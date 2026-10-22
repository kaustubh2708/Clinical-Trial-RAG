"""Evaluate retrieval (offline) and, optionally, end-to-end answers with a real LLM.

Usage:
    python -m eval.run_eval                          # dense vs hybrid retrieval, gate sweep
    python -m eval.run_eval --min-hit-rate 0.95      # exit 1 if the configured mode falls below (CI)
    OPENAI_API_KEY=sk-... python -m eval.run_eval --with-llm
    LLM_PROVIDER=ollama LLM_MODEL_NAME=qwen2.5:7b python -m eval.run_eval --with-llm --limit 10
"""

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from src.config import Settings
from src.ingestion.indexer import ProtocolIndexer, build_vector_store
from src.ingestion.parser import load_document
from src.ingestion.splitter import split_protocol, strip_context_prefix
from src.rag.chain import EligibilityRAGChain
from src.rag.embeddings import get_embeddings
from src.schemas import EligibilityQueryRequest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "sample_protocols"
GOLDEN = ROOT / "eval" / "golden_set.jsonl"
MODES = ("dense", "hybrid")
ANSWERABLE = ("lookup", "exact")  # paraphrased questions / questions naming an exact clinical term


@dataclass
class CaseResult:
    case: dict
    rank: int | None  # 1-based rank of the first chunk containing an expected keyword
    top_relevance: float  # best dense relevance; what the abstention gate sees
    code_match: bool  # hybrid gate bypass: query names a code/acronym present in the protocol
    contaminated: int
    retrieved: int


def load_cases() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def build_chain(settings: Settings, llm=None) -> EligibilityRAGChain:
    store = build_vector_store(get_embeddings(), settings.CHROMA_PERSIST_DIR, settings.CHROMA_COLLECTION_NAME)
    indexer = ProtocolIndexer(store)
    for path in sorted(SAMPLES.glob("NCT*.txt")):
        protocol_id = path.name.split("_")[0]
        chunks = split_protocol(load_document(path), protocol_id, path.name, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        indexer.index_chunks(chunks, protocol_id)
    return EligibilityRAGChain(store, indexer, llm, settings)


def evaluate_retrieval(chain: EligibilityRAGChain, cases: list[dict], top_k: int) -> list[CaseResult]:
    results = []
    for case in cases:
        retrieval = chain.retrieve(case["question"], case["protocol_id"], top_k)  # gate disabled for measurement
        rank = None
        for position, doc in enumerate(retrieval.chunks, start=1):
            text = strip_context_prefix(doc.page_content).lower()
            if any(keyword.lower() in text for keyword in case["expected_keywords"]):
                rank = position
                break
        results.append(CaseResult(
            case=case,
            rank=rank,
            top_relevance=retrieval.top_relevance,
            code_match=retrieval.code_match,
            contaminated=sum(doc.metadata["protocol_id"] != case["protocol_id"] for doc in retrieval.chunks),
            retrieved=len(retrieval.chunks),
        ))
    return results


def hit_rate(results: list[CaseResult]) -> float:
    return mean(r.rank is not None for r in results) if results else float("nan")


def mrr(results: list[CaseResult]) -> float:
    return mean(1 / r.rank if r.rank else 0.0 for r in results) if results else float("nan")


def retrieval_report(by_mode: dict[str, list[CaseResult]], thresholds: list[float], top_k: int) -> list[str]:
    first = next(iter(by_mode.values()))
    counts = {kind: sum(r.case["kind"] == kind for r in first) for kind in (*ANSWERABLE, "abstain")}
    lines = [
        f"## Retrieval (top_k={top_k}; {counts['lookup']} paraphrased, {counts['exact']} exact-term, "
        f"{counts['abstain']} out-of-scope questions)",
        "",
        "| Mode | Hit rate @k (all) | MRR (all) | MRR paraphrased | MRR exact-term | Contamination |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for mode, results in by_mode.items():
        answerable = [r for r in results if r.case["kind"] in ANSWERABLE]
        contamination = sum(r.contaminated for r in results) / max(1, sum(r.retrieved for r in results))
        lines.append(
            f"| {mode} | {hit_rate(answerable):.2f} | {mrr(answerable):.2f} | "
            f"{mrr([r for r in answerable if r.case['kind'] == 'lookup']):.2f} | "
            f"{mrr([r for r in answerable if r.case['kind'] == 'exact']):.2f} | {contamination:.2f} |"
        )

    lines += [
        "",
        "## Relevance gate sweep",
        "",
        "Blocked = best dense relevance below the threshold (hybrid also lets through queries naming a code "
        "found verbatim in the protocol).",
        "",
        "| MIN_RELEVANCE_SCORE | Mode | Abstention accuracy (out-of-scope blocked) | "
        "False abstention rate (answerable blocked) |",
        "| --- | --- | --- | --- |",
    ]
    for threshold in thresholds:
        for mode, results in by_mode.items():
            def blocked(r: CaseResult, threshold: float = threshold, mode: str = mode) -> bool:
                return r.top_relevance < threshold and not (mode == "hybrid" and r.code_match)

            abstains = [r for r in results if r.case["kind"] == "abstain"]
            answerable = [r for r in results if r.case["kind"] in ANSWERABLE]
            abstain_acc = mean(blocked(r) for r in abstains) if abstains else float("nan")
            lines.append(f"| {threshold:.2f} | {mode} | {abstain_acc:.2f} | {mean(blocked(r) for r in answerable):.2f} |")

    header = " | ".join(f"rank ({mode})" for mode in by_mode)
    lines += ["", "## Per-question retrieval", "", f"| id | kind | {header} | top dense relevance |",
              "| --- | --- | " + " | ".join("---" for _ in by_mode) + " | --- |"]
    for index, case_result in enumerate(first):
        ranks = " | ".join(str(results[index].rank or "-") for results in by_mode.values())
        lines.append(f"| {case_result.case['id']} | {case_result.case['kind']} | {ranks} | "
                     f"{case_result.top_relevance:.3f} |")
    return lines


def llm_report(settings: Settings, cases: list[dict], top_k: int) -> list[str]:
    from src.api.dependencies import get_llm

    llm = get_llm(settings)
    if llm is None:
        return ["## End-to-end (LLM)", "", "Skipped: no LLM configured (set OPENAI_API_KEY or LLM_PROVIDER=ollama)."]
    chain = build_chain(settings, llm)
    correct, labelled, quotes, verified, grounded, unsafe = 0, 0, 0, 0, 0, 0
    rows = []
    for case in cases:
        response = chain.query(EligibilityQueryRequest(query=case["question"], protocol_id=case["protocol_id"], top_k=top_k))
        if case.get("expected_status"):
            labelled += 1
            correct += response.status.value == case["expected_status"]
            # The dangerous error: telling a coordinator a patient may enrol when a criterion rules them out.
            unsafe += response.status.value == "ELIGIBLE" and case["expected_status"] == "NOT_ELIGIBLE"
        quotes += len(response.citations)
        verified += sum(c.quote_verified for c in response.citations)
        grounded += response.is_grounded
        rows.append(f"| {case['id']} | {case.get('expected_status', '-')} | {response.status.value} | {response.is_grounded} |")
    return [
        f"## End-to-end (LLM: {settings.LLM_PROVIDER}/{settings.LLM_MODEL_NAME}, mode={settings.RETRIEVAL_MODE}, "
        f"MIN_RELEVANCE_SCORE={settings.MIN_RELEVANCE_SCORE})",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Status accuracy | {correct / max(1, labelled):.2f} ({correct}/{labelled}) |",
        f"| Unsafe answers (ELIGIBLE when NOT_ELIGIBLE expected) | {unsafe} |",
        f"| Citation verification rate | {verified / max(1, quotes):.2f} ({verified}/{quotes}) |",
        f"| Grounded rate | {grounded / len(cases):.2f} |",
        "",
        "| id | expected | predicted | grounded |",
        "| --- | --- | --- | --- |",
        *rows,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-k", type=int, default=Settings().RETRIEVAL_TOP_K)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.15, 0.25, 0.30, 0.35, 0.45])
    parser.add_argument("--with-llm", action="store_true",
                        help="Also run end-to-end answers (OPENAI_API_KEY, or LLM_PROVIDER=ollama).")
    parser.add_argument("--limit", type=int, default=None,
                        help="With --with-llm: only the first N cases (local models are slow).")
    parser.add_argument("--min-hit-rate", type=float, default=None,
                        help="Exit with status 1 if the configured RETRIEVAL_MODE's hit rate is below this.")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "results.md")
    args = parser.parse_args()

    cases = load_cases()
    with tempfile.TemporaryDirectory() as tmp:
        base = Settings(CHROMA_PERSIST_DIR=tmp, CHROMA_COLLECTION_NAME="eval")
        by_mode = {}
        for mode in MODES:
            settings = base.model_copy(update={"MIN_RELEVANCE_SCORE": -1.0, "RETRIEVAL_MODE": mode})
            by_mode[mode] = evaluate_retrieval(build_chain(settings), cases, args.top_k)
        lines = ["# Evaluation results", "", *retrieval_report(by_mode, args.thresholds, args.top_k)]
        if args.with_llm:
            llm_settings = base.model_copy(update={"CHROMA_COLLECTION_NAME": "eval_llm"})
            lines += ["", *llm_report(llm_settings, cases[: args.limit], args.top_k)]

    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if args.min_hit_rate is not None:
        configured = [r for r in by_mode[base.RETRIEVAL_MODE] if r.case["kind"] in ANSWERABLE]
        if hit_rate(configured) < args.min_hit_rate:
            sys.exit(f"Hit rate {hit_rate(configured):.2f} is below the floor {args.min_hit_rate:.2f}")


if __name__ == "__main__":
    main()
