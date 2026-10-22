"""Evaluate retrieval (offline) and, if OPENAI_API_KEY is set, end-to-end answers.

Usage:
    python -m eval.run_eval                      # retrieval metrics only
    python -m eval.run_eval --thresholds 0.2 0.3 0.4
    OPENAI_API_KEY=sk-... python -m eval.run_eval --with-llm
"""

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import List

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


@dataclass
class CaseResult:
    case: dict
    rank: int | None  # 1-based rank of the first chunk containing an expected keyword
    top_score: float
    contaminated: int
    retrieved: int


def load_cases() -> List[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def build_chain(settings: Settings, llm=None) -> EligibilityRAGChain:
    store = build_vector_store(get_embeddings(), settings.CHROMA_PERSIST_DIR, settings.CHROMA_COLLECTION_NAME)
    indexer = ProtocolIndexer(store)
    for path in sorted(SAMPLES.glob("NCT*.txt")):
        protocol_id = path.name.split("_")[0]
        chunks = split_protocol(load_document(path), protocol_id, path.name, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        indexer.index_chunks(chunks, protocol_id)
    return EligibilityRAGChain(store, indexer, llm, settings)


def evaluate_retrieval(chain: EligibilityRAGChain, cases: List[dict], top_k: int) -> List[CaseResult]:
    results = []
    for case in cases:
        hits = chain.retrieve(case["question"], case["protocol_id"], top_k)  # gate disabled: MIN_RELEVANCE_SCORE=-1
        rank = None
        for position, (doc, _) in enumerate(hits, start=1):
            text = strip_context_prefix(doc.page_content).lower()
            if any(keyword.lower() in text for keyword in case["expected_keywords"]):
                rank = position
                break
        results.append(CaseResult(
            case=case,
            rank=rank,
            top_score=hits[0][1] if hits else -1.0,
            contaminated=sum(doc.metadata["protocol_id"] != case["protocol_id"] for doc, _ in hits),
            retrieved=len(hits),
        ))
    return results


def retrieval_report(results: List[CaseResult], thresholds: List[float], top_k: int) -> List[str]:
    lookups = [r for r in results if r.case["kind"] == "lookup"]
    abstains = [r for r in results if r.case["kind"] == "abstain"]
    hit_rate = mean(r.rank is not None for r in lookups)
    mrr = mean(1 / r.rank if r.rank else 0.0 for r in lookups)
    contamination = sum(r.contaminated for r in results) / max(1, sum(r.retrieved for r in results))

    lines = [
        f"## Retrieval (top_k={top_k}, {len(lookups)} lookup + {len(abstains)} out-of-scope questions)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Hit rate @{top_k} | {hit_rate:.2f} |",
        f"| MRR | {mrr:.2f} |",
        f"| Contamination rate | {contamination:.2f} |",
        "",
        "## Relevance gate sweep",
        "",
        "| MIN_RELEVANCE_SCORE | Abstention accuracy (out-of-scope blocked) | False abstention rate (answerable blocked) |",
        "| --- | --- | --- |",
    ]
    for threshold in thresholds:
        abstain_acc = mean(r.top_score < threshold for r in abstains) if abstains else float("nan")
        false_abstain = mean(r.top_score < threshold for r in lookups)
        lines.append(f"| {threshold:.2f} | {abstain_acc:.2f} | {false_abstain:.2f} |")

    lines += ["", "## Per-question retrieval", "", "| id | kind | rank of first correct chunk | top relevance |",
              "| --- | --- | --- | --- |"]
    for r in results:
        lines.append(f"| {r.case['id']} | {r.case['kind']} | {r.rank or '-'} | {r.top_score:.3f} |")
    return lines


def llm_report(settings: Settings, cases: List[dict], top_k: int) -> List[str]:
    from src.api.dependencies import get_llm

    llm = get_llm(settings)
    if llm is None:
        return ["## End-to-end (LLM)", "", "Skipped: OPENAI_API_KEY is not set."]
    chain = build_chain(settings, llm)
    correct, labelled, quotes, verified, grounded = 0, 0, 0, 0, 0
    rows = []
    for case in cases:
        response = chain.query(EligibilityQueryRequest(query=case["question"], protocol_id=case["protocol_id"], top_k=top_k))
        if case.get("expected_status"):
            labelled += 1
            correct += response.status.value == case["expected_status"]
        quotes += len(response.citations)
        verified += sum(c.quote_verified for c in response.citations)
        grounded += response.is_grounded
        rows.append(f"| {case['id']} | {case.get('expected_status', '-')} | {response.status.value} | {response.is_grounded} |")
    return [
        f"## End-to-end (LLM: {settings.LLM_MODEL_NAME}, MIN_RELEVANCE_SCORE={settings.MIN_RELEVANCE_SCORE})",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Status accuracy | {correct / max(1, labelled):.2f} ({correct}/{labelled}) |",
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
    parser.add_argument("--with-llm", action="store_true", help="Also run end-to-end answers (needs OPENAI_API_KEY).")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "results.md")
    args = parser.parse_args()

    cases = load_cases()
    with tempfile.TemporaryDirectory() as tmp:
        base = Settings(CHROMA_PERSIST_DIR=tmp, CHROMA_COLLECTION_NAME="eval")
        retrieval_settings = base.model_copy(update={"MIN_RELEVANCE_SCORE": -1.0})
        results = evaluate_retrieval(build_chain(retrieval_settings), cases, args.top_k)
        lines = ["# Evaluation results", "", *retrieval_report(results, args.thresholds, args.top_k)]
        if args.with_llm:
            llm_settings = base.model_copy(update={"CHROMA_COLLECTION_NAME": "eval_llm"})
            lines += ["", *llm_report(llm_settings, cases, args.top_k)]

    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
