"""Eligibility RAG pipeline: retrieve (protocol-scoped) -> gate -> LLM structured output -> verify -> respond."""

import logging
import time
from itertools import islice
from typing import Iterable, Iterator, List, Sequence, TypeVar

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from src.config import Settings
from src.ingestion.indexer import ProtocolIndexer
from src.ingestion.splitter import strip_context_prefix
from src.rag.guardrails import hash_for_log, sanitize_context, sha256_hex, verify_quote
from src.rag.prompt_templates import (
    INSUFFICIENT_CONTEXT_RESPONSE,
    QA_PROMPT,
    QA_PROMPT_VERSION,
    SCREENING_PROMPT,
    SCREENING_PROMPT_VERSION,
    format_context,
    format_criteria,
)
from src.schemas import (
    AuditInfo,
    CitationSource,
    CriterionAssessment,
    CriterionType,
    CriterionVerdictValue,
    EligibilityQueryRequest,
    EligibilityQueryResponse,
    EligibilityStatus,
    LLMAssessment,
    LLMScreeningAssessment,
    PatientScreeningRequest,
    ScreeningResponse,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

UNVERIFIED_EVIDENCE_CONCERN = (
    "Model evidence could not be verified verbatim against the protocol text; the status was downgraded to "
    "INDETERMINATE. Review the protocol manually."
)


class ProtocolNotFoundError(LookupError):
    """No chunks are indexed for the requested protocol_id."""


class NoCriteriaError(LookupError):
    """The protocol is indexed but no inclusion/exclusion criteria were detected."""


class LLMNotConfiguredError(RuntimeError):
    """An LLM call is needed but no LLM is configured (e.g. OPENAI_API_KEY missing)."""


class LLMInvocationError(RuntimeError):
    """The LLM call failed or returned output that did not match the schema."""


def _batched(items: Iterable[T], size: int) -> Iterator[List[T]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def aggregate_screening(criteria: Sequence[CriterionAssessment]) -> tuple[EligibilityStatus, List[str], List[str]]:
    """Combine per-criterion verdicts: all inclusions met AND no exclusion met.

    Returns (status, failing_chunk_ids, unresolved_chunk_ids).
    """
    failing = [
        c.chunk_id
        for c in criteria
        if (c.criterion_type is CriterionType.EXCLUSION and c.verdict is CriterionVerdictValue.MET)
        or (c.criterion_type is CriterionType.INCLUSION and c.verdict is CriterionVerdictValue.NOT_MET)
    ]
    unresolved = [c.chunk_id for c in criteria if c.verdict is CriterionVerdictValue.INSUFFICIENT_INFO]
    if failing:
        return EligibilityStatus.NOT_ELIGIBLE, failing, unresolved
    if unresolved:
        return EligibilityStatus.INDETERMINATE, failing, unresolved
    return EligibilityStatus.ELIGIBLE, failing, unresolved


def render_assessment(status: EligibilityStatus, rationale: str, citations: Sequence[CitationSource],
                      safety_concerns: Sequence[str]) -> str:
    """Render the STATUS / PROTOCOL EVIDENCE / SAFETY CONCERNS report from structured data."""
    if status is EligibilityStatus.CRITERIA_NOT_SPECIFIED:
        return INSUFFICIENT_CONTEXT_RESPONSE
    evidence_lines = [
        f'- [{c.section} | {c.chunk_id}] "{c.quote}" ({"verified" if c.quote_verified else "UNVERIFIED"})'
        for c in citations
    ] or ["- No protocol evidence cited."]
    concern_lines = [f"- {concern}" for concern in safety_concerns] or ["- None identified in the cited protocol text."]
    return "\n".join(
        [f"STATUS: {status.value}", rationale.strip(), "", "PROTOCOL EVIDENCE:", *evidence_lines, "",
         "SAFETY CONCERNS:", *concern_lines]
    )


class EligibilityRAGChain:
    def __init__(self, vector_store: Chroma, indexer: ProtocolIndexer, llm: BaseChatModel | None, settings: Settings):
        self.vector_store = vector_store
        self.indexer = indexer
        self.llm = llm
        self.settings = settings

    # ------------------------------------------------------------------ helpers

    def _require_llm(self) -> BaseChatModel:
        if self.llm is None:
            raise LLMNotConfiguredError("No LLM configured. Set OPENAI_API_KEY to enable assessments.")
        return self.llm

    def _invoke_structured(self, prompt, schema, variables: dict):
        chain = prompt | self._require_llm().with_structured_output(schema)
        try:
            result = chain.invoke(variables)
        except LLMNotConfiguredError:
            raise
        except Exception as exc:  # network errors, rate limits, schema validation failures
            raise LLMInvocationError(f"LLM call failed: {type(exc).__name__}") from exc
        if not isinstance(result, schema):
            raise LLMInvocationError("LLM returned output that does not match the expected schema.")
        return result

    def _audit(self, prompt_version: str, chunk_ids: List[str], context_hash: str | None, llm_called: bool) -> AuditInfo:
        return AuditInfo(
            model=self.settings.LLM_MODEL_NAME,
            prompt_version=prompt_version,
            retrieved_chunk_ids=chunk_ids,
            context_hash=context_hash,
            llm_called=llm_called,
        )

    def retrieve(self, query: str, protocol_id: str, top_k: int) -> List[tuple[Document, float]]:
        """Top-k chunks for one protocol with cosine relevance (1 - cosine distance), filtered by the gate."""
        results = self.vector_store.similarity_search_with_score(query, k=top_k, filter={"protocol_id": protocol_id})
        scored = []
        for doc, distance in results:
            if doc.metadata.get("protocol_id") != protocol_id:  # defence in depth; the filter should prevent this
                logger.error("Dropped chunk from %s while querying %s", doc.metadata.get("protocol_id"), protocol_id)
                continue
            relevance = 1.0 - float(distance)
            if relevance >= self.settings.MIN_RELEVANCE_SCORE:
                scored.append((doc, relevance))
        return scored

    # ------------------------------------------------------------------ Q&A

    def query(self, request: EligibilityQueryRequest) -> EligibilityQueryResponse:
        started = time.perf_counter()
        if not self.indexer.protocol_exists(request.protocol_id):
            raise ProtocolNotFoundError(request.protocol_id)

        retrieved = self.retrieve(request.query, request.protocol_id, request.top_k)
        chunks = [doc for doc, _ in retrieved]
        chunk_ids = [doc.metadata["chunk_id"] for doc in chunks]

        if not chunks:
            response = EligibilityQueryResponse(
                protocol_id=request.protocol_id,
                query=request.query,
                status=EligibilityStatus.CRITERIA_NOT_SPECIFIED,
                assessment=INSUFFICIENT_CONTEXT_RESPONSE,
                citations=[],
                is_grounded=False,
                audit=self._audit(QA_PROMPT_VERSION, [], None, llm_called=False),
            )
            self._log("query", request.query, response.status, chunk_ids, False, started)
            return response

        context = format_context(chunks)
        llm_output: LLMAssessment = self._invoke_structured(
            QA_PROMPT, LLMAssessment, {"protocol_id": request.protocol_id, "context": context, "query": request.query}
        )

        by_id = {doc.metadata["chunk_id"]: doc for doc in chunks}
        citations: List[CitationSource] = []
        all_verified = bool(llm_output.evidence)
        for item in llm_output.evidence:
            doc = by_id.get(item.chunk_id)
            if doc is None:  # model cited a chunk it was not given
                all_verified = False
                continue
            excerpt = strip_context_prefix(doc.page_content)
            verified = verify_quote(item.quote, excerpt)
            all_verified = all_verified and verified
            citations.append(
                CitationSource(
                    source_document=doc.metadata.get("source_document", "unknown"),
                    protocol_id=doc.metadata["protocol_id"],
                    excerpt=excerpt,
                    chunk_id=item.chunk_id,
                    section=doc.metadata.get("section"),
                    criterion_type=doc.metadata.get("criterion_type"),
                    quote=item.quote,
                    quote_verified=verified,
                )
            )

        status = llm_output.status
        safety_concerns = list(llm_output.safety_concerns)
        if status is EligibilityStatus.CRITERIA_NOT_SPECIFIED:
            citations, is_grounded = [], False
        else:
            is_grounded = all_verified
            if not is_grounded and status in (EligibilityStatus.ELIGIBLE, EligibilityStatus.NOT_ELIGIBLE):
                status = EligibilityStatus.INDETERMINATE
                safety_concerns.append(UNVERIFIED_EVIDENCE_CONCERN)

        response = EligibilityQueryResponse(
            protocol_id=request.protocol_id,
            query=request.query,
            status=status,
            assessment=render_assessment(status, llm_output.rationale, citations, safety_concerns),
            citations=citations,
            is_grounded=is_grounded,
            safety_concerns=safety_concerns if status is not EligibilityStatus.CRITERIA_NOT_SPECIFIED else [],
            audit=self._audit(QA_PROMPT_VERSION, chunk_ids, sha256_hex(context), llm_called=True),
        )
        self._log("query", request.query, response.status, chunk_ids, True, started)
        return response

    # ------------------------------------------------------------------ screening

    def screen(self, request: PatientScreeningRequest) -> ScreeningResponse:
        started = time.perf_counter()
        if not self.indexer.protocol_exists(request.protocol_id):
            raise ProtocolNotFoundError(request.protocol_id)
        criteria = self.indexer.get_criteria(request.protocol_id)
        if not criteria:
            raise NoCriteriaError(request.protocol_id)

        patient_summary = sanitize_context(request.patient_summary)
        verdicts = {}
        contexts = []
        for batch in _batched(criteria, self.settings.MAX_SCREENING_CRITERIA):
            criteria_block = format_criteria(batch)
            contexts.append(criteria_block)
            output: LLMScreeningAssessment = self._invoke_structured(
                SCREENING_PROMPT,
                LLMScreeningAssessment,
                {"protocol_id": request.protocol_id, "criteria": criteria_block, "patient_summary": patient_summary},
            )
            for verdict in output.verdicts:
                verdicts.setdefault(verdict.criterion_id, verdict)  # first verdict wins; unknown ids are ignored

        assessments: List[CriterionAssessment] = []
        is_grounded = True
        for criterion in criteria:
            meta = criterion.metadata
            verdict = verdicts.get(meta["chunk_id"])
            text = strip_context_prefix(criterion.page_content)
            if verdict is None:
                is_grounded = False
                assessments.append(CriterionAssessment(
                    chunk_id=meta["chunk_id"], criterion_type=meta["criterion_type"], criterion_text=text,
                    verdict=CriterionVerdictValue.INSUFFICIENT_INFO,
                    rationale="No verdict returned by the model; treated as insufficient information.",
                ))
                continue

            value, rationale, evidence_verified = verdict.verdict, verdict.rationale, False
            if value is not CriterionVerdictValue.INSUFFICIENT_INFO:
                evidence_verified = verify_quote(verdict.patient_evidence, request.patient_summary)
                if not evidence_verified:
                    is_grounded = False
                    value = CriterionVerdictValue.INSUFFICIENT_INFO
                    rationale = f"[Downgraded: patient evidence not found verbatim in the summary] {rationale}"
            assessments.append(CriterionAssessment(
                chunk_id=meta["chunk_id"], criterion_type=meta["criterion_type"], criterion_text=text,
                verdict=value, patient_evidence=verdict.patient_evidence or None,
                evidence_verified=evidence_verified, rationale=rationale,
            ))

        status, failing, unresolved = aggregate_screening(assessments)
        summary = (
            f"{status.value}: {len(assessments)} criteria checked; {len(failing)} failing, "
            f"{len(unresolved)} need more patient information. Decision support only; a qualified "
            "research coordinator must confirm eligibility."
        )
        chunk_ids = [c.metadata["chunk_id"] for c in criteria]
        response = ScreeningResponse(
            protocol_id=request.protocol_id,
            status=status,
            summary=summary,
            criteria=assessments,
            failing_criteria=failing,
            unresolved_criteria=unresolved,
            is_grounded=is_grounded,
            audit=self._audit(SCREENING_PROMPT_VERSION, chunk_ids, sha256_hex("\n".join(contexts)), llm_called=True),
        )
        self._log("screen", request.patient_summary, status, chunk_ids, True, started)
        return response

    def _log(self, kind: str, text: str, status: EligibilityStatus, chunk_ids: List[str], llm_called: bool,
             started: float) -> None:
        logger.info(
            "%s text_hash=%s status=%s chunks=%d llm_called=%s latency_ms=%.0f",
            kind, hash_for_log(text), status.value, len(chunk_ids), llm_called, (time.perf_counter() - started) * 1000,
        )
