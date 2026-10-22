"""Shared fixtures: offline fakes for embeddings and the LLM, isolated temp storage per test."""

import re
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableLambda

from app import app
from src.api.dependencies import get_llm, get_vector_store
from src.config import Settings, get_settings
from src.ingestion.indexer import ProtocolIndexer, build_vector_store
from src.rag.chain import EligibilityRAGChain
from src.schemas import (
    CriterionVerdictValue,
    EligibilityStatus,
    EvidenceItem,
    LLMAssessment,
    LLMCriterionVerdict,
    LLMScreeningAssessment,
)

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_protocols"
T2D_PROTOCOL = SAMPLE_DIR / "NCT90000001_t2d_protocol.txt"
NSCLC_PROTOCOL = SAMPLE_DIR / "NCT90000002_nsclc_protocol.txt"
T2D_ID, NSCLC_ID = "NCT90000001", "NCT90000002"

_CHUNK_BLOCK = re.compile(r'<chunk id="(?P<id>\w+)"[^>]*>\n\[[^\]]*\]\n(?P<text>.*?)\n</chunk>', re.DOTALL)
_CRITERION_BLOCK = re.compile(r'<criterion id="(?P<id>\w+)" type="(?P<type>\w+)">\n\[[^\]]*\]\n(?P<text>.*?)\n</criterion>', re.DOTALL)


class FakeStructuredLLM:
    """Stands in for ChatOpenAI. `with_structured_output(schema)` returns a runnable producing `schema` objects.

    By default Q&A answers quote the first chunk they are given (so quotes verify), and screening returns
    INSUFFICIENT_INFO for every criterion. Tests customise behaviour via `qa_responder` / `screening_rules`.
    """

    def __init__(
        self,
        qa_responder: Callable[[List[Tuple[str, str]]], LLMAssessment] | None = None,
        screening_rules: Dict[str, Tuple[CriterionVerdictValue, str]] | None = None,
        error: Exception | None = None,
    ):
        self.qa_responder = qa_responder
        self.screening_rules = screening_rules or {}
        self.error = error
        self.prompts: List[str] = []

    @property
    def call_count(self) -> int:
        return len(self.prompts)

    def with_structured_output(self, schema, **_):
        return RunnableLambda(lambda prompt_value: self._respond(schema, prompt_value.to_string()))

    def _respond(self, schema, prompt: str):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        if schema is LLMAssessment:
            chunks = [(m["id"], m["text"]) for m in _CHUNK_BLOCK.finditer(prompt)]
            if self.qa_responder is not None:
                return self.qa_responder(chunks)
            chunk_id, text = chunks[0]
            return LLMAssessment(
                status=EligibilityStatus.NOT_ELIGIBLE,
                evidence=[EvidenceItem(chunk_id=chunk_id, quote=text[:60], explanation="Directly stated.")],
                safety_concerns=[],
                rationale="Decided by the cited criterion.",
            )
        if schema is LLMScreeningAssessment:
            verdicts = []
            for match in _CRITERION_BLOCK.finditer(prompt):
                verdict, evidence = CriterionVerdictValue.INSUFFICIENT_INFO, ""
                for keyword, (rule_verdict, rule_evidence) in self.screening_rules.items():
                    if keyword.lower() in match["text"].lower():
                        verdict, evidence = rule_verdict, rule_evidence
                verdicts.append(
                    LLMCriterionVerdict(criterion_id=match["id"], verdict=verdict, patient_evidence=evidence,
                                        rationale="fake")
                )
            return LLMScreeningAssessment(verdicts=verdicts)
        raise AssertionError(f"Unexpected schema {schema}")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        CHROMA_PERSIST_DIR=str(tmp_path / "chroma"),
        UPLOAD_DIR=str(tmp_path / "uploads"),
        CHROMA_COLLECTION_NAME=f"test_{uuid.uuid4().hex[:8]}",
        MIN_RELEVANCE_SCORE=-1.0,  # fake embeddings give meaningless similarities; gate off unless a test sets it
        OPENAI_API_KEY=None,
    )


@pytest.fixture
def vector_store(settings: Settings):
    return build_vector_store(DeterministicFakeEmbedding(size=384), settings.CHROMA_PERSIST_DIR,
                              settings.CHROMA_COLLECTION_NAME)


@pytest.fixture
def indexer(vector_store) -> ProtocolIndexer:
    return ProtocolIndexer(vector_store)


@pytest.fixture
def fake_llm() -> FakeStructuredLLM:
    return FakeStructuredLLM()


@pytest.fixture
def make_chain(vector_store, indexer, settings):
    def _make(llm=None, **overrides) -> EligibilityRAGChain:
        chain_settings = settings.model_copy(update=overrides)
        return EligibilityRAGChain(vector_store=vector_store, indexer=indexer, llm=llm, settings=chain_settings)

    return _make


@pytest.fixture
def client(settings, vector_store, fake_llm):
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: fake_llm
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def upload(client: TestClient, path: Path, protocol_id: str, filename: str | None = None):
    with path.open("rb") as handle:
        return client.post(
            "/api/v1/protocols/index",
            files={"file": (filename or path.name, handle, "text/plain")},
            data={"protocol_id": protocol_id},
        )
