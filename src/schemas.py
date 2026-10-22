"""Request/response models for the API, plus the structured-output schemas the LLM must fill."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from src.config import get_settings

NCT_ID_PATTERN = r"^NCT\d{8}$"


class EligibilityStatus(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    INDETERMINATE = "INDETERMINATE"
    CRITERIA_NOT_SPECIFIED = "CRITERIA_NOT_SPECIFIED"


class CriterionType(StrEnum):
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"
    OTHER = "other"


class CriterionVerdictValue(StrEnum):
    MET = "MET"
    NOT_MET = "NOT_MET"
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"


# ---------------------------------------------------------------------------
# API requests
# ---------------------------------------------------------------------------


class EligibilityQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=5, max_length=1000, description="Eligibility question.")
    protocol_id: str = Field(..., pattern=NCT_ID_PATTERN, description="ClinicalTrials.gov ID, e.g. NCT04368728.")
    top_k: int = Field(default_factory=lambda: get_settings().RETRIEVAL_TOP_K, ge=1, le=10)

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class PatientScreeningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_id: str = Field(..., pattern=NCT_ID_PATTERN)
    patient_summary: str = Field(
        ...,
        min_length=20,
        max_length=4000,
        description="De-identified patient summary. Do not send names, MRNs or dates of birth.",
    )

    @field_validator("patient_summary", mode="before")
    @classmethod
    def strip_summary(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


# ---------------------------------------------------------------------------
# API responses
# ---------------------------------------------------------------------------


class CitationSource(BaseModel):
    source_document: str
    protocol_id: str
    excerpt: str = Field(..., description="Text of the cited protocol chunk.")
    chunk_id: str | None = None
    section: str | None = None
    criterion_type: CriterionType | None = None
    quote: str | None = Field(default=None, description="Verbatim quote the model cited from this chunk.")
    quote_verified: bool = False


class AuditInfo(BaseModel):
    """Everything needed to explain or replay an answer."""

    model: str
    llm_provider: str | None = None
    prompt_version: str
    retrieval_mode: str | None = None
    retrieved_chunk_ids: list[str]
    context_hash: str | None = Field(default=None, description="SHA-256 of the exact context sent to the LLM.")
    llm_called: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EligibilityQueryResponse(BaseModel):
    protocol_id: str
    query: str
    status: EligibilityStatus
    assessment: str
    citations: list[CitationSource]
    is_grounded: bool
    safety_concerns: list[str] = Field(default_factory=list)
    audit: AuditInfo


class ProtocolUploadResponse(BaseModel):
    message: str
    protocol_id: str
    source_filename: str
    chunks_indexed: int
    criteria_indexed: int = 0
    replaced_existing: bool = False


class CriterionAssessment(BaseModel):
    chunk_id: str
    criterion_type: CriterionType
    criterion_text: str
    verdict: CriterionVerdictValue
    patient_evidence: str | None = None
    evidence_verified: bool = False
    rationale: str


class ScreeningResponse(BaseModel):
    protocol_id: str
    status: EligibilityStatus
    summary: str
    criteria: list[CriterionAssessment]
    failing_criteria: list[str] = Field(default_factory=list, description="chunk_ids that make the patient ineligible.")
    unresolved_criteria: list[str] = Field(default_factory=list, description="chunk_ids lacking patient information.")
    is_grounded: bool
    audit: AuditInfo


class ProtocolSummary(BaseModel):
    protocol_id: str
    source_documents: list[str]
    chunk_count: int


# ---------------------------------------------------------------------------
# Structured-output schemas filled by the LLM
# ---------------------------------------------------------------------------


# Every field is length-bounded. With constrained decoding (Ollama, OpenAI JSON schema) the bounds become part of
# the grammar, so a model cannot loop inside a string or list; output beyond them fails validation (fail closed).
ShortText = Annotated[str, StringConstraints(max_length=300)]


class EvidenceItem(BaseModel):
    chunk_id: str = Field(..., max_length=64, description="ID of the protocol chunk the quote comes from.")
    quote: str = Field(..., max_length=400, description="Short exact span copied verbatim from that chunk.")
    explanation: str = Field(..., max_length=400, description="How this quote bears on the question.")


class LLMAssessment(BaseModel):
    status: EligibilityStatus
    evidence: list[EvidenceItem] = Field(..., max_length=6)
    safety_concerns: list[ShortText] = Field(..., max_length=8)
    rationale: str = Field(..., max_length=1500)


class LLMCriterionVerdict(BaseModel):
    criterion_id: str = Field(..., max_length=64, description="The id attribute of the criterion being judged.")
    verdict: CriterionVerdictValue
    patient_evidence: str = Field(
        ...,
        max_length=300,
        description="Exact text copied from the patient summary that supports the verdict; empty if none.",
    )
    rationale: str = Field(..., max_length=500)


class LLMScreeningAssessment(BaseModel):
    verdicts: list[LLMCriterionVerdict] = Field(..., max_length=200)
