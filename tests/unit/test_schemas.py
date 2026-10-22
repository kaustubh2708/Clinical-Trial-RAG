import pytest
from pydantic import ValidationError

from src.schemas import (
    EligibilityQueryRequest,
    EligibilityStatus,
    LLMAssessment,
    PatientScreeningRequest,
    ProtocolUploadResponse,
)


def test_valid_request_uses_default_top_k():
    request = EligibilityQueryRequest(query="Is eGFR below 30 excluded?", protocol_id="NCT04368728")
    assert request.top_k == 4


@pytest.mark.parametrize(
    "protocol_id",
    ["NCT123", "nct04368728", "NCT043687281", "NCT0436872A", " NCT04368728", "NCT04368728 ", "04368728", ""],
)
def test_invalid_nct_ids_are_rejected(protocol_id):
    with pytest.raises(ValidationError) as excinfo:
        EligibilityQueryRequest(query="Is eGFR below 30 excluded?", protocol_id=protocol_id)
    assert excinfo.value.errors()[0]["loc"] == ("protocol_id",)


@pytest.mark.parametrize("query", ["abc", "    hi    ", "x" * 1001])
def test_query_length_is_enforced_after_stripping(query):
    with pytest.raises(ValidationError):
        EligibilityQueryRequest(query=query, protocol_id="NCT04368728")


def test_query_is_stripped():
    request = EligibilityQueryRequest(query="   What is the age limit?  ", protocol_id="NCT04368728")
    assert request.query == "What is the age limit?"


@pytest.mark.parametrize("top_k", [0, 11, -1])
def test_top_k_bounds(top_k):
    with pytest.raises(ValidationError):
        EligibilityQueryRequest(query="What is the age limit?", protocol_id="NCT04368728", top_k=top_k)


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        EligibilityQueryRequest(query="What is the age limit?", protocol_id="NCT04368728", topk=3)


def test_screening_request_requires_meaningful_summary():
    with pytest.raises(ValidationError):
        PatientScreeningRequest(protocol_id="NCT04368728", patient_summary="58F, T2D")
    ok = PatientScreeningRequest(protocol_id="NCT04368728", patient_summary="58-year-old woman with type 2 diabetes.")
    assert ok.patient_summary.startswith("58")


def test_llm_assessment_parses_status_strings():
    parsed = LLMAssessment.model_validate(
        {"status": "CRITERIA_NOT_SPECIFIED", "evidence": [], "safety_concerns": [], "rationale": "n/a"}
    )
    assert parsed.status is EligibilityStatus.CRITERIA_NOT_SPECIFIED
    with pytest.raises(ValidationError):
        LLMAssessment.model_validate({"status": "MAYBE", "evidence": [], "safety_concerns": [], "rationale": ""})


def test_upload_response_defaults():
    response = ProtocolUploadResponse(message="ok", protocol_id="NCT04368728", source_filename="a.txt", chunks_indexed=3)
    assert response.replaced_existing is False and response.criteria_indexed == 0
