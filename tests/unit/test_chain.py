import pytest

from src.ingestion.parser import load_document
from src.ingestion.splitter import split_protocol
from src.rag.chain import (
    UNVERIFIED_EVIDENCE_CONCERN,
    LLMInvocationError,
    LLMNotConfiguredError,
    NoCriteriaError,
    ProtocolNotFoundError,
    aggregate_screening,
)
from src.rag.prompt_templates import INSUFFICIENT_CONTEXT_RESPONSE
from src.schemas import (
    CriterionAssessment,
    CriterionType,
    CriterionVerdictValue,
    EligibilityQueryRequest,
    EligibilityStatus,
    EvidenceItem,
    LLMAssessment,
    PatientScreeningRequest,
)
from tests.conftest import T2D_ID, T2D_PROTOCOL, FakeStructuredLLM

MET, NOT_MET, UNKNOWN = CriterionVerdictValue.MET, CriterionVerdictValue.NOT_MET, CriterionVerdictValue.INSUFFICIENT_INFO
INC, EXC = CriterionType.INCLUSION, CriterionType.EXCLUSION


@pytest.fixture
def indexed(indexer):
    chunks = split_protocol(load_document(T2D_PROTOCOL), T2D_ID, T2D_PROTOCOL.name)
    indexer.index_chunks(chunks, T2D_ID)
    return chunks


def _request(query="Are patients with eGFR below 30 excluded?", top_k=4):
    return EligibilityQueryRequest(query=query, protocol_id=T2D_ID, top_k=top_k)


def test_abstains_without_calling_llm_when_nothing_is_relevant(indexed, make_chain, fake_llm):
    chain = make_chain(fake_llm, MIN_RELEVANCE_SCORE=1.0)
    response = chain.query(_request())
    assert response.status is EligibilityStatus.CRITERIA_NOT_SPECIFIED
    assert response.assessment == INSUFFICIENT_CONTEXT_RESPONSE
    assert response.is_grounded is False and response.citations == []
    assert response.audit.llm_called is False
    assert fake_llm.call_count == 0


def test_abstention_works_without_an_llm(indexed, make_chain):
    assert make_chain(None, MIN_RELEVANCE_SCORE=1.0).query(_request()).audit.llm_called is False


def test_missing_llm_raises_when_an_answer_is_needed(indexed, make_chain):
    with pytest.raises(LLMNotConfiguredError):
        make_chain(None).query(_request())


def test_grounded_answer_has_verified_citations(indexed, make_chain, fake_llm):
    response = make_chain(fake_llm).query(_request())
    assert response.status is EligibilityStatus.NOT_ELIGIBLE
    assert response.is_grounded is True
    assert response.citations and all(c.quote_verified for c in response.citations)
    assert response.assessment.startswith("STATUS: NOT_ELIGIBLE")
    assert "PROTOCOL EVIDENCE:" in response.assessment and "SAFETY CONCERNS:" in response.assessment
    assert len(response.audit.retrieved_chunk_ids) == 4 and len(response.audit.context_hash) == 64


def test_fabricated_quote_downgrades_to_indeterminate(indexed, make_chain):
    def fabricate(chunks):
        return LLMAssessment(
            status=EligibilityStatus.ELIGIBLE,
            evidence=[EvidenceItem(chunk_id=chunks[0][0], quote="Patients with any kidney disease are welcome",
                                   explanation="made up")],
            safety_concerns=[],
            rationale="Looks fine.",
        )

    response = make_chain(FakeStructuredLLM(qa_responder=fabricate)).query(_request())
    assert response.status is EligibilityStatus.INDETERMINATE
    assert response.is_grounded is False
    assert response.citations[0].quote_verified is False
    assert UNVERIFIED_EVIDENCE_CONCERN in response.safety_concerns
    assert "UNVERIFIED" in response.assessment


def test_citing_a_chunk_that_was_not_retrieved_is_not_grounded(indexed, make_chain):
    def wrong_id(chunks):
        return LLMAssessment(
            status=EligibilityStatus.NOT_ELIGIBLE,
            evidence=[EvidenceItem(chunk_id="not-a-real-chunk", quote=chunks[0][1][:40], explanation="x")],
            safety_concerns=[],
            rationale="x",
        )

    response = make_chain(FakeStructuredLLM(qa_responder=wrong_id)).query(_request())
    assert response.is_grounded is False and response.status is EligibilityStatus.INDETERMINATE
    assert response.citations == []


def test_model_abstention_returns_the_exact_sentence(indexed, make_chain):
    def abstain(_):
        return LLMAssessment(status=EligibilityStatus.CRITERIA_NOT_SPECIFIED, evidence=[], safety_concerns=["x"],
                             rationale="nothing")

    response = make_chain(FakeStructuredLLM(qa_responder=abstain)).query(_request("Is coffee allowed on visit days?"))
    assert response.assessment == INSUFFICIENT_CONTEXT_RESPONSE
    assert response.citations == [] and response.safety_concerns == [] and response.is_grounded is False
    assert response.audit.llm_called is True


def test_unknown_protocol_raises(make_chain, fake_llm):
    with pytest.raises(ProtocolNotFoundError):
        make_chain(fake_llm).query(_request())


def test_llm_errors_are_wrapped(indexed, make_chain):
    with pytest.raises(LLMInvocationError):
        make_chain(FakeStructuredLLM(error=TimeoutError("boom"))).query(_request())


def _assessment(kind, verdict, cid="c"):
    return CriterionAssessment(chunk_id=cid, criterion_type=kind, criterion_text="t", verdict=verdict, rationale="r")


@pytest.mark.parametrize(
    "verdicts, expected",
    [
        ([(INC, MET), (EXC, NOT_MET)], EligibilityStatus.ELIGIBLE),
        ([(INC, MET), (EXC, MET)], EligibilityStatus.NOT_ELIGIBLE),  # any exclusion met
        ([(INC, NOT_MET), (EXC, NOT_MET)], EligibilityStatus.NOT_ELIGIBLE),  # an inclusion failed
        ([(INC, MET), (EXC, UNKNOWN)], EligibilityStatus.INDETERMINATE),
        ([(INC, UNKNOWN), (EXC, MET)], EligibilityStatus.NOT_ELIGIBLE),  # a definite failure beats unknowns
    ],
)
def test_aggregate_screening_rules(verdicts, expected):
    criteria = [_assessment(kind, verdict, f"c{i}") for i, (kind, verdict) in enumerate(verdicts)]
    status, _, _ = aggregate_screening(criteria)
    assert status is expected


PATIENT = (
    "62-year-old man with type 2 diabetes for 8 years on metformin 2000 mg daily. HbA1c 8.1%. "
    "BMI 31. Most recent eGFR 24 mL/min/1.73m2."
)


def test_screening_flags_met_exclusion(indexed, make_chain):
    llm = FakeStructuredLLM(screening_rules={"eGFR": (MET, "eGFR 24 mL/min/1.73m2")})
    response = make_chain(llm).screen(PatientScreeningRequest(protocol_id=T2D_ID, patient_summary=PATIENT))
    assert response.status is EligibilityStatus.NOT_ELIGIBLE
    assert len(response.criteria) == 16
    egfr = next(c for c in response.criteria if "eGFR" in c.criterion_text)
    assert egfr.verdict is MET and egfr.evidence_verified is True
    assert response.failing_criteria == [egfr.chunk_id]


def test_screening_downgrades_unverified_patient_evidence(indexed, make_chain):
    llm = FakeStructuredLLM(screening_rules={"eGFR": (MET, "eGFR is 12")})  # not in the summary
    response = make_chain(llm).screen(PatientScreeningRequest(protocol_id=T2D_ID, patient_summary=PATIENT))
    egfr = next(c for c in response.criteria if "eGFR" in c.criterion_text)
    assert egfr.verdict is UNKNOWN and egfr.rationale.startswith("[Downgraded")
    assert response.status is EligibilityStatus.INDETERMINATE and response.is_grounded is False


def test_screening_batches_and_fills_missing_verdicts(indexed, make_chain):
    llm = FakeStructuredLLM()
    response = make_chain(llm, MAX_SCREENING_CRITERIA=5).screen(
        PatientScreeningRequest(protocol_id=T2D_ID, patient_summary=PATIENT))
    assert llm.call_count == 4  # 16 criteria in batches of 5
    assert all(c.verdict is UNKNOWN for c in response.criteria)


def test_screening_requires_criteria(indexer, make_chain, fake_llm):
    chunks = [c for c in split_protocol(load_document(T2D_PROTOCOL), T2D_ID, "x.txt") if not c.metadata["is_criterion"]]
    indexer.index_chunks(chunks, T2D_ID)
    with pytest.raises(NoCriteriaError):
        make_chain(fake_llm).screen(PatientScreeningRequest(protocol_id=T2D_ID, patient_summary=PATIENT))


# ---------------------------------------------------------------- hybrid retrieval

@pytest.fixture
def nsclc_indexed(indexer):
    from tests.conftest import NSCLC_ID, NSCLC_PROTOCOL

    indexer.index_chunks(split_protocol(load_document(NSCLC_PROTOCOL), NSCLC_ID, NSCLC_PROTOCOL.name), NSCLC_ID)
    return NSCLC_ID


def test_hybrid_puts_exact_keyword_match_first(nsclc_indexed, make_chain):
    # Fake embeddings rank randomly; BM25 is the only signal, and RRF must surface its single match.
    result = make_chain(None, RETRIEVAL_MODE="hybrid").retrieve("CYP3A4 inhibitors", nsclc_indexed, 4)
    assert "CYP3A4" in result.chunks[0].page_content
    assert result.mode == "hybrid" and len(result.chunks) == 4


def test_code_in_protocol_passes_the_gate_only_in_hybrid_mode(indexed, make_chain, fake_llm):
    hybrid = make_chain(fake_llm, RETRIEVAL_MODE="hybrid", MIN_RELEVANCE_SCORE=1.0)
    result = hybrid.retrieve("MEN2 history", T2D_ID, 4)
    assert result.code_match is True and any("MEN2" in c.page_content for c in result.chunks)

    dense = make_chain(fake_llm, RETRIEVAL_MODE="dense", MIN_RELEVANCE_SCORE=1.0)
    assert dense.retrieve("MEN2 history", T2D_ID, 4).chunks == []


def test_off_topic_query_still_abstains_in_hybrid_mode(indexed, make_chain, fake_llm):
    chain = make_chain(fake_llm, RETRIEVAL_MODE="hybrid", MIN_RELEVANCE_SCORE=1.0)
    response = chain.query(_request("Where should visitors park their cars?"))
    assert response.status is EligibilityStatus.CRITERIA_NOT_SPECIFIED
    assert response.audit.retrieval_mode == "hybrid" and fake_llm.call_count == 0


def test_dense_mode_matches_previous_behaviour(indexed, make_chain, fake_llm):
    response = make_chain(fake_llm, RETRIEVAL_MODE="dense").query(_request(top_k=3))
    assert len(response.audit.retrieved_chunk_ids) == 3 and response.audit.retrieval_mode == "dense"


def test_llm_never_sees_the_embedding_prefix(indexed, make_chain, fake_llm):
    make_chain(fake_llm).query(_request())
    prompt = fake_llm.prompts[-1]
    assert f"[{T2D_ID} |" not in prompt  # prefix is for embeddings only
    assert ' section="' in prompt and ' type="' in prompt  # the same facts arrive as tag attributes


def test_prompts_render_with_literal_json_braces():
    from src.rag.prompt_templates import QA_PROMPT, SCREENING_PROMPT

    qa = QA_PROMPT.format_messages(protocol_id=T2D_ID, context="c", query="q")[0].content
    screen = SCREENING_PROMPT.format_messages(protocol_id=T2D_ID, criteria="c", patient_summary="p")[0].content
    assert '{"chunk_id"' in qa and '{"verdicts"' in screen


def test_screening_uses_short_labels_and_maps_them_back(indexed, make_chain):
    llm = FakeStructuredLLM(screening_rules={"eGFR": (MET, "eGFR 24 mL/min/1.73m2")})
    response = make_chain(llm).screen(PatientScreeningRequest(protocol_id=T2D_ID, patient_summary=PATIENT))
    prompt = llm.prompts[-1]
    assert '<criterion id="I1" type="inclusion">' in prompt and '<criterion id="E9" type="exclusion">' in prompt
    assert all(c.chunk_id not in prompt for c in response.criteria)  # hashes never shown to the model
    egfr = next(c for c in response.criteria if "eGFR" in c.criterion_text)
    assert egfr.verdict is MET and response.failing_criteria == [egfr.chunk_id]


def test_screening_downgrades_unrelated_evidence(indexed, make_chain):
    summary = PATIENT + " No history of pancreatitis."
    llm = FakeStructuredLLM(screening_rules={"Pregnant": (NOT_MET, "No history of pancreatitis")})
    response = make_chain(llm).screen(PatientScreeningRequest(protocol_id=T2D_ID, patient_summary=summary))
    pregnancy = next(c for c in response.criteria if "Pregnant" in c.criterion_text)
    assert pregnancy.verdict is UNKNOWN and "does not mention this criterion" in pregnancy.rationale


def test_short_patient_facts_count_as_evidence(indexed, make_chain):
    llm = FakeStructuredLLM(screening_rules={"Body mass index": (MET, "BMI 31")})
    response = make_chain(llm).screen(PatientScreeningRequest(protocol_id=T2D_ID, patient_summary=PATIENT))
    bmi = next(c for c in response.criteria if "Body mass index" in c.criterion_text)
    assert bmi.verdict is MET and bmi.evidence_verified is True


def test_query_never_confirms_eligibility(indexed, make_chain):
    def eligible(chunks):
        return LLMAssessment(
            status=EligibilityStatus.ELIGIBLE,
            evidence=[EvidenceItem(chunk_id=chunks[0][0], quote=chunks[0][1][:60], explanation="x")],
            safety_concerns=[],
            rationale="Looks fine.",
        )

    response = make_chain(FakeStructuredLLM(qa_responder=eligible)).query(_request())
    assert response.status is EligibilityStatus.INDETERMINATE
    assert response.is_grounded is True  # the quote was real; the conclusion is what cannot be trusted
    assert any("/protocols/screen" in concern for concern in response.safety_concerns)
