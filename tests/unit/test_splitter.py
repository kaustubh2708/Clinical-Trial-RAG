from langchain_core.documents import Document

from src.ingestion.parser import load_document
from src.ingestion.splitter import split_protocol, strip_context_prefix
from tests.conftest import NSCLC_ID, NSCLC_PROTOCOL, T2D_ID, T2D_PROTOCOL

REQUIRED_METADATA = {"protocol_id", "source_document", "section", "criterion_type", "is_criterion", "chunk_index",
                     "content_hash", "chunk_id"}


def _split_text(text: str, protocol_id: str = "NCT12345678", **kwargs):
    return split_protocol([Document(page_content=text, metadata={})], protocol_id, "test.txt", **kwargs)


def _criteria(chunks, kind):
    return [c for c in chunks if c.metadata["is_criterion"] and c.metadata["criterion_type"] == kind]


def test_one_chunk_per_criterion_with_correct_labels():
    chunks = split_protocol(load_document(T2D_PROTOCOL), T2D_ID, T2D_PROTOCOL.name)
    assert len(_criteria(chunks, "inclusion")) == 7
    assert len(_criteria(chunks, "exclusion")) == 9
    egfr = next(c for c in chunks if "eGFR" in c.page_content)
    assert egfr.metadata["criterion_type"] == "exclusion"
    assert egfr.metadata["section"] == "Exclusion Criteria"


def test_context_prefix_and_metadata_on_every_chunk():
    chunks = split_protocol(load_document(T2D_PROTOCOL), T2D_ID, T2D_PROTOCOL.name)
    for chunk in chunks:
        assert REQUIRED_METADATA <= chunk.metadata.keys()
        assert chunk.metadata["protocol_id"] == T2D_ID
        assert chunk.metadata["source_document"] == T2D_PROTOCOL.name
        assert chunk.page_content.startswith(f"[{T2D_ID} | {chunk.metadata['section']}]\n")


def test_wrapped_lines_and_sub_items_stay_with_their_criterion():
    t2d = split_protocol(load_document(T2D_PROTOCOL), T2D_ID, T2D_PROTOCOL.name)
    contraception = next(c for c in t2d if "contraception" in c.page_content)
    assert "30 days after the last dose" in contraception.page_content

    nsclc = split_protocol(load_document(NSCLC_PROTOCOL), NSCLC_ID, NSCLC_PROTOCOL.name)
    organ = next(c for c in nsclc if "Adequate organ function" in c.page_content)
    assert "Creatinine clearance" in organ.page_content
    assert len(_criteria(nsclc, "inclusion")) == 6


def test_chunk_ids_are_stable_and_unique():
    first = split_protocol(load_document(T2D_PROTOCOL), T2D_ID, T2D_PROTOCOL.name)
    second = split_protocol(load_document(T2D_PROTOCOL), T2D_ID, T2D_PROTOCOL.name)
    ids = [c.metadata["chunk_id"] for c in first]
    assert ids == [c.metadata["chunk_id"] for c in second]
    assert len(ids) == len(set(ids))


def test_header_only_sections_are_dropped():
    chunks = split_protocol(load_document(T2D_PROTOCOL), T2D_ID, T2D_PROTOCOL.name)
    assert not any(strip_context_prefix(c.page_content).strip() == "## 3. Eligibility Criteria" for c in chunks)


def test_long_prose_is_split_within_chunk_size():
    prose = "## Background\n" + " ".join(f"Sentence number {i} about the disease." for i in range(200))
    chunks = _split_text(prose, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 5
    assert all(len(strip_context_prefix(c.page_content)) <= 300 for c in chunks)
    assert all(c.metadata["section"] == "Background" for c in chunks)


def test_ctgov_style_bullets_and_numbered_pdf_headers():
    text = (
        "5.1 Inclusion Criteria\n"
        "* Age >= 18 years\n"
        "* ECOG 0-1\n"
        "5.2 Exclusion Criteria\n"
        "* Pregnancy\n"
        "6.1 Study Treatment\n"
        "Drug X is given daily.\n"
    )
    chunks = _split_text(text)
    assert [c.metadata["criterion_type"] for c in chunks if c.metadata["is_criterion"]] == [
        "inclusion", "inclusion", "exclusion"]
    treatment = next(c for c in chunks if "Drug X" in c.page_content)
    assert treatment.metadata["section"] == "Study Treatment"
    assert treatment.metadata["is_criterion"] is False


def test_inline_criteria_after_colon_and_mid_sentence_mentions():
    text = (
        "## Methods\n"
        "Inclusion criteria were applied by two reviewers.\n"
        "Exclusion Criteria: Prior chemotherapy within 4 weeks\n"
    )
    chunks = _split_text(text)
    methods = next(c for c in chunks if "two reviewers" in c.page_content)
    assert methods.metadata["section"] == "Methods"
    exclusion = [c for c in chunks if c.metadata["criterion_type"] == "exclusion"]
    assert len(exclusion) == 1 and "Prior chemotherapy" in exclusion[0].page_content


def test_prose_eligibility_falls_back_but_keeps_labels():
    text = "Exclusion Criteria:\nPatients with severe hepatic impairment or active infection are not eligible."
    chunks = _split_text(text)
    assert len(chunks) == 1
    assert chunks[0].metadata["criterion_type"] == "exclusion" and chunks[0].metadata["is_criterion"] is True


def test_page_numbers_are_kept_for_pdfs():
    pages = [
        Document(page_content="## Overview\nPhase 2 study.", metadata={"page": 0}),
        Document(page_content="Inclusion Criteria:\n1. Adults over 18", metadata={"page": 1}),
    ]
    chunks = split_protocol(pages, "NCT12345678", "p.pdf")
    criterion = next(c for c in chunks if c.metadata["is_criterion"])
    assert criterion.metadata["page"] == 2


def test_strip_context_prefix():
    assert strip_context_prefix("[NCT12345678 | Exclusion Criteria]\nPregnant") == "Pregnant"
    assert strip_context_prefix("No prefix here") == "No prefix here"
