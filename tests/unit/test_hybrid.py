import pytest

from src.rag.hybrid import BM25, code_terms, reciprocal_rank_fusion, tokenize


def test_tokenize_lowercases_drops_stopwords_and_stems_plurals():
    assert tokenize("What are the liver enzyme limits?") == ["liver", "enzyme", "limit"]
    assert tokenize("CYP3A4 inhibitors, eGFR 1.73m2 and PD-1") == ["cyp3a4", "inhibitor", "egfr", "1.73m2", "pd-1"]
    assert tokenize("class process") == ["class", "process"]  # "-ss" words are not stemmed


@pytest.mark.parametrize(
    "query, expected",
    [
        ("MEN2 history", ["men2"]),
        ("ULN limit for ALT", ["uln", "alt"]),
        ("CYP3A4 inhibitors", ["cyp3a4"]),
        ("Is the patient 45 years old?", []),  # pure numbers are not codes
        ("NCT90000001 parking", []),  # trial IDs are in every chunk prefix, so they never count
        ("Where should visitors park?", []),
    ],
)
def test_code_terms(query, expected):
    assert code_terms(query) == expected


def test_bm25_ranks_exact_term_first_and_skips_non_matches():
    corpus = [
        "Strong CYP3A4 inhibitors must be avoided.",
        "Age 18 years or older.",
        "Use of any SGLT2 inhibitor within 90 days.",
    ]
    bm25 = BM25(corpus)
    assert bm25.rank("CYP3A4 inhibitors", limit=10) == [0, 2]  # doc 1 has no matching term
    assert bm25.contains_any(["sglt2"]) and not bm25.contains_any(["anc"])


def test_bm25_prefers_rarer_terms():
    corpus = ["participant visit", "participant consent", "participant calcitonin"]
    scores = BM25(corpus).scores("participant calcitonin")
    assert scores[2] > scores[0] == scores[1] > 0


def test_reciprocal_rank_fusion():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=60)
    assert fused[0] == "a"  # 1/61 + 1/62 beats c's 1/63 + 1/61
    assert set(fused) == {"a", "b", "c"}
    assert reciprocal_rank_fusion([["x"], []]) == ["x"]
