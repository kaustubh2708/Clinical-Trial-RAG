"""Deterministic safety checks: context sanitisation, verbatim quote verification, privacy-safe hashing."""

import hashlib
import re
import unicodedata

MIN_QUOTE_CHARS = 8  # protocol quotes
MIN_PATIENT_EVIDENCE_CHARS = 4  # short facts such as "BMI 31" or "ECOG 1" are valid patient evidence

_DELIMITER_TAGS = re.compile(r"</?\s*(?:chunk|criterion|protocol_context|patient_summary)\b[^>]*>", re.IGNORECASE)
# Our own identifiers, which models sometimes paste into quotes as citation markers. They are never protocol text.
_ID_MARKERS = re.compile(r"\[(?:[0-9a-f]{32}|NCT\d{8}(?:\s*\|[^\]]*)?)\]")
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")
_WHITESPACE = re.compile(r"\s+")
_CHAR_MAP = str.maketrans(
    {
        "‘": "'", "’": "'", "‛": "'", "′": "'",
        "“": '"', "”": '"', "‟": '"', "″": '"',
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
        " ": " ",
    }
)


def sanitize_context(text: str) -> str:
    """Remove tag-like strings that could close or forge our prompt delimiters (prompt-injection defence)."""
    return _DELIMITER_TAGS.sub("", text)


def normalize_text(text: str) -> str:
    """Canonical form for comparing quotes: NFKC, unified quotes/dashes, lowercase, collapsed whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP).lower()
    return _WHITESPACE.sub(" ", text).strip()


def verify_quote(quote: str, source: str, min_chars: int = MIN_QUOTE_CHARS) -> bool:
    """True if `quote` appears verbatim (after normalisation) in `source`.

    A quote may elide text with "..." / "…"; each fragment must then appear in order.
    Chunk-ID and "[NCT... | Section]" markers are removed first; everything else must match.
    """
    fragments = [normalize_text(f) for f in _ELLIPSIS.split(_ID_MARKERS.sub(" ", quote))]
    fragments = [f.strip(" \"'") for f in fragments if f.strip(" \"'")]
    if not fragments or sum(len(f) for f in fragments) < min_chars:
        return False
    haystack = normalize_text(source)
    position = 0
    for fragment in fragments:
        found = haystack.find(fragment, position)
        if found < 0:
            return False
        position = found + len(fragment)
    return True


# Words that appear across many unrelated criteria, so sharing one proves nothing about relevance.
_GENERIC_TERMS = frozenset(
    ["history", "patient", "participant", "subject", "study", "screening", "prior", "current", "known", "any", "no", "not"]
)


def _content_terms(text: str) -> set[str]:
    from src.rag.hybrid import tokenize  # local import: hybrid has no dependency on guardrails

    terms = set()
    for token in tokenize(text):
        for part in token.split("-"):
            if part and not part.replace(".", "").isdigit() and part not in _GENERIC_TERMS:
                terms.add(part[:-1] if len(part) > 3 and part.endswith("s") and not part.endswith("ss") else part)
    return terms


def evidence_mentions_criterion(evidence: str, criterion: str) -> bool:
    """True if patient evidence shares at least one content term with the criterion.

    Catches a model citing a real but unrelated fact ("No history of pancreatitis" as evidence about pregnancy).
    It cannot catch wrong reasoning about a relevant fact, such as misreading a numeric threshold.
    """
    return bool(_content_terms(evidence) & _content_terms(criterion))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_for_log(text: str) -> str:
    """Short, non-reversible fingerprint for logging user/patient text without storing it."""
    return sha256_hex(text)[:16]
