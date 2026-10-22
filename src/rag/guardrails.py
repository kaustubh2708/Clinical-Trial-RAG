"""Deterministic safety checks: context sanitisation, verbatim quote verification, privacy-safe hashing."""

import hashlib
import re
import unicodedata

MIN_QUOTE_CHARS = 8

_DELIMITER_TAGS = re.compile(r"</?\s*(?:chunk|criterion|protocol_context|patient_summary)\b[^>]*>", re.IGNORECASE)
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


def verify_quote(quote: str, source: str) -> bool:
    """True if `quote` appears verbatim (after normalisation) in `source`.

    A quote may elide text with "..." / "…"; each fragment must then appear in order.
    """
    fragments = [normalize_text(f) for f in _ELLIPSIS.split(quote)]
    fragments = [f.strip(" \"'") for f in fragments if f.strip(" \"'")]
    if not fragments or sum(len(f) for f in fragments) < MIN_QUOTE_CHARS:
        return False
    haystack = normalize_text(source)
    position = 0
    for fragment in fragments:
        found = haystack.find(fragment, position)
        if found < 0:
            return False
        position = found + len(fragment)
    return True


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_for_log(text: str) -> str:
    """Short, non-reversible fingerprint for logging user/patient text without storing it."""
    return sha256_hex(text)[:16]
