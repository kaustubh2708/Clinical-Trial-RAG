"""Keyword retrieval (BM25) and Reciprocal Rank Fusion, used to make dense retrieval robust to exact terms.

Dense embeddings blur exact tokens such as "CYP3A4", "QTc" or "ULN"; BM25 matches them exactly.
Both rankings are merged with RRF, which needs only ranks, not comparable scores.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
# Codes and acronyms as typed by the user: contain a digit (CYP3A4, SGLT2) or are all caps (ALK, ULN).
_CODE = re.compile(r"\b(?:[A-Za-z\-]*\d[A-Za-z0-9\-]*|[A-Z]{2,}[A-Z0-9\-]*)\b")
_NCT_ID = re.compile(r"nct\d{8}")
STOPWORDS = frozenset(
    ["a", "an", "and", "any", "are", "as", "at", "be", "by", "can", "could", "do", "does", "for", "from", "has", "have", "how", "i", "if", "in", "is", "it", "of", "on", "or", "that", "the", "their", "there", "this", "to", "was", "were", "what", "when", "where", "which", "who", "will", "with"]
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed and a light plural stem ("limits" -> "limit")."""
    tokens = []
    for token in _TOKEN.findall(text.lower()):
        if token in STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


def code_terms(query: str) -> list[str]:
    """Code/acronym tokens in a query, lowercased ("MEN2 history" -> ["men2"]).

    Pure numbers and trial IDs (every chunk's prefix contains its NCT ID) are ignored.
    """
    terms: list[str] = []
    for match in _CODE.findall(query):
        terms.extend(t for t in tokenize(match) if any(c.isalpha() for c in t) and not _NCT_ID.fullmatch(t))
    return terms


class BM25:
    """Okapi BM25 over a small in-memory corpus (one protocol: tens to hundreds of chunks)."""

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(text)) for text in corpus]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        document_frequency: Counter = Counter()
        for doc in self.docs:
            document_frequency.update(doc.keys())
        n = len(self.docs)
        # "+1" variant keeps IDF positive even for terms present in most documents.
        self.idf: dict[str, float] = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in document_frequency.items()
        }

    def contains_any(self, terms: Sequence[str]) -> bool:
        """True if any term occurs verbatim (as a token) somewhere in the corpus."""
        return any(term in self.idf for term in terms)

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        results = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1.0))
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            results.append(score)
        return results

    def rank(self, query: str, limit: int) -> list[int]:
        """Indices of documents with a non-zero score, best first."""
        scored = [(score, index) for index, score in enumerate(self.scores(query)) if score > 0]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [index for _, index in scored[:limit]]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    """Merge ranked ID lists: score(id) = sum over lists of 1 / (k + rank). Ties keep first-seen order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: -scores[item])
