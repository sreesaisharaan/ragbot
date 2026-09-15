"""Deterministic hybrid retrieval used by the generation step."""
from __future__ import annotations

import logging
import re
from math import sqrt
from typing import Iterable, Sequence

from config import SIMILARITY_THRESHOLD, TOP_K
from models import DocumentChunk, RetrievedChunk

logger = logging.getLogger("ragbot.retrieval")
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_STOP_WORDS = {"a", "an", "and", "are", "be", "for", "how", "i", "in", "is", "it", "of", "on", "the", "to", "what", "when", "where", "which", "who"}
_QUERY_ALIASES = {
    "syllabus": {"course", "description", "unit", "units", "objective", "objectives"},
    "objective": {"objectives", "outcome", "outcomes", "purpose", "aim", "goal"},
    "objectives": {"objective", "outcome", "outcomes", "purpose", "aim", "goal"},
    "learning": {"course", "outcome", "outcomes", "pedagogy"},
}


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOP_WORDS}


def lexical_overlap(query: str, content: str) -> float:
    """Return normalized query-term coverage, with document-Q&A aliases."""
    query_tokens = _tokens(query)
    expanded = set(query_tokens)
    for token in query_tokens:
        expanded.update(_QUERY_ALIASES.get(token, set()))
    if not expanded:
        return 0.0
    return len(expanded & _tokens(content)) / len(expanded)


class InMemoryRetriever:
    def __init__(self, chunks: Iterable[DocumentChunk], threshold=SIMILARITY_THRESHOLD, top_k=TOP_K):
        self.chunks = list(chunks)
        self.threshold = threshold
        self.top_k = top_k

    def retrieve(self, query_embedding: Sequence[float], query: str | None = None) -> list[RetrievedChunk]:
        scored = []
        for chunk in self.chunks:
            similarity = cosine_similarity(query_embedding, chunk.embedding)
            lexical = lexical_overlap(query, chunk.content) if query else 0.0
            scored.append((RetrievedChunk(chunk, similarity), lexical))

        # Hashed vectors are deterministic but collision-prone. When lexical
        # evidence exists, use it to discard unrelated vector-collision hits;
        # keep vector ranking as the tie-breaker and as the no-lexical fallback.
        lexical_hits = [item for item in scored if item[1] > 0]
        if lexical_hits:
            candidates = [item for item in lexical_hits if item[0].similarity >= 0.0]
            candidates.sort(key=lambda item: (item[1], item[0].similarity, -item[0].chunk.chunk_index), reverse=True)
            eligible = [item[0] for item in candidates]
        else:
            eligible = [item[0] for item in scored if item[0].similarity >= self.threshold]
            eligible.sort(key=lambda item: (item.similarity, -item.chunk.chunk_index), reverse=True)

        logger.info(
            "[RAG][3-retrieval] chunks=%d query_dim=%d threshold=%.3f top_k=%d lexical_hits=%d eligible=%d",
            len(self.chunks), len(query_embedding), self.threshold, self.top_k, len(lexical_hits), len(eligible),
        )
        logger.info(
            "[RAG][3-scores] %s",
            [(item[0].chunk.filename, item[0].chunk.chunk_index, round(item[0].similarity, 4), round(item[1], 4)) for item in scored],
        )
        return eligible[: self.top_k]
