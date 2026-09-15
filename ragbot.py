"""Public RAGbot orchestration API."""
import logging
import re
from typing import Callable, Sequence

from models import RetrievedChunk
from prompts import SYSTEM_PROMPT, assemble_user_prompt

logger = logging.getLogger("ragbot.pipeline")
NOT_ENOUGH = "I don't have enough information in the uploaded documents to answer that."
_METADATA_RE = re.compile(r"\bhow many documents (?:have i uploaded|are uploaded|did i upload)\b", re.I)


def _validate_generation_payload(raw) -> tuple[str, list[dict]]:
    """Validate the local shape; JSON object mode is transport only."""
    if not isinstance(raw, dict):
        raise ValueError("generation result must be a JSON object")
    answer = raw.get("answer")
    sources = raw.get("sources")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("generation result must contain a non-empty answer")
    if not isinstance(sources, list):
        raise ValueError("generation result sources must be a list")
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("each source must be an object")
        if not isinstance(source.get("filename"), str):
            raise ValueError("source filename must be a string")
        index = source.get("chunk_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("source chunk_index must be an integer")
    return answer, sources


class RAGBot:
    def __init__(self, retriever, generator, embedder: Callable[[str], Sequence[float]], document_count: int | None = None):
        self.retriever = retriever
        self.generator = generator
        self.embedder = embedder
        self.document_count = document_count

    def answer(self, question: str) -> dict:
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        if _METADATA_RE.search(question) and self.document_count is not None:
            return {"answer": f"You have uploaded {self.document_count} documents.", "sources": [], "context_found": False}

        query_embedding = self.embedder(question)
        logger.info("[RAG][2-query-embedding] question=%r dim=%d nonzero=%d", question, len(query_embedding), sum(value != 0 for value in query_embedding))
        retrieved: list[RetrievedChunk] = self.retriever.retrieve(query_embedding, question)
        if not retrieved:
            logger.warning("[RAG][6-fallback] no eligible chunks; generation skipped question=%r", question)
            return {"answer": NOT_ENOUGH, "sources": [], "context_found": False}

        user_prompt = assemble_user_prompt(retrieved, question)
        logger.info("[RAG][5-llm-input] system_chars=%d user_chars=%d chunks=%d", len(SYSTEM_PROMPT), len(user_prompt), len(retrieved))
        logger.debug("[RAG][4-prompt] exact_user_prompt=%s", user_prompt)
        raw = self.generator.generate(SYSTEM_PROMPT, user_prompt)
        answer, returned_sources = _validate_generation_payload(raw)
        allowed = {(x.chunk.filename, x.chunk.chunk_index) for x in retrieved}
        sources = []
        for source in returned_sources:
            key = (source["filename"], source["chunk_index"])
            if key in allowed:
                entry = {"filename": source["filename"], "chunk_index": source["chunk_index"]}
                if entry not in sources:
                    sources.append(entry)
        # Preserve citations even when a model returns the answer text but omits the JSON list.
        if not sources:
            for item in retrieved:
                marker = f"[Source: {item.chunk.filename}, chunk {item.chunk.chunk_index}]"
                if marker in answer:
                    sources.append({"filename": item.chunk.filename, "chunk_index": item.chunk.chunk_index})
        logger.info("[RAG][6-result] context_found=True sources=%s", sources)
        return {"answer": answer, "sources": sources, "context_found": True}
