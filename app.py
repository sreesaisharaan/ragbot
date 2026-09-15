"""Deterministic local document ingestion and FastAPI application."""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from pathlib import Path, PurePath
from typing import Iterable

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import LOCAL_SIMILARITY_THRESHOLD, LOCAL_TOP_K
from models import DocumentChunk
from ragbot import RAGBot
from retrieval import InMemoryRetriever
from supabase_store import PersistenceError, SupabaseStore

BASE_DIR = Path(__file__).parent
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_VECTOR_SIZE = 128


def _load_local_env() -> None:
    """Load local dotenv-style settings without requiring python-dotenv.

    Prefer the normal `.env`; `.env.example` is supported because this project
    currently uses that filename locally. Existing process variables win.
    """
    for filename in (".env", ".env.example"):
        path = BASE_DIR / filename
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value.strip('"').strip("'")
        return


# Environment loading is explicit in run_server.py so importing app in tests stays offline.
logger = logging.getLogger("ragbot.ingestion")


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into ordered word chunks with a bounded overlap."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    return [" ".join(words[start : start + chunk_size]) for start in range(0, len(words), step)]


def local_embedding(text: str) -> list[float]:
    """Create a repeatable hashed bag-of-words embedding without network calls."""
    vector = [0.0] * _VECTOR_SIZE
    for token in _WORD_RE.findall(text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % _VECTOR_SIZE
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    result = [value / norm for value in vector] if norm else vector
    logger.debug("[RAG][2-embedding] input_chars=%d dim=%d norm=%.6f zero=%s", len(text), len(result), norm, not any(result))
    return result


class LocalGenerator:
    """Small offline generator used when GROQ_API_KEY is not configured."""

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        match = re.search(r"\[Source: ([^,]+), chunk (\d+)\]\n(.*?)(?=\n\n|\n---\nQuestion:)", user_prompt, re.S)
        if not match:
            return {"answer": "I don't have enough information in the uploaded documents to answer that.", "sources": []}
        filename, index, content = match.groups()
        answer = re.split(r"(?<=[.!?])\s+", content.strip(), maxsplit=1)[0]
        return {"answer": f"{answer} [Source: {filename}, chunk {index}]", "sources": [{"filename": filename, "chunk_index": int(index)}]}


class DocumentStore:
    def __init__(self, generator=None, persistence=None):
        self.chunks: list[DocumentChunk] = []
        self.filenames: set[str] = set()
        self.generator = generator or self._default_generator()
        self.persistence = persistence or SupabaseStore()
        if self.persistence.enabled:
            try:
                for row in self.persistence.load_chunks():
                    embedding = row["embedding"]
                    if isinstance(embedding, str):
                        embedding = [float(value) for value in embedding.strip("[]").split(",") if value.strip()]
                    if len(embedding) != _VECTOR_SIZE:
                        raise RuntimeError("stored embedding dimension does not match local embedder")
                    document = row.get("documents") or {}
                    chunk = DocumentChunk(
                        document.get("filename", row.get("filename", "unknown.txt")),
                        int(row["chunk_index"]), row["content"], embedding
                    )
                    self.chunks.append(chunk)
                    self.filenames.add(chunk.filename)
                    logger.info("[RAG][1-chunk-loaded] file=%r index=%d chars=%d preview=%r", chunk.filename, chunk.chunk_index, len(chunk.content), chunk.content[:200])
            except Exception as exc:
                raise RuntimeError("Supabase chunks could not be loaded") from exc

    @staticmethod
    def _default_generator():
        if os.getenv("GROQ_API_KEY"):
            from generator import GroqGenerator
            return GroqGenerator(api_key=os.environ["GROQ_API_KEY"])
        return LocalGenerator()

    def add_text(self, filename: str, text: str) -> list[dict]:
        new_chunks = []
        for index, content in enumerate(chunk_text(text)):
            new_chunks.append(DocumentChunk(filename, index, content, local_embedding(content)))
            logger.info("[RAG][1-chunk-created] file=%r index=%d chars=%d preview=%r", filename, index, len(content), content[:200])
        metadata = [{"filename": item.filename, "chunk_index": item.chunk_index} for item in new_chunks]
        inserted = self.persistence.persist_document(
            filename,
            text,
            [{**item, "content": chunk.content, "embedding": list(chunk.embedding)} for item, chunk in zip(metadata, new_chunks)],
        )
        if not inserted:
            return []
        self.chunks.extend(new_chunks)
        return metadata

    def record_chat(self, question: str, result: dict) -> None:
        self.persistence.persist_chat(question, result)

    def bot(self) -> RAGBot:
        return RAGBot(InMemoryRetriever(self.chunks, threshold=LOCAL_SIMILARITY_THRESHOLD, top_k=LOCAL_TOP_K), self.generator, local_embedding, len(self.filenames))

    def reset(self) -> None:
        self.chunks.clear()
        self.filenames.clear()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


app = FastAPI(title="RAGbot", version="1.1")
app.state.store = DocumentStore()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web")), name="static")


@app.get("/")
def home():
    return FileResponse(str(BASE_DIR / "web" / "index.html"))


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = PurePath(file.filename or "").name
    if not filename or not filename.lower().endswith(".txt"):
        raise HTTPException(status_code=415, detail="Only TXT files are supported in this MVP")
    try:
        text = (await file.read()).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="TXT files must be UTF-8 encoded") from exc
    if not text.strip():
        raise HTTPException(status_code=400, detail="TXT files must not be empty")
    try:
        chunks = app.state.store.add_text(filename, text)
    except PersistenceError as exc:
        raise HTTPException(status_code=503, detail="Document persistence is temporarily unavailable") from exc
    return {"filename": filename, "chunks_added": len(chunks), "chunks": chunks}


@app.post("/ask")
def ask(request: AskRequest):
    try:
        result = app.state.store.bot().answer(request.question)
        app.state.store.record_chat(request.question, result)
        return result
    except PersistenceError as exc:
        raise HTTPException(status_code=503, detail="Chat persistence is temporarily unavailable") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
