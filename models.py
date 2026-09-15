from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DocumentChunk:
    filename: str
    chunk_index: int
    content: str
    embedding: Sequence[float]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    similarity: float
