"""Data structures exchanged by ingestion and question-answering services."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedDocument:
    """Text extracted from one S3 object."""

    bucket: str
    key: str
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable fragment with stable source metadata."""

    id: str
    text: str
    source_bucket: str
    source_key: str
    chunk_index: int


@dataclass(frozen=True)
class RetrievedChunk:
    """A vector-search result returned to the answer generator."""

    id: str
    text: str
    source_bucket: str
    source_key: str
    chunk_index: int
    score: float


@dataclass(frozen=True)
class Answer:
    """Grounded LLM output and the chunks used to produce it."""

    answer: str
    sources: list[RetrievedChunk]
