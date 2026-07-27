"""LangChain text chunking for extracted PDF text."""

from __future__ import annotations

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import DocumentChunk, ExtractedDocument


class LangChainChunker:
    """Split documents while preserving deterministic source identifiers."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, document: ExtractedDocument) -> list[DocumentChunk]:
        """Return chunks that can be safely re-ingested without duplicate records."""

        pieces = self.splitter.split_text(document.text)
        return [
            DocumentChunk(
                id=self._chunk_id(document.bucket, document.key, index, text),
                text=text,
                source_bucket=document.bucket,
                source_key=document.key,
                chunk_index=index,
            )
            for index, text in enumerate(pieces)
            if text.strip()
        ]

    @staticmethod
    def _chunk_id(bucket: str, key: str, index: int, text: str) -> str:
        """Hash source location and content to create an idempotent OpenSearch id."""

        value = f"{bucket}:{key}:{index}:{text}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()
