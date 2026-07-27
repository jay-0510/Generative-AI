"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime configuration read once during application startup."""

    aws_region: str
    s3_bucket: str
    s3_prefix: str
    opensearch_url: str
    opensearch_index: str
    opensearch_service: str
    opensearch_username: str | None
    opensearch_password: str | None
    embedding_model: str
    embedding_dimension: int
    chat_model: str
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int
    retrieval_top_k: int
    max_context_characters: int

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables and fail on missing essentials."""

        def required(name: str) -> str:
            value = os.getenv(name)
            if not value:
                raise ValueError(f"{name} must be set. Copy .env.example to .env first.")
            return value

        def integer(name: str, default: int) -> int:
            value = int(os.getenv(name, str(default)))
            if value < 1:
                raise ValueError(f"{name} must be greater than zero.")
            return value

        settings = cls(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            s3_bucket=required("S3_BUCKET"),
            s3_prefix=os.getenv("S3_PREFIX", ""),
            opensearch_url=required("OPENSEARCH_URL").rstrip("/"),
            opensearch_index=os.getenv("OPENSEARCH_INDEX", "milestone-rag-documents"),
            opensearch_service=os.getenv("OPENSEARCH_SERVICE", "es"),
            opensearch_username=os.getenv("OPENSEARCH_USERNAME") or None,
            opensearch_password=os.getenv("OPENSEARCH_PASSWORD") or None,
            embedding_model=os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"),
            embedding_dimension=integer("BEDROCK_EMBEDDING_DIMENSION", 1024),
            chat_model=os.getenv("BEDROCK_CHAT_MODEL", "amazon.nova-lite-v1:0"),
            chunk_size=integer("CHUNK_SIZE", 900),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            embedding_batch_size=integer("EMBEDDING_BATCH_SIZE", 20),
            retrieval_top_k=integer("RETRIEVAL_TOP_K", 4),
            max_context_characters=integer("MAX_CONTEXT_CHARACTERS", 12000),
        )
        if settings.chunk_overlap < 0 or settings.chunk_overlap >= settings.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE.")
        return settings
