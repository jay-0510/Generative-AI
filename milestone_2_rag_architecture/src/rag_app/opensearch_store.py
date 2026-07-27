"""Amazon OpenSearch vector index creation, indexing, and retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from opensearchpy import OpenSearch, helpers

from .models import DocumentChunk, RetrievedChunk


class OpenSearchVectorStore:
    """Thin OpenSearch wrapper that keeps vector schema explicit."""

    def __init__(
        self,
        client: OpenSearch,
        index_name: str,
        embedding_dimension: int,
        use_document_ids: bool = True,
        supports_refresh: bool = True,
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension
        self.use_document_ids = use_document_ids
        self.supports_refresh = supports_refresh

    def ensure_index(self) -> None:
        """Create the k-NN index once; preserve an existing index and its data."""

        if self.client.indices.exists(index=self.index_name):
            return
        body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dimension,
                        "method": {"name": "hnsw", "space_type": "cosinesimil"},
                    },
                    "text": {"type": "text"},
                    "source_bucket": {"type": "keyword"},
                    "source_key": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                }
            },
        }
        self.client.indices.create(index=self.index_name, body=body)

    def upsert_chunks(self, chunks: Sequence[DocumentChunk], embeddings: Sequence[list[float]]) -> None:
        """Bulk-index chunks. Serverless collections generate IDs themselves."""

        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        actions = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            action = {
                "_op_type": "index",
                "_index": self.index_name,
                "_source": {
                    "text": chunk.text,
                    "embedding": embedding,
                    "source_bucket": chunk.source_bucket,
                    "source_key": chunk.source_key,
                    "chunk_index": chunk.chunk_index,
                },
            }
            if self.use_document_ids:
                action["_id"] = chunk.id
            actions.append(action)
        if actions:
            helpers.bulk(self.client, actions, refresh=False)

    def refresh(self) -> None:
        """Make a completed managed-domain ingestion immediately searchable."""

        if self.supports_refresh:
            self.client.indices.refresh(index=self.index_name)

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return the most similar document chunks for a query vector."""

        response = self.client.search(
            index=self.index_name,
            body={
                "size": top_k,
                "_source": ["text", "source_bucket", "source_key", "chunk_index"],
                "query": {"knn": {"embedding": {"vector": query_embedding, "k": top_k}}},
            },
        )
        return [
            RetrievedChunk(
                id=hit["_id"],
                text=hit["_source"]["text"],
                source_bucket=hit["_source"]["source_bucket"],
                source_key=hit["_source"]["source_key"],
                chunk_index=hit["_source"]["chunk_index"],
                score=float(hit.get("_score", 0.0)),
            )
            for hit in response["hits"]["hits"]
        ]
