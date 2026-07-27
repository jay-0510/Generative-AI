"""Composable ingestion, retrieval, and Bedrock answer-generation services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_aws import BedrockEmbeddings

from .chunking import LangChainChunker
from .document_loader import S3PdfLoader
from .models import Answer, RetrievedChunk
from .opensearch_store import OpenSearchVectorStore


@dataclass(frozen=True)
class IngestionReport:
    """Summary returned after processing an S3 prefix."""

    documents_seen: int
    documents_ingested: int
    chunks_indexed: int
    errors: list[str]


class IngestionService:
    """Execute S3 -> PyMuPDF -> LangChain -> Bedrock -> OpenSearch."""

    def __init__(
        self,
        loader: S3PdfLoader,
        chunker: LangChainChunker,
        embeddings: Any,
        store: OpenSearchVectorStore,
        embedding_batch_size: int,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embeddings = embeddings
        self.store = store
        self.embedding_batch_size = embedding_batch_size

    def ingest(self, bucket: str, prefix: str) -> IngestionReport:
        """Index every supported S3 object and continue after individual failures."""

        self.store.ensure_index()
        seen = ingested = indexed = 0
        errors: list[str] = []
        for key in self.loader.list_keys(bucket, prefix):
            seen += 1
            try:
                document = self.loader.extract(bucket, key)
                chunks = self.chunker.split(document)
                for offset in range(0, len(chunks), self.embedding_batch_size):
                    batch = chunks[offset : offset + self.embedding_batch_size]
                    vectors = self.embeddings.embed_documents([chunk.text for chunk in batch])
                    self.store.upsert_chunks(batch, vectors)
                    indexed += len(batch)
                ingested += 1
            except Exception as error:  # Keep a bad source file from blocking the rest of the prefix.
                errors.append(f"s3://{bucket}/{key}: {error}")
        if indexed:
            self.store.refresh()
        return IngestionReport(seen, ingested, indexed, errors)


class BedrockAnswerGenerator:
    """Generate source-grounded answers through the Bedrock Converse API."""

    def __init__(self, runtime_client: Any, model_id: str, max_context_characters: int) -> None:
        self.runtime = runtime_client
        self.model_id = model_id
        self.max_context_characters = max_context_characters

    def generate(self, question: str, sources: list[RetrievedChunk]) -> str:
        """Ask the LLM to answer only from retrieved passages."""

        context = self._context(sources)
        prompt = (
            "Answer the question using only the supplied context. "
            "If the context does not contain the answer, say that you do not know. "
            "Cite source labels such as [1] in the answer.\n\n"
            f"Context:\n{context or 'No relevant context was retrieved.'}\n\n"
            f"Question: {question}"
        )
        response = self.runtime.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 700, "temperature": 0.0},
        )
        return "".join(part.get("text", "") for part in response["output"]["message"]["content"])

    def _context(self, sources: list[RetrievedChunk]) -> str:
        """Limit context size while retaining a readable source label for each chunk."""

        parts: list[str] = []
        used = 0
        for number, source in enumerate(sources, start=1):
            section = f"[{number}] s3://{source.source_bucket}/{source.source_key}\n{source.text}\n"
            if used + len(section) > self.max_context_characters:
                break
            parts.append(section)
            used += len(section)
        return "\n".join(parts)


class RAGService:
    """Run the query embedding, OpenSearch retrieval, and Bedrock response flow."""

    def __init__(self, embeddings: Any, store: OpenSearchVectorStore, generator: BedrockAnswerGenerator) -> None:
        self.embeddings = embeddings
        self.store = store
        self.generator = generator

    def ask(self, question: str, top_k: int) -> Answer:
        """Return a grounded answer with the OpenSearch chunks used as evidence."""

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("question must not be empty.")
        sources = self.retrieve(cleaned_question, top_k)
        return Answer(answer=self.generator.generate(cleaned_question, sources), sources=sources)

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        """Retrieve evidence only; used by the offline retrieval-quality evaluator."""

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("question must not be empty.")
        query_embedding = self.embeddings.embed_query(cleaned_question)
        return self.store.search(query_embedding, top_k)


def create_embeddings(runtime_client: Any, model_id: str) -> BedrockEmbeddings:
    """Create the LangChain adapter used by both indexing and retrieval."""

    return BedrockEmbeddings(client=runtime_client, model_id=model_id)
