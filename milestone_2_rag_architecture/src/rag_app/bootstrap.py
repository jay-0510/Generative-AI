"""AWS client and service wiring shared by the API and command-line scripts."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

from .chunking import LangChainChunker
from .config import Settings
from .document_loader import S3PdfLoader
from .opensearch_store import OpenSearchVectorStore
from .services import BedrockAnswerGenerator, IngestionService, RAGService, create_embeddings


@dataclass(frozen=True)
class ApplicationServices:
    """Fully configured services for ingestion and question answering."""

    ingestion: IngestionService
    rag: RAGService


def build_services(settings: Settings) -> ApplicationServices:
    """Build AWS SDK clients using the standard boto3 credential provider chain."""

    session = boto3.Session(region_name=settings.aws_region)
    s3 = session.client("s3")
    bedrock_runtime = session.client("bedrock-runtime")
    store = OpenSearchVectorStore(
        client=_opensearch_client(session, settings),
        index_name=settings.opensearch_index,
        embedding_dimension=settings.embedding_dimension,
        use_document_ids=settings.opensearch_service != "aoss",
        supports_refresh=settings.opensearch_service != "aoss",
    )
    embeddings = create_embeddings(bedrock_runtime, settings.embedding_model)
    ingestion = IngestionService(
        loader=S3PdfLoader(s3),
        chunker=LangChainChunker(settings.chunk_size, settings.chunk_overlap),
        embeddings=embeddings,
        store=store,
        embedding_batch_size=settings.embedding_batch_size,
    )
    rag = RAGService(
        embeddings=embeddings,
        store=store,
        generator=BedrockAnswerGenerator(
            bedrock_runtime,
            settings.chat_model,
            settings.max_context_characters,
        ),
    )
    return ApplicationServices(ingestion=ingestion, rag=rag)


def _opensearch_client(session: boto3.Session, settings: Settings) -> OpenSearch:
    """Configure IAM signing for AWS domains or basic auth for local development."""

    parsed = urlparse(settings.opensearch_url)
    if not parsed.hostname:
        raise ValueError("OPENSEARCH_URL must include a host, for example https://search.example.com")
    if settings.opensearch_username and settings.opensearch_password:
        auth = (settings.opensearch_username, settings.opensearch_password)
    else:
        credentials = session.get_credentials()
        if credentials is None:
            raise ValueError("No AWS credentials found for OpenSearch request signing.")
        auth = AWSV4SignerAuth(credentials, settings.aws_region, settings.opensearch_service)
    return OpenSearch(
        hosts=[{"host": parsed.hostname, "port": parsed.port or 443}],
        http_auth=auth,
        use_ssl=parsed.scheme == "https",
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )
