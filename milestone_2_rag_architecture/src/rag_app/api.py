"""FastAPI interface for the query-time RAG pipeline."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .bootstrap import build_services
from .config import Settings
from .models import Answer
from .services import RAGService

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    """Input accepted by POST /ask."""

    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceResponse(BaseModel):
    """Citable document chunk returned with an answer."""

    s3_uri: str
    chunk_index: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    """Grounded answer and retrieval evidence."""

    answer: str
    sources: list[SourceResponse]


def create_app(rag_service: RAGService | None = None, settings: Settings | None = None) -> FastAPI:
    """Create an app; optional injected dependencies keep endpoint tests offline."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not hasattr(app.state, "rag"):
            load_dotenv()
            loaded_settings = Settings.from_env()
            services = build_services(loaded_settings)
            app.state.settings = loaded_settings
            app.state.rag = services.rag
        yield

    app = FastAPI(title="Milestone 2 AWS RAG API", version="1.0.0", lifespan=lifespan)
    if rag_service is not None:
        app.state.rag = rag_service
    if settings is not None:
        app.state.settings = settings

    @app.get("/health")
    def health() -> dict[str, str]:
        """Confirm that API dependencies were initialized."""

        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest, request: Request) -> AskResponse:
        """Embed a question, retrieve OpenSearch evidence, and generate with Bedrock."""

        settings = getattr(request.app.state, "settings", None)
        default_top_k = getattr(settings, "retrieval_top_k", 4)
        try:
            result: Answer = request.app.state.rag.ask(payload.question, payload.top_k or default_top_k)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            logger.exception("RAG request failed")
            raise HTTPException(status_code=502, detail="RAG service request failed.") from error
        return AskResponse(
            answer=result.answer,
            sources=[
                SourceResponse(
                    s3_uri=f"s3://{source.source_bucket}/{source.source_key}",
                    chunk_index=source.chunk_index,
                    score=source.score,
                    excerpt=source.text[:500],
                )
                for source in result.sources
            ],
        )

    return app


app = create_app()
