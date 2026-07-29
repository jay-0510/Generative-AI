"""
config/settings.py
===================

WHY THIS FILE EXISTS
---------------------
Every practical so far has scattered `os.getenv(...)` calls across files. The
moment you add a *second* observability backend (LangFuse, on top of
LangSmith) you end up with 6-8 environment variables that need to be read
consistently in more than one module (main.py, the graph nodes, the tests).

Centralizing them in one `pydantic-settings` object gives us:
  1. A single place to see every credential/knob the project needs.
  2. Type validation at startup (fail fast if AWS_REGION is missing, instead
     of failing 40 seconds into a Bedrock call with a cryptic boto3 error).
  3. A typed object (`settings.nova_lite_model_id`) instead of magic strings
     sprinkled through the codebase.

This is the ONLY file that should call `os.environ[...] = ...` to push
values into process environment (see `export_tracing_env()` below) — the
LangChain/LangGraph SDKs read LangSmith config directly from environment
variables, not from a config object we pass in, so we still need to expose
them there. Everything else in the project imports `settings` from here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Typed application configuration, loaded from environment variables / a
    `.env` file (see `.env.example` for the full list of keys).

    Using pydantic-settings (rather than plain `os.getenv`) means a missing
    *required* value raises a clear `ValidationError` at import time instead
    of an obscure error deep inside a Bedrock or LangGraph call.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ------------------------------------------------------------------ #
    # AWS / Bedrock
    # ------------------------------------------------------------------ #
    aws_region: str = Field(default="us-east-1", description="Bedrock is only available in a subset of regions.")

    # Practical constraint: Amazon Nova Micro / Nova Lite ONLY.
    # Nova Micro is text-only and the cheapest ($0.035 / $0.14 per 1M
    # input/output tokens) -> ideal for the classifier node, which does a
    # single-word routing decision and gains nothing from a bigger model.
    # Nova Lite adds light multimodal support and a larger context window
    # ($0.06 / $0.24 per 1M tokens) -> used for the SQL and RAG agents,
    # which have to reason over schema/retrieved context, not just classify.
    # (Rates are indicative Bedrock on-demand, us-east-1, pulled while this
    # project was written — always confirm current numbers at
    # https://aws.amazon.com/bedrock/pricing/ before trusting a cost report.)
    nova_micro_model_id: str = Field(default="amazon.nova-micro-v1:0")
    nova_lite_model_id: str = Field(default="amazon.nova-lite-v1:0")
    # Used only to embed documents for the RAG agent's vector store. This is
    # an *embedding* model, not a chat/completion model, so it does not
    # violate the "Nova only" constraint on generation.
    titan_embed_model_id: str = Field(default="amazon.titan-embed-text-v2:0")

    classifier_model_id: str = Field(default="amazon.nova-micro-v1:0")
    sql_agent_model_id: str = Field(default="amazon.nova-lite-v1:0")
    rag_agent_model_id: str = Field(default="amazon.nova-lite-v1:0")

    # ------------------------------------------------------------------ #
    # Local data stores
    # ------------------------------------------------------------------ #
    sqlite_db_path: str = Field(default="data/app.db", description="SQL agent's queryable database.")
    chroma_persist_dir: str = Field(default="data/chroma", description="RAG agent's vector store.")
    knowledge_base_dir: str = Field(default="data/knowledge_base")

    # ------------------------------------------------------------------ #
    # LangSmith (Part 1 of this practical)
    # ------------------------------------------------------------------ #
    # LangSmith reads these directly from `os.environ`, so we mirror them
    # here purely so the *rest of the app* has one place to check "is
    # tracing on?" without re-parsing env vars.
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="practical-12-tracing-observability", alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT")

    # ------------------------------------------------------------------ #
    # LangFuse (Part 2 of this practical)
    # ------------------------------------------------------------------ #
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    # Point this at https://cloud.langfuse.com (free tier, zero infra) or at
    # http://localhost:3000 if you are running the self-hosted stack in
    # docker-compose.yml. Both are valid for this practical — see README.
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")

    def export_tracing_env(self) -> None:
        """
        Push tracing config into `os.environ`.

        WHY: LangChain/LangGraph's tracing hooks and the LangFuse SDK read
        credentials straight from process environment variables at call
        time — they do not accept a settings object. pydantic-settings only
        *reads* env vars, it doesn't guarantee they are still set under the
        exact key names the SDKs expect (e.g. if they came from a renamed
        alias). Calling this once at startup (see `main.py`) guarantees both
        the modern `LANGSMITH_*` names and the older `LANGCHAIN_*` aliases
        (still used by some pinned LangChain versions) are present.
        """
        if self.langsmith_tracing:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
            os.environ["LANGSMITH_ENDPOINT"] = self.langsmith_endpoint
            # Legacy aliases some LangChain minor versions still check first.
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
            os.environ["LANGCHAIN_ENDPOINT"] = self.langsmith_endpoint


# Approximate Bedrock on-demand pricing, USD per 1,000 tokens.
# WHY a dict and not a hardcoded number in the cost function: Bedrock prices
# differ per model, and the whole point of this practical is comparing cost
# *between* nodes that use different models — so pricing has to be looked
# up per model_id, not assumed constant. Verify against the live pricing
# page before using this for a real budget decision.
NOVA_PRICING_PER_1K_TOKENS: dict[str, dict[Literal["input", "output"], float]] = {
    "amazon.nova-micro-v1:0": {"input": 0.000035, "output": 0.00014},
    "amazon.nova-lite-v1:0": {"input": 0.00006, "output": 0.00024},
}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached accessor so every module gets the *same* Settings instance
    instead of re-reading/re-validating the .env file on every import.
    `lru_cache` also makes it trivial to monkeypatch in tests
    (`get_settings.cache_clear()`), see tests/conftest.py.
    """
    return Settings()
