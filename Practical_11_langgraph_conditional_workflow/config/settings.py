"""
config/settings.py
-------------------
Centralized, environment-driven configuration for the whole project.

WHY A SINGLE SETTINGS MODULE
    Model IDs, the AWS region, file paths, and tuning constants (like the
    RAG relevance threshold) were previously scattered across whichever
    module happened to need them. Pulling them into one place means:
      1. There is exactly ONE place to look when you want to swap a model
         (e.g. amazon.nova-micro-v1:0 -> amazon.nova-lite-v1:0) or retune
         a threshold.
      2. Every other module imports *values*, not `os.environ.get(...)`
         calls -- so a missing/misspelled env var fails fast, in one
         place, with one clear error, instead of silently defaulting deep
         inside some unrelated function.

WHY DEFAULTS ARE ATTACHED TO NON-SECRET VALUES ONLY
    Model IDs, the region, and thresholds have sensible defaults baked in
    here because they're not sensitive and are fine to version-control.
    AWS/LangSmith/LangFuse credentials are NEVER given a default and are
    read directly from the environment wherever they're needed (via
    python-dotenv loading .env) -- see README.md / .env.example.
"""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
DB_PATH = DATA_DIR / "company_sample.db"
POLICY_DOC_PATH = SAMPLE_DOCS_DIR / "company_policies.txt"

# --- AWS Bedrock ---------------------------------------------------------
# Per this project's requirement: Amazon models only (Nova for chat, Titan
# for embeddings) -- no Anthropic/OpenAI/Google calls anywhere.
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
NOVA_MODEL_ID = os.environ.get("NOVA_MODEL_ID", "amazon.nova-micro-v1:0")
TITAN_EMBEDDING_MODEL_ID = os.environ.get(
    "TITAN_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
)
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))

# --- RAG tuning ------------------------------------------------------------
# FAISS's similarity_search_with_score uses L2 distance: LOWER is MORE
# similar. This threshold separates "genuinely about company policy"
# queries from off-topic ones on the sample corpus with Titan Text
# Embeddings V2 -- re-tune if you swap in your own documents or a
# different embedding model (see src/agents/rag_agent.py).
RAG_RELEVANCE_DISTANCE_THRESHOLD = float(
    os.environ.get("RAG_RELEVANCE_DISTANCE_THRESHOLD", "0.55")
)
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
RAG_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "300"))
RAG_CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "50"))

# --- Tracing (optional -- Practical 12 territory, included here so this
# project can opt in later without restructuring anything) -----------------
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "practical11-langgraph-conditional")
