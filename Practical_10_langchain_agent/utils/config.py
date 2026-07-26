"""
config.py
---------
Centralized settings for the agent, loaded from environment variables.
Why: keeps AWS region/model IDs and safety limits (max_iterations,
max_execution_time) in one place instead of hardcoded magic numbers
scattered across agent.py, bedrock_llm.py, and the notebook.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable settings snapshot, read once at import time."""

    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    bedrock_model_id: str = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
    bedrock_embedding_model_id: str = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1")

    # Safety guardrails for the agent loop (Practical 10 requirement):
    # max_iterations caps how many times the agent can call a tool before
    # being forcibly stopped; max_execution_time is a wall-clock backstop
    # in case a single tool call hangs rather than looping.
    max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "6"))
    max_execution_time_seconds: int = int(os.getenv("AGENT_MAX_EXECUTION_TIME", "60"))

    verbose: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"


settings = Settings()
