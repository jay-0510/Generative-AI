"""
main.py
-------
Command-line entry point for the project -- runs the same workflow the
notebook demonstrates, without needing Jupyter. Useful for a quick
end-to-end smoke test after setup, or for running inside the Docker
container defined in docker-compose.yml.

WHY THIS EXISTS ALONGSIDE THE NOTEBOOK
    The notebook (notebooks/) is the primary teaching artifact -- explained
    step by step, with markdown between cells. main.py is the "just run
    it" version: a single script covering setup, graph construction, and a
    few demo queries, for anyone who wants to verify the whole pipeline
    works without opening Jupyter at all.

Usage:
    python main.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")


def _check_credentials() -> bool:
    """Print which required credentials are present, and return whether
    all of them are -- used to fail with a clear message before attempting
    any model call, rather than partway through a run."""
    has_aws = bool(
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or os.environ.get("AWS_ACCESS_KEY_ID")
    )
    print(f"{'OK  ' if has_aws else 'MISSING'} - AWS Bedrock credentials")
    if not has_aws:
        print("See README.md 'Setting up AWS Bedrock' before running main.py.")
    return has_aws


def main() -> None:
    if not _check_credentials():
        sys.exit(1)

    # Import AFTER the credential check and AFTER sys.path/load_dotenv are
    # set up -- src.llm_invoke constructs its Bedrock clients at import
    # time and reads config.settings, which reads the now-loaded env vars.
    from config.db_init import ensure_sample_database
    from src.graph.workflow import build_graph

    ensure_sample_database()
    graph = build_graph()

    demo_queries = [
        "How many employees work in the Engineering department, and what's their average salary?",
        "What is the company's policy on remote work?",
        "Tell me about Ethan Wright.",
    ]

    for query in demo_queries:
        result = graph.invoke({"query": query})
        print(f"\nQUERY: {query}")
        print(f"  answered by: {result.get('source')}")
        print(f"  ANSWER     : {result.get('answer')}")


if __name__ == "__main__":
    main()
