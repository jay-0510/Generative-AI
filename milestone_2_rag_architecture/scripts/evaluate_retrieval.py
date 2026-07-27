"""Measure Recall@K for the manually curated evaluation question set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from rag_app.bootstrap import build_services
from rag_app.config import Settings


def main() -> None:
    """Retrieve evidence for each question without invoking the answer LLM."""

    parser = argparse.ArgumentParser(description="Evaluate OpenSearch retrieval quality.")
    parser.add_argument("--questions", default="evaluation/questions.json")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    load_dotenv()
    services = build_services(Settings.from_env())
    cases = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    hits = 0
    results = []
    for case in cases:
        retrieved = services.rag.retrieve(case["question"], args.top_k)
        source_keys = [item.source_key for item in retrieved]
        matched = case["expected_source_key"] in source_keys
        hits += matched
        results.append({"id": case["id"], "matched": matched, "retrieved_source_keys": source_keys})
    print(json.dumps({"recall_at_k": hits / len(cases), "cases": results}, indent=2))


if __name__ == "__main__":
    main()
