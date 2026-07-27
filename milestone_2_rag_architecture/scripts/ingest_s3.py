"""Command-line S3 ingestion entry point."""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from rag_app.bootstrap import build_services
from rag_app.config import Settings


def main() -> None:
    """Ingest an S3 prefix and return a machine-readable report."""

    load_dotenv()
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Ingest S3 documents into the RAG vector index.")
    parser.add_argument("--bucket", default=settings.s3_bucket)
    parser.add_argument("--prefix", default=settings.s3_prefix)
    args = parser.parse_args()
    report = build_services(settings).ingestion.ingest(args.bucket, args.prefix)
    print(json.dumps(report.__dict__, indent=2))
    if report.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
