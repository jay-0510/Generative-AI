"""S3 discovery and PyMuPDF text extraction."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import fitz

from .models import ExtractedDocument

SUPPORTED_PDF_EXTENSIONS = {".pdf"}


class S3PdfLoader:
    """Load text-based PDF documents directly from S3 with PyMuPDF."""

    def __init__(self, s3_client: Any) -> None:
        self.s3 = s3_client

    def list_keys(self, bucket: str, prefix: str) -> Iterator[str]:
        """Yield PDF document keys beneath an S3 prefix."""

        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if any(key.lower().endswith(extension) for extension in SUPPORTED_PDF_EXTENSIONS):
                    yield key

    def extract(self, bucket: str, key: str) -> ExtractedDocument:
        """Download one PDF and extract its page text with PyMuPDF."""

        response = self.s3.get_object(Bucket=bucket, Key=key)
        pdf_bytes = response["Body"].read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            text = "\n".join(page.get_text("text").strip() for page in document).strip()
        if not text.strip():
            raise ValueError(
                f"PyMuPDF returned no text for s3://{bucket}/{key}. "
                "Scanned PDFs require OCR before ingestion."
            )
        return ExtractedDocument(bucket=bucket, key=key, text=text)
