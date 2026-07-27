import fitz

from rag_app.document_loader import S3PdfLoader


class FakePaginator:
    def paginate(self, **_kwargs):
        return [{"Contents": [{"Key": "docs/guide.pdf"}, {"Key": "docs/readme.txt"}]}]


class FakeS3:
    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator()

    def get_object(self, **kwargs):
        assert kwargs == {"Bucket": "bucket", "Key": "docs/guide.pdf"}
        return {"Body": FakeBody(_pdf_with_text("First page\nSecond page"))}


class FakeBody:
    def __init__(self, content: bytes):
        self.content = content

    def read(self) -> bytes:
        return self.content


def _pdf_with_text(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def test_loader_filters_pdfs_and_extracts_text_with_pymupdf() -> None:
    loader = S3PdfLoader(FakeS3())

    assert list(loader.list_keys("bucket", "docs/")) == ["docs/guide.pdf"]
    document = loader.extract("bucket", "docs/guide.pdf")

    assert document.text == "First page\nSecond page"
