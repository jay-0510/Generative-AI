from rag_app.chunking import LangChainChunker
from rag_app.models import ExtractedDocument


def test_chunking_preserves_source_and_creates_stable_ids() -> None:
    document = ExtractedDocument(
        bucket="knowledge-bucket",
        key="policies/travel.pdf",
        text="Travel expenses need manager approval. " * 20,
    )
    chunker = LangChainChunker(chunk_size=100, chunk_overlap=20)

    first = chunker.split(document)
    second = chunker.split(document)

    assert len(first) > 1
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert all(chunk.source_key == "policies/travel.pdf" for chunk in first)
    assert first[1].chunk_index == 1
