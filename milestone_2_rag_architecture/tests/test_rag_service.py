import pytest

from rag_app.models import RetrievedChunk
from rag_app.services import RAGService


class FakeEmbeddings:
    def embed_query(self, text):
        assert text == "What is the policy?"
        return [0.1, 0.2]


class FakeStore:
    def search(self, embedding, top_k):
        assert embedding == [0.1, 0.2]
        assert top_k == 3
        return [RetrievedChunk("id-1", "Policy requires approval.", "bucket", "policy.pdf", 0, 0.95)]


class FakeGenerator:
    def generate(self, question, sources):
        assert question == "What is the policy?"
        assert len(sources) == 1
        return "Manager approval is required [1]."


def test_rag_service_returns_answer_and_sources() -> None:
    service = RAGService(FakeEmbeddings(), FakeStore(), FakeGenerator())

    result = service.ask("  What is the policy?  ", top_k=3)

    assert result.answer == "Manager approval is required [1]."
    assert result.sources[0].source_key == "policy.pdf"


def test_rag_service_rejects_blank_question() -> None:
    service = RAGService(FakeEmbeddings(), FakeStore(), FakeGenerator())

    with pytest.raises(ValueError, match="must not be empty"):
        service.ask("   ", top_k=3)
