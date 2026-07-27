from fastapi.testclient import TestClient

from rag_app.api import create_app
from rag_app.models import Answer, RetrievedChunk


class FakeRAGService:
    def ask(self, question, top_k):
        assert question == "Where is the policy?"
        assert top_k == 4
        return Answer(
            answer="It is in the policy document [1].",
            sources=[RetrievedChunk("id-1", "Policy text", "bucket", "docs/policy.pdf", 2, 0.88)],
        )


def test_ask_endpoint_returns_answer_and_source() -> None:
    with TestClient(create_app(rag_service=FakeRAGService())) as client:
        response = client.post("/ask", json={"question": "Where is the policy?"})

    assert response.status_code == 200
    assert response.json()["sources"][0]["s3_uri"] == "s3://bucket/docs/policy.pdf"


def test_ask_endpoint_validates_empty_question() -> None:
    with TestClient(create_app(rag_service=FakeRAGService())) as client:
        response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422
