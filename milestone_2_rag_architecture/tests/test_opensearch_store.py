from rag_app.models import DocumentChunk
from rag_app.opensearch_store import OpenSearchVectorStore


def test_serverless_indexing_omits_client_supplied_document_ids(monkeypatch) -> None:
    actions_seen = []

    def fake_bulk(_client, actions, refresh):
        assert refresh is False
        actions_seen.extend(actions)

    monkeypatch.setattr("rag_app.opensearch_store.helpers.bulk", fake_bulk)
    store = OpenSearchVectorStore(object(), "documents", 1024, use_document_ids=False)
    chunk = DocumentChunk("stable-id", "Policy text", "bucket", "policy.pdf", 0)

    store.upsert_chunks([chunk], [[0.1, 0.2]])

    assert "_id" not in actions_seen[0]
    assert actions_seen[0]["_source"]["text"] == "Policy text"


def test_managed_domain_indexing_uses_stable_document_ids(monkeypatch) -> None:
    actions_seen = []

    monkeypatch.setattr(
        "rag_app.opensearch_store.helpers.bulk", lambda _client, actions, refresh: actions_seen.extend(actions)
    )
    store = OpenSearchVectorStore(object(), "documents", 1024)
    chunk = DocumentChunk("stable-id", "Policy text", "bucket", "policy.pdf", 0)

    store.upsert_chunks([chunk], [[0.1, 0.2]])

    assert actions_seen[0]["_id"] == "stable-id"


def test_serverless_refresh_is_a_noop() -> None:
    class Indices:
        def refresh(self, **_kwargs):
            raise AssertionError("OpenSearch Serverless does not support index refresh")

    class Client:
        indices = Indices()

    OpenSearchVectorStore(Client(), "documents", 1024, supports_refresh=False).refresh()
