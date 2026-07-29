"""
tests/test_rag_agent.py
=========================

WHY THIS FILE EXISTS
---------------------
Like the SQL agent test, this pins down the node's control flow (retrieve
-> build prompt -> generate answer) without paying for real Bedrock
embedding + generation calls. The vector store itself
(`config.db_init.init_vector_store`) is faked out here rather than
exercised for real, since embedding calls are explicitly out of scope for
a fast unit test (see tests/test_setup_services.py docstring).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.rag_agent import rag_agent_node


@dataclass
class _FakeDoc:
    page_content: str


class _FakeRetriever:
    def __init__(self, docs: list[_FakeDoc]):
        self._docs = docs

    def invoke(self, query: str):
        return self._docs


class _FakeVectorStore:
    def __init__(self, docs: list[_FakeDoc]):
        self._docs = docs

    def as_retriever(self, search_kwargs=None):
        return _FakeRetriever(self._docs)


def test_rag_agent_retrieves_and_answers(monkeypatch, patch_chat_model):
    fake_docs = [_FakeDoc(page_content="Returns are accepted within 30 days.")]
    monkeypatch.setattr(
        "src.agents.rag_agent.init_vector_store",
        lambda: _FakeVectorStore(fake_docs),
    )
    patch_chat_model("src.agents.rag_agent", ["You can return your item within 30 days."])

    result = rag_agent_node({"query": "What is your return policy?", "session_id": "t1"})

    assert result["retrieved_docs"] == ["Returns are accepted within 30 days."]
    assert result["answer"] == "You can return your item within 30 days."
