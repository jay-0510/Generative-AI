"""
tests/test_rag_agent.py
-------------------------
Tests for src/agents/rag_agent.py's retrieval-confidence logic and node
behavior, plus route_after_rag (src/graph/workflow.py).

No real FAISS index or Titan embedding call happens here: `_get_vectorstore`
is monkeypatched to return a small fake object exposing only the one
method `retrieve_with_confidence` actually calls
(`similarity_search_with_score`), so the relevance-threshold LOGIC is
tested in isolation from the embedding model that would normally produce
those scores.
"""

from langchain_core.documents import Document

import src.agents.rag_agent as rag_module
from src.graph.workflow import route_after_rag


class _FakeVectorStore:
    """Stands in for a real FAISS vector store in tests -- exposes just
    enough surface (similarity_search_with_score) for
    retrieve_with_confidence to work against, with fully controlled,
    deterministic (document, score) pairs."""

    def __init__(self, results):
        self._results = results

    def similarity_search_with_score(self, query, k=3):
        return self._results


class TestRetrieveWithConfidence:
    def test_low_distance_score_is_considered_relevant(self, monkeypatch):
        doc = Document(page_content="Remote work is allowed up to 3 days/week.")
        monkeypatch.setattr(
            rag_module, "_get_vectorstore", lambda: _FakeVectorStore([(doc, 0.1)])
        )
        chunks, is_relevant = rag_module.retrieve_with_confidence("remote work policy")
        assert is_relevant is True
        assert chunks[0].page_content.startswith("Remote work")

    def test_high_distance_score_is_considered_not_relevant(self, monkeypatch):
        """A score well above RAG_RELEVANCE_DISTANCE_THRESHOLD means even
        the closest chunk isn't a real match -- retrieve_with_confidence
        must report that honestly rather than returning it anyway."""
        doc = Document(page_content="Completely unrelated chunk.")
        monkeypatch.setattr(
            rag_module, "_get_vectorstore", lambda: _FakeVectorStore([(doc, 5.0)])
        )
        _chunks, is_relevant = rag_module.retrieve_with_confidence("obscure question")
        assert is_relevant is False

    def test_empty_results_are_not_relevant(self, monkeypatch):
        monkeypatch.setattr(rag_module, "_get_vectorstore", lambda: _FakeVectorStore([]))
        chunks, is_relevant = rag_module.retrieve_with_confidence("anything")
        assert chunks == []
        assert is_relevant is False

    def test_relevance_uses_the_best_of_several_scores(self, monkeypatch):
        """Even if some retrieved chunks are weak matches, one strong
        match among them should still be enough to count as relevant."""
        weak = Document(page_content="weak match")
        strong = Document(page_content="strong match")
        monkeypatch.setattr(
            rag_module,
            "_get_vectorstore",
            lambda: _FakeVectorStore([(weak, 0.9), (strong, 0.05)]),
        )
        _chunks, is_relevant = rag_module.retrieve_with_confidence("query")
        assert is_relevant is True


class TestRagAgentNode:
    def test_reports_failure_without_calling_the_llm_when_not_relevant(self, monkeypatch):
        """When nothing relevant is found, the node must NOT call the LLM
        at all (to avoid hallucinating from irrelevant context) -- this
        test fails loudly if it ever does, via the RuntimeError below."""
        monkeypatch.setattr(
            rag_module, "retrieve_with_confidence", lambda query, k=3: ([], False)
        )

        def _should_not_be_called(messages):
            raise RuntimeError("invoke_chat must not be called when retrieval is not relevant")

        monkeypatch.setattr(rag_module, "invoke_chat", _should_not_be_called)

        result = rag_module.rag_agent_node({"query": "totally unrelated question"})
        assert result == {"rag_success": False, "source": "rag"}

    def test_returns_answer_when_relevant(self, monkeypatch):
        doc = Document(page_content="Employees accrue 15 vacation days per year.")
        monkeypatch.setattr(
            rag_module, "retrieve_with_confidence", lambda query, k=3: ([doc], True)
        )
        monkeypatch.setattr(
            rag_module, "invoke_chat", lambda messages: "You accrue 15 vacation days per year."
        )

        result = rag_module.rag_agent_node({"query": "How much vacation do I get?"})
        assert result["rag_success"] is True
        assert result["source"] == "rag"
        assert "15 vacation days" in result["answer"]


class TestRouteAfterRag:
    def test_success_routes_to_end(self):
        assert route_after_rag({"rag_success": True}) == "end"

    def test_failure_falls_back_to_sql_agent(self):
        """The core fallback behavior this project is built around: RAG
        failing routes to sql_agent, not to error."""
        assert route_after_rag({"rag_success": False}) == "sql_agent"

    def test_missing_flag_treated_as_failure(self):
        assert route_after_rag({}) == "sql_agent"
