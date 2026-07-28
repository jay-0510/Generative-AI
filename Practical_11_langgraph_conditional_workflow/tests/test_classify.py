"""
tests/test_classify.py
------------------------
Tests for src/agents/classifier.py and the routing decision that reads
its output (route_after_classification, in src/graph/workflow.py).

`invoke_chat` is monkeypatched on `src.agents.classifier` (the module
that imported it), NOT on `src.llm_invoke` (where it's defined) --
`from src.llm_invoke import invoke_chat` creates a new name binding
inside classifier.py's own namespace, so patching the original wouldn't
affect the already-imported reference classify_query_node actually calls.
"""

import src.agents.classifier as classifier_module
from src.graph.workflow import route_after_classification


class TestClassifyQueryNode:
    def test_sql_label_maps_to_sql_query_type(self, monkeypatch):
        monkeypatch.setattr(classifier_module, "invoke_chat", lambda messages: "SQL")
        result = classifier_module.classify_query_node(
            {"query": "How many employees are in Engineering?"}
        )
        assert result == {"query_type": "sql"}

    def test_rag_label_maps_to_rag_query_type(self, monkeypatch):
        monkeypatch.setattr(classifier_module, "invoke_chat", lambda messages: "RAG")
        result = classifier_module.classify_query_node(
            {"query": "What is the vacation policy?"}
        )
        assert result == {"query_type": "rag"}

    def test_unknown_label_maps_to_unknown_query_type(self, monkeypatch):
        monkeypatch.setattr(classifier_module, "invoke_chat", lambda messages: "UNKNOWN")
        result = classifier_module.classify_query_node({"query": "What's the weather today?"})
        assert result == {"query_type": "unknown"}

    def test_label_parsing_is_case_and_whitespace_insensitive(self, monkeypatch):
        """The model's raw reply may come back lowercase or with stray
        whitespace/newlines -- the node must still parse it correctly."""
        monkeypatch.setattr(classifier_module, "invoke_chat", lambda messages: "  sql\n")
        result = classifier_module.classify_query_node({"query": "..."})
        assert result == {"query_type": "sql"}

    def test_unrecognized_reply_defaults_to_unknown_not_a_crash(self, monkeypatch):
        """If the model ever replies with something that isn't SQL/RAG/
        UNKNOWN, the node should degrade to "unknown" (which this
        project's routing then tries RAG first for) rather than raising."""
        monkeypatch.setattr(classifier_module, "invoke_chat", lambda messages: "I'm not sure!")
        result = classifier_module.classify_query_node({"query": "..."})
        assert result == {"query_type": "unknown"}


class TestRouteAfterClassification:
    def test_sql_routes_to_sql_agent(self):
        assert route_after_classification({"query_type": "sql"}) == "sql_agent"

    def test_rag_routes_to_rag_agent(self):
        assert route_after_classification({"query_type": "rag"}) == "rag_agent"

    def test_unknown_routes_to_rag_agent_first(self):
        """Per this project's design, an ambiguous classification still
        tries the RAG path first rather than going straight to SQL or
        error."""
        assert route_after_classification({"query_type": "unknown"}) == "rag_agent"

    def test_missing_query_type_defaults_to_rag_agent(self):
        assert route_after_classification({}) == "rag_agent"
