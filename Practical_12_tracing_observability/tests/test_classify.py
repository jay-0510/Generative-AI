"""
tests/test_classify.py
========================

WHY THIS FILE EXISTS
---------------------
`classify_node` decides which specialist runs at all — a routing bug here
(e.g. "RAG" and "SQL" both matching the same branch) silently sends every
query to the wrong agent, which is a much harder bug to notice in
production than a crash would be. These tests pin down all three possible
outcomes (sql / rag / error) using a faked model response, so the routing
logic itself is verified without spending real Bedrock tokens.
"""

from __future__ import annotations

from src.agents.classifier import classify_node


def test_classify_routes_to_sql(patch_chat_model):
    patch_chat_model("src.agents.classifier", ["SQL"])
    result = classify_node({"query": "How many orders has Vikram placed?", "session_id": "t1"})
    assert result["category"] == "sql"


def test_classify_routes_to_rag(patch_chat_model):
    patch_chat_model("src.agents.classifier", ["RAG"])
    result = classify_node({"query": "What is the warranty on electronics?", "session_id": "t1"})
    assert result["category"] == "rag"


def test_classify_falls_back_to_error_on_unparseable_output(patch_chat_model):
    """
    If the model ever returns something that isn't clearly SQL or RAG, the
    node must fail SAFE into "error" rather than guessing — this is the
    behavior error_node.py depends on.
    """
    patch_chat_model("src.agents.classifier", ["I'm not sure what you mean"])
    result = classify_node({"query": "asdf", "session_id": "t1"})
    assert result["category"] == "error"
