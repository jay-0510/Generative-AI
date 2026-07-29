"""
tests/test_sql_agent.py
=========================

WHY THIS FILE EXISTS
---------------------
This node executes model-generated SQL against a real database file — the
two things most worth pinning down with tests are (1) a normal SELECT
round-trips correctly into a phrased answer, and (2) the safety guard
actually blocks a destructive statement BEFORE it reaches `sqlite3.connect(...)
.execute(...)`, since that guard is the only thing standing between a model
hallucination and real data loss.
"""

from __future__ import annotations

from config.db_init import init_sql_database
from src.agents.sql_agent import sql_agent_node


def test_sql_agent_executes_valid_select_and_phrases_answer(isolated_settings, patch_chat_model):
    init_sql_database()
    # First canned response = the "generated SQL", second = the "phrased answer".
    patch_chat_model(
        "src.agents.sql_agent",
        ["SELECT COUNT(*) as order_count FROM orders WHERE customer_id = 2;", "Vikram has placed 2 orders."],
    )

    result = sql_agent_node({"query": "How many orders has Vikram placed?", "session_id": "t1"})

    assert "order_count" in result["sql_result"]
    assert result["answer"] == "Vikram has placed 2 orders."
    assert "error" not in result


def test_sql_agent_blocks_unsafe_generated_sql(isolated_settings, patch_chat_model):
    init_sql_database()
    # Model "hallucinates" a destructive statement — must never execute.
    patch_chat_model("src.agents.sql_agent", ["DELETE FROM orders;"])

    result = sql_agent_node({"query": "delete everything", "session_id": "t1"})

    assert "error" in result
    assert "couldn't safely answer" in result["answer"]
