"""
src/agents/error_node.py
==========================

WHY THIS FILE EXISTS
---------------------
Two failure paths lead here: (1) the classifier couldn't confidently label
a query as "sql" or "rag", or (2) the SQL agent generated a query that
failed the safety check in utils.helpers.is_safe_select_query. Rather than
letting either failure propagate as an unhandled exception (which would
show up in LangSmith/LangFuse as a broken red trace with no user-facing
message), routing both into one explicit terminal node means:

  1. The user always gets a graceful, on-brand fallback message.
  2. Every failure is a NORMAL (not exceptional) node execution, so it
     still traces cleanly and shows up as its own span — useful when
     you're scanning a trace for "why did this run take unusually long"
     and want to rule out silent failures as the cause.

This node makes no LLM call at all, so — unlike the other three nodes — it
should cost $0.00 and take well under a millisecond in both trace UIs. That
makes it a useful sanity check: if error_node ever shows meaningful
latency, something upstream (e.g. a slow state merge) is worth
investigating.
"""

from __future__ import annotations

from langfuse import observe

from src.graph.state import AgentState
from utils.helpers import timed


@observe(name="error_node")
@timed("error_node")
def error_node(state: AgentState) -> dict:
    """
    LangGraph node: produce a graceful fallback answer for any query the
    classifier or a specialist couldn't confidently handle.
    """
    reason = state.get("error", "I wasn't able to confidently classify that question.")
    return {
        "answer": (
            "I'm sorry, I couldn't process that request confidently. "
            "Could you rephrase it, or ask about an order/product (data lookup) "
            "or a store policy (returns, shipping, warranty, loyalty)? "
            f"(internal note: {reason})"
        )
    }
