"""
src/graph/state.py
===================

WHY THIS FILE EXISTS
---------------------
LangGraph passes a single shared "state" dict between every node — each
node reads the keys it needs and returns a partial dict of updates, which
LangGraph merges into the running state. Defining that shape as a
`TypedDict` (rather than letting each node invent its own keys) means:

  1. Every node's input/output contract is documented in one place.
  2. Typos in state keys ("catagory" vs "category") get caught by a type
     checker instead of silently producing `None` three nodes later.
  3. When you look at a LangSmith/LangFuse trace, the state snapshot at
     each span matches this schema exactly, making traces much easier to
     read during the "find the slowest step" exercise.
"""

from __future__ import annotations

from typing import List, Literal, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state threaded through the classifier -> {sql_agent | rag_agent}
    -> END graph.

    `total=False` means no key is required up front — the graph starts with
    just `{"query": ..., "session_id": ...}` and each node adds its own
    keys as it runs, which is the normal LangGraph pattern for a routing
    graph where not every node populates every field.
    """

    # Set by the caller (main.py / notebook) before the graph runs.
    query: str
    session_id: str

    # Set by classifier_node.
    category: Optional[Literal["sql", "rag", "error"]]

    # Set by sql_agent_node.
    sql_query: Optional[str]
    sql_result: Optional[str]

    # Set by rag_agent_node.
    retrieved_docs: Optional[List[str]]

    # Set by whichever terminal node runs (sql_agent, rag_agent, or error_node).
    answer: Optional[str]
    error: Optional[str]
