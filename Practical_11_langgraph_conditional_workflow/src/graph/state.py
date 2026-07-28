"""
src/graph/state.py
-------------------
Defines the shared state schema that flows through every node in the
LangGraph workflow.

WHY A TypedDict
    LangGraph passes a single state object between nodes. Each node reads
    what it needs from state and returns a dict of ONLY the keys it wants
    to update; LangGraph merges that into the overall state automatically.
    Declaring the schema as a TypedDict gives editors/type-checkers
    visibility into exactly what keys exist, and documents -- in one place
    -- everything that flows through the graph.

WHY total=False
    Not every node sets every field on every run (e.g. `rag_success` is
    only ever set by the RAG agent, and only some runs even reach it).
    Marking the TypedDict as total=False means "all keys are optional,"
    matching how LangGraph actually uses it: the state dict grows
    incrementally as different nodes contribute fields.
"""

from typing import Optional, TypedDict


class GraphState(TypedDict, total=False):
    """Shared state passed between every node in the workflow.

    Attributes:
        query: The original natural-language question from the user.
        query_type: Set by the classifier -- one of "sql", "rag", or
            "unknown".
        answer: The final natural-language answer, once produced by
            whichever path succeeds (or the error node's fallback message).
        source: Which agent ultimately produced `answer` -- "rag", "sql",
            or "error".
        rag_success: Set by the RAG agent -- True if it found sufficiently
            relevant context and produced an answer, False otherwise
            (triggering the SQL fallback).
        sql_success: Set by the SQL agent -- True if it generated and
            executed a valid SQL query and produced an answer, False
            otherwise (triggering the error node).
        error_message: Set on a failed SQL attempt -- a short, technical
            reason that error_node turns into a user-facing message.
    """

    query: str
    query_type: Optional[str]
    answer: Optional[str]
    source: Optional[str]
    rag_success: Optional[bool]
    sql_success: Optional[bool]
    error_message: Optional[str]
