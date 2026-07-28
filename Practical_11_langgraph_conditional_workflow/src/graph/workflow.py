"""
src/graph/workflow.py
-----------------------
Routing decisions and graph assembly for the conditional workflow.

WHY ROUTING FUNCTIONS LIVE IN THIS MODULE
    Each routing function is a small, pure function: `(state) -> str`,
    read by LangGraph's `add_conditional_edges` to decide which node runs
    next. They're kept in the same module as `build_graph()` (rather than
    a separate file) because they only make sense in the context of the
    specific edges they're wired to just below them -- reading
    route_after_rag next to the `add_conditional_edges("rag_agent", ...)`
    call that uses it makes the routing table easy to verify at a glance.
    They're still plain, independently testable functions -- see
    tests/test_rag_agent.py and tests/test_sql_agent.py, which call them
    directly against hand-built state dicts.

WORKFLOW SHAPE

                              +-------------------+
                    START --> |  classify_query   |
                              +---------+---------+
                        "sql"           |            "rag" / "unknown"
                    +-------------------+-------------------+
                    v                                        v
             +-------------+                          +--------------+
             |  sql_agent  | <---------- fail -------- |   rag_agent  |
             +------+------+     (RAG didn't work      +------+-------+
                    |             out -> try SQL)              |
              success|                                    success|
                    v                                          v
                   END                                        END
                    |
                  fail
                    v
             +--------------+
             |  error_node  | --> END
             +--------------+

RAG is tried FIRST for anything that isn't unambiguously a SQL question.
If RAG doesn't work out, the graph automatically falls back to the SQL
agent before giving up -- and only if THAT also fails does the workflow
reach the explicit error node.
"""

from langgraph.graph import END, START, StateGraph

from src.agents import classify_query_node, error_node, rag_agent_node, sql_agent_node
from src.graph.state import GraphState


def route_after_classification(state: GraphState) -> str:
    """Route based on the classifier's decision.

    "sql"               -> go straight to the SQL agent (the question is
                            clearly about structured employee/product data).
    "rag" or "unknown"   -> try the RAG agent FIRST -- see route_after_rag
                            for what happens if that doesn't work out.
    """
    if state.get("query_type") == "sql":
        return "sql_agent"
    return "rag_agent"


def route_after_rag(state: GraphState) -> str:
    """Route after the RAG agent node runs.

    Success -> the workflow is done, route to END.
    Failure -> fall back to sql_agent rather than giving up immediately.
    """
    if state.get("rag_success"):
        return "end"
    return "sql_agent"


def route_after_sql(state: GraphState) -> str:
    """Route after the SQL agent node runs.

    Success -> the workflow is done, route to END.
    Failure -> both the primary/fallback attempts have now been exhausted,
               so route to the explicit error node for a graceful message
               instead of ending the run with no answer at all.
    """
    if state.get("sql_success"):
        return "end"
    return "error_node"


def build_graph():
    """Construct and compile the conditional workflow graph.

    Returns:
        A compiled LangGraph graph. Call `.invoke({"query": "..."})` to
        run it end to end, or `.get_graph().draw_mermaid()` /
        `.get_graph().draw_png()` to visualize its structure (see the
        notebook's visualization section).
    """
    builder = StateGraph(GraphState)

    # --- Register nodes -----------------------------------------------------
    builder.add_node("classify_query", classify_query_node)
    builder.add_node("rag_agent", rag_agent_node)
    builder.add_node("sql_agent", sql_agent_node)
    builder.add_node("error_node", error_node)

    # --- Wire edges ----------------------------------------------------------
    builder.add_edge(START, "classify_query")

    # Classifier fans out to exactly one of the two agent paths.
    builder.add_conditional_edges(
        "classify_query",
        route_after_classification,
        {"sql_agent": "sql_agent", "rag_agent": "rag_agent"},
    )

    # RAG succeeds -> END. RAG fails -> fall back to the SQL agent path.
    builder.add_conditional_edges(
        "rag_agent",
        route_after_rag,
        {"end": END, "sql_agent": "sql_agent"},
    )

    # SQL succeeds -> END. SQL fails (whether reached directly or via the
    # RAG fallback) -> the graceful error node.
    builder.add_conditional_edges(
        "sql_agent",
        route_after_sql,
        {"end": END, "error_node": "error_node"},
    )

    builder.add_edge("error_node", END)

    return builder.compile()
