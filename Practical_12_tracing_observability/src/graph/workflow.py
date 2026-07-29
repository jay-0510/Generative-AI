"""
src/graph/workflow.py
=======================

WHY THIS FILE EXISTS
---------------------
This is the only file that assembles the graph shape — every other module
(agents/*, state.py) is a self-contained piece that this file wires
together. Keeping "which node connects to which" in one place, separate
from "what each node does," means the graph topology can be read and
changed (e.g. adding a fourth specialist) without touching any node's
internal logic.

WHY LANGGRAPH (rather than a plain if/elif router function) — and why this
matters for THIS practical specifically: LangGraph instruments the graph
itself as a traced run tree. Every node execution becomes its own span in
LangSmith automatically (no code changes needed beyond the environment
variables in config/settings.py), with parent/child relationships that
mirror this exact `add_edge`/`add_conditional_edges` structure. A hand-
rolled if/elif router would still be traceable, but you'd have to add a
`@traceable` decorator to each branch yourself to get the same visual
breakdown — LangGraph gives it "for free," which is precisely the property
this practical is exploring.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.classifier import classify_node
from src.agents.error_node import error_node
from src.agents.rag_agent import rag_agent_node
from src.agents.sql_agent import sql_agent_node
from src.graph.state import AgentState


def _route_after_classification(state: AgentState) -> str:
    """
    Conditional-edge function: reads the category the classifier set and
    returns the name of the next node to run.

    WHY a separate function instead of inlining a lambda in `build_graph`:
    LangGraph's tracing labels conditional-edge spans with this function's
    `__name__`, so a named function (rather than an anonymous lambda) gives
    a readable label in the trace UI instead of "<lambda>".
    """
    category = state.get("category", "error")
    return {"sql": "sql_agent", "rag": "rag_agent"}.get(category, "error_node")


def build_graph():
    """
    Construct and compile the LangGraph workflow.

    Graph shape:

        classifier ──┬─(sql)──> sql_agent ──> END
                      ├─(rag)──> rag_agent ──> END
                      └─(error)─> error_node ──> END

    Returns:
        A compiled `CompiledGraph` — call `.invoke(state)` or `.stream(state)`
        on it. Both `main.py` and the notebook use this same function so
        there is exactly one place that defines the graph's structure.
    """
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("classifier", classify_node)
    graph_builder.add_node("sql_agent", sql_agent_node)
    graph_builder.add_node("rag_agent", rag_agent_node)
    graph_builder.add_node("error_node", error_node)

    graph_builder.set_entry_point("classifier")
    graph_builder.add_conditional_edges(
        "classifier",
        _route_after_classification,
        {"sql_agent": "sql_agent", "rag_agent": "rag_agent", "error_node": "error_node"},
    )
    graph_builder.add_edge("sql_agent", END)
    graph_builder.add_edge("rag_agent", END)
    graph_builder.add_edge("error_node", END)

    return graph_builder.compile()
