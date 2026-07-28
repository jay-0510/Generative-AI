"""
src/agents/error_node.py
--------------------------
The graph's explicit, graceful failure path -- reached only after BOTH the
RAG attempt and the SQL fallback attempt have failed to produce a
confident answer (see src/graph/workflow.py for the exact routing). The
workflow never just crashes or silently returns nothing; it always ends
with a clear, user-facing message.
"""

from src.graph.state import GraphState


def error_node(state: GraphState) -> dict:
    """Produce a graceful, user-facing message when no path could answer.

    Makes no LLM call -- there's nothing left to reason about once both
    agents have failed, and a template message is more predictable (and
    cheaper) than asking a model to explain a failure it wasn't involved
    in producing.
    """
    reason = state.get("error_message", "no matching data was found")
    message = (
        "I wasn't able to answer that confidently using either the "
        "knowledge base or the database. "
        f"({reason}) Try rephrasing the question, or ask something more "
        "specific to company policies (for the knowledge base) or "
        "employees/products (for the database)."
    )
    return {"answer": message, "source": "error"}
