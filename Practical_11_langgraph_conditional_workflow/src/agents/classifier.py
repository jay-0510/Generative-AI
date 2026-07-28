"""
src/agents/classifier.py
-------------------------
The classifier node: the entry point of the workflow, deciding whether a
query needs the SQL agent path or the RAG agent path.

WHY AN LLM CALL RATHER THAN KEYWORD MATCHING
    Phrasing varies too much for a fixed keyword list to generalize well
    (e.g. "how many people work in Engineering?" vs. "count employees per
    department" both need SQL but share almost no keywords). A small,
    cheap, temperature=0 Amazon Nova Micro call generalizes far better
    while staying fast -- exactly the kind of short classification task
    Nova Micro is built for.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from config.db_init import SCHEMA_DESCRIPTION
from src.graph.state import GraphState
from src.llm_invoke import invoke_chat

_CLASSIFIER_SYSTEM_PROMPT = f"""You are a query router. Decide whether the \
user's question should be answered by querying a SQL database or by \
retrieving from a document knowledge base.

The SQL database has this schema:
{SCHEMA_DESCRIPTION}
It holds structured records about employees and products: counts, \
averages, filtering, sorting, and other aggregates over those records.

The document knowledge base holds company POLICY text (remote work, \
vacation, onboarding, expenses, equipment, performance reviews, parental \
leave, code of conduct) -- conceptual/explanatory content, not structured \
records.

Reply with EXACTLY one word: SQL, RAG, or UNKNOWN. Reply UNKNOWN only if \
the question clearly matches neither domain at all."""


def classify_query_node(state: GraphState) -> dict:
    """Classify the incoming query as needing SQL, RAG, or UNKNOWN handling.

    Returns:
        {"query_type": "sql" | "rag" | "unknown"} -- read by
        route_after_classification in src/graph/workflow.py to choose
        which agent runs next.
    """
    messages = [
        SystemMessage(content=_CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=state["query"]),
    ]
    label = invoke_chat(messages).strip().upper()

    if "SQL" in label:
        query_type = "sql"
    elif "RAG" in label:
        query_type = "rag"
    else:
        query_type = "unknown"

    return {"query_type": query_type}
