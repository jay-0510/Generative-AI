"""
src/agents/sql_agent.py
-------------------------
The SQL agent node -- reached directly when the classifier says "sql", OR
as the fallback after the RAG agent reports rag_success=False (see
route_after_rag / route_after_classification in src/graph/workflow.py).

"Agent" here means the same Thought -> Action -> Observation shape used
throughout this course's agent practicals: reason about the question
(generate SQL), act (execute it through the safety guard in
utils/helpers.py), observe the result, and respond -- implemented
directly rather than via a generic AgentExecutor, so every step stays
inspectable inside a single graph node.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from config.db_init import SCHEMA_DESCRIPTION
from src.graph.state import GraphState
from src.llm_invoke import invoke_chat
from utils.helpers import execute_readonly_query

_SQL_GENERATION_PROMPT = f"""You write SQLite SELECT queries.

Schema:
{SCHEMA_DESCRIPTION}

Rules:
- Output ONLY the raw SQL query, nothing else: no explanation, no markdown \
code fences.
- Only ever write a single SELECT statement.
"""


def sql_agent_node(state: GraphState) -> dict:
    """Attempt to answer the query by generating and safely executing SQL.

    Sets sql_success=False (instead of raising) on any failure -- an
    unsafe/invalid query, a database error, or an empty result -- so
    route_after_sql can send the query on to error_node instead of
    crashing the whole graph run.
    """
    query = state["query"]

    generated_sql = invoke_chat(
        [
            SystemMessage(content=_SQL_GENERATION_PROMPT),
            HumanMessage(content=query),
        ]
    ).strip().strip("`").strip()

    try:
        columns, rows = execute_readonly_query(generated_sql)
    except Exception as exc:  # noqa: BLE001 - any failure here means "this attempt failed", not a crash
        return {
            "sql_success": False,
            "source": "sql",
            "error_message": f"SQL generation/execution failed: {exc}",
        }

    if not rows:
        return {
            "sql_success": False,
            "source": "sql",
            "error_message": "The generated query ran but returned no rows.",
        }

    result_preview = "\n".join(
        ", ".join(f"{col}={val}" for col, val in zip(columns, row)) for row in rows[:10]
    )
    summarize_prompt = (
        f"Question: {query}\n\nSQL query used: {generated_sql}\n\n"
        f"Query results:\n{result_preview}\n\n"
        "Write a concise, natural-language answer to the question using "
        "these results."
    )
    answer = invoke_chat([HumanMessage(content=summarize_prompt)])

    return {"answer": answer, "source": "sql", "sql_success": True}
