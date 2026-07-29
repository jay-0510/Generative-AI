"""
src/agents/sql_agent.py
=========================

WHY THIS FILE EXISTS
---------------------
This node handles the "SQL" branch of the routing graph: it turns a natural
-language question into SQL, executes it against the read-only SQLite
database from config/db_init.py, and turns the raw row results back into a
natural-language sentence. It is intentionally split into two LLM calls
(generate SQL, then phrase the answer) rather than one, so that:

  1. Each call is small and single-purpose (better traced spans — you can
     see in LangSmith/LangFuse exactly how much of this node's latency was
     "thinking of SQL" vs. "phrasing the answer").
  2. The SQL-generation call can be validated/rejected (see
     `utils.helpers.is_safe_select_query`) BEFORE anything touches the
     database, without also having committed to a final answer phrasing.

WHY NOVA LITE (not Micro): generating syntactically correct SQL against a
multi-table schema is a genuine reasoning task, not a single-label
classification — Nova Lite's larger context/greater capability justifies
its ~1.7x cost over Micro here. This is the "expensive" node this practical
expects you to be able to point to in a trace.
"""

from __future__ import annotations

import sqlite3

from langfuse import observe

from config.db_init import get_schema_description, init_sql_database
from config.settings import get_settings
from src.graph.state import AgentState
from src.llm_invoke import get_chat_model, invoke_with_usage
from utils.helpers import estimate_cost, is_safe_select_query, timed

_SQL_GENERATION_PROMPT = """You write SQLite SELECT queries. Given a database schema and a
question, output ONLY the SQL query — no markdown fences, no explanation.

Schema:
{schema}

Question: {question}

SQL query:"""

_ANSWER_PHRASING_PROMPT = """Answer the user's question in one short, natural sentence using
only the query result data below. Do not mention SQL or databases.

Question: {question}
Query result (rows): {rows}

Answer:"""


@observe(name="sql_agent_node")
@timed("sql_agent")
def sql_agent_node(state: AgentState) -> dict:
    """
    LangGraph node: turn `state["query"]` into SQL, run it, and phrase the
    result. Returns `sql_query`, `sql_result`, and `answer` state updates.
    """
    settings = get_settings()
    model = get_chat_model(settings.sql_agent_model_id, temperature=0.0)
    db_path = init_sql_database()  # idempotent: no-op if already set up.

    # --- Step 1: natural language -> SQL -------------------------------
    gen_prompt = _SQL_GENERATION_PROMPT.format(schema=get_schema_description(), question=state["query"])
    sql_response, gen_usage = invoke_with_usage(model, [("human", gen_prompt)])
    sql_query = sql_response.content.strip().strip("`").strip()
    gen_cost = estimate_cost(settings.sql_agent_model_id, gen_usage["input_tokens"], gen_usage["output_tokens"])
    print(f"[cost] sql_agent/generate ({settings.sql_agent_model_id}): ${gen_cost:.6f}")

    if not is_safe_select_query(sql_query):
        return {
            "sql_query": sql_query,
            "error": "Generated SQL failed the read-only SELECT safety check.",
            "answer": "I couldn't safely answer that from the database — could you rephrase the question?",
        }

    # --- Step 2: execute -------------------------------------------------
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(sql_query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
    finally:
        conn.close()
    result_rows = [dict(zip(columns, row)) for row in rows]

    # --- Step 3: rows -> natural language answer -------------------------
    answer_prompt = _ANSWER_PHRASING_PROMPT.format(question=state["query"], rows=result_rows)
    answer_response, ans_usage = invoke_with_usage(model, [("human", answer_prompt)])
    ans_cost = estimate_cost(settings.sql_agent_model_id, ans_usage["input_tokens"], ans_usage["output_tokens"])
    print(f"[cost] sql_agent/phrase ({settings.sql_agent_model_id}): ${ans_cost:.6f}")

    return {
        "sql_query": sql_query,
        "sql_result": str(result_rows),
        "answer": answer_response.content.strip(),
    }
