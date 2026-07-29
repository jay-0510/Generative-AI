"""
src/agents/classifier.py
=========================

WHY THIS FILE EXISTS
---------------------
The graph has two very different specialists (a text-to-SQL agent and a
document-retrieval agent) and one query at a time only ever needs one of
them. Rather than running both and discarding an answer, a cheap upfront
routing step decides which specialist to invoke — this is the "classifier
-> specialist" pattern that keeps cost/latency down in production agent
systems, and it is also *exactly* the kind of node that should use the
cheapest available model, because its output is a single label, not a
reasoned answer.

WHY NOVA MICRO SPECIFICALLY (not Nova Lite):
This is the cheapest text model on Bedrock ($0.035 / $0.14 per 1M tokens —
see config/settings.py) and the task is a one-word classification, well
within a small model's ability. Routing this node to Micro instead of Lite
is a deliberate, visible cost-optimization choice you can point to when
comparing "most expensive LLM call" in the trace UIs later: this node
should show up as the CHEAPEST call in both LangSmith and LangFuse, while
sql_agent/rag_agent (on Nova Lite, doing actual reasoning) cost more.

WHY PLAIN-TEXT PARSING INSTEAD OF TOOL-CALLING STRUCTURED OUTPUT:
`with_structured_output()` (via forced tool-calling) is the "proper" way to
get a typed classification out of most chat models, but small models can be
unreliable with a forced `tool_choice` on Bedrock's Converse API. For a
single-word routing decision, asking for plain text and validating it with
a strict `if/elif` is simpler, cheaper (no extra tool-definition tokens),
and just as robust — and it fails into `error_node` cleanly if the model
ever returns something unexpected.
"""

from __future__ import annotations

from langfuse import observe

from config.settings import get_settings
from src.graph.state import AgentState
from src.llm_invoke import get_chat_model, invoke_with_usage
from utils.helpers import estimate_cost, timed

_CLASSIFIER_SYSTEM_PROMPT = """You are a routing classifier for a customer-support assistant.
Given a user's question, decide which specialist should answer it:

- Reply with exactly "SQL" if the question is about specific data that
  lives in a database: customer orders, product prices, order history,
  quantities, dates, or counts (e.g. "how many keyboards did Vikram buy?").
- Reply with exactly "RAG" if the question is about general company policy
  or knowledge: returns, shipping, warranty, or loyalty program rules
  (e.g. "what is your return window?").

Reply with ONLY the single word SQL or RAG. No punctuation, no explanation.
"""


@observe(name="classifier_node")  # LangFuse: function-level span (Part 2 of this practical).
@timed("classifier")
def classify_node(state: AgentState) -> dict:
    """
    LangGraph node: read `state["query"]`, decide "sql" or "rag", return the
    routing decision as a partial state update.

    Note on tracing: this function is wrapped with LangFuse's `@observe()`
    decorator (the exact mechanism the practical asks us to add), which
    creates a named span every time this node runs. We do NOT also need a
    manual LangSmith decorator here — as long as `LANGSMITH_TRACING=true`
    is set in the environment (config/settings.py -> export_tracing_env),
    the `.invoke()` call on the Bedrock chat model below is captured
    automatically by LangChain's built-in tracer, nested correctly under
    this node because LangGraph itself is also traced automatically.
    That asymmetry — LangSmith needs zero code changes, LangFuse needs one
    decorator per function you want a span for — is itself one of the two
    UIs' most practical differences, worth calling out in the comparison
    write-up (see README "LangSmith vs LangFuse" section).
    """
    settings = get_settings()
    model = get_chat_model(settings.classifier_model_id, temperature=0.0)

    messages = [
        ("system", _CLASSIFIER_SYSTEM_PROMPT),
        ("human", state["query"]),
    ]
    response, usage = invoke_with_usage(model, messages)
    cost = estimate_cost(settings.classifier_model_id, usage["input_tokens"], usage["output_tokens"])
    print(f"[cost] classifier ({settings.classifier_model_id}): ${cost:.6f} ({usage['total_tokens']} tokens)")

    label = response.content.strip().upper()
    if "SQL" in label:
        category = "sql"
    elif "RAG" in label:
        category = "rag"
    else:
        # Model returned something unparseable -> fail safe into error_node
        # rather than guessing, so a bad classification never silently
        # reaches the wrong specialist.
        category = "error"

    return {"category": category}
