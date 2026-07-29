"""
utils/helpers.py
=================

WHY THIS FILE EXISTS
---------------------
Two small, reusable pieces of logic that don't belong to any single node:

  1. `timed()` — a decorator that measures wall-clock time locally. This is
     NOT a replacement for LangSmith/LangFuse latency numbers; it exists so
     you can *sanity-check* what the trace UIs report. If your own stopwatch
     says the SQL node took 400ms but LangSmith's trace says 4 seconds, that
     gap itself is useful signal (usually network/cold-start overhead) worth
     noting in the comparison write-up this practical asks for.

  2. `estimate_cost()` — turns a token-usage dict into a dollar estimate
     using the pricing table in config/settings.py. LangSmith can show cost
     automatically for model providers it has built-in pricing for; Bedrock
     Nova pricing is not always one of them, so we compute it ourselves and
     log it as run metadata (see src/agents/*.py) rather than relying on the
     UI to get the number right.
"""

from __future__ import annotations

import functools
import time
from typing import Callable, TypeVar

from config.settings import NOVA_PRICING_PER_1K_TOKENS

F = TypeVar("F", bound=Callable)


def timed(step_name: str) -> Callable[[F], F]:
    """
    Decorator that prints/records how long a node function took to run.

    Args:
        step_name: A short label (e.g. "classifier", "sql_agent") used in
                   the printed line and attached to the return value, so
                   the caller can aggregate step timings across a run
                   without re-deriving them from log text.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"[timed] {step_name}: {elapsed_ms:.1f} ms")
            if isinstance(result, dict):
                result.setdefault("_local_timings_ms", {})[step_name] = round(elapsed_ms, 1)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate USD cost of one LLM call from its token usage.

    WHY per-model lookup instead of one flat rate: Nova Micro and Nova Lite
    are priced differently (see config/settings.py), and the entire point
    of routing the classifier to Micro and the SQL/RAG agents to Lite is a
    cost trade-off — so the estimator has to reflect that difference to be
    useful for the "most expensive LLM call" comparison this practical asks
    for.

    Returns 0.0 for an unrecognized model_id rather than raising, since this
    is a best-effort estimate used for logging/comparison, not billing.
    """
    rates = NOVA_PRICING_PER_1K_TOKENS.get(model_id)
    if rates is None:
        return 0.0
    return (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]


def is_safe_select_query(sql: str) -> bool:
    """
    Guard-rail for the SQL agent: only allow single, read-only SELECT
    statements to actually execute against the database.

    WHY THIS EXISTS: a text-to-SQL LLM occasionally hallucinates a DROP/
    DELETE/UPDATE statement, especially under an ambiguous prompt. Since
    this node's whole job is to run *model-generated* SQL, we don't trust
    it by construction — this is a cheap, explicit allow-list check, not a
    substitute for running the DB user with read-only permissions in a
    real deployment (which you still should do).
    """
    normalized = sql.strip().strip(";").strip().lower()
    if not normalized.startswith("select"):
        return False
    forbidden_keywords = ("insert", "update", "delete", "drop", "alter", "attach", "pragma", ";")
    return not any(keyword in normalized for keyword in forbidden_keywords)
