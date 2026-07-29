"""
src/llm_invoke.py
==================

WHY THIS FILE EXISTS
---------------------
Three different nodes (classifier, sql_agent, rag_agent) each need a chat
model, and each is allowed to use a *different* Nova model (Micro vs Lite)
for cost/latency reasons (see config/settings.py). Without a shared
factory, each node would repeat the same `ChatBedrockConverse(...)`
boilerplate, and — more importantly for THIS practical — any tracing
callback we want attached to *every* LLM call would need to be duplicated
in three places too. Centralizing it here means:

  1. One line changes the model for cost experiments.
  2. One place to attach callbacks (e.g. the LangFuse CallbackHandler) so
     every node's LLM call is captured, not just the ones we remembered
     to instrument.

We use `ChatBedrockConverse` (not the older `ChatBedrock`) because it talks
to Bedrock's newer unified Converse API, which is what Nova models expect
and what returns clean `.usage_metadata` (input/output token counts) on
every response — and those counts are exactly what we need to compute
per-call cost for the "most expensive LLM call" part of this practical.
"""

from __future__ import annotations

from langchain_aws import ChatBedrockConverse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from config.settings import get_settings


def get_chat_model(model_id: str, temperature: float = 0.0) -> BaseChatModel:
    """
    Build a Bedrock chat model client for the given Nova model ID.

    Args:
        model_id: One of settings.nova_micro_model_id / nova_lite_model_id.
                  Passed explicitly (rather than hardcoded) so each node
                  stays a one-line decision about which model it needs.
        temperature: 0.0 by default — both the classifier and the SQL agent
                     need deterministic, repeatable output (a routing label
                     and valid SQL), not creative variation. The RAG agent
                     node overrides this slightly higher for more natural
                     prose answers.

    Returns:
        A LangChain `BaseChatModel` — every LangGraph node treats this the
        same way regardless of which Nova variant is underneath, via
        `.invoke(messages)`.
    """
    settings = get_settings()
    return ChatBedrockConverse(
        model="amazon.nova-micro-v1:0",
        region_name="us-east-1",
        temperature=0.7,
    )


def invoke_with_usage(model: BaseChatModel, messages: list, config: RunnableConfig | None = None):
    """
    Thin `.invoke()` wrapper that also returns token usage alongside the
    response, so callers don't have to know where `usage_metadata` lives
    on the AIMessage.

    WHY a wrapper instead of calling `.invoke()` directly everywhere:
    `usage_metadata` is populated by the Bedrock Converse integration but
    its exact shape (dict vs. object) has shifted between langchain-aws
    releases before. Isolating that lookup in one helper means a future
    version bump only breaks one function, not every agent node.

    Returns:
        (ai_message, usage_dict) where usage_dict has keys
        "input_tokens", "output_tokens", "total_tokens" (0 if unavailable).
    """
    response = model.invoke(messages, config=config)
    usage = getattr(response, "usage_metadata", None) or {}
    usage_dict = {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    return response, usage_dict
