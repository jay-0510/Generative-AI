"""
src/llm_invoke.py
------------------
The single place every agent goes through to talk to a model. Constructs
the shared Amazon Nova chat client and the Amazon Titan embeddings client,
and exposes one small wrapper function each agent calls instead of
importing `langchain_aws` directly.

WHY A DEDICATED "LLM INVOKE" MODULE
    Every agent (classifier, RAG, SQL) needs to call a chat model, and the
    RAG agent additionally needs an embeddings model. Without this module,
    each of those files would construct its own ChatBedrockConverse /
    BedrockEmbeddings instance, duplicating the model IDs, region, and
    temperature settings four times over. Centralizing it here means:
      1. There's exactly one place that knows how this project talks to
         AWS Bedrock -- swapping "amazon.nova-micro-v1:0" for
         "amazon.nova-lite-v1:0" (see config/settings.py) is a one-line
         change that affects every agent.
      2. Every agent module imports a plain function (`invoke_chat`), not
         a stateful client object -- easier to reason about and to mock
         in tests if you ever want to stub out model calls entirely.

WHY AMAZON NOVA + TITAN (AND NOTHING ELSE)
    This project is restricted to Amazon's own models end-to-end --
    Nova for chat, Titan for embeddings -- with no Anthropic, OpenAI, or
    Google model calls anywhere in the pipeline. `ChatBedrockConverse` and
    `BedrockEmbeddings` (both from `langchain_aws`) are Bedrock's
    LangChain-native clients; both talk to AWS using whichever credentials
    are present in the environment (an `AWS_BEARER_TOKEN_BEDROCK` API key,
    or a standard IAM access key pair -- see README.md).
"""

from typing import List

from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_core.messages import BaseMessage

from config.settings import AWS_REGION, LLM_TEMPERATURE, NOVA_MODEL_ID, TITAN_EMBEDDING_MODEL_ID

# Constructed once at import time and reused by every agent -- building
# these clients doesn't make a network call by itself (boto3 only talks to
# AWS when a method like .invoke() or .embed_query() actually runs), so
# this is safe to do at module load.
_chat_model = ChatBedrockConverse(
    model="amazon.nova-micro-v1:0",
    region_name="us-east-1",
    temperature=0.7,
)

_embedding_model = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1",
)


def invoke_chat(messages: List[BaseMessage]) -> str:
    """Send a list of messages to Amazon Nova and return the reply text.

    Args:
        messages: a list of LangChain message objects (SystemMessage,
            HumanMessage, etc.) -- the same input shape every LangChain
            chat model accepts.

    Returns:
        The model's reply as a plain string (`.content`), so calling
        agents never need to know they're talking to a
        `ChatBedrockConverse` object specifically.
    """
    response = _chat_model.invoke(messages)
    return response.content


def get_embedding_model() -> BedrockEmbeddings:
    """Return the shared Titan embeddings client.

    Exposed as a function (rather than importing `_embedding_model`
    directly) so callers go through one clear entry point, matching
    `invoke_chat` above -- and so a future swap to a different embeddings
    backend only touches this module.
    """
    return _embedding_model
