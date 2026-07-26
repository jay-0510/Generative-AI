"""
bedrock_llm.py
--------------
Builds the Bedrock chat model the agent reasons with. Isolated in its own
module so swapping models/providers later (e.g. a different Bedrock model,
or a non-Bedrock provider) only touches this file, not agent.py or the
notebook.

Uses ChatBedrockConverse (Bedrock's Converse API) rather than ChatBedrock
(InvokeModel). Converse is the model-agnostic Bedrock API and takes
`temperature` as a first-class argument, so Amazon Nova needs no
provider-specific `model_kwargs` body shaping the way InvokeModel does.
"""

from langchain_aws import ChatBedrockConverse

from utils.config import settings


def get_bedrock_llm(temperature: float = 0.0) -> ChatBedrockConverse:
    """
    Returns a ChatBedrockConverse instance configured from Settings.

    temperature=0.0 by default: agent reasoning (deciding which tool to call)
    benefits from determinism far more than creative variation does — a
    wandering temperature makes the ReAct Thought/Action steps less reliable.
    """
    return ChatBedrockConverse(
        model=settings.bedrock_model_id,
        region_name=settings.aws_region,
        temperature=temperature,
    )
