"""
conversation_memory.py
-----------------------
Wraps LangChain's ConversationBufferMemory so the agent can answer follow-up
questions that refer back to earlier turns (e.g. "what was that number
again?"). Kept in its own module rather than inlined in agent.py so memory
strategy (buffer vs. summary vs. windowed) can be swapped independently of
agent wiring.
"""

from langchain_classic.memory import ConversationBufferMemory


def get_conversation_memory() -> ConversationBufferMemory:
    """
    Returns a fresh ConversationBufferMemory instance.

    memory_key="chat_history" matches the placeholder name used in
    system_prompt.py's ReAct prompt template — the two must agree or the
    agent executor won't inject history into the prompt.

    return_messages=True: keeps history as structured message objects
    (HumanMessage/AIMessage) rather than one flattened string, which plays
    more reliably with chat-style models like Amazon Nova on Bedrock.
    """
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
