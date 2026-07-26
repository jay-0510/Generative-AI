"""
system_prompt.py
------------------
Builds the ReAct-style prompt template the agent reasons with.

Why a template (not an f-string built inline in agent.py): create_react_agent
needs specific placeholder variables — {tools}, {tool_names}, {input},
{agent_scratchpad} — filled in automatically by the agent executor at run
time. Keeping the template text separate from agent.py means the persona /
instructions can be edited without touching wiring code.
"""

from langchain_core.prompts import PromptTemplate

# {chat_history} is included so ConversationBufferMemory's messages get
# injected into the prompt — this is what gives the agent the ability to
# answer follow-up questions. Its key must match `memory_key` in
# conversation_memory.py exactly.
_REACT_TEMPLATE = """You are a careful, helpful assistant with access to the following tools:

{tools}

Use this exact format:

Question: the input question you must answer
Thought: reason about what to do next
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Rules:
- Only call a tool when it is actually needed to answer the question.
- If a tool result already answers the question, do not call another tool
  with the same input again — go straight to Final Answer.
- If you are unsure a tool call will help, say so in Final Answer instead
  of guessing repeatedly.

Previous conversation history:
{chat_history}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


def get_react_prompt() -> PromptTemplate:
    """Returns the PromptTemplate used to construct the ReAct agent."""
    return PromptTemplate.from_template(_REACT_TEMPLATE)
