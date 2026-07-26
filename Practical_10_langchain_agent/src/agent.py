"""
agent.py
----------
Assembles the final agent: wires together the LLM, tools, prompt, and
memory into a runnable AgentExecutor with safety limits applied. This is
the one function the notebook actually calls — everything else in src/ is
a building block this file composes.
"""

import re

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from src.prompts.system_prompt import get_react_prompt
from utils.config import settings

# Matches a trailing "Observation" (with or without a colon) left at the end
# of the model's generation.
_TRAILING_OBSERVATION = re.compile(r"\n\s*Observation\s*:?\s*$")


class BedrockReActOutputParser(ReActSingleInputOutputParser):
    """
    ReAct parser that tolerates Bedrock echoing back the stop sequence.

    create_react_agent binds stop=["\\nObservation"] so the model halts right
    after writing its Action Input — AgentExecutor then runs the tool and
    supplies the real Observation itself. Bedrock honours that stop, but
    unlike some providers it *includes* the matched text in the response,
    so a generation arrives as:

        Action: calculator_tool
        Action Input: "15 * 3"
        Observation

    The stock parser's Action Input regex is greedy to end-of-string, so the
    tool receives `15 * 3"\\nObservation` and every call fails ("unterminated
    string literal"). Stripping that trailing token before parsing restores
    the loop. Harmless on providers that omit the stop text — the regex then
    simply matches nothing.
    """

    def parse(self, text: str) -> AgentAction | AgentFinish:
        return super().parse(_TRAILING_OBSERVATION.sub("", text))


def build_agent(
    llm: BaseLanguageModel,
    tools: list[BaseTool],
    memory,
    max_iterations: int | None = None,
    max_execution_time: int | None = None,
    verbose: bool | None = None,
) -> AgentExecutor:
    """
    Builds and returns an AgentExecutor.

    max_iterations / max_execution_time (Practical 10 safety requirement):
    these cap how many Thought/Action/Observation loops the agent can run
    and how long it can run in total, so a task that would otherwise cause
    an infinite loop (e.g. the model repeatedly calling the same tool
    without making progress) gets forcibly stopped instead of hanging or
    burning unbounded LLM calls/cost.

    early_stopping_method="force": when a limit is hit, immediately return
    an "Agent stopped due to iteration limit or time limit" response instead
    of asking the LLM for one more generation to summarize (which could
    itself loop or fail).
    """
    prompt = get_react_prompt()
    react_agent = create_react_agent(llm, tools, prompt, output_parser=BedrockReActOutputParser())

    return AgentExecutor(
        agent=react_agent,
        tools=tools,
        memory=memory,
        max_iterations=max_iterations if max_iterations is not None else settings.max_iterations,
        max_execution_time=(
            max_execution_time if max_execution_time is not None else settings.max_execution_time_seconds
        ),
        early_stopping_method="force",
        verbose=verbose if verbose is not None else settings.verbose,
        handle_parsing_errors=True,
    )
