"""
src/main.py
------------
Assembles and runs a 2-agent CrewAI crew:

    1. football_commentator -- a Peter-Drury-style commentator delivering
       dramatic commentary on Cristiano Ronaldo and Lionel Messi
    2. singer_biographer      -- a music journalist producing a structured,
       factual summary of Karan Aujla's career

Runs the crew sequentially, inspects whether either agent's built-in
delegation tools were actually used during the run, and saves a single
combined markdown file (research_output.md, project root) containing
both agents' work plus a short delegation-inspection note.

WHY THE @CrewBase / YAML PATTERN
    CrewAI's recommended project layout (the one `crewai create crew`
    scaffolds) keeps agent/task *definitions* -- role, goal, backstory,
    description, expected_output -- in YAML under config/, separate from
    the *wiring* code (which LLM, what order, how to run it) in Python.
    That keeps each agent's "personality" easy to tweak without touching
    Python, and keeps this file focused purely on orchestration. See
    src/config/agents.yaml and src/config/tasks.yaml for the definitions
    this file wires together.

WHY AMAZON NOVA (VIA BEDROCK) AND NOTHING ELSE
    Per this project's requirement, every LLM call goes through Amazon
    Bedrock's Nova family -- no Anthropic, OpenAI, or Google model calls
    anywhere. CrewAI's `LLM` class talks to Bedrock (via LiteLLM under the
    hood) using a "bedrock/<model-id>" model string, authenticated with
    whichever AWS credentials are present in the environment.
"""

import contextlib
import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_FILE = PROJECT_ROOT / "research_output.md"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# The two tool names CrewAI attaches to any agent with allow_delegation=True
# (see crewai/tools/agent_tools/delegate_work_tool.py and
# ask_question_tool.py) -- used by inspect_delegation() below to detect
# whether either agent actually chose to use them during a run.
_DELEGATE_TOOL_NAME = "Delegate work to coworker"
_ASK_QUESTION_TOOL_NAME = "Ask question to coworker"


def build_llm() -> LLM:
    """Construct the shared Amazon Nova LLM client used by both agents.

    WHY A CROSS-REGION INFERENCE PROFILE ID BY DEFAULT
        On-demand (pay-per-token, no provisioned throughput) invocation of
        Nova models on Bedrock often requires an *inference profile* ID
        rather than the bare model ID -- calling "amazon.nova-micro-v1:0"
        directly can fail with "...isn't supported. Retry your request
        with the ID or ARN of an inference profile...". The documented
        fix is to prefix the model ID with a region-group code, e.g.
        "us.amazon.nova-micro-v1:0" for US regions. NOVA_MODEL_ID
        defaults to that safer form; override it in .env if you're using
        a different Bedrock region group (eu./apac.) or have confirmed
        the bare ID works in your account.
    """
    model_id = os.environ.get("NOVA_MODEL_ID", "us.amazon.nova-micro-v1:0")
    region = os.environ.get("AWS_REGION", "us-east-1")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
    return LLM(model=f"bedrock/{model_id}", temperature=temperature, aws_region_name=region)


_llm = build_llm()


@CrewBase
class FootballAndMusicCrew:
    """The 2-agent crew this practical asks for.

    CrewBase automatically loads agents_config / tasks_config from the
    YAML paths below (resolved relative to this file) and, for any Task
    whose YAML entry has an `agent:` field matching an `@agent`-decorated
    method name, wires that agent onto the task automatically -- which is
    why the Task() calls below don't pass an explicit `agent=` argument.
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def football_commentator(self) -> Agent:
        """The Peter-Drury-style football commentator.

        allow_delegation=True (set in agents.yaml) equips this agent with
        CrewAI's built-in delegation tools -- it CAN hand off part of its
        task to singer_biographer if the LLM decides that's useful, though
        with two fully independent topics it may reasonably choose not
        to. See inspect_delegation() below for how we check what actually
        happened after a run.
        """
        return Agent(config=self.agents_config["football_commentator"], llm=_llm)

    @agent
    def singer_biographer(self) -> Agent:
        """The music journalist / Karan Aujla biographer."""
        return Agent(config=self.agents_config["singer_biographer"], llm=_llm)

    @task
    def football_commentary_task(self) -> Task:
        return Task(config=self.tasks_config["football_commentary_task"])

    @task
    def singer_biography_task(self) -> Task:
        return Task(config=self.tasks_config["singer_biography_task"])

    @crew
    def crew(self) -> Crew:
        """Assemble the full crew.

        Process.sequential: the football task runs first, then the singer
        task. Appropriate here since the two topics are independent and
        neither task's output needs to feed into the other's prompt
        (compare to CrewAI's `context:` field in tasks.yaml, which this
        project deliberately doesn't use, for the same reason).
        """
        return Crew(
            agents=[self.football_commentator(), self.singer_biographer()],
            tasks=[self.football_commentary_task(), self.singer_biography_task()],
            process=Process.sequential,
            verbose=True,
        )


class _Tee:
    """A minimal stdout "tee": writes to every stream it's given.

    Why this exists: we want CrewAI's verbose execution trace to keep
    printing live to the real console (so you can watch the crew work as
    it runs), AND capture that same text so inspect_delegation() can
    scan it afterward -- without needing any extra dependency, this
    small class does both by fanning writes out to multiple streams.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def run_crew_and_capture_log(crew_instance: Crew):
    """Run the crew while capturing its verbose stdout for later inspection.

    Returns:
        (crew_output, captured_log) -- crew_output is CrewAI's normal
        CrewOutput object; captured_log is the full verbose text printed
        during the run, used by inspect_delegation() below.
    """
    buffer = io.StringIO()
    tee = _Tee(sys.stdout, buffer)
    with contextlib.redirect_stdout(tee):
        crew_output = crew_instance.kickoff()
    return crew_output, buffer.getvalue()


def inspect_delegation(captured_log: str) -> Dict[str, bool]:
    """Scan a crew run's captured verbose log for delegation activity.

    CrewAI implements delegation as two built-in tools attached to any
    agent with allow_delegation=True: "Delegate work to coworker" and
    "Ask question to coworker" (see crewai/tools/agent_tools/). Their
    exact names appear verbatim in the verbose execution log whenever an
    agent actually decides to use them. Scanning for those literal
    strings is a simple, dependency-free way to answer "did delegation
    actually happen?" without relying on any internal, version-specific
    CrewAI API for tool-call history.

    Args:
        captured_log: the full verbose text from one crew run (see
            run_crew_and_capture_log above).

    Returns:
        {"delegate_work_used": bool, "ask_question_used": bool,
         "any_delegation_detected": bool}
    """
    delegate_used = _DELEGATE_TOOL_NAME in captured_log
    ask_used = _ASK_QUESTION_TOOL_NAME in captured_log
    return {
        "delegate_work_used": delegate_used,
        "ask_question_used": ask_used,
        "any_delegation_detected": delegate_used or ask_used,
    }


def describe_delegation(delegation_report: Dict[str, bool]) -> str:
    """Turn inspect_delegation()'s result into one human-readable sentence
    for the saved markdown file."""
    if delegation_report["any_delegation_detected"]:
        used = []
        if delegation_report["delegate_work_used"]:
            used.append('"Delegate work to coworker"')
        if delegation_report["ask_question_used"]:
            used.append('"Ask question to coworker"')
        return f"Delegation WAS used during this run (tool(s): {', '.join(used)})."
    return (
        "No delegation occurred during this run -- both agents completed "
        "their own task independently. allow_delegation=True on both "
        "agents means they COULD have handed off work to each other; the "
        "model simply judged it unnecessary for these two independent "
        "topics."
    )


def save_combined_markdown(
    task_outputs: List, delegation_report: Dict[str, bool], output_path: Path
) -> str:
    """Combine both agents' task outputs into one markdown deliverable.

    WHY WE COMBINE MANUALLY INSTEAD OF USING CrewAI's TASK-LEVEL
    output_file
        A task's own `output_file` (settable in tasks.yaml) writes ONLY
        that task's output to disk. Since this project's deliverable is
        ONE file containing BOTH agents' work (research_output.md), we
        build the combined document here from the crew's tasks_output
        instead, giving full control over the final structure -- a
        title, a delegation-inspection note, then both sections in
        order -- rather than just whatever a single task would write on
        its own.

    Returns:
        The markdown content that was written (also returned so
        tests can assert on it without re-reading the file from disk).
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = [task_output.raw.strip() for task_output in task_outputs]

    content = (
        "# Practical 13 -- Football Commentary & Karan Aujla Career Summary\n\n"
        f"*Generated {generated_at} by a 2-agent CrewAI crew running on "
        "Amazon Nova (Bedrock).*\n\n"
        f"**Delegation inspection:** {describe_delegation(delegation_report)}\n\n"
        + "\n\n---\n\n".join(sections)
        + "\n"
    )

    output_path.write_text(content, encoding="utf-8")
    return content


def _check_credentials() -> bool:
    """Print which required credentials are present, and return whether
    all of them are -- used to fail with a clear message before spending
    any tokens, rather than partway through a run."""
    has_aws = bool(
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or os.environ.get("AWS_ACCESS_KEY_ID")
    )
    print(f"{'OK  ' if has_aws else 'MISSING'} - AWS Bedrock credentials")
    if not has_aws:
        print("See README.md 'Setting up AWS Bedrock' before running main.py.")
    return has_aws


def main() -> None:
    if not _check_credentials():
        sys.exit(1)

    OUTPUTS_DIR.mkdir(exist_ok=True)

    crew_instance = FootballAndMusicCrew().crew()
    crew_output, captured_log = run_crew_and_capture_log(crew_instance)

    delegation_report = inspect_delegation(captured_log)
    print("\n=== Delegation inspection ===")
    for key, value in delegation_report.items():
        print(f"  {key}: {value}")

    save_combined_markdown(crew_output.tasks_output, delegation_report, OUTPUT_FILE)
    print(f"\nSaved combined markdown output -> {OUTPUT_FILE}")

    # Also keep a timestamped copy in outputs/, so repeated runs build up a
    # history instead of each one silently overwriting the last -- unlike
    # research_output.md at the project root, which always holds the
    # latest run only.
    timestamped_copy = OUTPUTS_DIR / f"research_output_{datetime.now():%Y%m%d_%H%M%S}.md"
    timestamped_copy.write_text(OUTPUT_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Saved timestamped copy -> {timestamped_copy}")


if __name__ == "__main__":
    main()
