# Practical 13 — CrewAI Multi-Agent

A 2-agent CrewAI crew:

1. **`football_commentator`** — a commentator in the style of Peter Drury,
   delivering dramatic, theatrical commentary on Cristiano Ronaldo and
   Lionel Messi.
2. **`singer_biographer`** — a music journalist producing a structured,
   factual career summary of Punjabi singer/rapper Karan Aujla.

Both agents run on **Amazon Nova** via AWS Bedrock — no Anthropic, OpenAI,
or Google model calls anywhere in this project. The crew runs end to end,
its execution log is inspected for delegation activity, and both agents'
output is saved as a single combined markdown file.

## Objectives covered

- [x] 2-agent crew: football commentator (Ronaldo/Messi) + singer biographer (Karan Aujla)
- [x] Run end-to-end via `python src/main.py`
- [x] Inspect delegation (`inspect_delegation()` scans the run's verbose log)
- [x] Output saved as markdown (`research_output.md`)
- [x] Model: Amazon Nova (Bedrock) only — no Anthropic/OpenAI/Google calls

## Folder structure

```
practical_13_crew_multi_agent/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── research_output.md          <- the required markdown deliverable (sample included)
├── outputs/                     <- timestamped history of every run (gitignored contents)
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── main.py                   <- builds + runs the crew, inspects delegation, saves output
    └── config/
        ├── agents.yaml            <- agent definitions (role/goal/backstory)
        └── tasks.yaml              <- task definitions (description/expected_output)
```

> **Note on scope:** this matches the exact structure requested, with one
> small addition — `outputs/.gitkeep` — needed only so git tracks the
> otherwise-empty `outputs/` folder (git doesn't track empty directories).
> No separate `crew.py` is used; the `@CrewBase` crew class lives directly
> in `src/main.py`, since that's the file shown in the given layout.

## Prerequisites

- **Python 3.10+**
- An **AWS account** with Bedrock access.

### A note on cost

**AWS Bedrock has no permanent free tier** — it's pay-per-token from the
first call, and AWS requires a payment method on the account. Amazon Nova
Micro costs about **$0.035 per million input tokens**, so running this
crew (two short generation tasks) costs a small fraction of a cent. New
AWS accounts also typically receive introductory credits that comfortably
cover this.

## Setting up AWS Bedrock (Amazon Nova)

1. Sign in to (or create) an AWS account at <https://console.aws.amazon.com>.
2. Open the **Amazon Bedrock** console and select a region that supports
   Nova (e.g. **us-east-1**, N. Virginia).
3. **Model access**: AWS automatically enables serverless models —
   including the Nova family — for every account, so you generally don't
   need to manually request access. If a call still comes back
   access-denied, check Bedrock → **Model access** in the console.
4. Get credentials — two options:
   - **Easiest: a Bedrock API key.** Bedrock console → **API keys** →
     **Generate long-term API key**. Copy it into `.env` as
     `AWS_BEARER_TOKEN_BEDROCK`.
   - **Alternative: a standard IAM access key.** IAM → Users → create a
     user → attach a policy allowing `bedrock:InvokeModel` /
     `bedrock:Converse` (e.g. `AmazonBedrockLimitedAccess`) → **Security
     credentials** → **Create access key**. Copy both values into `.env`.
5. Set `AWS_REGION` in `.env` to match the region you enabled Bedrock in.

### A Nova-specific gotcha worth knowing about

On-demand (pay-per-token) invocation of Nova models can fail with:

```
Invocation of model ID amazon.nova-micro-v1:0 with on-demand throughput
isn't supported. Retry your request with the ID or ARN of an inference
profile that contains this model.
```

The fix is to use the **cross-region inference profile ID** instead of
the bare model ID — prefix it with a region-group code, e.g.
`us.amazon.nova-micro-v1:0` for US regions. `src/main.py`'s `build_llm()`
already defaults `NOVA_MODEL_ID` to that safer form, so you shouldn't hit
this — but if you override `NOVA_MODEL_ID` yourself and see this error,
this is why.

## Installation

```bash
# 1. Unzip / clone the project, then cd into it
cd practical_13_crew_multi_agent

# 2. Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your credentials
cp .env.example .env
# then open .env and fill in your real AWS values

# 5. Run the crew end to end
python src/main.py
```

### Dependencies installed (`requirements.txt`)

| Package         | Why it's needed                                                                                                                                     |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `crewai`        | `Agent`, `Task`, `Crew`, `Process`, `LLM`, and the `@CrewBase`/`@agent`/`@task`/`@crew` decorators used to wire `config/*.yaml` into a running crew |
| `boto3`         | The AWS SDK — what CrewAI's `LLM` class (via LiteLLM) actually uses to authenticate and call Bedrock's Converse API                                 |
| `python-dotenv` | Loads credentials from `.env`                                                                                                                       |

`crewai` already pulls in `pyyaml` (for `config/*.yaml`) as a transitive
dependency, so it isn't listed separately.

## How it works

### `src/config/agents.yaml` and `src/config/tasks.yaml`

CrewAI's recommended pattern keeps each agent's _personality_ (role,
goal, backstory) and each task's _brief_ (description, expected_output)
in YAML, separate from the Python code that wires them together. Both
agents have `allow_delegation: true` — see "Inspecting delegation" below
for what that actually does.

### `src/main.py`

- `build_llm()` — constructs one shared `LLM` pointed at Amazon Nova via
  Bedrock, reused by both agents.
- `FootballAndMusicCrew` — a `@CrewBase`-decorated class. Its `@agent` and
  `@task` methods load their config from the YAML files above; the
  `@crew` method assembles them into a `Process.sequential` crew (the
  football task runs, then the singer task — the two are independent, so
  no `context:` linking is used between them).
- `run_crew_and_capture_log()` — runs the crew while "teeing" its verbose
  output to both your terminal (so you see it live) and an in-memory
  buffer (so the next step can inspect it).
- `inspect_delegation()` — scans that captured log for CrewAI's two
  built-in delegation tool names (`"Delegate work to coworker"`,
  `"Ask question to coworker"`) to determine whether either agent actually
  used them.
- `save_combined_markdown()` — builds `research_output.md` from both
  agents' task outputs, with a short delegation-inspection note at the
  top.

### Inspecting delegation

Setting `allow_delegation: true` in `agents.yaml` **allows** an agent to
hand off work to its crewmate — it does not force it to. CrewAI attaches
two tools to any agent with delegation enabled, and the LLM decides at
run time whether using them is worthwhile.

Because this crew's two tasks are on **completely independent topics**
(football commentary vs. a singer's biography), there's rarely a good
reason for either agent to delegate — realistically, most runs will show
**no delegation**, and `research_output.md`'s delegation-inspection note
will say so. That's expected, correct behavior, not a bug: the crew is
correctly _configured_ to allow delegation; whether it's _used_ is the
model's call, run to run. If you want to actually observe delegation
firing, try modifying one task's description in `tasks.yaml` to require
information only the other agent would plausibly have.

## Running the crew

```bash
python src/main.py
```

This will:

1. Check your AWS credentials are present (fails fast with a clear
   message if not).
2. Build and run the crew (`Process.sequential` — football task, then
   singer task), printing CrewAI's live verbose trace to your terminal.
3. Print a delegation-inspection summary.
4. Write the combined result to `research_output.md` (project root).
5. Save a timestamped copy under `outputs/`.

## Sample output

`research_output.md` in this repo already contains a representative
example of what a run produces — both sections in full, in the exact
format `save_combined_markdown()` generates — so you have a complete
deliverable even before running the crew yourself with real credentials.
Running `python src/main.py` regenerates it with a fresh, live
AI-generated version.

## Troubleshooting

- **`AccessDeniedException` calling Nova** — check `AWS_REGION` in `.env`
  matches where you set up Bedrock, and that your IAM policy/API key
  grants `bedrock:InvokeModel`/`bedrock:Converse`.
- **"...isn't supported. Retry your request with the ID or ARN of an
  inference profile..."** — see "A Nova-specific gotcha worth knowing
  about" above; use the `us.amazon.nova-micro-v1:0`-style inference
  profile ID rather than the bare model ID.
- **`ModuleNotFoundError`** — the install step wasn't run, or ran into a
  different Python environment than the one running `python src/main.py`.
  Confirm you're using the same interpreter/venv for both.
- **Nothing in `research_output.md` changed after a re-run** — make sure
  you're looking at the project-root file, not a stale copy under
  `outputs/`; each run overwrites `research_output.md` but always adds a
  _new_, separate timestamped file under `outputs/`.
