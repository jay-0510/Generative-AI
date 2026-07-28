# Practical 11 — LangGraph Conditional Workflow (Amazon Nova / Bedrock)

A LangGraph workflow that routes each incoming query through a
**classifier node**, down either a **SQL agent path** or a **RAG agent
path**, with a graceful **error node** for anything neither path can
answer — visualized with `.get_graph().draw_mermaid()`. Runs end-to-end on
**Amazon Nova** (chat) and **Amazon Titan** (embeddings) via AWS Bedrock —
no Anthropic, OpenAI, or Google model calls anywhere in this project.

## Objectives covered

- [x] Classifier node → SQL agent path OR RAG agent path, based on query type
- [x] Error node for graceful failures
- [x] Graph visualized with `.get_graph().draw_mermaid()` — plus a real,
      locally-rendered `graph_workflow.png` already included in this repo
- [x] Model: Amazon Nova (chat) + Amazon Titan (embeddings) only
- [x] `tests/` — 45 passing pytest tests, runnable fully offline

## Design note: RAG-first, SQL-fallback

RAG is treated as the **default/primary** attempt for anything that isn't
unambiguously a SQL question. If the RAG agent can't find sufficiently
relevant context, the graph **automatically falls back to the SQL agent**
before giving up. Only if _both_ attempts fail does the workflow reach the
error node.

```
                              +-------------------+
                    START --> |  classify_query   |
                              +---------+---------+
                        "sql"           |            "rag" / "unknown"
                    +-------------------+-------------------+
                    v                                        v
             +-------------+                          +--------------+
             |  sql_agent  | <---------- fail -------- |   rag_agent  |
             +------+------+     (RAG didn't work      +------+-------+
                    |             out -> try SQL)              |
              success|                                    success|
                    v                                          v
                   END                                        END
                    |
                  fail
                    v
             +--------------+
             |  error_node  | --> END
             +--------------+
```

## Folder structure

```
practical_11_langgraph_conditional_workflow/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── docker-compose.yml
├── graph_workflow.png              <- the required diagram screenshot (already rendered)
├── main.py                         <- CLI entry point (run the workflow without Jupyter)
├── notebooks/
│   └── Practical_11_LangGraph_Conditional_Workflow.ipynb   <- run this
├── config/
│   ├── __init__.py
│   ├── db_init.py                  <- sample SQLite DB: schema + seed data
│   └── settings.py                  <- centralized config: model IDs, paths, RAG tuning
├── src/
│   ├── __init__.py
│   ├── llm_invoke.py                <- shared Amazon Nova + Titan access layer
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── classifier.py             <- classify_query_node
│   │   ├── rag_agent.py               <- rag_agent_node (retrieval + confidence check)
│   │   ├── sql_agent.py                <- sql_agent_node (generate + safely run SQL)
│   │   └── error_node.py                <- error_node
│   └── graph/
│       ├── __init__.py
│       ├── state.py                   <- GraphState TypedDict
│       └── workflow.py                 <- routing functions + build_graph()
├── utils/
│   ├── __init__.py
│   └── helpers.py                     <- the SELECT-only SQL safety guard
├── data/
│   └── sample_docs/
│       └── company_policies.txt   <- RAG-path knowledge base
│   (company_sample.db is generated automatically on first run — gitignored)
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_classify.py
    ├── test_rag_agent.py
    ├── test_sql_agent.py
    └── test_setup_services.py
```

> **Note on scope:** the `notebooks/` folder isn't part of the folder
> layout you shared, but the brief also explicitly asked for a Jupyter
> notebook — so it's included here as an addition, alongside (not instead
> of) everything else matching your structure exactly.

## Prerequisites

- **Python 3.10+**
- An **AWS account** with Bedrock access (see setup below).

### A note on cost

**AWS Bedrock has no permanent free tier** — it's pay-per-token from the
first call, and AWS requires a payment method on the account. The good
news: Amazon Nova Micro costs about **$0.035 per million input tokens**,
so running this entire notebook costs a small fraction of a cent. New AWS
accounts also typically receive introductory credits that comfortably
cover this.

## Setting up AWS Bedrock (Amazon Nova + Titan)

1. Sign in to (or create) an AWS account at <https://console.aws.amazon.com>.
2. Open the **Amazon Bedrock** console and select a region that supports
   Nova (e.g. **us-east-1**, N. Virginia).
3. **Model access**: as of late 2025, AWS automatically enables serverless
   models — including the Nova family — for every account, so you
   generally don't need to manually request access. If a call still comes
   back with an access-denied error, check Bedrock → **Model access** in
   the console and confirm Nova Micro shows as enabled.
4. Get credentials — two options:
   - **Easiest: a Bedrock API key.** In the Bedrock console, look for
     **API keys** in the left sidebar → **Generate long-term API key**.
     Copy it into `.env` as `AWS_BEARER_TOKEN_BEDROCK`.
   - **Alternative: a standard IAM access key.** IAM → Users → create a
     user → attach a policy allowing `bedrock:InvokeModel` /
     `bedrock:Converse` (e.g. `AmazonBedrockLimitedAccess`) → **Security
     credentials** → **Create access key**. Copy both values into `.env`.
5. Set `AWS_REGION` in `.env` to match the region you enabled Bedrock in.

If `AWS_BEARER_TOKEN_BEDROCK` doesn't authenticate in your environment,
upgrade boto3/botocore (`pip install -U boto3 botocore`) or fall back to
the IAM access key method — both are read automatically by
`langchain-aws`/boto3's standard AWS credential chain, so no code changes
are needed either way.

## Installation

```bash
# 1. Unzip / clone the project, then cd into it
cd practical_11_langgraph_conditional_workflow

# 2. Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your credentials
cp .env.example .env
# then open .env and fill in your real AWS values

# 5. Run the test suite (fully offline — no credentials actually needed)
pytest -v

# 6. Launch Jupyter, or run the CLI version instead
jupyter notebook notebooks/Practical_11_LangGraph_Conditional_Workflow.ipynb
# --- or ---
python main.py
```

### Running with Docker instead

```bash
docker compose up notebook   # Jupyter Lab at http://localhost:8888
# --- or ---
docker compose run --rm app  # runs main.py once and exits
```

Both services install `requirements.txt` inside the container on
startup and read credentials from your local `.env` via `env_file:` — see
`docker-compose.yml` for details.

### Dependencies installed (`requirements.txt`)

| Package                       | Why it's needed                                                              |
| ----------------------------- | ---------------------------------------------------------------------------- |
| `langchain`, `langchain-core` | Message primitives shared across agents                                      |
| `langchain-community`         | `FAISS` vector store, `TextLoader`                                           |
| `langchain-text-splitters`    | `RecursiveCharacterTextSplitter` for chunking the policy doc                 |
| `langchain-aws`               | `ChatBedrockConverse` (Nova) and `BedrockEmbeddings` (Titan)                 |
| `boto3`                       | AWS SDK — the actual client `langchain-aws` calls into                       |
| `langgraph`                   | `StateGraph`, conditional edges, `.get_graph().draw_mermaid()`               |
| `faiss-cpu`                   | Local, in-memory vector index for the RAG path                               |
| `python-dotenv`               | Loads credentials from `.env`                                                |
| `jupyter`, `ipykernel`        | Run the notebook itself                                                      |
| `pytest`                      | Runs `tests/`                                                                |
| `pygraphviz` (optional)       | Local rendering of `graph_workflow.png` — see the note in `requirements.txt` |

`sqlite3` (the SQL path's database driver) is part of the Python standard
library, so it needs no separate install.

## How each piece works

| Module                     | Role                                                                                                                                                                                     |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config/settings.py`       | Every model ID, file path, and tuning constant, read from env vars in exactly one place                                                                                                  |
| `config/db_init.py`        | Creates + seeds the sample SQLite database (idempotent)                                                                                                                                  |
| `utils/helpers.py`         | `execute_readonly_query` — rejects anything that isn't a single safe `SELECT`, since the SQL string comes from an LLM (untrusted input)                                                  |
| `src/llm_invoke.py`        | Constructs the shared `ChatBedrockConverse` (Nova) and `BedrockEmbeddings` (Titan) clients once; every agent calls `invoke_chat(messages)` instead of importing `langchain_aws` directly |
| `src/agents/classifier.py` | Labels the query `"sql"`, `"rag"`, or `"unknown"` via a temperature=0 Nova Micro call                                                                                                    |
| `src/agents/rag_agent.py`  | Retrieves from FAISS and checks the best match's similarity **distance against a threshold** (`config/settings.py`) before trusting it                                                   |
| `src/agents/sql_agent.py`  | Generates SQL, validates + executes it through `utils/helpers.py`, then summarizes the results                                                                                           |
| `src/agents/error_node.py` | One graceful, user-facing message — the graph's explicit failure path                                                                                                                    |
| `src/graph/state.py`       | `GraphState` — the shared schema passed between every node                                                                                                                               |
| `src/graph/workflow.py`    | The three routing functions + `build_graph()`                                                                                                                                            |

Both agent nodes report success/failure through **state flags**
(`rag_success`, `sql_success`) instead of raising exceptions — that's what
lets the routing functions in `src/graph/workflow.py`, not a `try/except`
block, decide what happens next.

## Visualizing and committing the graph diagram

`graph_workflow.png` (project root) is already included in this repo,
rendered with `.get_graph().draw_png()` — a **locally-rendered**
alternative to the network-dependent `draw_mermaid_png()` that uses
`pygraphviz` + the system `graphviz` library instead of calling out to the
Mermaid.Ink API. `.get_graph().draw_mermaid()` (the literal method this
practical asks for) still prints the raw Mermaid source in the notebook —
pure local computation either way.

The notebook's visualization cell tries three things in order:
`draw_png()` (local) → `draw_mermaid_png()` (needs internet) → printing
instructions for a manual screenshot via mermaid.live — so you'll always
end up with a usable image regardless of your environment.

## Test queries used in the notebook / main.py

| Query                                                                                       | Exercises                                                                  |
| ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| _"How many employees work in the Engineering department, and what's their average salary?"_ | Direct SQL path                                                            |
| _"What is the company's policy on remote work?"_                                            | Direct RAG path                                                            |
| _"Tell me about Ethan Wright."_                                                             | RAG fails (no biographical content in the policy docs) → falls back to SQL |
| _"What's the weather like in Tokyo today?"_                                                 | Both paths fail → graceful `error_node` message                            |

## Testing

```bash
pytest -v
```

All 45 tests run **fully offline** — `tests/conftest.py` supplies fake AWS
credentials so imports succeed, and every test that needs a model reply
monkeypatches `invoke_chat` at the point of use (in the agent module that
imported it) rather than actually calling Bedrock.

| File                     | What it tests                                                                                                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_classify.py`       | Label parsing (SQL/RAG/UNKNOWN, case/whitespace handling) + `route_after_classification`                                                                             |
| `test_rag_agent.py`      | The relevance-distance-threshold logic, the "never call the LLM on irrelevant context" rule, + `route_after_rag`                                                     |
| `test_sql_agent.py`      | The SQL safety guard (rejects everything but single `SELECT`s), `sql_agent_node`'s success/failure paths, + `route_after_sql`                                        |
| `test_setup_services.py` | `config/settings.py`'s values (Amazon-only model IDs, sane types/paths), `config/db_init.py`'s idempotent DB creation, and `src/llm_invoke.py`'s client construction |

## Troubleshooting

- **`AccessDeniedException` calling Nova** — check `AWS_REGION` in `.env`
  matches where you set up Bedrock, and that your IAM policy/API key grants
  `bedrock:InvokeModel`/`bedrock:Converse`.
- **`ModuleNotFoundError`** — the install cell wasn't run, or ran into a
  different Python environment than Jupyter is using. Run
  `import sys; print(sys.executable)` in a cell, then
  `!{sys.executable} -m pip install -r ../requirements.txt`, then restart
  the kernel.
- **`draw_png()` raises an ImportError** — `pygraphviz` (and the system
  `graphviz` library) aren't installed; see the note in `requirements.txt`
  for platform-specific install commands, or just rely on the notebook's
  automatic fallback to `draw_mermaid_png()` / manual screenshot.
- **`pytest` fails on a fresh clone** — run it from the project root (not
  from inside `tests/`) so `conftest.py` is picked up.
