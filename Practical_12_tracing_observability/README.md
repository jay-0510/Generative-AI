# Practical 12 — Tracing & Observability

A LangGraph agent (classifier → SQL agent / RAG agent → error fallback) built on
**Amazon Nova Micro / Nova Lite via AWS Bedrock**, instrumented with **both**
LangSmith and LangFuse so the two observability platforms can be compared
side by side.

> **Model constraint**: every LLM call in this project uses **Amazon Nova Micro**
> or **Amazon Nova Lite** on Bedrock. No Anthropic/Claude models are used anywhere
> in the agent logic (Bedrock Titan Embed is used only for embeddings, not
> generation, and is not a chat model).

---

## 1. What this project does

A user asks a customer-support-style question. A classifier decides whether it's
a **data lookup** ("how many orders has Vikram placed?") or a **policy/FAQ
question** ("what's your return window?"), then routes it to the matching
specialist:

```
                ┌──────────────────────┐
                │     classifier        │  Nova Micro — cheapest model,
                │  (routing decision)   │  single-label output
                └───────────┬───────────┘
             ┌───────────────┼───────────────┐
             │ sql            │ rag            │ error
             ▼                ▼                ▼
      ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
      │  sql_agent   │  │  rag_agent   │  │ error_node   │
      │ (Nova Lite)  │  │ (Nova Lite)  │  │ (no LLM call)│
      │ text→SQL,    │  │ Chroma       │  │ graceful      │
      │ execute,     │  │ retrieval +  │  │ fallback      │
      │ phrase answer│  │ generation   │  │ message       │
      └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
             └────────────────┬┴────────────────┘
                              ▼
                            END
```

See `graph_workflow.png` for the rendered diagram (generated from the actual
compiled graph — see the notebook, section 6).

This is the tracing/observability practical, not the agent-design practical —
so the agent itself is deliberately small (4 nodes, a 3-table SQLite DB, 4 FAQ
documents) in order to keep the focus on **instrumenting and reading traces**,
not on agent sophistication.

---

## 2. Folder structure

```
practical_12_tracing_observability/
├── config/
│   ├── __init__.py
│   ├── settings.py        # Central pydantic-settings config (models, keys, pricing table)
│   └── db_init.py         # Idempotent SQLite + Chroma vector store setup
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── classifier.py  # Routing node (Nova Micro)
│   │   ├── sql_agent.py   # Text-to-SQL node (Nova Lite)
│   │   ├── rag_agent.py   # Retrieval-augmented generation node (Nova Lite)
│   │   └── error_node.py  # Graceful fallback node
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py       # Shared AgentState TypedDict
│   │   └── workflow.py    # StateGraph construction/compilation
│   └── llm_invoke.py      # Shared Bedrock chat-model factory + usage helper
├── tests/
│   ├── conftest.py            # Fake Bedrock model + isolated temp data stores
│   ├── test_classify.py
│   ├── test_rag_agent.py
│   ├── test_setup_services.py
│   └── test_sql_agent.py
├── utils/
│   ├── __init__.py
│   └── helpers.py         # timing decorator, cost estimator, SQL safety guard
├── docker-compose.yml     # Self-hosted LangFuse v3 stack (optional — see §4)
├── graph_workflow.png     # Rendered diagram of the compiled graph
├── main.py                # Entry point: runs the graph with both tracers, analyzes runs
├── Practical_12_Tracing_Observability.ipynb
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 3. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in real values, see §4 and §5
```

**AWS credentials**: standard boto3 resolution — either run `aws configure`,
set `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in `.env`, or rely on an
IAM role if running on AWS infrastructure. You'll also need **model access
enabled** for Amazon Nova Micro, Nova Lite, and Titan Embed Text v2 in the
Bedrock console for your account/region (Bedrock → Model access).

---

## 4. Part 1 — LangSmith

1. Sign up free at <https://smith.langchain.com> and create an API key
   (Settings → API Keys).
2. In `.env`, set:
   ```
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=lsv2_pt_...
   LANGSMITH_PROJECT=practical-12-tracing-observability
   ```
3. Run `python main.py` (or the notebook). **No code changes to the graph
   are needed** — `config/settings.py`'s `export_tracing_env()` pushes these
   into `os.environ`, and LangChain/LangGraph's built-in tracer picks them
   up automatically for every node and every Bedrock call.
4. Open the LangSmith UI → your project → click into a trace. Each node
   (`classifier`, `sql_agent`, `rag_agent`) appears as a nested span with its
   own latency; each Bedrock `.invoke()` call appears as a child `llm` span.

**Finding the slowest step**: sort the trace's child runs by the "Latency"
column in the UI, or use `main.py`'s `analyze_langsmith_runs()`, which pulls
runs back via `langsmith.Client().list_runs()` and prints the slowest one
programmatically — see the notebook for the same walkthrough with output.

**Finding the most expensive LLM call**: LangSmith shows token counts per
`llm` run automatically; it may not know Bedrock Nova's per-token price
out of the box, so this project computes cost itself from
`config.settings.NOVA_PRICING_PER_1K_TOKENS` (see `utils/helpers.py`'s
`estimate_cost()` and the `[cost] ...` lines each node prints).

---

## 5. Part 2 — LangFuse

Pick **one**:

- **LangFuse Cloud (recommended for this practical)** — sign up free at
  <https://cloud.langfuse.com>, create a project, copy its public/secret
  keys into `.env`, leave `LANGFUSE_HOST` as the cloud URL. Zero infra.
- **Self-hosted** — `docker compose up -d` (see `docker-compose.yml`), open
  <http://localhost:3000>, create an account/project there instead, and set
  `LANGFUSE_HOST=http://localhost:3000` with those keys.

Instrumentation here is **not** purely environment-variable-driven — LangFuse
needs one of two mechanisms to see your code:

1. **`@observe()` decorator** (what this practical asks for) — every node
   function in `src/agents/*.py` is wrapped with `@observe(name="...")`,
   which creates a named span each time that node runs.
2. **LangChain `CallbackHandler`** — passed into `graph.invoke(state,
   config={"callbacks": [langfuse_handler]})` in `main.py`. This is what
   captures the actual Bedrock LLM call *inside* each node as a nested span
   with token usage — the decorator alone only sees the Python function
   boundary, not what happens inside the `.invoke()` call to Bedrock.

Both are used together in this project (see `main.py`'s
`run_with_langfuse_tracing()`), which is also the officially recommended
pattern when instrumenting a LangChain/LangGraph app with LangFuse.

Open the LangFuse UI → Traces to inspect latency and cost per span the same
way as in LangSmith.

---

## 6. Running it

```bash
python main.py          # runs both tracing passes + programmatic analysis
pytest tests/ -v         # fast unit tests — Bedrock calls are faked, no AWS cost
jupyter notebook Practical_12_Tracing_Observability.ipynb
```

The notebook walks through the same steps as `main.py` interactively, with
markdown cells explaining each piece, plus a template section at the end for
recording your own slowest-step / most-expensive-call findings and your
LangSmith-vs-LangFuse comparison.

---

## 7. LangSmith vs. LangFuse — comparison notes

Fill this in after you've actually run both against your own traces — the
table below is a starting scaffold of the two platforms' structural
differences, not a substitute for your own observations.

| | LangSmith | LangFuse |
|---|---|---|
| Hosting | Cloud-only (LangChain's own service) | Cloud **or** self-hosted (open source) |
| Setup for a LangChain/LangGraph app | Env vars only, zero code changes | Env vars **+** `@observe()` decorator and/or a `CallbackHandler` for full LLM-call detail |
| Cost visibility for Bedrock Nova specifically | Token counts shown automatically; $-cost may need a custom price map (this project supplies one) | Same — token counts automatic, $-cost benefits from a custom price map for non-built-in models |
| Best fit | Teams already all-in on LangChain tooling | Teams wanting open-source/self-hosted control, or using multiple LLM frameworks side by side |

**Your findings, after running the practical:**

- Slowest step observed: _fill in — which node, and its latency_
- Most expensive LLM call observed: _fill in — which node/model, and estimated cost_
- Which UI you preferred and why: _fill in — this is a judgment call the
  assignment wants from you directly, based on your own experience clicking
  through both dashboards._

---

## 8. Notes on design choices

- **Why Nova Micro for the classifier and Nova Lite for the two agents**:
  see the docstring in `config/settings.py` — it's a deliberate cost/latency
  trade-off you can point to directly in a trace comparison.
- **Why SQLite + Chroma instead of OpenSearch**: this practical is about
  tracing, not infrastructure — both stores need zero external services,
  keeping `docker compose up` needed only for the optional self-hosted
  LangFuse stack.
- **Why the SQL agent's generated SQL is validated before execution**: see
  `utils/helpers.py`'s `is_safe_select_query()` — a text-to-SQL node should
  never trust model output enough to execute it unchecked.
- Bedrock and LangFuse/LangSmith pricing and self-hosting requirements
  change fairly often — the version pins in `requirements.txt` and the
  pricing table in `config/settings.py` were accurate when this project was
  written; re-check the linked docs if something doesn't match what you see.
