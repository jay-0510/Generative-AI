# Practical 10: LangChain Agent with Tools

A LangChain ReAct agent with 3 tools (calculator, mock web search, RAG
retriever), conversation memory for follow-up questions, and a
`max_iterations` safety limit demonstrated against a task that would
otherwise loop forever.

---

## Table of Contents

1. [A Note on LangChain Version](#a-note-on-langchain-version)
2. [Folder Structure](#folder-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the Notebook](#running-the-notebook)
7. [What Each Piece Does](#what-each-piece-does)
8. [Memory: How Follow-Ups Work](#memory-how-follow-ups-work)
9. [Safety: The max_iterations Test](#safety-the-max_iterations-test)
10. [Switching to Real Bedrock](#switching-to-real-bedrock)
11. [Known Limitations](#known-limitations)

---

## A Note on LangChain Version

As of mid-2026, LangChain's agent API has been restructured around
`create_agent` (a LangGraph-based graph, replacing `AgentExecutor`). That
new API doesn't expose a literal `max_iterations` parameter — the
equivalent is a `recursion_limit` passed at invoke time. Since this
practical specifically asks for `max_iterations` and `AgentExecutor`, this
project deliberately uses `langchain-classic` (the officially maintained
package that preserves the pre-1.0 `AgentExecutor` / `create_react_agent`
API) instead of forcing a mismatch with the brief's wording. If your course
material has since moved to the new `create_agent` API, the safety concept
is identical — just renamed — and Section 6 of the notebook / this README's
[Known Limitations](#known-limitations) says exactly what would change.

---

## Folder Structure

```
practical_10_langchain_agent/
│
├── notebook/
│   └── practical_10_notebook.ipynb   # end-to-end demo: tools, memory, safety test
│
├── src/
│   ├── __init__.py
│   ├── memory/
│   │   ├── __init__.py
│   │   └── conversation_memory.py    # ConversationBufferMemory wrapper
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── system_prompt.py          # ReAct prompt template
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── calculator_tool.py        # safe AST-based calculator (no eval())
│   │   ├── rag_tool.py               # wraps a retriever as an agent tool
│   │   └── web_search_tool.py        # mock web search (per the brief)
│   ├── agent.py                      # assembles the AgentExecutor + safety limits
│   └── bedrock_llm.py                # ChatBedrock wrapper
│
├── utils/
│   └── config.py                     # environment-driven settings
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Requirement                        | Purpose                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------ |
| Python 3.10+                       | Runtime                                                                                    |
| AWS Account + Bedrock model access | Only needed for **real** LLM mode — the notebook runs fully offline by default (see below) |

**You do not need AWS credentials to run this notebook end-to-end.** Every
cell defaults to a scripted `FakeListLLM` and a local FAISS demo index, so
tool routing, memory, and the safety limit are all provable without any
cloud dependency. AWS is only required if you flip `USE_FAKE_LLM = False`
to run against real Claude on Bedrock.

---

## Installation

```bash
git clone <repository-url>
cd practical_10_langchain_agent

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Configuration

```bash
cp .env.example .env
```

Only needed for real Bedrock mode — fill in your AWS region and model IDs.
The default offline demo mode ignores these.

---

## Running the Notebook

```bash
jupyter notebook notebook/practical_10_notebook.ipynb
```

Run all cells top to bottom. Every cell is designed to execute without
AWS credentials by default (`USE_FAKE_LLM = True` in cell 2).

---

## What Each Piece Does

| Tool          | File                           | Notes                                                                                                                                                                                                                                                                        |
| ------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Calculator    | `src/tools/calculator_tool.py` | Uses Python's `ast` module to parse and evaluate arithmetic — **deliberately not `eval()`**, since passing an LLM-generated string straight to `eval()` is an arbitrary-code-execution risk. A malicious or malformed action input just fails to parse instead of executing. |
| Web search    | `src/tools/web_search_tool.py` | A **mock**, per the practical's own spec — returns canned, keyword-matched results, no live API call. Swappable for a real search API later without changing the tool's signature.                                                                                           |
| RAG retriever | `src/tools/rag_tool.py`        | Wraps any LangChain retriever as a tool. Ships with a small demo FAISS index (`FakeEmbeddings`, no AWS needed) so it's runnable out of the box — **swap in your actual Milestone 2 retriever for a real submission**; see the docstring in that file for exactly where.      |

`src/agent.py` ties these together with `create_react_agent` +
`AgentExecutor` from `langchain_classic`, applying `max_iterations` /
`max_execution_time` as safety limits.

---

## Memory: How Follow-Ups Work

`src/memory/conversation_memory.py` wraps `ConversationBufferMemory` with
`memory_key="chat_history"` — this key must match the `{chat_history}`
placeholder in `src/prompts/system_prompt.py`'s ReAct template, or the
executor won't inject history into the prompt at all.

The notebook proves this isn't just decorative: after turn 1, it renders
the prompt template directly with what's currently in memory and prints
the `chat_history` section, showing the prior Q&A actually landed in the
text the LLM sees for the follow-up question — not just that the second
answer happened to sound right.

---

## Safety: The max_iterations Test

The notebook scripts a fake LLM that **never emits `Final Answer`** — it
keeps reissuing the same tool call indefinitely, simulating a model stuck
in a reasoning loop. With `max_iterations=3`, `AgentExecutor` (using
`early_stopping_method="force"`) stops after exactly 3 tool calls and
returns `"Agent stopped due to iteration limit or time limit."` instead of
hanging or erroring.

`max_execution_time` is a second, independent backstop — it cuts the run
off on wall-clock time even if a _single_ tool call hangs (e.g. a slow
network call), which an iteration count alone wouldn't catch.

---

## Switching to Real Bedrock

1. `cp .env.example .env` and fill in your AWS region (and run
   `aws configure`, or set credentials as env vars).
2. In the notebook, set `USE_FAKE_LLM = False` and re-run from the top.
3. For the safety test in real mode, you can't script a guaranteed loop —
   instead try an intentionally adversarial prompt (an example is already
   in that cell) and observe whether/how the real model gets stopped by
   the same `max_iterations` limit.
4. Optionally replace `make_rag_tool()`'s demo FAISS index with your real
   Milestone 2 retriever: `make_rag_tool(retriever=your_retriever)`.

---
