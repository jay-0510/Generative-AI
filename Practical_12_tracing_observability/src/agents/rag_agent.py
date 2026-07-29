"""
src/agents/rag_agent.py
=========================

WHY THIS FILE EXISTS
---------------------
This node handles the "RAG" branch: policy/FAQ questions that aren't in the
structured database at all (return windows, warranty terms, etc.) and
instead live as short text documents embedded in the Chroma vector store
(config/db_init.py). Retrieval-augmented generation is used here — rather
than just asking the model to answer from memory — so the answer is
grounded in the actual (small, controlled) policy text instead of the
model's possibly-wrong prior knowledge about "a typical return policy."

WHY NOVA LITE: same reasoning as the SQL agent — this node has to read
retrieved context and produce a grounded, coherent answer, which is a step
up from the classifier's single-label task.

A slightly higher temperature (0.3, vs. 0.0 for classifier/SQL) is used
here on purpose: the SQL agent's output must be syntactically exact, but a
policy-FAQ answer benefits from a little more natural phrasing variation
without risking factual drift, since it's still grounded in retrieved text.
"""

from __future__ import annotations

from langfuse import observe

from config.db_init import init_vector_store
from config.settings import get_settings
from src.graph.state import AgentState
from src.llm_invoke import get_chat_model, invoke_with_usage
from utils.helpers import estimate_cost, timed

_RAG_ANSWER_PROMPT = """Answer the user's question using ONLY the context below. If the
context doesn't contain the answer, say you don't have that information.

Context:
{context}

Question: {question}

Answer:"""

_TOP_K_DOCS = 2


@observe(name="rag_agent_node")
@timed("rag_agent")
def rag_agent_node(state: AgentState) -> dict:
    """
    LangGraph node: retrieve relevant FAQ passages for `state["query"]` and
    generate a grounded answer. Returns `retrieved_docs` and `answer`.
    """
    settings = get_settings()
    model = get_chat_model(settings.rag_agent_model_id, temperature=0.3)

    vector_store = init_vector_store()  # idempotent: loads existing index if present.
    retriever = vector_store.as_retriever(search_kwargs={"k": _TOP_K_DOCS})
    retrieved_documents = retriever.invoke(state["query"])
    retrieved_texts = [doc.page_content for doc in retrieved_documents]

    prompt = _RAG_ANSWER_PROMPT.format(context="\n\n".join(retrieved_texts), question=state["query"])
    response, usage = invoke_with_usage(model, [("human", prompt)])
    cost = estimate_cost(settings.rag_agent_model_id, usage["input_tokens"], usage["output_tokens"])
    print(f"[cost] rag_agent ({settings.rag_agent_model_id}): ${cost:.6f} ({usage['total_tokens']} tokens)")

    return {
        "retrieved_docs": retrieved_texts,
        "answer": response.content.strip(),
    }
