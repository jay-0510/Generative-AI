"""
rag_tool.py
-------------
Wraps a retriever as a LangChain tool so the agent can call it like any
other tool. This module ships with a small in-memory FAISS demo index (a
handful of sample documents + FakeEmbeddings) purely so the tool is
runnable and testable without AWS credentials.

IMPORTANT: In your actual submission, replace `_build_demo_vectorstore()`
below with your real Milestone 2 RAG retriever (the one built over your
actual document set with real embeddings) — copy that module in rather
than rebuilding it here. This file is intentionally a thin, swappable
wrapper: `make_rag_tool(retriever)` accepts any LangChain BaseRetriever.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever

# Sample docs so the demo index has something to retrieve. Swap for your
# real Milestone 2 document set / retriever in production.
_DEMO_DOCS = [
    "The agent's calculator tool evaluates arithmetic expressions safely using "
    "an AST-based parser instead of Python's eval().",
    "The web search tool in this project is a mock — it returns canned results "
    "and does not call a live search API.",
    "AgentExecutor's max_iterations parameter stops the agent loop after a fixed "
    "number of tool calls, guarding against infinite loops.",
]


def _build_demo_vectorstore(embeddings=None) -> FAISS:
    """
    Builds a small FAISS index over `_DEMO_DOCS`.

    embeddings defaults to FakeEmbeddings (deterministic, no API calls) so
    this module works offline for testing. Pass a real embeddings object
    (e.g. BedrockEmbeddings) to build a production-quality index instead.
    """
    if embeddings is None:
        # Imported lazily so FakeEmbeddings (a testing utility) is only
        # pulled in when no real embeddings model is supplied.
        from langchain_community.embeddings import FakeEmbeddings

        embeddings = FakeEmbeddings(size=256)
    return FAISS.from_texts(_DEMO_DOCS, embeddings)


def make_rag_tool(retriever: VectorStoreRetriever | None = None):
    """
    Builds the @tool-decorated RAG retriever function.

    Accepts an optional pre-built retriever (e.g. your real Milestone 2
    retriever); if none is given, falls back to the small demo FAISS index
    above so the tool is usable out of the box.
    """
    active_retriever = retriever or _build_demo_vectorstore().as_retriever(search_kwargs={"k": 2})

    @tool
    def rag_retriever_tool(query: str) -> str:
        """
        Retrieves relevant passages from the indexed knowledge base for the
        given query. Use this when the question is about this project's own
        documented details (e.g. tool behavior, safety limits) rather than
        general world knowledge.
        """
        docs = active_retriever.invoke(query)
        if not docs:
            return "No relevant documents found in the knowledge base."
        return "\n---\n".join(doc.page_content for doc in docs)

    return rag_retriever_tool
