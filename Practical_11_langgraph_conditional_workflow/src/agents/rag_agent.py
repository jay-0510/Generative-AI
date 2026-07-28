"""
src/agents/rag_agent.py
-------------------------
The RAG agent node -- tried FIRST whenever the classifier says "rag" or
"unknown" (see route_after_classification in src/graph/workflow.py).

WHY THIS AGENT CAN HONESTLY REPORT FAILURE
    A vector search always returns *something* -- even for a query that
    has nothing to do with the indexed documents, FAISS will still hand
    back whichever chunks are "least dissimilar." This project's RAG ->
    SQL fallback (see route_after_rag in src/graph/workflow.py) depends on
    this agent being able to say "I don't have relevant context for this"
    rather than fabricating an answer from irrelevant chunks. It does that
    by checking the raw similarity distance of the best match against
    `RAG_RELEVANCE_DISTANCE_THRESHOLD` (config/settings.py), not just
    checking "did we get any results back."

WHY AMAZON TITAN EMBEDDINGS
    Per this project's requirement to stay on Amazon models end-to-end,
    the knowledge base is embedded with Titan (via
    `src.llm_invoke.get_embedding_model()`) rather than an OpenAI/Google
    embedding model.
"""

from typing import List, Tuple

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    POLICY_DOC_PATH,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_RELEVANCE_DISTANCE_THRESHOLD,
    RAG_TOP_K,
)
from src.graph.state import GraphState
from src.llm_invoke import get_embedding_model, invoke_chat

_vectorstore = None  # module-level cache, built once per process


def _build_vectorstore() -> FAISS:
    """Load, chunk, and embed the sample policy document into FAISS.

    Why RecursiveCharacterTextSplitter with a modest chunk size: the
    source document is organized as short policy sections, so relatively
    small chunks (with overlap) keep each chunk focused on one policy
    topic, which improves both retrieval precision and the
    relevance-threshold check below.
    """
    loader = TextLoader(str(POLICY_DOC_PATH), encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, get_embedding_model())


def _get_vectorstore() -> FAISS:
    """Return the cached vector store, building it on first use."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _build_vectorstore()
    return _vectorstore


def retrieve_with_confidence(query: str, k: int = RAG_TOP_K) -> Tuple[List[Document], bool]:
    """Retrieve the top-k chunks for `query` and judge whether they're relevant.

    Returns:
        (chunks, is_relevant). `is_relevant` is False when even the
        closest match is farther than RAG_RELEVANCE_DISTANCE_THRESHOLD --
        the signal rag_agent_node uses to report a failed attempt
        (triggering the SQL fallback) instead of answering from
        weak/unrelated context.
    """
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return [], False

    chunks = [doc for doc, _score in results]
    best_score = min(score for _doc, score in results)
    is_relevant = best_score <= RAG_RELEVANCE_DISTANCE_THRESHOLD
    return chunks, is_relevant


def rag_agent_node(state: GraphState) -> dict:
    """Attempt to answer the query using the document knowledge base.

    Sets rag_success=False (instead of raising, and instead of guessing an
    answer from weak context) whenever retrieval doesn't turn up
    sufficiently relevant chunks, so route_after_rag can send the query on
    to the SQL agent as a fallback.
    """
    query = state["query"]
    chunks, is_relevant = retrieve_with_confidence(query)

    if not is_relevant:
        # Deliberately do NOT call the LLM here: with no relevant context,
        # any "answer" it produced would likely be a hallucination. Failing
        # fast and honestly is what lets the SQL fallback do its job.
        return {"rag_success": False, "source": "rag"}

    context = "\n\n".join(chunk.page_content for chunk in chunks)
    prompt = (
        "Answer the question using ONLY the context below. If the context "
        "truly does not contain the answer, say so plainly instead of "
        "guessing.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    answer = invoke_chat([HumanMessage(content=prompt)])

    return {"answer": answer, "source": "rag", "rag_success": True}
