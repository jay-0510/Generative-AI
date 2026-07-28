"""
src/agents package
-------------------
The four LangGraph node functions, each in its own module so it can be
understood, tested, and modified independently:

    classifier.py -> classify_query_node: labels a query "sql", "rag", or
                     "unknown"
    rag_agent.py    -> rag_agent_node: tries to answer from the document
                     knowledge base FIRST (see this project's routing
                     design in src/graph/workflow.py)
    sql_agent.py     -> sql_agent_node: tries to answer by generating and
                     safely executing SQL -- reached directly for "sql"
                     queries, or as the RAG fallback
    error_node.py      -> error_node: the graph's graceful, explicit
                     failure path

Every node function has the same shape: `def node(state: GraphState) ->
dict`, reading what it needs from state and returning only the keys it
wants to update.
"""

from src.agents.classifier import classify_query_node
from src.agents.error_node import error_node
from src.agents.rag_agent import rag_agent_node
from src.agents.sql_agent import sql_agent_node

__all__ = ["classify_query_node", "rag_agent_node", "sql_agent_node", "error_node"]
