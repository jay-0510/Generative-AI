"""
src/agents package
===================
One module per LangGraph node: classifier.py (routing), sql_agent.py,
rag_agent.py, and error_node.py (fallback). Every node function has the
same signature — `(state: AgentState) -> dict` — which is the contract
LangGraph's `StateGraph.add_node` expects.
"""
