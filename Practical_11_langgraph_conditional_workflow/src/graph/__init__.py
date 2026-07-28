"""
src/graph package
------------------
    state.py    -> GraphState, the shared TypedDict passed between every node
    workflow.py -> build_graph(): wires the four agents (see src/agents/)
                   and the routing logic into a compiled LangGraph StateGraph

NOTE: this __init__.py deliberately does NOT re-export `build_graph` here
(no `from src.graph.workflow import build_graph`). workflow.py imports
from src.agents, and src.agents' submodules import GraphState from
src.graph.state -- eagerly importing workflow.py from this package's own
__init__.py would make that a circular import (Python must finish running
a package's __init__.py before it can import any of that package's
submodules, including state.py). Import what you need directly instead:
`from src.graph.workflow import build_graph` or
`from src.graph.state import GraphState`.
"""

