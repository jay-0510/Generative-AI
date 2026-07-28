"""
src package
-----------
The project's application code.

    llm_invoke.py -> shared Amazon Bedrock (Nova + Titan) access layer
                     every agent goes through
    agents/        -> the four LangGraph node functions: classifier,
                     rag_agent, sql_agent, error_node
    graph/          -> the shared state schema and the compiled
                     LangGraph workflow that wires the agents together
"""
