"""
web_search_tool.py
---------------------
A MOCK web search tool, as specified by the practical brief ("web search
mock") — it does not call a real search API. It returns canned, keyword-
matched results so the agent's tool-routing and reasoning can be exercised
and tested offline, without needing a live API key or network access.

To swap in a real search API later (e.g. Tavily, SerpAPI, Bing), replace the
body of `web_search_tool` with an actual HTTP call and keep the same
`@tool`-decorated signature — nothing else in agent.py needs to change.
"""

from langchain_core.tools import tool

# Canned "search index" keyed by topic keyword. Not exhaustive — this is a
# stand-in for a real search backend, purely to demonstrate the tool being
# selected and invoked by the agent.
_MOCK_RESULTS = {
    "langchain": "LangChain is a framework for building applications powered by language models, "
    "providing abstractions for chains, agents, memory, and tool use.",
    "bedrock": "Amazon Bedrock is a fully managed AWS service that provides access to foundation "
    "models (including Anthropic's Claude) via a single API.",
    "weather": "Mock result: weather data is not available from this offline mock tool.",
}

_DEFAULT_RESULT = "Mock result: no indexed information found for this query in the offline mock tool."


@tool
def web_search_tool(query: str) -> str:
    """
    Searches a small offline mock index and returns a canned result relevant
    to the query. This is a stand-in for a real web search API — use it when
    the question needs general/current information the other tools can't
    provide (the calculator can't do lookups, and the RAG tool only knows
    the indexed documents).
    """
    query_lower = query.lower()
    for keyword, result in _MOCK_RESULTS.items():
        if keyword in query_lower:
            return result
    return _DEFAULT_RESULT
