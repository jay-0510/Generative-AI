"""
tests package
-------------
pytest suite for this project. Every test runs fully offline -- no real
AWS/Bedrock network calls are made anywhere in this suite (see
conftest.py for the fake-credentials fixture, and each test file for how
`invoke_chat` is monkeypatched at the point of use rather than actually
called).

    test_classify.py         -> classifier.py's labeling logic +
                               route_after_classification
    test_rag_agent.py          -> rag_agent.py's retrieval-confidence logic
                               and node behavior + route_after_rag
    test_sql_agent.py           -> the SQL safety guard (utils/helpers.py),
                               sql_agent.py's node behavior, and
                               route_after_sql
    test_setup_services.py        -> config/settings.py, config/db_init.py,
                               and src/llm_invoke.py's construction

Run with: pytest -v   (from the project root)
"""
