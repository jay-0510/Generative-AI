"""
config/db_init.py
------------------
Creates and seeds the small sample SQLite database used by the SQL agent
path (src/agents/sql_agent.py).

WHY THIS LIVES IN config/ RATHER THAN src/agents/
    Creating the database is a one-time SETUP concern -- it needs to
    happen once, before the SQL agent can ever run a query against it --
    which is a different kind of responsibility than the SQL agent's own
    per-request logic (generate SQL, execute it, summarize results). Since
    settings.py already centralizes configuration/setup concerns, putting
    db_init.py alongside it keeps all "things that need to happen before
    the agents can run" in one place.

WHY A LOCAL, GENERATED SQLITE DATABASE
    Keeps the project fully self-contained -- no external database
    server, driver, or credentials required. The .db file itself is left
    out of git (see .gitignore) and rebuilt automatically the first time
    this module runs.
"""

import sqlite3

from config.settings import DB_PATH

# Embedded directly here (and imported by src/agents/sql_agent.py for its
# SQL-generation prompt) so the LLM knows exactly which tables/columns
# exist, without needing a separate live schema-introspection tool call.
SCHEMA_DESCRIPTION = """\
Table: employees(id INTEGER, name TEXT, department TEXT, salary INTEGER, hire_year INTEGER)
Table: products(id INTEGER, name TEXT, category TEXT, price REAL, stock INTEGER)
"""

_SAMPLE_EMPLOYEES = [
    (1, "Ava Thompson", "Engineering", 98000, 2019),
    (2, "Liam Carter", "Engineering", 87000, 2021),
    (3, "Sofia Ramirez", "Sales", 72000, 2020),
    (4, "Noah Patel", "Sales", 65000, 2022),
    (5, "Maria Chen", "Marketing", 71000, 2018),
    (6, "Ethan Wright", "Engineering", 105000, 2017),
    (7, "Grace Kim", "Marketing", 68000, 2023),
    (8, "Lucas Silva", "Support", 55000, 2021),
]

_SAMPLE_PRODUCTS = [
    (1, "Wireless Mouse", "Electronics", 19.99, 150),
    (2, "Mechanical Keyboard", "Electronics", 79.99, 80),
    (3, "Standing Desk", "Furniture", 349.99, 20),
    (4, "Desk Lamp", "Furniture", 24.99, 60),
    (5, "Noise-Cancelling Headphones", "Electronics", 129.99, 45),
    (6, "Office Chair", "Furniture", 189.99, 35),
]


def ensure_sample_database(db_path=None) -> None:
    """Create and populate the sample SQLite DB if it doesn't already exist.

    Idempotent: safe to call at the top of every run (main.py, the
    notebook, and every test in tests/test_setup_services.py all call this
    before touching the database). Accepts an optional `db_path` override
    purely so tests can point it at a throwaway file instead of the real
    project database. Defaults to None, resolved to
    config.settings.DB_PATH *at call time* rather than at import time --
    see utils/helpers.py's execute_readonly_query for why that distinction
    matters for testability.
    """
    if db_path is None:
        db_path = DB_PATH

    if db_path.exists():
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                salary INTEGER NOT NULL,
                hire_year INTEGER NOT NULL
            )"""
        )
        cur.execute(
            """CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                stock INTEGER NOT NULL
            )"""
        )
        cur.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", _SAMPLE_EMPLOYEES)
        cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", _SAMPLE_PRODUCTS)
        conn.commit()
    finally:
        conn.close()
