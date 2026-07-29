"""
tests/test_setup_services.py
==============================

WHY THIS FILE EXISTS
---------------------
`config/db_init.py` is infrastructure code that every other node depends
on — if the schema is wrong or seeding isn't actually idempotent, every
downstream node test would fail for a confusing, unrelated reason. Testing
it in isolation, first, makes later test failures much easier to diagnose.

The vector store (`init_vector_store`) is intentionally NOT unit-tested
here: it calls Bedrock (`BedrockEmbeddings`) for real, which needs live AWS
credentials and costs a few cents per run — outside the scope of what a
fast, offline unit test suite should do. It's exercised instead in
`main.py` / the notebook, which are meant to be run with real credentials.
"""

from __future__ import annotations

import sqlite3

from config.db_init import get_schema_description, init_sql_database


def test_init_sql_database_creates_expected_tables(isolated_settings):
    db_path = init_sql_database()

    conn = sqlite3.connect(db_path)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()

    assert {"customers", "products", "orders"}.issubset(tables)


def test_init_sql_database_is_idempotent(isolated_settings):
    """Calling init twice must not duplicate seed rows."""
    init_sql_database()
    db_path = init_sql_database()  # second call

    conn = sqlite3.connect(db_path)
    try:
        (count,) = conn.execute("SELECT COUNT(*) FROM customers").fetchone()
    finally:
        conn.close()

    assert count == 4  # matches len(_SEED_CUSTOMERS) in db_init.py


def test_schema_description_mentions_all_tables():
    description = get_schema_description()
    for table_name in ("customers", "products", "orders"):
        assert table_name in description
