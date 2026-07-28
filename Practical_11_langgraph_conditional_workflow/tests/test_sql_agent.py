"""
tests/test_sql_agent.py
-------------------------
Tests for the SQL safety guard (utils/helpers.py), src/agents/sql_agent.py's
node behavior, and route_after_sql (src/graph/workflow.py).
"""

import sqlite3

import pytest

import src.agents.sql_agent as sql_module
from config.db_init import ensure_sample_database
from src.graph.workflow import route_after_sql
from utils import helpers as helpers_module


@pytest.fixture
def temp_db(tmp_path):
    """A freshly-seeded, throwaway database for each test -- keeps SQL
    tests isolated from each other and from the real project database."""
    db_path = tmp_path / "test_company_sample.db"
    ensure_sample_database(db_path=db_path)
    return db_path


class TestExecuteReadonlyQuery:
    def test_valid_select_returns_rows(self, temp_db):
        columns, rows = helpers_module.execute_readonly_query(
            "SELECT name, department FROM employees WHERE department = 'Engineering'",
            db_path=temp_db,
        )
        assert columns == ["name", "department"]
        assert len(rows) == 3  # Ava, Liam, Ethan in the sample data

    def test_select_with_trailing_semicolon_is_allowed(self, temp_db):
        columns, _rows = helpers_module.execute_readonly_query(
            "SELECT * FROM products;", db_path=temp_db
        )
        assert "name" in columns

    @pytest.mark.parametrize(
        "unsafe_sql",
        [
            "DROP TABLE employees",
            "DELETE FROM employees WHERE id = 1",
            "UPDATE employees SET salary = 0",
            "INSERT INTO employees VALUES (99, 'x', 'x', 1, 2020)",
            "ATTACH DATABASE 'evil.db' AS evil",
            "PRAGMA table_info(employees)",
        ],
    )
    def test_non_select_statements_are_rejected(self, temp_db, unsafe_sql):
        with pytest.raises(ValueError):
            helpers_module.execute_readonly_query(unsafe_sql, db_path=temp_db)

    def test_stacked_statements_are_rejected(self, temp_db):
        with pytest.raises(ValueError):
            helpers_module.execute_readonly_query(
                "SELECT * FROM employees; DROP TABLE employees;", db_path=temp_db
            )

    def test_malformed_sql_raises_sqlite_error(self, temp_db):
        with pytest.raises(sqlite3.Error):
            helpers_module.execute_readonly_query("SELECT FROM WHERE", db_path=temp_db)


class TestSqlAgentNode:
    def test_success_path_returns_answer(self, monkeypatch, temp_db):
        # sql_agent_node calls execute_readonly_query with no db_path,
        # which resolves utils.helpers.DB_PATH at call time -- point that
        # at our throwaway, pre-seeded database for this test.
        monkeypatch.setattr(helpers_module, "DB_PATH", temp_db)

        replies = iter(
            [
                "SELECT COUNT(*) AS n FROM employees WHERE department='Engineering'",
                "There are 3 employees in Engineering.",
            ]
        )
        monkeypatch.setattr(sql_module, "invoke_chat", lambda messages: next(replies))

        result = sql_module.sql_agent_node({"query": "How many employees in Engineering?"})
        assert result["sql_success"] is True
        assert result["source"] == "sql"
        assert "3 employees" in result["answer"]

    def test_unsafe_generated_sql_is_reported_as_failure_not_raised(self, monkeypatch, temp_db):
        """If the model ever generates something destructive, sql_agent_node
        must catch it and report a normal failure state -- never let it
        propagate and crash the graph."""
        monkeypatch.setattr(helpers_module, "DB_PATH", temp_db)
        monkeypatch.setattr(sql_module, "invoke_chat", lambda messages: "DROP TABLE employees")

        result = sql_module.sql_agent_node({"query": "anything"})
        assert result["sql_success"] is False
        assert result["source"] == "sql"
        assert "error_message" in result

    def test_query_with_no_matching_rows_is_a_failure(self, monkeypatch, temp_db):
        monkeypatch.setattr(helpers_module, "DB_PATH", temp_db)
        monkeypatch.setattr(
            sql_module,
            "invoke_chat",
            lambda messages: "SELECT * FROM employees WHERE department='Legal'",
        )

        result = sql_module.sql_agent_node({"query": "employees in Legal?"})
        assert result["sql_success"] is False
        assert "no rows" in result["error_message"]


class TestRouteAfterSql:
    def test_success_routes_to_end(self):
        assert route_after_sql({"sql_success": True}) == "end"

    def test_failure_routes_to_error_node(self):
        assert route_after_sql({"sql_success": False}) == "error_node"

    def test_missing_flag_treated_as_failure(self):
        assert route_after_sql({}) == "error_node"
