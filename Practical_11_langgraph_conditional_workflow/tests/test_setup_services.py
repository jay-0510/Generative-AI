"""
tests/test_setup_services.py
------------------------------
Tests for the project's setup/configuration layer:
config/settings.py, config/db_init.py, and src/llm_invoke.py.

These are the modules every agent depends on before it can do anything
else -- if settings resolve to the wrong type, or the database isn't
seeded correctly, or the Bedrock clients don't construct, every agent
breaks the same way. Testing them in one file, separately from the
agents that consume them, makes failures here easy to tell apart from an
actual agent-logic bug.
"""

import sqlite3

import config.settings as settings_module
from config.db_init import SCHEMA_DESCRIPTION, ensure_sample_database


class TestSettings:
    def test_model_ids_are_amazon_models_only(self):
        """Locks in this project's core requirement: no Anthropic/OpenAI/
        Google model IDs anywhere in the default configuration."""
        assert settings_module.NOVA_MODEL_ID.startswith("amazon.")
        assert settings_module.TITAN_EMBEDDING_MODEL_ID.startswith("amazon.")

    def test_temperature_is_a_float_and_deterministic_by_default(self):
        assert isinstance(settings_module.LLM_TEMPERATURE, float)
        assert settings_module.LLM_TEMPERATURE == 0.0

    def test_rag_threshold_and_top_k_have_sane_types_and_ranges(self):
        assert isinstance(settings_module.RAG_RELEVANCE_DISTANCE_THRESHOLD, float)
        assert settings_module.RAG_RELEVANCE_DISTANCE_THRESHOLD > 0
        assert isinstance(settings_module.RAG_TOP_K, int)
        assert settings_module.RAG_TOP_K > 0

    def test_paths_are_resolved_under_the_project_root(self):
        assert settings_module.DATA_DIR.is_relative_to(settings_module.PROJECT_ROOT)
        assert settings_module.POLICY_DOC_PATH.is_relative_to(settings_module.DATA_DIR)

    def test_policy_document_referenced_by_settings_actually_exists(self):
        """A broken path here would silently break the RAG agent the
        first time it tries to load its knowledge base -- catch it here
        instead."""
        assert settings_module.POLICY_DOC_PATH.exists()


class TestDbInit:
    def test_schema_description_mentions_both_tables(self):
        assert "employees" in SCHEMA_DESCRIPTION
        assert "products" in SCHEMA_DESCRIPTION

    def test_creates_database_with_expected_tables(self, tmp_path):
        db_path = tmp_path / "test.db"
        ensure_sample_database(db_path=db_path)

        conn = sqlite3.connect(db_path)
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        conn.close()
        assert {"employees", "products"}.issubset(tables)

    def test_is_idempotent_and_does_not_duplicate_rows(self, tmp_path):
        db_path = tmp_path / "test.db"
        ensure_sample_database(db_path=db_path)
        ensure_sample_database(db_path=db_path)  # second call should no-op

        conn = sqlite3.connect(db_path)
        employee_count = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
        product_count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        conn.close()
        assert employee_count == 8
        assert product_count == 6

    def test_does_not_touch_an_already_existing_file(self, tmp_path):
        """If the DB file already exists, ensure_sample_database must
        leave it alone rather than silently recreating it -- this test
        would catch an accidental "always overwrite" regression."""
        db_path = tmp_path / "test.db"
        ensure_sample_database(db_path=db_path)

        original_mtime = db_path.stat().st_mtime
        ensure_sample_database(db_path=db_path)
        assert db_path.stat().st_mtime == original_mtime


class TestLlmInvoke:
    def test_chat_and_embedding_clients_construct_with_fake_credentials(self):
        """Confirms src/llm_invoke.py's module-level ChatBedrockConverse /
        BedrockEmbeddings construction succeeds and never attempts a real
        network call just by being imported (see conftest.py's
        fake_credentials fixture, which supplies the placeholder AWS
        values this relies on)."""
        import importlib

        from src import llm_invoke

        importlib.reload(llm_invoke)  # re-run module-level construction under fake creds
        assert llm_invoke._chat_model is not None
        assert llm_invoke._embedding_model is not None
        assert llm_invoke.get_embedding_model() is llm_invoke._embedding_model

    def test_invoke_chat_returns_the_models_content_field(self, monkeypatch):
        """invoke_chat() must unwrap AIMessage.content into a plain
        string -- verified here with a fake chat model standing in for
        ChatBedrockConverse, so no real Bedrock call happens."""
        from src import llm_invoke

        class _FakeResponse:
            content = "fake reply text"

        class _FakeChatModel:
            def invoke(self, messages):
                return _FakeResponse()

        monkeypatch.setattr(llm_invoke, "_chat_model", _FakeChatModel())
        assert llm_invoke.invoke_chat([]) == "fake reply text"
