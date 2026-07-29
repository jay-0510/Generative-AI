"""
tests/conftest.py
===================

WHY THIS FILE EXISTS
---------------------
Every node in this project calls out to Amazon Bedrock. Unit tests must NOT
make real network/AWS calls: that would make the test suite slow, flaky
(needs real credentials + network), AND cost real money on every CI run —
exactly the kind of hidden cost this practical is teaching you to notice.

The fixtures below fake out the LLM boundary (`get_chat_model`) with an
in-memory stand-in that returns pre-programmed responses, and point every
data store (SQLite DB, Chroma dir, FAQ docs) at a pytest `tmp_path` so
tests never touch the real `data/` folder used by `main.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from config.settings import get_settings


@dataclass
class _FakeAIMessage:
    """Minimal stand-in for a LangChain `AIMessage` — just enough surface
    area (`.content`, `.usage_metadata`) for `invoke_with_usage` to read."""

    content: str
    usage_metadata: dict = field(default_factory=lambda: {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


class FakeChatModel:
    """
    Stand-in for `ChatBedrockConverse` used in every test.

    Give it a list of canned string responses at construction time; each
    call to `.invoke()` pops the next one off the front. This mirrors how
    e.g. `sql_agent_node` calls the model twice in sequence (generate SQL,
    then phrase the answer) and needs a different canned response each time.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def invoke(self, messages: Any, config: Any = None) -> _FakeAIMessage:
        if not self._responses:
            raise AssertionError("FakeChatModel ran out of canned responses — add more in the test.")
        return _FakeAIMessage(content=self._responses.pop(0))


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """
    Point every on-disk data store at a fresh pytest tmp_path, so tests
    never read/write the real project `data/` directory, and don't
    interfere with each other or with a concurrently-running `main.py`.
    """
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test_app.db"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge_base"))
    get_settings.cache_clear()  # Force Settings() to re-read the patched env vars.
    yield get_settings()
    get_settings.cache_clear()  # Don't leak the test config into later tests.


@pytest.fixture
def patch_chat_model(monkeypatch):
    """
    Returns a helper `patch(module_path, responses)` that replaces
    `get_chat_model` in a given agent module with a factory returning a
    `FakeChatModel(responses)` — regardless of which model_id is requested.

    Usage in a test:
        patch_chat_model("src.agents.classifier", ["SQL"])
    """

    def _patch(module_path: str, responses: list[str]) -> FakeChatModel:
        fake_model = FakeChatModel(responses)
        monkeypatch.setattr(f"{module_path}.get_chat_model", lambda *args, **kwargs: fake_model)
        return fake_model

    return _patch
