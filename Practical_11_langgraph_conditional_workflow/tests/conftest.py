"""
tests/conftest.py
-------------------
Shared pytest configuration for the whole test suite.

WHY EVERY TEST GETS FAKE AWS CREDENTIALS (autouse fixture below)
    src/llm_invoke.py constructs its Bedrock clients (ChatBedrockConverse,
    BedrockEmbeddings) at IMPORT time. Constructing those clients doesn't
    make a network call by itself (boto3 only talks to AWS when a method
    like .invoke() is actually called), but it DOES require *some*
    credentials and region to be present in the environment, or
    construction raises. Setting harmless fake values here means every
    test file can `import src.agents...` (which transitively imports
    src.llm_invoke) without needing real AWS credentials, while no test in
    this suite ever calls .invoke() directly and therefore never actually
    reaches the network -- every test that needs an agent's LLM call
    monkeypatches `invoke_chat` at the point of use instead (see
    tests/test_classify.py, test_rag_agent.py, test_sql_agent.py for the
    pattern).

WHY sys.path IS SET UP HERE, AT THE PROJECT ROOT
    pytest automatically adds a conftest.py's directory to sys.path before
    collecting tests. Placing it at the project root -- the same level as
    the config/, src/, and utils/ packages -- means every test file can
    `import config...` / `import src...` / `import utils...` without each
    one needing its own sys.path hack, regardless of which directory
    pytest is invoked from.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def fake_credentials(monkeypatch):
    """Give every test harmless, fake AWS credentials.

    autouse=True means this runs before EVERY test in the suite without
    each test file needing to request it explicitly -- appropriate here
    because "don't accidentally touch a real cloud service" is a blanket
    rule for this whole offline test suite, not a per-test concern.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "FAKEKEYFORTESTONLY123")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fakeSecretForTestingOnly1234567890")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    yield
