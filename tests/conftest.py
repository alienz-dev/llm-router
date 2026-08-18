"""Shared test fixtures.

The suite is hermetic and enforces it: every acceptance criterion that once
needed a live provider is a fixture instead. OpenRouter free is 50 requests a
day, so a test suite that spends real calls is a test suite nobody can run twice.
"""
import asyncio
import os
import socket
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

# Set test env before importing app modules
os.environ["OPENROUTER_API_KEY"] = "test-key"
os.environ["DATABASE_PATH"] = ":memory:"


_real_connect = socket.socket.connect


def _blocked_connect(self, address):
    """Fail loudly rather than quietly spending a free-tier request."""
    host = address[0] if isinstance(address, tuple) else address
    if self.family in (socket.AF_INET, socket.AF_INET6) and host not in (
        "127.0.0.1", "::1", "localhost"
    ):
        raise RuntimeError(
            f"the test suite tried to reach {address}. Tests are hermetic — "
            "capture a fixture in tests/fixtures/ instead."
        )
    return _real_connect(self, address)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def test_db():
    """In-memory SQLite database for testing."""
    import aiosqlite
    from llm_router.db import _init_schema

    db = await aiosqlite.connect(":memory:")
    await db.execute("PRAGMA journal_mode=WAL")
    await _init_schema(db)
    yield db
    await db.close()


@pytest.fixture
def mock_adapter():
    """Mock adapter that returns successful responses."""
    from llm_router.adapters.base import AdapterResponse

    adapter = AsyncMock()
    adapter.chat_completion.return_value = AdapterResponse(
        response={
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
        quota=None,
        tokens_in=10,
        tokens_out=5,
        latency_ms=500.0,
    )
    adapter.stream_completion.return_value = iter(["Hello", " ", "World"])
    return adapter


@pytest.fixture
def mock_adapter_error():
    """Mock adapter that returns errors."""
    from llm_router.adapters.base import AdapterResponse

    adapter = AsyncMock()
    adapter.chat_completion.return_value = AdapterResponse(
        response={"error": "Provider unavailable"},
        quota=None,
        tokens_in=0,
        tokens_out=0,
        latency_ms=100.0,
    )
    return adapter
