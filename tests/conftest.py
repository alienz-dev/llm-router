"""Shared test fixtures."""
import asyncio
import os
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

# Set test env before importing app modules
os.environ["OPENROUTER_API_KEY"] = "test-key"
os.environ["DATABASE_PATH"] = ":memory:"


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
