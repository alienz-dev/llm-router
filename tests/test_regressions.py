"""Defects found by auditing the tree after the agent-ready sprint landed.

Each one is a specific wrong answer the router gave, not a style complaint.
"""
import asyncio
import json

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from llm_router.adapters.base import AdapterResponse, describe_exception
from llm_router.db import _init_schema
from llm_router.quota import QuotaManager
from llm_router.router import SmartRouter


@pytest.fixture
async def db(monkeypatch):
    import aiosqlite
    import llm_router.db as db_module

    conn = await aiosqlite.connect(":memory:")
    await _init_schema(conn)
    await conn.execute(
        "INSERT INTO providers (id, name, base_url) VALUES ('openrouter', 'or', 'https://x')")
    await conn.commit()

    async def _get_db():
        return conn

    monkeypatch.setattr(db_module, "get_db", _get_db)
    for module in ("db", "router", "quota", "health", "app", "capabilities"):
        monkeypatch.setattr(f"llm_router.{module}.get_db", _get_db, raising=False)
    yield conn
    await conn.close()


class TestModelHealthSuccessRate:
    """/v1/models/health divided success_count by (avg_latency_ms + success_count),
    so every model with any measured latency reported a success rate near zero."""

    @pytest.mark.asyncio
    async def test_success_rate_is_successes_over_attempts(self, db):
        from llm_router import app as app_module

        await db.execute(
            """INSERT INTO models (provider_id, model_id, display_name, active)
               VALUES ('openrouter', 'm1', 'm1', 1)""")
        await db.execute(
            """INSERT INTO model_health (provider_id, model_id, avg_latency_ms,
                                         success_count, failure_count)
               VALUES ('openrouter', 'm1', 1500.0, 3, 1)""")
        await db.commit()

        transport = ASGITransport(app=app_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            body = (await client.get("/v1/models/health")).json()

        model = body["models"][0]
        assert model["success_count"] == 3
        assert model["failure_count"] == 1
        assert model["success_rate"] == 0.75


class TestCapabilityProbeSurvivesAKeylessProvider:
    """probe_one returned early with no `provider` key when a key was missing;
    the caller read record["provider"] and took down the whole weekly cycle."""

    @pytest.mark.asyncio
    async def test_a_missing_key_does_not_crash_the_cycle(self, db, monkeypatch):
        from llm_router.capabilities import probe_one, run_capability_probe

        await db.execute(
            """INSERT INTO models (provider_id, model_id, display_name, active)
               VALUES ('huggingface', 'gpt2', 'gpt2', 1)""")
        await db.execute(
            """INSERT INTO models (provider_id, model_id, display_name, active)
               VALUES ('openrouter', 'm1', 'm1', 1)""")
        await db.commit()
        monkeypatch.setenv("HF_API_TOKEN", "")

        record = await probe_one(MagicMock(), "huggingface", "gpt2")
        assert record["provider"] == "huggingface"
        assert record["model"] == "gpt2"

        async def _one(client, provider_id, model_id):
            return await probe_one(client, provider_id, model_id)

        with patch("llm_router.capabilities.probe_one", new=_one), \
             patch("llm_router.capabilities.asyncio.sleep", new=AsyncMock()):
            stats = await run_capability_probe()

        assert stats["probed"] == 2  # it completed rather than raising

    @pytest.mark.asyncio
    async def test_an_unconfigured_provider_is_named_too(self, db):
        from llm_router.capabilities import probe_one

        record = await probe_one(MagicMock(), "not-a-provider", "m1")
        assert record["provider"] == "not-a-provider"
        assert record["tools"] == "no-config"


class TestAbandonedStreamIsStillBooked:
    """GeneratorExit is a BaseException, so `except Exception` missed a client
    hanging up — and the upstream request it had already spent went unrecorded."""

    @pytest.mark.asyncio
    async def test_a_disconnect_mid_stream_records_the_request(self, db):
        await db.execute(
            """INSERT INTO models (provider_id, model_id, display_name, active)
               VALUES ('openrouter', 'm1', 'm1', 1)""")
        await db.commit()

        async def source():
            for word in ("the", "client", "leaves", "now"):
                yield {"choices": [{"index": 0, "delta": {"content": word}}]}

        router = SmartRouter({"openrouter": MagicMock()}, QuotaManager())
        stream = router._metered_stream(
            "openrouter", "m1", source(), [{"role": "user", "content": "hi"}],
            {}, {"provider": "openrouter", "model": "m1"})

        assert (await stream.__anext__())["choices"][0]["delta"]["content"] == "the"
        await stream.aclose()
        for _ in range(5):  # let the scheduled accounting run
            await asyncio.sleep(0)

        async with db.execute(
            "SELECT COUNT(*) FROM quota_usage WHERE provider_id = 'openrouter'"
        ) as cur:
            assert (await cur.fetchone())[0] == 1

    @pytest.mark.asyncio
    async def test_a_fully_consumed_stream_is_booked_once(self, db):
        await db.execute(
            """INSERT INTO models (provider_id, model_id, display_name, active)
               VALUES ('openrouter', 'm1', 'm1', 1)""")
        await db.commit()

        async def source():
            yield {"choices": [{"index": 0, "delta": {"content": "hello"}}]}

        router = SmartRouter({"openrouter": MagicMock()}, QuotaManager())
        stream = router._metered_stream(
            "openrouter", "m1", source(), [{"role": "user", "content": "hi"}],
            {}, {"provider": "openrouter", "model": "m1"})
        [chunk async for chunk in stream]
        for _ in range(5):
            await asyncio.sleep(0)

        async with db.execute("SELECT COUNT(*) FROM quota_usage") as cur:
            assert (await cur.fetchone())[0] == 1


class TestTransportFailuresAreLegible:
    """str(httpx.ConnectTimeout()) is the empty string. The error carried no
    text, the circuit breaker matches on text, and so it never opened on a
    provider that had gone completely dark."""

    def test_an_empty_exception_is_still_named(self):
        assert describe_exception(httpx.ConnectTimeout("")) == "ConnectTimeout"
        assert describe_exception(httpx.ConnectError("no route")) == (
            "ConnectError: no route")

    def test_the_breaker_opens_on_silent_timeouts(self):
        from llm_router.circuit_breaker import BreakerState, ProviderBreaker

        breaker = ProviderBreaker()
        for _ in range(5):
            breaker.record_failure(describe_exception(httpx.ConnectTimeout("")))
        assert breaker.state == BreakerState.OPEN

    def test_the_breaker_opens_on_connection_errors(self):
        from llm_router.circuit_breaker import BreakerState, ProviderBreaker

        breaker = ProviderBreaker()
        for _ in range(5):
            breaker.record_failure(describe_exception(httpx.ConnectError("no route")))
        assert breaker.state == BreakerState.OPEN

    @pytest.mark.asyncio
    async def test_an_adapter_never_returns_a_blank_error(self):
        from llm_router.adapters.base import OpenAICompatibleAdapter

        adapter = OpenAICompatibleAdapter.__new__(OpenAICompatibleAdapter)
        adapter.provider_name = "openrouter"
        adapter.base_url = "https://example.test/v1"
        adapter.api_key = "k"
        adapter.client = MagicMock()
        adapter.client.post = AsyncMock(side_effect=httpx.ConnectTimeout(""))

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await adapter.chat_completion([{"role": "user", "content": "hi"}], "m")

        assert result.response["error"] == "ConnectTimeout"


class TestUnknownModelIsNotFound:
    """A model nobody has heard of came back as 502, so a client retried it."""

    @pytest.mark.asyncio
    async def test_unknown_model_is_404(self, db):
        result = await SmartRouter({}, QuotaManager()).route(
            [{"role": "user", "content": "hi"}], model_override="not-a-real-model")
        assert result.status == 404

    @pytest.mark.asyncio
    async def test_unknown_provider_is_404(self, db):
        result = await SmartRouter({}, QuotaManager()).route(
            [{"role": "user", "content": "hi"}], model_override="nope:some-model")
        assert result.status == 404

    @pytest.mark.asyncio
    async def test_the_app_returns_it_as_404(self, db):
        from llm_router import app as app_module

        router = SmartRouter({}, QuotaManager())
        with patch.object(app_module, "_router", router):
            transport = ASGITransport(app=app_module.app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/chat/completions", json={
                    "model": "not-a-real-model",
                    "messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 404
