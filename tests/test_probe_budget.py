"""The probe was the router's biggest spender, and it spent invisibly.

Every 2 hours it called every active model: 20 OpenRouter models x 12 cycles =
240 requests a day against a 50/day cap, and `record_usage` appeared nowhere in
probe.py, so none of it counted. Three controls now bound it — staleness, a
per-provider burst cap, and the quota manager itself — and what it does spend is
booked like any other request.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from llm_router.db import _init_schema
from llm_router.probe import (
    PROBE_MIN_AGE_HOURS, PROBE_QUOTA_FLOOR, RATE_LIMITED, probe_all_models,
    seed_known_limits,
)


@pytest.fixture(autouse=True)
def no_probe_delay(monkeypatch):
    """The 0.5s courtesy delay between probes is real behaviour, not something
    the suite should sit through."""
    monkeypatch.setattr("llm_router.probe.asyncio.sleep", AsyncMock())


@pytest.fixture
async def db(monkeypatch):
    import aiosqlite
    import llm_router.db as db_module

    conn = await aiosqlite.connect(":memory:")
    await _init_schema(conn)
    for pid in ("openrouter", "nvidia"):
        await conn.execute(
            "INSERT INTO providers (id, name, base_url, daily_reset_utc_hour) "
            "VALUES (?, ?, 'https://x', 0)", (pid, pid))
    await conn.commit()

    async def _get_db():
        return conn

    monkeypatch.setattr(db_module, "get_db", _get_db)
    for module in ("db", "probe", "quota", "health", "router"):
        monkeypatch.setattr(f"llm_router.{module}.get_db", _get_db, raising=False)
    yield conn
    await conn.close()


async def add_model(db, provider, model, probed_hours_ago=None):
    await db.execute(
        """INSERT INTO models (provider_id, model_id, display_name, context_length,
                               task_scores, discovered_at, active)
           VALUES (?, ?, ?, 8192, '{"general": 0.5}', '2026-08-01', 1)""",
        (provider, model, model),
    )
    if probed_hours_ago is not None:
        stamp = (datetime.now(timezone.utc)
                 - timedelta(hours=probed_hours_ago)).isoformat()
        await db.execute(
            """INSERT INTO model_health (provider_id, model_id, last_probe_at,
                                         last_probe_ok, success_count)
               VALUES (?, ?, ?, 1, 1)""",
            (provider, model, stamp),
        )
    await db.commit()


def _probe_returns(available=True):
    return AsyncMock(return_value={
        "available": available, "error": None if available else "model not found",
        "latency_ms": 10, "reached": True,
    })


class TestFreshnessGate:
    @pytest.mark.asyncio
    async def test_a_freshly_probed_model_is_left_alone(self, db):
        await add_model(db, "openrouter", "m1", probed_hours_ago=1)
        with patch("llm_router.probe.probe_model", new=_probe_returns()) as probed:
            results = await probe_all_models()
        assert probed.await_count == 0
        assert results == {}

    @pytest.mark.asyncio
    async def test_real_traffic_counts_as_freshness(self, db):
        """The router writes last_probe_at on every request through the same
        repository, so a model in use never needs a synthetic call."""
        from llm_router.health import ModelHealthRepository

        await add_model(db, "openrouter", "m1")
        await ModelHealthRepository().update("openrouter", "m1", success=True,
                                             latency_ms=120)
        with patch("llm_router.probe.probe_model", new=_probe_returns()) as probed:
            await probe_all_models()
        assert probed.await_count == 0

    @pytest.mark.asyncio
    async def test_a_stale_model_is_probed(self, db):
        await add_model(db, "openrouter", "m1",
                        probed_hours_ago=PROBE_MIN_AGE_HOURS + 1)
        with patch("llm_router.probe.probe_model", new=_probe_returns()) as probed:
            await probe_all_models()
        assert probed.await_count == 1

    @pytest.mark.asyncio
    async def test_never_probed_models_go_first(self, db):
        """An unknown model is what makes the router guess."""
        for i in range(3):
            await add_model(db, "openrouter", f"old-{i}",
                            probed_hours_ago=PROBE_MIN_AGE_HOURS + 10 + i)
        await add_model(db, "openrouter", "brand-new")

        with patch("llm_router.probe.probe_model", new=_probe_returns()) as probed:
            await probe_all_models(budget_per_provider=1)

        assert probed.await_args.args[2] == "brand-new"


class TestBudget:
    @pytest.mark.asyncio
    async def test_a_cycle_stops_at_the_per_provider_cap(self, db):
        for i in range(20):
            await add_model(db, "openrouter", f"m{i}")
        with patch("llm_router.probe.probe_model", new=_probe_returns()) as probed:
            await probe_all_models(budget_per_provider=8)
        assert probed.await_count == 8

    @pytest.mark.asyncio
    async def test_providers_are_budgeted_separately(self, db):
        for i in range(5):
            await add_model(db, "openrouter", f"m{i}")
            await add_model(db, "nvidia", f"n{i}")
        with patch("llm_router.probe.probe_model", new=_probe_returns()) as probed:
            results = await probe_all_models(budget_per_provider=2)
        assert probed.await_count == 4
        assert len(results["openrouter"]) == 2
        assert len(results["nvidia"]) == 2

    @pytest.mark.asyncio
    async def test_a_nearly_exhausted_provider_is_not_probed(self, db):
        """Liveness checks must not be what finishes off the daily budget."""
        await seed_known_limits()
        await add_model(db, "openrouter", "m1")
        now = datetime.now(timezone.utc).isoformat()
        spent = int(50 * (1 - PROBE_QUOTA_FLOOR)) + 1
        await db.executemany(
            "INSERT INTO quota_usage (provider_id, model_id, timestamp, tokens_in,"
            " tokens_out, success) VALUES ('openrouter', 'x', ?, 0, 0, 1)",
            [(now,) for _ in range(spent)],
        )
        await db.commit()

        with patch("llm_router.probe.probe_model", new=_probe_returns()) as probed:
            await probe_all_models()
        assert probed.await_count == 0

    @pytest.mark.asyncio
    async def test_what_the_probe_spends_is_recorded(self, db):
        await add_model(db, "openrouter", "m1")
        with patch("llm_router.probe.probe_model", new=_probe_returns()):
            await probe_all_models()
        async with db.execute(
            "SELECT COUNT(*) FROM quota_usage WHERE provider_id = 'openrouter'"
        ) as cur:
            assert (await cur.fetchone())[0] == 1

    @pytest.mark.asyncio
    async def test_a_request_that_never_reached_the_provider_is_not_charged(self, db):
        await add_model(db, "openrouter", "m1")
        unreachable = AsyncMock(return_value={"available": False, "error": "timeout"})
        with patch("llm_router.probe.probe_model", new=unreachable):
            await probe_all_models()
        async with db.execute("SELECT COUNT(*) FROM quota_usage") as cur:
            assert (await cur.fetchone())[0] == 0


class TestRateLimitedSweep:
    @pytest.mark.asyncio
    async def test_untouched_models_get_no_verdict(self, db):
        """They were reported unavailable, which incremented consecutive_failures
        on models nobody had called. Five is a hard skip from `auto`; five real
        HuggingFace models sat at four."""
        for i in range(5):
            await add_model(db, "openrouter", f"m{i}")

        calls = []

        async def _probe(client, pid, mid):
            calls.append(mid)
            return {"available": False, "reached": True, "latency_ms": 1,
                    "error": f"{RATE_LIMITED} (retry-after=60)"}

        with patch("llm_router.probe.probe_model", new=_probe):
            results = await probe_all_models()

        assert len(calls) == 1, "the sweep should stop at the first rate limit"
        assert len(results["openrouter"]) == 1
        assert "m1" not in results["openrouter"]

    def test_one_spelling_of_the_marker(self):
        """The producer said "rate limited", every check tested "rate_limited"."""
        import inspect

        from llm_router import probe

        source = inspect.getsource(probe)
        assert "rate limited (skipped)" not in source
