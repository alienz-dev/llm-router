"""Discovery and the probe disagree about which models exist. The probe wins.

Providers keep listing models that 404 when you call them. Discovery used to set
active = 1 unconditionally on every run, so the probe would switch a dead model
off and the next discovery cycle would switch it back on — and since the probe
only looks at active rows, nothing ever re-tested it. Six NVIDIA models were
live in that loop when this was written.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from llm_router.db import _init_schema


@pytest.fixture
async def db(monkeypatch):
    import aiosqlite
    import llm_router.db as db_module

    conn = await aiosqlite.connect(":memory:")
    await _init_schema(conn)
    await conn.execute(
        "INSERT INTO providers (id, name, base_url) VALUES ('nvidia', 'NVIDIA NIM', 'https://x')"
    )
    await conn.commit()

    async def _get_db():
        return conn

    monkeypatch.setattr(db_module, "get_db", _get_db)
    monkeypatch.setattr("llm_router.discovery.get_db", _get_db)
    monkeypatch.setattr("llm_router.probe.get_db", _get_db)
    monkeypatch.setattr("llm_router.health.get_db", _get_db)
    # The probe consults the quota ledger before spending a request, so the fake
    # database has to cover that module too.
    monkeypatch.setattr("llm_router.quota.get_db", _get_db)
    yield conn
    await conn.close()


async def _insert(db, model_id, active=1, reason=None, at=None):
    await db.execute(
        """INSERT INTO models (provider_id, model_id, display_name, context_length,
                               task_scores, discovered_at, active,
                               deactivated_reason, deactivated_at)
           VALUES ('nvidia', ?, ?, 4096, '{}', '2026-01-01', ?, ?, ?)""",
        (model_id, model_id, active, reason, at),
    )
    await db.commit()


async def _active(db, model_id) -> int:
    async with db.execute(
        "SELECT active FROM models WHERE model_id = ?", (model_id,)
    ) as cur:
        return (await cur.fetchone())[0]


DEAD = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
LIVE = "meta/llama-3.3-70b-instruct"


def _discovered(*model_ids):
    return {
        "nvidia": [
            {"model_id": m, "display_name": m, "context_length": 4096,
             "task_scores": json.dumps({"general": 0.5})}
            for m in model_ids
        ]
    }


class TestDiscoveryRespectsProbeVerdicts:
    @pytest.mark.asyncio
    async def test_does_not_revive_a_model_the_probe_killed(self, db):
        from llm_router.discovery import ModelDiscovery

        await _insert(db, DEAD, active=0, reason="model not found",
                      at=datetime.now(timezone.utc).isoformat())

        # The provider still lists it. That is not evidence it works.
        await ModelDiscovery().update_models_table(_discovered(DEAD))

        assert await _active(db, DEAD) == 0

    @pytest.mark.asyncio
    async def test_still_activates_models_nobody_deactivated(self, db):
        from llm_router.discovery import ModelDiscovery

        await ModelDiscovery().update_models_table(_discovered(LIVE))
        assert await _active(db, LIVE) == 1

    @pytest.mark.asyncio
    async def test_delisted_model_records_why(self, db):
        from llm_router.discovery import ModelDiscovery

        await _insert(db, LIVE, active=1)
        await ModelDiscovery().update_models_table(_discovered())  # no longer listed

        async with db.execute(
            "SELECT active, deactivated_reason, deactivated_at FROM models WHERE model_id = ?",
            (LIVE,),
        ) as cur:
            active, reason, at = await cur.fetchone()
        assert active == 0
        assert reason == "no longer listed by provider"
        assert at is not None


class TestProbeRechecksDeactivated:
    @pytest.mark.asyncio
    async def test_stale_deactivation_is_re_probed(self, db):
        """Otherwise a transient 404 is a life sentence."""
        from llm_router.probe import DEACTIVATED_RECHECK_DAYS, probe_all_models
        from unittest.mock import AsyncMock, patch

        stale = (datetime.now(timezone.utc)
                 - timedelta(days=DEACTIVATED_RECHECK_DAYS + 1)).isoformat()
        await _insert(db, DEAD, active=0, reason="model not found", at=stale)

        with patch("llm_router.probe.probe_model", new=AsyncMock(
            return_value={"available": True, "error": None, "latency_ms": 10}
        )) as probed:
            await probe_all_models()

        assert probed.await_count == 1, "stale deactivated model should be re-probed"

    @pytest.mark.asyncio
    async def test_fresh_deactivation_is_left_alone(self, db):
        from llm_router.probe import probe_all_models
        from unittest.mock import AsyncMock, patch

        fresh = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        await _insert(db, DEAD, active=0, reason="model not found", at=fresh)

        with patch("llm_router.probe.probe_model", new=AsyncMock(
            return_value={"available": True, "error": None, "latency_ms": 10}
        )) as probed:
            await probe_all_models()

        assert probed.await_count == 0


class TestProbeVerdictsAreRecorded:
    @pytest.mark.asyncio
    async def test_404_deactivates_with_a_reason(self, db):
        from llm_router.probe import run_probe_and_update
        from unittest.mock import AsyncMock, patch

        await _insert(db, DEAD, active=1)

        with patch("llm_router.probe.probe_all_models", new=AsyncMock(
            return_value={"nvidia": {DEAD: {"available": False, "error": "model not found"}}}
        )), patch("llm_router.probe.seed_known_limits", new=AsyncMock()):
            stats = await run_probe_and_update()

        async with db.execute(
            "SELECT active, deactivated_reason FROM models WHERE model_id = ?", (DEAD,)
        ) as cur:
            active, reason = await cur.fetchone()
        assert active == 0
        assert reason == "model not found"
        assert any(DEAD in entry for entry in stats["deactivated"])

    @pytest.mark.asyncio
    async def test_success_clears_the_deactivation(self, db):
        from llm_router.probe import run_probe_and_update
        from unittest.mock import AsyncMock, patch

        await _insert(db, DEAD, active=0, reason="model not found",
                      at=datetime.now(timezone.utc).isoformat())

        with patch("llm_router.probe.probe_all_models", new=AsyncMock(
            return_value={"nvidia": {DEAD: {"available": True, "error": None, "latency_ms": 5}}}
        )), patch("llm_router.probe.seed_known_limits", new=AsyncMock()):
            stats = await run_probe_and_update()

        async with db.execute(
            "SELECT active, deactivated_reason, deactivated_at FROM models WHERE model_id = ?",
            (DEAD,),
        ) as cur:
            active, reason, at = await cur.fetchone()
        assert (active, reason, at) == (1, None, None)
        assert stats["reactivated"] == [f"nvidia:{DEAD}"]


class TestZenFreeDiscovery:
    """opencode.ai zen serves 60+ models from one endpoint and marks none of them
    free — /v1/models returns id, object, created, owned_by and nothing else. The
    free set was therefore hardcoded to two ids, one of which this repo's own
    adapter blocks, so the router knew one of zen's seven free models and
    recorded its context as 200k when it is 1M.
    """

    ZEN_SERVED = {"data": [{"id": i} for i in [
        "claude-fable-5", "gpt-5.5-pro", "nemotron-3-ultra-free",
        "deepseek-v4-flash-free", "big-pickle", "some-paid-model",
    ]]}
    MODELS_DEV = {"opencode": {"models": {
        "nemotron-3-ultra-free": {"cost": {"input": 0, "output": 0},
                                  "limit": {"context": 1_000_000}, "tool_call": True},
        "deepseek-v4-flash-free": {"cost": {"input": 0, "output": 0},
                                   "limit": {"context": 200_000}, "tool_call": True},
        "big-pickle": {"cost": {"input": 0, "output": 0},
                       "limit": {"context": 200_000}, "tool_call": True},
        "gpt-5.5-pro": {"cost": {"input": 30, "output": 180},
                        "limit": {"context": 400_000}, "tool_call": True},
    }}}

    def _client(self, monkeypatch, models_dev_ok=True):
        import httpx

        from llm_router import discovery as discovery_module

        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, **kw):
                if "models.dev" in url:
                    if not models_dev_ok:
                        raise httpx.ConnectError("models.dev unreachable")
                    return FakeResponse(TestZenFreeDiscovery.MODELS_DEV)
                return FakeResponse(TestZenFreeDiscovery.ZEN_SERVED)

        monkeypatch.setattr(discovery_module.httpx, "AsyncClient", FakeClient)

    @pytest.mark.asyncio
    async def test_it_finds_every_free_model_zen_actually_serves(self, monkeypatch):
        from llm_router.discovery import ModelDiscovery

        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        self._client(monkeypatch)

        models = await ModelDiscovery()._discover_opencode()

        assert {m["model_id"] for m in models} == {
            "nemotron-3-ultra-free", "deepseek-v4-flash-free", "big-pickle"}

    @pytest.mark.asyncio
    async def test_it_records_the_real_context_window(self, monkeypatch):
        """1M, not the 200k that was hardcoded for every zen model."""
        from llm_router.discovery import ModelDiscovery

        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        self._client(monkeypatch)

        models = {m["model_id"]: m for m in await ModelDiscovery()._discover_opencode()}
        assert models["nemotron-3-ultra-free"]["context_length"] == 1_000_000

    @pytest.mark.asyncio
    async def test_paid_models_never_enter_the_catalogue(self, monkeypatch):
        """This router is for free tiers; gpt-5.5-pro is $30/$180 per 1M."""
        from llm_router.discovery import ModelDiscovery

        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        self._client(monkeypatch)

        models = await ModelDiscovery()._discover_opencode()
        assert "gpt-5.5-pro" not in {m["model_id"] for m in models}

    @pytest.mark.asyncio
    async def test_it_degrades_to_name_matching_when_the_catalogue_is_down(
        self, monkeypatch
    ):
        """models.dev being unreachable must not empty the provider."""
        from llm_router.discovery import ModelDiscovery

        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        self._client(monkeypatch, models_dev_ok=False)

        found = {m["model_id"] for m in await ModelDiscovery()._discover_opencode()}
        assert found == {"nemotron-3-ultra-free", "deepseek-v4-flash-free", "big-pickle"}
        assert "gpt-5.5-pro" not in found
