"""The free-tier budget has to be a real number before anything else means much.

Three defects made it fiction. `seed_known_limits` skipped every `_default` key,
so OpenRouter's 50 requests/day — the tightest constraint this router exists to
manage — was never written and `can_use` returned True for every request ever
made. `can_use` then counted usage per model, so an account-wide cap compared
against one model's count could never trip. And `get_all_quota_status` read a
`model_id = ''` row nothing wrote, so `quota_remaining_pct` was a constant 1.0
for every provider — which made the `headroom` term in the routing score inert.
"""
from datetime import datetime, timedelta, timezone

import pytest

from llm_router.db import _init_schema
from llm_router.probe import seed_known_limits
from llm_router.quota import QuotaManager

PROVIDERS = ("openrouter", "nvidia", "google")


@pytest.fixture
async def db(monkeypatch):
    import aiosqlite
    import llm_router.db as db_module

    conn = await aiosqlite.connect(":memory:")
    await _init_schema(conn)
    for pid in PROVIDERS:
        await conn.execute(
            "INSERT INTO providers (id, name, base_url, daily_reset_utc_hour) "
            "VALUES (?, ?, 'https://x', 0)", (pid, pid))
    await conn.commit()

    async def _get_db():
        return conn

    monkeypatch.setattr(db_module, "get_db", _get_db)
    for module in ("quota", "probe", "router", "health"):
        monkeypatch.setattr(f"llm_router.{module}.get_db", _get_db, raising=False)
    yield conn
    await conn.close()


async def spend(db, provider, model, n=1, tokens=0, success=True):
    now = datetime.now(timezone.utc).isoformat()
    await db.executemany(
        """INSERT INTO quota_usage (provider_id, model_id, timestamp,
                                    tokens_in, tokens_out, success)
           VALUES (?, ?, ?, ?, 0, ?)""",
        [(provider, model, now, tokens, 1 if success else 0) for _ in range(n)],
    )
    await db.commit()


class TestSeeding:
    @pytest.mark.asyncio
    async def test_the_account_wide_cap_is_written(self, db):
        await seed_known_limits()
        async with db.execute(
            "SELECT rpd FROM quota_limits WHERE provider_id = 'openrouter' AND model_id = ''"
        ) as cur:
            row = await cur.fetchone()
        assert row == (50,), "OpenRouter's 50/day cap never reached the database"

    @pytest.mark.asyncio
    async def test_hourly_and_daily_token_limits_survive_the_insert(self, db):
        """The INSERT only listed rpm/rpd/tpm, so rph and tpd were dropped."""
        await seed_known_limits()
        async with db.execute(
            "SELECT rph FROM quota_limits WHERE provider_id = 'huggingface' AND model_id = ''"
        ) as cur:
            assert (await cur.fetchone()) == (100,)

    @pytest.mark.asyncio
    async def test_per_model_limits_still_land_per_model(self, db):
        await seed_known_limits()
        async with db.execute(
            "SELECT rpm, rpd FROM quota_limits "
            "WHERE provider_id = 'google' AND model_id = 'gemini-3.1-pro-preview'"
        ) as cur:
            assert (await cur.fetchone()) == (5, 50)

    @pytest.mark.asyncio
    async def test_seeding_twice_changes_nothing(self, db):
        await seed_known_limits()
        await seed_known_limits()
        async with db.execute("SELECT COUNT(*) FROM quota_limits") as cur:
            first = (await cur.fetchone())[0]
        await seed_known_limits()
        async with db.execute("SELECT COUNT(*) FROM quota_limits") as cur:
            assert (await cur.fetchone())[0] == first


class TestAccountWideCounting:
    @pytest.mark.asyncio
    async def test_the_cap_counts_every_model_on_the_key(self, db):
        """The bug: 50/day spread over 20 models read as 2-3 per model, so the
        router happily spent 20x its budget."""
        await seed_known_limits()
        qm = QuotaManager()
        for i in range(49):
            await spend(db, "openrouter", f"model-{i % 10}")
        assert await qm.can_use("openrouter", "model-0", 100) is True

        await spend(db, "openrouter", "model-7")  # the 50th
        assert await QuotaManager().can_use("openrouter", "model-0", 100) is False

    @pytest.mark.asyncio
    async def test_another_providers_traffic_is_not_charged_to_this_one(self, db):
        await seed_known_limits()
        await spend(db, "nvidia", "some-model", n=200)
        assert await QuotaManager().can_use("openrouter", "model-0", 100) is True

    @pytest.mark.asyncio
    async def test_failed_requests_do_not_consume_the_budget(self, db):
        await seed_known_limits()
        await spend(db, "openrouter", "model-0", n=60, success=False)
        assert await QuotaManager().can_use("openrouter", "model-0", 100) is True

    @pytest.mark.asyncio
    async def test_a_per_model_cap_still_binds_that_model_alone(self, db):
        await seed_known_limits()
        await spend(db, "google", "gemini-3.1-pro-preview", n=50)
        qm = QuotaManager()
        assert await qm.can_use("google", "gemini-3.1-pro-preview", 10) is False
        assert await qm.can_use("google", "gemma-3-27b-it", 10) is True

    @pytest.mark.asyncio
    async def test_a_provider_with_no_known_limit_is_allowed(self, db):
        """Unknown is not zero — nvidia publishes nothing we have recorded."""
        await seed_known_limits()
        await spend(db, "nvidia", "mistralai/mistral-nemotron", n=500)
        assert await QuotaManager().can_use("nvidia", "mistralai/mistral-nemotron", 10) is True

    @pytest.mark.asyncio
    async def test_token_limits_look_ahead_at_the_completion(self, db):
        await seed_known_limits()
        await spend(db, "mistral", "any", tokens=499_000)
        # 499k used, 500k/min cap, and the estimate adds 1024 for the reply.
        assert await QuotaManager().can_use("mistral", "any", 2000) is False


class TestReportedHeadroom:
    @pytest.mark.asyncio
    async def test_headroom_moves_with_real_usage(self, db):
        """This is the number the routing score multiplies by. It was a constant."""
        await seed_known_limits()
        await spend(db, "openrouter", "model-1", n=25)
        status = {s["provider_id"]: s for s in await QuotaManager().get_all_quota_status()}

        assert status["openrouter"]["rpd_used"] == 25
        assert status["openrouter"]["rpd_limit"] == 50
        assert status["openrouter"]["quota_remaining_pct"] == pytest.approx(0.5)
        assert status["openrouter"]["limits_known"] is True

    @pytest.mark.asyncio
    async def test_unknown_limits_are_reported_as_unknown(self, db):
        """A full-looking bar must be readable as "we have no idea", not "plenty"."""
        await seed_known_limits()
        status = {s["provider_id"]: s for s in await QuotaManager().get_all_quota_status()}
        assert status["nvidia"]["limits_known"] is False
        assert status["nvidia"]["quota_remaining_pct"] == 1.0

    @pytest.mark.asyncio
    async def test_an_exhausted_provider_reports_unhealthy(self, db):
        await seed_known_limits()
        await spend(db, "openrouter", "model-1", n=50)
        status = {s["provider_id"]: s for s in await QuotaManager().get_all_quota_status()}
        assert status["openrouter"]["quota_remaining_pct"] == 0
        assert status["openrouter"]["healthy"] is False
