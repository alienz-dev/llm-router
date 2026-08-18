"""Capability flags: where they come from, and what NULL means.

The columns are worth nothing if they are filled wrong. Two failure modes matter
more than the rest: treating NULL as "unsupported" (which empties the candidate
set the day the migration lands), and letting a failed probe overwrite a good
reading with a guess.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from llm_router.capabilities import (
    CAPABILITY_TTL_DAYS, flags_from_verdicts, models_due_for_probe,
    record_capability, run_capability_probe, seed_from_inventory,
)
from llm_router.db import _init_schema

INVENTORY = [
    {"provider": "openrouter", "model": "nvidia/nemotron-3-nano-30b-a3b:free",
     "tools": "native", "json": "schema-ok"},
    {"provider": "deepseek", "model": "deepseek-v4-pro",
     "tools": "native", "json": "HTTP400"},
    {"provider": "openrouter", "model": "nvidia/nemotron-nano-9b-v2:free",
     "tools": "ignored", "json": "not-json"},
    {"provider": "openrouter", "model": "dead/model:free",
     "tools": "HTTP404", "json": "HTTP404"},
]


@pytest.fixture
async def db(monkeypatch, tmp_path):
    import aiosqlite
    import llm_router.db as db_module

    conn = await aiosqlite.connect(":memory:")
    await _init_schema(conn)
    for pid in ("openrouter", "deepseek"):
        await conn.execute(
            "INSERT INTO providers (id, name, base_url) VALUES (?, ?, 'https://x')",
            (pid, pid),
        )
    for record in INVENTORY:
        await conn.execute(
            """INSERT INTO models (provider_id, model_id, display_name, context_length,
                                   task_scores, discovered_at, active)
               VALUES (?, ?, ?, 8192, '{}', '2026-08-01', 1)""",
            (record["provider"], record["model"], record["model"]),
        )
    await conn.commit()

    async def _get_db():
        return conn

    monkeypatch.setattr(db_module, "get_db", _get_db)
    monkeypatch.setattr("llm_router.capabilities.get_db", _get_db)
    yield conn
    await conn.close()


@pytest.fixture
def inventory_file(tmp_path):
    path = tmp_path / "model-capabilities-2026-08-18.json"
    path.write_text(json.dumps(INVENTORY))
    return path


async def flags(db, model_id):
    async with db.execute(
        "SELECT supports_tools, supports_json_schema, capability_checked_at "
        "FROM models WHERE model_id = ?", (model_id,)
    ) as cur:
        return await cur.fetchone()


class TestVerdictMapping:
    def test_a_clean_tool_call_and_schema_are_yes(self):
        assert flags_from_verdicts("native", "schema-ok") == {
            "supports_tools": 1, "supports_json_schema": 1}

    def test_ignoring_tools_and_ignoring_the_schema_are_no(self):
        assert flags_from_verdicts("ignored", "json-off-schema") == {
            "supports_tools": 0, "supports_json_schema": 0}

    def test_a_tool_call_without_an_id_is_unusable(self):
        """No id means no way to match the result back on the next turn."""
        assert flags_from_verdicts("native-noid", "not-json")["supports_tools"] == 0

    def test_a_dead_model_teaches_us_nothing(self):
        """A 404 is about the model existing, not about what it can do."""
        assert flags_from_verdicts("HTTP404", "HTTP429") == {
            "supports_tools": None, "supports_json_schema": None}


class TestSeeding:
    @pytest.mark.asyncio
    async def test_the_known_agent_ready_models_get_their_flags(self, db, inventory_file):
        seeded = await seed_from_inventory(inventory_file)

        assert seeded == 3  # the 404 row teaches nothing, so nothing is written
        assert (await flags(db, "nvidia/nemotron-3-nano-30b-a3b:free"))[:2] == (1, 1)
        assert (await flags(db, "deepseek-v4-pro"))[:2] == (1, 0)
        assert (await flags(db, "nvidia/nemotron-nano-9b-v2:free"))[:2] == (0, 0)

    @pytest.mark.asyncio
    async def test_an_unknown_verdict_leaves_the_row_alone(self, db, inventory_file):
        await seed_from_inventory(inventory_file)
        assert (await flags(db, "dead/model:free")) == (None, None, None)

    @pytest.mark.asyncio
    async def test_the_reading_is_dated_by_the_file_not_by_now(self, db, inventory_file):
        """A two-month-old inventory should look stale to the re-probe."""
        await seed_from_inventory(inventory_file)
        checked = (await flags(db, "deepseek-v4-pro"))[2]
        assert checked.startswith("2026-08-18")

    @pytest.mark.asyncio
    async def test_seeding_twice_does_not_overwrite_a_live_result(self, db, inventory_file):
        await record_capability("deepseek", "deepseek-v4-pro",
                                {"supports_json_schema": 1})
        await seed_from_inventory(inventory_file)
        assert (await flags(db, "deepseek-v4-pro"))[1] == 1

    @pytest.mark.asyncio
    async def test_a_missing_inventory_is_not_an_error(self, db, tmp_path):
        assert await seed_from_inventory(tmp_path / "nope.json") == 0


class TestFreshness:
    @pytest.mark.asyncio
    async def test_never_checked_models_are_due(self, db):
        due = await models_due_for_probe()
        assert len(due) == 4

    @pytest.mark.asyncio
    async def test_a_fresh_reading_is_skipped(self, db):
        for record in INVENTORY:
            await record_capability(record["provider"], record["model"],
                                    {"supports_tools": 1})
        assert await models_due_for_probe() == []

    @pytest.mark.asyncio
    async def test_a_stale_reading_comes_back_up(self, db):
        stale = (datetime.now(timezone.utc)
                 - timedelta(days=CAPABILITY_TTL_DAYS + 1)).isoformat()
        await record_capability("deepseek", "deepseek-v4-pro",
                                {"supports_tools": 1}, checked_at=stale)
        for model in ("nvidia/nemotron-3-nano-30b-a3b:free",
                      "nvidia/nemotron-nano-9b-v2:free", "dead/model:free"):
            await record_capability("openrouter", model, {"supports_tools": 1})

        assert await models_due_for_probe() == [("deepseek", "deepseek-v4-pro")]

    @pytest.mark.asyncio
    async def test_the_per_provider_budget_is_respected(self, db):
        due = await models_due_for_probe(budget_per_provider=1)
        providers = [p for p, _ in due]
        assert providers.count("openrouter") == 1

    @pytest.mark.asyncio
    async def test_a_run_inside_the_window_makes_zero_provider_calls(self, db, monkeypatch):
        """The acceptance criterion: freshness is what keeps this affordable."""
        calls = []

        async def _never(*args, **kwargs):
            calls.append(args)
            return {}

        monkeypatch.setattr("llm_router.capabilities.probe_one", _never)
        for record in INVENTORY:
            await record_capability(record["provider"], record["model"],
                                    {"supports_tools": 1})

        stats = await run_capability_probe()

        assert calls == []
        assert stats == {"probed": 0, "updated": 0, "results": []}


class TestFailedProbesDoNotErase:
    @pytest.mark.asyncio
    async def test_a_404_does_not_wipe_a_known_capability(self, db):
        await record_capability("deepseek", "deepseek-v4-pro",
                                {"supports_tools": 1, "supports_json_schema": 1})
        await record_capability("deepseek", "deepseek-v4-pro",
                                flags_from_verdicts("HTTP404", "HTTP404"))
        assert (await flags(db, "deepseek-v4-pro"))[:2] == (1, 1)


AGENT_READY = [
    ("openrouter", "nvidia/nemotron-nano-12b-v2-vl:free"),
    ("agnes", "agnes-2.0-flash"),
    ("openrouter", "nvidia/nemotron-3-nano-30b-a3b:free"),
    ("openrouter", "google/gemma-4-26b-a4b-it:free"),
    ("agnes", "agnes-2.5-flash"),
    ("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free"),
    ("openrouter", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"),
    ("opencode", "nemotron-3-ultra-free"),
    ("nvidia", "mistralai/mistral-nemotron"),
]


class TestAgainstTheRealInventory:
    """The checked-in file is what a fresh database is seeded from, so the two
    have to agree — including on a database that predates the columns."""

    @pytest.mark.asyncio
    async def test_an_old_database_migrates_and_learns_the_agent_ready_nine(
        self, monkeypatch
    ):
        import aiosqlite
        import llm_router.db as db_module
        from llm_router.capabilities import DEFAULT_INVENTORY
        from tests.test_migrations import LEGACY_MODELS_DDL

        conn = await aiosqlite.connect(":memory:")
        try:
            await conn.executescript(LEGACY_MODELS_DDL)
            for provider, model in AGENT_READY:
                await conn.execute(
                    "INSERT INTO models (provider_id, model_id, active) VALUES (?, ?, 1)",
                    (provider, model),
                )
            await conn.commit()
            await _init_schema(conn)  # the startup path: migrations run here

            async def _get_db():
                return conn

            monkeypatch.setattr(db_module, "get_db", _get_db)
            monkeypatch.setattr("llm_router.capabilities.get_db", _get_db)
            await seed_from_inventory(DEFAULT_INVENTORY)

            async with conn.execute(
                "SELECT model_id, supports_tools, supports_json_schema FROM models"
            ) as cur:
                rows = await cur.fetchall()
            assert len(rows) == len(AGENT_READY)
            for model_id, tools, schema in rows:
                assert tools == 1, f"{model_id} should be tool-capable"
                assert schema == 1, f"{model_id} should honour a strict schema"
        finally:
            await conn.close()
