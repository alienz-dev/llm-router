"""Schema migrations must work on databases that already exist.

The failure this guards against: adding a column to the CREATE TABLE block in
db.py and shipping it. Every existing database only ever sees
CREATE TABLE IF NOT EXISTS, so the column never appears and the first UPDATE
against it raises 'no such column'.
"""
import aiosqlite
import pytest

from llm_router.db import COLUMN_MIGRATIONS, _columns, _init_schema, apply_migrations

# The models table exactly as it shipped before any capability work.
LEGACY_MODELS_DDL = """
    CREATE TABLE models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        display_name TEXT,
        context_length INTEGER,
        task_scores TEXT,
        is_free INTEGER DEFAULT 1,
        discovered_at TEXT,
        active INTEGER DEFAULT 1,
        UNIQUE(provider_id, model_id)
    );
"""

NEW_COLUMNS = [column for _, table, column, _ in COLUMN_MIGRATIONS if table == "models"]


@pytest.mark.asyncio
async def test_migrates_a_database_built_from_the_old_schema():
    db = await aiosqlite.connect(":memory:")
    try:
        await db.executescript(LEGACY_MODELS_DDL)
        await db.execute(
            "INSERT INTO models (provider_id, model_id, active) VALUES (?, ?, 1)",
            ("openrouter", "nvidia/nemotron-3-nano-30b-a3b:free"),
        )
        await db.commit()

        # Startup path: CREATE TABLE IF NOT EXISTS is a no-op here, migrations are not.
        await _init_schema(db)

        columns = await _columns(db, "models")
        for column in NEW_COLUMNS:
            assert column in columns, f"{column} missing after migration"

        # The pre-existing row survives, and reads NULL (unknown) for the new flags.
        async with db.execute(
            "SELECT supports_tools, capability_checked_at FROM models WHERE model_id = ?",
            ("nvidia/nemotron-3-nano-30b-a3b:free",),
        ) as cur:
            row = await cur.fetchone()
        assert row == (None, None)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_migrations_are_idempotent():
    db = await aiosqlite.connect(":memory:")
    try:
        await _init_schema(db)
        first = await apply_migrations(db)
        second = await apply_migrations(db)
        assert first == [], "init_schema should have applied everything already"
        assert second == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fresh_database_records_every_migration():
    db = await aiosqlite.connect(":memory:")
    try:
        await _init_schema(db)
        async with db.execute("SELECT name FROM schema_migrations") as cur:
            recorded = {row[0] for row in await cur.fetchall()}
        assert recorded == {name for name, _, _, _ in COLUMN_MIGRATIONS}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_db_path_honours_env_override(monkeypatch):
    from llm_router.db import db_path

    monkeypatch.setenv("DATABASE_PATH", "/tmp/scratch-router.db")
    assert db_path() == "/tmp/scratch-router.db"
    monkeypatch.setenv("DATABASE_PATH", "")
    assert db_path() != ""  # falls back to config, never empty
