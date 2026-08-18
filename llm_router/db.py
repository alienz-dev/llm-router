import logging
import os

import aiosqlite
from pathlib import Path

logger = logging.getLogger(__name__)

_db_connection: aiosqlite.Connection | None = None


def db_path() -> str:
    """Resolve the database path.

    DATABASE_PATH wins so tests (and one-off tooling) can point at ':memory:'
    or a scratch file without touching the dev database.
    """
    env_path = os.getenv("DATABASE_PATH", "")
    if env_path:
        return env_path
    from .config import get_config
    return get_config().database.path


async def get_db() -> aiosqlite.Connection:
    global _db_connection
    if _db_connection is None:
        path = db_path()
        _db_connection = await aiosqlite.connect(
            path if path == ":memory:" else Path(path)
        )
        if path != ":memory:":
            await _db_connection.execute("PRAGMA journal_mode=WAL")
        await _init_schema(_db_connection)
    return _db_connection


async def close_db():
    global _db_connection
    if _db_connection is not None:
        await _db_connection.close()
        _db_connection = None


async def _init_schema(db: aiosqlite.Connection):
    await db.executescript("""
        CREATE TABLE IF NOT EXISTS providers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            base_url TEXT NOT NULL,
            enabled INTEGER DEFAULT 1,
            daily_reset_utc_hour INTEGER DEFAULT 0,
            credits_remaining REAL
        );

        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider_id TEXT NOT NULL REFERENCES providers(id),
            model_id TEXT NOT NULL,
            display_name TEXT,
            context_length INTEGER,
            task_scores TEXT,
            is_free INTEGER DEFAULT 1,
            discovered_at TEXT,
            active INTEGER DEFAULT 1,
            UNIQUE(provider_id, model_id)
        );

        CREATE TABLE IF NOT EXISTS quota_limits (
            provider_id TEXT NOT NULL,
            model_id TEXT NOT NULL DEFAULT '',
            rpm INTEGER, rph INTEGER, rpd INTEGER,
            tpm INTEGER, tph INTEGER, tpd INTEGER,
            PRIMARY KEY (provider_id, model_id)
        );

        CREATE TABLE IF NOT EXISTS quota_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            tokens_in INTEGER DEFAULT 0,
            tokens_out INTEGER DEFAULT 0,
            success INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'pending',
            priority TEXT DEFAULT 'batch',
            task_type TEXT,
            messages TEXT NOT NULL,
            model_override TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            retries INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS job_results (
            job_id TEXT PRIMARY KEY REFERENCES jobs(id),
            provider_id TEXT,
            model_id TEXT,
            response TEXT,
            tokens_in INTEGER,
            tokens_out INTEGER,
            latency_ms REAL
        );

        CREATE TABLE IF NOT EXISTS model_health (
            provider_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            last_probe_at TEXT,
            last_probe_ok INTEGER DEFAULT 0,
            last_probe_latency_ms REAL,
            last_probe_error TEXT,
            consecutive_failures INTEGER DEFAULT 0,
            avg_latency_ms REAL,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            PRIMARY KEY (provider_id, model_id)
        );

        CREATE TABLE IF NOT EXISTS circuit_breaker_state (
            provider_id TEXT PRIMARY KEY,
            state TEXT NOT NULL DEFAULT 'closed',
            failure_count INTEGER DEFAULT 0,
            last_failure_at TEXT,
            last_error TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_quota_usage_ts ON quota_usage(provider_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, priority, created_at);
        CREATE INDEX IF NOT EXISTS idx_models_active ON models(active, provider_id);
    """)
    await db.commit()
    await apply_migrations(db)


# ── Migrations ──────────────────────────────────────────
# Columns added after the original schema shipped. Never edit the base DDL above
# for a new column — an existing database (agent-mini's, anyone's dev copy) only
# ever sees CREATE TABLE IF NOT EXISTS, so it would never gain the column.
# Each entry is (migration_name, table, column, column_declaration).
COLUMN_MIGRATIONS: list[tuple[str, str, str, str]] = [
    # Why a model was switched off, so a transient 404 does not become permanent.
    ("001_models_deactivated_reason", "models", "deactivated_reason", "TEXT"),
    ("002_models_deactivated_at", "models", "deactivated_at", "TEXT"),
    # Capability flags. NULL means unknown — never treat NULL as unsupported.
    ("003_models_supports_tools", "models", "supports_tools", "INTEGER"),
    ("004_models_supports_json_schema", "models", "supports_json_schema", "INTEGER"),
    ("005_models_supports_vision", "models", "supports_vision", "INTEGER"),
    ("006_models_capability_checked_at", "models", "capability_checked_at", "TEXT"),
]


async def _columns(db: aiosqlite.Connection, table: str) -> set[str]:
    async with db.execute(f"PRAGMA table_info({table})") as cur:
        return {row[1] for row in await cur.fetchall()}


async def apply_migrations(db: aiosqlite.Connection) -> list[str]:
    """Bring any database — fresh or years old — up to the current schema.

    Idempotent: a column that already exists is recorded and skipped, so this is
    safe to run on every startup.
    """
    from datetime import datetime, timezone

    async with db.execute("SELECT name FROM schema_migrations") as cur:
        applied = {row[0] for row in await cur.fetchall()}

    newly_applied = []
    table_columns: dict[str, set[str]] = {}
    for name, table, column, decl in COLUMN_MIGRATIONS:
        if name in applied:
            continue
        if table not in table_columns:
            table_columns[table] = await _columns(db, table)
        if column not in table_columns[table]:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            table_columns[table].add(column)
            newly_applied.append(name)
        await db.execute(
            "INSERT OR IGNORE INTO schema_migrations (name, applied_at) VALUES (?, ?)",
            (name, datetime.now(timezone.utc).isoformat()),
        )
    await db.commit()
    if newly_applied:
        logger.info("Applied %d schema migrations: %s", len(newly_applied), ", ".join(newly_applied))
    return newly_applied
