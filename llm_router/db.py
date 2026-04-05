import aiosqlite
from pathlib import Path

_db_connection: aiosqlite.Connection | None = None


async def get_db() -> aiosqlite.Connection:
    global _db_connection
    if _db_connection is None:
        db_path = Path("llm_router.db")
        _db_connection = await aiosqlite.connect(db_path)
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

        CREATE INDEX IF NOT EXISTS idx_quota_usage_ts ON quota_usage(provider_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, priority, created_at);
        CREATE INDEX IF NOT EXISTS idx_models_active ON models(active, provider_id);
    """)
    await db.commit()
