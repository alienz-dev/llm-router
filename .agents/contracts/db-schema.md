# DB Schema Contract

## SQLite (WAL mode) — db.py

```sql
PRAGMA journal_mode=WAL;

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
    task_scores TEXT,  -- JSON: {"code": 0.9, "reasoning": 0.8, ...}
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
    timestamp TEXT NOT NULL,  -- ISO 8601
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    success INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT DEFAULT 'pending',  -- pending, running, completed, failed
    priority TEXT DEFAULT 'batch',  -- immediate, batch
    task_type TEXT,
    messages TEXT NOT NULL,  -- JSON
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
    response TEXT,  -- JSON
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms REAL
);

CREATE INDEX IF NOT EXISTS idx_quota_usage_ts ON quota_usage(provider_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_models_active ON models(active, provider_id);
```

## Access Pattern
- `db.py` exports `async get_db() -> aiosqlite.Connection` (singleton, WAL mode set on first connect)
- All queries use parameterized SQL
- Connection shared across FastAPI lifespan
