"""Model health tracking — single source of truth for availability data."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from .db import get_db

logger = logging.getLogger(__name__)


class ModelHealthRepository:
    """Tracks per-model probe results, latency EMA, success/failure counts.

    Used by both the router (after every request) and the probe system
    (after every probe cycle). Single implementation eliminates the
    duplication that previously existed between router.py and probe.py.
    """

    async def update(
        self,
        provider_id: str,
        model_id: str,
        success: bool,
        latency_ms: float | None = None,
        error: str | None = None,
        now: str | None = None,
    ) -> None:
        """Upsert model_health row with result data.

        Args:
            provider_id: Provider slug (e.g., "openrouter")
            model_id: Model identifier
            success: Whether the request/probe succeeded
            latency_ms: Response latency (used for EMA)
            error: Error message if failed
            now: ISO timestamp (defaults to current UTC time)
        """
        db = await get_db()
        if now is None:
            now = datetime.now(timezone.utc).isoformat()

        # Fetch existing row for EMA calculation
        async with db.execute(
            "SELECT avg_latency_ms, consecutive_failures, success_count, failure_count "
            "FROM model_health WHERE provider_id = ? AND model_id = ?",
            (provider_id, model_id),
        ) as cur:
            row = await cur.fetchone()

        if row:
            old_avg, consec, suc, fail = row
        else:
            old_avg, consec, suc, fail = None, 0, 0, 0

        # EMA for latency
        if latency_ms is not None and latency_ms > 0:
            if old_avg is not None:
                avg_lat = old_avg * 0.7 + latency_ms * 0.3
            else:
                avg_lat = latency_ms
        else:
            avg_lat = old_avg

        if success:
            consec = 0
            suc = (suc or 0) + 1
        else:
            consec = (consec or 0) + 1
            fail = (fail or 0) + 1

        await db.execute(
            """INSERT INTO model_health
               (provider_id, model_id, last_probe_at, last_probe_ok,
                last_probe_latency_ms, last_probe_error,
                consecutive_failures, avg_latency_ms, success_count, failure_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(provider_id, model_id) DO UPDATE SET
               last_probe_at = excluded.last_probe_at,
               last_probe_ok = excluded.last_probe_ok,
               last_probe_latency_ms = excluded.last_probe_latency_ms,
               last_probe_error = excluded.last_probe_error,
               consecutive_failures = excluded.consecutive_failures,
               avg_latency_ms = excluded.avg_latency_ms,
               success_count = excluded.success_count,
               failure_count = excluded.failure_count""",
            (provider_id, model_id, now, 1 if success else 0,
             latency_ms, error, consec, avg_lat, suc, fail),
        )
        await db.commit()

    async def get_health(self, provider_id: str, model_id: str) -> dict | None:
        """Get health data for a specific model."""
        db = await get_db()
        async with db.execute(
            "SELECT * FROM model_health WHERE provider_id = ? AND model_id = ?",
            (provider_id, model_id),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return dict(row)

    async def get_all_health(self) -> list[dict]:
        """Get health data for all models."""
        db = await get_db()
        async with db.execute(
            "SELECT * FROM model_health ORDER BY success_count DESC"
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]
