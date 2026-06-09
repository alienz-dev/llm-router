import asyncio
import json
from datetime import datetime, timezone, timedelta

from .db import get_db
from .adapters.base import QuotaSnapshot


class QuotaManager:
    def __init__(self):
        self._cache_lock = asyncio.Lock()
        self._limits_cache: dict[str, dict] = {}
        self._cache_expires: dict[str, datetime] = {}

    async def estimate_tokens(self, messages: list[dict]) -> int:
        """len(text) / 4 heuristic across all message content."""
        total_text = ""
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                total_text += content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total_text += item.get("text", "")
        return max(1, len(total_text) // 4)

    async def can_use(
        self, provider: str, model_id: str, estimated_tokens: int
    ) -> bool:
        """Check RPM + TPM against sliding window usage AND known limits.
        Uses a single query for all usage data instead of 4 separate queries."""
        db = await get_db()
        now = datetime.now(timezone.utc)
        limits = await self._get_limits(provider, model_id)
        if not limits:
            return True

        # Get daily reset hour
        async with db.execute(
            "SELECT daily_reset_utc_hour FROM providers WHERE id = ?", (provider,)
        ) as cursor:
            row = await cursor.fetchone()
        reset_hour = row[0] if row else 0

        daily_start = now.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
        if now.hour < reset_hour:
            daily_start -= timedelta(days=1)

        minute_ago = (now - timedelta(minutes=1)).isoformat()
        daily_iso = daily_start.isoformat()
        total_estimated = estimated_tokens + 1024  # expected completion

        # Single query: get minute + daily usage in one shot
        usage_q = """
            SELECT
                SUM(CASE WHEN timestamp >= ? THEN 1 ELSE 0 END) AS rpm_count,
                SUM(CASE WHEN timestamp >= ? THEN tokens_in + tokens_out ELSE 0 END) AS tpm_used,
                SUM(CASE WHEN timestamp >= ? THEN 1 ELSE 0 END) AS rpd_count,
                SUM(CASE WHEN timestamp >= ? THEN tokens_in + tokens_out ELSE 0 END) AS tpd_used
            FROM quota_usage
            WHERE provider_id = ? AND model_id = ? AND success = 1
              AND timestamp >= ?
        """
        async with db.execute(
            usage_q,
            (minute_ago, minute_ago, daily_iso, daily_iso,
             provider, model_id, daily_iso),
        ) as cur:
            row = await cur.fetchone()

        if not row:
            return True

        rpm_count, tpm_used, rpd_count, tpd_used = (
            row[0] or 0, row[1] or 0, row[2] or 0, row[3] or 0
        )

        if limits.get("rpm") and rpm_count >= limits["rpm"]:
            return False
        if limits.get("tpm") and tpm_used + total_estimated > limits["tpm"]:
            return False
        if limits.get("rpd") and rpd_count >= limits["rpd"]:
            return False
        if limits.get("tpd") and tpd_used + total_estimated > limits["tpd"]:
            return False

        return True

    async def record_usage(
        self, provider: str, model_id: str,
        tokens_in: int, tokens_out: int, success: bool
    ) -> None:
        db = await get_db()
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            """INSERT INTO quota_usage
               (provider_id, model_id, timestamp, tokens_in, tokens_out, success)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (provider, model_id, now, tokens_in, tokens_out, 1 if success else 0),
        )
        await db.commit()

    async def update_limits(
        self, provider: str, model_id: str, snapshot: QuotaSnapshot
    ) -> None:
        if snapshot is None:
            return
        updates = {}
        if snapshot.rpm_limit is not None:
            updates["rpm"] = snapshot.rpm_limit
        if snapshot.rpd_limit is not None:
            updates["rpd"] = snapshot.rpd_limit
        if snapshot.tpm_limit is not None:
            updates["tpm"] = snapshot.tpm_limit
        if snapshot.tpd_limit is not None:
            updates["tpd"] = snapshot.tpd_limit
        if not updates:
            return

        db = await get_db()
        await db.execute(
            """INSERT INTO quota_limits (provider_id, model_id, rpm, rpd, tpm, tpd)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(provider_id, model_id) DO UPDATE SET
               rpm = COALESCE(excluded.rpm, rpm),
               rpd = COALESCE(excluded.rpd, rpd),
               tpm = COALESCE(excluded.tpm, tpm),
               tpd = COALESCE(excluded.tpd, tpd)""",
            (provider, model_id,
             updates.get("rpm"), updates.get("rpd"),
             updates.get("tpm"), updates.get("tpd")),
        )
        await db.commit()

        async with self._cache_lock:
            cache_key = f"{provider}:{model_id}"
            self._limits_cache.pop(cache_key, None)
            self._cache_expires.pop(cache_key, None)

    async def get_all_quota_status(self) -> list[dict]:
        db = await get_db()
        now = datetime.now(timezone.utc)

        async with db.execute(
            "SELECT id, daily_reset_utc_hour FROM providers WHERE enabled = 1"
        ) as cursor:
            providers = await cursor.fetchall()

        results = []
        for provider_id, reset_hour in providers:
            minute_ago = (now - timedelta(minutes=1)).isoformat()
            daily_start = now.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
            if now.hour < reset_hour:
                daily_start -= timedelta(days=1)
            daily_iso = daily_start.isoformat()

            limits = await self._get_limits(provider_id, "")

            usage_q = """
                SELECT COUNT(*), COALESCE(SUM(tokens_in + tokens_out), 0)
                FROM quota_usage
                WHERE provider_id = ? AND timestamp >= ? AND success = 1
            """

            async with db.execute(usage_q, (provider_id, minute_ago)) as cur:
                row = await cur.fetchone()
                rpm_used, tpm_used = (row[0], row[1]) if row else (0, 0)

            async with db.execute(usage_q, (provider_id, daily_iso)) as cur:
                row = await cur.fetchone()
                rpd_used, tpd_used = (row[0], row[1]) if row else (0, 0)

            pcts = []
            if limits.get("rpm"):
                pcts.append(max(0, (limits["rpm"] - rpm_used) / limits["rpm"]))
            if limits.get("tpm"):
                pcts.append(max(0, (limits["tpm"] - tpm_used) / limits["tpm"]))
            if limits.get("rpd"):
                pcts.append(max(0, (limits["rpd"] - rpd_used) / limits["rpd"]))
            if limits.get("tpd"):
                pcts.append(max(0, (limits["tpd"] - tpd_used) / limits["tpd"]))

            results.append({
                "provider_id": provider_id,
                "rpm_used": rpm_used,
                "rpm_limit": limits.get("rpm"),
                "tpm_used": tpm_used,
                "tpm_limit": limits.get("tpm"),
                "rpd_used": rpd_used,
                "rpd_limit": limits.get("rpd"),
                "tpd_used": tpd_used,
                "tpd_limit": limits.get("tpd"),
                "quota_remaining_pct": min(pcts) if pcts else 1.0,
                "healthy": (min(pcts) if pcts else 1.0) > 0,
            })

        return results

    async def _get_limits(self, provider: str, model_id: str) -> dict:
        cache_key = f"{provider}:{model_id}"
        async with self._cache_lock:
            if (cache_key in self._limits_cache
                    and datetime.now(timezone.utc) < self._cache_expires.get(cache_key, datetime.min.replace(tzinfo=timezone.utc))):
                return self._limits_cache[cache_key]

        db = await get_db()
        async with db.execute(
            "SELECT rpm, rph, rpd, tpm, tph, tpd FROM quota_limits WHERE provider_id = ? AND model_id = ?",
            (provider, model_id),
        ) as cursor:
            row = await cursor.fetchone()

        limits = {}
        if row:
            for i, key in enumerate(["rpm", "rph", "rpd", "tpm", "tph", "tpd"]):
                if row[i] is not None:
                    limits[key] = row[i]

        async with self._cache_lock:
            self._limits_cache[cache_key] = limits
            self._cache_expires[cache_key] = datetime.now(timezone.utc) + timedelta(minutes=5)

        return limits
