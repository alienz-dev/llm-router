import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta

from .db import get_db
from .adapters.base import QuotaSnapshot

logger = logging.getLogger(__name__)


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

    async def _window_starts(self, provider: str) -> tuple[str, str, str]:
        """(minute, hour, day) window starts as ISO strings, honouring the
        provider's own daily reset hour."""
        db = await get_db()
        now = datetime.now(timezone.utc)
        async with db.execute(
            "SELECT daily_reset_utc_hour FROM providers WHERE id = ?", (provider,)
        ) as cursor:
            row = await cursor.fetchone()
        reset_hour = row[0] if row else 0

        daily_start = now.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
        if now.hour < reset_hour:
            daily_start -= timedelta(days=1)

        return (
            (now - timedelta(minutes=1)).isoformat(),
            (now - timedelta(hours=1)).isoformat(),
            daily_start.isoformat(),
        )

    async def _usage(
        self, provider: str, model_id: str | None, minute: str, hour: str, day: str
    ) -> dict[str, int]:
        """Successful usage in each window. `model_id=None` counts the whole
        account, which is what an account-wide cap has to be measured against."""
        db = await get_db()
        scope = "" if model_id is None else " AND model_id = ?"
        args: list = [minute, minute, hour, hour, day, day, provider]
        if model_id is not None:
            args.append(model_id)
        args.append(day)

        async with db.execute(
            f"""
            SELECT
                SUM(CASE WHEN timestamp >= ? THEN 1 ELSE 0 END),
                SUM(CASE WHEN timestamp >= ? THEN tokens_in + tokens_out ELSE 0 END),
                SUM(CASE WHEN timestamp >= ? THEN 1 ELSE 0 END),
                SUM(CASE WHEN timestamp >= ? THEN tokens_in + tokens_out ELSE 0 END),
                SUM(CASE WHEN timestamp >= ? THEN 1 ELSE 0 END),
                SUM(CASE WHEN timestamp >= ? THEN tokens_in + tokens_out ELSE 0 END)
            FROM quota_usage
            WHERE provider_id = ?{scope} AND success = 1 AND timestamp >= ?
            """,
            args,
        ) as cur:
            row = await cur.fetchone()

        keys = ("rpm", "tpm", "rph", "tph", "rpd", "tpd")
        return {k: (row[i] or 0) if row else 0 for i, k in enumerate(keys)}

    async def can_use(
        self, provider: str, model_id: str, estimated_tokens: int
    ) -> bool:
        """Would this request stay inside every known limit?

        Two scopes, and the difference is the whole point. A limit stored against
        `model_id = ''` is an ACCOUNT limit — OpenRouter's 50 requests/day covers
        every model on the key — so it has to be counted across every model.
        Counting it per model, which is what this did, turned a 50/day budget
        into 50-per-model: with 20 active models, a 20x overrun before anything
        said no.
        """
        minute, hour, day = await self._window_starts(provider)
        total_estimated = estimated_tokens + 1024  # expected completion

        # Account-wide first: it is the tighter and more consequential of the two.
        scopes: list[tuple[str, str | None]] = [("", None)]
        if model_id:
            scopes.append((model_id, model_id))

        for limits_key, usage_scope in scopes:
            limits = await self._get_limits(provider, limits_key)
            if not limits:
                continue
            used = await self._usage(provider, usage_scope, minute, hour, day)
            for key in ("rpm", "rph", "rpd"):
                if limits.get(key) and used[key] >= limits[key]:
                    logger.info(
                        "Quota blocked %s%s: %s %d/%d",
                        provider, f":{model_id}" if usage_scope else " (account)",
                        key, used[key], limits[key],
                    )
                    return False
            for key in ("tpm", "tph", "tpd"):
                if limits.get(key) and used[key] + total_estimated > limits[key]:
                    logger.info(
                        "Quota blocked %s%s: %s %d+%d/%d",
                        provider, f":{model_id}" if usage_scope else " (account)",
                        key, used[key], total_estimated, limits[key],
                    )
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
        """Per-provider account-wide usage against account-wide limits.

        `quota_remaining_pct` feeds the `headroom` term in the router's score.
        Until the account-wide rows existed it was always 1.0 for every provider,
        which made that whole term a constant — the formula looked like it
        weighed quota and did not. `limits_known` says which providers still
        have no published cap, so a full-looking bar can be read as "unknown"
        rather than "plenty".
        """
        db = await get_db()
        async with db.execute(
            "SELECT id FROM providers WHERE enabled = 1"
        ) as cursor:
            providers = [row[0] for row in await cursor.fetchall()]

        results = []
        for provider_id in providers:
            minute, hour, day = await self._window_starts(provider_id)
            limits = await self._get_limits(provider_id, "")
            used = await self._usage(provider_id, None, minute, hour, day)

            pcts = [
                max(0.0, (limits[key] - used[key]) / limits[key])
                for key in ("rpm", "rph", "rpd", "tpm", "tph", "tpd")
                if limits.get(key)
            ]
            remaining = min(pcts) if pcts else 1.0

            results.append({
                "provider_id": provider_id,
                "rpm_used": used["rpm"], "rpm_limit": limits.get("rpm"),
                "rph_used": used["rph"], "rph_limit": limits.get("rph"),
                "rpd_used": used["rpd"], "rpd_limit": limits.get("rpd"),
                "tpm_used": used["tpm"], "tpm_limit": limits.get("tpm"),
                "tph_used": used["tph"], "tph_limit": limits.get("tph"),
                "tpd_used": used["tpd"], "tpd_limit": limits.get("tpd"),
                "limits_known": bool(pcts),
                "quota_remaining_pct": remaining,
                "healthy": remaining > 0,
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
