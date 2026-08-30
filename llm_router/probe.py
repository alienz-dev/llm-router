"""Probe discovered models to verify they're actually available and free."""
import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

import httpx

from .db import get_db
from .redact import redact, redact_error
from .health import ModelHealthRepository
from .quota import QuotaManager

logger = logging.getLogger(__name__)

# A model the probe switched off is re-checked this many days later. Without a
# re-check a transient 404 is permanent: probe_all_models only looks at active
# rows, and discovery no longer flips them back on.
DEACTIVATED_RECHECK_DAYS = 7

# The probe used to sweep every active model on every 2-hourly cycle: 20 active
# OpenRouter models x 12 cycles = 240 requests a day against a 50/day cap, none
# of it recorded. Three controls now bound it, and real traffic counts as
# freshness because the router writes last_probe_at on every request.
PROBE_MIN_AGE_HOURS = 24      # do not re-probe a model whose health is newer
PROBE_BUDGET_PER_PROVIDER = 8  # burst cap per cycle
PROBE_QUOTA_FLOOR = 0.25       # leave this share of the budget for real work

# Known free-tier quota limits (from research)
KNOWN_LIMITS = {
    # Google AI Studio (free tier, per project)
    "google": {
        "gemini-3.1-pro-preview":  {"rpm": 5,  "rpd": 50,   "tpm": 250000,  "reset_hour": 8},   # midnight PT = 08:00 UTC
        "gemma-3-27b-it":          {"rpm": 15, "rpd": 1500, "tpm": 1000000, "reset_hour": 8},
        "gemma-4-31b-it":          {"rpm": 15, "rpd": 1500, "tpm": 1000000, "reset_hour": 8},
    },
    # OpenRouter free tier
    "openrouter": {
        "_default": {"rpd": 50, "reset_hour": 0},  # 50 req/day without credits, UTC midnight
    },
    # Cerebras free tier
    "cerebras": {
        "_default": {"rpm": 30, "rpd": 14400, "tpd": 1000000, "reset_hour": 0},
    },
    # Groq free tier
    "groq": {
        "_default": {"rpm": 30, "tpm": 6000, "reset_hour": 0},
    },
    # Mistral free tier
    "mistral": {
        "_default": {"rpm": 2, "tpm": 500000, "reset_hour": 0},
    },
    # NIM - unknown, probe will discover
    "nvidia": {},
    # OpenCode Z - unknown, probe will discover
    "opencode": {},
    # DeepSeek - probe will discover
    "deepseek": {},
    # Cloudflare Workers AI
    "cloudflare": {
        "_default": {"rpm": 300, "reset_hour": 0},
    },
    # HuggingFace Inference
    "huggingface": {
        "_default": {"rph": 100, "reset_hour": 0},
    },
    # Kilo Gateway - probe will discover
    "kilo": {},
    # Agnes AI - probe will discover
    "agnes": {},
}

# The marker every rate-limit check tests for. It used to be spelled two ways
# — "rate_limited" in the check, "rate limited" in one of the producers — so the
# check silently never matched.
RATE_LIMITED = "rate_limited"


# Standard OpenAI-compatible probe config factory
def _openai_probe(url: str, key_env: str) -> dict:
    return {
        "url": url,
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "body": lambda model: {"messages": [{"role": "user", "content": "ok"}], "model": model, "max_tokens": 3},
        "key_env": key_env,
    }


# Provider API endpoints for probing
PROBE_ENDPOINTS = {
    "openrouter": _openai_probe(
        "https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "google": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        "headers": lambda key: {"Content-Type": "application/json"},
        "body": lambda model: {"contents": [{"parts": [{"text": "ok"}]}]},
        "key_env": "GOOGLE_AI_API_KEY",
        "url_builder": lambda model, key: f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
    },
    "nvidia": _openai_probe(
        "https://integrate.api.nvidia.com/v1/chat/completions", "NVIDIA_API_KEY"),
    "opencode": _openai_probe(
        "https://opencode.ai/zen/v1/chat/completions", "OPENCODE_API_KEY"),
    "deepseek": _openai_probe(
        "https://api.deepseek.com/v1/chat/completions", "DEEPSEEK_API_KEY"),
    "groq": _openai_probe(
        "https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "cerebras": _openai_probe(
        "https://api.cerebras.ai/v1/chat/completions", "CEREBRAS_API_KEY"),
    "mistral": _openai_probe(
        "https://api.mistral.ai/v1/chat/completions", "MISTRAL_API_KEY"),
    "kilo": _openai_probe(
        "https://api.kilo.ai/v1/chat/completions", "KILO_API_KEY"),
    "cloudflare": {
        "url": "",  # built by url_builder
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "body": lambda model: {"messages": [{"role": "user", "content": "ok"}]},
        "key_env": "CF_API_TOKEN",
        "url_builder": lambda model, key: f"https://api.cloudflare.com/client/v4/accounts/{os.getenv('CF_ACCOUNT_ID', '')}/ai/run/@cf/{model}",
    },
    "huggingface": {
        "url": "",  # built by url_builder
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "body": lambda model: {"inputs": "User: ok\nAssistant:", "parameters": {"max_new_tokens": 5}},
        "key_env": "HF_API_TOKEN",
        "url_builder": lambda model, key: f"https://api-inference.huggingface.co/models/{model}",
    },
    "agnes": _openai_probe(
        "https://apihub.agnes-ai.com/v1/chat/completions", "AGNES_API_KEY"),
}


async def probe_model(client: httpx.AsyncClient, provider_id: str, model_id: str) -> dict:
    """Probe a single model. Returns {available, error, latency_ms}."""
    endpoint = PROBE_ENDPOINTS.get(provider_id)
    if not endpoint:
        return {"available": False, "error": "no probe endpoint"}

    api_key = os.getenv(endpoint["key_env"], "")
    if not api_key:
        return {"available": False, "error": f"missing {endpoint['key_env']}"}

    try:
        import time
        t0 = time.monotonic()

        if "url_builder" in endpoint:
            url = endpoint["url_builder"](model_id, api_key)
            body = endpoint["body"](model_id)
        else:
            url = endpoint["url"]
            body = endpoint["body"](model_id)

        resp = await client.post(
            url,
            headers=endpoint["headers"](api_key),
            json=body,
            timeout=10,
        )
        latency = (time.monotonic() - t0) * 1000

        if resp.status_code == 200:
            return {"available": True, "error": None, "latency_ms": latency,
                    "reached": True}
        elif resp.status_code == 429:
            # Rate limited but model exists
            retry_after = resp.headers.get("retry-after", "")
            return {"available": False, "latency_ms": latency, "reached": True,
                    "error": f"{RATE_LIMITED} (retry-after={retry_after})"}
        elif resp.status_code == 401:
            try:
                err = resp.json().get("error", {})
                msg = err.get("message", resp.text)
            except Exception:
                msg = resp.text
            # This string is written to model_health.last_probe_error, which
            # /v1/models/health and /dashboard serve. Redact before truncating —
            # truncating a key only makes it shorter.
            return {"available": False, "error": f"auth/credits: {redact(msg)[:100]}",
                    "latency_ms": latency, "reached": True}
        elif resp.status_code == 404:
            return {"available": False, "error": "model not found", "latency_ms": latency,
                    "reached": True}
        else:
            return {"available": False, "error": f"HTTP {resp.status_code}", "latency_ms": latency,
                    "reached": True}

    except httpx.TimeoutException:
        return {"available": False, "error": "timeout"}
    except Exception as e:
        return {"available": False, "error": redact_error(e, 80)}


async def probe_all_models(
    min_age_hours: float = PROBE_MIN_AGE_HOURS,
    budget_per_provider: int = PROBE_BUDGET_PER_PROVIDER,
) -> dict[str, dict[str, dict]]:
    """Probe the models whose health data has gone stale.

    Returns {provider_id: {model_id: probe_result}}, covering only what was
    actually probed this cycle — a model that was skipped as fresh must not
    appear, or the caller would record a verdict nobody formed.

    Never-probed models go first: an unknown model is what makes the router
    guess. Real traffic refreshes `last_probe_at` through the same repository,
    so a model in active use is never probed synthetically.
    """
    import json as _json
    db = await get_db()
    recheck_before = (
        datetime.now(timezone.utc) - timedelta(days=DEACTIVATED_RECHECK_DAYS)
    ).isoformat()
    stale_before = (
        datetime.now(timezone.utc) - timedelta(hours=min_age_hours)
    ).isoformat()
    async with db.execute(
        """SELECT m.provider_id, m.model_id, m.task_scores
           FROM models m
           LEFT JOIN model_health mh ON m.provider_id = mh.provider_id
                                    AND m.model_id = mh.model_id
           WHERE (m.active = 1
                  OR (m.deactivated_at IS NOT NULL AND m.deactivated_at < ?))
             AND (mh.last_probe_at IS NULL OR mh.last_probe_at < ?)
           ORDER BY mh.last_probe_at IS NOT NULL, mh.last_probe_at, m.model_id""",
        (recheck_before, stale_before),
    ) as cur:
        models = await cur.fetchall()

    if not models:
        logger.info("Probe: every model's health is under %sh old — no calls made",
                    min_age_hours)
        return {}

    results = {}
    async with httpx.AsyncClient(timeout=30) as client:
        # Probe sequentially per provider (respect rate limits), parallel across providers
        provider_groups: dict[str, list[str]] = {}
        for pid, mid, scores_raw in models:
            # Skip image-only models — they don't support chat completions
            try:
                scores = _json.loads(scores_raw) if scores_raw else {}
            except (ValueError, TypeError):
                scores = {}
            if "image_generation" in scores and not any(
                k in scores for k in ("code", "reasoning", "summarize", "general")
            ):
                continue
            provider_groups.setdefault(pid, []).append(mid)

        # A provider running low on its daily budget must not have the rest of it
        # spent on liveness checks.
        quota = QuotaManager()
        headroom = {s["provider_id"]: s["quota_remaining_pct"]
                    for s in await quota.get_all_quota_status()}

        async def probe_provider(pid: str, mids: list[str]):
            r = {}
            if headroom.get(pid, 1.0) < PROBE_QUOTA_FLOOR:
                logger.info("Skipping probes for %s — only %.0f%% of its budget left",
                            pid, headroom.get(pid, 1.0) * 100)
                return pid, r
            for mid in mids[:budget_per_provider]:
                if not await quota.can_use(pid, mid, 16):
                    logger.info("Skipping probe of %s:%s — no quota", pid, mid)
                    continue
                r[mid] = await probe_model(client, pid, mid)
                if r[mid].get("reached"):
                    # The provider served this request and counted it, whatever
                    # it answered. Booking it is the difference between a budget
                    # the router manages and one it merely observes.
                    await quota.record_usage(pid, mid, 16, 3, True)
                # A rate-limited provider tells us nothing about its remaining
                # models, so stop — and report nothing for them. Recording them
                # as unavailable was recording an opinion we never formed: five
                # HuggingFace models reached 4 consecutive "failures" that way,
                # one short of being hard-skipped by the router.
                if not r[mid]["available"] and RATE_LIMITED in (r[mid].get("error") or ""):
                    logger.info("%s rate limited — skipping its remaining %d models",
                                pid, min(len(mids), budget_per_provider) - len(r))
                    break
                await asyncio.sleep(0.5)
            return pid, r

        tasks = [probe_provider(pid, mids) for pid, mids in provider_groups.items()]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for result in gathered:
            if isinstance(result, Exception):
                logger.error("Probe failed: %s", redact(result))
                continue
            pid, probe_results = result
            results[pid] = probe_results

    return results


# KNOWN_LIMITS uses this key for a cap that applies to the whole account rather
# than to one model. The quota tables spell the same idea as an empty model_id.
PROVIDER_WIDE = "_default"


async def seed_known_limits():
    """Seed known quota limits into the DB from research data.

    `_default` entries used to be skipped outright, so OpenRouter's 50/day cap —
    the single tightest constraint this router exists to manage — was never
    written, `can_use` found no limits, and it returned True for every request
    ever made. They are stored as the `model_id = ''` row the schema already
    reserves for account-wide limits.
    """
    db = await get_db()
    for provider_id, models in KNOWN_LIMITS.items():
        for model_id, limits in models.items():
            if model_id.startswith("_") and model_id != PROVIDER_WIDE:
                continue
            if model_id == PROVIDER_WIDE:
                model_id = ""
            await db.execute(
                """INSERT INTO quota_limits (provider_id, model_id, rpm, rph, rpd, tpm, tph, tpd)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(provider_id, model_id) DO UPDATE SET
                   rpm = COALESCE(excluded.rpm, rpm),
                   rph = COALESCE(excluded.rph, rph),
                   rpd = COALESCE(excluded.rpd, rpd),
                   tpm = COALESCE(excluded.tpm, tpm),
                   tph = COALESCE(excluded.tph, tph),
                   tpd = COALESCE(excluded.tpd, tpd)""",
                (provider_id, model_id,
                 limits.get("rpm"), limits.get("rph"), limits.get("rpd"),
                 limits.get("tpm"), limits.get("tph"), limits.get("tpd")),
            )
            # Also update provider reset hour if known
            if "reset_hour" in limits:
                await db.execute(
                    "UPDATE providers SET daily_reset_utc_hour = ? WHERE id = ?",
                    (limits["reset_hour"], provider_id),
                )
    await db.commit()
    logger.info("Seeded known quota limits")


async def run_probe_and_update() -> dict:
    """Full probe cycle: test all models, update active status, seed known limits."""
    await seed_known_limits()
    results = await probe_all_models()

    db = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    health_repo = ModelHealthRepository()
    stats = {"probed": 0, "available": 0, "unavailable": 0,
             "deactivated": [], "reactivated": []}

    for provider_id, model_results in results.items():
        for model_id, probe in model_results.items():
            stats["probed"] += 1
            available = probe["available"]
            latency_ms = probe.get("latency_ms")
            error = probe.get("error")

            # Write model_health for every probe result
            await health_repo.update(
                provider_id, model_id, available, latency_ms, error, now
            )

            if available:
                stats["available"] += 1
                async with db.execute(
                    "SELECT active FROM models WHERE provider_id = ? AND model_id = ?",
                    (provider_id, model_id),
                ) as cur:
                    row = await cur.fetchone()
                if row and not row[0]:
                    stats["reactivated"].append(f"{provider_id}:{model_id}")
                    logger.info("Reactivated %s:%s — probe succeeded", provider_id, model_id)
                await db.execute(
                    """UPDATE models SET active = 1, deactivated_reason = NULL,
                           deactivated_at = NULL
                       WHERE provider_id = ? AND model_id = ?""",
                    (provider_id, model_id),
                )
            else:
                stats["unavailable"] += 1
                error = error or "unknown"
                if "auth/credits" in error or "not found" in error:
                    await db.execute(
                        """UPDATE models SET active = 0, deactivated_reason = ?,
                               deactivated_at = ?
                           WHERE provider_id = ? AND model_id = ?""",
                        (error, now, provider_id, model_id),
                    )
                    stats["deactivated"].append(f"{provider_id}:{model_id} ({error})")
                    logger.warning("Deactivated %s:%s - %s", provider_id, model_id, error)
                elif RATE_LIMITED in error:
                    logger.info("Rate limited: %s:%s - %s", provider_id, model_id, error)
                else:
                    logger.warning("Probe failed: %s:%s - %s", provider_id, model_id, error)

    await db.commit()
    return stats
