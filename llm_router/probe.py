"""Probe discovered models to verify they're actually available and free."""
import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

import httpx

from .db import get_db
from .redact import redact, redact_error
from .health import ModelHealthRepository

logger = logging.getLogger(__name__)

# A model the probe switched off is re-checked this many days later. Without a
# re-check a transient 404 is permanent: probe_all_models only looks at active
# rows, and discovery no longer flips them back on.
DEACTIVATED_RECHECK_DAYS = 7

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
            return {"available": True, "error": None, "latency_ms": latency}
        elif resp.status_code == 429:
            # Rate limited but model exists
            retry_after = resp.headers.get("retry-after", "")
            return {"available": False, "error": f"rate_limited (retry-after={retry_after})", "latency_ms": latency}
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
                    "latency_ms": latency}
        elif resp.status_code == 404:
            return {"available": False, "error": "model not found", "latency_ms": latency}
        else:
            return {"available": False, "error": f"HTTP {resp.status_code}", "latency_ms": latency}

    except httpx.TimeoutException:
        return {"available": False, "error": "timeout"}
    except Exception as e:
        return {"available": False, "error": redact_error(e, 80)}


async def probe_all_models() -> dict[str, dict[str, dict]]:
    """Probe all active models in the DB. Returns {provider_id: {model_id: probe_result}}."""
    import json as _json
    db = await get_db()
    recheck_before = (
        datetime.now(timezone.utc) - timedelta(days=DEACTIVATED_RECHECK_DAYS)
    ).isoformat()
    async with db.execute(
        """SELECT provider_id, model_id, task_scores FROM models
           WHERE active = 1
              OR (deactivated_at IS NOT NULL AND deactivated_at < ?)""",
        (recheck_before,),
    ) as cur:
        models = await cur.fetchall()

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

        async def probe_provider(pid: str, mids: list[str]):
            r = {}
            rate_limited = False
            for mid in mids:
                if rate_limited:
                    r[mid] = {"available": False, "error": "provider rate limited (skipped)"}
                    continue
                r[mid] = await probe_model(client, pid, mid)
                # If provider is rate limited, skip remaining models for this provider
                if not r[mid]["available"] and "rate_limited" in r[mid].get("error", ""):
                    rate_limited = True
                    continue
                # Small delay between probes
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


async def seed_known_limits():
    """Seed known quota limits into the DB from research data."""
    db = await get_db()
    for provider_id, models in KNOWN_LIMITS.items():
        for model_id, limits in models.items():
            if model_id.startswith("_"):
                continue
            await db.execute(
                """INSERT INTO quota_limits (provider_id, model_id, rpm, rpd, tpm)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(provider_id, model_id) DO UPDATE SET
                   rpm = COALESCE(excluded.rpm, rpm),
                   rpd = COALESCE(excluded.rpd, rpd),
                   tpm = COALESCE(excluded.tpm, tpm)""",
                (provider_id, model_id,
                 limits.get("rpm"), limits.get("rpd"), limits.get("tpm")),
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
                elif "rate_limited" in error:
                    logger.info("Rate limited: %s:%s - %s", provider_id, model_id, error)
                else:
                    logger.warning("Probe failed: %s:%s - %s", provider_id, model_id, error)

    await db.commit()
    return stats
