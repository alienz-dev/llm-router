"""What each model can actually do, and how we found out.

Two sources feed the same four columns. A verified inventory
(`docs/model-capabilities-*.json`, 33 models probed live) seeds them once, and a
weekly probe keeps them honest — weekly, not on the 2-hourly health cycle,
because two extra calls per model per cycle would be roughly 720 OpenRouter
requests a day against a 50/day cap.

Three rules, learned the hard way:

* **NULL means unknown, not unsupported.** On the day the migration lands every
  row is NULL, and a `WHERE supports_tools = 1` filter would empty the candidate
  set for every structured request.
* **Results are a snapshot with noise.** A second run of the same probe an hour
  later flipped two models from schema-ok to malformed output. Everything
  carries `capability_checked_at` and is re-probed; nothing is trusted forever.
* **A provider ceiling wins.** These flags narrow within what an adapter can
  express (ADR-01); they never widen it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from .db import get_db
from .redact import redact

logger = logging.getLogger(__name__)

# How long a capability reading stays good. Shorter than it sounds: free tiers
# move without notice — Cerebras cut its free catalogue from ~12 models to 2 in
# May 2026 with no deprecation notice.
CAPABILITY_TTL_DAYS = 7

# Models probed per provider per cycle. A ReAct loop already competes for the
# same 50 requests/day; capability probing must not be what exhausts them.
PROBE_BUDGET_PER_PROVIDER = 8

DEFAULT_INVENTORY = (
    Path(__file__).resolve().parent.parent / "docs" / "model-capabilities-2026-08-18.json"
)

TOOLS_PROBE = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"},
                           "unit": {"type": "string", "enum": ["c", "f"]}},
            "required": ["city"],
        },
    },
}]

SCHEMA_PROBE = {
    "type": "json_schema",
    "json_schema": {
        "name": "job_fit",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"score": {"type": "integer"}, "reason": {"type": "string"}},
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
    },
}

# Verdicts that are evidence about the *model*. Anything else — a 404, a 429, a
# transport error — says nothing about capability and must leave the flag alone.
_TOOLS_YES = {"native"}
_TOOLS_NO = {"ignored", "text-leak", "native-noid", "native-broken", "HTTP400"}
_JSON_YES = {"schema-ok"}
_JSON_NO = {"not-json", "json-off-schema", "HTTP400"}


def classify_tools(body: dict) -> tuple[str, str]:
    try:
        msg = body["choices"][0]["message"]
    except Exception:
        return "malformed", ""
    calls = msg.get("tool_calls")
    if calls:
        try:
            fn = calls[0]["function"]
            args = json.loads(fn["arguments"])
            # No id means no way to match the tool result back on the next turn,
            # which is the whole point of the loop.
            has_id = bool(calls[0].get("id"))
            note = f"{fn['name']}({','.join(args)})" + ("" if has_id else " NO-ID")
            return ("native" if has_id else "native-noid"), note
        except Exception as e:
            return "native-broken", type(e).__name__
    content = msg.get("content") or ""
    if "get_weather" in content:
        return "text-leak", content[:70].replace("\n", " ")
    return "ignored", content[:70].replace("\n", " ")


def classify_json(body: dict) -> tuple[str, str]:
    try:
        content = body["choices"][0]["message"]["content"] or ""
    except Exception:
        return "malformed", ""
    try:
        obj = json.loads(content)
    except Exception:
        return "not-json", content[:60].replace("\n", " ")
    if isinstance(obj, dict) and {"score", "reason"} <= set(obj):
        return "schema-ok", json.dumps(obj)[:60]
    return "json-off-schema", json.dumps(obj)[:60]


def flags_from_verdicts(tools: str, json_verdict: str) -> dict[str, int | None]:
    """Turn probe verdicts into column values. Unknown stays unknown."""
    return {
        "supports_tools": 1 if tools in _TOOLS_YES else (0 if tools in _TOOLS_NO else None),
        "supports_json_schema": (
            1 if json_verdict in _JSON_YES else (0 if json_verdict in _JSON_NO else None)
        ),
    }


async def record_capability(
    provider_id: str, model_id: str, flags: dict, checked_at: str | None = None,
    only_if_unknown: bool = False,
) -> bool:
    """Write capability flags for one model. Returns whether a row changed.

    A flag whose value is None is left untouched — a failed probe must not erase
    what an earlier successful one found.
    """
    db = await get_db()
    known = {k: v for k, v in flags.items() if v is not None}
    if not known:
        return False

    checked_at = checked_at or datetime.now(timezone.utc).isoformat()
    assignments = ", ".join(f"{col} = ?" for col in known)
    sql = (f"UPDATE models SET {assignments}, capability_checked_at = ? "
           "WHERE provider_id = ? AND model_id = ?")
    args = [*known.values(), checked_at, provider_id, model_id]
    if only_if_unknown:
        sql += " AND capability_checked_at IS NULL"
    cur = await db.execute(sql, args)
    await db.commit()
    return cur.rowcount > 0


async def seed_from_inventory(path: Path | str | None = None,
                              overwrite: bool = False) -> int:
    """Fill capability columns from a verified inventory file.

    Only touches models discovery already knows about, and by default only rows
    that have never been checked — a live probe result always outranks a file.
    """
    path = Path(path or DEFAULT_INVENTORY)
    if not path.exists():
        logger.info("No capability inventory at %s — nothing to seed", path)
        return 0

    try:
        records = json.loads(path.read_text())
    except (ValueError, OSError) as e:
        logger.warning("Could not read capability inventory %s: %s", path, redact(e))
        return 0

    # Date the reading by the file, not by now: a two-month-old inventory should
    # look stale to the re-probe, because it is.
    checked_at = _inventory_date(path)
    seeded = 0
    for record in records:
        provider, model = record.get("provider"), record.get("model")
        if not provider or not model:
            continue
        flags = flags_from_verdicts(record.get("tools", ""), record.get("json", ""))
        if await record_capability(provider, model, flags, checked_at,
                                   only_if_unknown=not overwrite):
            seeded += 1
    if seeded:
        logger.info("Seeded capability flags for %d models from %s", seeded, path.name)
    return seeded


def _inventory_date(path: Path) -> str:
    stem = path.stem.rsplit("-", 3)
    if len(stem) == 4:
        try:
            return datetime.strptime("-".join(stem[1:]), "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            ).isoformat()
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


async def models_due_for_probe(
    ttl_days: int = CAPABILITY_TTL_DAYS,
    budget_per_provider: int = PROBE_BUDGET_PER_PROVIDER,
) -> list[tuple[str, str]]:
    """Active models whose capability reading is missing or stale.

    Never-checked models come first: an unknown flag is what makes the router
    guess, and a stale one at least reflects something that was once true.
    """
    db = await get_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=ttl_days)).isoformat()
    async with db.execute(
        """SELECT provider_id, model_id FROM models
           WHERE active = 1
             AND (capability_checked_at IS NULL OR capability_checked_at < ?)
           ORDER BY capability_checked_at IS NOT NULL, capability_checked_at, model_id""",
        (cutoff,),
    ) as cur:
        rows = await cur.fetchall()

    per_provider: dict[str, int] = {}
    due = []
    for provider_id, model_id in rows:
        if per_provider.get(provider_id, 0) >= budget_per_provider:
            continue
        per_provider[provider_id] = per_provider.get(provider_id, 0) + 1
        due.append((provider_id, model_id))
    return due


async def probe_one(client: httpx.AsyncClient, provider_id: str, model_id: str) -> dict:
    """Two calls against the provider's own endpoint: a tool call and a strict schema."""
    from .config import get_config

    # Every return from here carries provider/model: the caller reads them off
    # the record, and an early return without them took down the whole cycle,
    # not just this model.
    unknown = {"provider": provider_id, "model": model_id,
               "tools_note": "", "json_note": ""}

    provider = get_config().providers.get(provider_id)
    if not provider:
        return {**unknown, "tools": "no-config", "json": "no-config"}
    api_key = os.getenv(provider.api_key_env, "")
    if not api_key:
        return {**unknown, "tools": "no-key", "json": "no-key"}

    url = provider.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def _call(extra: dict, prompt: str) -> tuple[dict | None, str]:
        try:
            resp = await client.post(url, headers=headers, timeout=90, json={
                "model": model_id, "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}], **extra,
            })
        except Exception as e:
            return None, f"ERR{type(e).__name__}"
        if resp.status_code != 200:
            return None, f"HTTP{resp.status_code}"
        try:
            return resp.json(), ""
        except ValueError:
            return None, "malformed"

    body, failure = await _call(
        {"tools": TOOLS_PROBE, "tool_choice": "auto"},
        "What is the weather in Sydney? Use the tool.",
    )
    tools_verdict, tools_note = classify_tools(body) if body else (failure, "")

    body, failure = await _call(
        {"response_format": SCHEMA_PROBE},
        "Rate this job fit 0-100: Python backend role, candidate is a Python dev. "
        "Respond as JSON.",
    )
    json_verdict, json_note = classify_json(body) if body else (failure, "")

    return {"provider": provider_id, "model": model_id,
            "tools": tools_verdict, "tools_note": redact(tools_note),
            "json": json_verdict, "json_note": redact(json_note)}


async def run_capability_probe(
    ttl_days: int = CAPABILITY_TTL_DAYS,
    budget_per_provider: int = PROBE_BUDGET_PER_PROVIDER,
) -> dict:
    """One capability cycle. Returns what it spent and what it learned."""
    due = await models_due_for_probe(ttl_days, budget_per_provider)
    if not due:
        logger.info("Capability probe: nothing stale — no provider calls made")
        return {"probed": 0, "updated": 0, "results": []}

    results = []
    async with httpx.AsyncClient() as client:
        # Sequential per provider, concurrent across providers: the same shape
        # the health probe uses, for the same rate-limit reason.
        by_provider: dict[str, list[str]] = {}
        for provider_id, model_id in due:
            by_provider.setdefault(provider_id, []).append(model_id)

        async def _provider(provider_id: str, model_ids: list[str]) -> list[dict]:
            out = []
            for model_id in model_ids:
                out.append(await probe_one(client, provider_id, model_id))
                await asyncio.sleep(0.5)
            return out

        gathered = await asyncio.gather(
            *[_provider(p, m) for p, m in by_provider.items()], return_exceptions=True
        )

    updated = 0
    for group in gathered:
        if isinstance(group, Exception):
            logger.error("Capability probe failed: %s", redact(group))
            continue
        for record in group:
            results.append(record)
            provider_id, model_id = record.get("provider"), record.get("model")
            if not provider_id or not model_id:
                logger.warning("Capability probe returned a record with no model: %r", record)
                continue
            flags = flags_from_verdicts(record.get("tools", ""), record.get("json", ""))
            if await record_capability(provider_id, model_id, flags):
                updated += 1

    logger.info("Capability probe: %d models, %d flags updated", len(results), updated)
    return {"probed": len(results), "updated": updated, "results": results}
