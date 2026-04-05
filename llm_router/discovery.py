import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import httpx

from .db import get_db

logger = logging.getLogger(__name__)


# Default task scores by model name pattern
_DEFAULT_SCORES = {
    "codestral": {"code": 0.95, "reasoning": 0.7, "summarize": 0.6, "general": 0.7},
    "qwen": {"code": 0.85, "reasoning": 0.8, "summarize": 0.8, "general": 0.8},
    "gpt-oss": {"code": 0.9, "reasoning": 0.9, "summarize": 0.85, "general": 0.9},
    "llama": {"code": 0.8, "reasoning": 0.85, "summarize": 0.8, "general": 0.8},
    "gemini": {"code": 0.85, "reasoning": 0.9, "summarize": 0.85, "general": 0.85},
    "mistral": {"code": 0.8, "reasoning": 0.85, "summarize": 0.8, "general": 0.8},
    "nemotron": {"code": 0.85, "reasoning": 0.85, "summarize": 0.8, "general": 0.85},
}
_FALLBACK_SCORES = {"code": 0.5, "reasoning": 0.5, "summarize": 0.5, "general": 0.5}


def _get_task_scores(model_id: str) -> str:
    lower = model_id.lower()
    for pattern, scores in _DEFAULT_SCORES.items():
        if pattern in lower:
            return json.dumps(scores)
    return json.dumps(_FALLBACK_SCORES)


class ModelDiscovery:
    """Discovers free models from provider APIs and updates the models table."""

    async def discover_all(self) -> dict[str, list[dict]]:
        tasks = {
            "openrouter": self._discover_openrouter(),
            "cerebras": self._discover_cerebras(),
            "groq": self._discover_groq(),
            "kilo": self._discover_kilo(),
        }
        results = {}
        gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for provider_id, result in zip(tasks.keys(), gathered):
            if isinstance(result, Exception):
                logger.error("Discovery failed for %s: %s", provider_id, result)
                results[provider_id] = []
            else:
                results[provider_id] = result
                logger.info("%s: discovered %d models", provider_id, len(result))
        return results

    async def _discover_openrouter(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get("https://openrouter.ai/api/v1/models")
            resp.raise_for_status()
        models = []
        for m in resp.json().get("data", []):
            pricing = m.get("pricing", {})
            if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
                models.append({
                    "model_id": m["id"],
                    "display_name": m.get("name", m["id"]),
                    "context_length": m.get("context_length", 4096),
                    "task_scores": _get_task_scores(m["id"]),
                })
        return models

    async def _discover_cerebras(self) -> list[dict]:
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            return []
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://api.cerebras.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
        return [
            {
                "model_id": m["id"],
                "display_name": m.get("id"),
                "context_length": m.get("context_window", 128000),
                "task_scores": _get_task_scores(m["id"]),
            }
            for m in resp.json().get("data", [])
        ]

    async def _discover_groq(self) -> list[dict]:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return []
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
        return [
            {
                "model_id": m["id"],
                "display_name": m.get("id"),
                "context_length": m.get("context_window", 8192),
                "task_scores": _get_task_scores(m["id"]),
            }
            for m in resp.json().get("data", [])
        ]

    async def _discover_kilo(self) -> list[dict]:
        api_key = os.getenv("KILO_API_KEY")
        if not api_key:
            return []
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://api.kilo.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
        return [
            {
                "model_id": m["id"],
                "display_name": m.get("name", m["id"]),
                "context_length": m.get("context_length", 4096),
                "task_scores": _get_task_scores(m["id"]),
            }
            for m in resp.json().get("data", [])
        ]

    async def update_models_table(self, discovered: dict[str, list[dict]]) -> dict[str, dict]:
        db = await get_db()
        now = datetime.now(timezone.utc).isoformat()
        stats = {}

        for provider_id, models in discovered.items():
            async with db.execute(
                "SELECT model_id FROM models WHERE provider_id = ? AND active = 1",
                (provider_id,),
            ) as cursor:
                existing_ids = {row[0] for row in await cursor.fetchall()}

            discovered_ids = {m["model_id"] for m in models}
            added = discovered_ids - existing_ids
            removed = existing_ids - discovered_ids

            # Mark removed as inactive
            for mid in removed:
                await db.execute(
                    "UPDATE models SET active = 0 WHERE provider_id = ? AND model_id = ?",
                    (provider_id, mid),
                )

            # Upsert discovered
            for m in models:
                await db.execute(
                    """INSERT INTO models
                       (provider_id, model_id, display_name, context_length, task_scores, is_free, discovered_at, active)
                       VALUES (?, ?, ?, ?, ?, 1, ?, 1)
                       ON CONFLICT(provider_id, model_id) DO UPDATE SET
                       display_name = excluded.display_name,
                       context_length = excluded.context_length,
                       task_scores = excluded.task_scores,
                       active = 1""",
                    (provider_id, m["model_id"], m["display_name"],
                     m["context_length"], m["task_scores"], now),
                )

            await db.commit()

            if added:
                logger.info("%s: +%d models: %s", provider_id, len(added), added)
            if removed:
                logger.info("%s: -%d models: %s", provider_id, len(removed), removed)

            stats[provider_id] = {
                "total": len(models),
                "added": len(added),
                "removed": len(removed),
                "active": len(discovered_ids),
            }

        return stats
