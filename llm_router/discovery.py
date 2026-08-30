import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import httpx

from .db import get_db
from .redact import redact

logger = logging.getLogger(__name__)


# Default task scores by model name pattern
_DEFAULT_SCORES = {
    "codestral": {"code": 0.95, "reasoning": 0.7, "summarize": 0.6, "general": 0.7},
    "qwen": {"code": 0.85, "reasoning": 0.8, "summarize": 0.8, "general": 0.8},
    "qwen3-coder": {"code": 0.98, "reasoning": 0.85, "summarize": 0.8, "general": 0.9},
    "gpt-oss": {"code": 0.9, "reasoning": 0.9, "summarize": 0.85, "general": 0.9},
    "llama": {"code": 0.8, "reasoning": 0.85, "summarize": 0.8, "general": 0.8},
    "llama-3.3-70b": {"code": 0.85, "reasoning": 0.85, "summarize": 0.85, "general": 0.85},
    "gemini": {"code": 0.85, "reasoning": 0.9, "summarize": 0.85, "general": 0.85},
    "mistral": {"code": 0.8, "reasoning": 0.85, "summarize": 0.8, "general": 0.8},
    "nemotron-3-ultra": {"code": 0.95, "reasoning": 0.98, "summarize": 0.9, "general": 0.95},
    "nemotron-3-super": {"code": 0.9, "reasoning": 0.9, "summarize": 0.85, "general": 0.9},
    "nemotron": {"code": 0.85, "reasoning": 0.85, "summarize": 0.8, "general": 0.85},
    "deepseek-v4": {"code": 0.95, "reasoning": 0.9, "summarize": 0.85, "general": 0.9},
    "agnes": {"code": 0.75, "reasoning": 0.75, "summarize": 0.7, "general": 0.75},
}
_FALLBACK_SCORES = {"code": 0.5, "reasoning": 0.5, "summarize": 0.5, "general": 0.5}


def _get_task_scores(model_id: str) -> str:
    lower = model_id.lower()
    for pattern, scores in _DEFAULT_SCORES.items():
        if pattern in lower:
            return json.dumps(scores)
    return json.dumps(_FALLBACK_SCORES)


# Provider priority multiplier — lower = deprioritized
# Applied to final score in router ranking
PROVIDER_PRIORITY = {
    "openrouter": 1.2,   # 1000 req/day free tier — try first
    "nvidia": 1.1,       # reliable fallback
    "opencode": 1.0,     # Nemotron via OpenCode
    "deepseek": 1.0,     # Direct API (new)
    "agnes": 1.0,        # Agnes AI (free, no visible limits)
    "google": 0.5,       # Quota exhausted
    "cerebras": 0.9,
    "groq": 0.9,
}


class ModelDiscovery:
    """Discovers free models from provider APIs and updates the models table."""

    async def _discover_openai_models(
        self,
        api_key_env: str,
        url: str,
        default_context: int = 4096,
        context_key: str = "context_length",
        name_key: str = "name",
        filter_fn=None,
    ) -> list[dict]:
        """Generic discovery for OpenAI-compatible /v1/models endpoints.

        Args:
            api_key_env: Environment variable name for the API key
            url: Full URL to the models endpoint
            default_context: Default context length if not in response
            context_key: JSON key for context length in model object
            name_key: JSON key for display name ("name" or "id")
            filter_fn: Optional callable(model_dict) -> bool to filter models
        """
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            return []
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
            resp.raise_for_status()
        models = []
        for m in resp.json().get("data", []):
            if filter_fn and not filter_fn(m):
                continue
            models.append({
                "model_id": m["id"],
                "display_name": m.get(name_key, m["id"]),
                "context_length": m.get(context_key, default_context),
                "task_scores": _get_task_scores(m["id"]),
            })
        return models

    async def discover_all(self) -> dict[str, list[dict]]:
        tasks = {
            "openrouter": self._discover_openrouter(),
            "cerebras": self._discover_cerebras(),
            "groq": self._discover_groq(),
            "kilo": self._discover_kilo(),
            "google": self._discover_google(),
            "opencode": self._discover_opencode(),
            "nvidia": self._discover_nim(),
            "deepseek": self._discover_deepseek(),
            "mistral": self._discover_mistral(),
            "cloudflare": self._discover_cloudflare(),
            "huggingface": self._discover_huggingface(),
            "agnes": self._discover_agnes(),
        }
        results = {}
        gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for provider_id, result in zip(tasks.keys(), gathered):
            if isinstance(result, Exception):
                # httpx puts the request URL in the message, and Google's carries ?key=.
                logger.error("Discovery failed for %s: %s", provider_id, redact(result))
                results[provider_id] = []
            else:
                results[provider_id] = result
                logger.info("%s: discovered %d models", provider_id, len(result))
        return results

    async def _discover_cerebras(self) -> list[dict]:
        return await self._discover_openai_models(
            "CEREBRAS_API_KEY", "https://api.cerebras.ai/v1/models",
            default_context=128000, context_key="context_window", name_key="id",
        )

    async def _discover_groq(self) -> list[dict]:
        return await self._discover_openai_models(
            "GROQ_API_KEY", "https://api.groq.com/openai/v1/models",
            default_context=8192, context_key="context_window", name_key="id",
        )

    async def _discover_google(self) -> list[dict]:
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            return []
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            )
            resp.raise_for_status()
        models = []
        for m in resp.json().get("models", []):
            methods = m.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                continue
            model_id = m["name"].split("/")[-1]
            models.append({
                "model_id": model_id,
                "display_name": m.get("displayName", model_id),
                "context_length": m.get("inputTokenLimit", 32768),
                "task_scores": _get_task_scores(model_id),
            })
        allowed = [m for m in models if m["model_id"] in self._GOOGLE_KEEP]
        rest = [m for m in models if m["model_id"] not in self._GOOGLE_KEEP]
        return self._dedup_series(rest) + allowed

    def _dedup_series(self, models: list[dict]) -> list[dict]:
        """Keep only the best model per series (e.g. gemma-3 -> only 27b, gemini -> only latest pro)."""
        import re

        # Group by series prefix (e.g. "gemma-3", "gemini-2.5", "llama-3.3")
        series: dict[str, list[dict]] = {}
        for m in models:
            mid = m["model_id"].lower()
            # Extract series key: family-version (e.g. gemma-3, gemini-2.5, llama-3.3)
            match = re.match(r"^(openai/)?([\w.-]+?)[-\s](\d+(?:\.\d+)?)", mid)
            if match:
                key = f"{match.group(2)}-{match.group(3)}"
            else:
                key = mid.split(":")[0]  # full id as key for non-matching
            series.setdefault(key, []).append(m)

        best = []
        for key, variants in series.items():
            if len(variants) == 1:
                best.append(variants[0])
                continue
            # Score each variant: prefer pro > flash > lite, larger > smaller, newer > older
            def score(m):
                mid = m["model_id"].lower()
                s = 0
                # Tier: pro > flash > lite
                if "pro" in mid: s += 100
                elif "flash" in mid: s += 50
                elif "lite" in mid: s += 10
                # Size: extract parameter count for open models (e.g. 27b, 120b)
                size_match = re.search(r"(\d+)b", mid)
                if size_match: s += int(size_match.group(1))
                # Preview/newer bonus
                if "preview" in mid: s += 5
                # Penalize TTS, image, robotics, computer-use, customtools specializations
                for skip in ["tts", "image", "robotics", "computer-use", "customtools", "lyria", "clip"]:
                    if skip in mid: s -= 200
                return s
            winner = max(variants, key=score)
            best.append(winner)
        return best

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
        # Allowlist takes priority over dedup — extract kept models first,
        # then dedup the rest, then combine.
        allowed = [m for m in models if m["model_id"] in self._OPENROUTER_KEEP]
        rest = [m for m in models if m["model_id"] not in self._OPENROUTER_KEEP]
        return self._dedup_series(rest) + allowed

    # Explicit allowlists — only keep the most capable model per family
    _OPENROUTER_KEEP = {
        "meta-llama/llama-3.3-70b-instruct:free",    # best llama
        "nousresearch/hermes-3-llama-3.1-405b:free",  # best Nous
        "nvidia/nemotron-3-ultra-550b-a55b:free",     # best nemotron (550B MoE)
        "nvidia/nemotron-3-super-120b-a12b:free",     # nemotron super (120B)
        "openai/gpt-oss-120b:free",                   # best gpt-oss
        "qwen/qwen3-coder:free",                      # best qwen (coding)
        "z-ai/glm-4.5-air:free",                      # best GLM
        "minimax/minimax-m2.5:free",                   # minimax (OpenRouter version)
        "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",  # best dolphin
        "arcee-ai/trinity-large-preview:free",        # unique
        "openrouter/elephant-alpha",                  # unique
        "openrouter/free",                            # unique
        "liquid/lfm-2.5-1.2b-thinking:free",         # unique LFM thinking
        "google/gemma-4-31b-it:free",                 # best Gemma 4
        "google/gemma-3-4b-it:free",                  # fast triage (small gemma-3)
    }
    _GOOGLE_KEEP = {
        "gemini-3.1-pro-preview",      # best Gemini
        "gemma-4-31b-it",              # best Gemma 4
        "gemma-3-27b-it",              # best Gemma 3
    }
    _NIM_KEEP = {
        # Meta/Llama
        "meta/llama-3.3-70b-instruct",
        "meta/llama-4-maverick-17b-128e-instruct",
        # DeepSeek
        "deepseek-ai/deepseek-v3.2",
        # Mistral
        "mistralai/mistral-large-3-675b-instruct-2512",
        "mistralai/devstral-2-123b-instruct-2512",
        # Qwen
        "qwen/qwen3.5-397b-a17b",
        "qwen/qwen3-coder-480b-a35b-instruct",
        # Kimi
        "moonshotai/kimi-k2.5",
        # GLM
        "z-ai/glm5",
        # NVIDIA Nemotron
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        # Microsoft Phi
        "microsoft/phi-4-mini-instruct",
        # Yi
        "01-ai/yi-large",
        # Mistral Nemotron bridge
        "mistralai/mistral-nemotron",
        # Jamba
        "ai21labs/jamba-1.5-large-instruct",
        # Seed
        "bytedance/seed-oss-36b-instruct",
        # DBRX
        "databricks/dbrx-instruct",
        # Magistral
        "mistralai/magistral-small-2506",
    }

    async def _discover_kilo(self) -> list[dict]:
        return await self._discover_openai_models(
            "KILO_API_KEY", "https://api.kilo.ai/v1/models",
        )

    # opencode.ai zen serves 60+ models from one endpoint and marks none of them
    # free — /v1/models returns id, object, created, owned_by and nothing else.
    # models.dev is the catalogue opencode's own CLI reads, and it carries cost
    # and context. Intersecting the two gives what is both *served today* and
    # *free today*, which a hardcoded pair could never track: the previous list
    # named two models, one of which this repo's own adapter blocks, so the
    # router knew one of zen's seven free models.
    MODELS_DEV_URL = "https://models.dev/api.json"

    # Used only when models.dev is unreachable. Zen's free ids end in "-free",
    # with one long-standing exception.
    _ZEN_FREE_FALLBACK = {"big-pickle"}

    async def _zen_free_catalogue(self, client: httpx.AsyncClient) -> dict[str, dict]:
        """{model_id: {"context": int, "tools": bool}} for zen's cost-zero models."""
        try:
            resp = await client.get(self.MODELS_DEV_URL, timeout=20)
            resp.raise_for_status()
            catalogue = resp.json().get("opencode", {}).get("models", {})
        except Exception as e:
            logger.warning("models.dev unavailable, falling back to name matching: %s",
                           redact(e))
            return {}

        free = {}
        for model_id, meta in catalogue.items():
            cost = meta.get("cost") or {}
            if cost.get("input") == 0 and cost.get("output") == 0:
                free[model_id] = {
                    "context": (meta.get("limit") or {}).get("context"),
                    "tools": meta.get("tool_call"),
                }
        return free

    async def _discover_opencode(self) -> list[dict]:
        api_key = os.getenv("OPENCODE_API_KEY")
        if not api_key:
            return []
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://opencode.ai/zen/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            served = [m["id"] for m in resp.json().get("data", [])]
            free = await self._zen_free_catalogue(client)

        models = []
        for model_id in served:
            if free:
                if model_id not in free:
                    continue
                context = free[model_id].get("context") or 200000
            else:
                if not (model_id.endswith("-free") or model_id in self._ZEN_FREE_FALLBACK):
                    continue
                context = 200000
            models.append({
                "model_id": model_id,
                "display_name": model_id,
                "context_length": context,
                "task_scores": _get_task_scores(model_id),
            })
        return models

    async def _discover_nim(self) -> list[dict]:
        return await self._discover_openai_models(
            "NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1/models",
            name_key="id",
            filter_fn=lambda m: m["id"] in self._NIM_KEEP,
        )

    async def _discover_deepseek(self) -> list[dict]:
        return await self._discover_openai_models(
            "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1/models",
            default_context=65536, name_key="id",
        )

    async def _discover_mistral(self) -> list[dict]:
        return await self._discover_openai_models(
            "MISTRAL_API_KEY", "https://api.mistral.ai/v1/models",
            default_context=32768, context_key="max_context_length",
        )

    async def _discover_cloudflare(self) -> list[dict]:
        api_token = os.getenv("CF_API_TOKEN")
        account_id = os.getenv("CF_ACCOUNT_ID")
        if not api_token or not account_id:
            return []
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search",
                headers={"Authorization": f"Bearer {api_token}"},
            )
            resp.raise_for_status()
        models = []
        for m in resp.json().get("result", []):
            task = m.get("task", {})
            if task.get("name") not in ("Text Generation", "Question Answering"):
                continue
            model_id = m["name"]
            models.append({
                "model_id": model_id,
                "display_name": m.get("description", model_id),
                "context_length": 4096,
                "task_scores": _get_task_scores(model_id),
            })
        return models

    async def _discover_huggingface(self) -> list[dict]:
        # HuggingFace Inference API — curated list of popular free text-generation models
        _HF_MODELS = [
            {"model_id": "mistralai/Mistral-7B-Instruct-v0.3", "display_name": "Mistral 7B Instruct v0.3", "context_length": 32768},
            {"model_id": "HuggingFaceH4/zephyr-7b-beta", "display_name": "Zephyr 7B Beta", "context_length": 32768},
            {"model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct", "display_name": "Llama 3.1 8B Instruct", "context_length": 131072},
            {"model_id": "Qwen/Qwen2.5-72B-Instruct", "display_name": "Qwen 2.5 72B Instruct", "context_length": 32768},
            {"model_id": "microsoft/Phi-3.5-mini-instruct", "display_name": "Phi 3.5 Mini Instruct", "context_length": 131072},
        ]
        return [
            {**m, "task_scores": _get_task_scores(m["model_id"])}
            for m in _HF_MODELS
        ]

    async def _discover_agnes(self) -> list[dict]:
        models = await self._discover_openai_models(
            "AGNES_API_KEY", "https://apihub.agnes-ai.com/v1/models",
            default_context=32768, name_key="id",
            filter_fn=lambda m: "video" not in m["id"],
        )
        # Override task scores for image models
        for m in models:
            if "image" in m["model_id"]:
                m["task_scores"] = json.dumps({"image_generation": 0.9})
        return models

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
                    """UPDATE models SET active = 0,
                           deactivated_reason = COALESCE(deactivated_reason, 'no longer listed by provider'),
                           deactivated_at = COALESCE(deactivated_at, ?)
                       WHERE provider_id = ? AND model_id = ?""",
                    (now, provider_id, mid),
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
                       -- A provider keeps listing models that 404 when you call them.
                       -- Discovery must not undo a deactivation the probe made from
                       -- evidence; only the probe clears deactivated_reason.
                       active = CASE WHEN models.deactivated_reason IS NULL
                                     THEN 1 ELSE models.active END""",
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
