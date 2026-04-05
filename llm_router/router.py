import json
import re
from typing import AsyncIterator
from dataclasses import dataclass

from .adapters.base import AdapterResponse, BaseAdapter
from .quota import QuotaManager
from .db import get_db


@dataclass
class ModelCandidate:
    provider_id: str
    model_id: str
    adapter: BaseAdapter
    capability_score: float
    quota_headroom_pct: float
    final_score: float


class SmartRouter:
    def __init__(self, adapters: dict[str, BaseAdapter], quota_manager: QuotaManager):
        self.adapters = adapters
        self.quota_manager = quota_manager

    async def route(
        self, messages: list[dict], model_override: str | None = None,
        stream: bool = False, **kwargs
    ) -> AdapterResponse | AsyncIterator[str]:
        if model_override:
            return await self._route_direct(messages, model_override, stream, **kwargs)

        task_type = self.classify_task(messages)
        estimated_tokens = await self.quota_manager.estimate_tokens(messages)
        candidates = await self._get_ranked_candidates(task_type, estimated_tokens)

        if not candidates:
            return AdapterResponse(
                response={"error": "No available models meet quota requirements"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0,
            )

        last_error = None
        for candidate in candidates[:3]:
            if stream:
                return candidate.adapter.stream_completion(
                    messages, candidate.model_id, **kwargs
                )

            response = await candidate.adapter.chat_completion(
                messages, candidate.model_id, **kwargs
            )

            # Check if adapter returned an error
            if "error" in response.response:
                last_error = response.response["error"]
                await self.quota_manager.record_usage(
                    candidate.provider_id, candidate.model_id, 0, 0, False
                )
                continue

            await self.quota_manager.record_usage(
                candidate.provider_id, candidate.model_id,
                response.tokens_in, response.tokens_out, True,
            )
            if response.quota:
                await self.quota_manager.update_limits(
                    candidate.provider_id, candidate.model_id, response.quota
                )

            # Tag which provider/model was used
            response.response.setdefault("_router", {
                "provider": candidate.provider_id,
                "model": candidate.model_id,
                "task_type": task_type,
            })
            return response

        return AdapterResponse(
            response={"error": f"All fallback attempts failed: {last_error}"},
            quota=None, tokens_in=0, tokens_out=0, latency_ms=0,
        )

    async def _route_direct(
        self, messages: list[dict], model_override: str, stream: bool, **kwargs
    ) -> AdapterResponse | AsyncIterator[str]:
        if ":" in model_override:
            provider_id, model_id = model_override.split(":", 1)
        else:
            # Try to find model in DB
            db = await get_db()
            async with db.execute(
                "SELECT provider_id FROM models WHERE model_id = ? AND active = 1 LIMIT 1",
                (model_override,),
            ) as cur:
                row = await cur.fetchone()
            if not row:
                return AdapterResponse(
                    response={"error": f"Model not found: {model_override}"},
                    quota=None, tokens_in=0, tokens_out=0, latency_ms=0,
                )
            provider_id = row[0]
            model_id = model_override

        if provider_id not in self.adapters:
            return AdapterResponse(
                response={"error": f"Unknown provider: {provider_id}"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0,
            )

        estimated_tokens = await self.quota_manager.estimate_tokens(messages)
        if not await self.quota_manager.can_use(provider_id, model_id, estimated_tokens):
            return AdapterResponse(
                response={"error": f"Quota exceeded for {provider_id}:{model_id}"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0,
            )

        adapter = self.adapters[provider_id]
        if stream:
            return adapter.stream_completion(messages, model_id, **kwargs)

        response = await adapter.chat_completion(messages, model_id, **kwargs)
        if "error" not in response.response:
            await self.quota_manager.record_usage(
                provider_id, model_id, response.tokens_in, response.tokens_out, True
            )
            if response.quota:
                await self.quota_manager.update_limits(provider_id, model_id, response.quota)
        return response

    def classify_task(self, messages: list[dict]) -> str:
        text = " ".join(
            msg.get("content", "") for msg in messages if isinstance(msg.get("content"), str)
        ).lower()

        code_patterns = [
            r"\bfunction\b", r"\bimplement\b", r"\bdebug\b", r"\brefactor\b",
            r"```", r"\bdef\s+\w+", r"\bclass\s+\w+",
        ]
        reasoning_patterns = [
            r"\banalyze\b", r"\bcompare\b", r"\bexplain why\b",
            r"\bstep by step\b", r"\bevaluate\b",
        ]
        summarize_patterns = [
            r"\bsummarize\b", r"\btldr\b", r"\bkey points\b", r"\bbrief\b",
        ]

        for p in code_patterns:
            if re.search(p, text):
                return "code"
        for p in reasoning_patterns:
            if re.search(p, text):
                return "reasoning"
        for p in summarize_patterns:
            if re.search(p, text):
                return "summarize"
        return "general"

    async def _get_ranked_candidates(
        self, task_type: str, estimated_tokens: int
    ) -> list[ModelCandidate]:
        db = await get_db()

        async with db.execute("""
            SELECT m.provider_id, m.model_id, m.task_scores
            FROM models m
            JOIN providers p ON m.provider_id = p.id
            WHERE m.active = 1 AND p.enabled = 1
        """) as cursor:
            rows = await cursor.fetchall()

        # Build quota headroom map
        quota_statuses = await self.quota_manager.get_all_quota_status()
        headroom_map = {s["provider_id"]: s["quota_remaining_pct"] for s in quota_statuses}

        candidates = []
        for provider_id, model_id, task_scores_raw in rows:
            if provider_id not in self.adapters:
                continue
            if not await self.quota_manager.can_use(provider_id, model_id, estimated_tokens):
                continue

            try:
                task_scores = json.loads(task_scores_raw) if task_scores_raw else {}
            except (json.JSONDecodeError, TypeError):
                task_scores = {}

            capability = task_scores.get(task_type, 0.5)
            headroom = headroom_map.get(provider_id, 0.5)
            score = capability * (headroom + 0.1)

            # Groq penalty for long prompts
            if provider_id == "groq":
                tpm_limit = await self._get_tpm_limit(provider_id, model_id)
                if tpm_limit and estimated_tokens > tpm_limit * 0.3:
                    score *= min(1.0, 0.5 * tpm_limit / estimated_tokens)

            candidates.append(ModelCandidate(
                provider_id=provider_id,
                model_id=model_id,
                adapter=self.adapters[provider_id],
                capability_score=capability,
                quota_headroom_pct=headroom,
                final_score=score,
            ))

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates

    async def _get_tpm_limit(self, provider_id: str, model_id: str) -> int | None:
        db = await get_db()
        async with db.execute(
            "SELECT tpm FROM quota_limits WHERE provider_id = ? AND model_id = ?",
            (provider_id, model_id),
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else None
