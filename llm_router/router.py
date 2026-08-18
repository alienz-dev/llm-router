import json
import re
import logging
from datetime import datetime, timezone
from typing import AsyncIterator
from dataclasses import dataclass

from .adapters.base import AdapterResponse, BaseAdapter
from .quota import QuotaManager
from .db import get_db
from .config import get_config
from .circuit_breaker import CircuitBreakerManager
from .health import ModelHealthRepository

logger = logging.getLogger(__name__)


@dataclass
class ModelCandidate:
    provider_id: str
    model_id: str
    adapter: BaseAdapter
    capability_score: float
    quota_headroom_pct: float
    final_score: float


class SmartRouter:
    def __init__(
        self,
        adapters: dict[str, BaseAdapter],
        quota_manager: QuotaManager,
        health_repo: ModelHealthRepository | None = None,
    ):
        self.adapters = adapters
        self.quota_manager = quota_manager
        self.circuit_breaker = CircuitBreakerManager()
        self.health_repo = health_repo or ModelHealthRepository()

    async def route(
        self, messages: list[dict], model_override: str | None = None,
        stream: bool = False, **kwargs
    ) -> AdapterResponse | AsyncIterator[str]:
        if model_override and model_override.strip().lower() not in ("auto", ""):
            return await self._route_direct(messages, model_override, stream, **kwargs)

        # No model override or "auto" — use smart routing

        task_type = self.classify_task(messages)
        estimated_tokens = await self.quota_manager.estimate_tokens(messages)
        candidates = await self._get_ranked_candidates(task_type, estimated_tokens)

        if not candidates:
            return AdapterResponse(
                response={"error": "No available models meet quota requirements"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0, status=503,
            )

        last_error = None
        last_status = None
        failed_providers = set()
        for candidate in candidates[:5]:
            # Skip providers that already failed in this request
            if candidate.provider_id in failed_providers:
                continue
            # Skip providers with open circuit breaker
            if not self.circuit_breaker.is_available(candidate.provider_id):
                logger.debug("Skipping %s — circuit breaker %s",
                    candidate.provider_id,
                    self.circuit_breaker.get_breaker(candidate.provider_id).state.value)
                continue

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
                last_status = response.status
                failed_providers.add(candidate.provider_id)
                # Record to circuit breaker — may trip it open
                tripped = self.circuit_breaker.record_failure(
                    candidate.provider_id, last_error
                )
                if tripped:
                    logger.warning("Circuit breaker OPEN for %s: %s",
                        candidate.provider_id, last_error[:100])
                await self.quota_manager.record_usage(
                    candidate.provider_id, candidate.model_id, 0, 0, False
                )
                # Update model health with failure
                await self.health_repo.update(
                    candidate.provider_id, candidate.model_id,
                    success=False, error=last_error,
                )
                continue

            # Success — record to circuit breaker (may close it if half-open)
            self.circuit_breaker.record_success(candidate.provider_id)
            await self.quota_manager.record_usage(
                candidate.provider_id, candidate.model_id,
                response.tokens_in, response.tokens_out, True,
            )
            # Update model health with success + latency
            await self.health_repo.update(
                candidate.provider_id, candidate.model_id,
                success=True, latency_ms=response.latency_ms,
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
                "fallback": len(failed_providers) > 0,
            })
            return response

        return AdapterResponse(
            response={"error": f"All fallback attempts failed: {last_error}"},
            quota=None, tokens_in=0, tokens_out=0, latency_ms=0, status=last_status,
        )

    async def route_image(
        self, prompt: str, model_override: str | None = None, **kwargs
    ) -> AdapterResponse:
        """Route an image generation request to an image-capable provider."""
        if model_override and model_override.strip().lower() not in ("auto", ""):
            return await self._route_image_direct(prompt, model_override, **kwargs)

        # Smart routing — find image-capable models
        candidates = await self._get_image_candidates()
        if not candidates:
            return AdapterResponse(
                response={"error": "No available image generation models"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0,
            )

        last_error = None
        for candidate in candidates[:3]:
            if not self.circuit_breaker.is_available(candidate.provider_id):
                continue

            response = await candidate.adapter.generate_image(
                prompt, candidate.model_id, **kwargs
            )

            if "error" in response.response:
                last_error = response.response["error"]
                self.circuit_breaker.record_failure(candidate.provider_id, last_error)
                await self.health_repo.update(
                    candidate.provider_id, candidate.model_id,
                    success=False, error=last_error,
                )
                continue

            self.circuit_breaker.record_success(candidate.provider_id)
            await self.health_repo.update(
                candidate.provider_id, candidate.model_id,
                success=True, latency_ms=response.latency_ms,
            )
            response.response.setdefault("_router", {
                "provider": candidate.provider_id,
                "model": candidate.model_id,
            })
            return response

        return AdapterResponse(
            response={"error": f"All image generation attempts failed: {last_error}"},
            quota=None, tokens_in=0, tokens_out=0, latency_ms=0,
        )

    async def _route_image_direct(
        self, prompt: str, model_override: str, **kwargs
    ) -> AdapterResponse:
        """Direct route for image generation with a specific model."""
        db = await get_db()
        async with db.execute(
            "SELECT provider_id FROM models WHERE model_id = ? AND active = 1 LIMIT 1",
            (model_override,),
        ) as cur:
            row = await cur.fetchone()
        if row:
            provider_id = row[0]
            model_id = model_override
        elif ":" in model_override:
            provider_id, model_id = model_override.split(":", 1)
        else:
            return AdapterResponse(
                response={"error": f"Model not found: {model_override}"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0, status=404,
            )

        if provider_id not in self.adapters:
            return AdapterResponse(
                response={"error": f"Unknown provider: {provider_id}"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0, status=404,
            )

        if not self.circuit_breaker.is_available(provider_id):
            return AdapterResponse(
                response={"error": f"Provider {provider_id} unavailable (circuit breaker)"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0,
            )

        adapter = self.adapters[provider_id]
        response = await adapter.generate_image(prompt, model_id, **kwargs)

        if "error" not in response.response:
            self.circuit_breaker.record_success(provider_id)
            await self.health_repo.update(
                provider_id, model_id, success=True, latency_ms=response.latency_ms,
            )
        else:
            self.circuit_breaker.record_failure(provider_id, response.response["error"])
            await self.health_repo.update(
                provider_id, model_id, success=False, error=response.response["error"],
            )
        return response

    async def _get_image_candidates(self) -> list[ModelCandidate]:
        """Get image-capable models ranked by score."""
        db = await get_db()
        async with db.execute("""
            SELECT m.provider_id, m.model_id, m.task_scores,
                   mh.success_count, mh.failure_count, mh.avg_latency_ms
            FROM models m
            JOIN providers p ON m.provider_id = p.id
            LEFT JOIN model_health mh ON m.provider_id = mh.provider_id
                                      AND m.model_id = mh.model_id
            WHERE m.active = 1 AND p.enabled = 1
        """) as cursor:
            rows = await cursor.fetchall()

        candidates = []
        for row in rows:
            provider_id, model_id, task_scores_raw = row[0], row[1], row[2]
            suc_count, fail_count, avg_latency = row[3], row[4], row[5]

            if provider_id not in self.adapters:
                continue

            try:
                task_scores = json.loads(task_scores_raw) if task_scores_raw else {}
            except (json.JSONDecodeError, TypeError):
                task_scores = {}

            img_capability = task_scores.get("image_generation", 0)
            if img_capability <= 0:
                continue

            # Success rate
            total = (suc_count or 0) + (fail_count or 0)
            success_rate = (suc_count or 0) / total if total > 0 else 0.5

            # Latency penalty
            latency_mult = 0.5 if avg_latency and avg_latency > 30000 else 1.0

            score = img_capability * success_rate * latency_mult

            cfg = get_config()
            provider_cfg = cfg.providers.get(provider_id)
            score *= provider_cfg.priority if provider_cfg else 1.0

            candidates.append(ModelCandidate(
                provider_id=provider_id,
                model_id=model_id,
                adapter=self.adapters[provider_id],
                capability_score=img_capability,
                quota_headroom_pct=0,
                final_score=score,
            ))

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates

    async def _route_direct(
        self, messages: list[dict], model_override: str, stream: bool, **kwargs
    ) -> AdapterResponse | AsyncIterator[str]:
        # Always check DB first (handles model names with : like OpenRouter's model:free)
        db = await get_db()
        async with db.execute(
            "SELECT provider_id FROM models WHERE model_id = ? AND active = 1 LIMIT 1",
            (model_override,),
        ) as cur:
            row = await cur.fetchone()
        if row:
            provider_id = row[0]
            model_id = model_override
        elif ":" in model_override:
            # Fall back to provider:model parsing
            provider_id, model_id = model_override.split(":", 1)
        else:
            return AdapterResponse(
                response={"error": f"Model not found: {model_override}"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0,
            )

        if provider_id not in self.adapters:
            return AdapterResponse(
                response={"error": f"Unknown provider: {provider_id}"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0,
            )

        # Check circuit breaker — if open, try fallback to other providers with same model family
        if not self.circuit_breaker.is_available(provider_id):
            breaker = self.circuit_breaker.get_breaker(provider_id)
            logger.info("Direct route blocked by circuit breaker (%s: %s), attempting fallback",
                provider_id, breaker.state.value)
            fallback = await self._try_fallback_for_model(
                messages, model_override, provider_id, stream, **kwargs
            )
            if fallback:
                return fallback
            return AdapterResponse(
                response={"error": f"Provider {provider_id} unavailable (circuit breaker {breaker.state.value}): {breaker.last_error}"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0, status=503,
            )

        estimated_tokens = await self.quota_manager.estimate_tokens(messages)
        if not await self.quota_manager.can_use(provider_id, model_id, estimated_tokens):
            return AdapterResponse(
                response={"error": f"Quota exceeded for {provider_id}:{model_id}"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0, status=429,
            )

        adapter = self.adapters[provider_id]
        if stream:
            return adapter.stream_completion(messages, model_id, **kwargs)

        response = await adapter.chat_completion(messages, model_id, **kwargs)
        if "error" not in response.response:
            self.circuit_breaker.record_success(provider_id)
            await self.quota_manager.record_usage(
                provider_id, model_id, response.tokens_in, response.tokens_out, True
            )
            if response.quota:
                await self.quota_manager.update_limits(provider_id, model_id, response.quota)
            await self.health_repo.update(
                provider_id, model_id, success=True, latency_ms=response.latency_ms,
            )
        else:
            # Record failure to circuit breaker
            self.circuit_breaker.record_failure(provider_id, response.response["error"])
            await self.health_repo.update(
                provider_id, model_id, success=False, error=response.response["error"],
            )
        return response

    async def _try_fallback_for_model(
        self, messages: list[dict], original_model: str, blocked_provider: str,
        stream: bool, **kwargs
    ) -> AdapterResponse | None:
        """When a direct-route provider is circuit-broken, find alternative models."""
        # Extract model family (e.g., "gemma" from "google/gemma-3-4b-it:free")
        model_base = original_model.split("/")[-1].split(":")[0].lower()
        # Get first 2 meaningful segments (e.g., "gemma-3" or "gemini-2.0")
        parts = model_base.split("-")
        family = parts[0] if parts else model_base

        db = await get_db()
        async with db.execute("""
            SELECT m.provider_id, m.model_id
            FROM models m
            JOIN providers p ON m.provider_id = p.id
            WHERE m.active = 1 AND p.enabled = 1
              AND m.provider_id != ?
              AND LOWER(m.model_id) LIKE ?
            ORDER BY m.model_id
            LIMIT 3
        """, (blocked_provider, f"%{family}%")) as cur:
            rows = await cur.fetchall()

        for provider_id, model_id in rows:
            if not self.circuit_breaker.is_available(provider_id):
                continue
            if not await self.quota_manager.can_use(
                provider_id, model_id,
                await self.quota_manager.estimate_tokens(messages)
            ):
                continue
            adapter = self.adapters.get(provider_id)
            if not adapter:
                continue
            if stream:
                return adapter.stream_completion(messages, model_id, **kwargs)
            response = await adapter.chat_completion(messages, model_id, **kwargs)
            if "error" not in response.response:
                self.circuit_breaker.record_success(provider_id)
                response.response.setdefault("_router", {
                    "provider": provider_id,
                    "model": model_id,
                    "task_type": "fallback",
                    "fallback": True,
                    "original_request": original_model,
                })
                return response
        return None

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
            SELECT m.provider_id, m.model_id, m.task_scores,
                   mh.last_probe_ok, mh.last_probe_at, mh.avg_latency_ms,
                   mh.consecutive_failures, mh.success_count, mh.failure_count
            FROM models m
            JOIN providers p ON m.provider_id = p.id
            LEFT JOIN model_health mh ON m.provider_id = mh.provider_id
                                      AND m.model_id = mh.model_id
            WHERE m.active = 1 AND p.enabled = 1
        """) as cursor:
            rows = await cursor.fetchall()

        # Build quota headroom map
        quota_statuses = await self.quota_manager.get_all_quota_status()
        headroom_map = {s["provider_id"]: s["quota_remaining_pct"] for s in quota_statuses}

        now = datetime.now(timezone.utc)
        candidates = []
        for row in rows:
            provider_id, model_id, task_scores_raw = row[0], row[1], row[2]
            last_probe_ok, last_probe_at, avg_latency = row[3], row[4], row[5]
            consec_failures, suc_count, fail_count = row[6], row[7], row[8]

            if provider_id not in self.adapters:
                continue
            if not await self.quota_manager.can_use(provider_id, model_id, estimated_tokens):
                continue

            # Hard skip: 5+ consecutive probe failures
            if consec_failures and consec_failures >= 5:
                continue

            try:
                task_scores = json.loads(task_scores_raw) if task_scores_raw else {}
            except (json.JSONDecodeError, TypeError):
                task_scores = {}

            capability = task_scores.get(task_type, 0.5)
            headroom = headroom_map.get(provider_id, 0.5)

            # --- Availability multiplier ---
            # Freshness: how recently was the model probed?
            if last_probe_at:
                try:
                    probe_time = datetime.fromisoformat(last_probe_at)
                    if probe_time.tzinfo is None:
                        probe_time = probe_time.replace(tzinfo=timezone.utc)
                    probe_age_h = (now - probe_time).total_seconds() / 3600
                except (ValueError, TypeError):
                    probe_age_h = 999
                if probe_age_h < 2:
                    freshness = 1.0
                elif probe_age_h < 6:
                    freshness = 0.7
                elif probe_age_h < 24:
                    freshness = 0.4
                else:
                    freshness = 0.1
            else:
                freshness = 0.2  # never probed — allow but low priority

            # Success rate
            total = (suc_count or 0) + (fail_count or 0)
            if total > 0:
                success_rate = max(0.1, min(1.0, (suc_count or 0) / total))
            else:
                success_rate = 0.5  # unknown — neutral

            # Consecutive failure penalty (soft circuit break)
            if consec_failures and consec_failures >= 3:
                success_rate *= 0.2

            # Latency penalty
            if avg_latency and avg_latency > 10000:
                latency_mult = 0.5
            elif avg_latency and avg_latency > 5000:
                latency_mult = 0.8
            else:
                latency_mult = 1.0

            availability = freshness * success_rate * latency_mult

            score = capability * (headroom + 0.1) * availability

            # Apply provider priority multiplier
            cfg = get_config()
            provider_mult = cfg.providers.get(provider_id, None)
            provider_mult = provider_mult.priority if provider_mult else 1.0
            score *= provider_mult

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

