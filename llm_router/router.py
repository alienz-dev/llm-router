import asyncio
import json
import re
import logging
import time
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


# Statuses that describe the *request* rather than the model behind it. A
# provider answering "this response_format type is unavailable" is not evidence
# that the model is unhealthy — and recording it as one would evict a working
# model from `auto` after five refusals (0.2x penalty at 3, hard skip at 5).
# They are also not worth retrying elsewhere: the caller's own fallback path is
# what should run, and it only runs if it sees the refusal (ADR-01).
_REQUEST_SHAPED_STATUSES = frozenset({400, 422})

# How many DISTINCT providers one `auto` request may try. Candidates are scored
# per model, and one provider can own every top slot — OpenRouter is 20 of 36
# routable models at the highest priority. Since a failure marks the whole
# provider failed, five OpenRouter candidates meant one attempt and four skips:
# a 429 killed the request without agnes, nvidia or opencode ever being asked.
FALLBACK_PROVIDERS = 3


def required_capabilities(params: dict, stream: bool = False) -> set[str]:
    """What this request needs a model to actually be able to do."""
    required = set()
    if params.get("tools"):
        required.add("tools")
    fmt = params.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        required.add("json_schema")
    if stream:
        required.add("streaming")
    return required


def adapter_gaps(adapter, required: set[str]) -> set[str]:
    """Capabilities the *provider* cannot express. A hard ceiling: per-model
    probe results narrow within it and never widen it (ADR-01)."""
    supports = getattr(adapter, "supports", None) or {}
    return {cap for cap in required if not supports.get(cap, True)}


def model_gaps(flags: dict, required: set[str]) -> set[str]:
    """Capabilities a model was *probed* to lack.

    NULL means unknown, not unsupported — on the day the capability migration
    lands every row is NULL, and treating that as "no" would empty the candidate
    set for every structured request.
    """
    gaps = set()
    for cap in required:
        value = flags.get(cap)
        if value is not None and not value:
            gaps.add(cap)
    return gaps


def _capability_error(missing: set[str], where: str) -> AdapterResponse:
    """ADR-01: when nothing can do the job, say so — never fall through to a
    model that will ignore the request and answer anyway.

    Phrased the way a provider phrases its own refusal ("not supported"), and
    returned as the same 400. When the router refuses on a model's behalf from
    probe data, a caller must be able to handle it exactly as it handles the
    provider saying it directly — otherwise pre-empting the call, which saves a
    request and four seconds, would break the fallback it was meant to trigger.
    """
    caps = ", ".join(sorted(missing))
    return AdapterResponse(
        response={"error": f"{caps} is not supported by {where}. The router does "
                           f"not downgrade a request to make it fit — choose a "
                           f"capable model or handle this refusal."},
        quota=None, tokens_in=0, tokens_out=0, latency_ms=0, status=400,
    )


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
        required = required_capabilities(kwargs, stream)
        candidates = await self._get_ranked_candidates(
            task_type, estimated_tokens, required
        )

        if not candidates:
            if required:
                # Distinguish "nothing can do this" from "nothing has quota" —
                # they need different actions from the caller.
                unfiltered = await self._get_ranked_candidates(task_type, estimated_tokens)
                if unfiltered:
                    return _capability_error(required, "auto routing")
            return AdapterResponse(
                response={"error": "No available models meet quota requirements"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0, status=503,
            )

        last_error = None
        last_status = None
        failed_providers = set()
        for candidate in self._spread_across_providers(candidates):
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
                return self._metered_stream(
                    candidate.provider_id, candidate.model_id,
                    candidate.adapter.stream_completion(
                        messages, candidate.model_id, **kwargs
                    ),
                    messages, kwargs,
                    meta={"provider": candidate.provider_id, "model": candidate.model_id,
                          "task_type": task_type},
                )

            response = await candidate.adapter.chat_completion(
                messages, candidate.model_id, **kwargs
            )

            # Check if adapter returned an error
            if "error" in response.response:
                last_error = response.response["error"]
                last_status = response.status
                if response.status in _REQUEST_SHAPED_STATUSES:
                    # The provider rejected the request, not the model. Hand it
                    # straight back: the caller's fallback keys on seeing it, and
                    # burning four more providers on a request they will also
                    # reject just spends free-tier budget to arrive at the same
                    # answer more slowly.
                    await self._record_capability_refusal(
                        candidate.provider_id, candidate.model_id, kwargs, last_error
                    )
                    return response
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
            # 404, not 502: the caller named something that does not exist, and
            # retrying will not change that. The image path five hundred lines
            # up already got this right.
            return AdapterResponse(
                response={"error": f"Model not found: {model_override}"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0, status=404,
            )

        if provider_id not in self.adapters:
            return AdapterResponse(
                response={"error": f"Unknown provider: {provider_id}"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0, status=404,
            )

        # Capability gate, before anything is spent. A direct route is the path
        # a caller picked deliberately, so a mismatch is worth saying out loud.
        required = required_capabilities(kwargs, stream)
        if required:
            missing = adapter_gaps(self.adapters[provider_id], required)
            missing |= model_gaps(
                await self._model_capability_flags(provider_id, model_id), required
            )
            if missing:
                return _capability_error(missing, f"{provider_id}:{model_id}")

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
            return self._metered_stream(
                provider_id, model_id,
                adapter.stream_completion(messages, model_id, **kwargs),
                messages, kwargs,
                meta={"provider": provider_id, "model": model_id},
            )

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
        elif response.status in _REQUEST_SHAPED_STATUSES:
            # A refusal is not ill health. Five schema refusals would otherwise
            # cross the consecutive-failure threshold and evict a model that
            # works perfectly well for everything else.
            await self._record_capability_refusal(
                provider_id, model_id, kwargs, response.response["error"]
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

        required = required_capabilities(kwargs, stream)
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
            # A LIKE '%family%' match is a loose way to pick a replacement, so
            # this is exactly where an incapable model would otherwise arrive.
            if required and (
                adapter_gaps(adapter, required)
                or model_gaps(
                    await self._model_capability_flags(provider_id, model_id), required
                )
            ):
                continue
            if stream:
                return self._metered_stream(
                    provider_id, model_id,
                    adapter.stream_completion(messages, model_id, **kwargs),
                    messages, kwargs,
                    meta={"provider": provider_id, "model": model_id,
                          "task_type": "fallback", "fallback": True,
                          "original_request": original_model},
                )
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

    @staticmethod
    def _spread_across_providers(
        candidates: list[ModelCandidate], limit: int = FALLBACK_PROVIDERS
    ) -> list[ModelCandidate]:
        """The best candidate from each of the first `limit` distinct providers.

        Taking the top N models instead meant the fallback chain could be one
        provider N times over — and one failure retires all of them at once,
        because failure is tracked per provider. Spreading is the difference
        between "OpenRouter is rate limited" ending the request and it moving on.
        """
        chosen: list[ModelCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.provider_id in seen:
                continue
            seen.add(candidate.provider_id)
            chosen.append(candidate)
            if len(chosen) >= limit:
                break
        return chosen

    async def _metered_stream(
        self, provider_id: str, model_id: str, source: AsyncIterator[dict],
        messages: list[dict], params: dict, meta: dict,
    ) -> AsyncIterator[dict]:
        """Pass a provider's stream through, and account for it.

        Streamed requests used to return the adapter's iterator directly: no
        quota recorded, no health updated, no circuit-breaker success. Streamed
        traffic was invisible to the free-tier budgeting this whole design rests
        on, so a streaming client could quietly exhaust a daily cap that the
        router still believed was untouched.
        """
        estimated_in = await self.quota_manager.estimate_tokens(messages)
        started = time.monotonic()
        usage: dict | None = None
        completion_chars = 0
        first = True
        settled = False

        async def _fail(error: str) -> None:
            nonlocal settled
            if settled:
                return
            settled = True
            self.circuit_breaker.record_failure(provider_id, error)
            await self.quota_manager.record_usage(provider_id, model_id, 0, 0, False)
            await self.health_repo.update(
                provider_id, model_id, success=False, error=error
            )

        async def _succeed() -> tuple[int, int]:
            nonlocal settled
            tokens_in = (usage or {}).get("prompt_tokens", estimated_in)
            tokens_out = (usage or {}).get(
                "completion_tokens", max(1, completion_chars // 4)
            )
            if settled:
                return tokens_in, tokens_out
            settled = True
            self.circuit_breaker.record_success(provider_id)
            await self.quota_manager.record_usage(
                provider_id, model_id, tokens_in, tokens_out, True
            )
            await self.health_repo.update(
                provider_id, model_id, success=True,
                latency_ms=(time.monotonic() - started) * 1000,
            )
            return tokens_in, tokens_out

        try:
            async for chunk in source:
                if isinstance(chunk, str):  # adapters that predate the dict contract
                    completion_chars += len(chunk)
                    yield chunk
                    continue
                if "error" in chunk:
                    if chunk.get("status") not in _REQUEST_SHAPED_STATUSES:
                        await _fail(str(chunk["error"]))
                    yield chunk
                    return
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if isinstance(delta.get("content"), str):
                        completion_chars += len(delta["content"])
                    for call in delta.get("tool_calls") or []:
                        completion_chars += len(
                            (call.get("function") or {}).get("arguments") or ""
                        )
                if first:
                    chunk.setdefault("_router", meta)
                    first = False
                yield chunk
        except GeneratorExit:
            # The client hung up. The upstream request was still made and still
            # counted against the free tier, so it has to be booked — but a
            # closing async generator must not await its own cleanup, so hand
            # the write to the loop and let the close finish.
            asyncio.get_running_loop().create_task(_succeed())
            raise
        except Exception as e:
            from .redact import redact_error

            error = redact_error(e) or type(e).__name__
            await _fail(error)
            yield {"error": error}
            return

        tokens_in, tokens_out = await _succeed()

        wants_usage = bool((params.get("stream_options") or {}).get("include_usage"))
        if wants_usage and usage is None:
            # The caller asked for usage and the provider sent none. Give it the
            # router's own estimate, labelled as one — an unlabelled guess is
            # worse than no number.
            yield {
                "choices": [],
                "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out,
                          "total_tokens": tokens_in + tokens_out},
                "_router": {**meta, "usage_estimated": True},
            }

    async def _model_capability_flags(self, provider_id: str, model_id: str) -> dict:
        """Probed capability flags for one model. NULL stays NULL — unknown."""
        db = await get_db()
        async with db.execute(
            "SELECT supports_tools, supports_json_schema FROM models "
            "WHERE provider_id = ? AND model_id = ?",
            (provider_id, model_id),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return {}
        return {"tools": row[0], "json_schema": row[1]}

    async def _record_capability_refusal(
        self, provider_id: str, model_id: str, params: dict, error: str
    ) -> None:
        """Narrow a model's capability flags from evidence.

        A 400 naming response_format or tools is the provider telling us
        something the capability probe would have to spend a request to learn.
        Writing it down is what stops `auto` picking the same model for the same
        kind of request tomorrow. Health is deliberately untouched.
        """
        lowered = (error or "").lower()
        updates = {}
        if params.get("response_format") is not None and (
            "response_format" in lowered or "json_schema" in lowered
            or "json schema" in lowered
        ):
            updates["supports_json_schema"] = 0
        if params.get("tools") and (
            "tool" in lowered or "function call" in lowered or "function_call" in lowered
        ):
            updates["supports_tools"] = 0
        if not updates:
            logger.info("Request rejected by %s:%s — %s", provider_id, model_id, error[:200])
            return

        db = await get_db()
        assignments = ", ".join(f"{col} = ?" for col in updates)
        await db.execute(
            f"UPDATE models SET {assignments}, capability_checked_at = ? "
            "WHERE provider_id = ? AND model_id = ?",
            (*updates.values(), datetime.now(timezone.utc).isoformat(),
             provider_id, model_id),
        )
        await db.commit()
        logger.info("Capability narrowed for %s:%s — %s (provider said: %s)",
                    provider_id, model_id, ", ".join(updates), error[:120])

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
        self, task_type: str, estimated_tokens: int,
        required: set[str] | None = None,
    ) -> list[ModelCandidate]:
        db = await get_db()

        async with db.execute("""
            SELECT m.provider_id, m.model_id, m.task_scores,
                   mh.last_probe_ok, mh.last_probe_at, mh.avg_latency_ms,
                   mh.consecutive_failures, mh.success_count, mh.failure_count,
                   m.supports_tools, m.supports_json_schema
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
            if required:
                # Provider ceiling first, then what the model itself was probed
                # to lack. NULL flags mean unknown and stay in the running.
                if adapter_gaps(self.adapters[provider_id], required):
                    continue
                if model_gaps({"tools": row[9], "json_schema": row[10]}, required):
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
                # Bands follow the probe rotation (PROBE_MIN_AGE_HOURS = 24).
                # They used to assume a 2-hourly sweep of every model, which is
                # what made that sweep cost 240 requests a day; against a 24h
                # rotation those bands would have parked every model at 0.1.
                # Real traffic writes last_probe_at too, so a model in use stays
                # at the top band without ever being probed.
                if probe_age_h < 24:
                    freshness = 1.0
                elif probe_age_h < 72:
                    freshness = 0.7
                elif probe_age_h < 24 * 7:
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

