from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator
from abc import ABC, abstractmethod

import httpx

from ..redact import redact, redact_error

logger = logging.getLogger(__name__)


@dataclass
class QuotaSnapshot:
    rpm_limit: int | None = None
    rpm_remaining: int | None = None
    rpd_limit: int | None = None
    rpd_remaining: int | None = None
    tpm_limit: int | None = None
    tpm_remaining: int | None = None
    tpd_limit: int | None = None
    tpd_remaining: int | None = None
    reset_seconds: float | None = None


@dataclass
class AdapterResponse:
    response: dict[str, Any]          # OpenAI-format response dict
    quota: QuotaSnapshot | None       # Parsed from response headers
    tokens_in: int                     # From response usage or estimated
    tokens_out: int                    # From response usage or estimated
    latency_ms: float
    # Upstream HTTP status when this is an error. Callers distinguish "the
    # provider rejected this request" (4xx, act on it now) from "the provider
    # broke" (5xx / None, retry is reasonable). Without it every failure looks
    # like a 502 and well-behaved clients retry a request that will never work.
    status: int | None = None


def content_chunk(content: str, finish_reason: str | None = None) -> dict:
    """A minimal OpenAI streaming chunk, for providers that have no stream of
    their own and hand us the whole answer at once."""
    return {"choices": [{"index": 0, "delta": {"content": content},
                         "finish_reason": finish_reason}]}


class BaseAdapter(ABC):
    provider_name: str

    # What this adapter can express on the wire, regardless of the model behind
    # it. A provider-level "no" is a hard ceiling: per-model probe results narrow
    # within it and never widen it (ADR-01). A model that calls tools happily
    # through OpenRouter does not gain tools through an adapter that flattens
    # every message into a prompt string.
    supports: dict[str, bool] = {"tools": True, "json_schema": True, "streaming": True}

    # The router decided these. A caller-supplied `model` that reached the
    # payload would re-target the upstream to something never scored, never
    # quota-checked and never recorded.
    RESERVED_PARAMS = frozenset({"messages", "model", "stream"})

    def _allowed_params(self) -> set[str]:
        """Caller parameters this provider is allowed to receive.

        Some providers 400 on unknown fields, so forwarding everything blindly
        turns requests that worked yesterday into failures — and those failures
        would land on the model's health score.
        """
        from ..config import get_config

        cfg = get_config()
        provider = cfg.providers.get(self.provider_name)
        if provider is not None and provider.passthrough_params is not None:
            return set(provider.passthrough_params)
        return set(cfg.routing.passthrough_params)

    def _clean_params(self, kwargs: dict) -> dict:
        """Strip what the caller must not set and what this provider will not take."""
        allowed = self._allowed_params()
        clean = {}
        for key, value in kwargs.items():
            if key in self.RESERVED_PARAMS:
                logger.warning(
                    "%s: dropping reserved parameter %r from a caller request",
                    self.provider_name, key,
                )
                continue
            if key in allowed:
                clean[key] = value
        return clean

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry with exponential backoff on 429/503 and connection errors.

        Every error string is redacted here, at the point of capture: these end
        up in logs, in `model_health.last_probe_error`, and in the body the
        caller receives, and a provider that echoes our Authorization header (or
        a Google URL carrying `?key=`) would otherwise publish it.
        """
        for attempt in range(3):
            try:
                return await func(*args, **kwargs)
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 503) and attempt < 2:
                    await asyncio.sleep((2 ** attempt) + random.uniform(0, 1))
                    continue
                return {
                    "error": f"HTTP {e.response.status_code}: {redact(e.response.text)}",
                    "status": e.response.status_code,
                }
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt < 2:
                    await asyncio.sleep((2 ** attempt) + random.uniform(0, 1))
                    continue
                return {"error": redact_error(e)}
            except Exception as e:
                return {"error": redact_error(e)}

    def _error_response(self, result: dict, start_time: float) -> "AdapterResponse":
        """Wrap a failed request, keeping the upstream status off the wire body."""
        status = result.pop("status", None)
        return AdapterResponse(
            result, None, 0, 0, (time.time() - start_time) * 1000, status=status
        )

    @abstractmethod
    async def chat_completion(
        self, messages: list[dict], model_id: str, **kwargs
    ) -> AdapterResponse: ...

    @abstractmethod
    async def stream_completion(
        self, messages: list[dict], model_id: str, **kwargs
    ) -> AsyncIterator[str]: ...

    async def generate_image(self, prompt: str, model_id: str, **kwargs) -> AdapterResponse:
        """Generate an image. Override in subclasses that support it."""
        return AdapterResponse(
            response={"error": f"{self.provider_name} does not support image generation"},
            quota=None, tokens_in=0, tokens_out=0, latency_ms=0,
        )

    @abstractmethod
    async def list_models(self) -> list[dict]: ...

    @abstractmethod
    async def close(self): ...


class OpenAICompatibleAdapter(BaseAdapter):
    """Base class for providers with an OpenAI-compatible /v1/chat/completions endpoint.

    Subclasses only need to set provider_name and base_url. Override _parse_quota()
    for custom rate-limit header parsing. Override _pre_request() for pre-flight checks
    (e.g., blocked models, credit checks).
    """

    provider_name: str = ""
    base_url: str = ""

    def __init__(self, api_key: str, base_url: str | None = None, timeout: int = 30):
        self.api_key = api_key
        if base_url:
            self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=timeout)

    def _auth_headers(self) -> dict[str, str]:
        """Return authorization headers. Override for non-Bearer auth."""
        return {"Authorization": f"Bearer {self.api_key}"}

    def _parse_quota(self, headers: httpx.Headers) -> QuotaSnapshot | None:
        """Parse rate-limit headers into a QuotaSnapshot. Override per provider."""
        return None

    def _pre_request(self, model_id: str) -> AdapterResponse | None:
        """Pre-flight check before making a request. Return an AdapterResponse
        to short-circuit (e.g., blocked model, no credits), or None to proceed."""
        return None

    async def chat_completion(
        self, messages: list[dict], model_id: str, **kwargs
    ) -> AdapterResponse:
        # Pre-flight check
        pre = self._pre_request(model_id)
        if pre is not None:
            return pre

        start_time = time.time()

        params = self._clean_params(kwargs)

        async def _make_request():
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=self._auth_headers(),
                # Router-owned keys last: they must win over anything a caller
                # managed to smuggle through.
                json={**params, "messages": messages, "model": model_id},
            )
            response.raise_for_status()
            return response

        result = await self._retry_with_backoff(_make_request)
        if isinstance(result, dict) and "error" in result:
            return self._error_response(result, start_time)

        data = result.json()
        usage = data.get("usage", {})

        return AdapterResponse(
            response=data,
            quota=self._parse_quota(result.headers),
            tokens_in=usage.get("prompt_tokens", len(str(messages)) // 4),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def stream_completion(
        self, messages: list[dict], model_id: str, **kwargs
    ) -> AsyncIterator[dict]:
        """Yield the provider's chunks verbatim.

        This used to extract `delta.content` and discard the rest, which meant
        tool-call fragments — `index`, `id`, partial `arguments` — could not
        survive the trip, and `usage` never arrived. Errors are yielded as
        `{"error": …}` rather than as assistant content: a stream failure that
        reads like a model answer is worse than one that fails.
        """
        pre = self._pre_request(model_id)
        if pre is not None:
            yield {"error": pre.response.get("error", "blocked"), "status": pre.status}
            return

        params = self._clean_params(kwargs)
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._auth_headers(),
                json={**params, "messages": messages, "model": model_id, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        continue
        except httpx.HTTPStatusError as e:
            try:
                body = (await e.response.aread()).decode("utf-8", "replace")
            except Exception:
                body = ""
            yield {"error": f"HTTP {e.response.status_code}: {redact(body)}",
                   "status": e.response.status_code}
        except Exception as e:
            yield {"error": redact_error(e)}

    async def generate_image(self, prompt: str, model_id: str, **kwargs) -> AdapterResponse:
        """Generate an image via OpenAI-compatible /v1/images/generations endpoint."""
        start_time = time.time()

        async def _make_request():
            payload = {**kwargs, "prompt": prompt, "model": model_id}
            response = await self.client.post(
                f"{self.base_url}/images/generations",
                headers=self._auth_headers(),
                json=payload,
            )
            response.raise_for_status()
            return response

        result = await self._retry_with_backoff(_make_request)
        if isinstance(result, dict) and "error" in result:
            return self._error_response(result, start_time)

        data = result.json()
        return AdapterResponse(
            response=data,
            quota=self._parse_quota(result.headers),
            tokens_in=len(prompt) // 4,
            tokens_out=0,
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def list_models(self) -> list[dict]:
        try:
            response = await self.client.get(
                f"{self.base_url}/models",
                headers=self._auth_headers(),
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception:
            return []

    async def close(self):
        await self.client.aclose()