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


class BaseAdapter(ABC):
    provider_name: str

    @abstractmethod
    async def chat_completion(
        self, messages: list[dict], model_id: str, **kwargs
    ) -> AdapterResponse: ...

    @abstractmethod
    async def stream_completion(
        self, messages: list[dict], model_id: str, **kwargs
    ) -> AsyncIterator[str]: ...

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

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry with exponential backoff on 429/503 and connection errors."""
        for attempt in range(3):
            try:
                return await func(*args, **kwargs)
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 503) and attempt < 2:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait)
                    continue
                return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt < 2:
                    await asyncio.sleep((2 ** attempt) + random.uniform(0, 1))
                    continue
                return {"error": str(e)}
            except Exception as e:
                return {"error": str(e)}

    async def chat_completion(
        self, messages: list[dict], model_id: str, **kwargs
    ) -> AdapterResponse:
        # Pre-flight check
        pre = self._pre_request(model_id)
        if pre is not None:
            return pre

        start_time = time.time()

        async def _make_request():
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=self._auth_headers(),
                json={"messages": messages, "model": model_id, **kwargs},
            )
            response.raise_for_status()
            return response

        result = await self._retry_with_backoff(_make_request)
        if isinstance(result, dict) and "error" in result:
            return AdapterResponse(result, None, 0, 0, (time.time() - start_time) * 1000)

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
    ) -> AsyncIterator[str]:
        # Pre-flight check
        pre = self._pre_request(model_id)
        if pre is not None:
            yield pre.response.get("error", "blocked")
            return

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._auth_headers(),
                json={"messages": messages, "model": model_id, "stream": True, **kwargs},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line.strip() != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield str(e)

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