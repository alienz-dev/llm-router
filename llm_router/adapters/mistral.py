import asyncio
import json
import random
import time
from typing import AsyncIterator
import httpx
from .base import BaseAdapter, AdapterResponse, QuotaSnapshot

class MistralAdapter(BaseAdapter):
    provider_name = "mistral"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1"
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        for attempt in range(3):
            try:
                return await func(*args, **kwargs)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < 2:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait)
                    continue
                return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return {"error": str(e)}
    
    def _parse_quota(self, headers) -> QuotaSnapshot:
        return QuotaSnapshot(
            rpm_limit=int(headers.get("x-ratelimit-limit-requests", 0)) or None,
            rpm_remaining=int(headers.get("x-ratelimit-remaining-requests", 0)) or None,
            tpm_limit=int(headers.get("x-ratelimit-limit-tokens", 0)) or None,
            tpm_remaining=int(headers.get("x-ratelimit-remaining-tokens", 0)) or None,
            reset_seconds=float(headers.get("x-ratelimit-reset-requests", 0)) or None
        )
    
    async def chat_completion(self, messages: list[dict], model_id: str, **kwargs) -> AdapterResponse:
        start_time = time.time()
        
        async def _make_request():
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"messages": messages, "model": model_id, **kwargs},
                    timeout=30
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
            tokens_out=usage.get("completion_tokens", len(data.get("choices", [{}])[0].get("message", {}).get("content", "")) // 4),
            latency_ms=(time.time() - start_time) * 1000
        )
    
    async def stream_completion(self, messages: list[dict], model_id: str, **kwargs) -> AsyncIterator[str]:
        async def _make_request():
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"messages": messages, "model": model_id, "stream": True, **kwargs},
                    timeout=30
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            yield line[6:]
        
        try:
            async for chunk in _make_request():
                yield chunk
        except Exception as e:
            yield json.dumps({"error": str(e)})
    
    async def list_models(self) -> list[dict]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10
                )
                response.raise_for_status()
                return response.json().get("data", [])
        except Exception:
            return []