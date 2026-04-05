import asyncio
import json
import random
import time
from typing import AsyncIterator
import httpx
from .base import BaseAdapter, AdapterResponse, QuotaSnapshot

class GroqAdapter(BaseAdapter):
    provider_name = "groq"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.client = httpx.AsyncClient(timeout=30)
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
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
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
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
            tokens_out=usage.get("completion_tokens", len(data.get("choices", [{}])[0].get("message", {}).get("content", "")) // 4),
            latency_ms=(time.time() - start_time) * 1000
        )
    
    async def stream_completion(self, messages: list[dict], model_id: str, **kwargs) -> AsyncIterator[str]:
        try:
            async with self.client.stream(
                "POST", f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
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
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception:
            return []

    async def close(self):
        await self.client.aclose()