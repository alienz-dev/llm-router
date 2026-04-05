import asyncio
import json
import random
import time
from typing import AsyncIterator
import httpx
from .base import BaseAdapter, AdapterResponse, QuotaSnapshot

class GoogleAdapter(BaseAdapter):
    provider_name = "google"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
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
        # Google doesn't provide quota in headers, return None for local tracking
        return None
    
    def _convert_to_google_format(self, messages: list[dict]) -> dict:
        system_parts = []
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append({"text": msg["content"]})
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        payload = {"contents": contents}
        if system_parts:
            payload["system_instruction"] = {"parts": system_parts}
        return payload
    
    def _convert_from_google_format(self, response: dict) -> dict:
        candidates = response.get("candidates", [])
        if not candidates:
            return {"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        
        content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        usage = response.get("usageMetadata", {})
        
        return {
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0)
            }
        }
    
    async def chat_completion(self, messages: list[dict], model_id: str, **kwargs) -> AdapterResponse:
        start_time = time.time()
        
        async def _make_request():
            google_payload = self._convert_to_google_format(messages)
            response = await self.client.post(
                f"{self.base_url}/models/{model_id}:generateContent?key={self.api_key}",
                json=google_payload,
            )
            response.raise_for_status()
            return response
        
        result = await self._retry_with_backoff(_make_request)
        if isinstance(result, dict) and "error" in result:
            return AdapterResponse(result, None, 0, 0, (time.time() - start_time) * 1000)
        
        google_data = result.json()
        openai_data = self._convert_from_google_format(google_data)
        usage = openai_data.get("usage", {})
        
        return AdapterResponse(
            response=openai_data,
            quota=self._parse_quota(result.headers),
            tokens_in=usage.get("prompt_tokens", len(str(messages)) // 4),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=(time.time() - start_time) * 1000
        )
    
    async def stream_completion(self, messages: list[dict], model_id: str, **kwargs) -> AsyncIterator[str]:
        # Google streaming requires different endpoint - simplified non-streaming for now
        result = await self.chat_completion(messages, model_id, **kwargs)
        if "error" in result.response:
            yield str(result.response.get("error", "Unknown error"))
        else:
            content = result.response.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                yield content
    
    async def list_models(self) -> list[dict]:
        try:
            response = await self.client.get(
                f"{self.base_url}/models?key={self.api_key}",
            )
            response.raise_for_status()
            models = response.json().get("models", [])
            return [{"id": m["name"].split("/")[-1], "object": "model"} for m in models]
        except Exception:
            return []

    async def close(self):
        await self.client.aclose()