import asyncio
import json
import random
import time
from typing import AsyncIterator
import httpx
from .base import BaseAdapter, AdapterResponse, QuotaSnapshot

class CloudflareAdapter(BaseAdapter):
    provider_name = "cloudflare"
    
    def __init__(self, account_id: str, api_token: str):
        self.account_id = account_id
        self.api_token = api_token
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"
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
        # Cloudflare doesn't provide quota headers, return None for local tracking
        return None
    
    def _convert_to_cf_format(self, messages: list[dict]) -> dict:
        # Cloudflare expects different format
        return {"messages": messages}
    
    def _convert_from_cf_format(self, response: dict) -> dict:
        # Convert Cloudflare response to OpenAI format
        if "result" in response:
            result = response["result"]
            if isinstance(result, dict) and "response" in result:
                content = result["response"]
            elif isinstance(result, str):
                content = result
            else:
                content = str(result)
        else:
            content = str(response)
        
        return {
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(str(content)) // 4,
                "completion_tokens": len(content) // 4,
                "total_tokens": len(content) // 2
            }
        }
    
    async def chat_completion(self, messages: list[dict], model_id: str, **kwargs) -> AdapterResponse:
        start_time = time.time()
        
        async def _make_request():
            cf_payload = self._convert_to_cf_format(messages)
            response = await self.client.post(
                f"{self.base_url}/@cf/{model_id}",
                headers={"Authorization": f"Bearer {self.api_token}"},
                json=cf_payload,
            )
            response.raise_for_status()
            return response
        
        result = await self._retry_with_backoff(_make_request)
        if isinstance(result, dict) and "error" in result:
            return AdapterResponse(result, None, 0, 0, (time.time() - start_time) * 1000)
        
        cf_data = result.json()
        openai_data = self._convert_from_cf_format(cf_data)
        usage = openai_data.get("usage", {})
        
        return AdapterResponse(
            response=openai_data,
            quota=self._parse_quota(result.headers),
            tokens_in=usage.get("prompt_tokens", len(str(messages)) // 4),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=(time.time() - start_time) * 1000
        )
    
    async def stream_completion(self, messages: list[dict], model_id: str, **kwargs) -> AsyncIterator[str]:
        # Cloudflare doesn't support streaming - fallback to non-streaming
        result = await self.chat_completion(messages, model_id, **kwargs)
        if "error" in result.response:
            yield str(result.response.get("error", "Unknown error"))
        else:
            content = result.response.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                yield content
    
    async def list_models(self) -> list[dict]:
        # Cloudflare models are predefined - return common ones
        return [
            {"id": "llama-2-7b-chat-fp16", "object": "model"},
            {"id": "mistral-7b-instruct-v0.1", "object": "model"},
            {"id": "codellama-7b-instruct-awq", "object": "model"}
        ]

    async def close(self):
        await self.client.aclose()