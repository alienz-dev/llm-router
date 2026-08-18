import json
import time
from typing import AsyncIterator
import httpx
from .base import BaseAdapter, AdapterResponse, QuotaSnapshot

class HuggingFaceAdapter(BaseAdapter):
    provider_name = "huggingface"
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api-inference.huggingface.co/models"
        self.client = httpx.AsyncClient(timeout=30)
    
    def _parse_quota(self, headers) -> QuotaSnapshot:
        # HuggingFace doesn't provide quota headers, return None for local tracking
        return None
    
    def _convert_to_hf_format(self, messages: list[dict]) -> dict:
        # HuggingFace expects different format - combine messages into single input
        combined_text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                combined_text += f"System: {content}\n"
            elif role == "user":
                combined_text += f"User: {content}\n"
            elif role == "assistant":
                combined_text += f"Assistant: {content}\n"
        
        combined_text += "Assistant:"
        return {"inputs": combined_text}
    
    def _convert_from_hf_format(self, response: list | dict) -> dict:
        # Convert HuggingFace response to OpenAI format
        if isinstance(response, list) and len(response) > 0:
            content = response[0].get("generated_text", "")
            # Remove the input prompt from generated text
            if "Assistant:" in content:
                content = content.split("Assistant:")[-1].strip()
        elif isinstance(response, dict):
            content = response.get("generated_text", str(response))
        else:
            content = str(response)
        
        return {
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(content) // 4,
                "completion_tokens": len(content) // 4,
                "total_tokens": len(content) // 2
            }
        }
    
    async def chat_completion(self, messages: list[dict], model_id: str, **kwargs) -> AdapterResponse:
        start_time = time.time()
        
        async def _make_request():
            hf_payload = self._convert_to_hf_format(messages)
            response = await self.client.post(
                f"{self.base_url}/{model_id}",
                headers={"Authorization": f"Bearer {self.api_token}"},
                json=hf_payload,
            )
            response.raise_for_status()
            return response
        
        result = await self._retry_with_backoff(_make_request)
        if isinstance(result, dict) and "error" in result:
            return self._error_response(result, start_time)
        
        hf_data = result.json()
        openai_data = self._convert_from_hf_format(hf_data)
        usage = openai_data.get("usage", {})
        
        return AdapterResponse(
            response=openai_data,
            quota=self._parse_quota(result.headers),
            tokens_in=usage.get("prompt_tokens", len(str(messages)) // 4),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=(time.time() - start_time) * 1000
        )
    
    async def stream_completion(self, messages: list[dict], model_id: str, **kwargs) -> AsyncIterator[str]:
        # HuggingFace doesn't support streaming - fallback to non-streaming
        result = await self.chat_completion(messages, model_id, **kwargs)
        if "error" in result.response:
            yield str(result.response.get("error", "Unknown error"))
        else:
            content = result.response.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                yield content
    
    async def list_models(self) -> list[dict]:
        # HuggingFace has many models - return popular free ones
        return [
            {"id": "microsoft/DialoGPT-medium", "object": "model"},
            {"id": "facebook/blenderbot-400M-distill", "object": "model"},
            {"id": "microsoft/DialoGPT-small", "object": "model"},
            {"id": "gpt2", "object": "model"}
        ]

    async def close(self):
        await self.client.aclose()