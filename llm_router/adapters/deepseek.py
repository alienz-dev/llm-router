from .base import OpenAICompatibleAdapter, AdapterResponse


class DeepSeekAdapter(OpenAICompatibleAdapter):
    provider_name = "deepseek"

    def __init__(self, api_key: str):
        super().__init__(api_key, base_url="https://api.deepseek.com/v1", timeout=60)

    async def chat_completion(self, messages, model_id, **kwargs):
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4096
        result = await super().chat_completion(messages, model_id, **kwargs)
        # Handle reasoning_content field (DeepSeek reasoning models)
        if "error" not in result.response:
            choice = result.response.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")
            if reasoning and not content:
                msg["content"] = reasoning
        return result

    async def stream_completion(self, messages, model_id, **kwargs):
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4096
        async for chunk in super().stream_completion(messages, model_id, **kwargs):
            yield chunk
