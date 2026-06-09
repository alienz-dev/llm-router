from .base import OpenAICompatibleAdapter, AdapterResponse


class NvidiaAdapter(OpenAICompatibleAdapter):
    provider_name = "nvidia"

    def __init__(self, api_key: str):
        super().__init__(api_key, base_url="https://integrate.api.nvidia.com/v1")
        self.credits_remaining = None
        self.auto_disabled = False

    def _pre_request(self, model_id: str):
        if self.auto_disabled:
            return AdapterResponse(
                response={"error": "NIM auto-disabled: credits < 10"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0,
            )
        return None

    def _parse_quota(self, headers):
        # NIM doesn't provide rate headers
        return None

    def _update_credits(self, response_data: dict):
        """Extract and track credits from response, auto-disable if < 10."""
        if "credits_used" in response_data:
            if self.credits_remaining is not None:
                self.credits_remaining -= response_data["credits_used"]
        elif "usage" in response_data and "credits" in response_data["usage"]:
            self.credits_remaining = response_data["usage"]["credits"]
        if self.credits_remaining is not None and self.credits_remaining < 10:
            self.auto_disabled = True

    async def chat_completion(self, messages, model_id, **kwargs):
        result = await super().chat_completion(messages, model_id, **kwargs)
        if "error" not in result.response:
            self._update_credits(result.response)
        return result

    async def list_models(self):
        if self.auto_disabled:
            return []
        return await super().list_models()
