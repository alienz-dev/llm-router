from .base import OpenAICompatibleAdapter, AdapterResponse


class OpenCodeAdapter(OpenAICompatibleAdapter):
    provider_name = "opencode"

    # Models to exclude — verified unreliable or not free
    _BLOCKED_MODELS = {"nemotron-3-super-free", "gpt-5-nano"}

    def __init__(self, api_key: str):
        super().__init__(api_key, base_url="https://opencode.ai/zen/v1")

    def _pre_request(self, model_id: str):
        if model_id in self._BLOCKED_MODELS:
            return AdapterResponse(
                response={"error": f"Model {model_id} is blocked (unreliable or not free)"},
                quota=None, tokens_in=0, tokens_out=0, latency_ms=0,
            )
        return None

    def _parse_quota(self, headers):
        from .base import QuotaSnapshot
        return QuotaSnapshot(
            rpm_limit=int(headers.get("x-ratelimit-limit-requests", 0)) or None,
            rpm_remaining=int(headers.get("x-ratelimit-remaining-requests", 0)) or None,
            tpm_limit=int(headers.get("x-ratelimit-limit-tokens", 0)) or None,
            tpm_remaining=int(headers.get("x-ratelimit-remaining-tokens", 0)) or None,
            reset_seconds=float(headers.get("x-ratelimit-reset-requests", 0)) or None,
        )
