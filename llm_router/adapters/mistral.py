from .base import OpenAICompatibleAdapter


class MistralAdapter(OpenAICompatibleAdapter):
    provider_name = "mistral"

    def __init__(self, api_key: str):
        super().__init__(api_key, base_url="https://api.mistral.ai/v1")

    def _parse_quota(self, headers):
        from .base import QuotaSnapshot
        return QuotaSnapshot(
            rpm_limit=int(headers.get("x-ratelimit-limit-requests", 0)) or None,
            rpm_remaining=int(headers.get("x-ratelimit-remaining-requests", 0)) or None,
            tpm_limit=int(headers.get("x-ratelimit-limit-tokens", 0)) or None,
            tpm_remaining=int(headers.get("x-ratelimit-remaining-tokens", 0)) or None,
            reset_seconds=float(headers.get("x-ratelimit-reset-requests", 0)) or None,
        )
