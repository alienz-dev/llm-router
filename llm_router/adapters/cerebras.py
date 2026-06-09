from .base import OpenAICompatibleAdapter


class CerebrasAdapter(OpenAICompatibleAdapter):
    provider_name = "cerebras"

    def __init__(self, api_key: str):
        super().__init__(api_key, base_url="https://api.cerebras.ai/v1")

    def _parse_quota(self, headers):
        from .base import QuotaSnapshot
        return QuotaSnapshot(
            rpd_limit=int(headers.get("x-ratelimit-limit-requests-day", 0)) or None,
            rpd_remaining=int(headers.get("x-ratelimit-remaining-requests-day", 0)) or None,
            tpm_limit=int(headers.get("x-ratelimit-limit-tokens-minute", 0)) or None,
            tpm_remaining=int(headers.get("x-ratelimit-remaining-tokens-minute", 0)) or None,
            reset_seconds=float(headers.get("x-ratelimit-reset-tokens-minute", 0)) or None,
        )
