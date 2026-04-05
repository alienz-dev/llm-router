# Quota Manager Contract

## Interface (quota.py)

```python
class QuotaManager:
    async def estimate_tokens(self, messages: list[dict]) -> int:
        """len(text) / 4 heuristic across all message content."""

    async def can_use(
        self, provider: str, model_id: str, estimated_tokens: int
    ) -> bool:
        """Check RPM + TPM against sliding window usage AND known limits.
        Factors in estimated_tokens + expected completion (default 1024)."""

    async def record_usage(
        self, provider: str, model_id: str,
        tokens_in: int, tokens_out: int, success: bool
    ) -> None:
        """Write to quota_usage table."""

    async def update_limits(
        self, provider: str, model_id: str, snapshot: QuotaSnapshot
    ) -> None:
        """Merge QuotaSnapshot from response headers into quota_limits."""

    async def get_all_quota_status(self) -> list[dict]:
        """Per-provider quota status for dashboard/API."""
```

## Sliding Windows
- RPM: usage in last 60 seconds
- RPH: usage in last 3600 seconds
- RPD/TPD: usage since daily reset (per-provider `daily_reset_utc_hour`)

## Pre-flight Logic
1. Estimate prompt tokens: `len(all_message_text) / 4`
2. Add expected completion tokens (default 1024)
3. Check: `rpm_used < rpm_limit` AND `(estimated_total_tokens + tpm_used) < tpm_limit`
4. Check daily limits similarly
5. Return False if any limit would be exceeded
