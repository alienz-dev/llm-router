# Adapter Interface Contract

## Base ABC (adapters/base.py)

```python
from dataclasses import dataclass
from typing import Any, AsyncIterator
from abc import ABC, abstractmethod

@dataclass
class QuotaSnapshot:
    rpm_limit: int | None = None
    rpm_remaining: int | None = None
    rpd_limit: int | None = None
    rpd_remaining: int | None = None
    tpm_limit: int | None = None
    tpm_remaining: int | None = None
    tpd_limit: int | None = None
    tpd_remaining: int | None = None
    reset_seconds: float | None = None

@dataclass
class AdapterResponse:
    response: dict[str, Any]          # OpenAI-format response dict
    quota: QuotaSnapshot | None       # Parsed from response headers
    tokens_in: int                     # From response usage or estimated
    tokens_out: int                    # From response usage or estimated
    latency_ms: float

class BaseAdapter(ABC):
    provider_name: str

    @abstractmethod
    async def chat_completion(
        self, messages: list[dict], model_id: str, **kwargs
    ) -> AdapterResponse: ...

    @abstractmethod
    async def stream_completion(
        self, messages: list[dict], model_id: str, **kwargs
    ) -> AsyncIterator[str]: ...

    @abstractmethod
    async def list_models(self) -> list[dict]: ...
```

## Error Handling
- On 429: retry with exponential backoff (1s, 2s, 4s + jitter), max 3 retries
- After 3 retries: return structured error (do NOT raise) so router can fallback to next provider
- On 5xx: single retry after 1s, then error
- On auth error (401/403): return error immediately, no retry
