# Router Contract

## Interface (router.py)

```python
class SmartRouter:
    async def route(
        self, messages: list[dict], model_override: str | None = None,
        stream: bool = False, **kwargs
    ) -> AdapterResponse | AsyncIterator[str]:
        """
        1. If model_override: check quota, route directly
        2. Classify task type from messages
        3. Estimate tokens
        4. Filter models where can_use(provider, model, tokens) is True
        5. Rank by (capability_score * quota_headroom_pct)
        6. Try top model; on failure, fallback to next (up to 3)
        """

    def classify_task(self, messages: list[dict]) -> str:
        """Returns: 'code', 'reasoning', 'summarize', or 'general'"""
```

## Task Classification Keywords
- `code`: "function", "implement", "debug", "refactor", "```", "def ", "class "
- `reasoning`: "analyze", "compare", "explain why", "step by step", "evaluate"
- `summarize`: "summarize", "tldr", "key points", "brief"
- `general`: default fallback

## Ranking Formula
```
score = model.task_scores[task_type] * (provider_quota_remaining_pct + 0.1)
```
Groq penalty: multiply score by `min(1.0, 0.5 * tpm_limit / estimated_tokens)` when estimated_tokens > tpm_limit * 0.3

## Fallback
On 429 or adapter error from selected provider → try next ranked model (up to 3 total attempts).
