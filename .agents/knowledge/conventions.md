# Naming Collision Warnings

These MUST be followed by all agents working on this codebase:

1. **app.py**: Do NOT use `status` as a local variable name if importing job status enums — use `job_status` or similar
2. **router.py**: Do NOT use `model` as a local if importing from `models.py` — use `llm_model` or `model_record`
3. **adapters/**: The `models` module name collides with `pydantic.models` — always use explicit imports
4. **General**: When adding imports, check for existing local variables with the same name. Use aliases to avoid shadowing.

# Code Conventions

- TDD: tests before implementation (when tests are requested)
- Exponential backoff with jitter on 429: wait 1s, 2s, 4s (max 3 retries)
- Token estimation: `len(text) / 4` — no tiktoken
- SQLite WAL mode: `PRAGMA journal_mode=WAL` on first connection
- All API keys via env vars, never hardcoded
- Git branches: `agent/{id}/{feature}`
