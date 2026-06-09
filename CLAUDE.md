# llm-router — Project Instructions

## What This Is
Free-tier LLM gateway that aggregates 11 providers behind a single OpenAI-compatible API with availability-aware smart routing, quota management, circuit breakers, and batch processing.

## Key Files
- `llm_router/app.py` — FastAPI routes, lifespan wiring, adapter construction
- `llm_router/router.py` — Smart routing with availability scoring, fallback chain, task classification
- `llm_router/discovery.py` — Model discovery from all 11 providers, task scores, provider priority
- `llm_router/probe.py` — Model health probing, known limits, health data updates
- `llm_router/circuit_breaker.py` — Per-provider circuit breaker with DB persistence
- `llm_router/quota.py` — Sliding-window quota manager (RPM/RPH/RPD/TPM/TPH/TPD)
- `llm_router/scheduler.py` — APScheduler: batch processing, discovery, probing, recovery
- `llm_router/adapters/base.py` — BaseAdapter ABC, AdapterResponse, QuotaSnapshot
- `llm_router/adapters/` — 11 provider adapters (7 are near-identical OpenAI-compatible)
- `config.yaml` — Provider definitions, scheduler windows, server settings
- `.env` — API keys (gitignored, never committed)

## Commands
```bash
uv sync                          # Install dependencies
uv run llm-router serve          # Start server on :8642
uv run llm-router discover       # Run model discovery once
uv run llm-router status         # Print quota status table
uv run pytest tests/ -v          # Run all tests
uv run pytest tests/test_circuit_breaker.py -v  # Run specific test file
```

## Conventions
- Python 3.11+, all async code uses aiosqlite
- Provider IDs: lowercase slugs (`openrouter`, `google`, `deepseek`)
- Model format: `provider:model_id` for direct routing, `auto` for smart routing
- API keys: always in `.env`, referenced by env var name in config.yaml
- Errors: returned as `{"error": "string"}` inside AdapterResponse.response
- Circuit breaker: persisted to `circuit_breaker_state` table on every transition
- Health data: persisted to `model_health` table after every request and probe cycle
- Task classification: regex-based (code/reasoning/summarize/general)
- Scoring: `capability × (quota_headroom + 0.1) × availability × provider_priority`

## What NOT To Do
- Do not use synchronous sqlite3 — always aiosqlite
- Do not commit API keys or .env contents
- Do not break the OpenAI-compatible `/v1/chat/completions` contract
- Do not add dependencies without checking existing ones first
- Do not modify DB schema without considering backward compatibility
- Do not remove the fallback chain or circuit breaker logic
- Do not hardcode provider priority — it should come from config
