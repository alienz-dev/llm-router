# AGENTS.md — llm-router

## Overview
Free-tier LLM gateway aggregating 11 providers behind a single OpenAI-compatible API with availability-aware smart routing, quota management, circuit breakers, and batch processing.

## Tech Stack
- Python 3.11+, FastAPI, uvicorn
- aiosqlite (WAL mode), httpx, APScheduler
- Pydantic v2, Click CLI, PyYAML, python-dotenv
- Build: hatchling, Package manager: uv

## Commands
```bash
uv sync                          # Install dependencies
uv run llm-router serve          # Start server (default :8642)
uv run llm-router discover       # Run model discovery
uv run llm-router status         # Print quota status table
uv run llm-router process        # Process pending batch jobs
uv run pytest tests/ -v          # Run test suite
uv run pytest tests/ -v --tb=short -x  # Stop on first failure
```

## Agent Roles
| Role | Purpose |
|------|---------|
| Supervisor | Orchestrate refactoring phases, verify changes, manage dependencies |
| Coder | Implement adapter dedup, health repository, discovery helpers |
| Reviewer | Verify no regressions, test coverage, API compatibility |

## Architecture
```
llm_router/
├── app.py              # FastAPI routes + lifespan wiring
├── router.py           # Smart routing with availability scoring
├── discovery.py        # Model discovery from 11 providers
├── probe.py            # Model health probing
├── circuit_breaker.py  # Per-provider circuit breaker (DB-persisted)
├── quota.py            # Sliding-window quota manager
├── scheduler.py        # APScheduler: batch, discovery, probe, recovery
├── queue.py            # Job queue for batch/immediate processing
├── db.py               # SQLite schema + connection
├── config.py           # YAML config loader
├── models.py           # Pydantic request/response models
├── cli.py              # Click CLI
└── adapters/
    ├── base.py         # BaseAdapter + AdapterResponse + QuotaSnapshot
    ├── openrouter.py   # OpenRouter adapter
    ├── google.py       # Google AI Studio adapter (format conversion)
    ├── deepseek.py     # DeepSeek adapter (reasoning_content)
    ├── cerebras.py     # Cerebras adapter
    ├── groq.py         # Groq adapter
    ├── mistral.py      # Mistral adapter
    ├── nvidia.py       # NVIDIA NIM adapter (credit tracking)
    ├── kilo.py         # Kilo Gateway adapter
    ├── cloudflare.py   # Cloudflare Workers AI (different API format)
    ├── huggingface.py  # HuggingFace Inference (different API format)
    └── opencode.py     # OpenCode Z adapter (blocked models)
```

## Refactoring Phases
See `plans/refactoring.md` for the full plan. Phases:
1. Extract `OpenAICompatibleAdapter` base class (eliminate ~700 lines of duplication)
2. Extract `ModelHealthRepository` (deduplicate router.py + probe.py)
3. Extract discovery helper (deduplicate 6 identical `_discover_*` methods)
4. Fix N+1 queries in quota checking
5. Add `quota_usage` cleanup job
6. Forward `max_tokens`/`temperature` to adapters
7. Make `PROVIDER_PRIORITY` configurable
8. Add auth middleware
9. Standardize error responses
10. Fix silent exception swallowing

## Conventions
- All adapters extend `BaseAdapter` and implement `chat_completion`, `stream_completion`, `list_models`, `close`
- Provider IDs are lowercase slugs: `openrouter`, `google`, `deepseek`, etc.
- Model IDs use `provider:model` format for direct routing, `auto` for smart routing
- DB schema uses `CREATE TABLE IF NOT EXISTS` for forward compatibility
- All async code uses `aiosqlite`, never synchronous sqlite3
- API keys are always in `.env`, never in config files
- Circuit breaker state is persisted to DB on every transition

## What NOT To Do
- Do not add synchronous DB calls (everything must be async)
- Do not hardcode API keys or secrets
- Do not break the OpenAI-compatible API contract
- Do not add dependencies without checking if existing ones suffice
- Do not modify the DB schema without considering existing databases
- Do not remove the fallback chain logic
