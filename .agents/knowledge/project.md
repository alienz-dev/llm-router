# Project Overview

## Name
llm-router — Free-Tier LLM Gateway

## Purpose
Personal quota-aware gateway that aggregates 9 free-tier LLM providers behind a single OpenAI-compatible API with smart routing, batch scheduling, and model discovery.

## Tech Stack
- Python 3.11+, FastAPI, uvicorn
- aiosqlite (WAL mode) for persistence
- httpx for async HTTP to providers
- APScheduler for batch processing + discovery cron
- PyYAML for config, click for CLI
- Build tool: uv with pyproject.toml

## Architecture
```
HTTP API (FastAPI :8642)
  → Smart Router (sync path) / Job Queue (batch path)
    → Quota Manager (SQLite WAL: usage ledger + token estimator)
      → Provider Adapters (9 providers, each normalizes to common interface)
```

## Key Constraints
- No LiteLLM — use httpx directly for provider calls to control quota header parsing
- Token estimation: `len(text) / 4` heuristic — no tiktoken
- Groq TPM is 6K — router MUST check estimated tokens before routing
- NIM credits are finite — auto-disable when < 10 credits
- SQLite WAL mode required for concurrent reads
- Cloudflare and HuggingFace use non-OpenAI API formats — adapters must translate
- Daily reset times differ per provider — store per-provider `daily_reset_utc_hour`
