# Environment

## Development Machine
WSL2 Linux, user mingl

## Code Location
`~/work-enhancement/llm-router/`

## Credentials
API keys via environment variables — see `.env.example`:
- `OPENROUTER_API_KEY`, `GOOGLE_AI_API_KEY`, `CEREBRAS_API_KEY`
- `GROQ_API_KEY`, `MISTRAL_API_KEY`, `NVIDIA_API_KEY`
- `KILO_API_KEY`, `CF_ACCOUNT_ID`, `CF_API_TOKEN`
- `HF_API_TOKEN`

No secrets in config.yaml — keys referenced by env var name only.

## Build & Run
```bash
uv sync                          # install deps
uv run python -m llm_router      # start server on :8642
uv run llm-router serve           # CLI alternative
```

## Production
Runs as systemd service via `llm-router.service`. Port 8642.
