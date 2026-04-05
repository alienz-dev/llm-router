# LLM Router

Free-tier LLM gateway that aggregates 9 providers behind a single OpenAI-compatible API with smart routing, quota management, and batch processing.

## Features

- **Smart Routing** — task-aware model selection with quota-based ranking
- **Quota Management** — real-time tracking with sliding windows and per-provider daily resets
- **Batch Processing** — queue non-urgent requests for overnight processing
- **Model Discovery** — auto-discover free models from OpenRouter, Cerebras, Groq, Kilo
- **Fallback Chain** — automatic failover between providers on rate limits
- **OpenAI Compatible** — drop-in replacement for OpenAI API clients

## Architecture

```
HTTP API (FastAPI :8642)
  → Smart Router (sync) / Job Queue (batch)
    → Quota Manager (SQLite WAL)
      → Provider Adapters (9 providers)
```

## Supported Providers

| Provider | Free Tier | Signup |
|---|---|---|
| OpenRouter | 20 RPM, 50 RPD | https://openrouter.ai |
| Google AI Studio | 15 RPM, 1K RPD, 250K TPM | https://aistudio.google.com |
| Cerebras | 30 RPM, 14.4K RPD, 1M TPD | https://inference.cerebras.ai |
| Groq | 30 RPM, 6K TPM | https://console.groq.com |
| Mistral | 2 RPM, 500K TPM | https://console.mistral.ai |
| NVIDIA NIM | 1000 credits total | https://build.nvidia.com |
| Kilo Gateway | Varies | https://gateway.kilo.ai |
| Cloudflare Workers AI | 300 RPM, 10K neurons/day | https://dash.cloudflare.com |
| Hugging Face | ~100 req/hour | https://huggingface.co |

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd llm-router
uv sync

# 2. Configure API keys
cp .env.example .env
# Edit .env with your provider API keys

# 3. Start server
uv run llm-router serve

# 4. Test
curl http://localhost:8642/health
curl -X POST http://localhost:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

## CLI

```bash
llm-router serve              # Start server (default :8642)
llm-router serve --port 9000  # Custom port
llm-router discover           # Run model discovery once
llm-router status             # Print quota status table
llm-router process            # Process pending batch jobs
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/chat/completions` | Chat completions (sync + streaming) |
| GET | `/v1/models` | List active free models |
| GET | `/v1/quota` | Per-provider quota status |
| GET | `/dashboard` | System dashboard (JSON) |
| POST | `/jobs` | Submit batch/immediate job |
| GET | `/jobs/{id}` | Job status |
| GET | `/jobs/{id}/result` | Job result |
| POST | `/jobs/process` | Manual batch trigger |
| GET | `/health` | Health check |

### Chat Completions

```bash
# Auto-route to best available model
curl -X POST http://localhost:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Write a Python function to sort a list"}]}'

# Route to specific provider:model
curl -X POST http://localhost:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "model": "groq:llama-3.3-70b-versatile"}'

# Streaming
curl -X POST http://localhost:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

Response headers include `x-llm-router-provider` and `x-llm-router-model` indicating which provider/model was used.

### Batch Jobs

```bash
# Submit batch job (processed overnight)
curl -X POST http://localhost:8642/jobs \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Summarize quantum computing"}], "priority": "batch"}'

# Submit immediate job
curl -X POST http://localhost:8642/jobs \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "priority": "immediate"}'

# Check status / get result
curl http://localhost:8642/jobs/{job_id}
curl http://localhost:8642/jobs/{job_id}/result
```

## Use as OpenAI Drop-in

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8642/v1", api_key="unused")
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Configuration

`config.yaml` defines providers, scheduler windows, and database path. API keys are always in `.env` (never in config).

Key settings:
- `providers.*.daily_reset_utc_hour` — when daily quota resets (0=midnight UTC, 8=midnight Pacific)
- `scheduler.batch_window_start/end` — overnight batch processing window (default 00:00-06:00 AEST)
- `scheduler.discovery_interval_hours` — model discovery frequency (default 6h)

## Systemd Service

```bash
sudo cp llm-router.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llm-router
```

## How Routing Works

1. Classify task type from message content (code/reasoning/summarize/general)
2. Estimate token count (`len(text) / 4`)
3. Filter models where quota allows the request
4. Rank by `capability_score × quota_headroom_pct`
5. Groq penalized for long prompts (6K TPM limit)
6. Try top model; on failure, fallback to next (up to 3 attempts)

## License

MIT
