# LLM Router

Free-tier LLM gateway that aggregates 11 providers behind a single OpenAI-compatible API with availability-aware smart routing, quota management, circuit breakers, and batch processing.

## Features

- **Availability-Aware Routing** — models scored by real-time health (probe freshness, success rate, latency)
- **Smart Task Classification** — routes code/reasoning/summarize/general tasks to best-fit models
- **Quota Management** — real-time tracking with sliding windows and per-provider daily resets
- **Circuit Breakers** — per-provider circuit breakers with DB persistence across restarts
- **Model Discovery** — auto-discovers free models from all 11 providers
- **Model Probing** — verifies model availability every 2h, updates health data
- **Batch Processing** — queue non-urgent requests for overnight processing
- **Fallback Chain** — automatic failover between providers on rate limits
- **OpenAI Compatible** — drop-in replacement for OpenAI API clients

## Architecture

```
HTTP API (FastAPI :8642)
  → Smart Router (sync) / Job Queue (batch)
    → Availability Scoring (model_health + circuit_breaker)
      → Quota Manager (SQLite WAL)
        → Provider Adapters (11 providers)
          → Probe System (every 2h)
          → Discovery System (every 6h)
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
| DeepSeek | Pay-per-use | https://platform.deepseek.com |
| OpenCode Z | Free tier | https://opencode.ai |

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd llm-router
uv sync

# 2. Configure API keys
cp .env.example .env
# Edit .env with your provider API keys (at least one required)

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
| POST | `/v1/chat/completions` | Chat completions (sync + SSE streaming) |
| GET | `/v1/models` | List active free models |
| GET | `/v1/models/health` | Per-model availability, latency, success rate |
| GET | `/v1/quota` | Per-provider quota status |
| GET | `/v1/providers/health` | Circuit breaker + quota combined status |
| GET | `/dashboard` | System dashboard (models, jobs, quota, health) |
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

Response headers:
- `x-llm-router-provider` — which provider was used
- `x-llm-router-model` — which model was used
- `x-llm-router-fallback` — "true" if fallback occurred
- `x-llm-router-original-model` — original model if fallback occurred

### Model Health

```bash
# Check per-model availability data
curl -s http://localhost:8642/v1/models/health | python3 -m json.tool
```

Returns per-model: `last_probe_at`, `last_probe_ok`, `avg_latency_ms`, `success_count`, `failure_count`, `consecutive_failures`, `success_rate`.

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
- `scheduler.batch_window_start/end` — batch processing window in UTC
- `scheduler.discovery_interval_hours` — model discovery frequency (default 6h)

## Systemd Service

Edit `llm-router.service` to set your user and install path, then:

```bash
sudo cp llm-router.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llm-router
```

## How Routing Works

1. **Task Classification** — regex-based classification into code/reasoning/summarize/general
2. **Token Estimation** — `len(text) / 4` heuristic
3. **Candidate Scoring** — for each active model:
   ```
   score = capability × (quota_headroom + 0.1) × availability × provider_priority
   ```
4. **Availability** — computed from probe freshness, success rate, and latency:
   - Freshness: 1.0 if probed <2h ago, decaying to 0.1 after 24h
   - Success rate: rolling window, penalized 5x if 3+ consecutive failures
   - Latency penalty: 0.8 if >5s avg, 0.5 if >10s avg
   - Hard skip: models with 5+ consecutive failures excluded
5. **Fallback** — tries top 5 candidates sequentially; on failure, records to circuit breaker and tries next
6. **Circuit Breaker** — per-provider, trips after 5 failures in 60s, auto-recovers after 30s cooldown, persisted to DB

## Monitoring

- `GET /health` — basic health check
- `GET /dashboard` — system overview with top models health
- `GET /v1/quota` — per-provider quota status
- `GET /v1/providers/health` — circuit breaker + quota combined
- `GET /v1/models/health` — per-model availability data
- `llm-router status` — CLI quota table

## License

MIT
