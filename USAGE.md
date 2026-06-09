# LLM Router — Usage Guide

LLM Router is a self-hosted gateway that aggregates free-tier LLM providers behind a single OpenAI-compatible API. It classifies incoming requests by task type (code, reasoning, summarization, general), scores available models by capability, quota headroom, and real-time availability, and routes to the best option — with automatic fallback if a provider is rate-limited or down.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
git clone <repo-url>
cd llm-router
uv sync
cp .env.example .env
```

Edit `.env` and add API keys for the providers you want to use. You don't need all 11 — the router works with whatever providers are configured.

## Provider Setup

Each provider offers a free tier. Sign up and get an API key:

### OpenRouter (20 RPM, 50 RPD)
1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Go to **Keys** page → **Create Key**
3. Set `OPENROUTER_API_KEY` in `.env`

### Google AI Studio (15 RPM, 1K RPD, 250K TPM)
1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Click **Get API Key** → **Create API key in new project**
3. Set `GOOGLE_AI_API_KEY` in `.env`

### Cerebras (30 RPM, 14.4K RPD, 1M TPD)
1. Sign up at [inference.cerebras.ai](https://inference.cerebras.ai)
2. Get API key from the dashboard
3. Set `CEREBRAS_API_KEY` in `.env`

### Groq (30 RPM, 6K TPM)
1. Sign up at [console.groq.com](https://console.groq.com)
2. Go to **API Keys** → **Create API Key**
3. Set `GROQ_API_KEY` in `.env`

### Mistral (2 RPM, 500K TPM)
1. Sign up at [console.mistral.ai](https://console.mistral.ai) (phone verification required)
2. Go to **API Keys** → **Create new key**
3. Set `MISTRAL_API_KEY` in `.env`

### NVIDIA NIM (1000 credits total)
1. Join the developer program at [build.nvidia.com](https://build.nvidia.com)
2. Go to any NIM model page → **Get API Key**
3. Set `NVIDIA_API_KEY` in `.env`

### Kilo Gateway
1. Sign up at [kilo.ai](https://kilo.ai)
2. Get API key from the dashboard
3. Set `KILO_API_KEY` in `.env`

### Cloudflare Workers AI (300 RPM, 10K neurons/day)
1. Create account at [dash.cloudflare.com](https://dash.cloudflare.com)
2. Go to **AI** → **Workers AI** to find your **Account ID** (also in the URL)
3. Go to **My Profile** → **API Tokens** → **Create Token** with **Workers AI** read permission
4. Set `CF_ACCOUNT_ID` and `CF_API_TOKEN` in `.env`

### Hugging Face (~100 req/hour)
1. Sign up at [huggingface.co](https://huggingface.co)
2. Go to **Settings** → **Access Tokens** → **New token** (read permission is sufficient)
3. Set `HF_API_TOKEN` in `.env`

### DeepSeek
1. Sign up at [platform.deepseek.com](https://platform.deepseek.com)
2. Go to **API Keys** → **Create API Key**
3. Set `DEEPSEEK_API_KEY` in `.env`

### OpenCode Z
1. Sign up at [opencode.ai](https://opencode.ai)
2. Get API key from the dashboard
3. Set `OPENCODE_API_KEY` in `.env`

## Configuration

### config.yaml

```yaml
server:
  host: "0.0.0.0"
  port: 8642

database:
  path: "llm_router.db"    # SQLite WAL — auto-created on first run

scheduler:
  batch_window_start: "14:00"   # UTC — batch jobs run in this window
  batch_window_end: "20:00"
  discovery_interval_hours: 6   # How often to re-discover models

providers:
  openrouter:
    name: "OpenRouter"
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: "OPENROUTER_API_KEY"       # Reads from this env var
    daily_reset_utc_hour: 0                 # When daily quotas reset
  # ... (9 providers pre-configured)
```

- `api_key_env` — the env var name (not the key itself). Keys live in `.env`.
- `daily_reset_utc_hour` — when the provider resets daily limits (used for quota tracking).
- Providers without a configured API key are automatically disabled.

### Environment Variables

All provider keys are set in `.env`. The only required config is at least one provider key. Everything else has sensible defaults.

## Running

### Development
```bash
llm-router serve
# or with auto-reload:
llm-router serve --reload
# custom host/port:
llm-router serve --host 127.0.0.1 --port 9000
```

### First Run

On first start, run model discovery to populate the database:
```bash
llm-router discover
```
This queries each configured provider for available free models and stores them in SQLite. The scheduler re-runs discovery every 6 hours automatically.

### systemd (Production)

```ini
[Unit]
Description=LLM Router
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/llm-router
ExecStart=/path/to/llm-router/.venv/bin/llm-router serve
EnvironmentFile=/path/to/llm-router/.env
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Usage Examples

### OpenAI Drop-In (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8642/v1",
    api_key="unused",  # no auth required
)

response = client.chat.completions.create(
    model="auto",  # let the router pick the best model
    messages=[{"role": "user", "content": "Explain Python generators"}],
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Write a haiku about code"}],
    stream=True,
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### curl — Chat Completion

```bash
# Non-streaming
curl -s http://localhost:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'

# Streaming (SSE)
curl -N http://localhost:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }'
```

### curl — List Models

```bash
curl -s http://localhost:8642/v1/models | python3 -m json.tool
```

### curl — Quota Status

```bash
curl -s http://localhost:8642/v1/quota | python3 -m json.tool
```

### curl — Submit Batch Job

```bash
# Submit a job for overnight processing
curl -s -X POST http://localhost:8642/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Summarize the history of Unix"}],
    "model": "auto"
  }'
# Returns: {"job_id": "..."}

# Check job status
curl -s http://localhost:8642/jobs/{job_id}

# Get result (when complete)
curl -s http://localhost:8642/jobs/{job_id}/result
```

### curl — Process Batch Jobs Manually

```bash
curl -s -X POST http://localhost:8642/jobs/process
```

### CLI — Quota Status

```bash
llm-router status
# Output:
# Provider          RPM          TPM        RPD  Healthy
# ----------------------------------------------------------
# openrouter       2/20      1200/∞       12/50     ✅
# google           0/15         0/250000    0/1000   ✅
# groq            15/30      3200/6000     ∞/∞      ✅
# ...
```

### CLI — Process Batch Jobs

```bash
llm-router process
```

## How Routing Works

1. **Task Classification** — the router analyzes the message content and classifies it as one of: `code`, `reasoning`, `summarize`, or `general`. Classification uses regex pattern matching on the prompt text.

2. **Candidate Scoring** — for each active model, the router computes:
   ```
   score = capability_score × (quota_headroom + 0.1) × availability × provider_priority
   ```
   - `capability_score` — per-model, per-task score from discovery (0–1)
   - `quota_headroom` — percentage of remaining quota for that provider (0–1)
   - `availability` — freshness × success_rate × latency_penalty (see below)
   - `provider_priority` — per-provider multiplier (openrouter=1.2, nvidia=1.1, etc.)

3. **Availability Tracking** — every request and probe updates per-model health data:
   - **Freshness**: 1.0 if probed <2h ago, decaying to 0.1 after 24h
   - **Success rate**: rolling `success_count / total`, penalized 5x if 3+ consecutive failures
   - **Latency penalty**: 0.8 if >5s avg, 0.5 if >10s avg
   - **Hard skip**: models with 5+ consecutive failures are excluded entirely

4. **Groq Penalty** — Groq has tight TPM limits. If the estimated token count exceeds 30% of the model's TPM limit, the score is penalized proportionally. This prevents Groq from being selected for long prompts.

5. **Fallback Chain** — candidates are sorted by score. The router tries the top candidate first. On failure (rate limit, timeout, error), it falls back to the next candidate. All providers are tried before returning an error.

6. **Circuit Breaker** — per-provider circuit breaker with 3 states (CLOSED/OPEN/HALF_OPEN). Trips after 5 failures in 60s, auto-recovers after 30s cooldown. State is persisted to DB across restarts.

## Monitoring

- **Dashboard**: `http://localhost:8642/dashboard` — JSON with provider status, job counts, top models health, circuit breakers
- **Quota API**: `GET /v1/quota` — JSON quota status for all providers
- **Provider Health**: `GET /v1/providers/health` — circuit breaker + quota combined status
- **Model Health**: `GET /v1/models/health` — per-model availability, latency, success rate
- **Health**: `GET /health` — basic health check
- **CLI**: `llm-router status` — terminal-friendly quota table

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No models found" / empty model list | Run `llm-router discover` to populate the database |
| All providers exhausted (500) | Check `llm-router status` — quotas may be depleted. Wait for daily reset or add more provider keys |
| 429 errors in logs | Normal — the router automatically falls back to the next provider. Only a problem if ALL providers return 429 |
| SSL errors (Zscaler) | Set `REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt` or the Zscaler CA bundle path |
| Specific provider always fails | Check the API key is valid. Run `llm-router discover` and verify the provider appears in output |
| Stale models after provider changes | Run `llm-router discover` — it refreshes the model list and deactivates removed models |
| Batch jobs not processing | Check scheduler config in `config.yaml` — jobs only process within the batch window (default: 14:00–20:00 UTC). Use `llm-router process` to run manually |

## Endpoints Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (OpenAI-compatible, sync + streaming) |
| `GET` | `/v1/models` | List active free models |
| `GET` | `/v1/models/health` | Per-model availability, latency, success rate |
| `GET` | `/v1/quota` | Quota status for all providers |
| `GET` | `/v1/providers/health` | Circuit breaker + quota combined status |
| `GET` | `/health` | Health check |
| `GET` | `/dashboard` | System dashboard (models, jobs, quota, health) |
| `POST` | `/jobs` | Submit batch/immediate job |
| `GET` | `/jobs/{id}` | Get job status |
| `GET` | `/jobs/{id}/result` | Get job result |
| `POST` | `/jobs/process` | Trigger batch processing |
