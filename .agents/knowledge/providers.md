# Provider Reference

## API Formats
All providers use OpenAI-compatible chat completions format EXCEPT:
- **Cloudflare Workers AI**: `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/{model}` — different request/response schema
- **Hugging Face**: `https://api-inference.huggingface.co/models/{model}` — different format

## Quota Header Parsing

| Provider | Header Pattern |
|---|---|
| OpenRouter | `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`, `x-ratelimit-limit-tokens` |
| Cerebras | `x-ratelimit-limit-requests-day`, `x-ratelimit-remaining-requests-day`, `x-ratelimit-limit-tokens-minute`, `x-ratelimit-remaining-tokens-minute` |
| Groq | `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`, `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens` |
| Mistral | Standard OpenAI-style `x-ratelimit-*` |
| Google AI Studio | Quota info in response metadata, not headers — track locally |
| NVIDIA NIM | Credit-based — no rate headers, track credits from response |
| Kilo Gateway | OpenAI-compatible headers |
| Cloudflare | No rate headers — track locally against 300 RPM / 10K neurons/day |
| Hugging Face | No rate headers — track locally against ~100 req/hour |

## Free-Tier Limits (April 2026)

| Provider | RPM | RPD | TPM/TPD | Notes |
|---|---|---|---|---|
| OpenRouter | 20 | 50 | — | Low daily quota; spread load |
| Google AI Studio | 15/10/5 | 1000/250/100 | 250K TPM | Flash-Lite is workhorse |
| Cerebras | 30 | 14.4K | 1M TPD | Fastest inference; high daily |
| Groq | 30 | — | 6K TPM | **Very low TPM — short prompts only** |
| Mistral | 2 | — | 500K TPM, 1B/mo | Low RPM but massive monthly |
| NVIDIA NIM | varies | — | 1000 credits total | **Finite credits; auto-disable** |
| Kilo Gateway | varies | — | — | Rotates free models |
| Cloudflare | 300 | — | 10K neurons/day | Non-OpenAI format |
| Hugging Face | ~100/hr | — | — | Non-OpenAI format |

## Daily Reset Times
- OpenRouter/Groq: midnight UTC (hour 0)
- Cerebras: rolling windows
- Google AI Studio: midnight Pacific (hour 8 UTC)
- Others: midnight UTC default
