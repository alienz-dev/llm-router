# Refactoring Plan — llm-router

## Context

The codebase works but has ~900 lines of duplicated code across adapters and discovery methods, N+1 query problems, and missing features that competitors (LiteLLM, Portkey, pLLM, ProxyGateLLM) have. This plan addresses structural issues first, then borrows high-value patterns from competitors.

---

## Phase 1: Extract `OpenAICompatibleAdapter` Base Class

**Problem:** 7 of 11 adapters (groq, cerebras, mistral, kilo, openrouter, opencode, nvidia) are near-identical copy-paste (~700 lines total). They share `_retry_with_backoff`, `chat_completion`, `stream_completion`, `list_models`, `close` — differing only in `provider_name`, `base_url`, and sometimes `_parse_quota` header names.

**Solution:** Create `OpenAICompatibleAdapter(BaseAdapter)` in `adapters/base.py` that implements all shared logic. Each adapter becomes ~15 lines: class definition + overrides.

**Files:**
- `llm_router/adapters/base.py` — add `OpenAICompatibleAdapter` class
- `llm_router/adapters/groq.py` — collapse to ~10 lines
- `llm_router/adapters/cerebras.py` — collapse to ~15 lines (custom `_parse_quota`)
- `llm_router/adapters/mistral.py` — collapse to ~10 lines
- `llm_router/adapters/kilo.py` — collapse to ~10 lines
- `llm_router/adapters/openrouter.py` — collapse to ~15 lines (custom `_parse_quota`)
- `llm_router/adapters/opencode.py` — collapse to ~20 lines (`_BLOCKED_MODELS`)
- `llm_router/adapters/nvidia.py` — collapse to ~25 lines (credit tracking override)
- `llm_router/adapters/deepseek.py` — collapse to ~20 lines (`reasoning_content` + 60s timeout)

**Non-OpenAI adapters (keep custom):**
- `google.py` — different API format (generateContent), fake streaming
- `cloudflare.py` — different API format (`/ai/run/@cf/`), fake streaming
- `huggingface.py` — different API format (text-generation), fake streaming

**Estimated reduction:** ~600 lines eliminated

---

## Phase 2: Extract `ModelHealthRepository`

**Problem:** `_update_model_health` is duplicated between `router.py:396-452` and `probe.py:230-281` with slightly different SQL (probe uses `excluded.*`, router uses `COALESCE`). This is a bug vector.

**Solution:** Single `ModelHealthRepository` class in a new `llm_router/health.py` file, used by both router and probe.

**Files:**
- `llm_router/health.py` — new file with `ModelHealthRepository`
- `llm_router/router.py` — replace `_update_model_health` with repository call
- `llm_router/probe.py` — replace `_update_model_health` with repository call

**API:**
```python
class ModelHealthRepository:
    async def update(self, provider_id, model_id, success, latency_ms=None, error=None)
    async def get_health(self, provider_id, model_id) -> dict | None
    async def get_all_health(self) -> list[dict]
```

---

## Phase 3: Extract Discovery Helper

**Problem:** 6 discovery methods (`_discover_cerebras`, `_discover_groq`, `_discover_kilo`, `_discover_opencode`, `_discover_deepseek`, `_discover_mistral`) are structurally identical — same `httpx.AsyncClient(timeout=30)`, same `GET` with bearer auth, same response parsing.

**Solution:** Single `_discover_openai_models(api_key_env, url, default_ctx_len)` helper method.

**Files:**
- `llm_router/discovery.py` — add helper, refactor 6 methods to use it

**Estimated reduction:** ~120 lines eliminated

---

## Phase 4: Fix N+1 Queries in Quota Checking

**Problem:** `quota.py:can_use` executes up to 4 separate queries per call (RPM, TPM, RPD, TPD). Called once per candidate model in `_get_ranked_candidates`. With 50 models = 200+ DB queries per request.

**Solution:** Consolidate into a single query with pre-fetched usage data. Add a `get_usage_summary(provider_id, model_id)` method that returns all limits + current usage in one query.

**Files:**
- `llm_router/quota.py` — add `get_usage_summary()`, refactor `can_use()` to use it
- `llm_router/router.py` — batch-fetch usage summaries before ranking

---

## Phase 5: Add `quota_usage` Cleanup Job

**Problem:** `quota_usage` table grows unboundedly. Every request inserts a row. Over time, sliding-window queries become slow.

**Solution:** Add a scheduled cleanup job that deletes rows older than 48 hours.

**Files:**
- `llm_router/scheduler.py` — add cleanup job (daily, delete rows > 48h old)
- `llm_router/quota.py` — add `cleanup_old_usage()` method

---

## Phase 6: Forward `max_tokens`/`temperature` to Adapters

**Problem:** `ChatCompletionRequest` accepts `max_tokens`, `temperature`, etc. but `app.py` never passes them to `_router.route()`. They are silently dropped.

**Solution:** Pass `**kwargs` from the request through to adapters.

**Files:**
- `llm_router/models.py` — ensure all OpenAI params are captured
- `llm_router/app.py` — forward `req.max_tokens`, `req.temperature`, etc.
- `llm_router/router.py` — pass kwargs through to adapter calls

---

## Phase 7: Make `PROVIDER_PRIORITY` Configurable

**Problem:** `PROVIDER_PRIORITY` dict in `discovery.py:42-50` is hardcoded. Users can't adjust priorities without editing source code.

**Solution:** Add `priority` field to `ProviderConfig` in `config.yaml`. Default to current values.

**Files:**
- `llm_router/config.py` — add `priority: float = 1.0` to `ProviderConfig`
- `llm_router/config.yaml` — add `priority` to each provider
- `llm_router/discovery.py` — read from config instead of hardcoded dict
- `llm_router/router.py` — read from config instead of importing hardcoded dict

---

## Phase 8: Add Auth Middleware

**Problem:** API is completely open. Anyone who can reach the port can drain quotas.

**Solution:** Optional bearer token middleware. If `API_KEY` env var is set, require it in `Authorization: Bearer <key>` header. If not set, allow all (backward compatible).

**Files:**
- `llm_router/app.py` — add `APIKeyMiddleware`
- `.env.example` — add `API_KEY=` entry

---

## Phase 9: Standardize Error Responses

**Problem:** Errors are `{"error": "string"}` but OpenAI convention uses `{"error": {"message": ..., "type": ..., "code": ...}}`.

**Solution:** Add error response model, update all error returns.

**Files:**
- `llm_router/models.py` — add `ErrorResponse` model
- `llm_router/app.py` — update error handling in routes
- `llm_router/router.py` — return structured errors

---

## Phase 10: Fix Silent Exception Swallowing

**Problem:** `queue.py:87` and `queue.py:137` catch exceptions and discard them. `probe.py:146` uses bare `except:`.

**Solution:** Add logging to all exception handlers. Replace bare `except:` with `except Exception:`.

**Files:**
- `llm_router/queue.py` — add `logger.error` to exception handlers
- `llm_router/probe.py` — fix bare `except:` to `except Exception:`

---

## Features to Borrow (After Refactoring)

### A. Model Groups (from LiteLLM)
Group equivalent models across providers. Users call `fast-chat`, router picks best available provider.

### B. Named Routes (from pLLM)
Formalize task classification into named routes: `fast`, `smart`, `coding`, `reasoning`. Users specify route instead of model.

### C. Model Aliases (from VoidLLM)
Friendly names: `default`/`coding`/`reasoning` that map to the best available model.

### D. Config-as-Code Routing (from Portkey)
Per-request JSON DSL for retry count, fallback behavior, timeout override.

### E. Key Rotation with Cooldown (from LLM-API-Key-Proxy)
When a free-tier key hits its rate limit, mark it with a cooldown timer and rotate to the next key.

---

## Execution Order

```
Phase 1 (adapter dedup) ──→ Phase 2 (health repo) ──→ Phase 3 (discovery helper)
                                                            │
                                                            ▼
                                              Phase 4 (N+1 fix) ──→ Phase 5 (cleanup)
                                                            │
                                                            ▼
                                              Phase 6 (forward params)
                                                            │
                                                            ▼
                                              Phase 7 (configurable priority)
                                                            │
                                                            ▼
                                              Phase 8 (auth) ──→ Phase 9 (errors) ──→ Phase 10 (logging)
```

Phases 1-3 can be done in parallel. Phase 4 depends on Phase 2. Phases 5-10 are independent.

## Verification

After each phase:
1. `uv run pytest tests/ -v` — all tests pass
2. `uv run llm-router serve` — server starts cleanly
3. `curl http://localhost:8642/health` — health check OK
4. `curl -X POST http://localhost:8642/v1/chat/completions ...` — auto-routing works
5. No regressions in existing functionality
