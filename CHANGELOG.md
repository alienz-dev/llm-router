# Changelog

Append-only. One dated line per shipped change. Current state lives in `STATUS.md`.

## 2026-08-19 (later)

- The free-tier ledger is real. Account-wide caps (OpenRouter's 50/day) are seeded as the
  `model_id = ''` row the schema reserved, counted across every model on the key, and
  reported with a `limits_known` flag. `can_use` had been returning True for every request
  ever made, and `quota_remaining_pct` was a constant 1.0 — which made the `headroom` term
  in the routing score inert.
- The probe stopped eating the budget it measures: 24h staleness window, 8 models per
  provider per cycle, skips a provider under 25% headroom, and records what it spends.
  Startup went from 39 provider requests to 0. Router freshness bands widened to match.
- `auto` spreads its fallback across three distinct providers instead of the top five
  models, so one rate-limited provider no longer ends the request.
- Fixed: `/v1/models/health` success rate (divided by `avg_latency_ms`); rate-limited sweeps
  marking uncalled models unhealthy; the weekly capability probe crashing on a keyless
  provider; a mid-stream client disconnect going unbooked; silent transport timeouts never
  opening the breaker; unknown models returning 502 instead of 404.
- Logging is configured. Nothing in the package had ever called `basicConfig`, so every
  `logger.info` was discarded.

## 2026-08-19

- Credentials are redacted where the error string is built, not where it is printed —
  logs, `model_health.last_probe_error`, circuit-breaker state and the body a caller
  receives. `API_KEY` auth uses `hmac.compare_digest`; the `?key=` fallback is gone.
- `tools` / `tool_choice` are forwarded, and an agent's second turn no longer 422s:
  assistant messages with `content: null` + `tool_calls`, `role: "tool"` replies and
  multimodal content arrays all round-trip.
- Streaming passes provider chunks through verbatim, so a streamed tool call
  reassembles; `stream_options.include_usage` is honoured; the provider's
  `finish_reason` is no longer overwritten; and streamed requests record quota,
  health and circuit-breaker success for the first time.
- Routing enforces capabilities on all three paths. No capable model is an explicit
  400 phrased like a provider's own refusal. A 400/422 refusal is returned at once
  and narrows the model's capability flags instead of counting against its health.
- `routing.passthrough_params` in `config.yaml` is the one-line revert for the whole
  passthrough wave.
- Per-model capability flags seeded from the verified inventory and re-probed weekly
  on a per-provider budget; NULL means unknown, and a failed probe never erases a
  good reading.
- Batch jobs carry `response_format` / `tools` through to the provider.
- `/health` asserts a database round-trip and a live adapter; `/v1/models` returns
  provider-prefixed ids; startup skips discovery and probing when the data is fresh.
- `deploy/` replaces the systemd unit with launchd plists, an idempotent installer,
  a dedicated heartbeat job and a runbook. Not yet installed on agent-mini.
- Test suite enforces its own hermeticity (124 tests, no network). `scripts/agent_gate.sh`
  runs the live proof — nexus `callStructured` plus a LangGraph ReAct turn — green.

## 2026-08-18

- Landed the image-generation routing and Agnes provider work carried over from agent-mini.
- Added a schema migration mechanism — there was none; any new column would have been
  invisible to every existing database.
- `get_db()` honours `DATABASE_PATH`, which the test suite had been setting all along while
  the code read a hardcoded path.
- Moved the live-server smoke script out of `tests/`, where it crashed pytest collection;
  `uv run pytest tests/` runs again — 67 tests, ~1.3s, no network.
- `response_format` is forwarded to providers verbatim and never rewritten by the router.
- Upstream HTTP status codes survive: 400/404/413/422/429 reach the caller unchanged, 5xx and
  auth failures against our own provider keys become 502. Error bodies moved to the OpenAI
  shape, top-level rather than nested under FastAPI's `detail`.
- Discovery no longer re-activates models the probe found dead; deactivations carry a reason
  and timestamp and are re-probed after 7 days.
- Added `scripts/probe_capabilities.py` and a verified capability inventory for 33 live models,
  and `scripts/nexus_gate.ts`, which runs nexus `callStructured` through the router.
