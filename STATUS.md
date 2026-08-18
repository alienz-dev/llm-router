# STATUS — llm-router

Current state only. Rewritten in place, never appended. History lives in `CHANGELOG.md`.

## What exists and works

- **OpenAI-compatible API** on `:8642` — `/v1/chat/completions` (sync + SSE),
  `/v1/models`, `/v1/images/generations`, plus `/v1/quota`, `/v1/providers/health`,
  `/v1/models/health`, `/dashboard`, and the `/jobs` batch endpoints.
- **12 provider adapters**, 6 with keys set: openrouter, google, nvidia, deepseek, opencode,
  agnes. The other 6 (cerebras, groq, mistral, kilo, cloudflare, huggingface) are configured
  and idle.
- **Smart routing** — task classification, availability scoring
  (`capability × (quota_headroom + 0.1) × availability × provider_priority`), a fallback
  chain, per-provider circuit breakers persisted across restarts, and sliding-window quota
  tracking.
- **Structured output works end to end.** `response_format` is forwarded verbatim and never
  rewritten; providers that refuse `json_schema` return their own 400, fast, so a caller's
  fallback fires instead of retrying a 502. Proven against the real consumer by
  `scripts/nexus_gate.ts` (nexus `callStructured`, three cases).
- **Schema migrations** — additive columns applied idempotently at startup and recorded in
  `schema_migrations`. `DATABASE_PATH` is honoured, so tests no longer write the dev database.
- **Model lifecycle is coherent** — the probe's verdict wins over discovery's listing;
  deactivations carry a reason and timestamp and are re-probed after 7 days.
- **Tests:** `uv run pytest tests/` — 67 tests, ~1.3s, no network. The live-server smoke
  script is `scripts/smoke_live_server.py` and is not part of that run.

## In progress

- Branch **`sprint/agent-ready`** — 5 commits ahead of `master`, unmerged, unpushed. Contains
  everything above plus the carried-over image-generation and Agnes work. The plan it
  executes is `plans/SPRINT-agent-ready.md`.

## Known broken

- **Tool calling does not work.** `tools`/`tool_choice` are dropped silently (HTTP 200, model
  answers as if it had no tools). Assistant messages with `content: null` + `tool_calls`, and
  `role: "tool"` messages, are rejected with 422 — so a LangGraph agent dies on turn 2.
- **Streaming is thin and unmetered.** Only `delta.content` survives; `stream_options.include_usage`
  is ignored; streamed requests bypass quota recording, health updates and the circuit breaker
  entirely. A stream failure is yielded as assistant content.
- **No per-model capability metadata.** The columns exist (migrations 003-006) but nothing
  fills or reads them, so routing cannot avoid a model that ignores tools or refuses schemas.
- **API keys leak.** `discovery.py:114`, `probe.py:73`, `google.py:79,114` log URLs containing
  `?key=`, and `probe.py` writes the error into `model_health.last_probe_error`, which
  `/v1/models/health` and `/dashboard` serve without auth by default. `GOOGLE_AI_API_KEY` is
  in `llm-router.log` in plaintext and needs rotating.
- **Batch jobs drop request parameters.** `queue.py:116` calls `route()` with no kwargs, so a
  queued job loses `response_format` exactly as the sync path used to.
- **Not deployed.** agent-mini has no launchd job and nothing listening on 8642; its database
  was last written 27 Jul. `llm-router.service` in this repo is a systemd unit, which
  agent-mini does not use.
- **`/v1/models` returns bare model ids** without the provider prefix, so ids collide across
  providers and a client cannot pick one for direct routing from that list.

## Next

1. Redact secrets at the point of capture, then rotate `GOOGLE_AI_API_KEY`.
2. Deploy on agent-mini — launchd, explicit bind, `API_KEY` set, a `/health` that asserts a
   DB round-trip, and a heartbeat from a dedicated job.
3. Then either the agent wire (tools + streaming, which must ship together) or the
   job-hunter cutover — see `plans/`.

## Reference

- Verified model capabilities: `docs/model-capabilities-2026-08-18.json`
  (33 models probed; 18 return native `tool_calls`, 9 also honour strict `json_schema`).
  Re-run with `uv run python scripts/probe_capabilities.py`.
- Machine roles and where this should run: the `env` skill.
