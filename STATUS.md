# STATUS — llm-router

Current state only. Rewritten in place, never appended. History lives in `CHANGELOG.md`.

## What exists and works

- **OpenAI-compatible API** on `:8642` — `/v1/chat/completions` (sync + SSE),
  `/v1/models`, `/v1/images/generations`, plus `/v1/quota`, `/v1/providers/health`,
  `/v1/models/health`, `/dashboard`, and the `/jobs` batch endpoints.
- **12 provider adapters**, 6 with keys set: openrouter, google, nvidia, deepseek,
  opencode, agnes. The other 6 (cerebras, groq, mistral, kilo, cloudflare,
  huggingface) are configured and idle.
- **Agent traffic works end to end.** `tools` / `tool_choice` are forwarded, an
  assistant message with `content: null` + `tool_calls` and a `role: "tool"` reply
  both round-trip, and multimodal content arrays are accepted. Proven live by a
  LangGraph `create_react_agent` turn through the router.
- **Streaming carries the whole delta and is metered.** Provider chunks pass through
  verbatim, so a streamed tool call reassembles into valid JSON arguments;
  `stream_options.include_usage` is honoured; the router no longer overwrites the
  provider's `finish_reason`; and streamed requests record quota, health and
  circuit-breaker success, which they previously bypassed entirely.
- **Structured output is honest.** `response_format` is forwarded verbatim and never
  rewritten. A model known not to support a strict schema is refused with a 400 the
  caller's own fallback recognises, rather than answering with prose.
- **Routing is capability-aware** on all three paths — `auto`, direct, and the family
  fallback. Zero capable candidates is an explicit 400. A provider's refusal
  (400/422) is returned immediately and does not count as ill health; it narrows
  that model's capability flags instead.
- **Per-model capability flags** are seeded from the verified inventory, dated by the
  file, and re-probed weekly with a per-provider budget. NULL means unknown.
- **The free-tier budget is measured, not assumed.** Account-wide caps are counted across
  every model on the key, `quota_remaining_pct` is a real number feeding the routing score,
  and `limits_known` marks the providers that publish no cap. The probe spends against the
  same ledger: 24h staleness window, 8 models per provider per cycle, and it stands down
  under 25% headroom. A startup that used to cost 39 provider requests now costs 0.
- **`auto` falls back across providers, not across one provider's models** — a rate-limited
  OpenRouter no longer ends the request.
- **Logging is configured**, so the router says what it is doing.
- **Secrets are redacted at the point of capture** — logs, `model_health`, circuit
  breaker state and the body a caller receives. `API_KEY` auth uses
  `hmac.compare_digest` and no longer accepts a key in the query string.
- **Batch jobs carry their parameters.** A queued job's `response_format` / `tools`
  reach the provider.
- **Tests:** `uv run pytest tests/` — 124 tests, ~1.5s, and the suite now *enforces*
  no network: a connect to anything but loopback fails with a pointer to
  `tests/fixtures/`.
- **The live gate:** `bash scripts/agent_gate.sh` — nexus `callStructured` (three
  cases) plus a LangGraph ReAct turn, against real providers. Currently green.
  Costs roughly 10 free-tier requests.

## In progress

- Branch **`sprint/agent-ready`** — 9 commits ahead of `master`, unmerged, unpushed.
  It executes `plans/SPRINT-agent-ready.md`; every ticket is done except the deploy.

## Known broken

- **Still not deployed, and the deploy is one command away.** `deploy/` holds the
  launchd plists, an idempotent installer and the runbook; `/health` asserts a
  database round-trip and a live adapter; `scripts/heartbeat.sh` is a dedicated
  job. What is missing is getting this branch onto agent-mini, which needs a push:

      git push ssh://agent/Users/ding/projects/llm-router sprint/agent-ready
      ssh agent 'cd ~/projects/llm-router && git checkout sprint/agent-ready && uv sync'
      ssh agent 'echo "API_KEY=$(openssl rand -hex 24)" >> ~/projects/llm-router/.env'
      ssh agent 'cd ~/projects/llm-router && bash deploy/install.sh'

  agent-mini's uncommitted pre-switch state is saved at
  `~/llm-router-preswitch-20260819.patch`.
- **agent-mini is running the old build, exposed.** A hand-started process is
  listening on `*:8642` — every interface, no `API_KEY`, pre-redaction code. That is
  an unauthenticated pass-through to eleven provider accounts on the LAN. The deploy
  above replaces it with a loopback-bound, authenticated launchd job.
- **`GOOGLE_AI_API_KEY` is in plaintext** in `/Users/ding/projects/llm-router/llm-router.log`
  on agent-mini and needs rotating. New leaks are closed; this one predates the fix.
- **The DeepSeek account has no balance** — every call returns HTTP 402. This is not
  a router fault, and it also means job-hunter's DeepSeek-direct default is dead.
- **Nothing is watching the service.** `HEALTHCHECKS_URL` is unset, so the heartbeat
  checks and logs but pages nobody. Create a check with a 10-minute period.
- **Google, Cloudflare and HuggingFace declare `tools: False`** and are excluded from
  agent traffic by design. Google's native function-calling translation is deferred,
  not attempted and broken.
- **Reasoning models leak their scratchpad into `content`** — `nemotron-3.5-lightning`
  returns its full "Here's a thinking process:" preamble as message content. Unowned
  by any ticket; either consumers tolerate it or the router splits `reasoning` out.
- **Nothing writes a per-request log.** `quota_usage` has no status, latency or request id
  and is deleted after 48h, so "why was that answer slow / wrong" is unanswerable after the
  fact. The next observability step.
- **Six providers have no key set** (cerebras, groq, mistral, kilo, cloudflare,
  huggingface) and are dead weight in scoring. Groq is the strongest free
  tool-calling tier available — 30 RPM / 14,400 RPD — and a signup is worth more
  than any routing change.

## Next

1. Deploy (the four commands above), then rotate `GOOGLE_AI_API_KEY`.
2. Set `HEALTHCHECKS_URL` so a dead router pages instead of going unnoticed.
3. Sign up for Groq; it changes what `auto` can do for agent traffic more than
   anything left in the code. Its free tier is also the only one whose published limits
   would give the new ledger real headroom to route against.
4. A `request_log` table where `record_usage` already writes — the first data that is not
   derivable from anything else, and the thing that makes a bad run diagnosable.
5. Then the job-hunter cutover — its `LlmProvider` seam (DESIGN.md D11) is what this
   sprint made implementable. That is job-hunter's ticket to schedule, and its
   DeepSeek-direct default must not switch until the router has run green for a week.

## Reference

- **opencode.ai zen free tier is metered per model, not per account.** Measured
  19 Aug: `nemotron-3-ultra-free`, `nemotron-3.5-lightning-free`, `hy3-free` and
  `laguna-s-2.1-free` answered, while `deepseek-v4-flash-free`, `mimo-v2.5-free` and
  `big-pickle` returned `FreeUsageLimitError` for at least ten minutes. The limit applies
  with or without an API key, so it is not per-credential either. Treat any single zen
  model as intermittently unavailable rather than free-and-unlimited.

- Deployment runbook: `deploy/README.md`.
- Verified model capabilities: `docs/model-capabilities-2026-08-18.json`
  (33 models probed; 18 return native `tool_calls`, 9 also honour strict
  `json_schema`). Refresh with `uv run python scripts/probe_capabilities.py`.
- The passthrough kill switch is `routing.passthrough_params` in `config.yaml`.
- Machine roles and where this should run: the `env` skill.
