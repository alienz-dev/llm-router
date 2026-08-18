# Sprint Plan — Agent-Ready Router

**Goal.** Make llm-router something the nexus funnel can point at without silently losing its
structured output, and something job-hunter's `LlmProvider` (DESIGN.md D11) can later
implement against: OpenAI-wire-correct for structured output and tool calling,
capability-aware in its routing, and running on agent-mini where its death is loud.

**Not a goal:** adopting LiteLLM, the Responses API, an MCP server, or a LangChain `ChatModel`
layer. job-hunter's D11 already fixed the seam — a provider interface — and this sprint only
makes a second implementation of it possible. Writing that implementation is job-hunter's
sprint, not this one.

Draft reviewed adversarially 18 Aug 2026; the ticket order below is the post-review order, and
every code reference in the diagnosis was re-verified against the working tree.

**Progress:** every ticket is shipped on branch `sprint/agent-ready` except the
deployment half of T6 — the artifacts exist and are tested, but the branch has not been
pushed to agent-mini, so nothing is running there yet. Current state is in `STATUS.md`;
what shipped is in `CHANGELOG.md`.

---

## Why now — what was measured on 18 Aug 2026

Verified against the live router and live providers, not inferred.

**1. The router silently drops `tools`.** `ChatCompletionRequest` (`llm_router/models.py:21-26`)
declares only `messages, model, stream, max_tokens, temperature`; Pydantic's default
`extra="ignore"` discards the rest without complaint. Through the router,
`openrouter:nvidia/nemotron-3-nano-30b-a3b:free` answered *"I currently don't have access to
live weather data or tools"* — HTTP 200. The same model and prompt called directly returned a
well-formed `tool_calls` object. Silent, not a 422: the caller cannot detect it.

**2. A LangGraph turn 2 is a 422.** `ChatMessage.content` is `str`, non-optional
(`models.py:16-18`), and there are no `tool_calls` / `tool_call_id` fields, so the standard
replay — assistant message with `content: null` + `tool_calls`, then a `role: "tool"` message —
is rejected. Multimodal content arrays 422 for the same reason. Both verified by curl.

**3. `response_format` is dropped, and that is what breaks the funnel.** nexus's
`callStructured` (`~/projects/nexus/src/llm/structured.ts:89-118`) sends
`response_format: json_schema` and falls back to `json_object` + schema-in-prompt **only when
the provider errors**. DeepSeek returns exactly `"This response_format type is unavailable
now"`, which is why job-hunter works today against DeepSeek direct. Through the router the
field vanishes, nothing errors, the fallback never fires, prose comes back, `jsonrepair` fails,
and the call burns its retry budget before throwing. This is the shape of the 25 Jul – 9 Aug
incident: 9,919 listings scraped, 38 scored.

**4. Errors come back as 502, which makes it worse.** `app.py:157` wraps every upstream failure
as HTTP 502; `nexus/src/llm/client.ts:121` retries on any status ≥ 500 with exponential backoff
up to 30s. So a provider's clean, fast 400 ("response_format unavailable") is converted into
three slow retries and then a throw — the fallback path nexus was designed around never runs.

**5. Streaming carries content only.** `adapters/base.py:170` extracts `delta.content` and
nothing else; `stream_options: {include_usage: true}` is ignored (verified: no usage chunk);
`[DONE]` is correct. Tool-call deltas cannot survive this path. Worse, streaming returns the
adapter's iterator directly (`router.py:72-75, 336-337, 393-394`) — **no quota recording, no
health update, no circuit-breaker success** — so streamed traffic is invisible to the free-tier
budgeting this whole design rests on. And `base.py:176` yields the exception string as
assistant content, so a stream failure reads as a model answer.

**6. Routing is capability-blind, and two paths bypass routing entirely.** Scoring is
`capability × (quota_headroom + 0.1) × availability × provider_priority`
(`router.py:523-529`) where "capability" is a task score; no model carries a tool-calling or
schema flag. Beyond that, `_route_direct` (`router.py:285-356`) and `_try_fallback_for_model`
(`router.py:358-406`) go straight to the adapter — and the family fallback picks its
replacement on a bare `LIKE '%family%'` match (`router.py:376`).

**7. It is not running.** agent-mini has no llm-router launchd job, no process, nothing on
8642; its `llm_router.db` was last written 27 Jul. The repo's `llm-router.service` is a systemd
unit, which agent-mini does not use. `config.yaml:2` binds `0.0.0.0` and `APIKeyMiddleware`
only mounts when `API_KEY` is set (`app.py:117-122`) — the default posture is an
unauthenticated pass-through to the provider accounts.

**8. Discovery re-animates dead models.** `discovery.py:405-409` sets `active = 1`
unconditionally on conflict, while `probe.py:170` only probes `active = 1` rows. So the probe
deactivates a dead model and the next discovery run revives it. Six NVIDIA models are active
right now and return HTTP 404.

**9. An API key is printed to the log.** `discovery.py:114` logs the stringified httpx error
whose message embeds the URL built at `discovery.py:139` (`?key={api_key}`). The same shape
exists at `probe.py:73-77` and `google.py:79,114`, and `probe.py:161-162` writes `str(e)[:80]`
into `model_health.last_probe_error` — which is served by `/v1/models/health` and `/dashboard`,
unauthenticated by default. The current `GOOGLE_AI_API_KEY` is sitting in `llm-router.log` in
plaintext and needs rotating.

**10. There is no schema migration mechanism at all.** `db.py:24-116` is
`CREATE TABLE IF NOT EXISTS` and nothing else — no `schema_version`, no `ALTER TABLE` anywhere
in the repo. Any new column is invisible to every existing database, including agent-mini's.

---

## Verified model inventory — 33 models probed live

Two probes per model, direct to the provider: a `tools` call and a strict `json_schema` call.
Raw results in `docs/model-capabilities-2026-08-18.json`; re-runnable via
`uv run python scripts/probe_capabilities.py`.

**Agent-ready — native `tool_calls` *and* strict `json_schema` (9):**

| Model | latency |
|---|---|
| `openrouter:nvidia/nemotron-nano-12b-v2-vl:free` | 1.3s |
| `agnes:agnes-2.0-flash` | 1.3s |
| `openrouter:nvidia/nemotron-3-nano-30b-a3b:free` | 1.8s |
| `openrouter:google/gemma-4-26b-a4b-it:free` | 2.1s |
| `agnes:agnes-2.5-flash` | 2.7s |
| `openrouter:nvidia/nemotron-3-ultra-550b-a55b:free` | 2.7s |
| `openrouter:nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | 7.6s |
| `opencode:nemotron-3-ultra-free` | 10.2s |
| `nvidia:mistralai/mistral-nemotron` | 21.6s |

**Tools yes, strict schema no (9):** `deepseek:deepseek-v4-pro`, `deepseek:deepseek-v4-flash`,
`agnes:agnes-2.5-pro`, `agnes:agnes-2.5-pro-alpha`,
`openrouter:nvidia/nemotron-3.5-lightning:free`,
`openrouter:nvidia/nemotron-3-super-120b-a12b:free`, `openrouter:cohere/north-mini-code:free`,
`openrouter:dots-studio/dots-3-note-preview:free`, `openrouter:openrouter/free`. These need the
`json_object` + schema-in-prompt path — which is nexus's own fallback, and must stay nexus's
decision.

**Ignore tools entirely (2):** `openrouter:nvidia/nemotron-nano-9b-v2:free`,
`openrouter:poolside/laguna-xs-2.1:free` — must never receive agent traffic.

**Dead or wrong-modality (13):** six NVIDIA 404s, two Agnes image models, two Lyria music
models, a content-safety classifier, and three rate-limited at probe time.

**Stability caveat:** a second run of the same probe an hour later flipped two models
(`nemotron-nano-12b-v2-vl` and the omni-reasoning model) from `schema-ok` to malformed/not-json
output. Capability results are a snapshot with noise, not a constant — which is exactly why
they need a `checked_at` and a re-probe cadence rather than a one-time table.

**Quota reality that shapes the design:** OpenRouter free is 20 RPM and **50 requests/day**
without ≥$10 lifetime credit. A ReAct loop spends several calls per task, so OpenRouter alone
cannot carry agent traffic — spreading across agnes / opencode / nvidia / deepseek is the
entire reason this router exists. Six of twelve providers (cerebras, groq, mistral, kilo,
cloudflare, huggingface) have **no key set** and are dead weight in scoring until they do.

---

## ADR-01 — who owns a downgrade, and who owns capability truth

Two rules the whole sprint hangs on. Write them down before any ticket starts.

**The router never rewrites `response_format`.** If a provider cannot honour a schema, the
router surfaces that provider's refusal — status and body — and the *caller* decides what to
do. A server-side downgrade from `json_schema` to `json_object` looks helpful and is the 25 Jul
incident with extra steps: nexus only injects the schema into the prompt on *its own* fallback
path, so a silent server-side downgrade produces valid JSON of arbitrary shape, which zod
rejects after burning the retry budget.

**Provider-level "no" beats model-level "yes".** Adapter declarations (R6) are a hard ceiling;
probe results (R7) can only narrow within them. A model probed tool-capable through one
provider does not become tool-capable through an adapter that cannot express tools.

**And when no capable candidate exists, `auto` fails loudly** — an explicit 400 naming the
missing capability, never a silent fall-through to a model that ignores it.

---

## Tickets

### Wave 0 — clear the ground (half a day, blocks everything)

**✅ T0 · Land or shelve the working tree.** *SHIPPED (62c14f4) — plus the `error_response` NameError fix.*

 +263 uncommitted lines sit across the exact files
this sprint edits (`router.py` +153, `base.py` +34, `app.py` +20, `models.py` +13,
`probe.py` +18, `discovery.py` +15) plus untracked `adapters/agnes.py`. It carries a live bug:
`error_response` is imported function-locally at `app.py:156` inside `chat_completions` but
called at `app.py:182` in the image route — that error path raises `NameError`. Two of the nine
agent-ready models are Agnes models behind that untracked adapter.
*Accept:* clean `git status`, image error path returns a 502 body instead of raising.

**✅ T1 · A schema migration mechanism.** *SHIPPED (e8e828d) — `schema_migrations` + six columns declared; test builds an old-schema database and migrates it.*

 `PRAGMA table_info` → `ALTER TABLE ADD COLUMN`, applied
idempotently at startup, plus a `schema_version` row. Nothing else in this sprint can add a
column until this exists.
*Owns:* `llm_router/db.py`.
*Accept:* a test that opens a database built from the *old* DDL, runs startup, and finds the
new columns — not a test against a freshly-created schema.

### Wave 1 — the nexus unblock (one day, ships on its own)

This wave alone makes the funnel safe to point at the router. It touches no message shapes and
no capability system.

**✅ T2 · Forward `response_format`.** *SHIPPED (450df23).*

 Add it to `ChatCompletionRequest` and pass it to the
adapter. Nothing else — no `extra="allow"`, no tools, no message-shape change. Per ADR-01 the
router never rewrites or downgrades it.
*Owns:* `llm_router/models.py`, `llm_router/app.py`.
*Accept:* a `json_schema` request to `openrouter:google/gemma-4-26b-a4b-it:free` through the
router returns schema-conforming JSON; the same request to `deepseek:deepseek-v4-pro` returns
DeepSeek's own 400.

**✅ T3 · Return the upstream status, not 502.** *SHIPPED (450df23) — gate met: nexus falls back on attempt 0, DeepSeek path 4.0s.*

 Carry the provider's status code on
`AdapterResponse` (`adapters/base.py:30-36` has no status field today) and surface 4xx as 4xx.
This is what lets nexus fall back immediately instead of retrying a 502 three times with
backoff. Consumers of the `{"error": str}` convention all change: five sites in `router.py`,
`queue.py:119-120`, and the bespoke adapters that build their own error dicts
(`google.py:26,86`, cloudflare, huggingface). Also decide and document what `auto` does with a
capability refusal — today the loop swallows it into `"All fallback attempts failed: …"`
(`router.py:82-100`) and the provider's words are lost.
*Owns:* `adapters/base.py`, `adapters/*.py`, `router.py`, `app.py`, `queue.py`.
*Accept:* the mechanical gate is **nexus's retry count, not the error text** — a
`callStructured` against `deepseek:deepseek-v4-pro` through the router falls back on attempt 0
with no retry sleep; measured elapsed under 5s where today it is 3 retries with exponential
backoff.

**✅ T4 · Discovery stops resurrecting dead models.** *SHIPPED (1b70681).*

 Remove the unconditional `active = 1` at
`discovery.py:405-409`; add `deactivated_reason` and a backoff re-check so a transient 404 does
not become a permanent graveyard (`probe.py:170` only probes `active = 1`, so a deactivated
model is otherwise never seen again).
*Owns:* `discovery.py`, `probe.py`, plus a column via T1.
*Accept:* the six NVIDIA 404s stay `active = 0` across a discovery run; a model deactivated
7 days ago is re-checked once.

**✅ T5 · Stop logging secrets.** *SHIPPED (66cc2b8) — redaction at the point of capture; `compare_digest`; no `?key=` fallback. Rotating `GOOGLE_AI_API_KEY` is still outstanding and is Ming's to do.*

**T5 · Stop logging secrets.** Redact query strings and `Authorization` headers at the point of
capture, not at the point of logging — the sinks are `discovery.py:114`, `probe.py:73-77`,
`google.py:79,114`, and `probe.py:161-162` which writes the error into `model_health`, served
publicly by `/v1/models/health` and `/dashboard`. Rotate `GOOGLE_AI_API_KEY` afterwards. Fold
in the one-line `hmac.compare_digest` fix at `app.py:37` and drop the `?key=` query-param
auth fallback at `app.py:40`.
*Owns:* `discovery.py`, `probe.py`, `adapters/google.py`, `app.py`.
*Accept:* after a forced failure, `grep -r AIza` across logs **and** a scan of
`model_health.last_probe_error` **and** the `/v1/models/health` response all return nothing.

**◐ T6 · Deploy on agent-mini, properly.** *ARTIFACTS SHIPPED (c947c6c) — launchd plists, idempotent installer, dedicated heartbeat job, a `/health` that asserts a DB round-trip and a live adapter, systemd unit deleted. NOT INSTALLED: pushing the branch to agent-mini needs an approval this session did not have. Commands are in `STATUS.md`.*

**T6 · Deploy on agent-mini, properly.** launchd plist (not the repo's systemd unit), an
explicit bind decision (`127.0.0.1` unless something off-box needs it) with `API_KEY` set
either way, a `/health` that actually asserts a DB round-trip and one adapter rather than
returning `{"status":"ok"}` unconditionally (`app.py:126-128`), and a healthchecks.io ping from
a **dedicated** job — pinging from the discovery cycle means a wedged HTTP server with a live
scheduler never pages. Delete or clearly mark `llm-router.service`. Ships after T5, never
before.
*Owns:* new `deploy/`, `app.py`, `llm-router.service`.
*Accept:* survives a reboot; `nc -z` from another host fails if bound to loopback; killing the
process pages within the stated grace period.

### Wave 2 — the agent wire (one day; T7 and T8 ship together or not at all)

**✅ T7 · Message shape and tool parameters.** *SHIPPED (c947c6c) — reserved-key deny-list, per-provider allow-list, and `routing.passthrough_params` as the kill switch.*

**T7 · Message shape and tool parameters.** `ChatMessage.content: str | list[dict] | None`,
plus `tool_calls`, `tool_call_id`, `name`; request gains `tools`, `tool_choice`,
`parallel_tool_calls`, `stream_options`, `stop`, `seed`, `top_p`, `n`, `user`. Three guards the
draft missed:
- **Reserved keys.** `base.py:129` and `base.py:163` splat `**kwargs` *after* `messages`,
  `model` and `stream`, so a caller-supplied `model` would re-target the upstream to something
  the router never scored, quota-checked, or recorded. Deny-list them explicitly.
- **Per-provider allow-list.** Some providers 400 on unknown fields. Forwarding `seed`/`stop`/
  `presence_penalty` blindly turns requests that worked yesterday into failures — and via T9
  those failures would degrade model health.
- **A kill switch.** A `passthrough_params` config flag defaulting to today's behaviour makes
  this whole wave reversible in one line.
Two things are already safe and narrow the blast radius: `quota.py:22-25` handles multimodal
content arrays, and `router.py:409-411` guards `classify_task` with `isinstance(content, str)`.
*Owns:* `models.py`, `app.py`, `adapters/base.py`, `config.yaml`.
*Accept:* `tests/fixtures/langgraph_turn2.json` returns 200; a `tools` request returns
`tool_calls` with non-empty `id` and `finish_reason: "tool_calls"`; a body carrying
`"model"` as an extra key does not change the upstream target.

**✅ T8 · Stream the whole delta, and meter it.** *SHIPPED (c947c6c) — verified live: a streamed tool call reassembles, usage arrives before `[DONE]`, and the request lands in `quota_usage`.*

**T8 · Stream the whole delta, and meter it.** Pass the provider's `delta` through verbatim so
`tool_calls` fragments keep `index`/`id`/partial `arguments`; honour
`stream_options.include_usage`; stop yielding exception strings as assistant content
(`base.py:176`, `base.py:155`); and record quota, health and breaker success for streamed
requests, which today bypass all three. **T7 without T8 is a regression** — once `tools` is
forwarded, a streaming tool-call response yields nothing at all, so HTTP 200 with an empty
completion replaces today's wrong-but-visible prose. If T8 must slip, T7 ships with an explicit
400 on `tools` + `stream: true`.
*Owns:* `adapters/base.py`, `app.py:190-204`, `router.py:72-75,336-337,393-394`.
*Accept:* a streamed tool call reassembles into valid JSON arguments; the last chunk before
`[DONE]` carries `usage`; a streamed request increments quota usage.

**✅ T9 · Capability enforcement, in one place, on every path.** *SHIPPED (c947c6c).*

**T9 · Capability enforcement, in one place, on every path.** `BaseAdapter` declares
`supports = {"tools": …, "json_schema": …, "streaming": …}`; `google`, `cloudflare` and
`huggingface` declare `tools: False` (Google's native function-calling translation is
deliberately deferred). Enforcement covers all three routing paths — `auto`, `_route_direct`
(`router.py:285-356`) and the family fallback (`router.py:358-406`), whose `LIKE '%family%'`
match is exactly the "a retry must not land on an incapable model" case. Zero capable
candidates → explicit 400 per ADR-01. And **a capability refusal must not count as a health
failure**: `router.py:96-99` and `:352-355` call `health_repo.update(success=False)` on every
error, `router.py:470` hard-skips at 5 consecutive failures and `:510` applies a 0.2× penalty
at 3 — five schema refusals would evict a healthy model from `auto`.
*Owns:* `adapters/*.py`, `router.py`.
*Accept:* a fake-adapter test proves a `tools` request never reaches an adapter declaring
`tools: False`, at any position in any of the three paths; five refusals leave the model's
health score unchanged.

**✅ T10 · The batch path gets the same treatment.** *SHIPPED (c947c6c).*

**T10 · The batch path gets the same treatment.** `queue.py:116` calls `route()` with no
kwargs, so a queued job's `tools`/`response_format` are dropped exactly as the sync path drops
them today — the same defect at a second endpoint.
*Owns:* `queue.py`, `models.py:52`.
*Accept:* a queued job with `response_format` produces schema-conforming output.

### Wave 3 — capability data and proof (one day)

**✅ T11 · Capability columns, on a budget.** *SHIPPED (8d5ec1c).*

**T11 · Capability columns, on a budget.** `supports_tools`, `supports_json_schema`,
`supports_vision`, `capability_checked_at` on `models`, via T1's migration, seeded from
`docs/model-capabilities-2026-08-18.json`. **NULL means unknown, not unsupported** — on the day
the migration lands every row is NULL, and a `WHERE supports_tools = 1` filter would empty the
candidate set. Capability probing runs **weekly**, not on the 2-hourly health cycle: two extra
calls per model per cycle would be ~720 OpenRouter requests/day against a 50/day cap. The probe
reads `capability_checked_at` and skips fresh rows.
*Owns:* `db.py`, `probe.py`, `scripts/probe_capabilities.py`.
*Accept:* an old-schema database migrates and the nine known agent-ready models carry
`supports_tools = 1`; a second run inside the freshness window makes zero provider calls.

**✅ T12 · Hermetic tests.** *SHIPPED (4c71528) — 124 tests, and the suite now fails any attempt to reach the network.*

**T12 · Hermetic tests.** Every acceptance criterion above that hits a live provider is
non-repeatable against a 50/day budget. Capture fixtures for the `tool_calls` shape, DeepSeek's
`response_format` 400, and a streaming tool-call delta sequence. This also unblocks the rest:
`get_db()` is a module-level singleton hardcoded to `Path("llm_router.db")` (`db.py:4-14`) and
`DATABASE_PATH` is set in `tests/conftest.py:11` but read nowhere — so tests currently write to
the dev database, and `tests/test_router.py` re-implements the scoring formula inside the test
file because it cannot reach the real one.
*Owns:* `tests/`, `db.py`.
*Accept:* the suite passes with no network access.

**✅ T13 · Prove it against the real consumer.** *SHIPPED (4c71528) — `bash scripts/agent_gate.sh`, green. The schema-refusing case moved off DeepSeek: that account returns HTTP 402 Insufficient Balance.*

**T13 · Prove it against the real consumer.** A nexus `callStructured` run through the router —
the funnel's judge prompt, 10 real listings, all 10 parsing — plus a LangGraph
`create_react_agent` smoke test over `ChatOpenAI(base_url=…)` in a scratch venv. The LangGraph
test is not because job-hunter's graph uses the LangChain model layer (D8/D11 say it does not)
but because it is the strictest available check on the wire protocol.
*Owns:* `tests/test_openai_compat.py`, `scripts/`.
*Accept:* both run green from one command. **Explicitly out of scope:** writing the API-key
`LlmProvider` implementation in job-hunter — that is cross-repo work during job-hunter's own
sprint, and it is their ticket to schedule.

---

## Sequencing

**Wave 0 → Wave 1** is the coherent minimum and is genuinely about a day and a half:
T0, T1, T2, T3, T4, T5, T6. It ships the entire nexus win, stops `auto` picking dead models,
closes the key leak, and means nobody discovers the router is down three weeks later. It adds
no half-built capability system and no message-shape risk.

**Wave 2** is the agent wire and is all-or-nothing across T7+T8. **Wave 3** is only worth
starting once Wave 2 is green.

## Risks

- **job-hunter must not switch its default this sprint.** DeepSeek-direct stays primary in
  `src/lib/context.ts`; the router is opt-in via `LLM_ENDPOINT` until it has run green for a
  week. The 25 Jul incident was caused by exactly that dependency.
- **Free tiers move without notice.** Cerebras cut its free catalogue from ~12 models to 2 in
  May 2026 with no deprecation notice. Capability flags carry `checked_at` and are re-probed,
  never trusted indefinitely — and the flip observed between two probe runs an hour apart says
  the same thing at a shorter timescale.
- **Groq is the strongest free tool-calling tier available** (30 RPM / 14,400 RPD) and has no
  key set. A signup before the sprint is worth more than any routing improvement inside it.
- **Reasoning models leak their scratchpad into `content`** — `nemotron-3.5-lightning` returned
  its full "Here's a thinking process:" preamble as message content. Either consumers tolerate
  it or the router splits `reasoning` out; unowned by any ticket so far.
