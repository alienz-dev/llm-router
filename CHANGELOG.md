# Changelog

Append-only. One dated line per shipped change. Current state lives in `STATUS.md`.

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
