#!/bin/bash
# Ping healthchecks.io only when the router actually answers.
#
# Runs as its own launchd job, not from the server's scheduler: a process whose
# HTTP listener has wedged but whose APScheduler is still ticking would ping
# happily from the inside and page nobody. This asks the same question a caller
# asks — over the socket — and /health asserts a database round-trip and at
# least one live adapter before saying ok.
#
# HEALTHCHECKS_URL lives in .env alongside the provider keys. Unset means "not
# wired yet": check locally, log, and exit 0 rather than failing every 5 minutes.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${LLM_ROUTER_HEALTH_HOST:-127.0.0.1}"
PORT="${LLM_ROUTER_PORT:-8642}"

if [[ -f "$ROOT/.env" ]]; then
    HEALTHCHECKS_URL="$(grep -E '^HEALTHCHECKS_URL=' "$ROOT/.env" | tail -1 | cut -d= -f2- | tr -d '"'"'"'')"
fi
HEALTHCHECKS_URL="${HEALTHCHECKS_URL:-}"

body="$(curl -fsS -m 10 "http://${HOST}:${PORT}/health" 2>&1)"
rc=$?
stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ $rc -ne 0 ]]; then
    echo "$stamp UNHEALTHY (curl rc=$rc): $body"
    [[ -n "$HEALTHCHECKS_URL" ]] && curl -fsS -m 10 --data-raw "$body" "${HEALTHCHECKS_URL}/fail" >/dev/null
    exit 1
fi

echo "$stamp ok $body"
if [[ -n "$HEALTHCHECKS_URL" ]]; then
    curl -fsS -m 10 "$HEALTHCHECKS_URL" >/dev/null || echo "$stamp ping failed"
else
    echo "$stamp HEALTHCHECKS_URL unset — nothing is watching this service"
fi
