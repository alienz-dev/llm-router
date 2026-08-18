#!/bin/bash
# The acceptance gate for the agent-ready sprint, in one command.
#
#   bash scripts/agent_gate.sh
#
# Two proofs against a real router and real providers:
#   1. nexus `callStructured` — the funnel's own structured-output path, including
#      the case where a provider refuses a schema and nexus must fall back fast.
#   2. A LangGraph ReAct turn — the strictest available check on the tool-calling
#      wire, including the turn-2 replay that used to 422.
#
# Spends roughly 8-10 provider requests. OpenRouter free is 50/day.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8642}"
ROUTER_URL="http://127.0.0.1:${PORT}/v1"
NEXUS="${NEXUS_DIR:-$HOME/projects/nexus}"
started_here=""

cleanup() {
    [[ -n "$started_here" ]] && kill "$started_here" 2>/dev/null
}
trap cleanup EXIT

if ! curl -fsS -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "── starting the router on :${PORT}"
    (cd "$ROOT" && .venv/bin/llm-router serve --host 127.0.0.1 --port "$PORT" \
        >/tmp/agent-gate-router.log 2>&1) &
    started_here=$!
    for _ in $(seq 1 40); do
        curl -fsS -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
        sleep 1
    done
fi

if ! curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "the router never became healthy — see /tmp/agent-gate-router.log" >&2
    exit 1
fi

rc=0

echo
echo "── nexus callStructured ──────────────────────────────"
if [[ -d "$NEXUS" ]]; then
    (cd "$NEXUS" && ROUTER_URL="$ROUTER_URL" npx --yes tsx "$ROOT/scripts/nexus_gate.ts") || rc=1
else
    echo "SKIP  no nexus checkout at $NEXUS"
fi

echo
echo "── LangGraph ReAct turn ──────────────────────────────"
ROUTER_URL="$ROUTER_URL" uv run --with langchain-openai --with langgraph \
    python "$ROOT/scripts/langgraph_smoke.py" || rc=1

echo
[[ $rc -eq 0 ]] && echo "agent gate: GREEN" || echo "agent gate: RED"
exit $rc
