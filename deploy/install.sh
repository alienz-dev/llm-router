#!/bin/bash
# Install llm-router as a launchd service on agent-mini. Run it there, not here.
#
#   ssh agent 'cd ~/projects/llm-router && bash deploy/install.sh'
#
# Idempotent: re-running reloads both jobs with whatever is currently in deploy/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
LOGS="/Users/Shared/hub/logs"
JOBS=(com.llmrouter.serve com.llmrouter.heartbeat)
UID_NUM="$(id -u)"

if [[ ! -x "$ROOT/.venv/bin/llm-router" ]]; then
    echo "no .venv/bin/llm-router in $ROOT — run 'uv sync' first" >&2
    exit 1
fi

# Loopback-bound, but the key is still required: anything that can reach
# 127.0.0.1 on this box (every other job, every user) can otherwise spend the
# whole provider set.
if ! grep -qE '^API_KEY=.+' "$ROOT/.env" 2>/dev/null; then
    echo "API_KEY is not set in $ROOT/.env — refusing to expose the provider keys" >&2
    echo "  generate one:  echo \"API_KEY=\$(openssl rand -hex 24)\" >> $ROOT/.env" >&2
    exit 1
fi

mkdir -p "$AGENTS" "$LOGS"

for job in "${JOBS[@]}"; do
    launchctl bootout "gui/$UID_NUM/$job" 2>/dev/null || true
    # The plists hardcode /Users/ding — refuse to install them anywhere else
    # rather than half-install a job that will never start.
    if ! grep -q "$ROOT" "$ROOT/deploy/$job.plist"; then
        echo "$job.plist points somewhere other than $ROOT — edit it first" >&2
        exit 1
    fi
    cp "$ROOT/deploy/$job.plist" "$AGENTS/$job.plist"
    launchctl bootstrap "gui/$UID_NUM" "$AGENTS/$job.plist"
    echo "loaded $job"
done

echo "waiting for the server to answer…"
for _ in $(seq 1 20); do
    if curl -fsS -m 2 http://127.0.0.1:8642/health >/dev/null 2>&1; then
        echo "healthy: $(curl -fsS -m 5 http://127.0.0.1:8642/health)"
        exit 0
    fi
    sleep 1
done

echo "server did not come up — tail $LOGS/llm-router.log" >&2
exit 1
