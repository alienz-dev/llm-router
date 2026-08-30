# Deploying llm-router

It runs on **agent-mini** (`ssh agent`), under launchd, as user `ding`. The MacBook is
for development; it sleeps and travels, and a gateway that half the funnel depends on
cannot live there.

```bash
ssh agent 'cd ~/projects/llm-router && git pull && uv sync && bash deploy/install.sh'
```

`install.sh` is idempotent — it reloads both jobs from `deploy/` and refuses to finish
unless `/health` answers.

## The two jobs

| Job | What it does |
|---|---|
| `com.llmrouter.serve` | The server. `KeepAlive` restarts it on any non-zero exit; `RunAtLoad` brings it back after a reboot. |
| `com.llmrouter.heartbeat` | Every 5 minutes, curls `/health` **over the socket** and pings healthchecks.io. Separate on purpose: a wedged HTTP listener with a live APScheduler would ping happily from inside the process. |

Logs: `/Users/Shared/hub/logs/llm-router.log` and `…/llm-router-heartbeat.log`.

## Bind and auth

Bound to `127.0.0.1`. Nothing off-box needs it — job-hunter and the nexus funnel run on
agent-mini too — and an unauthenticated `0.0.0.0` listener is a pass-through to eleven
provider accounts. `API_KEY` is required regardless of bind, because every other job and
user on this machine can reach loopback:

```bash
echo "API_KEY=$(openssl rand -hex 24)" >> ~/projects/llm-router/.env
```

To use it from the MacBook, forward the port rather than opening the bind:

```bash
ssh -N -L 8642:127.0.0.1:8642 agent
curl -H "Authorization: Bearer $API_KEY" http://127.0.0.1:8642/v1/models
```

## Paging

`scripts/heartbeat.sh` pings `HEALTHCHECKS_URL` (in `.env`, next to the provider keys) on
success and `$HEALTHCHECKS_URL/fail` with the failure body on failure. **Until that URL is
set, nothing is watching this service** — the script says so in its log every 5 minutes.
Create a check at healthchecks.io with a 10-minute period and a 5-minute grace.

## What `/health` asserts

A database round-trip and at least one constructed adapter. It makes no provider calls —
a free-tier request budget is not something to spend on liveness. It is exempt from
`API_KEY` auth so the heartbeat needs no secret.

## Verifying

```bash
ssh agent 'launchctl list | grep llmrouter'                  # both jobs, PID + last exit
ssh agent 'curl -s localhost:8642/health'                    # {"status":"ok", …}
ssh agent 'nc -z 192.168.31.170 8642 && echo REACHABLE'      # must NOT print REACHABLE
ssh agent 'tail -20 /Users/Shared/hub/logs/llm-router-heartbeat.log'
```
