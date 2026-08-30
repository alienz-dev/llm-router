"""FastAPI application with all routes."""
import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .adapters import ADAPTERS
from .capabilities import seed_from_inventory
from .logging_setup import configure_logging
from .db import get_db, close_db
from .adapters.base import AdapterResponse
from .models import (
    ChatCompletionRequest,
    ImageGenerationRequest,
    JobSubmission,
    error_response,
)
from .quota import QuotaManager
from .redact import redact_error
from .queue import JobQueue
from .router import SmartRouter
from .scheduler import AppScheduler

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Optional bearer token auth. If API_KEY env var is set, require it in requests.
    Health endpoint is always exempt."""

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        # Always allow health check
        if request.url.path == "/health":
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        # compare_digest, not ==: a plain comparison leaks the key one byte at a
        # time to anyone who can time the response.
        if hmac.compare_digest(auth, f"Bearer {self.api_key}"):
            return await call_next(request)
        # No ?key= fallback: query strings land in access logs, proxy logs and
        # Referer headers, which is exactly how this key would leak next.
        return Response(content='{"error":{"message":"Invalid API key","type":"auth_error"}}', status_code=401, media_type="application/json")

# --- Globals wired in lifespan ---
_router: SmartRouter | None = None
_quota: QuotaManager | None = None
_queue: JobQueue | None = None
_scheduler: AppScheduler | None = None


def _build_adapters() -> dict:
    """Instantiate adapters from env vars."""
    from .config import get_config
    cfg = get_config()
    adapters = {}
    for pid, pcfg in cfg.providers.items():
        key = os.getenv(pcfg.api_key_env, "")
        if not key:
            logger.warning("Skipping provider %s: %s not set", pid, pcfg.api_key_env)
            continue
        if pid == "cloudflare":
            acct = os.getenv(pcfg.account_id_env or "", "")
            if acct:
                adapters[pid] = ADAPTERS[pid](acct, key)
            else:
                logger.warning("Skipping provider cloudflare: %s not set", pcfg.account_id_env)
        elif pid in ADAPTERS:
            adapters[pid] = ADAPTERS[pid](key)
    if not adapters:
        logger.error("No providers configured! Set API keys in .env")
    else:
        logger.info("Initialized %d provider adapters: %s", len(adapters), ", ".join(adapters.keys()))
    return adapters


async def _seed_providers():
    """Seed providers table from config."""
    from .config import get_config
    cfg = get_config()
    db = await get_db()
    for pid, pcfg in cfg.providers.items():
        await db.execute(
            """INSERT OR IGNORE INTO providers (id, name, base_url, enabled, daily_reset_utc_hour)
               VALUES (?, ?, ?, 1, ?)""",
            (pid, pcfg.name, pcfg.base_url, pcfg.daily_reset_utc_hour),
        )
    await db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _router, _quota, _queue, _scheduler
    configure_logging()
    await get_db()
    await _seed_providers()
    # Idempotent: only fills rows that have never been checked, so a live probe
    # result always outranks the file.
    await seed_from_inventory()

    adapters = _build_adapters()
    _quota = QuotaManager()
    _router = SmartRouter(adapters, _quota)
    # Restore circuit breaker state from previous run
    await _router.circuit_breaker.load_state()
    _queue = JobQueue(_router)
    _scheduler = AppScheduler(_queue)
    await _scheduler.start()

    yield

    await _scheduler.stop()
    # Close all adapter clients
    for adapter in adapters.values():
        await adapter.close()
    await close_db()


app = FastAPI(title="LLM Router", version="0.1.0", lifespan=lifespan)

# Optional API key auth — set API_KEY env var to enable
_api_key = os.getenv("API_KEY", "")
if _api_key:
    app.add_middleware(APIKeyMiddleware, api_key=_api_key)
    logger.info("API key auth enabled")
else:
    logger.info("API key auth disabled (set API_KEY env var to enable)")


# Upstream statuses that describe the *request* rather than the provider's health.
# These reach the caller unchanged so it can act immediately instead of retrying
# something that will never succeed. Everything else — 5xx, auth failures against
# our own provider keys, connection errors — becomes a 502: the caller's request
# was fine, our upstream was not.
_PASSTHROUGH_STATUSES = frozenset({400, 404, 413, 422, 429})


def _upstream_error(result, fallback_type: str = "upstream_error") -> JSONResponse:
    """Render an adapter/router failure as an OpenAI-shaped error response.

    The error object sits at the top level, where OpenAI SDK clients look for it —
    FastAPI's HTTPException would nest it under "detail".
    """
    status = result.status if result.status in _PASSTHROUGH_STATUSES else 502
    return JSONResponse(
        status_code=status,
        content=error_response(result.response["error"], fallback_type),
    )


# ── Health ──────────────────────────────────────────────
@app.get("/health")
async def health(response: Response):
    """Assert the two things a wedged process loses first.

    The old handler returned {"status": "ok"} without touching anything, so a
    process with a dead database connection and zero adapters still looked
    healthy. That is worse than having no check at all, because the heartbeat
    believes it and nobody is paged.

    Deliberately cheap — no provider calls. This runs on a short interval and a
    free-tier request budget is not something to spend on liveness.
    """
    checks: dict[str, Any] = {}
    healthy = True

    try:
        db = await get_db()
        async with db.execute("SELECT COUNT(*) FROM providers") as cur:
            row = await cur.fetchone()
        checks["database"] = {"ok": True, "providers": row[0]}
    except Exception as e:
        healthy = False
        checks["database"] = {"ok": False, "error": redact_error(e, 200)}

    adapters = getattr(_router, "adapters", None) or {}
    checks["adapters"] = {"ok": bool(adapters), "count": len(adapters),
                          "providers": sorted(adapters)}
    if not adapters:
        healthy = False

    if not healthy:
        response.status_code = 503
    return {"status": "ok" if healthy else "unhealthy", "checks": checks}


# ── OpenAI-compatible: chat completions ─────────────────
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, response: Response):
    # exclude_none, so an assistant message that only calls a tool arrives as
    # {"role": "assistant", "tool_calls": [...]} rather than carrying an explicit
    # content: null that some providers reject.
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    extra_kwargs = req.passthrough_params()

    if req.stream:
        stream_iter = await _router.route(
            messages, model_override=req.model, stream=True, **extra_kwargs,
        )
        # The router can refuse before any stream exists — no capable model, no
        # quota, circuit breaker open.
        if not hasattr(stream_iter, "__aiter__"):
            return _upstream_error(stream_iter)
        # Look at the first chunk before committing to 200. Once the SSE body has
        # started there is no status left to send, and a client's retry logic
        # keys on the status.
        first, stream_iter = await _peek_stream(stream_iter)
        if isinstance(first, dict) and "error" in first:
            return _upstream_error(AdapterResponse(
                response={"error": first["error"]}, quota=None, tokens_in=0,
                tokens_out=0, latency_ms=0, status=first.get("status"),
            ))
        return StreamingResponse(
            _sse_wrap(stream_iter, req.model or "auto"),
            media_type="text/event-stream",
        )

    result = await _router.route(
        messages, model_override=req.model, stream=False, **extra_kwargs,
    )
    if "error" in result.response:
        return _upstream_error(result)

    meta = result.response.pop("_router", {})
    response.headers["x-llm-router-provider"] = meta.get("provider", "unknown")
    response.headers["x-llm-router-model"] = meta.get("model", req.model or "auto")
    if meta.get("fallback"):
        response.headers["x-llm-router-fallback"] = "true"
        if meta.get("original_request"):
            response.headers["x-llm-router-original-model"] = meta["original_request"]
    return result.response


# ── OpenAI-compatible: image generation ─────────────
@app.post("/v1/images/generations")
async def image_generations(req: ImageGenerationRequest, response: Response):
    result = await _router.route_image(
        prompt=req.prompt, model_override=req.model,
        n=req.n, size=req.size, response_format=req.response_format,
    )
    if "error" in result.response:
        return _upstream_error(result)
    meta = result.response.pop("_router", {})
    response.headers["x-llm-router-provider"] = meta.get("provider", "unknown")
    response.headers["x-llm-router-model"] = meta.get("model", req.model or "auto")
    return result.response


async def _peek_stream(stream):
    """Pull the first chunk, and hand back an iterator that still starts with it."""
    iterator = stream.__aiter__()
    try:
        first = await iterator.__anext__()
    except StopAsyncIteration:
        first = None

    async def _rewound():
        if first is not None:
            yield first
        async for chunk in iterator:
            yield chunk

    return first, _rewound()


async def _sse_wrap(stream, model_name: str):
    """Re-emit the provider's chunks, adding only what the wire requires.

    The chunk is passed through rather than rebuilt: `tool_calls` fragments carry
    `index`, `id` and partial `arguments`, and rebuilding around `delta.content`
    — which is what this did — silently dropped all of it, so a streamed tool
    call arrived as an empty completion.
    """
    chunk_id = f"chatcmpl-{int(time.time())}"
    saw_error = False
    saw_finish = False
    try:
        async for data in stream:
            if isinstance(data, str):
                # Legacy adapters that yield plain content.
                data = {"choices": [{"index": 0, "delta": {"content": data},
                                     "finish_reason": None}]}
            if "error" in data:
                saw_error = True
                yield f"data: {json.dumps(error_response(str(data['error']), 'upstream_error'))}\n\n"
                break
            data.setdefault("id", chunk_id)
            data.setdefault("object", "chat.completion.chunk")
            data.setdefault("created", int(time.time()))
            data.setdefault("model", data.pop("_model", None) or model_name)
            if any(c.get("finish_reason") for c in data.get("choices") or []):
                saw_finish = True
            yield f"data: {json.dumps(data)}\n\n"
        # Only close the stream ourselves if the provider never did. Appending
        # finish_reason: "stop" after the provider already said "tool_calls"
        # tells a client the model stopped talking when it actually asked for a
        # tool — which is the whole conversation an agent is having.
        if not saw_error and not saw_finish:
            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps(error_response(redact_error(e), 'upstream_error'))}\n\n"
        yield "data: [DONE]\n\n"


# ── OpenAI-compatible: models ───────────────────────────
@app.get("/v1/models")
async def list_models():
    db = await get_db()
    async with db.execute("""
        SELECT m.model_id, m.display_name, m.context_length, p.name, m.provider_id
        FROM models m JOIN providers p ON m.provider_id = p.id
        WHERE m.active = 1 AND p.enabled = 1
    """) as cur:
        rows = await cur.fetchall()
    return {
        "object": "list",
        "data": [
            {"id": f"{r[4]}:{r[0]}", "object": "model", "created": int(time.time()),
             "owned_by": r[3], "root": r[0], "context_length": r[2]}
            for r in rows
        ],
    }


# ── Quota ───────────────────────────────────────────────
@app.get("/v1/quota")
async def get_quota():
    return {"providers": await _quota.get_all_quota_status()}


# ── Provider Health (circuit breaker status) ────────────
@app.get("/v1/providers/health")
async def provider_health():
    """Circuit breaker status for all providers. Consumers can check this
    to see which providers are available before making requests."""
    breaker_status = _router.circuit_breaker.get_all_status()
    quota_status = await _quota.get_all_quota_status()
    quota_map = {q["provider_id"]: q for q in quota_status}

    providers = {}
    for pid, bstate in breaker_status.items():
        providers[pid] = {
            **bstate,
            "quota": quota_map.get(pid, {}),
            "available": bstate["state"] == "closed" and (
                quota_map.get(pid, {}).get("healthy", True)
            ),
        }
    return {"providers": providers}


# ── Model Health ────────────────────────────────────────
@app.get("/v1/models/health")
async def model_health():
    """Per-model availability and health data."""
    db = await get_db()
    async with db.execute("""
        SELECT mh.provider_id, mh.model_id, mh.last_probe_at, mh.last_probe_ok,
               mh.last_probe_latency_ms, mh.last_probe_error,
               mh.consecutive_failures, mh.avg_latency_ms,
               mh.success_count, mh.failure_count, m.display_name
        FROM model_health mh
        JOIN models m ON mh.provider_id = m.provider_id AND mh.model_id = m.model_id
        ORDER BY mh.success_count DESC
    """) as cur:
        rows = await cur.fetchall()
    models = []
    for r in rows:
        # success_count + failure_count. This read r[7] (avg_latency_ms) for
        # years, so every model reported a success rate near zero.
        total = (r[8] or 0) + (r[9] or 0)
        models.append({
            "provider_id": r[0], "model_id": r[1], "display_name": r[10],
            "last_probe_at": r[2], "last_probe_ok": bool(r[3]),
            "last_probe_latency_ms": r[4], "last_probe_error": r[5],
            "consecutive_failures": r[6], "avg_latency_ms": r[7],
            "success_count": r[8], "failure_count": r[9],
            "success_rate": round(r[8] / total, 3) if total > 0 else None,
        })
    return {"models": models}


# ── Dashboard ───────────────────────────────────────────
@app.get("/dashboard")
async def dashboard():
    db = await get_db()
    async with db.execute("SELECT COUNT(*) FROM models WHERE active = 1") as cur:
        model_count = (await cur.fetchone())[0]
    async with db.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status") as cur:
        job_counts = {r[0]: r[1] for r in await cur.fetchall()}
    quota_status = await _quota.get_all_quota_status()
    breaker_status = _router.circuit_breaker.get_all_status()

    # Top models by health
    async with db.execute("""
        SELECT mh.provider_id, mh.model_id, mh.avg_latency_ms,
               mh.success_count, mh.failure_count, mh.consecutive_failures,
               mh.last_probe_ok
        FROM model_health mh
        JOIN models m ON mh.provider_id = m.provider_id AND mh.model_id = m.model_id
        WHERE m.active = 1
        ORDER BY mh.success_count DESC LIMIT 20
    """) as cur:
        health_rows = await cur.fetchall()
    top_models = []
    for r in health_rows:
        total = (r[3] or 0) + (r[4] or 0)
        top_models.append({
            "provider_id": r[0], "model_id": r[1],
            "avg_latency_ms": round(r[2], 1) if r[2] else None,
            "success_rate": round(r[3] / total, 3) if total > 0 else None,
            "consecutive_failures": r[5],
            "last_probe_ok": bool(r[6]),
        })

    return {
        "active_models": model_count,
        "jobs": job_counts,
        "providers": quota_status,
        "circuit_breakers": breaker_status,
        "top_models_health": top_models,
    }


# ── Jobs ────────────────────────────────────────────────
@app.post("/jobs")
async def submit_job(req: JobSubmission):
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    job_id = await _queue.submit_job(
        messages=messages, priority=req.priority,
        model_override=req.model, task_type=req.task_type,
        params=req.passthrough_params(),
    )
    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    result = await _queue.get_job_status(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found")
    return result


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    result = await _queue.get_job_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found or not completed")
    return result


@app.post("/jobs/process")
async def process_jobs():
    n = await _scheduler.process_now()
    return {"processed": n}
