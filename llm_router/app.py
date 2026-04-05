"""FastAPI application with all routes."""
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from .adapters import ADAPTERS
from .db import get_db, close_db
from .models import ChatCompletionRequest, JobSubmission
from .quota import QuotaManager
from .queue import JobQueue
from .router import SmartRouter
from .scheduler import AppScheduler

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
            continue
        if pid == "cloudflare":
            acct = os.getenv(pcfg.account_id_env or "", "")
            if acct:
                adapters[pid] = ADAPTERS[pid](acct, key)
        elif pid in ADAPTERS:
            adapters[pid] = ADAPTERS[pid](key)
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
    await get_db()
    await _seed_providers()

    adapters = _build_adapters()
    _quota = QuotaManager()
    _router = SmartRouter(adapters, _quota)
    _queue = JobQueue(_router)
    _scheduler = AppScheduler(_queue)
    await _scheduler.start()

    yield

    await _scheduler.stop()
    await close_db()


app = FastAPI(title="LLM Router", version="0.1.0", lifespan=lifespan)


# ── Health ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── OpenAI-compatible: chat completions ─────────────────
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, response: Response):
    messages = [m.model_dump() for m in req.messages]

    if req.stream:
        stream_iter = await _router.route(messages, model_override=req.model, stream=True)
        return StreamingResponse(
            _sse_wrap(stream_iter, req.model or "auto"),
            media_type="text/event-stream",
        )

    result = await _router.route(messages, model_override=req.model, stream=False)
    if "error" in result.response:
        raise HTTPException(status_code=502, detail=result.response["error"])

    meta = result.response.pop("_router", {})
    response.headers["x-llm-router-provider"] = meta.get("provider", "unknown")
    response.headers["x-llm-router-model"] = meta.get("model", req.model or "auto")
    return result.response


async def _sse_wrap(stream, model_name: str):
    chunk_id = f"chatcmpl-{int(time.time())}"
    try:
        async for data in stream:
            chunk = {
                "id": chunk_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model_name,
                "choices": [{"index": 0, "delta": {"content": data}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"


# ── OpenAI-compatible: models ───────────────────────────
@app.get("/v1/models")
async def list_models():
    db = await get_db()
    async with db.execute("""
        SELECT m.model_id, m.display_name, m.context_length, p.name
        FROM models m JOIN providers p ON m.provider_id = p.id
        WHERE m.active = 1 AND p.enabled = 1
    """) as cur:
        rows = await cur.fetchall()
    return {
        "object": "list",
        "data": [
            {"id": r[0], "object": "model", "created": int(time.time()), "owned_by": r[3]}
            for r in rows
        ],
    }


# ── Quota ───────────────────────────────────────────────
@app.get("/v1/quota")
async def get_quota():
    return {"providers": await _quota.get_all_quota_status()}


# ── Dashboard ───────────────────────────────────────────
@app.get("/dashboard")
async def dashboard():
    db = await get_db()
    async with db.execute("SELECT COUNT(*) FROM models WHERE active = 1") as cur:
        model_count = (await cur.fetchone())[0]
    async with db.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status") as cur:
        job_counts = {r[0]: r[1] for r in await cur.fetchall()}
    quota_status = await _quota.get_all_quota_status()
    return {
        "active_models": model_count,
        "jobs": job_counts,
        "providers": quota_status,
    }


# ── Jobs ────────────────────────────────────────────────
@app.post("/jobs")
async def submit_job(req: JobSubmission):
    messages = [m.model_dump() for m in req.messages]
    job_id = await _queue.submit_job(
        messages=messages, priority=req.priority,
        model_override=req.model, task_type=req.task_type,
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
