"""Job queue manager for batch processing."""
import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from .db import get_db
from .redact import redact

logger = logging.getLogger(__name__)


class JobQueue:
    def __init__(self, router):
        self.router = router

    async def submit_job(
        self,
        messages: list[dict],
        priority: str = "batch",
        model_override: str | None = None,
        task_type: str | None = None,
        params: dict | None = None,
    ) -> str:
        """Queue a request. `params` is the same passthrough set the synchronous
        endpoint sends — response_format, tools and the rest. Dropping them here
        reproduced the exact defect the sync path had: a job that asked for a
        schema got prose back, and nothing errored."""
        job_id = str(uuid.uuid4())
        db = await get_db()
        await db.execute(
            """INSERT INTO jobs (id, status, priority, task_type, messages,
                                 model_override, created_at, params)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id, "pending", priority,
                task_type or self.router.classify_task(messages),
                json.dumps(messages), model_override,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(params) if params else None,
            ),
        )
        await db.commit()

        if priority == "immediate":
            asyncio.create_task(self._process_job(job_id))

        return job_id

    async def get_job_status(self, job_id: str) -> dict | None:
        db = await get_db()
        async with db.execute("SELECT id, status, priority, task_type, messages, model_override, created_at, started_at, completed_at, retries FROM jobs WHERE id = ?", (job_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0], "status": row[1], "priority": row[2],
            "task_type": row[3], "created_at": row[6],
            "started_at": row[7], "completed_at": row[8],
        }

    async def get_job_result(self, job_id: str) -> dict | None:
        db = await get_db()
        async with db.execute(
            """SELECT jr.job_id, jr.provider_id, jr.model_id, jr.response,
                      jr.tokens_in, jr.tokens_out, jr.latency_ms, j.status
               FROM job_results jr JOIN jobs j ON jr.job_id = j.id
               WHERE jr.job_id = ?""",
            (job_id,),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return {
            "job_id": row[0], "provider_id": row[1], "model_id": row[2],
            "response": json.loads(row[3]) if row[3] else None,
            "tokens_in": row[4], "tokens_out": row[5],
            "latency_ms": row[6], "status": row[7],
        }

    async def process_batch_jobs(self, limit: int = 50) -> int:
        db = await get_db()
        async with db.execute(
            """SELECT id FROM jobs
               WHERE status = 'pending' AND priority = 'batch'
               ORDER BY created_at LIMIT ?""",
            (limit,),
        ) as cur:
            job_ids = [row[0] for row in await cur.fetchall()]

        processed = 0
        for job_id in job_ids:
            try:
                await self._process_job(job_id)
                processed += 1
            except Exception as e:
                logger.error("Failed to process job %s: %s", job_id, redact(e))
        return processed

    async def _process_job(self, job_id: str) -> None:
        db = await get_db()
        async with db.execute(
            "SELECT messages, model_override, retries, params FROM jobs WHERE id = ?",
            (job_id,),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return

        messages = json.loads(row[0])
        model_override = row[1]
        retries = row[2]
        # These come back out of a database row. Anything the router owns is
        # stripped here rather than exploding at the call site: route() takes
        # messages/model_override/stream by name, so a stray key is a TypeError.
        params = {
            k: v for k, v in (json.loads(row[3]) if row[3] else {}).items()
            if k not in {"messages", "model", "model_override", "stream"}
        }

        await db.execute(
            "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), job_id),
        )
        await db.commit()

        try:
            import time
            t0 = time.monotonic()
            result = await self.router.route(
                messages=messages, model_override=model_override, stream=False, **params
            )
            latency_ms = (time.monotonic() - t0) * 1000

            if "error" in result.response:
                raise RuntimeError(result.response["error"])

            router_meta = result.response.get("_router", {})
            await db.execute(
                """INSERT INTO job_results
                   (job_id, provider_id, model_id, response, tokens_in, tokens_out, latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id, router_meta.get("provider"),
                    router_meta.get("model"),
                    json.dumps(result.response), result.tokens_in,
                    result.tokens_out, latency_ms,
                ),
            )
            await db.execute(
                "UPDATE jobs SET status = 'completed', completed_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), job_id),
            )
            await db.commit()

        except Exception as e:
            logger.error("Job %s failed (attempt %d): %s", job_id, retries + 1, redact(e))
            if retries < 3:
                await db.execute(
                    "UPDATE jobs SET retries = retries + 1, status = 'pending' WHERE id = ?",
                    (job_id,),
                )
            else:
                await db.execute(
                    "UPDATE jobs SET status = 'failed', completed_at = ? WHERE id = ?",
                    (datetime.now(timezone.utc).isoformat(), job_id),
                )
            await db.commit()
