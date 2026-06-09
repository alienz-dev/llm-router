"""APScheduler: batch processing + model discovery + probing."""
import logging
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .config import get_config
from .discovery import ModelDiscovery
from .probe import run_probe_and_update
from .queue import JobQueue

logger = logging.getLogger(__name__)


class AppScheduler:
    """Unified scheduler for batch processing and model discovery."""

    def __init__(self, job_queue: JobQueue):
        self.scheduler = AsyncIOScheduler()
        self.job_queue = job_queue
        self.discovery = ModelDiscovery()
        self._router = None  # Set during start()

    async def start(self):
        # Access router from the job queue
        self._router = self.job_queue.router

        # Run discovery + probe immediately on startup
        await self._run_discovery()
        await self._run_probe()

        cfg = get_config()
        start_hour = int(cfg.scheduler.batch_window_start.split(":")[0])
        end_hour = int(cfg.scheduler.batch_window_end.split(":")[0])
        interval = cfg.scheduler.discovery_interval_hours

        # Batch processing: every 30 min during configured window
        self.scheduler.add_job(
            self._process_batch,
            CronTrigger(minute="0,30", hour=f"{start_hour}-{end_hour}", timezone=ZoneInfo("UTC")),
            id="batch_processor",
            max_instances=1,
        )

        # Model discovery: every configured interval
        self.scheduler.add_job(
            self._run_discovery,
            IntervalTrigger(hours=interval),
            id="model_discovery",
            max_instances=1,
        )

        # Model probing: every 2 hours (check if models still free/available)
        self.scheduler.add_job(
            self._run_probe,
            IntervalTrigger(hours=2),
            id="model_probe",
            max_instances=1,
        )

        # Circuit breaker recovery probe: every 5 minutes
        self.scheduler.add_job(
            self._run_recovery_probe,
            IntervalTrigger(minutes=5),
            id="recovery_probe",
            max_instances=1,
        )

        # Quota usage cleanup: daily, delete rows older than 48 hours
        self.scheduler.add_job(
            self._cleanup_quota_usage,
            IntervalTrigger(hours=24),
            id="quota_cleanup",
            max_instances=1,
        )

        self.scheduler.start()
        logger.info("Scheduler started (batch: %d-%d UTC, discovery: every %dh, probe: every 2h, recovery: every 5min, cleanup: daily)", start_hour, end_hour, interval)

    async def stop(self):
        self.scheduler.shutdown(wait=False)

    async def process_now(self) -> int:
        return await self.job_queue.process_batch_jobs(limit=100)

    async def _process_batch(self):
        try:
            n = await self.job_queue.process_batch_jobs(limit=100)
            if n:
                logger.info("Processed %d batch jobs", n)
        except Exception as e:
            logger.error("Batch processing failed: %s", e)

    async def _run_discovery(self):
        try:
            logger.info("Running model discovery...")
            discovered = await self.discovery.discover_all()
            stats = await self.discovery.update_models_table(discovered)
            total = sum(s["active"] for s in stats.values())
            added = sum(s["added"] for s in stats.values())
            removed = sum(s["removed"] for s in stats.values())
            logger.info("Discovery: %d active (+%d, -%d)", total, added, removed)
        except Exception as e:
            logger.error("Discovery failed: %s", e)

    async def _run_probe(self):
        try:
            logger.info("Running model probe...")
            stats = await run_probe_and_update()
            logger.info(
                "Probe: %d probed, %d available, %d unavailable, %d deactivated",
                stats["probed"], stats["available"], stats["unavailable"],
                len(stats.get("deactivated", [])),
            )
            for d in stats.get("deactivated", []):
                logger.warning("  Deactivated: %s", d)
        except Exception as e:
            logger.error("Probe failed: %s", e)

    async def _run_recovery_probe(self):
        """Test providers in HALF_OPEN state to see if they've recovered.
        Runs every 5 minutes — much faster than the 2h model probe."""
        if not self._router:
            return
        try:
            half_open = self._router.circuit_breaker.get_half_open_providers()
            if not half_open:
                return

            logger.info("Recovery probe: testing %d half-open providers: %s",
                len(half_open), ", ".join(half_open))

            import time as _time
            for provider_id in half_open:
                # Find any active model for this provider
                from .db import get_db
                db = await get_db()
                async with db.execute(
                    "SELECT model_id FROM models WHERE provider_id = ? AND active = 1 LIMIT 1",
                    (provider_id,),
                ) as cur:
                    row = await cur.fetchone()
                if not row:
                    continue

                model_id = row[0]
                adapter = self._router.adapters.get(provider_id)
                if not adapter:
                    continue

                try:
                    t0 = _time.monotonic()
                    response = await adapter.chat_completion(
                        [{"role": "user", "content": "ok"}], model_id, max_tokens=3
                    )
                    latency = (_time.monotonic() - t0) * 1000

                    if "error" not in response.response:
                        self._router.circuit_breaker.record_success(provider_id)
                        logger.info("Recovery probe OK: %s (%.0fms) — breaker CLOSED",
                            provider_id, latency)
                    else:
                        error = response.response["error"]
                        self._router.circuit_breaker.record_failure(provider_id, error)
                        logger.info("Recovery probe FAIL: %s — %s (breaker stays OPEN)",
                            provider_id, error[:80])
                except Exception as e:
                    self._router.circuit_breaker.record_failure(provider_id, str(e))
                    logger.info("Recovery probe ERROR: %s — %s", provider_id, str(e)[:80])

        except Exception as e:
            logger.error("Recovery probe failed: %s", e)

    async def _cleanup_quota_usage(self):
        """Delete quota_usage rows older than 48 hours to prevent unbounded growth."""
        try:
            from .db import get_db
            from datetime import datetime, timezone, timedelta
            db = await get_db()
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
            async with db.execute(
                "DELETE FROM quota_usage WHERE timestamp < ?", (cutoff,)
            ) as cur:
                deleted = cur.rowcount
            await db.commit()
            if deleted:
                logger.info("Cleaned up %d old quota_usage rows", deleted)
        except Exception as e:
            logger.error("Quota cleanup failed: %s", e)
