"""APScheduler: batch processing + model discovery."""
import logging
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .config import get_config
from .discovery import ModelDiscovery
from .queue import JobQueue

logger = logging.getLogger(__name__)


class AppScheduler:
    """Unified scheduler for batch processing and model discovery."""

    def __init__(self, job_queue: JobQueue):
        self.scheduler = AsyncIOScheduler()
        self.job_queue = job_queue
        self.discovery = ModelDiscovery()

    async def start(self):
        # Run discovery immediately on startup
        await self._run_discovery()

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

        self.scheduler.start()
        logger.info("Scheduler started (batch: %d-%d UTC, discovery: every %dh)", start_hour, end_hour, interval)

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
