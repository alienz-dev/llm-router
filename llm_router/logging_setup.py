"""Turn the logging on.

Nothing in this package ever called `basicConfig`/`dictConfig`, so every
`logger.info` written here — which breaker tripped, which model was quota
blocked, what discovery changed, why a capability flag moved — was discarded,
and warnings printed through `logging.lastResort` with no timestamp and no
logger name. On a machine you only reach by ssh, that is the difference between
diagnosing something and guessing.
"""
from __future__ import annotations

import logging
import os

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_configured = False


def configure_logging(level: str | int | None = None) -> None:
    """Idempotent. `LOG_LEVEL` overrides; INFO by default.

    Only the root logger is touched — uvicorn configures its own and does not
    propagate, so its access log is unaffected and nothing is duplicated.
    """
    global _configured
    if _configured:
        return
    _configured = True

    level = level or os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(level=level, format=DEFAULT_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S%z")
    # httpx logs every request at INFO, which on a 50-requests/day budget is
    # noise that hides the lines that matter.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
