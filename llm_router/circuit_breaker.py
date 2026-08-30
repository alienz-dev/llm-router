"""Provider-level circuit breaker for handling transient provider failures.

State machine per provider:
  CLOSED   → normal operation, requests flow through
  OPEN     → provider failing, skip entirely, try next candidate
  HALF-OPEN → recovery window, allow one test request

Error classification:
  Transient (DO trip breaker): 502, 503, 504, timeout, connection error
  Expected   (DON'T trip):     429 (rate limit), 400 (bad request), 401/403 (auth)
  Permanent  (DON'T trip):     404 (handled by probe/deactivation)
"""
import time
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class BreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# Error codes that indicate provider is down (should trip breaker)
TRIP_CODES = frozenset({502, 503, 504})

# Error codes that are expected/transient (don't trip breaker)
EXPECTED_CODES = frozenset({400, 401, 403, 404, 429})

# String patterns in error messages that indicate provider failure
TRIP_PATTERNS = frozenset({
    "502", "503", "504",
    "timeout", "timed out",
    "connection reset", "connection refused",
    "server error", "service unavailable",
    "bad gateway", "gateway timeout",
    # httpx transport exception names. These arrive as "ConnectError: ..." and
    # are the clearest possible evidence a provider is down — but several carry
    # no message at all, so matching on prose alone missed them entirely.
    "connecterror", "connecttimeout", "readtimeout", "writetimeout",
    "pooltimeout", "readerror", "writeerror", "remoteprotocolerror",
    "proxyerror", "networkerror",
})


@dataclass
class ProviderBreaker:
    """Circuit breaker state for a single provider."""
    state: BreakerState = BreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_state_change: float = 0.0
    last_error: str = ""
    consecutive_failures: int = 0
    # Sliding window: timestamps of recent failures
    failure_window: list[float] = field(default_factory=list)

    def record_failure(self, error: str, status_code: int | None = None):
        """Record a failure. Returns True if breaker tripped."""
        now = time.monotonic()

        # Check if this is a trip-worthy error
        should_trip = False
        if status_code and status_code in TRIP_CODES:
            should_trip = True
        elif any(p in error.lower() for p in TRIP_PATTERNS):
            should_trip = True

        if not should_trip:
            # Expected error (429, 400, etc.) — don't count against breaker
            return False

        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure_time = now
        self.last_error = error[:200]

        # Add to sliding window (keep last 60 seconds)
        self.failure_window.append(now)
        self.failure_window = [t for t in self.failure_window if now - t < 60]

        # Trip if threshold exceeded
        if len(self.failure_window) >= 5 and self.state == BreakerState.CLOSED:
            self._transition(BreakerState.OPEN, "5 failures in 60s")
            return True

        return False

    def record_success(self):
        """Record a success. Closes breaker if half-open."""
        self.success_count += 1
        self.consecutive_failures = 0
        if self.state == BreakerState.HALF_OPEN:
            self._transition(BreakerState.CLOSED, "recovery confirmed")
            logger.info("Circuit breaker CLOSED for provider (recovery confirmed)")

    def should_allow(self) -> bool:
        """Should a request be allowed through this provider?"""
        now = time.monotonic()

        if self.state == BreakerState.CLOSED:
            return True

        if self.state == BreakerState.OPEN:
            # After cooldown, move to half-open
            if now - self.last_state_change > 30:  # 30s cooldown
                self._transition(BreakerState.HALF_OPEN, "cooldown elapsed")
                return True  # Allow one test request
            return False

        # HALF_OPEN: allow request (it's the recovery test)
        return True

    def _transition(self, new_state: BreakerState, reason: str):
        old = self.state
        self.state = new_state
        self.last_state_change = time.monotonic()
        if new_state == BreakerState.CLOSED:
            self.failure_window.clear()
            self.consecutive_failures = 0
        logger.info("Circuit breaker %s → %s (%s)", old.value, new_state.value, reason)

    def to_dict(self) -> dict:
        now = time.monotonic()
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "recent_failures_60s": len(self.failure_window),
            "seconds_since_last_change": round(now - self.last_state_change, 1),
            "open_for_seconds": (
                round(max(0, 30 - (now - self.last_state_change)), 1)
                if self.state == BreakerState.OPEN else None
            ),
        }


class CircuitBreakerManager:
    """Manages circuit breakers for all providers."""

    def __init__(self):
        self._breakers: dict[str, ProviderBreaker] = defaultdict(ProviderBreaker)
        self._db_loaded = False

    def get_breaker(self, provider_id: str) -> ProviderBreaker:
        return self._breakers[provider_id]

    def is_available(self, provider_id: str) -> bool:
        """Check if provider is available (breaker allows requests)."""
        return self._breakers[provider_id].should_allow()

    def record_failure(self, provider_id: str, error: str, status_code: int | None = None) -> bool:
        """Record a provider failure. Returns True if breaker tripped."""
        b = self._breakers[provider_id]
        old_state = b.state
        tripped = b.record_failure(error, status_code)
        if b.state != old_state:
            # State transitioned — persist asynchronously
            try:
                import asyncio
                asyncio.get_running_loop().create_task(self.save_state(provider_id))
            except RuntimeError:
                pass  # no event loop running (e.g. tests)
        return tripped

    def record_success(self, provider_id: str):
        """Record a provider success."""
        b = self._breakers[provider_id]
        old_state = b.state
        b.record_success()
        if b.state != old_state:
            try:
                import asyncio
                asyncio.get_running_loop().create_task(self.save_state(provider_id))
            except RuntimeError:
                pass

    def get_all_status(self) -> dict[str, dict]:
        """Get status of all breakers for the health endpoint."""
        return {pid: b.to_dict() for pid, b in self._breakers.items()}

    def get_open_providers(self) -> set[str]:
        """Get set of providers currently in OPEN state."""
        return {
            pid for pid, b in self._breakers.items()
            if b.state == BreakerState.OPEN
        }

    def get_half_open_providers(self) -> set[str]:
        """Get set of providers currently in HALF_OPEN state (need recovery test)."""
        return {
            pid for pid, b in self._breakers.items()
            if b.state == BreakerState.HALF_OPEN
        }

    def force_close(self, provider_id: str):
        """Force-close a breaker (manual override / recovery probe success)."""
        b = self._breakers[provider_id]
        old_state = b.state
        if b.state != BreakerState.CLOSED:
            b._transition(BreakerState.CLOSED, "manual override")
            try:
                import asyncio
                asyncio.get_running_loop().create_task(self.save_state(provider_id))
            except RuntimeError:
                pass

    async def load_state(self):
        """Load persisted circuit breaker state from DB on startup."""
        from .db import get_db
        db = await get_db()
        async with db.execute(
            "SELECT provider_id, state, failure_count, last_error FROM circuit_breaker_state"
        ) as cur:
            rows = await cur.fetchall()
        for provider_id, state_str, failure_count, last_error in rows:
            try:
                state = BreakerState(state_str)
            except ValueError:
                continue
            b = self._breakers[provider_id]
            b.state = state
            b.failure_count = failure_count or 0
            b.last_error = last_error or ""
            if state == BreakerState.OPEN:
                # Set last_state_change so cooldown logic works
                b.last_state_change = time.monotonic()
            logger.info("Restored circuit breaker for %s: %s (%d failures)",
                        provider_id, state_str, failure_count or 0)
        self._db_loaded = True

    async def save_state(self, provider_id: str):
        """Persist circuit breaker state to DB on every transition."""
        from .db import get_db
        b = self._breakers[provider_id]
        db = await get_db()
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            """INSERT INTO circuit_breaker_state
               (provider_id, state, failure_count, last_failure_at, last_error, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(provider_id) DO UPDATE SET
               state = excluded.state,
               failure_count = excluded.failure_count,
               last_failure_at = excluded.last_failure_at,
               last_error = excluded.last_error,
               updated_at = excluded.updated_at""",
            (provider_id, b.state.value, b.failure_count,
             now if b.last_failure_time else None,
             b.last_error, now),
        )
        await db.commit()
