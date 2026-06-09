"""Tests for circuit breaker."""
import time
from llm_router.circuit_breaker import (
    BreakerState, ProviderBreaker, CircuitBreakerManager,
)


class TestProviderBreaker:
    def test_initial_state(self):
        b = ProviderBreaker()
        assert b.state == BreakerState.CLOSED
        assert b.should_allow() is True

    def test_expected_errors_dont_trip(self):
        b = ProviderBreaker()
        # 429 (rate limit) should not trip
        tripped = b.record_failure("rate limited", status_code=429)
        assert tripped is False
        assert b.state == BreakerState.CLOSED

    def test_transient_errors_trip(self):
        b = ProviderBreaker()
        # 5 trips in quick succession should open the breaker
        for i in range(4):
            tripped = b.record_failure("server error", status_code=503)
            assert tripped is False
        tripped = b.record_failure("server error", status_code=503)
        assert tripped is True
        assert b.state == BreakerState.OPEN

    def test_open_blocks_requests(self):
        b = ProviderBreaker()
        for _ in range(5):
            b.record_failure("503", status_code=503)
        assert b.should_allow() is False

    def test_half_open_after_cooldown(self):
        b = ProviderBreaker()
        for _ in range(5):
            b.record_failure("503", status_code=503)
        assert b.state == BreakerState.OPEN

        # Simulate cooldown elapsed
        b.last_state_change = time.monotonic() - 31
        assert b.should_allow() is True
        assert b.state == BreakerState.HALF_OPEN

    def test_success_closes_half_open(self):
        b = ProviderBreaker()
        b.state = BreakerState.HALF_OPEN
        b.record_success()
        assert b.state == BreakerState.CLOSED
        assert b.consecutive_failures == 0

    def test_half_open_allows_test_request(self):
        b = ProviderBreaker()
        b.state = BreakerState.HALF_OPEN
        b.last_state_change = time.monotonic() - 31
        # In HALF_OPEN, should_allow returns True (it's the recovery test)
        assert b.should_allow() is True
        # Failures in HALF_OPEN are recorded but don't trip the breaker
        # (trip only happens from CLOSED state). The recovery probe in
        # scheduler.py handles re-tripping via circuit_breaker.record_failure.
        b.record_failure("503", status_code=503)
        assert b.failure_count == 1

    def test_string_patterns_trip(self):
        b = ProviderBreaker()
        for _ in range(5):
            b.record_failure("connection refused")
        assert b.state == BreakerState.OPEN

    def test_to_dict(self):
        b = ProviderBreaker()
        d = b.to_dict()
        assert d["state"] == "closed"
        assert d["failure_count"] == 0
        assert d["consecutive_failures"] == 0


class TestCircuitBreakerManager:
    def test_is_available_default(self):
        mgr = CircuitBreakerManager()
        assert mgr.is_available("any_provider") is True

    def test_record_failure_trips(self):
        mgr = CircuitBreakerManager()
        for _ in range(5):
            mgr.record_failure("test", "503", 503)
        assert mgr.is_available("test") is False

    def test_get_open_providers(self):
        mgr = CircuitBreakerManager()
        for _ in range(5):
            mgr.record_failure("bad_provider", "503", 503)
        assert "bad_provider" in mgr.get_open_providers()

    def test_force_close(self):
        mgr = CircuitBreakerManager()
        for _ in range(5):
            mgr.record_failure("test", "503", 503)
        assert mgr.is_available("test") is False
        mgr.force_close("test")
        assert mgr.is_available("test") is True

    def test_get_all_status(self):
        mgr = CircuitBreakerManager()
        mgr.record_success("a")
        mgr.record_success("b")
        status = mgr.get_all_status()
        assert "a" in status
        assert "b" in status
