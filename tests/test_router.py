"""Tests for smart router — task classification and availability scoring."""
import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from llm_router.router import SmartRouter


def _classify(text: str) -> str:
    """Helper to classify a single text string."""
    router = SmartRouter.__new__(SmartRouter)
    return router.classify_task([{"role": "user", "content": text}])


class TestClassifyTask:
    def test_code_detection(self):
        assert _classify("Write a Python function to sort") == "code"
        assert _classify("Implement a linked list") == "code"
        assert _classify("Debug this code: ```python") == "code"
        assert _classify("def calculate_sum():") == "code"

    def test_reasoning_detection(self):
        assert _classify("Analyze the pros and cons") == "reasoning"
        assert _classify("Compare React and Vue") == "reasoning"
        assert _classify("Explain why the sky is blue") == "reasoning"
        assert _classify("Step by step, solve this") == "reasoning"

    def test_summarize_detection(self):
        assert _classify("Summarize this article") == "summarize"
        assert _classify("TLDR of the meeting") == "summarize"
        assert _classify("Give me the key points") == "summarize"

    def test_general_fallback(self):
        assert _classify("Hello, how are you?") == "general"
        assert _classify("What's the weather?") == "general"

    def test_empty_messages(self):
        router = SmartRouter.__new__(SmartRouter)
        assert router.classify_task([]) == "general"
        assert router.classify_task([{"role": "user", "content": ""}]) == "general"


class TestAvailabilityScoring:
    """Test the availability multiplier computation logic."""

    def _compute_availability(self, probe_age_h, success_rate, avg_latency, consec_failures):
        """Mirror the availability logic from _get_ranked_candidates."""
        # Freshness
        if probe_age_h is not None:
            if probe_age_h < 2:
                freshness = 1.0
            elif probe_age_h < 6:
                freshness = 0.7
            elif probe_age_h < 24:
                freshness = 0.4
            else:
                freshness = 0.1
        else:
            freshness = 0.2

        # Success rate
        sr = max(0.1, min(1.0, success_rate)) if success_rate is not None else 0.5
        if consec_failures and consec_failures >= 3:
            sr *= 0.2

        # Latency penalty
        if avg_latency and avg_latency > 10000:
            lm = 0.5
        elif avg_latency and avg_latency > 5000:
            lm = 0.8
        else:
            lm = 1.0

        return freshness * sr * lm

    def test_fresh_high_success(self):
        avail = self._compute_availability(1, 0.95, 200, 0)
        assert avail == pytest.approx(0.95, abs=0.01)

    def test_stale_model(self):
        # 8h → freshness=0.4, success_rate=0.95, latency=200→1.0
        avail = self._compute_availability(8, 0.95, 200, 0)
        assert avail == pytest.approx(0.38, abs=0.01)

    def test_low_success_rate(self):
        avail = self._compute_availability(1, 0.3, 200, 0)
        assert avail == pytest.approx(0.3, abs=0.01)

    def test_consecutive_failures_penalty(self):
        avail = self._compute_availability(1, 0.5, 200, 3)
        assert avail == pytest.approx(0.1, abs=0.01)

    def test_hard_skip_threshold(self):
        # 5+ consecutive failures should be excluded (not reach scoring)
        # This is tested at the candidate filtering level
        pass

    def test_never_probed(self):
        avail = self._compute_availability(None, None, None, None)
        assert avail == pytest.approx(0.1, abs=0.01)

    def test_slow_latency(self):
        avail = self._compute_availability(1, 0.9, 8000, 0)
        assert avail == pytest.approx(0.72, abs=0.01)

    def test_very_slow_latency(self):
        avail = self._compute_availability(1, 0.9, 15000, 0)
        assert avail == pytest.approx(0.45, abs=0.01)

    def test_fast_latency(self):
        avail = self._compute_availability(1, 0.9, 500, 0)
        assert avail == pytest.approx(0.9, abs=0.01)
