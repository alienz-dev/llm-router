"""Tests for probe system."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from llm_router.probe import PROBE_ENDPOINTS, KNOWN_LIMITS, probe_model
from llm_router.health import ModelHealthRepository


class TestProbeEndpoints:
    def test_all_providers_have_endpoints(self):
        expected = {
            "openrouter", "google", "nvidia", "opencode",
            "deepseek", "groq", "cerebras", "mistral",
            "kilo", "cloudflare", "huggingface",
        }
        assert set(PROBE_ENDPOINTS.keys()) == expected

    def test_openai_compatible_endpoints(self):
        """Standard providers should have url, headers, body, key_env."""
        for pid in ["deepseek", "groq", "cerebras", "mistral", "kilo"]:
            ep = PROBE_ENDPOINTS[pid]
            assert "url" in ep
            assert "headers" in ep
            assert "body" in ep
            assert "key_env" in ep
            # Should produce valid headers
            headers = ep["headers"]("test-key")
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test-key"
            # Should produce valid body
            body = ep["body"]("test-model")
            assert body["model"] == "test-model"

    def test_google_has_url_builder(self):
        ep = PROBE_ENDPOINTS["google"]
        assert "url_builder" in ep
        url = ep["url_builder"]("gemini-pro", "test-key")
        assert "gemini-pro" in url
        assert "test-key" in url

    def test_cloudflare_has_url_builder(self):
        ep = PROBE_ENDPOINTS["cloudflare"]
        assert "url_builder" in ep

    def test_huggingface_has_url_builder(self):
        ep = PROBE_ENDPOINTS["huggingface"]
        assert "url_builder" in ep
        url = ep["url_builder"]("mistral-7b", "test-key")
        assert "mistral-7b" in url


class TestKnownLimits:
    def test_all_configured_providers_have_limits(self):
        """Every provider in config should have an entry (even if empty)."""
        expected = {
            "google", "openrouter", "cerebras", "groq", "mistral",
            "nvidia", "opencode", "deepseek", "cloudflare", "huggingface", "kilo",
        }
        assert set(KNOWN_LIMITS.keys()) == expected

    def test_google_limits_structure(self):
        google = KNOWN_LIMITS["google"]
        for model_id, limits in google.items():
            assert "rpm" in limits or "rpd" in limits or "tpm" in limits


class TestUpdateModelHealth:
    @pytest.mark.asyncio
    async def test_insert_new_health(self, test_db):
        """Should create a new model_health row."""
        import llm_router.db as db_module
        original = db_module._db_connection
        db_module._db_connection = test_db
        try:
            repo = ModelHealthRepository()
            await repo.update(
                "test_provider", "test_model",
                success=True, latency_ms=500.0, error=None,
                now="2025-01-01T00:00:00+00:00",
            )
            async with test_db.execute(
                "SELECT last_probe_ok, avg_latency_ms, success_count FROM model_health WHERE provider_id = ? AND model_id = ?",
                ("test_provider", "test_model"),
            ) as cur:
                row = await cur.fetchone()
            assert row is not None
            assert row[0] == 1  # last_probe_ok
            assert row[1] == pytest.approx(500.0)  # avg_latency_ms
            assert row[2] == 1  # success_count
        finally:
            db_module._db_connection = original

    @pytest.mark.asyncio
    async def test_update_existing_health(self, test_db):
        """Should update existing row with EMA latency."""
        import llm_router.db as db_module
        original = db_module._db_connection
        db_module._db_connection = test_db
        try:
            repo = ModelHealthRepository()
            now = "2025-01-01T00:00:00+00:00"
            await repo.update("p", "m", True, 1000.0, None, now)
            await repo.update("p", "m", True, 500.0, None, now)
            async with test_db.execute(
                "SELECT avg_latency_ms, success_count FROM model_health WHERE provider_id = 'p' AND model_id = 'm'"
            ) as cur:
                row = await cur.fetchone()
            # EMA: 1000 * 0.7 + 500 * 0.3 = 850
            assert row[0] == pytest.approx(850.0)
            assert row[1] == 2
        finally:
            db_module._db_connection = original

    @pytest.mark.asyncio
    async def test_failure_increments_consecutive(self, test_db):
        """Should increment consecutive_failures on failure."""
        import llm_router.db as db_module
        original = db_module._db_connection
        db_module._db_connection = test_db
        try:
            repo = ModelHealthRepository()
            now = "2025-01-01T00:00:00+00:00"
            await repo.update("p", "m", False, None, "error", now)
            await repo.update("p", "m", False, None, "error", now)
            async with test_db.execute(
                "SELECT consecutive_failures, failure_count FROM model_health WHERE provider_id = 'p' AND model_id = 'm'"
            ) as cur:
                row = await cur.fetchone()
            assert row[0] == 2  # consecutive_failures
            assert row[1] == 2  # failure_count
        finally:
            db_module._db_connection = original
