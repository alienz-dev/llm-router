"""Credentials must not survive into a log line, a health row, or an error body.

Google puts the API key in the query string, so every httpx error built from a
Google URL embeds it; the rest echo the Authorization header back inside error
bodies. Both used to be logged verbatim, and probe errors were additionally
written to model_health.last_probe_error, which /v1/models/health and /dashboard
serve unauthenticated. One live GOOGLE_AI_API_KEY reached llm-router.log that way.
"""
import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llm_router.redact import redact, redact_error

FAKE_GOOGLE_KEY = "AIzaSyD-fake-key-for-tests-0123456789ab"
FAKE_BEARER = "sk-or-v1-fake0123456789abcdef0123456789abcdef"


@pytest.fixture
def google_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_AI_API_KEY", FAKE_GOOGLE_KEY)
    return FAKE_GOOGLE_KEY


class TestRedact:
    def test_masks_a_key_query_param(self):
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={FAKE_GOOGLE_KEY}"
        assert FAKE_GOOGLE_KEY not in redact(url)
        assert "key=***" in redact(url)

    def test_masks_a_value_that_came_from_our_env(self, monkeypatch):
        monkeypatch.setenv("SOME_PROVIDER_API_KEY", "zzz-not-a-known-shape-zzz-1234")
        assert "zzz-not-a-known-shape" not in redact("failed with zzz-not-a-known-shape-zzz-1234")

    def test_masks_a_bearer_header(self):
        assert FAKE_BEARER not in redact(f"Authorization: Bearer {FAKE_BEARER}")

    def test_masks_a_key_shape_we_never_held(self):
        """A second provider's key inside a shared error body."""
        assert "AIzaSy" not in redact("upstream said AIzaSyOTHER0123456789abcdefghijklmno")

    def test_leaves_ordinary_text_alone(self):
        msg = "HTTP 404: model not found"
        assert redact(msg) == msg

    def test_redacts_before_truncating(self):
        """Truncating first would leave a usable prefix of the key in the row."""
        exc = Exception(f"connect to https://x/y?key={FAKE_GOOGLE_KEY} failed")
        out = redact_error(exc, 80)
        assert len(out) <= 80
        assert "AIzaSy" not in out


class TestAdapterErrorsAreRedacted:
    def _adapter(self):
        from llm_router.adapters.base import OpenAICompatibleAdapter

        adapter = OpenAICompatibleAdapter.__new__(OpenAICompatibleAdapter)
        adapter.provider_name = "test"
        adapter.base_url = "https://example.test/v1"
        adapter.api_key = FAKE_BEARER
        adapter.client = MagicMock()
        return adapter

    @pytest.mark.asyncio
    async def test_echoed_key_never_reaches_the_error_body(self, monkeypatch):
        """The body a client receives is the same string we log."""
        monkeypatch.setenv("TEST_PROVIDER_API_KEY", FAKE_BEARER)
        adapter = self._adapter()
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        response = httpx.Response(
            401, text=f'{{"error":"bad key: {FAKE_BEARER}"}}', request=request
        )
        adapter.client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError("401", request=request, response=response)
        )

        result = await adapter.chat_completion([{"role": "user", "content": "hi"}], "m")

        assert FAKE_BEARER not in result.response["error"]
        assert result.status == 401

    @pytest.mark.asyncio
    async def test_connection_error_carrying_a_url_is_redacted(self, google_key):
        adapter = self._adapter()
        adapter.client.post = AsyncMock(
            side_effect=httpx.ConnectError(f"failed to connect to /v1?key={google_key}")
        )
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await adapter.chat_completion([{"role": "user", "content": "hi"}], "m")
        assert google_key not in result.response["error"]


class TestProbeErrorsAreRedacted:
    """These land in model_health.last_probe_error, which /v1/models/health serves."""

    @pytest.mark.asyncio
    async def test_401_body_is_redacted(self, google_key):
        from llm_router import probe

        client = MagicMock()
        client.post = AsyncMock(return_value=httpx.Response(
            401,
            text=f'{{"error":{{"message":"invalid key {google_key}"}}}}',
            request=httpx.Request("POST", "https://x/v1/chat/completions"),
        ))
        with patch.dict(probe.os.environ, {"GOOGLE_AI_API_KEY": google_key}):
            result = await probe.probe_model(client, "google", "gemini-2.0-flash")

        assert google_key not in result["error"]

    @pytest.mark.asyncio
    async def test_exception_is_redacted(self, google_key):
        from llm_router import probe

        client = MagicMock()
        client.post = AsyncMock(
            side_effect=Exception(f"GET https://x/models?key={google_key} timed out")
        )
        result = await probe.probe_model(client, "google", "gemini-2.0-flash")
        assert google_key not in result["error"]


class TestApiKeyAuth:
    @pytest.mark.asyncio
    async def test_query_param_no_longer_authenticates(self):
        """A key in the URL lands in access logs, proxy logs and Referer headers."""
        from httpx import ASGITransport, AsyncClient
        from llm_router.app import APIKeyMiddleware
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/v1/quota")
        async def quota():
            return {"ok": True}

        app.add_middleware(APIKeyMiddleware, api_key="secret-key-value")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            via_query = await client.get("/v1/quota?key=secret-key-value")
            via_header = await client.get(
                "/v1/quota", headers={"authorization": "Bearer secret-key-value"}
            )

        assert via_query.status_code == 401
        assert via_header.status_code == 200
