"""OpenAI wire-protocol compatibility.

These guard the two things that made the router unusable for structured output:
response_format being silently dropped, and every upstream failure — including a
clean 400 that says "I can't do that" — being reported as a 502, which makes
well-behaved clients retry a request that will never succeed.
"""
import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from llm_router.adapters.base import AdapterResponse, OpenAICompatibleAdapter


class FakeRouter:
    """Captures what the app layer forwards, and returns what it is told to."""

    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def route(self, messages, model_override=None, stream=False, **kwargs):
        self.calls.append({"messages": messages, "model": model_override,
                           "stream": stream, "kwargs": kwargs})
        if self.result is not None:
            return self.result
        return AdapterResponse(
            response={"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                      "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            quota=None, tokens_in=1, tokens_out=1, latency_ms=1.0,
        )


async def _post(body, router):
    from llm_router import app as app_module

    with patch.object(app_module, "_router", router):
        transport = ASGITransport(app=app_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/v1/chat/completions", json=body)


JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "fit", "strict": True, "schema": {
        "type": "object",
        "properties": {"score": {"type": "integer"}},
        "required": ["score"],
        "additionalProperties": False,
    }},
}


class TestResponseFormatPassthrough:
    @pytest.mark.asyncio
    async def test_json_schema_reaches_the_adapter(self):
        router = FakeRouter()
        resp = await _post({
            "model": "auto",
            "messages": [{"role": "user", "content": "score it"}],
            "response_format": JSON_SCHEMA,
        }, router)

        assert resp.status_code == 200
        assert router.calls[0]["kwargs"]["response_format"] == JSON_SCHEMA

    @pytest.mark.asyncio
    async def test_not_sent_when_absent(self):
        """An unset response_format must not become a value the provider sees."""
        router = FakeRouter()
        await _post({"model": "auto", "messages": [{"role": "user", "content": "hi"}]}, router)
        assert "response_format" not in router.calls[0]["kwargs"]

    @pytest.mark.asyncio
    async def test_router_does_not_rewrite_it(self):
        """ADR-01: json_schema is never downgraded to json_object server-side."""
        router = FakeRouter()
        await _post({
            "model": "auto",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": JSON_SCHEMA,
        }, router)
        assert router.calls[0]["kwargs"]["response_format"]["type"] == "json_schema"


def _error(status, message="This response_format type is unavailable now"):
    return AdapterResponse(
        response={"error": f"HTTP {status}: {message}"},
        quota=None, tokens_in=0, tokens_out=0, latency_ms=1.0, status=status,
    )


class TestUpstreamStatus:
    @pytest.mark.asyncio
    async def test_provider_400_is_not_laundered_into_502(self):
        """The case that matters: a client's own fallback path keys on this."""
        resp = await _post(
            {"model": "deepseek:deepseek-v4-pro",
             "messages": [{"role": "user", "content": "hi"}],
             "response_format": JSON_SCHEMA},
            FakeRouter(_error(400)),
        )
        assert resp.status_code == 400
        assert "unavailable" in resp.json()["error"]["message"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [404, 413, 422, 429])
    async def test_request_shaped_failures_pass_through(self, status):
        resp = await _post(
            {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
            FakeRouter(_error(status)),
        )
        assert resp.status_code == status

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [500, 502, 503, None])
    async def test_provider_side_failures_become_502(self, status):
        """The caller's request was fine; our upstream was not."""
        resp = await _post(
            {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
            FakeRouter(_error(status)),
        )
        assert resp.status_code == 502

    @pytest.mark.asyncio
    async def test_upstream_auth_failure_is_not_reported_as_the_callers_fault(self):
        """401 means *our* provider key is bad — surfacing it would send the
        caller off checking a key that is fine."""
        resp = await _post(
            {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
            FakeRouter(_error(401)),
        )
        assert resp.status_code == 502

    @pytest.mark.asyncio
    async def test_error_body_is_openai_shaped(self):
        """Top level, not nested under FastAPI's "detail"."""
        resp = await _post(
            {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
            FakeRouter(_error(400)),
        )
        body = resp.json()
        assert set(body) == {"error"}
        assert body["error"]["type"] == "upstream_error"
        assert isinstance(body["error"]["message"], str)


class TestAdapterCarriesStatus:
    @pytest.mark.asyncio
    async def test_status_is_captured_and_kept_off_the_wire_body(self):
        adapter = OpenAICompatibleAdapter.__new__(OpenAICompatibleAdapter)
        adapter.provider_name = "test"
        adapter.base_url = "https://example.test/v1"
        adapter.api_key = "k"
        adapter.client = MagicMock()

        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        response = httpx.Response(400, text="response_format unavailable", request=request)
        adapter.client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError("400", request=request, response=response)
        )

        result = await adapter.chat_completion([{"role": "user", "content": "hi"}], "m")

        assert result.status == 400
        assert "unavailable" in result.response["error"]
        # "status" must not ride along in the JSON body the client receives.
        assert "status" not in result.response

    @pytest.mark.asyncio
    async def test_connection_failure_has_no_status(self):
        adapter = OpenAICompatibleAdapter.__new__(OpenAICompatibleAdapter)
        adapter.provider_name = "test"
        adapter.base_url = "https://example.test/v1"
        adapter.api_key = "k"
        adapter.client = MagicMock()
        adapter.client.post = AsyncMock(side_effect=httpx.ConnectError("no route"))

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await adapter.chat_completion([{"role": "user", "content": "hi"}], "m")

        assert result.status is None
