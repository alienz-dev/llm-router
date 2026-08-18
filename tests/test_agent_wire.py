"""The wire an agent actually speaks: tool calls, replayed turns, live streams.

Three defects met here. `tools` was dropped silently — HTTP 200, and the model
answered as if it had none. Turn 2 of any agent loop 422'd, because an assistant
message with `content: null` + `tool_calls` and a `role: "tool"` message had
nowhere to go in the schema. And streaming carried `delta.content` only, so a
streamed tool call would have arrived as an empty completion.

Everything here is hermetic: a 50-requests-a-day free tier is not a test budget.
"""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from llm_router.adapters.base import AdapterResponse

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name):
    text = (FIXTURES / name).read_text()
    return json.loads(text) if name.endswith(".json") else text


class RecordingRouter:
    """Stands in for SmartRouter at the app boundary."""

    def __init__(self, result=None, stream=None):
        self.result = result
        self.stream = stream
        self.calls = []

    async def route(self, messages, model_override=None, stream=False, **kwargs):
        self.calls.append({"messages": messages, "model": model_override,
                           "stream": stream, "kwargs": kwargs})
        if stream:
            return self.stream
        return self.result or AdapterResponse(
            response={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            quota=None, tokens_in=1, tokens_out=1, latency_ms=1.0,
        )


async def post(body, router):
    from llm_router import app as app_module

    with patch.object(app_module, "_router", router):
        transport = ASGITransport(app=app_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/v1/chat/completions", json=body)


# ── T7: message shape and tool parameters ───────────────────────────────
class TestAgentMessageShapes:
    @pytest.mark.asyncio
    async def test_langgraph_turn_two_is_accepted(self):
        """The replay that used to 422 and kill the loop on its second turn."""
        router = RecordingRouter(AdapterResponse(
            response=fixture("tool_call_response.json"),
            quota=None, tokens_in=132, tokens_out=21, latency_ms=800.0,
        ))
        resp = await post(fixture("langgraph_turn2.json"), router)

        assert resp.status_code == 200
        sent = router.calls[0]["messages"]
        assert sent[2]["tool_calls"][0]["id"] == "call_hM3kPq"
        assert "content" not in sent[2], "content: null must not be forwarded as a value"
        assert sent[3]["tool_call_id"] == "call_hM3kPq"

    @pytest.mark.asyncio
    async def test_tool_calls_survive_the_response(self):
        router = RecordingRouter(AdapterResponse(
            response=fixture("tool_call_response.json"),
            quota=None, tokens_in=132, tokens_out=21, latency_ms=800.0,
        ))
        body = (await post(fixture("langgraph_turn2.json"), router)).json()
        call = body["choices"][0]["message"]["tool_calls"][0]

        assert call["id"]
        assert json.loads(call["function"]["arguments"]) == {"city": "Melbourne"}
        assert body["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_tools_reach_the_adapter(self):
        router = RecordingRouter()
        await post(fixture("langgraph_turn2.json"), router)
        kwargs = router.calls[0]["kwargs"]
        assert kwargs["tools"][0]["function"]["name"] == "get_weather"
        assert kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_multimodal_content_array_is_accepted(self):
        router = RecordingRouter()
        resp = await post({
            "model": "auto",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
            ]}],
        }, router)
        assert resp.status_code == 200
        assert isinstance(router.calls[0]["messages"][0]["content"], list)

    @pytest.mark.asyncio
    async def test_unset_parameters_are_not_invented(self):
        router = RecordingRouter()
        await post({"model": "auto", "messages": [{"role": "user", "content": "hi"}]}, router)
        assert router.calls[0]["kwargs"] == {}


class TestReservedParameters:
    """A caller must not be able to re-target the upstream. `model` in the body
    selects the *route*; it must never end up in the provider payload, where it
    would reach a model nothing scored, quota-checked or recorded."""

    def _adapter(self):
        from llm_router.adapters.base import OpenAICompatibleAdapter

        adapter = OpenAICompatibleAdapter.__new__(OpenAICompatibleAdapter)
        adapter.provider_name = "openrouter"
        adapter.base_url = "https://example.test/v1"
        adapter.api_key = "k"
        adapter.client = MagicMock()
        return adapter

    @pytest.mark.asyncio
    async def test_a_smuggled_model_key_does_not_reach_the_payload(self):
        import httpx

        adapter = self._adapter()
        captured = {}

        async def _post(url, headers=None, json=None):
            captured.update(json)
            return httpx.Response(
                200, json={"choices": [], "usage": {}},
                request=httpx.Request("POST", url),
            )

        adapter.client.post = _post
        await adapter.chat_completion(
            [{"role": "user", "content": "hi"}], "the-routed-model",
            model="something-else", stream=True, temperature=0.5,
        )

        assert captured["model"] == "the-routed-model"
        assert captured["messages"] == [{"role": "user", "content": "hi"}]
        assert "stream" not in captured
        assert captured["temperature"] == 0.5

    def test_reserved_keys_are_dropped_before_the_payload_is_built(self):
        adapter = self._adapter()
        clean = adapter._clean_params({
            "messages": [{"role": "user", "content": "nope"}],
            "model": "something-else",
            "stream": True,
            "temperature": 0.5,
        })
        assert clean == {"temperature": 0.5}

    def test_the_kill_switch_narrows_what_is_forwarded(self):
        """routing.passthrough_params is the one-line revert for this whole wave."""
        from llm_router.config import Config, ProviderConfig, RoutingConfig

        adapter = self._adapter()
        narrowed = Config(
            routing=RoutingConfig(passthrough_params=["max_tokens", "temperature",
                                                      "response_format"]),
            providers={"openrouter": ProviderConfig(
                name="OpenRouter", base_url="https://x", api_key_env="OPENROUTER_API_KEY")},
        )
        with patch("llm_router.config.get_config", return_value=narrowed):
            clean = adapter._clean_params({"tools": [{"type": "function"}],
                                           "temperature": 0.2})
        assert clean == {"temperature": 0.2}

    def test_a_provider_can_be_narrowed_on_its_own(self):
        from llm_router.config import Config, ProviderConfig, RoutingConfig

        adapter = self._adapter()
        cfg = Config(
            routing=RoutingConfig(),
            providers={"openrouter": ProviderConfig(
                name="OpenRouter", base_url="https://x", api_key_env="K",
                passthrough_params=["max_tokens"])},
        )
        with patch("llm_router.config.get_config", return_value=cfg):
            clean = adapter._clean_params({"seed": 7, "max_tokens": 10})
        assert clean == {"max_tokens": 10}


# ── T8: streaming ───────────────────────────────────────────────────────
async def _chunks_from(sse_text):
    for block in sse_text.strip().split("\n\n"):
        payload = block.replace("data: ", "", 1).strip()
        if payload and payload != "[DONE]":
            yield json.loads(payload)


def _parse_sse(body):
    out = []
    for line in body.splitlines():
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            out.append(json.loads(line[6:]))
    return out


class TestStreamingCarriesTheWholeDelta:
    @pytest.mark.asyncio
    async def test_a_streamed_tool_call_reassembles(self):
        """delta.content-only extraction turned this into an empty completion."""
        router = RecordingRouter(stream=_chunks_from(fixture("stream_tool_call.txt")))
        from llm_router import app as app_module

        with patch.object(app_module, "_router", router):
            transport = ASGITransport(app=app_module.app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/chat/completions", json={
                    "model": "auto", "stream": True,
                    "messages": [{"role": "user", "content": "weather?"}],
                    "tools": fixture("langgraph_turn2.json")["tools"],
                })
                body = resp.text

        assert resp.status_code == 200
        chunks = _parse_sse(body)
        arguments = "".join(
            call.get("function", {}).get("arguments", "")
            for chunk in chunks
            for choice in chunk.get("choices", [])
            for call in (choice.get("delta") or {}).get("tool_calls") or []
        )
        assert json.loads(arguments) == {"city": "Melbourne"}
        assert body.rstrip().endswith("data: [DONE]")

    @pytest.mark.asyncio
    async def test_usage_arrives_before_done(self):
        router = RecordingRouter(stream=_chunks_from(fixture("stream_tool_call.txt")))
        from llm_router import app as app_module

        with patch.object(app_module, "_router", router):
            transport = ASGITransport(app=app_module.app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/chat/completions", json={
                    "model": "auto", "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream_options": {"include_usage": True},
                })
                chunks = _parse_sse(resp.text)

        with_usage = [c for c in chunks if c.get("usage")]
        assert with_usage, "no usage chunk reached the client"
        assert with_usage[-1]["usage"]["total_tokens"] == 153

    @pytest.mark.asyncio
    async def test_stream_options_reach_the_provider(self):
        router = RecordingRouter(stream=_chunks_from("data: {}\n\n"))
        await post({"model": "auto", "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream_options": {"include_usage": True}}, router)
        assert router.calls[0]["kwargs"]["stream_options"] == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_a_stream_failure_is_not_served_as_an_answer(self):
        """base.py used to yield the exception string as assistant content, so a
        transport failure read as a model reply."""
        async def failing():
            yield {"error": "HTTP 429: rate limited", "status": 429}

        router = RecordingRouter(stream=failing())
        resp = await post({"model": "auto", "stream": True,
                           "messages": [{"role": "user", "content": "hi"}]}, router)

        assert resp.status_code == 429
        assert "error" in resp.json()

    @pytest.mark.asyncio
    async def test_router_refusal_before_the_stream_is_an_http_error(self):
        router = RecordingRouter()
        router.stream = AdapterResponse(
            response={"error": "No available models meet quota requirements"},
            quota=None, tokens_in=0, tokens_out=0, latency_ms=0, status=503,
        )
        resp = await post({"model": "auto", "stream": True,
                           "messages": [{"role": "user", "content": "hi"}]}, router)
        assert resp.status_code == 502  # 503 is provider-side, laundered to 502
        assert "quota" in resp.json()["error"]["message"]
