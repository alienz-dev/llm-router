"""A request must never reach a model that cannot do what it asks.

Scoring was capability-blind: `capability` in the formula is a *task* score, and
no model carried a tool-calling or schema flag. Two paths skipped scoring
entirely — a direct route, and the family fallback, which picks a replacement on
a bare LIKE '%family%' match. So "retry somewhere else" could land a tools
request on a model that ignores tools and answers anyway, with HTTP 200.

ADR-01: a provider-level "no" is a ceiling; probe results narrow within it. When
nothing can do the job, say so — never downgrade the request to make it fit.
"""
import json
from datetime import datetime, timezone

import pytest

from llm_router.adapters.base import AdapterResponse
from llm_router.db import _init_schema
from llm_router.quota import QuotaManager
from llm_router.router import SmartRouter


class FakeAdapter:
    """Records what it was asked to do, and refuses on demand."""

    def __init__(self, provider_name, supports=None, error=None, status=None):
        self.provider_name = provider_name
        self.supports = supports or {"tools": True, "json_schema": True, "streaming": True}
        self.error = error
        self.status = status
        self.calls = []

    async def chat_completion(self, messages, model_id, **kwargs):
        self.calls.append({"model": model_id, "kwargs": kwargs})
        if self.error:
            return AdapterResponse(response={"error": self.error}, quota=None,
                                   tokens_in=0, tokens_out=0, latency_ms=1.0,
                                   status=self.status)
        return AdapterResponse(
            response={"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                      "usage": {"prompt_tokens": 5, "completion_tokens": 5}},
            quota=None, tokens_in=5, tokens_out=5, latency_ms=10.0,
        )

    async def stream_completion(self, messages, model_id, **kwargs):
        self.calls.append({"model": model_id, "kwargs": kwargs, "stream": True})
        yield {"choices": [{"index": 0, "delta": {"content": "ok"},
                            "finish_reason": None}]}


@pytest.fixture
async def db(monkeypatch):
    import aiosqlite
    import llm_router.db as db_module

    conn = await aiosqlite.connect(":memory:")
    await _init_schema(conn)
    for pid in ("openrouter", "google", "deepseek"):
        await conn.execute(
            "INSERT INTO providers (id, name, base_url) VALUES (?, ?, 'https://x')",
            (pid, pid),
        )
    await conn.commit()

    async def _get_db():
        return conn

    for module in ("db", "router", "health", "quota", "queue"):
        monkeypatch.setattr(f"llm_router.{module}.get_db", _get_db, raising=False)
    monkeypatch.setattr(db_module, "get_db", _get_db)
    yield conn
    await conn.close()


SCORES = json.dumps({"code": 0.9, "reasoning": 0.9, "summarize": 0.9, "general": 0.9})


async def add_model(db, provider, model, *, tools=None, json_schema=None, active=1):
    await db.execute(
        """INSERT INTO models (provider_id, model_id, display_name, context_length,
                               task_scores, discovered_at, active,
                               supports_tools, supports_json_schema)
           VALUES (?, ?, ?, 8192, ?, '2026-08-01', ?, ?, ?)""",
        (provider, model, model, SCORES, active, tools, json_schema),
    )
    await db.commit()


def router_with(**adapters):
    return SmartRouter(adapters, QuotaManager())


TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
SCHEMA = {"type": "json_schema", "json_schema": {"name": "fit", "strict": True,
                                                 "schema": {"type": "object"}}}
ASK = [{"role": "user", "content": "what is the weather"}]


class TestAutoRouting:
    @pytest.mark.asyncio
    async def test_an_incapable_provider_never_receives_a_tools_request(self, db):
        await add_model(db, "google", "gemini-2.0-flash")
        await add_model(db, "openrouter", "nvidia/nemotron-3-nano-30b-a3b:free")
        google = FakeAdapter("google", supports={"tools": False, "json_schema": False,
                                                 "streaming": True})
        openrouter = FakeAdapter("openrouter")

        result = await router_with(google=google, openrouter=openrouter).route(
            ASK, model_override="auto", tools=TOOLS
        )

        assert "error" not in result.response
        assert google.calls == []
        assert openrouter.calls[0]["kwargs"]["tools"] == TOOLS

    @pytest.mark.asyncio
    async def test_no_capable_model_is_an_explicit_400(self, db):
        """Never a silent fall-through to a model that ignores tools (ADR-01)."""
        await add_model(db, "google", "gemini-2.0-flash")
        google = FakeAdapter("google", supports={"tools": False, "json_schema": False,
                                                 "streaming": True})

        result = await router_with(google=google).route(ASK, model_override="auto",
                                                        tools=TOOLS)

        assert result.status == 400
        assert "tools" in result.response["error"]
        assert google.calls == []

    @pytest.mark.asyncio
    async def test_no_candidates_at_all_is_still_a_503(self, db):
        """"Nothing can do this" and "nothing is available" need different actions."""
        result = await router_with(google=FakeAdapter("google")).route(
            ASK, model_override="auto", tools=TOOLS
        )
        assert result.status == 503

    @pytest.mark.asyncio
    async def test_a_probed_model_no_is_respected(self, db):
        await add_model(db, "openrouter", "poolside/laguna-xs-2.1:free", tools=0)
        await add_model(db, "openrouter", "nvidia/nemotron-3-nano-30b-a3b:free", tools=1)
        openrouter = FakeAdapter("openrouter")

        await router_with(openrouter=openrouter).route(ASK, model_override="auto",
                                                       tools=TOOLS)

        assert openrouter.calls[0]["model"] == "nvidia/nemotron-3-nano-30b-a3b:free"

    @pytest.mark.asyncio
    async def test_null_capability_means_unknown_not_unsupported(self, db):
        """On the day the migration lands every row is NULL. Filtering on
        `= 1` would empty the candidate set for every structured request."""
        await add_model(db, "openrouter", "some/new-model:free")  # both flags NULL
        openrouter = FakeAdapter("openrouter")

        result = await router_with(openrouter=openrouter).route(
            ASK, model_override="auto", response_format=SCHEMA
        )

        assert "error" not in result.response
        assert openrouter.calls[0]["kwargs"]["response_format"] == SCHEMA


class TestDirectRouting:
    @pytest.mark.asyncio
    async def test_direct_route_to_an_incapable_provider_is_refused(self, db):
        await add_model(db, "google", "gemini-2.0-flash")
        google = FakeAdapter("google", supports={"tools": False, "json_schema": False,
                                                 "streaming": True})

        result = await router_with(google=google).route(
            ASK, model_override="google:gemini-2.0-flash", tools=TOOLS
        )

        assert result.status == 400
        assert google.calls == []

    @pytest.mark.asyncio
    async def test_direct_route_to_a_probed_no_is_refused(self, db):
        await add_model(db, "openrouter", "poolside/laguna-xs-2.1:free", tools=0)
        openrouter = FakeAdapter("openrouter")

        result = await router_with(openrouter=openrouter).route(
            ASK, model_override="poolside/laguna-xs-2.1:free", tools=TOOLS
        )

        assert result.status == 400
        assert openrouter.calls == []


class TestFamilyFallback:
    @pytest.mark.asyncio
    async def test_a_retry_does_not_land_on_an_incapable_model(self, db):
        """The LIKE '%family%' match is exactly where this used to happen."""
        await add_model(db, "deepseek", "deepseek-v4-pro")
        await add_model(db, "google", "deepseek-ish-mirror")  # matches the family
        deepseek = FakeAdapter("deepseek")
        google = FakeAdapter("google", supports={"tools": False, "json_schema": False,
                                                 "streaming": True})
        router = router_with(deepseek=deepseek, google=google)
        # Trip deepseek's breaker so the direct route has to look elsewhere.
        for _ in range(6):
            router.circuit_breaker.record_failure("deepseek", "503 service unavailable")
        assert not router.circuit_breaker.is_available("deepseek")

        result = await router.route(ASK, model_override="deepseek:deepseek-v4-pro",
                                    tools=TOOLS)

        assert google.calls == [], "the fallback landed on a model that ignores tools"
        assert deepseek.calls == []
        assert "error" in result.response


class TestRefusalsAreNotIllHealth:
    @pytest.mark.asyncio
    async def test_five_refusals_leave_the_health_score_alone(self, db):
        """5 consecutive failures is a hard skip and 3 is a 0.2x penalty, so
        refusing a schema five times would evict a perfectly healthy model."""
        await add_model(db, "deepseek", "deepseek-v4-pro")
        deepseek = FakeAdapter(
            "deepseek", error="HTTP 400: This response_format type is unavailable now",
            status=400,
        )
        router = router_with(deepseek=deepseek)

        for _ in range(5):
            result = await router.route(ASK, model_override="deepseek:deepseek-v4-pro",
                                        response_format=SCHEMA)
            assert result.status == 400

        async with db.execute(
            "SELECT consecutive_failures, failure_count FROM model_health "
            "WHERE model_id = 'deepseek-v4-pro'"
        ) as cur:
            row = await cur.fetchone()
        assert row is None, "a refusal must not be recorded as a failure"
        assert router.circuit_breaker.is_available("deepseek")

    @pytest.mark.asyncio
    async def test_a_refusal_narrows_the_capability_flag(self, db):
        """Free evidence: the provider just told us. Writing it down is what
        stops `auto` picking the same model for the same request tomorrow."""
        await add_model(db, "deepseek", "deepseek-v4-pro")
        deepseek = FakeAdapter(
            "deepseek", error="HTTP 400: This response_format type is unavailable now",
            status=400,
        )

        await router_with(deepseek=deepseek).route(
            ASK, model_override="deepseek:deepseek-v4-pro", response_format=SCHEMA
        )

        async with db.execute(
            "SELECT supports_json_schema, supports_tools, capability_checked_at "
            "FROM models WHERE model_id = 'deepseek-v4-pro'"
        ) as cur:
            schema_ok, tools_ok, checked = await cur.fetchone()
        assert schema_ok == 0
        assert tools_ok is None, "unrelated capabilities stay unknown"
        assert checked is not None

    @pytest.mark.asyncio
    async def test_a_real_failure_still_counts(self, db):
        await add_model(db, "deepseek", "deepseek-v4-pro")
        deepseek = FakeAdapter("deepseek", error="HTTP 500: upstream exploded",
                               status=500)

        await router_with(deepseek=deepseek).route(
            ASK, model_override="deepseek:deepseek-v4-pro"
        )

        async with db.execute(
            "SELECT consecutive_failures FROM model_health WHERE model_id = 'deepseek-v4-pro'"
        ) as cur:
            row = await cur.fetchone()
        assert row[0] == 1


class TestStreamingIsMetered:
    @pytest.mark.asyncio
    async def test_a_streamed_request_records_quota_and_health(self, db):
        """Streamed traffic used to bypass quota, health and the breaker
        entirely — invisible to the free-tier budgeting this design rests on."""
        await add_model(db, "openrouter", "nvidia/nemotron-3-nano-30b-a3b:free")
        openrouter = FakeAdapter("openrouter")

        stream = await router_with(openrouter=openrouter).route(
            ASK, model_override="auto", stream=True
        )
        chunks = [chunk async for chunk in stream]

        assert chunks
        async with db.execute("SELECT COUNT(*) FROM quota_usage") as cur:
            assert (await cur.fetchone())[0] == 1
        async with db.execute(
            "SELECT success_count FROM model_health "
            "WHERE model_id = 'nvidia/nemotron-3-nano-30b-a3b:free'"
        ) as cur:
            assert (await cur.fetchone())[0] == 1


class TestBatchPath:
    @pytest.mark.asyncio
    async def test_a_queued_job_keeps_its_response_format(self, db):
        """queue.py called route() with no kwargs — the same defect the sync
        path had, at a second endpoint."""
        from llm_router.queue import JobQueue

        await add_model(db, "openrouter", "nvidia/nemotron-3-nano-30b-a3b:free")
        openrouter = FakeAdapter("openrouter")
        router = router_with(openrouter=openrouter)
        queue = JobQueue(router)

        job_id = await queue.submit_job(
            ASK, priority="batch", params={"response_format": SCHEMA, "temperature": 0.1}
        )
        await queue.process_batch_jobs()

        assert openrouter.calls[0]["kwargs"]["response_format"] == SCHEMA
        assert openrouter.calls[0]["kwargs"]["temperature"] == 0.1
        result = await queue.get_job_result(job_id)
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_router_owned_keys_in_a_stored_job_cannot_break_the_call(self, db):
        from llm_router.queue import JobQueue

        await add_model(db, "openrouter", "m1")
        openrouter = FakeAdapter("openrouter")
        queue = JobQueue(router_with(openrouter=openrouter))

        await queue.submit_job(ASK, priority="batch",
                               params={"messages": [], "model": "other", "seed": 1})
        await queue.process_batch_jobs()

        assert openrouter.calls[0]["model"] == "m1"
        assert "messages" not in openrouter.calls[0]["kwargs"]
