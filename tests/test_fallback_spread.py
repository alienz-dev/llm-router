"""One provider must not be able to own the whole fallback chain.

Candidates are scored per MODEL, and OpenRouter is 20 of 36 routable models at
the highest configured priority — so `candidates[:5]` was routinely five
OpenRouter models. Failure is tracked per PROVIDER, so the first 429 retired all
five: the request died without agnes, nvidia or opencode ever being asked.
"""
import json

import pytest

from llm_router.adapters.base import AdapterResponse
from llm_router.db import _init_schema
from llm_router.quota import QuotaManager
from llm_router.router import FALLBACK_PROVIDERS, ModelCandidate, SmartRouter


class FakeAdapter:
    supports = {"tools": True, "json_schema": True, "streaming": True}

    def __init__(self, name, error=None, status=None):
        self.provider_name = name
        self.error = error
        self.status = status
        self.calls = []

    async def chat_completion(self, messages, model_id, **kwargs):
        self.calls.append(model_id)
        if self.error:
            return AdapterResponse(response={"error": self.error}, quota=None,
                                   tokens_in=0, tokens_out=0, latency_ms=1.0,
                                   status=self.status)
        return AdapterResponse(
            response={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            quota=None, tokens_in=1, tokens_out=1, latency_ms=5.0)


SCORES = json.dumps({"code": 0.9, "reasoning": 0.9, "summarize": 0.9, "general": 0.9})


@pytest.fixture
async def db(monkeypatch):
    import aiosqlite
    import llm_router.db as db_module

    conn = await aiosqlite.connect(":memory:")
    await _init_schema(conn)
    for pid in ("openrouter", "agnes", "nvidia", "opencode"):
        await conn.execute(
            "INSERT INTO providers (id, name, base_url) VALUES (?, ?, 'https://x')",
            (pid, pid))
    await conn.commit()

    async def _get_db():
        return conn

    monkeypatch.setattr(db_module, "get_db", _get_db)
    for module in ("db", "router", "quota", "health"):
        monkeypatch.setattr(f"llm_router.{module}.get_db", _get_db, raising=False)
    yield conn
    await conn.close()


async def add_model(db, provider, model):
    await db.execute(
        """INSERT INTO models (provider_id, model_id, display_name, context_length,
                               task_scores, discovered_at, active)
           VALUES (?, ?, ?, 8192, ?, '2026-08-01', 1)""",
        (provider, model, model, SCORES))
    await db.commit()


def candidate(provider, model, score):
    return ModelCandidate(provider_id=provider, model_id=model, adapter=None,
                          capability_score=1.0, quota_headroom_pct=1.0,
                          final_score=score)


class TestSpread:
    def test_one_candidate_per_provider(self):
        chosen = SmartRouter._spread_across_providers([
            candidate("openrouter", "a", 0.9),
            candidate("openrouter", "b", 0.8),
            candidate("openrouter", "c", 0.7),
            candidate("agnes", "d", 0.6),
            candidate("nvidia", "e", 0.5),
        ])
        assert [(c.provider_id, c.model_id) for c in chosen] == [
            ("openrouter", "a"), ("agnes", "d"), ("nvidia", "e")]

    def test_it_keeps_the_best_model_from_each(self):
        chosen = SmartRouter._spread_across_providers([
            candidate("agnes", "worse", 0.9),
            candidate("agnes", "better", 0.99),
        ])
        assert [c.model_id for c in chosen] == ["worse"], "input is score-ordered"

    def test_it_stops_at_the_provider_limit(self):
        chosen = SmartRouter._spread_across_providers(
            [candidate(f"p{i}", "m", 1.0 - i / 10) for i in range(10)])
        assert len(chosen) == FALLBACK_PROVIDERS

    def test_a_single_provider_still_gets_one_attempt(self):
        chosen = SmartRouter._spread_across_providers(
            [candidate("openrouter", "a", 0.9), candidate("openrouter", "b", 0.8)])
        assert len(chosen) == 1


class TestRoutingFallsThrough:
    @pytest.mark.asyncio
    async def test_a_rate_limited_provider_does_not_end_the_request(self, db):
        """The case that mattered: OpenRouter caps out at 50/day and every auto
        request died 429 instead of spilling to a provider with budget left."""
        for i in range(6):
            await add_model(db, "openrouter", f"or-{i}")
        await add_model(db, "agnes", "agnes-2.5-flash")

        openrouter = FakeAdapter("openrouter", error="HTTP 429: rate limited",
                                 status=429)
        agnes = FakeAdapter("agnes")
        router = SmartRouter({"openrouter": openrouter, "agnes": agnes},
                             QuotaManager())

        result = await router.route([{"role": "user", "content": "hi"}],
                                    model_override="auto")

        assert "error" not in result.response
        assert agnes.calls == ["agnes-2.5-flash"]
        assert len(openrouter.calls) == 1, "one attempt per provider, not five"
        assert result.response["_router"]["provider"] == "agnes"

    @pytest.mark.asyncio
    async def test_it_still_gives_up_when_every_provider_fails(self, db):
        await add_model(db, "openrouter", "or-1")
        await add_model(db, "agnes", "ag-1")
        adapters = {p: FakeAdapter(p, error="HTTP 500: boom", status=500)
                    for p in ("openrouter", "agnes")}

        result = await SmartRouter(adapters, QuotaManager()).route(
            [{"role": "user", "content": "hi"}], model_override="auto")

        assert "All fallback attempts failed" in result.response["error"]
        assert result.status == 500
