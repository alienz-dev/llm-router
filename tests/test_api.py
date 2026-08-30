"""Tests for API endpoints."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestHealthEndpoint:
    """/health is what the heartbeat trusts, so it has to assert something.

    Unconditional {"status": "ok"} meant a process with a dead database and no
    adapters still reported healthy — and a wedged server with a live scheduler
    would never have paged.
    """

    async def _get(self):
        from llm_router.app import app
        from httpx import AsyncClient, ASGITransport

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/health")

    @pytest.mark.asyncio
    async def test_unhealthy_without_adapters(self):
        """No lifespan here, so nothing is wired — that must not read as ok."""
        resp = await self._get()
        assert resp.status_code == 503
        assert resp.json()["status"] == "unhealthy"
        assert resp.json()["checks"]["adapters"]["ok"] is False

    @pytest.mark.asyncio
    async def test_healthy_when_db_and_adapters_are_live(self):
        from llm_router import app as app_module

        router = MagicMock()
        router.adapters = {"openrouter": object()}
        with patch.object(app_module, "_router", router):
            resp = await self._get()
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["checks"]["database"]["ok"] is True
        assert body["checks"]["adapters"]["count"] == 1

    @pytest.mark.asyncio
    async def test_dead_database_is_reported(self):
        from llm_router import app as app_module

        router = MagicMock()
        router.adapters = {"openrouter": object()}
        with patch.object(app_module, "_router", router), \
             patch.object(app_module, "get_db", AsyncMock(side_effect=OSError("disk gone"))):
            resp = await self._get()
        assert resp.status_code == 503
        assert resp.json()["checks"]["database"]["ok"] is False


class TestModelsEndpoint:
    @pytest.mark.asyncio
    async def test_models_returns_list(self):
        """Models endpoint should return a list structure."""
        from llm_router.app import app
        from httpx import AsyncClient, ASGITransport

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "object" in data
        assert data["object"] == "list"
        assert "data" in data
