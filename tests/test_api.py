"""Tests for API endpoints."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self):
        from llm_router.app import app
        from httpx import AsyncClient, ASGITransport

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


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
