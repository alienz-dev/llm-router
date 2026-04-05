from fastapi import FastAPI
from contextlib import asynccontextmanager
from .db import get_db, close_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_db()
    yield
    await close_db()


app = FastAPI(
    title="LLM Router",
    description="Free-Tier LLM Gateway",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
