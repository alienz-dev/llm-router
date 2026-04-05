from pydantic import BaseModel
from dataclasses import dataclass
from typing import Any


# API request/response models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


class JobSubmission(BaseModel):
    messages: list[ChatMessage]
    task_type: str | None = None
    priority: str = "batch"  # "immediate" or "batch"
    model: str | None = None
    callback_url: str | None = None


class JobStatus(BaseModel):
    id: str
    status: str
    priority: str
    task_type: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None


# Internal dataclasses
@dataclass
class QuotaSnapshot:
    rpm_limit: int | None = None
    rpm_remaining: int | None = None
    rpd_limit: int | None = None
    rpd_remaining: int | None = None
    tpm_limit: int | None = None
    tpm_remaining: int | None = None
    tpd_limit: int | None = None
    tpd_remaining: int | None = None
    reset_seconds: float | None = None


@dataclass
class AdapterResponse:
    response: dict[str, Any]
    quota: QuotaSnapshot | None
    tokens_in: int
    tokens_out: int
    latency_ms: float
    error: str | None = None
