from pydantic import BaseModel
from typing import Any

from .adapters.base import QuotaSnapshot, AdapterResponse  # canonical location


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
