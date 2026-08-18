from pydantic import BaseModel
from typing import Any

from .adapters.base import QuotaSnapshot, AdapterResponse  # canonical location


def error_response(message: str, error_type: str = "server_error", code: str | None = None) -> dict:
    """Standardized OpenAI-compatible error response."""
    err = {"message": message, "type": error_type}
    if code:
        err["code"] = code
    return {"error": err}


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


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str | None = None
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "url"  # "url" or "b64_json"


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[dict[str, Any]]


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
