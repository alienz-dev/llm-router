from pydantic import BaseModel
from typing import Any, ClassVar

from .adapters.base import QuotaSnapshot, AdapterResponse  # canonical location


def error_response(message: str, error_type: str = "server_error", code: str | None = None) -> dict:
    """Standardized OpenAI-compatible error response."""
    err = {"message": message, "type": error_type}
    if code:
        err["code"] = code
    return {"error": err}


# API request/response models
class ChatMessage(BaseModel):
    """One message on the OpenAI wire.

    `content` is optional and may be a list: an assistant message that only calls
    a tool carries `content: null`, and a multimodal message carries a list of
    parts. Both used to 422 here, which is why turn 2 of any agent loop died.
    """

    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    # Assistant messages replaying a tool call, and the tool results that answer them.
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    # Forwarded verbatim to the provider. The router never rewrites or downgrades
    # it: a caller that asks for json_schema and gets a provider that cannot do it
    # must see the provider's own refusal, because that is what its fallback keys on.
    response_format: dict[str, Any] | None = None
    # Tool calling. Dropped silently until now — the model answered as if it had
    # no tools, with HTTP 200, so the caller could not tell.
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    # Sampling and streaming knobs a real OpenAI client sends.
    stream_options: dict[str, Any] | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    top_p: float | None = None
    n: int | None = None
    user: str | None = None

    # Fields forwarded to the adapter, in the order a provider sees them.
    # `messages`, `model` and `stream` are deliberately absent: they are the
    # router's to decide, and a caller must not be able to re-target the upstream.
    PASSTHROUGH_FIELDS: ClassVar[tuple[str, ...]] = (
        "max_tokens", "temperature", "response_format", "tools", "tool_choice",
        "parallel_tool_calls", "stream_options", "stop", "seed", "top_p", "n", "user",
    )

    def passthrough_params(self) -> dict[str, Any]:
        """Only what the caller actually set — an unset field must not become a
        value the provider sees."""
        return {
            name: value
            for name in self.PASSTHROUGH_FIELDS
            if (value := getattr(self, name)) is not None
        }


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
    # A queued job is the same request as a synchronous one. These were dropped
    # on the floor — queue.py called route() with no kwargs — so a batch job
    # asking for a schema got prose, exactly as the sync path used to.
    max_tokens: int | None = None
    temperature: float | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    top_p: float | None = None

    PASSTHROUGH_FIELDS: ClassVar[tuple[str, ...]] = (
        "max_tokens", "temperature", "response_format", "tools", "tool_choice",
        "parallel_tool_calls", "stop", "seed", "top_p",
    )

    def passthrough_params(self) -> dict[str, Any]:
        return {
            name: value
            for name in self.PASSTHROUGH_FIELDS
            if (value := getattr(self, name)) is not None
        }


class JobStatus(BaseModel):
    id: str
    status: str
    priority: str
    task_type: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None
