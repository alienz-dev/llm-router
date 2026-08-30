"""Scrub credentials out of strings at the point of capture.

Provider failures carry our keys. Google puts the key in the query string, so
every httpx error message built from a Google URL embeds it; the rest echo the
`Authorization` header back inside error bodies. Those strings are logged, and
probe errors are *also* written to `model_health.last_probe_error`, which
`/v1/models/health` and `/dashboard` serve — unauthenticated unless `API_KEY`
is set. There is no second chance once an unredacted string is stored, so
redaction belongs where the string is built, not where it is printed.
"""
from __future__ import annotations

import os
import re

MASK = "***"

# Env vars whose *values* must never appear in a log line or an error body.
_SECRET_NAME_HINTS = ("API_KEY", "APIKEY", "TOKEN", "SECRET", "PASSWORD")

# Anything shorter than this is too likely to be a substring of ordinary text
# ("ok", "1234") to be worth replacing.
_MIN_SECRET_LEN = 12

# Query parameters that carry credentials, in a URL or a form body.
_QUERY_PARAM_RE = re.compile(
    r"(?i)\b(key|api[-_]?key|access[-_]?token|auth[-_]?token|token|password)"
    r"(=|%3D)([^&\s\"'<>\\]+)"
)

# `Authorization: Bearer …`, however it got stringified.
_BEARER_RE = re.compile(r"(?i)\b(bearer|basic)\s+([A-Za-z0-9\-._~+/=]{8,})")

# Known key shapes, for the case where the value did not come from our own env
# (a second key in a shared error body, a copy-pasted curl).
_KEY_SHAPE_RE = re.compile(
    r"\b("
    r"AIza[0-9A-Za-z\-_]{20,}"        # Google
    r"|sk-[A-Za-z0-9\-_]{16,}"        # OpenAI-style, DeepSeek, OpenRouter
    r"|nvapi-[A-Za-z0-9\-_]{16,}"     # NVIDIA
    r"|hf_[A-Za-z0-9]{16,}"           # Hugging Face
    r"|gsk_[A-Za-z0-9]{16,}"          # Groq
    r"|csk-[A-Za-z0-9\-_]{16,}"       # Cerebras
    r")\b"
)


def _env_secrets() -> list[str]:
    """Current secret values, longest first so a prefix never masks a superset."""
    values = set()
    for name, value in os.environ.items():
        if len(value) < _MIN_SECRET_LEN:
            continue
        if any(hint in name.upper() for hint in _SECRET_NAME_HINTS):
            values.add(value)
    return sorted(values, key=len, reverse=True)


def redact(value: object) -> str:
    """Return `value` as a string with every credential we can recognise masked."""
    text = value if isinstance(value, str) else str(value)
    if not text:
        return text

    for secret in _env_secrets():
        if secret in text:
            text = text.replace(secret, MASK)

    text = _QUERY_PARAM_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{MASK}", text)
    text = _BEARER_RE.sub(lambda m: f"{m.group(1)} {MASK}", text)
    text = _KEY_SHAPE_RE.sub(MASK, text)
    return text


def redact_error(exc: BaseException, limit: int | None = None) -> str:
    """Stringify an exception safely. Redact first, truncate second — truncating
    a secret only shortens it."""
    text = redact(exc)
    return text[:limit] if limit else text
