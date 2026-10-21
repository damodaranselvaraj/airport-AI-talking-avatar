"""Secret scrubbing for anything that could leave the server (errors, logs).

Mirrors the approach in ``ask.py``: mask every configured secret value and any
key-looking token. API responses never include exception text at all; this is
the second line of defence for log lines and for diagnostics.
"""
from __future__ import annotations

import re
from typing import Iterable

_SECRET_RE = re.compile(r"\b(?:sk|key|co)-[A-Za-z0-9_\-]{8,}")
_MIN_SECRET_LEN = 8


def secret_values(settings: object) -> list[str]:
    """Collect configured secret values from a Settings-like object."""
    values: list[str] = []
    for name in ("openai_api_key", "typesafe_api_key", "cohere_api_key", "langsmith_api_key"):
        value = getattr(settings, name, None)
        value = value.get_secret_value() if hasattr(value, "get_secret_value") else value
        if isinstance(value, str) and len(value) >= _MIN_SECRET_LEN:
            values.append(value)
    return values


def scrub(text: str, secrets: Iterable[str] = ()) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return _SECRET_RE.sub("***", text)
