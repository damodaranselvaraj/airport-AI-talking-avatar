"""JSON-lines protocol between the FastAPI app and the Ditto worker process.

Standard library only, so the same file runs in the app's ``.venv`` (Python
3.12) and in the isolated ``ditto`` conda env (Python 3.10). The worker loads
it by file path; keep it free of imports from the rest of ``app``.

Requests (app -> worker, one JSON object per line on stdin)::

    {"op": "render", "job_id": "...", "source": "<abs png>", "audio": "<abs wav>", "output": "<abs mp4>"}
    {"op": "check_face", "job_id": "...", "source": "<abs png>"}
    {"op": "quit"}

Replies (worker -> app, one JSON object per line on a private channel)::

    {"type": "ready", "load_seconds": 12.3}
    {"type": "result", "job_id": "...", "ok": true, "seconds": 41.0, ...}
    {"type": "result", "job_id": "...", "ok": false, "error_code": "no_face", "seconds": 0.4}

Replies never contain exception text or file contents, only a short error code.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

MAX_LINE_BYTES = 16 * 1024
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

OP_RENDER = "render"
OP_CHECK_FACE = "check_face"
OP_QUIT = "quit"

# Error codes the worker may return.
NO_FACE = "no_face"
BAD_AUDIO = "bad_audio"
BAD_REQUEST = "bad_request"
RENDER_FAILED = "render_failed"
ERROR_CODES = frozenset({NO_FACE, BAD_AUDIO, BAD_REQUEST, RENDER_FAILED})


class ProtocolError(ValueError):
    """A line could not be parsed or failed validation. Message is safe to log."""


@dataclass(frozen=True)
class Request:
    op: str
    job_id: str = ""
    source: str = ""
    audio: str = ""
    output: str = ""


def encode(message: dict[str, Any]) -> str:
    """One compact JSON object terminated by a newline."""
    return json.dumps(message, separators=(",", ":"), ensure_ascii=True) + "\n"


def decode(line: str) -> dict[str, Any]:
    if len(line.encode("utf-8", errors="replace")) > MAX_LINE_BYTES:
        raise ProtocolError("line too long")
    try:
        message = json.loads(line)
    except (ValueError, TypeError) as exc:
        raise ProtocolError("not valid JSON") from exc
    if not isinstance(message, dict):
        raise ProtocolError("message must be a JSON object")
    return message


def _abs_path(message: dict[str, Any], key: str) -> str:
    value = message.get(key)
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProtocolError(f"{key} must be a non-empty string")
    if not os.path.isabs(value):
        raise ProtocolError(f"{key} must be an absolute path")
    return value


def parse_request(line: str) -> Request:
    """Parse and validate one request line. Raises ``ProtocolError``."""
    message = decode(line)
    op = message.get("op")
    if op == OP_QUIT:
        return Request(op=OP_QUIT)
    if op not in (OP_RENDER, OP_CHECK_FACE):
        raise ProtocolError("unknown op")
    job_id = message.get("job_id")
    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
        raise ProtocolError("invalid job_id")
    if op == OP_CHECK_FACE:
        return Request(op=op, job_id=job_id, source=_abs_path(message, "source"))
    return Request(
        op=op,
        job_id=job_id,
        source=_abs_path(message, "source"),
        audio=_abs_path(message, "audio"),
        output=_abs_path(message, "output"),
    )


def request_line(
    op: str,
    job_id: str = "",
    source: str = "",
    audio: str = "",
    output: str = "",
) -> str:
    """Build a request line (used by the app side)."""
    if op == OP_QUIT:
        return encode({"op": OP_QUIT})
    message: dict[str, Any] = {"op": op, "job_id": job_id, "source": source}
    if op == OP_RENDER:
        message.update(audio=audio, output=output)
    return encode(message)


def ready_message(load_seconds: float) -> dict[str, Any]:
    return {"type": "ready", "load_seconds": round(load_seconds, 2)}


def ok_result(job_id: str, seconds: float, **extra: Any) -> dict[str, Any]:
    return {"type": "result", "job_id": job_id, "ok": True, "seconds": round(seconds, 2), **extra}


def error_result(job_id: str, code: str, seconds: float = 0.0) -> dict[str, Any]:
    if code not in ERROR_CODES:
        code = RENDER_FAILED
    return {
        "type": "result",
        "job_id": job_id,
        "ok": False,
        "error_code": code,
        "seconds": round(seconds, 2),
    }


@dataclass(frozen=True)
class Result:
    job_id: str
    ok: bool
    seconds: float
    error_code: Optional[str] = None
    extra: Optional[dict[str, Any]] = None


def parse_reply(line: str) -> tuple[str, Any]:
    """Parse a worker reply into ``("ready", load_seconds)`` or ``("result", Result)``."""
    message = decode(line)
    kind = message.get("type")
    if kind == "ready":
        return "ready", float(message.get("load_seconds", 0.0))
    if kind == "result":
        job_id = message.get("job_id")
        if not isinstance(job_id, str):
            raise ProtocolError("result without job_id")
        extra = {k: v for k, v in message.items() if k not in {"type", "job_id", "ok", "seconds", "error_code"}}
        code = message.get("error_code")
        return "result", Result(
            job_id=job_id,
            ok=bool(message.get("ok")),
            seconds=float(message.get("seconds", 0.0)),
            error_code=code if isinstance(code, str) and code in ERROR_CODES else (RENDER_FAILED if not message.get("ok") else None),
            extra=extra or None,
        )
    raise ProtocolError("unknown reply type")
