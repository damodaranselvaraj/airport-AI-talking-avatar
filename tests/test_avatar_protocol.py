"""Worker protocol: parsing, validation and reply shapes (no GPU, no Ditto)."""
from __future__ import annotations

import os

import pytest

from app.avatar import protocol as p

ABS = os.path.abspath(os.sep + "tmp")


def _abs(name: str) -> str:
    return os.path.join(ABS, name)


def test_render_request_roundtrip():
    line = p.request_line(p.OP_RENDER, "job-1", source=_abs("f.png"), audio=_abs("a.wav"), output=_abs("o.mp4"))
    assert line.endswith("\n") and "\n" not in line[:-1]
    req = p.parse_request(line)
    assert req == p.Request(p.OP_RENDER, "job-1", _abs("f.png"), _abs("a.wav"), _abs("o.mp4"))


def test_check_face_request_needs_only_source():
    req = p.parse_request(p.request_line(p.OP_CHECK_FACE, "chk_2", source=_abs("f.png")))
    assert req.op == p.OP_CHECK_FACE and req.audio == "" and req.output == ""


def test_quit_request():
    assert p.parse_request(p.request_line(p.OP_QUIT)).op == p.OP_QUIT


@pytest.mark.parametrize(
    "line",
    [
        "",
        "not json",
        "[1, 2]",
        '{"op": "nope"}',
        '{"op": "render"}',
        '{"op": "render", "job_id": "bad id!", "source": "/x", "audio": "/a", "output": "/o"}',
        '{"op": "render", "job_id": "a", "source": "relative.png", "audio": "/a", "output": "/o"}',
        '{"op": "check_face", "job_id": "a", "source": ""}',
        '{"op": "check_face", "job_id": "a", "source": 5}',
    ],
)
def test_bad_requests_are_rejected(line):
    with pytest.raises(p.ProtocolError):
        p.parse_request(line)


def test_overlong_line_rejected():
    with pytest.raises(p.ProtocolError):
        p.decode("x" * (p.MAX_LINE_BYTES + 1))


def test_ready_and_ok_reply_parse():
    kind, value = p.parse_reply(p.encode(p.ready_message(12.345)))
    assert kind == "ready" and value == pytest.approx(12.35)
    kind, res = p.parse_reply(p.encode(p.ok_result("j", 3.14159, audio_seconds=2.0, vram_peak_mb=900)))
    assert kind == "result" and res.ok and res.seconds == pytest.approx(3.14)
    assert res.extra == {"audio_seconds": 2.0, "vram_peak_mb": 900}


def test_error_reply_has_code_and_no_text():
    line = p.encode(p.error_result("j", p.NO_FACE, 0.4))
    _, res = p.parse_reply(line)
    assert not res.ok and res.error_code == p.NO_FACE
    assert set(p.decode(line)) == {"type", "job_id", "ok", "error_code", "seconds"}


def test_unknown_error_code_is_normalised():
    assert p.error_result("j", "Traceback: boom")["error_code"] == p.RENDER_FAILED
    _, res = p.parse_reply('{"type":"result","job_id":"j","ok":false,"error_code":"weird"}')
    assert res.error_code == p.RENDER_FAILED


def test_unknown_reply_type_rejected():
    with pytest.raises(p.ProtocolError):
        p.parse_reply('{"type": "banana"}')
