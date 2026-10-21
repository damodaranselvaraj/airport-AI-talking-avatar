"""A stand-in for scripts/ditto_worker.py that speaks the same protocol, instantly.

Behaviour is chosen with environment variables so tests can drive the real
``DittoWorker`` / ``AvatarService`` code without a GPU:

* ``STUB_CRASH_MARKER`` - path; if it does not exist the worker creates it and
  exits abruptly in the middle of its first render (a crash that heals on restart).
* ``STUB_HANG_RENDER``  - "1": renders never reply (simulates a hung process).
* ``STUB_NO_FACE``      - "1": every ``check_face`` reports ``no_face``.
* ``STUB_START_DELAY``  - seconds to sleep before reporting ready.
* ``STUB_EXIT_BEFORE_READY`` - "1": exit immediately (a worker that cannot start).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("avatar_protocol", ROOT / "app" / "avatar" / "protocol.py")
proto = importlib.util.module_from_spec(spec)
sys.modules["avatar_protocol"] = proto
spec.loader.exec_module(proto)


def send(message: dict) -> None:
    sys.stdout.write(proto.encode(message))
    sys.stdout.flush()


def main() -> int:
    if os.environ.get("STUB_EXIT_BEFORE_READY") == "1":
        return 3
    time.sleep(float(os.environ.get("STUB_START_DELAY", "0")))
    send(proto.ready_message(0.01))
    marker = os.environ.get("STUB_CRASH_MARKER")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = proto.parse_request(line)
        except proto.ProtocolError:
            send(proto.error_result("unknown", proto.BAD_REQUEST))
            continue
        if req.op == proto.OP_QUIT:
            break
        if req.op == proto.OP_CHECK_FACE:
            if os.environ.get("STUB_NO_FACE") == "1":
                send(proto.error_result(req.job_id, proto.NO_FACE, 0.01))
            else:
                send(proto.ok_result(req.job_id, 0.01))
            continue
        if marker and not os.path.exists(marker):
            Path(marker).write_text("crashed")
            os._exit(1)
        if os.environ.get("STUB_HANG_RENDER") == "1":
            time.sleep(3600)
        Path(req.output).parent.mkdir(parents=True, exist_ok=True)
        Path(req.output).write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 2048)
        send(proto.ok_result(req.job_id, 0.01, audio_seconds=1.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
