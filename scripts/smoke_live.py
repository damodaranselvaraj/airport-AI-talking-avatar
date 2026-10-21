"""Smoke-test a RUNNING kiosk server end to end (real models, real APIs).

    python scripts/smoke_live.py                      # default questions
    python scripts/smoke_live.py "Where is gate C22?" # your own
    python scripts/smoke_live.py --voice              # also: Kokoro audio -> /api/transcribe
    python scripts/smoke_live.py --avatar             # also: wait for the talking-face MP4 and save it

Prints status, text, citation, timings and whether audio became available. It
never prints API keys (it has none: they live only in the server process).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULTS = [
    "Hello",
    "Where is gate C22?",
    "Where is the prayer room?",
    "What is the status of flight NIA 442?",
    "Where can I buy a ticket to the moon?",
]


def call(base: str, path: str, payload: dict | None = None, raw: bytes | None = None, headers: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else raw
    hdrs = {"Content-Type": "application/json"} if payload is not None else (headers or {})
    req = urllib.request.Request(base + path, data=data, headers=hdrs, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers, err.read()


def multipart(field: str, filename: str, content: bytes, ctype: str) -> tuple[bytes, dict]:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def ask(base: str, session: str, text: str, input_type: str = "typed") -> dict:
    t0 = time.perf_counter()
    status, _, body = call(base, "/api/ask", {"session_id": session, "text": text, "input_type": input_type})
    elapsed = time.perf_counter() - t0
    out = json.loads(body)
    out["_http"], out["_text_ms"] = status, round(elapsed * 1000)
    if status == 200 and out.get("audio_url"):
        t1 = time.perf_counter()
        while time.perf_counter() - t1 < 60:
            s, _, audio = call(base, out["audio_url"])
            if s != 202:
                out["_audio"] = (s, len(audio), round((time.perf_counter() - t0) * 1000))
                break
            time.sleep(0.25)
    return out


def fetch_video(base: str, out: dict, wait_s: float, save_dir: Path) -> None:
    """Poll /api/video until it is ready (or fails / times out); save the MP4 and probe its streams."""
    url, t0 = out["video_url"], time.perf_counter()
    while time.perf_counter() - t0 < wait_s:
        status, _, data = call(base, url)
        if status != 202:
            break
        time.sleep(1.0)
    else:
        out["_video"] = ("timeout", 0, round(time.perf_counter() - t0, 1))
        return
    elapsed = round(time.perf_counter() - t0, 1)
    if status != 200:
        out["_video"] = (status, 0, elapsed)
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"smoke_{out['turn_id']}.mp4"
    path.write_bytes(data)
    streams = "?"
    if shutil.which("ffprobe"):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True,
        )
        streams = ",".join(sorted(set(probe.stdout.split()))) or "?"
    out["_video"] = (200, len(data), elapsed, streams, str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="*")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--avatar", action="store_true", help="wait for each turn's talking-face video and save it")
    ap.add_argument("--video-wait", type=float, default=150.0, help="seconds to wait for each video")
    ap.add_argument("--voice", action="store_true", help="synthesize a question with Kokoro and transcribe it")
    args = ap.parse_args()
    session = uuid.uuid4().hex

    for q in args.questions or DEFAULTS:
        r = ask(args.base, session, q)
        print(f"\nQ: {q}")
        if r["_http"] != 200:
            print(f"   HTTP {r['_http']}: {r}")
            continue
        cites = ", ".join(c["source"] for c in r["citations"]) or "-"
        print(f"   [{r['status']}] tier={r['evidence_tier']} text in {r['_text_ms']} ms; source: {cites}")
        print(f"   A: {r['text']}")
        print(f"   audio: {r.get('_audio', 'n/a')}  (status, bytes, ms since ask)")
        if args.avatar:
            if r.get("video_url"):
                fetch_video(args.base, r, args.video_wait, Path(__file__).resolve().parents[1] / "sessions")
                print(f"   video: {r['_video']}  (status, bytes, s waited, streams, file)")
            else:
                print("   video: none for this turn (not spoken, or the avatar is off)")

    if args.voice:
        from app.text_to_speech.kokoro_service import KokoroSynthesizer

        question = "Where is gate C22?"
        wav = KokoroSynthesizer().synthesize(question)
        body, hdrs = multipart("audio", "question.wav", wav, "audio/wav")
        t0 = time.perf_counter()
        status, _, resp = call(args.base, "/api/transcribe", raw=body, headers=hdrs)
        print(f"\nVOICE: /api/transcribe -> HTTP {status} in {round((time.perf_counter() - t0) * 1000)} ms")
        print(f"   {resp.decode()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
