"""Benchmark the warm Ditto worker: one-time load, then per-job render times.

Run with the Ditto conda env's Python (it also launches the worker with it)::

    & "$env:USERPROFILE\\anaconda3\\envs\\ditto\\python.exe" scripts\\ditto_bench.py

It cuts the Ditto example audio into 3 s, 6 s and 12 s clips, sends a
``check_face`` job and then the three renders to ONE worker process, and prints
load time, per-job time, seconds of render per second of audio, and GPU memory
(from nvidia-smi). Use the numbers to set ``AVATAR_TIMEOUT_SECONDS``.

Options: --photo, --audio, --python, --keep (keep the rendered mp4s).
"""
from __future__ import annotations

import argparse
import importlib.util
import queue
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DITTO_REPO = PROJECT_ROOT / "third_party" / "ditto-talkinghead"
WORKER = PROJECT_ROOT / "scripts" / "ditto_worker.py"
LOG_PATH = PROJECT_ROOT / "sessions" / "ditto_bench_worker.log"


def _load_protocol():
    path = PROJECT_ROOT / "app" / "avatar" / "protocol.py"
    spec = importlib.util.spec_from_file_location("avatar_protocol", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["avatar_protocol"] = module
    spec.loader.exec_module(module)
    return module


proto = _load_protocol()


def cut_wav(src: Path, dst: Path, seconds: float) -> float:
    """Write the first ``seconds`` of ``src`` to ``dst`` (looping if src is shorter)."""
    with wave.open(str(src), "rb") as r:
        params = r.getparams()
        frames = r.readframes(r.getnframes())
    width = params.sampwidth * params.nchannels
    need = int(params.framerate * seconds) * width
    data = (frames * (need // max(len(frames), 1) + 1))[:need]
    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(data)
    return len(data) / width / params.framerate


def gpu_used_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except Exception:  # noqa: BLE001
        return None


class WorkerProcess:
    def __init__(self, python: str) -> None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(LOG_PATH, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            [python, str(WORKER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, encoding="utf-8", bufsize=1,
        )
        self.replies: "queue.Queue[tuple[str, object]]" = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                try:
                    self.replies.put(proto.parse_reply(line))
                except proto.ProtocolError:
                    pass
        self.replies.put(("eof", None))

    def wait(self, kind: str, timeout: float):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"no {kind} within {timeout:.0f}s (see {LOG_PATH})")
            got_kind, value = self.replies.get(timeout=remaining)
            if got_kind == "eof":
                raise RuntimeError(f"worker exited (code {self.proc.poll()}); see {LOG_PATH}")
            if got_kind == kind:
                return value

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

    def close(self) -> None:
        try:
            self.send(proto.request_line(proto.OP_QUIT))
            self.proc.wait(timeout=20)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        self._log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Warm Ditto worker benchmark")
    parser.add_argument("--photo", default=str(DITTO_REPO / "example" / "image.png"))
    parser.add_argument("--audio", default=str(DITTO_REPO / "example" / "audio.wav"))
    parser.add_argument("--python", default=sys.executable, help="Python of the ditto env")
    parser.add_argument("--durations", default="3,6,12", help="comma-separated seconds")
    parser.add_argument("--timeout", type=float, default=900.0, help="per-job timeout (s)")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    photo, audio = Path(args.photo).resolve(), Path(args.audio).resolve()
    for p in (WORKER, photo, audio):
        if not p.exists():
            print(f"[bench] missing: {p}", file=sys.stderr)
            return 2

    durations = [float(x) for x in args.durations.split(",") if x.strip()]
    workdir = Path(tempfile.mkdtemp(prefix="ditto_bench_"))
    idle_vram = gpu_used_mb()
    print(f"[bench] GPU memory in use before start: {idle_vram} MiB")

    worker = WorkerProcess(args.python)
    failures = 0
    try:
        t0 = time.perf_counter()
        load_s = worker.wait("ready", timeout=600)
        print(f"[bench] worker ready: model load {load_s:.1f}s (wall {time.perf_counter() - t0:.1f}s)")
        print(f"[bench] GPU memory after load: {gpu_used_mb()} MiB")

        worker.send(proto.request_line(proto.OP_CHECK_FACE, "check1", source=str(photo)))
        res = worker.wait("result", timeout=args.timeout)
        print(f"[bench] check_face: ok={res.ok} {res.seconds:.1f}s")
        failures += 0 if res.ok else 1

        rows = []
        peak_used = gpu_used_mb() or 0
        for i, seconds in enumerate(durations, 1):
            wav = workdir / f"in{i}.wav"
            actual = cut_wav(audio, wav, seconds)
            out = workdir / f"out{i}.mp4"
            worker.send(proto.request_line(proto.OP_RENDER, f"bench{i}", source=str(photo), audio=str(wav), output=str(out)))
            res = worker.wait("result", timeout=args.timeout)
            peak_used = max(peak_used, gpu_used_mb() or 0)
            size_mb = out.stat().st_size / 1048576 if out.exists() else 0.0
            ratio = res.seconds / actual if actual else float("nan")
            rows.append((i, actual, res.seconds, ratio, res.ok, res.error_code, size_mb, (res.extra or {}).get("vram_peak_mb")))
            failures += 0 if res.ok else 1
            print(f"[bench] job {i}: audio {actual:5.1f}s -> render {res.seconds:6.1f}s "
                  f"({ratio:4.1f}x) ok={res.ok} {res.error_code or ''} {size_mb:.2f} MB")

        print("\n job | audio s | render s | render/audio | ok | torch peak MiB")
        for i, actual, secs, ratio, ok, _code, _size, vram in rows:
            print(f" {i:>3} | {actual:7.1f} | {secs:8.1f} | {ratio:12.2f} | {'Y' if ok else 'N'}  | {vram}")
        print(f"\n[bench] one-time load {load_s:.1f}s; nvidia-smi peak {peak_used} MiB of GPU memory in use")
        if rows and all(r[4] for r in rows):
            print("[bench] suggested AVATAR_TIMEOUT_SECONDS >= "
                  f"{int(max(r[2] for r in rows) * 1.5) + 5} (1.5x slowest job + margin; scale for ~9 s answers)")
        if args.keep:
            print(f"[bench] outputs kept in {workdir}")
    finally:
        worker.close()
        if not args.keep:
            for f in workdir.glob("*"):
                f.unlink(missing_ok=True)
            workdir.rmdir()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
