"""Task 2 — Ditto de-risk spike.

Standalone check that Ditto PyTorch inference works end to end on this machine:
a photo + a WAV -> a lip-synced MP4. Exits non-zero if no valid video is
produced, so it doubles as a smoke test.

Run it with the isolated Ditto conda env's Python (so the heavy deps and the
NumPy blend patch are in scope):

    & "$env:USERPROFILE\anaconda3\envs\ditto\python.exe" scripts\ditto_spike.py

Defaults to the Ditto repo's bundled example assets; override with --photo /
--audio / --out to prove "any photo + any WAV".
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DITTO_REPO = PROJECT_ROOT / "third_party" / "ditto-talkinghead"
CKPT = PROJECT_ROOT / "checkpoints"
DATA_ROOT = CKPT / "ditto_pytorch"
CFG_PKL = CKPT / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Ditto de-risk spike")
    parser.add_argument("--photo", default=str(DITTO_REPO / "example" / "image.png"))
    parser.add_argument("--audio", default=str(DITTO_REPO / "example" / "audio.wav"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "sessions" / "ditto_spike.mp4"))
    args = parser.parse_args()

    for p in (DITTO_REPO / "inference.py", DATA_ROOT, CFG_PKL, Path(args.photo), Path(args.audio)):
        if not Path(p).exists():
            print(f"[spike] MISSING required path: {p}", file=sys.stderr)
            return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    cmd = [
        sys.executable,
        "inference.py",
        "--data_root", str(DATA_ROOT),
        "--cfg_pkl", str(CFG_PKL),
        "--audio_path", str(args.audio),
        "--source_path", str(args.photo),
        "--output_path", str(out_path),
    ]
    print("[spike] running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(DITTO_REPO))
    if result.returncode != 0:
        print(f"[spike] inference exited {result.returncode}", file=sys.stderr)
        return result.returncode

    # Clean up Ditto's intermediate file and assert a real video exists.
    tmp = Path(str(out_path) + ".tmp.mp4")
    if tmp.exists():
        tmp.unlink()

    if not out_path.exists() or out_path.stat().st_size < 100_000:
        print(f"[spike] FAIL: no valid output at {out_path}", file=sys.stderr)
        return 1

    print(f"[spike] OK: {out_path} ({out_path.stat().st_size / 1_048_576:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
