"""Type text -> hear it -> see the avatar talk.

Chains the project's real TTS (Kokoro) to Ditto so you can test the talking
head with your own words, without needing a pre-recorded WAV:

    text -> Kokoro WAV (CPU) -> Ditto (GPU) -> MP4

Run it with the isolated Ditto conda env's Python (Kokoro + Ditto deps live
there):

    & "$env:USERPROFILE\anaconda3\envs\ditto\python.exe" scripts\text_to_avatar.py --text "Where is gate C22?"

Options:
    --text   what the avatar should say (required)
    --photo  face image (default: Ditto example image)
    --voice  Kokoro voice id (default: af_heart)
    --lang   Kokoro lang_code (default: a = American English)
    --out    output mp4 (default: sessions/text_avatar.mp4)

This is a preview of the Stage 2 TTS + Stage 3 avatar path, used here as a
manual test tool.
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
SAMPLE_RATE = 24000


def synthesize(text: str, voice: str, lang: str, wav_path: Path) -> None:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=lang)
    chunks = [np.asarray(audio, dtype=np.float32) for _, _, audio in pipeline(text, voice=voice)]
    if not chunks:
        raise RuntimeError("Kokoro produced no audio for the given text.")
    audio = np.concatenate(chunks)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), audio, SAMPLE_RATE)
    print(f"[tts] wrote {wav_path} ({len(audio) / SAMPLE_RATE:.2f}s)")


def animate(photo: Path, wav_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    cmd = [
        sys.executable,
        "inference.py",
        "--data_root", str(DATA_ROOT),
        "--cfg_pkl", str(CFG_PKL),
        "--audio_path", str(wav_path),
        "--source_path", str(photo),
        "--output_path", str(out_path),
    ]
    print("[ditto] running inference...")
    rc = subprocess.run(cmd, cwd=str(DITTO_REPO)).returncode
    tmp = Path(str(out_path) + ".tmp.mp4")
    if tmp.exists():
        tmp.unlink()
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Text -> Kokoro -> Ditto avatar")
    parser.add_argument("--text", required=True)
    parser.add_argument("--photo", default=str(DITTO_REPO / "example" / "image.png"))
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--lang", default="a")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "sessions" / "text_avatar.mp4"))
    args = parser.parse_args()

    for p in (DITTO_REPO / "inference.py", DATA_ROOT, CFG_PKL, Path(args.photo)):
        if not Path(p).exists():
            print(f"[error] missing required path: {p}", file=sys.stderr)
            return 2

    out_path = Path(args.out)
    wav_path = out_path.with_suffix(".wav")

    synthesize(args.text, args.voice, args.lang, wav_path)
    rc = animate(Path(args.photo), wav_path, out_path)
    if rc != 0:
        print(f"[error] Ditto inference exited {rc}", file=sys.stderr)
        return rc

    if not out_path.exists() or out_path.stat().st_size < 100_000:
        print(f"[error] no valid output at {out_path}", file=sys.stderr)
        return 1

    print(f"[ok] {out_path} ({out_path.stat().st_size / 1_048_576:.2f} MB) — open it to watch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
