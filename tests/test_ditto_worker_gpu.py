"""Real Ditto worker on the GPU: render twice on one warm process.

Opt-in (needs the ``ditto`` conda env, checkpoints and a CUDA GPU)::

    pytest -m gpu tests/test_ditto_worker_gpu.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DITTO_PY = Path(os.environ.get("DITTO_PYTHON", Path.home() / "anaconda3" / "envs" / "ditto" / "python.exe"))
EXAMPLE = ROOT / "third_party" / "ditto-talkinghead" / "example"
CKPT = ROOT / "checkpoints" / "ditto_pytorch"

pytestmark = pytest.mark.gpu


def _bench():
    spec = importlib.util.spec_from_file_location("ditto_bench", ROOT / "scripts" / "ditto_bench.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ditto_bench"] = module
    spec.loader.exec_module(module)
    return module


def _flat_gray_png(width: int, height: int, level: int) -> bytes:
    """A plain RGB PNG with no face in it (stdlib only)."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    row = b"\x00" + bytes([level]) * (width * 3)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )


@pytest.fixture(scope="module")
def worker():
    if not (DITTO_PY.exists() and CKPT.exists() and (EXAMPLE / "image.png").exists()):
        pytest.skip("Ditto env/checkpoints/example assets not installed")
    bench = _bench()
    proc = bench.WorkerProcess(str(DITTO_PY))
    try:
        proc.wait("ready", timeout=600)
        yield bench, proc
    finally:
        proc.close()


def test_two_renders_on_one_warm_worker(worker, tmp_path):
    bench, proc = worker
    proto = bench.proto
    photo = str(EXAMPLE / "image.png")

    proc.send(proto.request_line(proto.OP_CHECK_FACE, "chk", source=photo))
    assert proc.wait("result", timeout=300).ok

    seconds = []
    for i in (1, 2):
        wav = tmp_path / f"in{i}.wav"
        bench.cut_wav(EXAMPLE / "audio.wav", wav, 2.0)
        out = tmp_path / f"out{i}.mp4"
        proc.send(proto.request_line(proto.OP_RENDER, f"r{i}", source=photo, audio=str(wav), output=str(out)))
        res = proc.wait("result", timeout=900)
        assert res.ok, res.error_code
        assert out.stat().st_size > 5_000
        assert not Path(str(out) + ".tmp.mp4").exists()
        seconds.append(res.seconds)
    # Second job reuses the loaded models; it must not pay the model-load cost again.
    assert seconds[1] < seconds[0] * 1.5


def test_face_free_image_reports_no_face(worker, tmp_path):
    bench, proc = worker
    proto = bench.proto
    blank = tmp_path / "blank.png"
    blank.write_bytes(_flat_gray_png(256, 256, 127))
    proc.send(proto.request_line(proto.OP_CHECK_FACE, "nf", source=str(blank)))
    res = proc.wait("result", timeout=300)
    assert not res.ok and res.error_code == proto.NO_FACE
