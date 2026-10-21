"""Warm Ditto worker: loads the models once, then renders many jobs.

Run with the isolated Ditto conda env's Python (the FastAPI app starts it for
you; run it by hand only to debug)::

    & "$env:USERPROFILE\\anaconda3\\envs\\ditto\\python.exe" scripts\\ditto_worker.py

Protocol: JSON lines. Requests arrive on stdin; replies go to a private copy of
the original stdout. Everything else Ditto, tqdm or ffmpeg prints is sent to
stderr, so it can never corrupt the protocol. See ``app/avatar/protocol.py``.

Replies carry only short error codes, never exception text or file contents.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DITTO_REPO = PROJECT_ROOT / "third_party" / "ditto-talkinghead"
CKPT = PROJECT_ROOT / "checkpoints"
DEFAULT_DATA_ROOT = CKPT / "ditto_pytorch"
DEFAULT_CFG_PKL = CKPT / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl"
FPS = 25
AUDIO_SR = 16000
# Longest side of the rendered video. The UI shows the avatar in a ~352 CSS px circle, so 1024 is
# ample; Ditto's own default is 1920, which makes the per-frame warp/blend/encode work much slower.
DEFAULT_MAX_SIZE = 1024
MIN_MAX_SIZE = 256
REGISTRATION_CACHE_SIZE = 8
_STILL_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def _max_size_from_env(environ=None) -> int:
    """DITTO_MAX_SIZE override; invalid or implausibly small values fall back to the default."""
    raw = (os.environ if environ is None else environ).get("DITTO_MAX_SIZE", "")
    try:
        value = int(str(raw).strip())
    except ValueError:
        return DEFAULT_MAX_SIZE
    return value if value >= MIN_MAX_SIZE else DEFAULT_MAX_SIZE


MAX_SIZE = _max_size_from_env()


def _load_protocol():
    """Load app/avatar/protocol.py by path (stdlib only; avoids importing the app)."""
    path = PROJECT_ROOT / "app" / "avatar" / "protocol.py"
    spec = importlib.util.spec_from_file_location("avatar_protocol", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["avatar_protocol"] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


proto = _load_protocol()


def _open_private_channel():
    """Return a text writer on a duplicate of stdout, then point fd 1 at stderr."""
    channel = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8", newline="\n")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return channel


def _ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


class CachedRegistrar:
    """LRU cache around Ditto's ``avatar_registrar`` for still images.

    Registration (face detection, landmarks, appearance and motion extraction) is deterministic for
    a given image file and crop settings, so repeated renders of the same traveler face can skip it.
    The key includes the file's mtime and size, so a rewritten file is registered again. The cache
    keeps a private master copy and every call returns a deep copy, so nothing downstream can
    mutate cached state. Failures are never cached. Videos and unknown suffixes are not cached.
    """

    def __init__(self, inner, maxsize: int = REGISTRATION_CACHE_SIZE) -> None:
        self._inner = inner
        self._maxsize = max(1, int(maxsize))
        self._entries: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __getattr__(self, name):  # only called for attributes not found on the wrapper
        return getattr(self._inner, name)

    @staticmethod
    def _key(source, max_dim, kwargs):
        if Path(str(source)).suffix.lower() not in _STILL_IMAGE_SUFFIXES:
            return None
        try:
            st = os.stat(source)
            real = os.path.realpath(source)
        except (OSError, TypeError, ValueError):
            return None  # let the inner registrar report the problem as it always did
        crop = tuple(sorted((k, repr(v)) for k, v in kwargs.items()))
        return (real, st.st_mtime_ns, st.st_size, repr(max_dim), crop)

    def _prune_stale(self) -> None:
        """Drop entries whose photo file is gone (deleted on reset/expiry) so no face data lingers."""
        with self._lock:
            for key in [k for k in self._entries if not os.path.exists(k[0])]:
                del self._entries[key]

    def __call__(self, source, *args, **kwargs):
        self._prune_stale()
        # n_frames is irrelevant for still images (check_face passes 1, setup passes -1), so it is
        # kept out of the key. Positional extras are not understood here: bypass the cache.
        key = None
        if not args:
            lookup = dict(kwargs)
            max_dim = lookup.pop("max_dim", None)
            lookup.pop("n_frames", None)
            key = self._key(source, max_dim, lookup)
        if key is None:
            return self._inner(source, *args, **kwargs)

        with self._lock:
            master = self._entries.get(key)
            if master is not None:
                self._entries.move_to_end(key)
                self.hits += 1
        if master is not None:
            print(f"[worker] registration cache hit ({len(self._entries)} cached)", file=sys.stderr)
            return copy.deepcopy(master)

        result = self._inner(source, *args, **kwargs)  # raises on failure: nothing is stored
        master = copy.deepcopy(result)
        with self._lock:
            self.misses += 1
            for old in [k for k in self._entries if k[0] == key[0]]:
                del self._entries[old]  # the same path with an older mtime/size can never hit again
            self._entries[key] = master
            self._entries.move_to_end(key)
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)
        print(f"[worker] registration cache miss ({len(self._entries)} cached)", file=sys.stderr)
        return result


def compute_roi(M, frame_h: int, frame_w: int, size: int = 512, margin: int = 2):
    """Bounding box ``(x0, y0, x1, y1)`` of the warped ``size`` x ``size`` crop, clamped to the frame.

    The crop is extended by one source pixel on each side because bilinear sampling against
    warpAffine's constant-zero border still gives non-zero weight there; ``margin`` adds a little
    more slack for rounding. Returns None when the box does not intersect the frame.
    """
    lo, hi = -1.0, float(size)
    xs, ys = [], []
    for cx, cy in ((lo, lo), (hi, lo), (hi, hi), (lo, hi)):
        xs.append(M[0][0] * cx + M[0][1] * cy + M[0][2])
        ys.append(M[1][0] * cx + M[1][1] * cy + M[1][2])
    values = xs + ys
    if not all(math.isfinite(float(v)) for v in values):
        return None
    x0 = max(0, math.floor(min(xs)) - margin)
    y0 = max(0, math.floor(min(ys)) - margin)
    x1 = min(frame_w, math.ceil(max(xs)) + margin)
    y1 = min(frame_h, math.ceil(max(ys)) + margin)
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1), int(y1)


class RoiPutBack:
    """Drop-in for Ditto's ``PutBack`` that only warps and blends inside the crop's bounding box.

    Ditto warps the 512x512 render and its mask over the whole (large) frame and blends every pixel,
    although the mask is zero outside the warped crop. Here the work is limited to the bounding box;
    the rest of the frame is copied unchanged, which equals the original result there. Inside the
    box the math is the original's, shifted by the box origin (differences of at most one level can
    come from warpAffine's fixed-point rounding). Returns a new array on every call.
    """

    def __init__(self, mask_ori_float) -> None:
        import numpy as np

        self.mask_ori_float = np.ascontiguousarray(mask_ori_float, dtype=np.float32)
        self.size = int(self.mask_ori_float.shape[0])

    def __call__(self, frame_rgb, render_image, M_c2o):
        import numpy as np

        h, w = frame_rgb.shape[:2]
        M = np.asarray(M_c2o, dtype=np.float64)[:2, :]
        if np.isfinite(M).all():
            roi = compute_roi(M, h, w, self.size)
            if roi is None:
                return frame_rgb.copy()  # crop entirely outside the frame: nothing to blend
        else:
            roi = (0, 0, w, h)  # garbage matrix: keep the original full-frame behaviour
        try:
            return self._blend(frame_rgb, render_image, M, roi)
        except Exception:  # noqa: BLE001 - stock PutBack would not fail here: retry on the full frame
            traceback.print_exc()
            return self._blend(frame_rgb, render_image, M, (0, 0, w, h))

    def _blend(self, frame_rgb, render_image, M, roi):
        import cv2
        import numpy as np

        out = frame_rgb.copy()
        x0, y0, x1, y1 = roi
        shifted = M.copy()
        shifted[0, 2] -= x0
        shifted[1, 2] -= y0
        dsize = (x1 - x0, y1 - y0)
        mask = cv2.warpAffine(self.mask_ori_float, shifted, dsize=dsize, flags=cv2.INTER_LINEAR).clip(0, 1)
        warped = cv2.warpAffine(render_image, shifted, dsize=dsize, flags=cv2.INTER_LINEAR)
        mask = mask[:, :, None]
        blended = mask * warped + (1.0 - mask) * out[y0:y1, x0:x1]
        np.clip(blended, 0, 255, out=blended)
        out[y0:y1, x0:x1] = blended.astype(np.uint8)
        return out


def install_speedups(sdk) -> None:
    """Swap in the cached registrar and ROI putback; on any failure keep Ditto's originals."""
    try:
        putback = RoiPutBack(sdk.putback.mask_ori_float)
        sdk.avatar_registrar = CachedRegistrar(sdk.avatar_registrar)
        sdk.putback = putback
    except Exception:  # noqa: BLE001 - the worker must still start with stock behaviour
        traceback.print_exc()


class Worker:
    def __init__(self, cfg_pkl: Path, data_root: Path) -> None:
        # Imported here: Ditto's modules live in the cloned repo, not on sys.path by default.
        from stream_pipeline_offline import StreamSDK  # type: ignore

        self.sdk = StreamSDK(str(cfg_pkl), str(data_root))
        install_speedups(self.sdk)
        self.ffmpeg = _ffmpeg_exe()
        print(f"[worker] max_size={MAX_SIZE}", file=sys.stderr)

    # ------------------------------------------------------------------ helpers
    def _abort_pipeline(self) -> None:
        """Stop and join pipeline threads after a failed render so the next job starts clean."""
        sdk = self.sdk
        try:
            sdk.stop_event.set()
            for thread in getattr(sdk, "thread_list", []):
                thread.join(timeout=15)
            try:
                sdk.writer.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                sdk.writer_pbar.close()
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    @staticmethod
    def _remove(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    # --------------------------------------------------------------- operations
    def check_face(self, req) -> dict:
        started = time.perf_counter()
        kw = self.sdk.default_kwargs
        try:
            self.sdk.avatar_registrar(
                req.source,
                max_dim=MAX_SIZE,
                n_frames=1,
                crop_scale=kw.get("crop_scale", 2.3),
                crop_vx_ratio=kw.get("crop_vx_ratio", 0),
                crop_vy_ratio=kw.get("crop_vy_ratio", -0.125),
                crop_flag_do_rot=kw.get("crop_flag_do_rot", True),
            )
        except Exception:  # noqa: BLE001 - the registrar raises TypeError when no face is found
            traceback.print_exc()
            return proto.error_result(req.job_id, proto.NO_FACE, time.perf_counter() - started)
        return proto.ok_result(req.job_id, time.perf_counter() - started)

    def render(self, req) -> dict:
        import librosa
        import numpy as np

        started = time.perf_counter()
        if not os.path.isfile(req.source):
            return proto.error_result(req.job_id, proto.BAD_REQUEST)
        try:
            audio, _ = librosa.core.load(req.audio, sr=AUDIO_SR)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return proto.error_result(req.job_id, proto.BAD_AUDIO)
        if len(audio) < AUDIO_SR // 10:  # under 0.1 s: nothing to animate
            return proto.error_result(req.job_id, proto.BAD_AUDIO)
        audio_seconds = len(audio) / AUDIO_SR
        n_frames = math.ceil(audio_seconds * FPS)

        os.makedirs(os.path.dirname(req.output), exist_ok=True)
        self._remove(req.output)
        sdk = self.sdk
        tmp_path = req.output + ".tmp.mp4"
        try:
            try:
                sdk.setup(req.source, req.output, max_size=MAX_SIZE)
            except Exception:  # noqa: BLE001 - registration failed (no face / bad image)
                traceback.print_exc()
                return proto.error_result(req.job_id, proto.NO_FACE, time.perf_counter() - started)
            try:
                sdk.setup_Nd(N_d=n_frames, fade_in=-1, fade_out=-1, ctrl_info={})
                aud_feat = sdk.wav2feat.wav2feat(np.asarray(audio, dtype=np.float32))
                sdk.audio2motion_queue.put(aud_feat)
                sdk.close()  # flushes the pipeline and joins its threads
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                self._abort_pipeline()
                return proto.error_result(req.job_id, proto.RENDER_FAILED, time.perf_counter() - started)

            mux = subprocess.run(
                [
                    self.ffmpeg, "-loglevel", "error", "-y",
                    "-i", tmp_path, "-i", req.audio,
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-shortest",
                    req.output,
                ],
                stdin=subprocess.DEVNULL,
                stdout=sys.stderr,
                stderr=sys.stderr,
            )
            if mux.returncode != 0 or not os.path.isfile(req.output) or os.path.getsize(req.output) < 1000:
                self._remove(req.output)
                return proto.error_result(req.job_id, proto.RENDER_FAILED, time.perf_counter() - started)
        finally:
            self._remove(tmp_path)

        extra = {"audio_seconds": round(audio_seconds, 2), "frames": n_frames}
        try:
            import torch

            if torch.cuda.is_available():
                extra["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1048576)
        except Exception:  # noqa: BLE001
            pass
        return proto.ok_result(req.job_id, time.perf_counter() - started, **extra)

    def handle(self, req) -> dict:
        try:
            if req.op == proto.OP_CHECK_FACE:
                return self.check_face(req)
            return self.render(req)
        except Exception:  # noqa: BLE001 - never let one job kill the worker
            traceback.print_exc()
            return proto.error_result(req.job_id, proto.RENDER_FAILED)


def main() -> int:
    parser = argparse.ArgumentParser(description="Warm Ditto worker (JSON lines over stdio)")
    parser.add_argument("--repo", default=str(DITTO_REPO))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--cfg-pkl", default=str(DEFAULT_CFG_PKL))
    args = parser.parse_args()

    channel = _open_private_channel()
    write_lock = threading.Lock()

    def send(message: dict) -> None:
        with write_lock:
            channel.write(proto.encode(message))
            channel.flush()

    repo = Path(args.repo)
    for required in (repo / "inference.py", Path(args.data_root), Path(args.cfg_pkl)):
        if not required.exists():
            print(f"[worker] missing required path: {required}", file=sys.stderr)
            return 2

    sys.path.insert(0, str(repo))
    os.chdir(repo)  # Ditto resolves some relative paths from its own root

    load_started = time.perf_counter()
    try:
        worker = Worker(Path(args.cfg_pkl), Path(args.data_root))
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 3
    send(proto.ready_message(time.perf_counter() - load_started))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = proto.parse_request(line)
        except proto.ProtocolError as exc:
            print(f"[worker] ignoring bad request: {exc}", file=sys.stderr)
            send(proto.error_result("unknown", proto.BAD_REQUEST))
            continue
        if req.op == proto.OP_QUIT:
            break
        send(worker.handle(req))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
