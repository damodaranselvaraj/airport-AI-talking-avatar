# Ditto setup

Ditto runs in its **own isolated conda env** (`ditto`, Python 3.10) so its heavy,
version-pinned dependencies never collide with the FastAPI app's `.venv`
(Python 3.12). The kiosk app calls Ditto as a subprocess.

**Verified result:** example photo + 15.76 s audio → `sessions/ditto_spike.mp4`
(1432×1432, H.264 + AAC) in ~2 minutes on the RTX 4050 (6 GB). Ditto owns the
GPU; the ONNX aux models fall back to CPU.

## 1. Clone the inference repo

```powershell
git clone --depth 1 https://github.com/antgroup/ditto-talkinghead third_party/ditto-talkinghead
```

## 2. Create the env and install deps

```powershell
conda create -y -n ditto python=3.10
$py = "$env:USERPROFILE\anaconda3\envs\ditto\python.exe"

# Torch 2.5.1 + CUDA 12.1 (driver supports it; wheels bundle the CUDA runtime)
& $py -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# PyTorch-path deps (TensorRT / polygraphy intentionally skipped)
& $py -m pip install numpy==2.0.1 audioread==3.0.1 cffi==1.17.1 cython==3.0.11 `
  decorator==5.1.1 filetype==1.2.0 imageio==2.36.1 imageio-ffmpeg==0.5.1 joblib==1.4.2 `
  lazy-loader==0.4 librosa==0.10.2.post1 llvmlite==0.43.0 msgpack==1.1.0 numba==0.60.0 `
  opencv-python-headless==4.10.0.84 packaging==24.2 platformdirs==4.3.6 pooch==1.8.2 `
  pycparser==2.22 scikit-image==0.25.0 scikit-learn==1.6.0 scipy==1.15.0 soundfile==0.13.0 `
  soxr==0.5.0.post1 threadpoolctl==3.5.0 tifffile==2024.12.12 tqdm==4.67.1 colored

# Aux-model runtimes omitted from upstream environment.yaml, plus einops (LMDM)
& $py -m pip install onnxruntime==1.19.2 mediapipe einops "huggingface_hub[hf_transfer]"
```

## 3. The Windows blend patch (required)

Upstream compiles `core/utils/blend/blend.pyx` via `pyximport` at import time,
which needs the MSVC C++ Build Tools. To avoid that toolchain we replaced
`third_party/ditto-talkinghead/core/utils/blend/__init__.py` with a vectorized
NumPy equivalent (same math as upstream's own `PutBackNumpy`):

```python
import numpy as np

def blend_images_cy(mask_warped, frame_warped, frame_rgb, result):
    mask = mask_warped[:, :, None]
    blended = mask * frame_warped + (1.0 - mask) * frame_rgb
    np.clip(blended, 0, 255, out=blended)
    result[...] = blended.astype(np.uint8)
    return result
```

> `third_party/` is gitignored, so re-apply this patch after re-cloning.

## 4. Download the PyTorch checkpoints

Only the PyTorch subset (config pkl + `ditto_pytorch/`) — not the TensorRT engines:

```powershell
& $py -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='digital-avatar/ditto-talkinghead', local_dir='checkpoints', allow_patterns=['ditto_cfg/v0.4_hubert_cfg_pytorch.pkl','ditto_pytorch/aux_models/*','ditto_pytorch/models/*'])"
```

## 5. Run the spike

```powershell
& $py scripts\ditto_spike.py
# -> sessions/ditto_spike.mp4 ; prints "[spike] OK: ..."
```

Override assets to prove any photo + any WAV:

```powershell
& $py scripts\ditto_spike.py --photo path\to\face.png --audio path\to\turn.wav --out sessions\test.mp4
```

## 6. The warm worker (what the app actually runs)

The spike in step 5 starts a fresh `inference.py` process per video and reloads every
model each time (about two minutes for 16 s of speech). The kiosk instead runs
`scripts/ditto_worker.py` once, in the `ditto` env, and keeps the models on the GPU.

- The FastAPI app starts it (`AVATAR_ENABLED=true`), restarts it with a back-off if it
  dies, and stops it on shutdown. You normally never run it by hand.
- Protocol: one JSON object per line over stdin/stdout (`app/avatar/protocol.py`).
  Ditto's own prints, tqdm bars and ffmpeg output go to stderr, which the app writes to
  `sessions/ditto_worker.log` (overwritten at each start). Replies carry only short error
  codes (`no_face`, `bad_audio`, `bad_request`, `render_failed`), never exception text.
- Jobs: `render` (face image + WAV -> MP4 with the voice muxed in) and `check_face`
  (face registration only, used to refuse photos with no face).
- One job at a time. The app queues them by priority: live turns, then face checks,
  then background phrase clips. A running job is never interrupted.

Settings (all optional; see `.env.example`): `DITTO_PYTHON`, `DITTO_REPO`,
`DITTO_DATA_ROOT`, `DITTO_CFG_PKL`, `AVATAR_STARTUP_TIMEOUT_SECONDS`,
`AVATAR_TIMEOUT_SECONDS`.

### Benchmark it

```powershell
& $py scripts\ditto_bench.py
```

Measured on the RTX 4050 Laptop (6 GB): models load in about 10 s and use about 3 GB;
a render peaks at about 4.3 GB. Warm renders take about 4x the audio length (3 s of
speech 17 s, 6 s 25 s, 12 s 47 s); `check_face` about 5 s.

### Tests that use the real worker

```powershell
pytest -m gpu tests/test_ditto_worker_gpu.py
```

### Known limits

- The ONNX aux models still run on the CPU (`CUDAExecutionProvider` is not in the
  `onnxruntime` build used here). That is part of why a render is about 4x real time.
  Installing `onnxruntime-gpu` for the matching CUDA version is a Stage 4 speed-up.
- Watch VRAM on longer clips: about 4.3 GB of the 6 GB is used while rendering.
- Phrase clips: `python scripts/prerender_clips.py` renders the default concierge's
  fixed phrases once (14 clips, about 8 minutes). The server also fills in missing ones
  in the background at startup.