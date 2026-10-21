"""CPU-only tests for the Ditto worker speed-ups: max_size env, registration cache, ROI putback.

No GPU or model files needed. The ROI equivalence tests need OpenCV (present in the ``ditto``
conda env, not in ``.venv``) and are skipped when it is missing.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_worker():
    spec = importlib.util.spec_from_file_location("ditto_worker", ROOT / "scripts" / "ditto_worker.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ditto_worker"] = module
    spec.loader.exec_module(module)
    return module


dw = _load_worker()


# ------------------------------------------------------------------ max_size env
@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 1024), ("768", 768), (" 512 ", 512), ("abc", 1024), ("0", 1024), ("-5", 1024), ("100", 1024), ("", 1024)],
)
def test_max_size_from_env(raw, expected):
    env = {} if raw is None else {"DITTO_MAX_SIZE": raw}
    assert dw._max_size_from_env(env) == expected


# ------------------------------------------------------------------ registration cache
class FakeRegistrar:
    def __init__(self):
        self.calls = 0
        self.extra_attr = "inner"

    def __call__(self, source, max_dim=1920, n_frames=-1, **kwargs):
        self.calls += 1
        if Path(source).name.startswith("bad"):
            raise TypeError("no face")
        return {
            "x_s_info_lst": [{"kp": np.arange(6, dtype=np.float32)}],
            "img_rgb_lst": [np.full((4, 4, 3), self.calls, dtype=np.uint8)],
            "sc": np.zeros(3),
            "call": self.calls,
        }


def _img(tmp_path, name, payload=b"x"):
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


CROP = dict(crop_scale=2.3, crop_vx_ratio=0, crop_vy_ratio=-0.125, crop_flag_do_rot=True)


def test_cache_hit_on_same_file(tmp_path):
    inner = FakeRegistrar()
    cache = dw.CachedRegistrar(inner)
    src = _img(tmp_path, "a.png")
    first = cache(src, max_dim=1024, n_frames=1, **CROP)
    second = cache(src, max_dim=1024, n_frames=-1, **CROP)  # n_frames differs: same key for stills
    assert inner.calls == 1
    assert (cache.hits, cache.misses) == (1, 1)
    assert second["call"] == first["call"]
    np.testing.assert_array_equal(second["img_rgb_lst"][0], first["img_rgb_lst"][0])


def test_cache_returns_independent_copies(tmp_path):
    cache = dw.CachedRegistrar(FakeRegistrar())
    src = _img(tmp_path, "a.png")
    first = cache(src, max_dim=1024, **CROP)
    first["img_rgb_lst"][0][:] = 99
    first["x_s_info_lst"][0]["kp"][:] = -1
    first["x_s_info_lst"].append("junk")
    second = cache(src, max_dim=1024, **CROP)
    assert second["img_rgb_lst"][0].max() == 1
    assert second["x_s_info_lst"][0]["kp"].tolist() == list(range(6))
    assert len(second["x_s_info_lst"]) == 1
    second["img_rgb_lst"][0][:] = 7
    third = cache(src, max_dim=1024, **CROP)
    assert third["img_rgb_lst"][0].max() == 1


def test_cache_miss_after_content_change(tmp_path):
    inner = FakeRegistrar()
    cache = dw.CachedRegistrar(inner)
    src = _img(tmp_path, "a.png", b"one")
    cache(src, max_dim=1024, **CROP)
    Path(src).write_bytes(b"different length")
    cache(src, max_dim=1024, **CROP)
    assert inner.calls == 2


def test_cache_miss_after_mtime_change_only(tmp_path):
    inner = FakeRegistrar()
    cache = dw.CachedRegistrar(inner)
    src = _img(tmp_path, "a.png", b"same")
    cache(src, max_dim=1024, **CROP)
    st = os.stat(src)
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    cache(src, max_dim=1024, **CROP)
    assert inner.calls == 2


def test_cache_miss_on_different_options(tmp_path):
    inner = FakeRegistrar()
    cache = dw.CachedRegistrar(inner)
    src = _img(tmp_path, "a.png")
    cache(src, max_dim=1024, **CROP)
    cache(src, max_dim=768, **CROP)
    cache(src, max_dim=1024, **{**CROP, "crop_scale": 2.0})
    assert inner.calls == 3


def test_cache_lru_bound_and_order(tmp_path):
    inner = FakeRegistrar()
    cache = dw.CachedRegistrar(inner, maxsize=3)
    files = [_img(tmp_path, f"f{i}.png", b"x" * (i + 1)) for i in range(5)]
    for f in files[:3]:
        cache(f, max_dim=1024, **CROP)
    cache(files[0], max_dim=1024, **CROP)  # touch f0 -> f1 is now least recently used
    cache(files[3], max_dim=1024, **CROP)  # evicts f1
    assert len(cache) == 3
    calls = inner.calls
    cache(files[0], max_dim=1024, **CROP)
    cache(files[2], max_dim=1024, **CROP)
    cache(files[3], max_dim=1024, **CROP)
    assert inner.calls == calls  # f0, f2, f3 still cached
    cache(files[1], max_dim=1024, **CROP)
    assert inner.calls == calls + 1  # f1 was evicted
    assert len(cache) == 3


def test_cache_does_not_store_failures(tmp_path):
    inner = FakeRegistrar()
    cache = dw.CachedRegistrar(inner)
    src = _img(tmp_path, "bad.png")
    for _ in range(2):
        with pytest.raises(TypeError):
            cache(src, max_dim=1024, **CROP)
    assert inner.calls == 2
    assert len(cache) == 0


def test_cache_bypasses_non_still_images(tmp_path):
    inner = FakeRegistrar()
    cache = dw.CachedRegistrar(inner)
    src = _img(tmp_path, "clip.mp4")
    cache(src, max_dim=1024, n_frames=10, **CROP)
    cache(src, max_dim=1024, n_frames=10, **CROP)
    assert inner.calls == 2
    assert len(cache) == 0


def test_cache_missing_file_reaches_inner(tmp_path):
    inner = FakeRegistrar()
    cache = dw.CachedRegistrar(inner)
    cache(str(tmp_path / "missing.png"), max_dim=1024, **CROP)
    assert inner.calls == 1
    assert len(cache) == 0


def test_cache_drops_entries_for_deleted_files(tmp_path):
    inner = FakeRegistrar()
    cache = dw.CachedRegistrar(inner)
    gone = _img(tmp_path, "gone.png", b"a")
    kept = _img(tmp_path, "kept.png", b"bb")
    cache(gone, max_dim=1024, **CROP)
    cache(kept, max_dim=1024, **CROP)
    assert len(cache) == 2
    os.remove(gone)
    cache(kept, max_dim=1024, **CROP)  # any later call prunes the deleted photo's entry
    assert len(cache) == 1
    assert cache.hits == 1
    # nothing keyed on the deleted file remains reachable
    assert all(key[0] != os.path.realpath(gone) for key in cache._entries)


def test_cache_keeps_one_entry_per_path(tmp_path):
    inner = FakeRegistrar()
    cache = dw.CachedRegistrar(inner)
    src = _img(tmp_path, "a.png", b"one")
    cache(src, max_dim=1024, **CROP)
    Path(src).write_bytes(b"rewritten")
    cache(src, max_dim=1024, **CROP)
    assert len(cache) == 1  # the superseded version is dropped, not left resident


def test_cache_delegates_attributes():
    assert dw.CachedRegistrar(FakeRegistrar()).extra_attr == "inner"


# ------------------------------------------------------------------ ROI geometry
def _affine(scale=1.0, angle_deg=0.0, tx=0.0, ty=0.0):
    a = math.radians(angle_deg)
    return np.array(
        [[scale * math.cos(a), -scale * math.sin(a), tx], [scale * math.sin(a), scale * math.cos(a), ty], [0, 0, 1]],
        dtype=np.float32,
    )


def test_roi_inside_is_smaller_than_frame_and_contains_crop():
    x0, y0, x1, y1 = dw.compute_roi(_affine(0.6, 0, 100, 80), 720, 1000)
    assert (x0, y0) == (97, 77)  # floor(100 - 0.6) - margin, floor(80 - 0.6) - margin
    assert x1 < 1000 and y1 < 720
    assert x1 >= 100 + 0.6 * 512 and y1 >= 80 + 0.6 * 512


def test_roi_clamped_past_right_bottom():
    assert dw.compute_roi(_affine(1.0, 0, 300, 400), 480, 640) == (297, 397, 640, 480)


def test_roi_clamped_past_left_top():
    x0, y0, x1, y1 = dw.compute_roi(_affine(1.0, 0, -200, -100), 480, 640)
    assert (x0, y0) == (0, 0)
    assert x1 < 640 and y1 < 480


def test_roi_none_when_outside_or_nan():
    assert dw.compute_roi(_affine(1.0, 0, 2000, 0), 480, 640) is None
    assert dw.compute_roi(_affine(1.0, 0, -2000, 0), 480, 640) is None
    nan = _affine()
    nan[0, 2] = np.nan
    assert dw.compute_roi(nan, 480, 640) is None


# ------------------------------------------------------------------ ROI putback equivalence
def _reference_putback(mask2d, frame_rgb, render_image, M, cv2):
    """Ditto's PutBack semantics: full-frame warp, float blend, truncating uint8 cast."""
    h, w = frame_rgb.shape[:2]
    mask_warped = cv2.warpAffine(mask2d, M[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR).clip(0, 1)
    frame_warped = cv2.warpAffine(render_image, M[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR)
    m = mask_warped[:, :, None]
    blended = m * frame_warped + (1.0 - m) * frame_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


def _feathered_mask(cv2, size=512):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.sqrt(((xx - size / 2) / (size * 0.45)) ** 2 + ((yy - size / 2) / (size * 0.45)) ** 2)
    mask = np.clip(1.5 - 1.5 * r, 0, 1).astype(np.float32)
    return cv2.GaussianBlur(mask, (0, 0), 12).astype(np.float32)


def _noise(cv2, rng, h, w):
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 3)


CASES = {
    "inside_small": ((480, 640), _affine(0.6, 0, 150, 90)),
    "inside_rotated": ((600, 640), _affine(0.8, 15, 120, 20)),
    "past_right_bottom": ((480, 640), _affine(1.0, 0, 300, 250)),
    "past_left_top": ((480, 640), _affine(1.0, 0, -200, -150)),
    "past_all_edges": ((480, 640), _affine(2.0, 0, -300, -300)),
    "non_square_portrait": ((720, 480), _affine(0.7, -10, 40, 200)),
    "identity_full_frame": ((512, 512), _affine()),
    "fully_outside": ((480, 640), _affine(1.0, 0, 5000, 5000)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_roi_putback_matches_original(name):
    cv2 = pytest.importorskip("cv2")
    (h, w), M = CASES[name]
    rng = np.random.default_rng(0)
    mask = _feathered_mask(cv2)
    frame = _noise(cv2, rng, h, w)
    render = _noise(cv2, rng, 512, 512)
    frame_before = frame.copy()

    roi_pb = dw.RoiPutBack(mask)
    out = roi_pb(frame, render, M)
    ref = _reference_putback(mask, frame, render, M, cv2)

    assert out.dtype == np.uint8 and out.shape == frame.shape
    assert np.abs(out.astype(int) - ref.astype(int)).max() <= 1
    assert not np.shares_memory(out, frame)
    np.testing.assert_array_equal(frame, frame_before)

    roi = dw.compute_roi(M, h, w)
    outside = np.ones((h, w), dtype=bool)
    if roi is not None:
        x0, y0, x1, y1 = roi
        outside[y0:y1, x0:x1] = False
    np.testing.assert_array_equal(out[outside], frame[outside])
    if name == "fully_outside":
        np.testing.assert_array_equal(out, frame)


def test_roi_putback_nan_matrix_falls_back_to_full_frame():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    frame = _noise(cv2, rng, 200, 300)
    M = _affine()
    M[0, 2] = np.nan
    out = dw.RoiPutBack(_feathered_mask(cv2))(frame, _noise(cv2, rng, 512, 512), M)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_roi_putback_retries_full_frame_when_roi_blend_fails():
    pb = dw.RoiPutBack(np.ones((512, 512), dtype=np.float32))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    seen = []

    def fake_blend(frame_rgb, render_image, M, roi):
        seen.append(roi)
        if len(seen) == 1:
            raise RuntimeError("boom")
        return "full-frame result"

    pb._blend = fake_blend
    assert pb(frame, None, _affine(1.0, 0, 100, 100)) == "full-frame result"
    assert seen[0] != (0, 0, 640, 480)
    assert seen[1] == (0, 0, 640, 480)


@pytest.mark.parametrize("name", ["inside_small", "past_right_bottom"])
def test_roi_putback_matches_ditto_putback(name):
    cv2 = pytest.importorskip("cv2")
    repo = ROOT / "third_party" / "ditto-talkinghead"
    if not repo.exists():
        pytest.skip("third_party/ditto-talkinghead not present")
    sys.path.insert(0, str(repo))
    try:
        from core.atomic_components.putback import PutBack  # type: ignore
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot import Ditto PutBack: {exc}")
    finally:
        sys.path.remove(str(repo))
    (h, w), M = CASES[name]
    rng = np.random.default_rng(2)
    frame = _noise(cv2, rng, h, w)
    render = _noise(cv2, rng, 512, 512)
    original = PutBack()
    ours = dw.RoiPutBack(original.mask_ori_float)
    ref = original(frame, render, M).copy()
    assert np.abs(ours(frame, render, M).astype(int) - ref.astype(int)).max() <= 1
