"""Traveler photo sanitising: decode, auto-rotate, strip metadata, re-encode to PNG.

Whatever the upload claims to be, only pixels survive: EXIF (including GPS and
camera data), ICC profiles, comments and any trailing bytes are dropped because
the output is a freshly encoded PNG. The result is capped in size so Ditto's
face registration stays fast.
"""
from __future__ import annotations

import io

from PIL import Image, ImageOps

MAX_PIXELS_IN = 25_000_000   # reject decompression bombs before decoding pixels
MIN_SIDE = 128               # smaller than this cannot hold a usable face
MAX_SIDE_OUT = 1280
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}


class BadImage(ValueError):
    """The upload is not a usable photo. The message is safe to show or log."""


def sanitize_photo(data: bytes) -> bytes:
    """Return PNG bytes for ``data`` or raise ``BadImage``."""
    if not data:
        raise BadImage("empty image")
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ALLOWED_FORMATS:
                raise BadImage("unsupported image type")
            width, height = img.size
            if width * height > MAX_PIXELS_IN:
                raise BadImage("image is too large")
            if min(width, height) < MIN_SIDE:
                raise BadImage("image is too small")
            img.load()
            oriented = ImageOps.exif_transpose(img)  # honour phone rotation before metadata is dropped
            rgb = oriented.convert("RGB")
    except BadImage:
        raise
    except Exception as exc:  # noqa: BLE001 - Pillow raises many distinct types for bad input
        raise BadImage("not a readable image") from exc

    if max(rgb.size) > MAX_SIDE_OUT:
        rgb.thumbnail((MAX_SIDE_OUT, MAX_SIDE_OUT), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    rgb.save(out, format="PNG")  # no exif=, no icc_profile=: nothing but pixels is written
    return out.getvalue()
