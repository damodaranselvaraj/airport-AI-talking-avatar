"""Pre-render the default concierge's fixed-phrase videos (greeting, refusals, ...).

Run this once with the app's ``.venv`` (it starts the Ditto worker itself) so
the first traveler already hears the greeting instantly::

    python scripts/prerender_clips.py            # render whatever is missing
    python scripts/prerender_clips.py --force    # re-render everything

The set is every spoken fixed phrase in ``config/responses.json``: greeting, thanks,
goodbye, fallback, refusals and guardrail messages (identical wording is rendered once).

Clips land in ``cache/clips/`` (gitignored) and are keyed by the concierge image
and by the exact phrase text and voice, so editing ``config/responses.json`` or
swapping the concierge image only re-renders what changed. The server also
renders missing clips at startup (``CLIPS_PRERENDER_ON_START``, on by default) and the
kiosk shows "Warming up" until they exist; this script just lets you do it ahead of
time and see the timings. Stop the server first: both would use the one GPU.

Needs the same ``.env`` as the server (``AVATAR_ENABLED`` is not required here).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-render the default concierge's phrase clips")
    parser.add_argument("--force", action="store_true", help="delete existing default clips first")
    parser.add_argument("--timeout", type=float, default=3600.0, help="give up after this many seconds")
    args = parser.parse_args()

    from app.avatar.clips import ClipCache, ClipRenderer, file_digest, load_phrases
    from app.avatar.models import DEFAULT_FACE_KEY, FaceRef
    from app.avatar.service import AvatarService
    from app.services import find_concierge_image
    from app.text_to_speech.kokoro_service import KokoroSynthesizer
    from config.settings import get_settings

    settings = get_settings()
    face_path = find_concierge_image(settings)
    face = FaceRef(path=face_path, key=DEFAULT_FACE_KEY)
    phrases = load_phrases(settings.responses_path, settings.default_language)
    cache = ClipCache(
        settings.clips_dir,
        phrases,
        voice=settings.kokoro_voice,
        language=settings.default_language,
        default_digest=file_digest(face_path),
    )
    if args.force:
        shutil.rmtree(cache.face_dir(face), ignore_errors=True)

    total = cache.total()
    missing = cache.missing(face)
    print(
        f"[clips] concierge image: {face_path.name}; {total} distinct clips "
        f"({len(phrases)} phrases), {len(missing)} to render"
    )
    if not missing:
        print("[clips] nothing to do")
        return 0

    synthesizer = KokoroSynthesizer(settings.kokoro_voice, settings.kokoro_lang)
    print("[clips] loading Kokoro...")
    synthesizer.warm_up()
    avatar = AvatarService.from_settings(settings)
    print("[clips] starting the Ditto worker...")
    if not avatar.start(wait_seconds=settings.avatar_startup_timeout_seconds):
        print("[clips] the Ditto worker did not become ready; see sessions/ditto_worker.log", file=sys.stderr)
        avatar.stop()
        return 2

    started = time.perf_counter()
    try:
        renderer = ClipRenderer(cache, avatar, synthesizer)
        renderer.schedule(face)
        finished, shown = False, -1
        while not finished:
            remaining = args.timeout - (time.perf_counter() - started)
            if remaining <= 0:
                break
            finished = renderer.wait_idle(min(5.0, remaining))
            done = total - len(cache.missing(face))
            if done != shown:
                shown = done
                print(f"[clips] {done} of {total} ready ({time.perf_counter() - started:.0f}s)")
    finally:
        avatar.stop()
    elapsed = time.perf_counter() - started

    still_missing = cache.missing(face)
    print(f"[clips] rendered {len(missing) - len(still_missing)}/{len(missing)} in {elapsed:.0f}s")
    if still_missing or not finished:
        print(f"[clips] missing: {', '.join(still_missing) or 'timed out'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
