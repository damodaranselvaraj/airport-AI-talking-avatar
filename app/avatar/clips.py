"""Pre-rendered clips for the fixed spoken phrases (greeting, refusals, ...).

Fixed phrases come from ``config/responses.json`` and are spoken verbatim, so a
turn whose text equals one of them can play a cached video at once instead of
waiting for Ditto. Clips are keyed by face and by a hash of (language, voice,
text): editing a phrase or changing the voice simply misses the cache and the
clip is rendered again.

Only the default concierge has clips. A traveler's photo is only ever animated for a
grounded answer, live, so nothing of it is pre-rendered or cached here.

Layout under ``cache/clips/``::

    audio/<phrase hash>.wav                 Kokoro audio
    default-<image hash>/<phrase hash>.mp4  the default concierge (survives restarts)

Changing the concierge image, a phrase or the voice only re-renders what changed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
from functools import partial
from pathlib import Path
from typing import Optional, Protocol

from app.avatar.models import DEFAULT_FACE_KEY, FaceRef, JobOutcome, Priority, QueueFull

logger = logging.getLogger(__name__)

_CLIP_RE = re.compile(r"^[0-9a-f]{16}\.mp4$")
_WAV_RE = re.compile(r"^[0-9a-f]{16}\.wav$")
# Most useful first: what a traveler hears most often is rendered earliest.
PRIORITY_ORDER = ["greeting", "fallback", "refusal.flight_status", "refusal.live_gate", "thanks", "goodbye"]


def load_phrases(responses_path: str | Path, language: str = "en") -> dict[str, str]:
    """Spoken fixed phrases in priority order: ``{"greeting": "...", "refusal.visa": "...", ...}``.

    Guardrail messages (``guardrail.EMPTY_INPUT`` ...) are spoken too. Two names may share the same
    text; the cache keys clips by text, so ``ClipCache`` renders each distinct text once.
    ``error`` is excluded on purpose: it is text only (``spoken=False``).
    """
    data = json.loads(Path(responses_path).read_text(encoding="utf-8"))
    block = data["languages"][language]
    flat: dict[str, str] = {}
    for name in ("greeting", "thanks", "goodbye", "fallback"):
        if isinstance(block.get(name), str) and block[name].strip():
            flat[name] = block[name].strip()
    for name, text in (block.get("refusal") or {}).items():
        if isinstance(text, str) and text.strip():
            flat[f"refusal.{name}"] = text.strip()
    for name, text in (block.get("guardrail") or {}).items():
        if isinstance(text, str) and text.strip():
            flat[f"guardrail.{name}"] = text.strip()
    ordered = {k: flat[k] for k in PRIORITY_ORDER if k in flat}
    ordered.update({k: v for k, v in flat.items() if k not in ordered})
    return ordered


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


class ClipCache:
    def __init__(
        self,
        root: str | Path,
        phrases: dict[str, str],
        voice: str,
        language: str,
        default_digest: str,
    ) -> None:
        self._root = Path(root)
        self._phrases = dict(phrases)
        self._voice = voice
        self._lang = language
        self._default_dir = f"default-{default_digest}"
        self._by_hash = {self.phrase_hash(text): text for text in self._phrases.values()}
        # One entry per distinct text (the first name wins), in priority order.
        seen: set[str] = set()
        self._unique: list[tuple[str, str]] = []
        for name, text in self._phrases.items():
            digest = self.phrase_hash(text)
            if digest not in seen:
                seen.add(digest)
                self._unique.append((name, text))
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / "audio").mkdir(exist_ok=True)
        self.purge_stale()

    # ----------------------------------------------------------------- keys
    def phrase_hash(self, text: str) -> str:
        return hashlib.sha256(f"{self._lang}|{self._voice}|{text}".encode("utf-8")).hexdigest()[:16]

    def is_phrase(self, text: str) -> bool:
        return self.phrase_hash(text) in self._by_hash

    def ordered_phrases(self) -> list[tuple[str, str]]:
        """``(name, text)`` in priority order, one entry per distinct text."""
        return list(self._unique)

    def missing(self, face: FaceRef) -> list[str]:
        """Names of the distinct fixed phrases that have no finished clip for this face yet."""
        return [name for name, text in self._unique if self.lookup(face, text) is None]

    def total(self) -> int:
        """How many distinct clips make up the full set."""
        return len(self._unique)

    # ---------------------------------------------------------------- paths
    def face_dir(self, face: FaceRef) -> Path:
        if face.key != DEFAULT_FACE_KEY:
            raise ValueError("clips exist only for the default concierge")
        return self._root / self._default_dir

    def clip_path(self, face: FaceRef, text: str) -> Path:
        return self.face_dir(face) / f"{self.phrase_hash(text)}.mp4"

    def part_path(self, face: FaceRef, text: str) -> Path:
        return self.face_dir(face) / f"{self.phrase_hash(text)}.part.mp4"

    def wav_path(self, text: str) -> Path:
        return self._root / "audio" / f"{self.phrase_hash(text)}.wav"

    def lookup(self, face: FaceRef, text: str) -> Optional[Path]:
        """The finished clip for the concierge and this exact text, or ``None`` (always ``None`` for a traveler)."""
        if face.key != DEFAULT_FACE_KEY or not self.is_phrase(text):
            return None
        path = self.clip_path(face, text)
        return path if path.is_file() else None

    def cached_wav(self, text: str) -> Optional[Path]:
        path = self.wav_path(text)
        return path if self.is_phrase(text) and path.is_file() else None

    # ------------------------------------------------------------- cleaning
    def purge_stale(self) -> None:
        """Drop old default-face dirs, half-written files and stale phrases.

        Also removes a leftover ``faces/`` directory: earlier versions pre-rendered clips for
        traveler photos, and none of that must remain on disk.
        """
        shutil.rmtree(self._root / "faces", ignore_errors=True)
        for entry in self._root.iterdir():
            if entry.is_dir() and entry.name.startswith("default-") and entry.name != self._default_dir:
                shutil.rmtree(entry, ignore_errors=True)
        keep = set(self._by_hash)
        current = self._root / self._default_dir
        if current.is_dir():
            for f in current.iterdir():
                if not (_CLIP_RE.match(f.name) and f.stem in keep):
                    f.unlink(missing_ok=True)
        for f in (self._root / "audio").iterdir():
            if not (_WAV_RE.match(f.name) and f.stem in keep):
                f.unlink(missing_ok=True)


class _Synthesizer(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class _Avatar(Protocol):
    def render(self, job_id, source, audio, output, *, priority, face_key, deadline_s, on_done) -> None: ...


class ClipRenderer:
    """Queues background renders for the default concierge's missing clips, one phrase at a time."""

    def __init__(
        self,
        cache: ClipCache,
        avatar: _Avatar,
        synthesizer: _Synthesizer,
        deadline_s: float = 3600.0,
    ) -> None:
        self._cache = cache
        self._avatar = avatar
        self._synth = synthesizer
        self._deadline = deadline_s
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._outstanding = 0
        self._scheduling: set[str] = set()

    # --------------------------------------------------------------- public
    def schedule(self, face: FaceRef) -> bool:
        """Start queueing the concierge's missing clips in the background. False if already doing so."""
        if face.key != DEFAULT_FACE_KEY:
            raise ValueError("only the default concierge has pre-rendered clips")
        with self._lock:
            if face.key in self._scheduling:
                return False
            self._scheduling.add(face.key)
            self._outstanding += 1  # held by the scheduler thread until it has queued everything
        thread = threading.Thread(target=self._run, args=(face,), name="clip-schedule", daemon=True)
        thread.start()
        return True

    @property
    def busy(self) -> bool:
        """True while clips are being queued or rendered (the kiosk shows "warming up")."""
        with self._lock:
            return self._outstanding > 0

    def wait_idle(self, timeout: float) -> bool:
        """Block until nothing is being scheduled or rendered. Used by tests and the pre-render script."""
        with self._idle:
            return self._idle.wait_for(lambda: self._outstanding == 0, timeout)

    # ------------------------------------------------------------- internals
    def _bump(self, delta: int) -> None:
        with self._idle:
            self._outstanding += delta
            if self._outstanding <= 0:
                self._idle.notify_all()

    def _ensure_wav(self, text: str) -> Path:
        path = self._cache.wav_path(text)
        if path.is_file():
            return path
        data = self._synth.synthesize(text)
        part = path.with_suffix(".part")
        part.write_bytes(data)
        os.replace(part, path)
        return path

    def _run(self, face: FaceRef) -> None:
        try:
            for _name, text in self._cache.ordered_phrases():
                if self._cache.lookup(face, text) is not None:
                    continue
                try:
                    wav = self._ensure_wav(text)
                    part = self._cache.part_path(face, text)
                    part.parent.mkdir(parents=True, exist_ok=True)
                except Exception as exc:  # noqa: BLE001 - skip this phrase, keep going
                    logger.warning("clip audio not prepared: %s", type(exc).__name__)
                    continue
                self._bump(1)
                try:
                    self._avatar.render(
                        f"clip-{face.key[:8]}-{self._cache.phrase_hash(text)}",
                        face.path,
                        wav,
                        part,
                        priority=Priority.BACKGROUND,
                        face_key=face.key,
                        deadline_s=self._deadline,
                        on_done=partial(self._done, face, text),
                    )
                except (QueueFull, RuntimeError, ValueError) as exc:
                    self._bump(-1)
                    logger.info("clip render not queued: %s", type(exc).__name__)
                    if isinstance(exc, (QueueFull, RuntimeError)):
                        break
        finally:
            with self._lock:
                self._scheduling.discard(face.key)
            self._bump(-1)

    def _done(self, face: FaceRef, text: str, outcome: JobOutcome) -> None:
        try:
            part, final = self._cache.part_path(face, text), self._cache.clip_path(face, text)
            if outcome.ok and part.is_file():
                os.replace(part, final)
            else:
                part.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("clip finalise failed: %s", type(exc).__name__)
        finally:
            self._bump(-1)
