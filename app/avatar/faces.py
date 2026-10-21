"""Traveler faces: session-only photos with a server-side expiry backstop.

A photo is stored as ``faces/<random hex>.png`` (the name carries no session or
identity), tied to a session in memory, and removed when:

* the traveler resets or clears (``delete``),
* the session goes ``backstop_seconds`` without a turn or keep-alive (``sweep``),
* the server shuts down, or at the next start (leftovers from a crash).

The expiry clock is paused while a turn for that face is rendering, so a slow
video never pulls the photo out from under itself; it restarts when the turn
finishes. Photo bytes and file names are never logged.
"""
from __future__ import annotations

import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from app.avatar.models import FaceRef

logger = logging.getLogger(__name__)


_OWN_FILE_RE = re.compile(r"^[0-9a-f]{32}\.png$")  # never touch .gitkeep or other files in faces/
_SWEEP_INTERVAL_SECONDS = 1.0
_PENDING_PAUSE_CAP_SECONDS = 300.0  # a stuck turn may not keep a photo alive forever


@dataclass
class _Face:
    ref: FaceRef
    session_id: str
    expires_at: float
    pending: int = 0
    pending_since: float = 0.0


class FaceStore:
    def __init__(
        self,
        directory: str | Path,
        backstop_seconds: float = 60.0,
        on_expire: Optional[Callable[[str], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._dir = Path(directory)
        self._backstop = float(backstop_seconds)
        self._on_expire = on_expire
        self._clock = clock
        self._by_session: dict[str, _Face] = {}
        self._lock = threading.Lock()
        self._doomed: list[Path] = []          # files that could not be removed yet; retried by sweep
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._dir.mkdir(parents=True, exist_ok=True)
        self.purge_stale_files()

    # --------------------------------------------------------------- files
    def purge_stale_files(self) -> None:
        """Remove photos left by a previous run. Only our own random-named PNGs are touched."""
        for path in self._dir.glob("*.png"):
            if _OWN_FILE_RE.match(path.name):
                self._remove(path)

    def _remove(self, path: Path) -> None:
        """Delete a photo; if the OS refuses (file briefly in use), retry on the next sweep."""
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("could not delete a traveler photo yet; will retry")
            self._doomed.append(path)

    def stage(self, png: bytes) -> FaceRef:
        """Write sanitised PNG bytes to a new random file. Not tied to a session until ``activate``."""
        key = secrets.token_hex(16)
        path = self._dir / f"{key}.png"
        path.write_bytes(png)
        return FaceRef(path=path, key=key)

    def discard(self, ref: FaceRef) -> None:
        self._remove(ref.path)

    # ------------------------------------------------------------ sessions
    def activate(self, session_id: str, ref: FaceRef) -> Optional[str]:
        """Make ``ref`` this session's face. Returns the replaced face's key, if any."""
        replaced: Optional[str] = None
        with self._lock:
            old = self._by_session.get(session_id)
            if old is not None:
                replaced = old.ref.key
                self._remove(old.ref.path)
            self._by_session[session_id] = _Face(ref, session_id, self._clock() + self._backstop)
        self._ensure_sweeper()
        return replaced

    def get(self, session_id: str) -> Optional[FaceRef]:
        with self._lock:
            face = self._by_session.get(session_id)
            return face.ref if face else None

    def touch(self, session_id: str) -> bool:
        """Keep-alive: push the expiry out. False if the session has no custom face."""
        with self._lock:
            face = self._by_session.get(session_id)
            if face is None:
                return False
            face.expires_at = self._clock() + self._backstop
            return True

    def delete(self, session_id: str) -> Optional[str]:
        """Remove the session's photo now. Returns its key (so queued jobs can be cancelled)."""
        with self._lock:
            face = self._by_session.pop(session_id, None)
            if face is None:
                return None
            self._remove(face.ref.path)
            return face.ref.key

    def turn_started(self, face_key: str) -> None:
        with self._lock:
            face = self._find(face_key)
            if face is not None:
                face.pending += 1
                face.pending_since = self._clock()

    def turn_finished(self, face_key: str) -> None:
        """A turn using this face is done; the idle clock restarts from now."""
        with self._lock:
            face = self._find(face_key)
            if face is not None:
                face.pending = max(face.pending - 1, 0)
                face.expires_at = self._clock() + self._backstop

    def _find(self, face_key: str) -> Optional[_Face]:
        for face in self._by_session.values():
            if face.ref.key == face_key:
                return face
        return None

    def has_key(self, face_key: str) -> bool:
        """Is this photo still some session's current face?"""
        with self._lock:
            return self._find(face_key) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_session)

    # -------------------------------------------------------------- expiry
    def sweep(self) -> list[str]:
        """Delete expired photos; returns their keys. Also retries stuck deletions."""
        now = self._clock()
        expired: list[str] = []
        with self._lock:
            for session_id, face in list(self._by_session.items()):
                paused = face.pending > 0 and (now - face.pending_since) < _PENDING_PAUSE_CAP_SECONDS
                if face.expires_at < now and not paused:
                    del self._by_session[session_id]
                    self._remove(face.ref.path)
                    expired.append(face.ref.key)
            retry, self._doomed = self._doomed, []
        for path in retry:
            self._remove(path)
        for key in expired:
            if self._on_expire is not None:
                try:
                    self._on_expire(key)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("face expiry callback failed: %s", type(exc).__name__)
        return expired

    def _ensure_sweeper(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="face-sweeper", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(_SWEEP_INTERVAL_SECONDS):
            self.sweep()

    def close(self) -> None:
        """Stop the sweeper and delete every traveler photo."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3)
        with self._lock:
            faces, self._by_session = list(self._by_session.values()), {}
            for face in faces:
                self._remove(face.ref.path)
            retry, self._doomed = self._doomed, []
        for path in retry:
            self._remove(path)
