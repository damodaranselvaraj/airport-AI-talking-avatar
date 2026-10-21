"""Per-turn talking-face videos (Ditto output) with a TTL.

Mirrors ``AudioStore``: turn IDs are server-generated random hex looked up in
memory, never used as paths; files live in ``sessions/`` (gitignored) and are
deleted after ``ttl_seconds`` and at shutdown. A video contains the Kokoro
audio track, which is derived from the approved answer text, never from
traveler audio.

Entries can also point at a *shared* file (a cached phrase clip). Those are
served but never deleted by this store; the clip cache owns them.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.audio_store import TURN_ID_RE, AudioState

_OWN_FILE_RE = re.compile(r"^[0-9a-f]{32}\.mp4$")  # never touch other files in sessions/


@dataclass
class _Entry:
    state: AudioState
    expires_at: float
    path: Optional[Path] = None
    shared: bool = False


class VideoStore:
    def __init__(self, directory: str | Path, ttl_seconds: int = 600) -> None:
        self._dir = Path(directory)
        self._ttl = ttl_seconds
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._dir.mkdir(parents=True, exist_ok=True)
        self.purge_stale_files()

    def purge_stale_files(self) -> None:
        """Remove leftover turn MP4s from a previous run (other files are left alone)."""
        for path in self._dir.glob("*.mp4"):
            if _OWN_FILE_RE.match(path.name):
                path.unlink(missing_ok=True)

    def path_for(self, turn_id: str) -> Path:
        if not TURN_ID_RE.match(turn_id):
            raise ValueError("invalid turn id")
        return self._dir / f"{turn_id}.mp4"

    def begin(self, turn_id: str) -> None:
        self.sweep()
        with self._lock:
            self._entries[turn_id] = _Entry(AudioState.PENDING, time.monotonic() + self._ttl)

    def ready_from_shared(self, turn_id: str, path: Path) -> None:
        """Serve an existing shared file (cached clip) for this turn without owning it."""
        self.sweep()
        with self._lock:
            self._entries[turn_id] = _Entry(AudioState.READY, time.monotonic() + self._ttl, Path(path), shared=True)

    def finish(self, turn_id: str, ok: bool) -> None:
        """Mark the render done. The file at ``path_for(turn_id)`` is the result when ``ok``."""
        path = self.path_for(turn_id)
        with self._lock:
            entry = self._entries.get(turn_id)
            if entry is None:  # swept while rendering: drop whatever was written
                path.unlink(missing_ok=True)
                return
            if ok and path.is_file():
                entry.state, entry.path = AudioState.READY, path
                return
            entry.state = AudioState.FAILED
        path.unlink(missing_ok=True)

    def fail(self, turn_id: str) -> None:
        self.finish(turn_id, ok=False)

    def get(self, turn_id: str) -> tuple[Optional[AudioState], Optional[Path]]:
        if not TURN_ID_RE.match(turn_id):
            return None, None
        with self._lock:
            entry = self._entries.get(turn_id)
            if entry is None or entry.expires_at < time.monotonic():
                return None, None
            return entry.state, entry.path

    def sweep(self) -> None:
        now = time.monotonic()
        with self._lock:
            expired = [tid for tid, e in self._entries.items() if e.expires_at < now]
            for tid in expired:
                entry = self._entries.pop(tid)
                if entry.path and not entry.shared:
                    entry.path.unlink(missing_ok=True)

    def close(self) -> None:
        with self._lock:
            for entry in self._entries.values():
                if entry.path and not entry.shared:
                    entry.path.unlink(missing_ok=True)
            self._entries.clear()
