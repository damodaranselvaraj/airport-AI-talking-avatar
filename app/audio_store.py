"""Per-turn generated audio (TTS output) with a TTL.

Only synthesized speech is stored here, never traveler audio. Turn IDs are
server-generated random hex and are looked up in memory; they are never
interpreted as filesystem paths. Files live in ``sessions/`` (gitignored) and
are deleted after ``ttl_seconds`` and at shutdown.
"""
from __future__ import annotations

import enum
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

TURN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_OWN_FILE_RE = re.compile(r"^[0-9a-f]{32}\.wav$")  # never touch other files in sessions/


class AudioState(str, enum.Enum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


@dataclass
class _Entry:
    state: AudioState
    expires_at: float
    path: Optional[Path] = None


class AudioStore:
    def __init__(self, directory: str | Path, ttl_seconds: int = 600) -> None:
        self._dir = Path(directory)
        self._ttl = ttl_seconds
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._dir.mkdir(parents=True, exist_ok=True)
        self.purge_stale_files()

    def purge_stale_files(self) -> None:
        """Remove leftover turn WAVs from a previous run (other files are left alone)."""
        for path in self._dir.glob("*.wav"):
            if _OWN_FILE_RE.match(path.name):
                path.unlink(missing_ok=True)

    @staticmethod
    def new_turn_id() -> str:
        return uuid.uuid4().hex

    def begin(self, turn_id: str) -> None:
        self.sweep()
        with self._lock:
            self._entries[turn_id] = _Entry(AudioState.PENDING, time.monotonic() + self._ttl)

    def put(self, turn_id: str, wav: bytes) -> None:
        path = self._dir / f"{turn_id}.wav"
        path.write_bytes(wav)
        with self._lock:
            entry = self._entries.get(turn_id)
            if entry is None:  # swept while rendering; drop the file
                path.unlink(missing_ok=True)
                return
            entry.state, entry.path = AudioState.READY, path

    def fail(self, turn_id: str) -> None:
        with self._lock:
            entry = self._entries.get(turn_id)
            if entry:
                entry.state = AudioState.FAILED

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
                if entry.path:
                    entry.path.unlink(missing_ok=True)

    def close(self) -> None:
        with self._lock:
            for entry in self._entries.values():
                if entry.path:
                    entry.path.unlink(missing_ok=True)
            self._entries.clear()
