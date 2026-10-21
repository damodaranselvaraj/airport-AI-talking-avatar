"""Shared types for the avatar subsystem (no heavy imports)."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


DEFAULT_FACE_KEY = "default"  # FaceRef.key of the concierge; traveler photos use a random hex key


class Priority(enum.IntEnum):
    """Lower runs first. A running job is never interrupted."""

    LIVE = 0         # a traveler is waiting for this turn
    FACE_CHECK = 1   # validating an uploaded photo
    BACKGROUND = 2   # pre-rendering fixed phrase clips


class JobStatus(str, enum.Enum):
    OK = "ok"
    FAILED = "failed"            # worker ran the job and reported an error
    EXPIRED = "expired"          # deadline passed while the job was still queued
    CANCELLED = "cancelled"      # removed from the queue (face deleted, shutdown)
    UNAVAILABLE = "unavailable"  # worker not running / could not be started / died


class FaceCheck(str, enum.Enum):
    OK = "ok"
    NO_FACE = "no_face"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class JobOutcome:
    job_id: str
    status: JobStatus
    error_code: Optional[str] = None
    seconds: float = 0.0
    extra: Optional[dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.status is JobStatus.OK


@dataclass
class Job:
    job_id: str
    op: str
    source: Path
    priority: Priority
    deadline: float                     # absolute, on the queue's monotonic clock
    on_done: Callable[[JobOutcome], None]
    audio: Optional[Path] = None
    output: Optional[Path] = None
    face_key: Optional[str] = None      # lets a deleted face cancel its queued jobs
    seq: int = field(default=0, compare=False)


@dataclass(frozen=True)
class FaceRef:
    """A face that can speak: its image file and a stable key for cancelling its queued jobs."""

    path: Path
    key: str


class QueueFull(Exception):
    """The avatar queue is at capacity; the caller should fall back to text and audio."""
