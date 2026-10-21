"""Priority job queue with deadlines and cancel-by-face.

Ordering is (priority, arrival). A job whose deadline has passed before it
reaches the front is reported ``EXPIRED`` and skipped, so a stale background
render never delays a traveler. Callbacks always run outside the queue lock.
"""
from __future__ import annotations

import heapq
import itertools
import logging
import threading
import time
from typing import Callable, Optional

from app.avatar.models import Job, JobOutcome, JobStatus, QueueFull

logger = logging.getLogger(__name__)


def notify(job: Job, outcome: JobOutcome) -> None:
    """Run a job's callback; a failing callback must never break the dispatcher."""
    try:
        job.on_done(outcome)
    except Exception as exc:  # noqa: BLE001
        logger.warning("avatar job callback failed: %s", type(exc).__name__)


class AvatarQueue:
    def __init__(self, max_pending: int = 64, clock: Callable[[], float] = time.monotonic) -> None:
        self._max = max_pending
        self._clock = clock
        self._heap: list[tuple[int, int, Job]] = []
        self._seq = itertools.count()
        self._cond = threading.Condition()
        self._closed = False

    def now(self) -> float:
        return self._clock()

    def __len__(self) -> int:
        with self._cond:
            return len(self._heap)

    def put(self, job: Job) -> None:
        """Queue a job. Raises ``QueueFull`` (at capacity) or ``RuntimeError`` (closed)."""
        with self._cond:
            if self._closed:
                raise RuntimeError("avatar queue is closed")
            if len(self._heap) >= self._max:
                raise QueueFull()
            job.seq = next(self._seq)
            heapq.heappush(self._heap, (int(job.priority), job.seq, job))
            self._cond.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[Job]:
        """Next runnable job, or ``None`` on timeout / when closed. Expired jobs are skipped."""
        end = None if timeout is None else self._clock() + timeout
        while True:
            expired: list[Job] = []
            job: Optional[Job] = None
            with self._cond:
                while not self._heap and not self._closed:
                    remaining = None if end is None else end - self._clock()
                    if remaining is not None and remaining <= 0:
                        return None
                    self._cond.wait(remaining)
                if self._closed and not self._heap:
                    return None
                now = self._clock()
                while self._heap:
                    _, _, candidate = heapq.heappop(self._heap)
                    if candidate.deadline < now:
                        expired.append(candidate)
                        continue
                    job = candidate
                    break
            for stale in expired:
                notify(stale, JobOutcome(stale.job_id, JobStatus.EXPIRED))
            if job is not None:
                return job
            if end is not None and self._clock() >= end:
                return None

    def cancel_face(self, face_key: str) -> int:
        """Remove every queued job for ``face_key``. Returns how many were cancelled."""
        with self._cond:
            doomed = [entry for entry in self._heap if entry[2].face_key == face_key]
            if doomed:
                self._heap = [entry for entry in self._heap if entry[2].face_key != face_key]
                heapq.heapify(self._heap)
        for _, _, job in doomed:
            notify(job, JobOutcome(job.job_id, JobStatus.CANCELLED))
        return len(doomed)

    def close(self) -> int:
        """Stop accepting work and cancel everything still queued."""
        with self._cond:
            self._closed = True
            doomed = [entry[2] for entry in self._heap]
            self._heap = []
            self._cond.notify_all()
        for job in doomed:
            notify(job, JobOutcome(job.job_id, JobStatus.CANCELLED))
        return len(doomed)
