"""The avatar service the HTTP layer talks to.

Owns the Ditto worker process and a priority queue. A single dispatcher thread
runs one job at a time (the GPU can only do one), so ordering decisions live in
the queue: live turns first, then face checks, then background clip renders.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from app.avatar import protocol
from app.avatar.models import (
    FaceCheck,
    Job,
    JobOutcome,
    JobStatus,
    Priority,
    QueueFull,
)
from app.avatar.queue import AvatarQueue, notify
from app.avatar.worker_client import (
    DittoWorker,
    WorkerDied,
    WorkerTimeout,
    WorkerUnavailable,
)

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)

_JOB_ID_RE = protocol.JOB_ID_RE
_MAX_BACKOFF_SECONDS = 60.0


class AvatarService:
    """Warm Ditto worker + queue. ``ready`` is True once the models are loaded."""

    def __init__(
        self,
        worker: DittoWorker,
        queue: Optional[AvatarQueue] = None,
        startup_timeout: float = 180.0,
        hard_job_timeout: float = 300.0,
    ) -> None:
        self._worker = worker
        self._queue = queue or AvatarQueue()
        self._startup_timeout = startup_timeout
        self._hard_job_timeout = hard_job_timeout
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._failures = 0
        self._next_start_allowed = 0.0

    # ---------------------------------------------------------- construction
    @classmethod
    def from_settings(cls, settings: "Settings") -> "AvatarService":
        root = Path.cwd()

        def resolve(value: str) -> Path:
            path = Path(value).expanduser()
            return path if path.is_absolute() else (root / path).resolve()

        script = resolve(settings.ditto_worker_script)
        command = [
            str(resolve(settings.ditto_python)),
            str(script),
            "--repo", str(resolve(settings.ditto_repo)),
            "--data-root", str(resolve(settings.ditto_data_root)),
            "--cfg-pkl", str(resolve(settings.ditto_cfg_pkl)),
        ]
        worker = DittoWorker(command, log_path=resolve(settings.sessions_dir) / "ditto_worker.log")
        return cls(
            worker,
            AvatarQueue(settings.avatar_max_queue),
            startup_timeout=settings.avatar_startup_timeout_seconds,
            # A healthy render takes seconds; this only catches a hung process.
            hard_job_timeout=max(settings.avatar_timeout_seconds * 3.0, 120.0),
        )

    # -------------------------------------------------------------- lifecycle
    def start(self, wait_seconds: float = 0.0) -> bool:
        """Start the worker and dispatcher. Optionally wait for the models to load."""
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._dispatch, name="avatar-dispatch", daemon=True)
            self._thread.start()
        self._worker.start()
        if wait_seconds > 0:
            return self._worker.wait_ready(wait_seconds)
        return self.ready

    def stop(self) -> None:
        self._stop.set()
        self._queue.close()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5)
        self._worker.stop()

    @property
    def ready(self) -> bool:
        return self._worker.ready

    @property
    def pending(self) -> int:
        return len(self._queue)

    # -------------------------------------------------------------------- API
    def render(
        self,
        job_id: str,
        source: Path,
        audio: Path,
        output: Path,
        *,
        priority: Priority = Priority.LIVE,
        face_key: Optional[str] = None,
        deadline_s: float = 90.0,
        on_done: Callable[[JobOutcome], None],
    ) -> None:
        """Queue a render. ``on_done`` runs on the dispatcher thread. Raises ``QueueFull``."""
        self._submit(
            Job(
                job_id=self._valid_id(job_id),
                op=protocol.OP_RENDER,
                source=Path(source).resolve(),
                audio=Path(audio).resolve(),
                output=Path(output).resolve(),
                priority=priority,
                deadline=self._queue.now() + deadline_s,
                on_done=on_done,
                face_key=face_key,
            )
        )

    def check_face(self, source: Path, *, timeout_s: float = 30.0) -> FaceCheck:
        """Blocking: is there a usable face in ``source``? Never raises for a bad photo."""
        done = threading.Event()
        holder: list[JobOutcome] = []

        def finished(outcome: JobOutcome) -> None:
            holder.append(outcome)
            done.set()

        job_id = f"check-{time.monotonic_ns():x}"
        try:
            self._submit(
                Job(
                    job_id=job_id,
                    op=protocol.OP_CHECK_FACE,
                    source=Path(source).resolve(),
                    priority=Priority.FACE_CHECK,
                    deadline=self._queue.now() + timeout_s,
                    on_done=finished,
                )
            )
        except QueueFull:
            return FaceCheck.UNAVAILABLE
        if not done.wait(timeout_s) or not holder:
            return FaceCheck.UNAVAILABLE
        outcome = holder[0]
        if outcome.ok:
            return FaceCheck.OK
        if outcome.status is JobStatus.FAILED and outcome.error_code == protocol.NO_FACE:
            return FaceCheck.NO_FACE
        return FaceCheck.UNAVAILABLE

    def cancel_face(self, face_key: str) -> int:
        return self._queue.cancel_face(face_key)

    # ------------------------------------------------------------- internals
    @staticmethod
    def _valid_id(job_id: str) -> str:
        if not _JOB_ID_RE.match(job_id):
            raise ValueError("invalid job id")
        return job_id

    def _submit(self, job: Job) -> None:
        self._queue.put(job)

    def _dispatch(self) -> None:
        while not self._stop.is_set():
            job = self._queue.get(timeout=0.5)
            if job is None:
                continue
            try:
                outcome = self._execute(job)
            except Exception as exc:  # noqa: BLE001 - the dispatcher must survive anything
                logger.error("avatar job crashed: %s", type(exc).__name__)
                outcome = JobOutcome(job.job_id, JobStatus.FAILED, protocol.RENDER_FAILED)
            notify(job, outcome)

    def _ensure_worker(self, job: Job) -> bool:
        """Make sure a ready worker exists, restarting a dead one with backoff."""
        if self._worker.ready:
            return True
        now = time.monotonic()
        if not self._worker.alive:
            if now < self._next_start_allowed:
                return False
            self._worker.start()
        remaining = max(job.deadline - self._queue.now(), 0.0)
        if self._worker.wait_ready(min(self._startup_timeout, remaining)):
            self._failures = 0
            return True
        if not self._worker.alive:  # exited while starting: back off before the next attempt
            self._failures += 1
            self._next_start_allowed = time.monotonic() + min(2.0 ** (self._failures - 1), _MAX_BACKOFF_SECONDS)
            logger.warning("Ditto worker failed to start (attempt %d)", self._failures)
        return False

    def _execute(self, job: Job) -> JobOutcome:
        if not self._ensure_worker(job):
            return JobOutcome(job.job_id, JobStatus.UNAVAILABLE)
        if job.op == protocol.OP_RENDER:
            assert job.audio is not None and job.output is not None
            line = protocol.request_line(
                job.op, job.job_id, str(job.source), str(job.audio), str(job.output)
            )
        else:
            line = protocol.request_line(job.op, job.job_id, str(job.source))
        try:
            result = self._worker.run(line, job.job_id, self._hard_job_timeout)
        except WorkerUnavailable:
            return JobOutcome(job.job_id, JobStatus.UNAVAILABLE)
        except (WorkerDied, WorkerTimeout) as exc:
            logger.warning("Ditto worker lost during a job: %s", type(exc).__name__)
            self._failures += 1
            self._next_start_allowed = time.monotonic() + min(2.0 ** (self._failures - 1), _MAX_BACKOFF_SECONDS)
            return JobOutcome(job.job_id, JobStatus.FAILED, protocol.RENDER_FAILED)
        if result.ok:
            self._failures = 0
            return JobOutcome(job.job_id, JobStatus.OK, seconds=result.seconds, extra=result.extra)
        return JobOutcome(
            job.job_id, JobStatus.FAILED, result.error_code or protocol.RENDER_FAILED, result.seconds
        )
