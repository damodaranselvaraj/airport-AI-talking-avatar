"""Spawns and talks to the Ditto worker process (see ``scripts/ditto_worker.py``).

One worker, one job at a time. The worker lives in the isolated ``ditto`` conda
env; this module only needs the standard library. If the process dies or hangs
the caller gets an exception and the next ``start()`` spawns a fresh one.
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

from app.avatar import protocol

logger = logging.getLogger(__name__)


class WorkerUnavailable(Exception):
    """No ready worker process to send the job to."""


class WorkerDied(Exception):
    """The process exited while a job was running."""


class WorkerTimeout(Exception):
    """The job ran past its hard timeout; the process was killed."""


class _Process:
    """One spawned worker plus the thread that reads its replies."""

    def __init__(self, cmd: Sequence[str], log_path: Optional[Path]) -> None:
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self._log = None
        stderr = subprocess.DEVNULL
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(log_path, "w", encoding="utf-8", errors="replace")
            stderr = self._log
        self.proc = subprocess.Popen(
            list(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        self.ready = threading.Event()
        self.load_seconds = 0.0
        self.replies: "queue.Queue[Optional[protocol.Result]]" = queue.Queue()
        self._reader = threading.Thread(target=self._read, name="ditto-worker-reader", daemon=True)
        self._reader.start()

    def _read(self) -> None:
        stdout = self.proc.stdout
        assert stdout is not None
        try:
            for line in stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    kind, value = protocol.parse_reply(line)
                except protocol.ProtocolError:
                    continue  # stray output on the channel; ignore it
                if kind == "ready":
                    self.load_seconds = value
                    self.ready.set()
                else:
                    self.replies.put(value)
        finally:
            self.replies.put(None)  # EOF: the process is gone

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def send(self, line: str) -> None:
        stdin = self.proc.stdin
        assert stdin is not None
        stdin.write(line)
        stdin.flush()

    def kill(self) -> None:
        try:
            self.proc.kill()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass

    def quit(self, grace: float = 10.0) -> None:
        try:
            self.send(protocol.request_line(protocol.OP_QUIT))
            self.proc.wait(timeout=grace)
        except Exception:  # noqa: BLE001
            pass
        self.kill()


class DittoWorker:
    def __init__(self, command: Sequence[str], log_path: Optional[Path] = None) -> None:
        self._command = list(command)
        self._log_path = log_path
        self._proc: Optional[_Process] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Spawn a worker if none is running. Returns at once; use ``wait_ready``."""
        with self._lock:
            if self._proc is not None and self._proc.alive:
                return
            if self._proc is not None:
                self._proc.kill()
            logger.info("starting Ditto worker")
            self._proc = _Process(self._command, self._log_path)

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None:
            proc.quit()

    @property
    def alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.alive

    @property
    def ready(self) -> bool:
        proc = self._proc
        return proc is not None and proc.alive and proc.ready.is_set()

    def wait_ready(self, timeout: float) -> bool:
        """Wait until the worker has loaded its models. False if it exits or times out."""
        end = time.monotonic() + max(timeout, 0.0)
        while True:
            proc = self._proc
            if proc is None or not proc.alive:
                return False
            remaining = end - time.monotonic()
            if proc.ready.wait(min(0.25, max(remaining, 0.0))):
                return proc.alive
            if remaining <= 0:
                return False

    # ----------------------------------------------------------------- jobs
    def run(self, line: str, job_id: str, timeout: float) -> protocol.Result:
        """Send one request line and wait for its result.

        Raises ``WorkerUnavailable`` (no ready process), ``WorkerDied`` (exited
        mid-job) or ``WorkerTimeout`` (killed after ``timeout`` seconds).
        """
        proc = self._proc
        if proc is None or not proc.alive or not proc.ready.is_set():
            raise WorkerUnavailable()
        try:
            proc.send(line)
        except (OSError, ValueError) as exc:
            raise WorkerDied() from exc
        end = time.monotonic() + timeout
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                logger.warning("Ditto worker exceeded %.0fs; restarting it", timeout)
                self._discard(proc)
                raise WorkerTimeout()
            try:
                item = proc.replies.get(timeout=remaining)
            except queue.Empty:
                continue
            if item is None:
                raise WorkerDied()
            if item.job_id == job_id:
                return item
            # a late reply for an earlier job: drop it

    def _discard(self, proc: _Process) -> None:
        with self._lock:
            if self._proc is proc:
                self._proc = None
        proc.kill()
