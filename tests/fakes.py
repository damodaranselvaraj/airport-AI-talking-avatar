"""Test doubles shared across the suite (no network, no models)."""
from __future__ import annotations

import io
import struct
import wave
from pathlib import Path
from typing import Optional, Sequence

from app.workflow.answer import GeneratedAnswer
from app.workflow.contracts import ErrorState, RetrievedPassage, Transcript

ROOT = Path(__file__).resolve().parents[1]

GATE_C22 = RetrievedPassage(
    chunk_id="gates-c22",
    source="data/gates.md",
    score=0.93,
    text=(
        "## Gate C22 (demo gate)\n**Gate C22 is in Terminal 2, Concourse C, Level 2.**\n"
        "From T2 security (central checkpoint): follow signs for Gates C, then walk past C12. "
        "Allow 12 minutes."
    ),
)
PRAYER = RetrievedPassage(
    chunk_id="faq-prayer",
    source="data/faq.md",
    score=0.41,
    text="The main prayer room is Terminal 2, Level 1, airside. It is open 24 hours.",
)


class FakeRetriever:
    """Stands in for HybridRetriever. Records calls so tests can assert none happened."""

    def __init__(
        self,
        passages: Sequence[RetrievedPassage] = (GATE_C22,),
        error: Optional[ErrorState] = None,
        raises: Optional[Exception] = None,
    ) -> None:
        self.passages = list(passages)
        self.error = error
        self.raises = raises
        self.calls: list[str] = []

    def retrieve_with_status(self, query: str, top_k: int):
        self.calls.append(query)
        if self.raises:
            raise self.raises
        return list(self.passages)[:top_k], self.error


class FakeGenerator:
    def __init__(
        self,
        answer: Optional[GeneratedAnswer] = None,
        raises: Optional[Exception] = None,
        delay: float = 0.0,
    ) -> None:
        self.answer = answer or GeneratedAnswer(
            True, "Gate C22 is in Terminal 2, Concourse C, Level 2.", [0]
        )
        self.raises = raises
        self.delay = delay
        self.calls = 0

    def generate(self, question, passages, language="en"):
        self.calls += 1
        if self.delay:
            import time

            time.sleep(self.delay)
        if self.raises:
            raise self.raises
        return self.answer



class FakeTranscriber:
    ready = True

    def __init__(self, text: str = "Where is gate C22?", raises: Optional[Exception] = None) -> None:
        self.text = text
        self.raises = raises
        self.calls = 0

    def transcribe(self, data: bytes) -> Transcript:
        self.calls += 1
        if self.raises:
            raise self.raises
        return Transcript(text=self.text, duration_s=2.0, confidence=0.9)


def silent_wav(seconds: float = 0.2, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(rate * seconds))
    return buf.getvalue()


class FakeSynthesizer:
    ready = True

    def __init__(self, raises: Optional[Exception] = None) -> None:
        self.raises = raises
        self.texts: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.texts.append(text)
        if self.raises:
            raise self.raises
        return silent_wav()


class FakeAvatar:
    """Stands in for the Ditto ``AvatarService``: no GPU, no subprocess.

    ``auto=True`` completes each render at once (writing a small placeholder mp4);
    ``auto=False`` keeps jobs pending until the test calls ``complete``.
    """

    FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 2048

    def __init__(
        self,
        ready: bool = True,
        auto: bool = True,
        face_check: "FaceCheck" = None,  # type: ignore[assignment]
        fail_code: Optional[str] = None,
        queue_full: bool = False,
        mp4: Optional[bytes] = None,
    ) -> None:
        from app.avatar.models import FaceCheck

        self.mp4 = mp4 if mp4 is not None else self.FAKE_MP4
        self.ready = ready
        self.auto = auto
        self.face_check_result = face_check or FaceCheck.OK
        self.fail_code = fail_code
        self.queue_full = queue_full
        self.renders: list[dict] = []
        self.checked: list[Path] = []
        self.cancelled: list[str] = []
        self.stopped = False
        self.starts = 0
        self._pending: dict[str, tuple] = {}

    def render(self, job_id, source, audio, output, *, priority=None, face_key=None, deadline_s=90.0, on_done):
        from app.avatar.models import Priority, QueueFull

        if self.queue_full:
            raise QueueFull()
        self.renders.append(
            dict(
                job_id=job_id,
                source=Path(source),
                audio=Path(audio),
                output=Path(output),
                priority=priority if priority is not None else Priority.LIVE,
                face_key=face_key,
                deadline_s=deadline_s,
            )
        )
        self._pending[job_id] = (Path(output), on_done)
        if self.auto:
            self.complete(job_id)

    def complete(self, job_id: str, ok: Optional[bool] = None, status=None) -> None:
        from app.avatar.models import JobOutcome, JobStatus

        output, on_done = self._pending.pop(job_id)
        success = (self.fail_code is None) if ok is None else ok
        if status is not None:
            on_done(JobOutcome(job_id, status))
        elif success:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(self.mp4)
            on_done(JobOutcome(job_id, JobStatus.OK, seconds=0.01))
        else:
            on_done(JobOutcome(job_id, JobStatus.FAILED, self.fail_code or "render_failed"))

    def start(self, wait_seconds: float = 0.0) -> bool:
        self.starts += 1  # no-op: the real service only spawns a worker if none is alive
        return self.ready

    def check_face(self, source, *, timeout_s: float = 30.0):
        self.checked.append(Path(source))
        return self.face_check_result

    def cancel_face(self, face_key: str) -> int:
        self.cancelled.append(face_key)
        return 0

    def stop(self) -> None:
        self.stopped = True
