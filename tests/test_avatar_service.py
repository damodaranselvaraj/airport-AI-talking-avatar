"""Avatar queue ordering/deadlines/cancel, and the real worker client + service
driven by a stub worker process (no GPU)."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from app.avatar.models import FaceCheck, Job, JobOutcome, JobStatus, Priority, QueueFull
from app.avatar.queue import AvatarQueue
from app.avatar.service import AvatarService
from app.avatar.worker_client import DittoWorker

STUB = Path(__file__).with_name("stub_worker.py")


# ------------------------------------------------------------------ queue ----
class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def make_job(job_id: str, priority=Priority.LIVE, deadline=1000.0, face_key=None, sink=None) -> Job:
    return Job(
        job_id=job_id,
        op="render",
        source=Path("/f.png"),
        audio=Path("/a.wav"),
        output=Path("/o.mp4"),
        priority=priority,
        deadline=deadline,
        on_done=(sink.append if sink is not None else (lambda _o: None)),
        face_key=face_key,
    )


def test_queue_orders_by_priority_then_arrival():
    q = AvatarQueue(clock=Clock())
    for jid, prio in [("bg1", Priority.BACKGROUND), ("live1", Priority.LIVE), ("chk", Priority.FACE_CHECK),
                      ("bg2", Priority.BACKGROUND), ("live2", Priority.LIVE)]:
        q.put(make_job(jid, prio))
    order = [q.get(timeout=0).job_id for _ in range(5)]
    assert order == ["live1", "live2", "chk", "bg1", "bg2"]


def test_queue_skips_expired_jobs_and_reports_them():
    clock = Clock()
    q = AvatarQueue(clock=clock)
    outcomes: list[JobOutcome] = []
    q.put(make_job("old", deadline=clock.t + 5, sink=outcomes))
    q.put(make_job("fresh", Priority.BACKGROUND, deadline=clock.t + 500))
    clock.t += 10  # "old" is now past its deadline
    assert q.get(timeout=0).job_id == "fresh"
    assert [(o.job_id, o.status) for o in outcomes] == [("old", JobStatus.EXPIRED)]


def test_queue_cancel_face_removes_only_that_faces_jobs():
    q = AvatarQueue(clock=Clock())
    outcomes: list[JobOutcome] = []
    q.put(make_job("a1", Priority.BACKGROUND, face_key="A", sink=outcomes))
    q.put(make_job("b1", Priority.BACKGROUND, face_key="B", sink=outcomes))
    q.put(make_job("a2", Priority.BACKGROUND, face_key="A", sink=outcomes))
    q.put(make_job("live", Priority.LIVE))
    assert q.cancel_face("A") == 2
    assert sorted(o.job_id for o in outcomes if o.status is JobStatus.CANCELLED) == ["a1", "a2"]
    assert [q.get(timeout=0).job_id for _ in range(2)] == ["live", "b1"]
    assert q.get(timeout=0) is None


def test_queue_full_and_close():
    q = AvatarQueue(max_pending=2, clock=Clock())
    outcomes: list[JobOutcome] = []
    q.put(make_job("1", sink=outcomes))
    q.put(make_job("2", sink=outcomes))
    with pytest.raises(QueueFull):
        q.put(make_job("3"))
    assert q.close() == 2
    assert {o.status for o in outcomes} == {JobStatus.CANCELLED}
    with pytest.raises(RuntimeError):
        q.put(make_job("4"))


def test_get_blocks_until_a_job_arrives():
    q = AvatarQueue()
    got: list[Job] = []
    t = threading.Thread(target=lambda: got.append(q.get(timeout=5)))
    t.start()
    time.sleep(0.1)
    q.put(make_job("late", deadline=q.now() + 100))  # real clock here, not the fake one
    t.join(timeout=5)
    assert got and got[0].job_id == "late"


# ------------------------------------------------- service with stub worker --
def make_service(**kwargs) -> AvatarService:
    worker = DittoWorker([sys.executable, str(STUB)])
    return AvatarService(worker, AvatarQueue(), **kwargs)


@pytest.fixture
def service():
    svc = make_service()
    yield svc
    svc.stop()


def render_and_wait(svc: AvatarService, tmp_path: Path, job_id: str, timeout: float = 20.0, **kw) -> JobOutcome:
    done = threading.Event()
    box: list[JobOutcome] = []

    def finished(outcome: JobOutcome) -> None:
        box.append(outcome)
        done.set()

    svc.render(job_id, tmp_path / "f.png", tmp_path / "a.wav", tmp_path / f"{job_id}.mp4", on_done=finished, **kw)
    assert done.wait(timeout), "job never finished"
    return box[0]


def test_service_renders_through_the_worker_process(service, tmp_path):
    assert service.start(wait_seconds=20) and service.ready
    outcome = render_and_wait(service, tmp_path, "turn-1")
    assert outcome.ok and outcome.extra == {"audio_seconds": 1.0}
    assert (tmp_path / "turn-1.mp4").stat().st_size > 1000


def test_service_runs_jobs_one_at_a_time_in_priority_order(service, tmp_path):
    service.start(wait_seconds=20)
    order: list[str] = []
    lock = threading.Lock()
    done = threading.Event()

    def record(outcome: JobOutcome) -> None:
        with lock:
            order.append(outcome.job_id)
            if len(order) == 3:
                done.set()

    # Hold the dispatcher so all three are queued before any starts.
    gate = threading.Event()
    service._queue.put(Job("gate", "check_face", tmp_path / "f.png", Priority.LIVE, service._queue.now() + 60,
                           lambda _o: gate.wait(5)))
    service.render("bg", tmp_path / "f.png", tmp_path / "a.wav", tmp_path / "bg.mp4",
                   priority=Priority.BACKGROUND, on_done=record)
    service.render("live", tmp_path / "f.png", tmp_path / "a.wav", tmp_path / "live.mp4",
                   priority=Priority.LIVE, on_done=record)
    service.render("bg2", tmp_path / "f.png", tmp_path / "a.wav", tmp_path / "bg2.mp4",
                   priority=Priority.BACKGROUND, on_done=record)
    gate.set()
    assert done.wait(20)
    assert order == ["live", "bg", "bg2"]


def test_check_face_ok_and_no_face(monkeypatch, tmp_path):
    svc = make_service()
    try:
        svc.start(wait_seconds=20)
        assert svc.check_face(tmp_path / "f.png", timeout_s=20) is FaceCheck.OK
    finally:
        svc.stop()
    monkeypatch.setenv("STUB_NO_FACE", "1")
    svc = make_service()
    try:
        svc.start(wait_seconds=20)
        assert svc.check_face(tmp_path / "f.png", timeout_s=20) is FaceCheck.NO_FACE
    finally:
        svc.stop()


def test_worker_crash_fails_that_job_and_a_later_job_restarts_it(monkeypatch, tmp_path):
    monkeypatch.setenv("STUB_CRASH_MARKER", str(tmp_path / "crashed.marker"))
    svc = make_service()
    try:
        svc.start(wait_seconds=20)
        first = render_and_wait(svc, tmp_path, "boom")
        assert first.status is JobStatus.FAILED and first.error_code == "render_failed"
        assert (tmp_path / "crashed.marker").exists()
        time.sleep(1.2)  # first restart backoff is 1 s
        second = render_and_wait(svc, tmp_path, "healed", timeout=40)
        assert second.ok
        assert svc.ready
    finally:
        svc.stop()


def test_hung_worker_is_killed_after_the_hard_timeout(monkeypatch, tmp_path):
    monkeypatch.setenv("STUB_HANG_RENDER", "1")
    svc = make_service(hard_job_timeout=1.0)
    try:
        svc.start(wait_seconds=20)
        outcome = render_and_wait(svc, tmp_path, "hang", timeout=20)
        assert outcome.status is JobStatus.FAILED
        assert not svc.ready  # the hung process was discarded
    finally:
        svc.stop()


def test_worker_that_cannot_start_reports_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("STUB_EXIT_BEFORE_READY", "1")
    svc = make_service(startup_timeout=5)
    try:
        assert svc.start(wait_seconds=5) is False
        outcome = render_and_wait(svc, tmp_path, "nope", timeout=20)
        assert outcome.status is JobStatus.UNAVAILABLE
        assert not svc.ready
    finally:
        svc.stop()


def test_job_ids_are_validated(service, tmp_path):
    with pytest.raises(ValueError):
        service.render("bad id!", tmp_path / "f.png", tmp_path / "a.wav", tmp_path / "o.mp4", on_done=lambda _o: None)


def test_stop_cancels_queued_jobs(tmp_path):
    svc = make_service()
    outcomes: list[JobOutcome] = []
    # Never started: nothing consumes the queue, so jobs stay queued until stop().
    svc.render("q1", tmp_path / "f.png", tmp_path / "a.wav", tmp_path / "o.mp4", on_done=outcomes.append)
    svc.stop()
    assert [o.status for o in outcomes] == [JobStatus.CANCELLED]
