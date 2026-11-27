"""Stage 3 failure paths with the REAL avatar queue and a stub worker (no GPU)."""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.avatar.queue import AvatarQueue
from app.avatar.service import AvatarService
from app.avatar.worker_client import DittoWorker
from app.main import create_app
from tests.fakes import FakeAvatar

SESSION = "kiosk-session-0001"
STUB = Path(__file__).with_name("stub_worker.py")


def png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (300, 300), (150, 100, 80)).save(out, format="PNG")
    return out.getvalue()


def wait(client, url, timeout=5.0):
    deadline = time.time() + timeout
    while True:
        r = client.get(url)
        if r.status_code != 202 or time.time() > deadline:
            return r
        time.sleep(0.02)


@pytest.fixture
def concierge(tmp_path):
    path = tmp_path / "concierge.png"
    path.write_bytes(png())
    return path


def test_with_the_worker_not_ready_a_grounded_answer_is_text_and_voice_only(build_services, concierge):
    # A real queue whose worker was never started, so avatar.ready is False. Per the
    # unchanged design-T8 worker_not_ready safety net, the turn degrades to text + voice
    # with no video (video_url is None); deleting the photo still removes it from disk.
    avatar = AvatarService(DittoWorker([sys.executable, str(STUB)]), AvatarQueue())
    svc = build_services(avatar=avatar, default_face=concierge)
    ref = svc.faces.stage(png())
    svc.faces.activate(SESSION, ref)
    with TestClient(create_app(svc)) as client:
        body = client.post("/api/ask", json={"session_id": SESSION, "text": "Where is gate C22?"}).json()
        assert body["video_url"] is None                   # worker not ready -> audio-only, nothing queued
        assert wait(client, body["audio_url"]).status_code == 200

        client.request("DELETE", "/api/face", json={"session_id": SESSION})
        assert not ref.path.exists()


def test_worker_that_cannot_start_means_text_and_voice_only(build_services, concierge, monkeypatch):
    monkeypatch.setenv("STUB_EXIT_BEFORE_READY", "1")
    avatar = AvatarService(DittoWorker([sys.executable, str(STUB)]), AvatarQueue(), startup_timeout=3)
    avatar.start()
    svc = build_services(avatar=avatar, default_face=concierge)
    with TestClient(create_app(svc)) as client:
        assert client.get("/health").json()["components"]["avatar"] is False
        body = client.post("/api/ask", json={"session_id": SESSION, "text": "Hello"}).json()
        # Greeting with no pre-rendered clip + a worker that cannot start: text + voice only,
        # no video (greeting_clip_missing), the unchanged design-T8 safety net.
        assert body["text"] and body["video_url"] is None
        assert wait(client, body["audio_url"]).status_code == 200


def test_worker_crash_mid_turn_degrades_then_a_later_turn_recovers(build_services, concierge, tmp_path, monkeypatch):
    # Uses a grounded answer: unlike a greeting (always audio-only per design T8), a grounded
    # answer renders live with a ready worker, so this exercises the worker crash -> recover path.
    monkeypatch.setenv("STUB_CRASH_MARKER", str(tmp_path / "crashed"))
    avatar = AvatarService(DittoWorker([sys.executable, str(STUB)]), AvatarQueue())
    assert avatar.start(wait_seconds=20)
    svc = build_services(avatar=avatar, default_face=concierge)
    with TestClient(create_app(svc)) as client:
        first = client.post("/api/ask", json={"session_id": SESSION, "text": "Where is gate C22?"}).json()
        assert wait(client, first["video_url"], timeout=20).status_code == 410   # the worker died on it
        assert wait(client, first["audio_url"]).status_code == 200               # the voice still works
        # A worker_not_ready turn kicks off a restart (lazily, per the design-T8 safety net), then
        # audio-only until it is back up. Keep asking until a later turn renders a video again.
        deadline = time.time() + 40
        video = None
        while time.time() < deadline:
            time.sleep(1.3)                                                      # restart backoff
            later = client.post("/api/ask", json={"session_id": SESSION, "text": "Where is gate C22?"}).json()
            if later["video_url"] is None:                                       # worker still not ready
                continue
            r = wait(client, later["video_url"], timeout=40)
            if r.status_code == 200:
                video = r
                break
        assert video is not None and video.content[:12] == b"\x00\x00\x00\x18ftypmp42"


def test_queue_full_still_returns_the_answer_and_voice(build_services, concierge):
    svc = build_services(avatar=FakeAvatar(queue_full=True), default_face=concierge)
    with TestClient(create_app(svc)) as client:
        body = client.post("/api/ask", json={"session_id": SESSION, "text": "Where is gate C22?"}).json()
        assert body["status"] == "answered" and "Terminal 2" in body["text"]
        assert wait(client, body["video_url"]).status_code == 410
        assert wait(client, body["audio_url"]).status_code == 200


def test_error_bodies_never_contain_paths_or_exception_text(build_services, concierge):
    svc = build_services(avatar=FakeAvatar(fail_code="render_failed"), default_face=concierge)
    ref = svc.faces.stage(png())
    svc.faces.activate(SESSION, ref)
    with TestClient(create_app(svc)) as client:
        body = client.post("/api/ask", json={"session_id": SESSION, "text": "Where is gate C22?"}).json()
        r = wait(client, body["video_url"])
        assert r.status_code == 410
        for text in (r.text, str(body)):
            assert str(ref.path) not in text and ref.key not in text and "render_failed" not in text
        up = client.post(
            "/api/face",
            files={"image": ("x.png", b"garbage", "image/png")},
            data={"session_id": SESSION, "consent": "true"},
        )
        assert up.status_code == 422 and "garbage" not in up.text and "Traceback" not in up.text
