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


def test_deleting_the_photo_while_its_render_is_queued_cancels_the_video(build_services, concierge):
    # A real queue whose dispatcher never runs, so the render stays queued.
    avatar = AvatarService(DittoWorker([sys.executable, str(STUB)]), AvatarQueue())
    svc = build_services(avatar=avatar, default_face=concierge)
    ref = svc.faces.stage(png())
    svc.faces.activate(SESSION, ref)
    with TestClient(create_app(svc)) as client:
        body = client.post("/api/ask", json={"session_id": SESSION, "text": "Where is gate C22?"}).json()
        for _ in range(100):                               # Kokoro (fake) finishes, the job is queued
            if len(avatar._queue) == 1:
                break
            time.sleep(0.02)
        assert client.get(body["video_url"]).status_code == 202

        client.request("DELETE", "/api/face", json={"session_id": SESSION})

        r = wait(client, body["video_url"])
        assert r.status_code == 410                       # cancelled with the photo -> text + voice fallback
        assert wait(client, body["audio_url"]).status_code == 200
        assert not ref.path.exists()


def test_worker_that_cannot_start_means_text_and_voice_only(build_services, concierge, monkeypatch):
    monkeypatch.setenv("STUB_EXIT_BEFORE_READY", "1")
    avatar = AvatarService(DittoWorker([sys.executable, str(STUB)]), AvatarQueue(), startup_timeout=3)
    avatar.start()
    svc = build_services(avatar=avatar, default_face=concierge)
    with TestClient(create_app(svc)) as client:
        assert client.get("/health").json()["components"]["avatar"] is False
        body = client.post("/api/ask", json={"session_id": SESSION, "text": "Hello"}).json()
        assert body["text"] and body["video_url"]
        assert wait(client, body["video_url"], timeout=20).status_code == 410
        assert wait(client, body["audio_url"]).status_code == 200


def test_worker_crash_mid_turn_degrades_then_a_later_turn_recovers(build_services, concierge, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_CRASH_MARKER", str(tmp_path / "crashed"))
    avatar = AvatarService(DittoWorker([sys.executable, str(STUB)]), AvatarQueue())
    assert avatar.start(wait_seconds=20)
    svc = build_services(avatar=avatar, default_face=concierge)
    with TestClient(create_app(svc)) as client:
        first = client.post("/api/ask", json={"session_id": SESSION, "text": "Hello"}).json()
        assert wait(client, first["video_url"], timeout=20).status_code == 410   # the worker died on it
        assert wait(client, first["audio_url"]).status_code == 200               # the voice still works
        time.sleep(1.3)                                                          # restart backoff
        second = client.post("/api/ask", json={"session_id": SESSION, "text": "Hello"}).json()
        r = wait(client, second["video_url"], timeout=40)
        assert r.status_code == 200 and r.content[:12] == b"\x00\x00\x00\x18ftypmp42"


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
