"""Per-turn talking-face video over HTTP, with a fake Ditto (no GPU)."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.avatar.models import Priority
from app.main import create_app
from tests.fakes import FakeAvatar, FakeSynthesizer

SESSION = "kiosk-session-0001"


def ask(client, text, **extra):
    return client.post("/api/ask", json={"session_id": SESSION, "text": text, **extra})


def wait(client, url, timeout=3.0):
    deadline = time.time() + timeout
    while True:
        r = client.get(url)
        if r.status_code != 202 or time.time() > deadline:
            return r
        time.sleep(0.02)


@pytest.fixture
def concierge(tmp_path):
    path = tmp_path / "concierge.png"
    path.write_bytes(b"not-a-real-png")  # the fake avatar never opens it
    return path


@pytest.fixture
def avatar_client(build_services, concierge):
    avatar = FakeAvatar()
    svc = build_services(avatar=avatar, default_face=concierge)
    with TestClient(create_app(svc)) as c:
        c.services, c.avatar = svc, avatar  # type: ignore[attr-defined]
        yield c


def test_spoken_turn_gets_a_video_url_and_a_playable_video(avatar_client, concierge):
    body = ask(avatar_client, "Where is gate C22?").json()
    assert body["video_url"] == f"/api/video/{body['turn_id']}"
    assert body["audio_url"] == f"/api/audio/{body['turn_id']}"  # kept as the fallback
    r = wait(avatar_client, body["video_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert r.headers["cache-control"] == "no-store"
    assert r.content == FakeAvatar.FAKE_MP4


def test_render_job_gets_the_turn_audio_default_face_and_deadline(avatar_client, concierge):
    body = ask(avatar_client, "Where is gate C22?").json()
    wait(avatar_client, body["video_url"])
    (job,) = avatar_client.avatar.renders
    assert job["job_id"] == body["turn_id"]
    assert job["source"] == concierge
    assert job["audio"].name == f"{body['turn_id']}.wav" and job["audio"].suffix == ".wav"
    assert job["output"].name == f"{body['turn_id']}.mp4"
    assert job["priority"] is Priority.LIVE
    assert job["deadline_s"] == avatar_client.services.settings.avatar_timeout_seconds


def test_video_is_pending_until_the_render_finishes(build_services, concierge):
    avatar = FakeAvatar(auto=False)
    with TestClient(create_app(build_services(avatar=avatar, default_face=concierge))) as c:
        body = ask(c, "Where is gate C22?").json()
        wait(c, body["audio_url"])  # Kokoro finished, so the job has been handed over
        for _ in range(100):
            if avatar.renders:
                break
            time.sleep(0.02)
        r = c.get(body["video_url"])
        assert r.status_code == 202 and r.json() == {"status": "pending"}
        avatar.complete(body["turn_id"])
        assert c.get(body["video_url"]).status_code == 200


def test_failed_render_is_410_and_audio_still_works(build_services, concierge):
    avatar = FakeAvatar(fail_code="no_face")
    with TestClient(create_app(build_services(avatar=avatar, default_face=concierge))) as c:
        body = ask(c, "Where is gate C22?").json()
        r = wait(c, body["video_url"])
        assert r.status_code == 410
        assert r.json()["detail"]["code"] == "video_failed"
        assert "no_face" not in r.text
        assert wait(c, body["audio_url"]).status_code == 200


def test_queue_full_degrades_to_410(build_services, concierge):
    with TestClient(create_app(build_services(avatar=FakeAvatar(queue_full=True), default_face=concierge))) as c:
        body = ask(c, "Where is gate C22?").json()
        assert wait(c, body["video_url"]).status_code == 410
        assert wait(c, body["audio_url"]).status_code == 200


def test_tts_failure_means_no_video_either(build_services, concierge):
    avatar = FakeAvatar()
    svc = build_services(avatar=avatar, default_face=concierge, synthesizer=FakeSynthesizer(raises=RuntimeError("tts down")))
    with TestClient(create_app(svc)) as c:
        body = ask(c, "Where is gate C22?").json()
        r = wait(c, body["video_url"])
        assert r.status_code == 410 and "tts down" not in r.text
        assert avatar.renders == []


def test_unspoken_error_turn_has_no_video(build_services, concierge):
    from tests.agents.helpers import make_agent_workflow
    from tests.fakes import FakeRetriever

    wf, _, _, _ = make_agent_workflow(retriever=FakeRetriever(raises=RuntimeError("down")))
    with TestClient(create_app(build_services(workflow=wf, avatar=FakeAvatar(), default_face=concierge))) as c:
        body = ask(c, "Where is gate C22?").json()
    assert body["status"] == "error" and body["spoken"] is False
    assert body["video_url"] is None and body["audio_url"] is None


def test_avatar_disabled_keeps_stage_two_behaviour(client):
    body = ask(client, "Where is gate C22?").json()
    assert body["video_url"] is None and body["audio_url"]
    assert client.get(f"/api/video/{'a' * 32}").status_code == 404


def test_no_default_face_means_no_video(build_services):
    with TestClient(create_app(build_services(avatar=FakeAvatar(), default_face=None))) as c:
        assert ask(c, "Hello").json()["video_url"] is None


def test_no_synthesizer_means_no_video(build_services, concierge):
    with TestClient(create_app(build_services(avatar=FakeAvatar(), default_face=concierge, synthesizer=None))) as c:
        body = ask(c, "Hello").json()
    assert body["audio_url"] is None and body["video_url"] is None


@pytest.mark.parametrize("turn_id", ["a" * 32, "../../etc/passwd", "short", "A" * 32])
def test_unknown_or_malformed_turn_ids_are_404(avatar_client, turn_id):
    assert avatar_client.get(f"/api/video/{turn_id}").status_code == 404


def test_videos_are_deleted_on_shutdown(build_services, concierge, tmp_path):
    svc = build_services(avatar=FakeAvatar(), default_face=concierge)
    with TestClient(create_app(svc)) as c:
        wait(c, ask(c, "Where is gate C22?").json()["video_url"])
        assert list((tmp_path / "sessions").glob("*.mp4"))
    assert not list((tmp_path / "sessions").glob("*.mp4"))


def test_only_own_mp4_files_are_purged_at_start(settings, tmp_path):
    from app.video_store import VideoStore

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    own, other = sessions / f"{'a' * 32}.mp4", sessions / "ditto_spike.mp4"
    own.write_bytes(b"x")
    other.write_bytes(b"x")
    VideoStore(sessions, 600)
    assert not own.exists() and other.exists()


def test_expired_video_is_swept(tmp_path):
    from app.video_store import VideoStore

    store = VideoStore(tmp_path, ttl_seconds=0)
    tid = "b" * 32
    store.begin(tid)
    store.path_for(tid).write_bytes(b"x")
    store.finish(tid, ok=True)
    time.sleep(0.01)
    assert store.get(tid) == (None, None)
    store.sweep()
    assert not store.path_for(tid).exists()


def test_shared_clip_is_served_but_never_deleted(tmp_path):
    from app.audio_store import AudioState
    from app.video_store import VideoStore

    clip = tmp_path / "cache" / "hello.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"clip")
    store = VideoStore(tmp_path / "sessions", ttl_seconds=0)
    tid = "c" * 32
    store.ready_from_shared(tid, clip)
    store.sweep()
    store.close()
    assert clip.exists()
    store2 = VideoStore(tmp_path / "sessions", ttl_seconds=600)
    store2.ready_from_shared(tid, clip)
    assert store2.get(tid) == (AudioState.READY, clip)


def test_finish_after_sweep_drops_the_file(tmp_path):
    from app.video_store import VideoStore

    store = VideoStore(tmp_path, ttl_seconds=600)
    tid = "d" * 32
    store.path_for(tid).write_bytes(b"late")  # written after the entry vanished
    store.finish(tid, ok=True)
    assert not store.path_for(tid).exists()


# ------------------------------------------------- page config + concierge ---
def test_config_for_the_page_has_timers_and_flags_but_no_secrets_or_paths(avatar_client, concierge):
    r = avatar_client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "avatar_enabled": True,
        "has_concierge": True,
        "avatar_timeout_seconds": 180.0,
        "face_idle_warn_seconds": 30,
        "face_idle_countdown_seconds": 15,
        "face_keepalive_seconds": 20,
        "max_face_bytes": 5_000_000,
    }
    assert str(concierge) not in r.text and "sk-" not in r.text


def test_config_when_the_avatar_is_off(client):
    body = client.get("/api/config").json()
    assert body["avatar_enabled"] is False and body["has_concierge"] is False


def test_concierge_image_is_served_when_the_avatar_is_on(avatar_client, concierge):
    r = avatar_client.get("/api/concierge")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content == concierge.read_bytes()


def test_concierge_image_404_when_the_avatar_is_off_or_missing(client, build_services):
    assert client.get("/api/concierge").status_code == 404
    with TestClient(create_app(build_services(avatar=FakeAvatar(), default_face=None))) as c:
        assert c.get("/api/concierge").status_code == 404
