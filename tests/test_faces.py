"""Traveler photos: sanitising, consent, expiry, keep-alive, deletion (fake Ditto, no GPU)."""
from __future__ import annotations

import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.avatar.faces import FaceStore
from app.avatar.images import BadImage, sanitize_photo
from app.avatar.models import FaceCheck, FaceRef
from app.main import create_app
from tests.fakes import FakeAvatar

SESSION = "kiosk-session-0001"
OTHER = "kiosk-session-0002"


def photo(size=(400, 300), fmt="PNG", color=(180, 120, 90), exif: bytes | None = None) -> bytes:
    img = Image.new("RGB", size, color)
    out = io.BytesIO()
    kwargs = {"exif": exif} if exif else {}
    img.save(out, format=fmt, **kwargs)
    return out.getvalue()


def exif_with(orientation: int | None = None, make: str | None = None) -> bytes:
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
    if make is not None:
        exif[0x010F] = make
    return exif.tobytes()


def upload(client, data=None, session=SESSION, consent="true", name="me.png", ctype="image/png"):
    data = photo() if data is None else data
    return client.post(
        "/api/face",
        files={"image": (name, data, ctype)},
        data={"session_id": session, "consent": consent},
    )


def face_files(svc) -> list[Path]:
    return sorted(Path(svc.settings.faces_dir).glob("*.png"))


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
    path.write_bytes(photo())
    return path


@pytest.fixture
def setup(build_services, concierge):
    avatar = FakeAvatar()
    svc = build_services(avatar=avatar, default_face=concierge)
    with TestClient(create_app(svc)) as client:
        yield client, svc, avatar


# ------------------------------------------------------------ sanitising ----
def test_sanitize_outputs_a_clean_png_without_metadata():
    src = photo(fmt="JPEG", exif=exif_with(make="SecretCam 3000"))
    assert b"SecretCam" in src
    out = sanitize_photo(src)
    img = Image.open(io.BytesIO(out))
    assert img.format == "PNG" and img.mode == "RGB"
    assert b"SecretCam" not in out and b"Exif" not in out
    assert len(img.getexif()) == 0 and not img.info.get("icc_profile")


def test_sanitize_applies_exif_rotation_before_dropping_it():
    src = photo(size=(400, 200), fmt="JPEG", exif=exif_with(orientation=6))  # camera turned 90 degrees
    out = Image.open(io.BytesIO(sanitize_photo(src)))
    assert out.size == (200, 400)


def test_sanitize_downscales_large_photos():
    out = Image.open(io.BytesIO(sanitize_photo(photo(size=(3000, 2000)))))
    assert max(out.size) == 1280


@pytest.mark.parametrize(
    "data",
    [b"", b"not an image at all", photo(fmt="GIF"), photo(size=(64, 64)), photo(size=(6000, 6000))],
    ids=["empty", "garbage", "gif", "tiny", "pixel-bomb"],
)
def test_sanitize_rejects_unusable_input(data):
    with pytest.raises(BadImage):
        sanitize_photo(data)


# ---------------------------------------------------------------- upload ----
def test_upload_stores_a_clean_random_named_png_and_checks_it_for_a_face(setup):
    client, svc, avatar = setup
    src = photo(fmt="JPEG", exif=exif_with(make="SecretCam 3000"))
    r = upload(client, src, name="IMG_1234_John_Smith.jpg", ctype="image/jpeg")
    assert r.status_code == 200 and r.json() == {"status": "ok", "expires_in_seconds": 60}
    (stored,) = face_files(svc)
    assert stored.name != "IMG_1234_John_Smith.jpg" and len(stored.stem) == 32
    assert b"SecretCam" not in stored.read_bytes()
    assert avatar.checked == [stored]
    assert str(stored) not in r.text and stored.stem not in r.text


@pytest.mark.parametrize("consent", ["", "false", "no", "yes please"])
def test_consent_is_required_and_nothing_is_stored(setup, consent):
    client, svc, avatar = setup
    r = upload(client, consent=consent)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "consent_required"
    assert face_files(svc) == [] and avatar.checked == []


def test_consent_field_missing_entirely_is_refused(setup):
    client, svc, _ = setup
    r = client.post("/api/face", files={"image": ("a.png", photo(), "image/png")}, data={"session_id": SESSION})
    assert r.status_code == 422 and face_files(svc) == []


def test_oversize_upload_is_413(setup):
    client, svc, avatar = setup
    svc.settings.max_face_bytes = 1000
    r = upload(client, photo(size=(900, 900), fmt="BMP"))
    assert r.status_code == 413 and r.json()["detail"]["code"] == "too_large"
    assert face_files(svc) == [] and avatar.checked == []


@pytest.mark.parametrize("data", [b"plain text", photo(fmt="GIF"), photo(size=(50, 50))])
def test_unusable_files_are_bad_image(setup, data):
    client, svc, avatar = setup
    r = upload(client, data)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "bad_image"
    assert face_files(svc) == [] and avatar.checked == []


def test_photo_without_a_face_is_rejected_and_not_kept(build_services, concierge):
    avatar = FakeAvatar(face_check=FaceCheck.NO_FACE)
    svc = build_services(avatar=avatar, default_face=concierge)
    with TestClient(create_app(svc)) as client:
        r = upload(client)
        assert r.status_code == 422 and r.json()["detail"]["code"] == "no_face"
        assert face_files(svc) == []
        assert client.post("/api/face/keepalive", json={"session_id": SESSION}).json() == {"active": False}


def test_worker_unavailable_rejects_and_does_not_keep_the_photo(build_services, concierge):
    svc = build_services(avatar=FakeAvatar(face_check=FaceCheck.UNAVAILABLE), default_face=concierge)
    with TestClient(create_app(svc)) as client:
        r = upload(client)
        assert r.status_code == 503 and r.json()["detail"]["code"] == "avatar_unavailable"
        assert face_files(svc) == []


def test_avatar_disabled_means_faces_unavailable(client):
    r = upload(client)
    assert r.status_code == 503 and r.json()["detail"]["code"] == "avatar_unavailable"


@pytest.mark.parametrize("session", ["short", "../../etc/passwd", "x" * 65])
def test_bad_session_id_is_rejected(setup, session):
    client, svc, _ = setup
    r = upload(client, session=session)
    assert r.status_code == 422 and face_files(svc) == []


def test_replacing_a_photo_deletes_the_old_one_and_cancels_its_jobs(setup):
    client, svc, avatar = setup
    upload(client)
    (first,) = face_files(svc)
    upload(client, photo(color=(10, 200, 10)))
    (second,) = face_files(svc)
    assert second != first and not first.exists()
    assert avatar.cancelled == [first.stem]


def test_each_session_has_its_own_photo(setup):
    client, svc, _ = setup
    upload(client, session=SESSION)
    upload(client, session=OTHER)
    assert len(face_files(svc)) == 2


# -------------------------------------------------- using the photo in a turn
def test_a_grounded_answer_uses_the_travelers_photo_and_others_use_the_default(setup, concierge):
    client, svc, avatar = setup
    upload(client)
    (stored,) = face_files(svc)
    mine = client.post("/api/ask", json={"session_id": SESSION, "text": "Where is gate C22?"}).json()
    other = client.post("/api/ask", json={"session_id": OTHER, "text": "Where is gate C22?"}).json()
    assert mine["status"] == "answered"
    wait(client, mine["video_url"])
    wait(client, other["video_url"])
    by_job = {j["job_id"]: j for j in avatar.renders}
    assert by_job[mine["turn_id"]]["source"] == stored and by_job[mine["turn_id"]]["face_key"] == stored.stem
    assert by_job[other["turn_id"]]["source"] == concierge and by_job[other["turn_id"]]["face_key"] == "default"


def test_delete_removes_the_photo_cancels_jobs_and_restores_the_default(setup, concierge):
    client, svc, avatar = setup
    upload(client)
    (stored,) = face_files(svc)
    r = client.request("DELETE", "/api/face", json={"session_id": SESSION})
    assert r.status_code == 200 and r.json() == {"deleted": True}
    assert not stored.exists() and avatar.cancelled == [stored.stem]
    again = client.request("DELETE", "/api/face", json={"session_id": SESSION})
    assert again.json() == {"deleted": False}  # idempotent
    body = client.post("/api/ask", json={"session_id": SESSION, "text": "Where is gate C22?"}).json()
    wait(client, body["video_url"])
    assert avatar.renders[-1]["source"] == concierge


def test_keepalive_reports_whether_a_photo_is_active(setup):
    client, _, _ = setup
    assert client.post("/api/face/keepalive", json={"session_id": SESSION}).json() == {"active": False}
    upload(client)
    assert client.post("/api/face/keepalive", json={"session_id": SESSION}).json() == {"active": True}
    assert client.post("/api/face/keepalive", json={"session_id": "bad"}).status_code == 422


def test_shutdown_deletes_every_photo(build_services, concierge):
    svc = build_services(avatar=FakeAvatar(), default_face=concierge)
    with TestClient(create_app(svc)) as client:
        upload(client, session=SESSION)
        upload(client, session=OTHER)
        assert len(face_files(svc)) == 2
    assert face_files(svc) == []


def test_face_pending_turn_keeps_the_photo_until_the_video_is_done(build_services, concierge):
    clock = Clock()
    avatar = FakeAvatar(auto=False)
    svc = build_services(avatar=avatar, default_face=concierge)
    svc.faces = FaceStore(svc.settings.faces_dir, 60, on_expire=avatar.cancel_face, clock=clock)
    with TestClient(create_app(svc)) as client:
        upload(client)
        body = client.post("/api/ask", json={"session_id": SESSION, "text": "Where is gate C22?"}).json()
        for _ in range(100):  # wait until the render has been handed to the (fake) worker
            if avatar.renders:
                break
            time.sleep(0.02)
        clock.t += 200  # well past the 60 s backstop, but the render is still pending
        assert svc.faces.sweep() == []
        assert len(face_files(svc)) == 1
        avatar.complete(body["turn_id"])
        assert svc.faces.sweep() == []  # the idle clock restarted when the turn finished
        clock.t += 61
        assert len(svc.faces.sweep()) == 1 and face_files(svc) == []


# ------------------------------------------------------------- FaceStore ----
class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def store(tmp_path):
    clock = Clock()
    expired: list[str] = []
    s = FaceStore(tmp_path / "faces", backstop_seconds=60, on_expire=expired.append, clock=clock)
    s.clock, s.expired = clock, expired  # type: ignore[attr-defined]
    yield s
    s.close()


def test_store_expires_after_the_backstop_and_reports_the_key(store, tmp_path):
    ref = store.stage(b"png")
    store.activate("s1", ref)
    store.clock.t += 59
    assert store.sweep() == [] and ref.path.exists()
    store.clock.t += 2
    assert store.sweep() == [ref.key]
    assert not ref.path.exists() and store.get("s1") is None
    assert store.expired == [ref.key]


def test_store_touch_extends_expiry(store):
    ref = store.stage(b"png")
    store.activate("s1", ref)
    store.clock.t += 50
    assert store.touch("s1") is True
    store.clock.t += 50
    assert store.sweep() == [] and ref.path.exists()
    store.clock.t += 20
    assert store.sweep() == [ref.key]
    assert store.touch("s1") is False


def test_store_pending_turn_pauses_expiry_but_only_for_a_while(store):
    ref = store.stage(b"png")
    store.activate("s1", ref)
    store.turn_started(ref.key)
    store.clock.t += 200
    assert store.sweep() == []
    store.clock.t += 200  # beyond the pause cap: a stuck turn cannot keep a photo forever
    assert store.sweep() == [ref.key]


def test_store_delete_and_replace_remove_files(store):
    a, b = store.stage(b"a"), store.stage(b"b")
    assert store.activate("s1", a) is None
    assert store.activate("s1", b) == a.key
    assert not a.path.exists() and b.path.exists()
    assert store.delete("s1") == b.key and not b.path.exists()
    assert store.delete("s1") is None


def test_store_purges_only_its_own_leftovers_at_start(tmp_path):
    folder = tmp_path / "faces"
    folder.mkdir()
    mine, keep1, keep2 = folder / f"{'a' * 32}.png", folder / "me.png", folder / ".gitkeep"
    for f in (mine, keep1, keep2):
        f.write_bytes(b"x")
    FaceStore(folder).close()
    assert not mine.exists() and keep1.exists() and keep2.exists()


def test_store_retries_a_deletion_the_os_refused(store, monkeypatch):
    ref = store.stage(b"png")
    store.activate("s1", ref)
    real_unlink = Path.unlink
    calls = {"n": 0}

    def flaky(self, missing_ok=False):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("in use")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky)
    store.delete("s1")
    assert ref.path.exists()           # first attempt failed
    store.sweep()                      # the retry succeeds
    assert not ref.path.exists()


def test_background_sweeper_deletes_without_any_request(tmp_path):
    s = FaceStore(tmp_path / "faces", backstop_seconds=0.2)
    try:
        ref = s.stage(b"png")
        s.activate("s1", ref)
        deadline = time.time() + 5
        while ref.path.exists() and time.time() < deadline:
            time.sleep(0.1)
        assert not ref.path.exists()
    finally:
        s.close()
