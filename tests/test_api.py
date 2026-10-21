"""Tasks 2/6/7/8/10: HTTP API behaviour with fakes."""
import time

import pytest

from app.speech_to_text.errors import NoSpeechError, UndecodableAudioError
from tests.conftest import SECRETS
from tests.fakes import FakeSynthesizer, FakeTranscriber

SESSION = "kiosk-session-0001"


def ask(client, text, **extra):
    return client.post("/api/ask", json={"session_id": SESSION, "text": text, **extra})


def wait_for_audio(client, url, timeout=3.0):
    deadline = time.time() + timeout
    while True:
        r = client.get(url)
        if r.status_code != 202 or time.time() > deadline:
            return r
        time.sleep(0.02)


# ---------------------------------------------------------------- /api/ask
def test_grounded_answer_has_text_and_citation(client):
    r = ask(client, "Where is gate C22?")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered"
    assert "Terminal 2" in body["text"]
    assert body["citations"] == [{"source": "data/gates.md", "chunk_id": "gates-c22"}]
    assert body["audio_url"].startswith("/api/audio/")


def test_trap_is_refused_over_http(client):
    r = ask(client, "What is the status of flight NIA 442?")
    body = r.json()
    assert body["status"] == "refused" and body["citations"] == []
    assert client.services.retriever.calls == []


def test_text_is_returned_before_audio_exists(build_services, client):
    r = ask(client, "Hello")
    assert r.json()["status"] == "greeting" and r.json()["text"]


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": SESSION, "text": ""},
        {"session_id": SESSION, "text": "   "},
        {"session_id": "short", "text": "hi"},
        {"session_id": "../../etc/passwd/xx", "text": "hi"},
        {"session_id": SESSION, "text": "x" * 2001},
        {"session_id": SESSION, "text": "hi", "language": "english!"},
        {"session_id": SESSION, "text": "hi", "input_type": "telepathy"},
        {"text": "hi"},
    ],
)
def test_invalid_ask_requests_are_rejected_without_echo(client, payload):
    r = client.post("/api/ask", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "invalid_request"
    assert "etc/passwd" not in r.text and "telepathy" not in r.text


def test_question_longer_than_setting_is_rejected(client):
    r = ask(client, "w" * (client.services.settings.max_question_chars + 1))
    assert r.status_code == 422 and r.json()["detail"]["code"] == "too_long"


def test_error_status_is_text_only_with_no_audio(build_services):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.agents.helpers import make_agent_workflow
    from tests.fakes import FakeRetriever

    wf, _, _, _ = make_agent_workflow(retriever=FakeRetriever(raises=RuntimeError("down")))
    svc = build_services(workflow=wf)
    with TestClient(create_app(svc)) as c:
        body = ask(c, "Where is gate C22?").json()
    assert body["status"] == "error" and body["spoken"] is False and body["audio_url"] is None


# ---------------------------------------------------------------- audio
def test_audio_becomes_available_after_background_render(client):
    url = ask(client, "Where is gate C22?").json()["audio_url"]
    r = wait_for_audio(client, url)
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    assert client.services.synthesizer.texts == ["Gate C22 is in Terminal 2, Concourse C, Level 2."]


def test_audio_failure_degrades_to_410(build_services):
    from fastapi.testclient import TestClient

    from app.main import create_app

    svc = build_services(synthesizer=FakeSynthesizer(raises=RuntimeError("tts down")))
    with TestClient(create_app(svc)) as c:
        body = ask(c, "Hello").json()
        assert body["text"]  # caption still delivered
        r = wait_for_audio(c, body["audio_url"])
        assert r.status_code == 410
        assert "tts down" not in r.text


def test_no_synthesizer_means_no_audio_url(build_services):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(build_services(synthesizer=None))) as c:
        assert ask(c, "Hello").json()["audio_url"] is None


@pytest.mark.parametrize(
    "turn_id", ["0" * 32, "abc", "..%2f..%2fsecret", "%2e%2e%2f%2e%2e%2fsecret", "A" * 32, "a" * 33]
)
def test_unknown_or_malicious_turn_ids_are_404(client, turn_id):
    assert client.get(f"/api/audio/{turn_id}").status_code == 404


def test_audio_files_are_deleted_on_shutdown(build_services, tmp_path):
    from fastapi.testclient import TestClient

    from app.main import create_app

    svc = build_services()
    with TestClient(create_app(svc)) as c:
        wait_for_audio(c, ask(c, "Hello").json()["audio_url"])
        assert list((tmp_path / "sessions").glob("*.wav"))
    assert not list((tmp_path / "sessions").glob("*.wav"))


def test_expired_audio_is_swept(build_services, settings):
    from app.audio_store import AudioStore

    # Negative TTL = already expired; avoids depending on the OS clock's tick (~15 ms on Windows).
    store = AudioStore(settings.sessions_dir, ttl_seconds=-1)
    tid = store.new_turn_id()
    store.begin(tid)
    store.put(tid, b"RIFFxxxx")
    assert store.get(tid) == (None, None)
    store.sweep()
    assert not list(__import__("pathlib").Path(settings.sessions_dir).glob("*.wav"))


def test_store_only_purges_its_own_wavs(tmp_path):
    from app.audio_store import AudioStore

    d = tmp_path / "s"
    d.mkdir()
    (d / "ditto_spike.mp4").write_bytes(b"x")
    (d / "text_avatar.wav").write_bytes(b"x")
    (d / ("a" * 32 + ".wav")).write_bytes(b"x")
    AudioStore(d)
    assert (d / "ditto_spike.mp4").exists() and (d / "text_avatar.wav").exists()
    assert not (d / ("a" * 32 + ".wav")).exists()


# ---------------------------------------------------------------- /api/transcribe
def test_transcribe_returns_transcript_and_never_runs_workflow(client):
    r = client.post("/api/transcribe", files={"audio": ("clip.webm", b"abc", "audio/webm")})
    assert r.status_code == 200
    body = r.json()
    assert body["transcript"] == "Where is gate C22?"
    assert body["duration_s"] == 2.0 and body["confidence"] == 0.9
    assert client.services.retriever.calls == []      # edit-then-confirm is structural
    assert client.services.generator.calls == 0
    assert client.services.synthesizer.texts == []


def test_transcribe_rejects_empty_upload(client):
    r = client.post("/api/transcribe", files={"audio": ("clip.webm", b"", "audio/webm")})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "empty_audio"
    assert client.services.transcriber.calls == 0


def test_transcribe_rejects_oversize_upload(client):
    big = b"\0" * (client.services.settings.max_audio_bytes + 1)
    r = client.post("/api/transcribe", files={"audio": ("clip.webm", big, "audio/webm")})
    assert r.status_code == 413 and r.json()["detail"]["code"] == "audio_too_large"
    assert client.services.transcriber.calls == 0


def test_transcribe_requires_audio_field(client):
    assert client.post("/api/transcribe").status_code == 422


@pytest.mark.parametrize(
    "exc,status,code",
    [(NoSpeechError(), 422, "no_speech"), (UndecodableAudioError(), 415, "undecodable_audio"),
     (RuntimeError("secret internals sk-abc"), 500, "stt_failed")],
)
def test_transcribe_errors_are_safe(build_services, exc, status, code):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(build_services(transcriber=FakeTranscriber(raises=exc)))) as c:
        r = c.post("/api/transcribe", files={"audio": ("c.webm", b"abc", "audio/webm")})
    assert r.status_code == status and r.json()["detail"]["code"] == code
    assert "secret internals" not in r.text


def test_transcribe_unavailable_without_stt(build_services):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(build_services(transcriber=None))) as c:
        r = c.post("/api/transcribe", files={"audio": ("c.webm", b"abc", "audio/webm")})
    assert r.status_code == 503


# ---------------------------------------------------------------- headers
def test_security_headers_present(client):
    r = client.get("/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert "permissions-policy" in r.headers
    assert "access-control-allow-origin" not in r.headers  # CORS is not enabled
    assert client.post("/api/ask", json={"session_id": SESSION, "text": "Hello"}).headers["cache-control"] == "no-store"


def test_cross_origin_preflight_is_not_allowed(client):
    r = client.options(
        "/api/ask",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in r.headers


def test_secrets_never_appear_in_any_response(client):
    values = list(SECRETS.values())
    responses = [
        client.get("/health"),
        client.get("/"),
        client.get("/kiosk.js"),
        client.get("/kiosk.css"),
        client.get("/openapi.json"),
        ask(client, "Where is gate C22?"),
        ask(client, "Hello"),
        client.post("/api/ask", json={"nope": 1}),
        client.get("/api/audio/" + "0" * 32),
        client.post("/api/transcribe", files={"audio": ("c.webm", b"", "audio/webm")}),
    ]
    for r in responses:
        for secret in values:
            assert secret not in r.text, f"{r.request.url} leaked a secret"
            assert secret.encode() not in r.content


# ---------------------------------------------------------------- UI + hardening
def test_interactive_docs_and_schema_are_not_exposed(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_ui_is_served_and_is_csp_clean(client):
    import re

    html = client.get("/").text
    assert "<title>Dam International Airport" in html
    assert 'src="/kiosk.js"' in html and 'href="/kiosk.css"' in html
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), "inline script would violate the CSP"
    assert " style=" not in html and "onclick=" not in html
    assert not re.search(r"https?://", html), "UI must not load anything from other origins"
    js = client.get("/kiosk.js").text
    assert "innerHTML" not in js, "server text must only be inserted with textContent"
    assert "api_key" not in js.lower() and "authorization" not in js.lower()
    assert not re.search(r"https?://", js)


def test_files_outside_the_ui_folder_are_not_served(client):
    for path in ("/.env", "/config/settings.py", "/app/main.py", "/..%2f.env", "/%2e%2e/.env"):
        assert client.get(path).status_code == 404
