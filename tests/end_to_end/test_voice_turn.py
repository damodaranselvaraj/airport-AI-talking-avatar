"""Stage 2 gate, end to end with fakes: speak -> transcript -> correct -> Ask -> text + audio.

Mirrors exactly what the browser does (see app/ui/kiosk.js).
"""
import time

from fastapi.testclient import TestClient

from app.main import create_app
from tests.fakes import FakeTranscriber

SESSION = "e2e-session-0001"


def wait_audio(client, url, timeout=3.0):
    end = time.time() + timeout
    while True:
        r = client.get(url)
        if r.status_code != 202 or time.time() > end:
            return r
        time.sleep(0.02)


def test_spoken_question_with_correction_gets_cited_spoken_answer(build_services):
    # Whisper mishears the gate; the traveler fixes it before asking.
    svc = build_services(transcriber=FakeTranscriber("Where is Kate C22?"))
    with TestClient(create_app(svc)) as client:
        t = client.post("/api/transcribe", files={"audio": ("q.webm", b"opus-bytes", "audio/webm")})
        assert t.status_code == 200 and t.json()["transcript"] == "Where is Kate C22?"

        # Edit-then-confirm: nothing has been answered, retrieved, or spoken yet.
        assert svc.retriever.calls == [] and svc.generator.calls == 0 and svc.synthesizer.texts == []

        corrected = t.json()["transcript"].replace("Kate", "gate")
        a = client.post(
            "/api/ask",
            json={"session_id": SESSION, "text": corrected, "input_type": "voice"},
        )
        body = a.json()
        assert body["status"] == "answered"
        assert "Terminal 2" in body["text"]
        assert body["citations"][0]["source"] == "data/gates.md"
        assert svc.retriever.calls == ["Where is gate C22?"]  # the corrected text, not the raw transcript

        audio = wait_audio(client, body["audio_url"])
        assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
        assert svc.synthesizer.texts == [body["text"]]  # TTS speaks the approved text verbatim


def test_spoken_trap_is_refused_and_still_spoken(build_services):
    svc = build_services(transcriber=FakeTranscriber("What is the status of flight DIA 442?"))
    with TestClient(create_app(svc)) as client:
        text = client.post("/api/transcribe", files={"audio": ("q.webm", b"x", "audio/webm")}).json()["transcript"]
        body = client.post("/api/ask", json={"session_id": SESSION, "text": text, "input_type": "voice"}).json()
        assert body["status"] == "refused" and body["citations"] == []
        assert wait_audio(client, body["audio_url"]).status_code == 200
        assert svc.retriever.calls == []
