"""Drive the real kiosk page in headless Chrome/Edge against an IN-PROCESS fake server.

No GPU, no API keys, no network: the server is the real FastAPI app wired to
fakes (fake Ditto that finishes when this script says so, fake Kokoro, fake
Whisper). That makes the avatar flow deterministic, so we can assert exactly
what the traveler sees while the video is "rendering":

    python scripts/browser_ui_check.py            # all scenarios
    python scripts/browser_ui_check.py --only hold fallback

Exits non-zero on the first failed assertion.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os  # noqa: E402

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("TYPESAFE_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY", "test-key")

import uvicorn  # noqa: E402
from browser_smoke import BROWSERS, Page  # noqa: E402
from PIL import Image  # noqa: E402

from app.avatar.clips import ClipCache, ClipRenderer, file_digest, load_phrases  # noqa: E402
from app.avatar.models import DEFAULT_FACE_KEY, FaceRef  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services import Services  # noqa: E402
from app.audio_store import AudioStore  # noqa: E402
from config.settings import Settings  # noqa: E402
from tests.agents.helpers import make_agent_workflow  # noqa: E402
from tests.fakes import FakeAvatar, FakeSynthesizer, FakeTranscriber  # noqa: E402

HTTP_PORT = 8765
CDP_PORT = 9334
BASE = f"http://127.0.0.1:{HTTP_PORT}"

# Short timers so the scenarios run in seconds. The page reads them from /api/config.
AVATAR_TIMEOUT = 4
WARN, COUNTDOWN, BACKSTOP = 3, 4, 30


# ------------------------------------------------------------------ fixtures
def make_mp4(path: Path) -> bytes:
    """A real 2 s H.264 + AAC clip so Chrome can decode and play it."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is needed to build a playable test clip")
    subprocess.run(
        [ffmpeg, "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=0x446688:s=256x256:d=2:r=25",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True,
    )
    return path.read_bytes()


def face_png(color=(180, 120, 90), size=(400, 400)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format="PNG")
    return out.getvalue()


class Harness:
    def __init__(self, tmp: Path) -> None:
        concierge = tmp / "concierge.png"
        concierge.write_bytes(face_png((40, 90, 150)))
        self.avatar = FakeAvatar(auto=False, mp4=make_mp4(tmp / "clip.mp4"))
        self.settings = Settings(
            _env_file=None,
            openai_api_key="x", typesafe_api_key="x", cohere_api_key="x",
            sessions_dir=str(tmp / "sessions"), faces_dir=str(tmp / "faces"),
            avatar_timeout_seconds=AVATAR_TIMEOUT,
            face_idle_warn_seconds=WARN, face_idle_countdown_seconds=COUNTDOWN,
            face_ttl_backstop_seconds=BACKSTOP,
        )
        workflow, _, _, _ = make_agent_workflow()
        synthesizer = FakeSynthesizer()
        # The concierge's phrase clips. Nothing is scheduled until the "warmup" scenario does it,
        # so every other scenario sees a kiosk that is simply not warming up.
        self.clips = ClipCache(
            tmp / "clips", load_phrases(ROOT / "config" / "responses.json", "en"),
            voice=self.settings.kokoro_voice, language="en", default_digest=file_digest(concierge),
        )
        self.renderer = ClipRenderer(self.clips, self.avatar, synthesizer)
        self.services = Services(
            settings=self.settings, workflow=workflow,
            audio=AudioStore(self.settings.sessions_dir, 600),
            transcriber=FakeTranscriber(), synthesizer=synthesizer,
            avatar=self.avatar, default_face=concierge,
            clips=self.clips, clip_renderer=self.renderer,
        )
        self.server = uvicorn.Server(
            uvicorn.Config(create_app(self.services), host="127.0.0.1", port=HTTP_PORT, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.touches: list[float] = []                     # every keep-alive the server receives
        real_touch = self.services.faces.touch

        def counting_touch(session_id: str) -> bool:
            self.touches.append(time.time())
            return real_touch(session_id)

        self.services.faces.touch = counting_touch  # type: ignore[method-assign]
        self.concierge = concierge

    def face_files(self) -> list[Path]:
        return sorted(Path(self.settings.faces_dir).glob("*.png"))

    def start(self) -> None:
        self.thread.start()
        for _ in range(100):
            try:
                urllib.request.urlopen(BASE + "/health", timeout=1)
                return
            except Exception:
                time.sleep(0.1)
        raise SystemExit("fake server did not start")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)

    def wait_for_render(self, count: int = 1, timeout: float = 10.0) -> list[str]:
        end = time.time() + timeout
        while time.time() < end:
            if len(self.avatar.renders) >= count:
                return [r["job_id"] for r in self.avatar.renders]
            time.sleep(0.05)
        raise TimeoutError("the page never asked Ditto for a video")

    def reset(self) -> None:
        self.touches.clear()
        self.avatar.checked.clear()
        self.avatar.renders.clear()
        self.avatar._pending.clear()
        self.avatar.fail_code = None
        self.avatar.face_check_result = __import__("app.avatar.models", fromlist=["FaceCheck"]).FaceCheck.OK


# --------------------------------------------------------------- the browser
class Browser:
    def __init__(self, tmp: Path) -> None:
        path = next((b for b in BROWSERS if Path(b).exists()), None)
        if not path:
            raise SystemExit("No Chrome/Edge found.")
        self.proc = subprocess.Popen(
            [path, "--headless=new", f"--remote-debugging-port={CDP_PORT}", f"--user-data-dir={tmp / 'profile'}",
             "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
             "--autoplay-policy=no-user-gesture-required", "--no-first-run", "--disable-gpu", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json", timeout=2))
                self.ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise SystemExit("could not reach the browser")
        self.page = Page(self.ws_url)
        self.page.call("Page.enable")

    def open(self) -> Page:
        self.page.call("Page.navigate", url=BASE + "/")
        self.page.wait_for("document.readyState === 'complete'", 15, "page load")
        self.page.wait_for("window.__kioskReady === true || document.getElementById('face') !== null", 5, "page init")
        time.sleep(0.5)  # let /api/config land
        return self.page

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


# -------------------------------------------------------------------- checks
def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}")


def ask(page: Page, text: str) -> None:
    page.js(f"""(() => {{ const q = document.getElementById('question'); q.value = {json.dumps(text)};
                q.dispatchEvent(new Event('input')); document.getElementById('ask-btn').click(); }})()""")


def state(page: Page) -> str:
    return page.js("document.querySelector('main.kiosk').dataset.state")


def status(page: Page) -> str:
    return page.js("document.getElementById('status').textContent")


def panel_hidden(page: Page) -> bool:
    return bool(page.js("document.getElementById('answer-panel').hidden"))


def scenario_load(h: Harness, page: Page) -> None:
    print("scenario: page loads with the concierge face")
    check(not page.js("document.getElementById('face').hidden"), "face area is shown (avatar enabled)")
    page.wait_for("document.getElementById('face-still').naturalWidth > 0", 10, "concierge image")
    check(True, "default concierge image loaded from /api/concierge")
    check(state(page) == "idle" and status(page) == "Ready", "state is idle and status says Ready")


def scenario_hold(h: Harness, page: Page) -> None:
    print("scenario: text and voice are held until the video is ready, then all start together")
    ask(page, "Where is gate C22?")
    page.wait_for("document.querySelector('main.kiosk').dataset.state === 'thinking'", 5, "thinking state")
    (job,) = h.wait_for_render(1)
    page.wait_for("document.getElementById('status').textContent.startsWith('Preparing the avatar')", 5, "avatar wait message")
    check(True, "status says 'Preparing the avatar...' while Ditto renders")
    time.sleep(1.5)                                   # Kokoro audio is long since ready, but must stay held
    check(panel_hidden(page), "answer text is NOT shown yet")
    check(not page.js("document.getElementById('player').getAttribute('src')"), "voice is NOT playing yet")
    check(bool(page.js("document.getElementById('ask-btn').disabled")), "Ask is disabled while waiting")
    check(not page.js("document.getElementById('clear-btn').disabled"), "Clear stays usable so the wait can be cancelled")

    h.avatar.complete(job)                            # Ditto finishes
    page.wait_for("!document.getElementById('answer-panel').hidden", 10, "answer to appear")
    check("Terminal 2" in page.js("document.getElementById('answer-text').textContent"), "answer text appears with the video")
    page.wait_for("document.getElementById('avatar-video').src.startsWith('blob:')", 10, "video src")
    page.wait_for("document.getElementById('avatar-video').currentTime > 0 || document.querySelector('main.kiosk').dataset.state === 'speaking'", 10, "video playing")
    check(True, "video is playing (lips move) in the face area")
    check(not page.js("document.getElementById('player').getAttribute('src')"), "no second audio track: the voice is inside the video")
    page.wait_for("document.querySelector('main.kiosk').dataset.state === 'idle'", 15, "back to idle after the clip ends")
    check(page.js("document.getElementById('avatar-video').hidden"), "video hides and the still face returns after speaking")
    check(not panel_hidden(page), "answer text stays on screen")


def scenario_fallback_on_failure(h: Harness, page: Page) -> None:
    print("scenario: render fails -> text appears and the Kokoro voice plays")
    ask(page, "Where is the prayer room?")
    (job,) = h.wait_for_render(1)
    check(panel_hidden(page), "text is held while rendering")
    h.avatar.complete(job, ok=False)
    page.wait_for("!document.getElementById('answer-panel').hidden", 10, "answer to appear")
    check(True, "text appears after the failure")
    note = page.js("document.getElementById('audio-note').hidden ? '' : document.getElementById('audio-note').textContent")
    check("could not be shown" in note, "a short notice explains the avatar is unavailable")
    page.wait_for("document.getElementById('player').src.startsWith('blob:')", 15, "Kokoro audio")
    check(True, "the Kokoro voice is played over the still face")
    check(page.js("document.getElementById('avatar-video').hidden"), "no video is shown")


def scenario_fallback_on_timeout(h: Harness, page: Page) -> None:
    print(f"scenario: render never finishes -> after ~{AVATAR_TIMEOUT + 5}s the page falls back")
    started = time.time()
    ask(page, "Where is gate C22?")
    h.wait_for_render(1)
    page.wait_for("!document.getElementById('answer-panel').hidden", AVATAR_TIMEOUT + 15, "fallback after timeout")
    waited = time.time() - started
    check(AVATAR_TIMEOUT <= waited <= AVATAR_TIMEOUT + 12, f"fell back after {waited:.1f}s (limit {AVATAR_TIMEOUT}s + grace)")
    note = page.js("document.getElementById('audio-note').hidden ? '' : document.getElementById('audio-note').textContent")
    check("could not be shown" in note, "notice shown")
    page.wait_for("document.getElementById('player').src.startsWith('blob:')", 15, "Kokoro audio")
    check(True, "Kokoro voice plays")


def scenario_clear_cancels_wait(h: Harness, page: Page) -> None:
    print("scenario: Clear while waiting cancels the turn")
    ask(page, "Where is gate C22?")
    (job,) = h.wait_for_render(1)
    page.js("document.getElementById('clear-btn').click()")
    check(state(page) == "idle" and panel_hidden(page), "page is idle with no answer")
    check(not page.js("document.getElementById('ask-btn').disabled"), "Ask is usable again")
    h.avatar.complete(job)                            # a late video must not pop up
    time.sleep(2.0)
    check(panel_hidden(page) and not page.js("document.getElementById('avatar-video').getAttribute('src')"),
          "the late video is ignored")


def set_file(page: Page, data: bytes, name: str = "me.png", mime: str = "image/png") -> None:
    """Hand the page a file the way the file chooser would."""
    b64 = base64.b64encode(data).decode()
    page.js(f"""(() => {{ const bytes = Uint8Array.from(atob('{b64}'), c => c.charCodeAt(0));
        const dt = new DataTransfer(); dt.items.add(new File([bytes], '{name}', {{type: '{mime}'}}));
        const input = document.getElementById('face-file'); input.files = dt.files;
        input.dispatchEvent(new Event('change')); }})()""")


def face_status(page: Page) -> str:
    return page.js("document.getElementById('face-status').textContent")


def still_src(page: Page) -> str:
    return page.js("document.getElementById('face-still').getAttribute('src')")


def upload_my_face(h: Harness, page: Page) -> None:
    page.js("document.getElementById('face-consent').click()")
    set_file(page, face_png())
    page.wait_for("document.getElementById('face-status').textContent.startsWith('Done')", 10, "photo accepted")


def scenario_face_upload(h: Harness, page: Page) -> None:
    print("scenario: upload your own photo (consent first), talk with it, reset")
    check(not page.js("document.getElementById('face-panel').hidden"), "face picker is shown")
    set_file(page, face_png())                         # no consent yet
    time.sleep(0.5)
    check("tick the box" in face_status(page), "without consent the page asks for it")
    check(h.avatar.checked == [] and h.face_files() == [], "nothing was sent to the server")

    upload_my_face(h, page)
    check(still_src(page).startswith("blob:"), "the still face is now the traveler's photo")
    check(not page.js("document.getElementById('face-reset-btn').hidden"), "'Use default concierge' appears")
    (mine,) = h.face_files()
    check(len(h.avatar.checked) == 1, "the server checked the photo for a face")

    ask(page, "Where is gate C22?")
    (job,) = h.wait_for_render(1)
    render = h.avatar.renders[-1]
    check(render["source"] == mine and render["face_key"] == mine.stem, "the video is rendered from the traveler's photo, not the concierge")
    h.avatar.complete(job)
    page.wait_for("!document.getElementById('answer-panel').hidden", 10, "answer")
    page.wait_for("document.querySelector('main.kiosk').dataset.state === 'idle'", 15, "idle again")

    page.js("document.getElementById('face-reset-btn').click()")
    page.wait_for("document.getElementById('face-reset-btn').hidden", 5, "reset")
    check(still_src(page) == "/api/concierge", "the default concierge is back")
    time.sleep(0.5)
    check(h.face_files() == [], "the photo was deleted from the server")
    check(not page.js("document.getElementById('face-consent').checked"), "consent is cleared for the next photo")


def scenario_face_rejected(h: Harness, page: Page) -> None:
    print("scenario: a photo without a face is refused")
    from app.avatar.models import FaceCheck

    h.avatar.face_check_result = FaceCheck.NO_FACE
    page.js("document.getElementById('face-consent').click()")
    set_file(page, face_png())
    page.wait_for("document.getElementById('face-status').classList.contains('error')", 10, "error message")
    check("couldn't find a clear face" in face_status(page), "the message explains what to do")
    check(still_src(page) == "/api/concierge" and page.js("document.getElementById('face-reset-btn').hidden"),
          "the concierge stays and there is nothing to reset")
    check(h.face_files() == [], "the rejected photo was not kept")


def scenario_face_camera(h: Harness, page: Page) -> None:
    print("scenario: take a photo with the webcam")
    page.js("document.getElementById('face-camera-btn').click()")
    check("tick the box" in face_status(page) and page.js("document.getElementById('camera').hidden"),
          "the camera does not open before consent")
    page.js("document.getElementById('face-consent').click()")
    page.js("document.getElementById('face-camera-btn').click()")
    page.wait_for("!document.getElementById('camera').hidden", 10, "camera preview")
    page.wait_for("document.getElementById('camera-preview').videoWidth > 0", 10, "camera frames")
    check(True, "camera preview is live")
    page.js("document.getElementById('camera-snap').click()")
    page.wait_for("document.getElementById('face-status').textContent.startsWith('Done')", 15, "photo accepted")
    check(page.js("document.getElementById('camera').hidden"), "the camera is switched off after capture")
    check(still_src(page).startswith("blob:") and len(h.face_files()) == 1, "the captured photo is the avatar")


def scenario_idle_handover(h: Harness, page: Page) -> None:
    print(f"scenario: after {WARN}s idle 'Still there?' appears; ignoring it resets after {COUNTDOWN}s")
    upload_my_face(h, page)
    ask(page, "Where is gate C22?")
    (job,) = h.wait_for_render(1)
    h.avatar.complete(job)
    page.wait_for("document.querySelector('main.kiosk').dataset.state === 'idle' && !document.getElementById('answer-panel').hidden", 15, "answer shown, idle")
    spoken_done = time.time()

    page.wait_for("!document.getElementById('idle-dialog').hidden", WARN + 6, "'Still there?' prompt")
    waited = time.time() - spoken_done
    check(WARN - 1 <= waited <= WARN + 4, f"prompt appeared {waited:.1f}s after speaking ended (limit {WARN}s)")
    check(page.js("document.activeElement.id") == "idle-stay", "focus moves to the 'Yes, I'm here' button")
    check(page.js("document.getElementById('idle-dialog').getAttribute('role')") == "alertdialog", "it is an alert dialog")
    first = page.js("document.getElementById('idle-desc').textContent")
    time.sleep(1.6)
    second = page.js("document.getElementById('idle-desc').textContent")
    check(first != second and "photo will be removed" in first, "a visible countdown is ticking and mentions the photo")

    touches_before = len(h.touches)
    page.js("document.getElementById('idle-stay').click()")
    check(page.js("document.getElementById('idle-dialog').hidden"), "'Yes, I'm here' closes the prompt")
    time.sleep(0.5)
    check(len(h.touches) > touches_before, "the server was told to keep the photo")
    check(still_src(page).startswith("blob:") and not panel_hidden(page), "photo and answer are still there")

    page.wait_for("!document.getElementById('idle-dialog').hidden", WARN + 4, "prompt again after more quiet")
    page.wait_for("document.getElementById('idle-dialog').hidden", COUNTDOWN + 4, "automatic reset")
    time.sleep(0.5)
    check(still_src(page) == "/api/concierge", "default concierge returned")
    check(panel_hidden(page) and page.js("document.getElementById('question').value") == "", "question box and answer panel are cleared")
    check(h.face_files() == [], "the photo is deleted on the server")
    check("next traveler" in status(page), "the reset is announced in the status line")


def scenario_idle_ignored_when_nothing_to_reset(h: Harness, page: Page) -> None:
    print("scenario: no prompt when there is nothing to reset")
    time.sleep(WARN + 2)
    check(page.js("document.getElementById('idle-dialog').hidden"), "a fresh page is left alone")


def scenario_idle_paused_while_rendering(h: Harness, page: Page) -> None:
    print("scenario: the prompt never appears while an answer is being prepared")
    upload_my_face(h, page)
    ask(page, "Where is gate C22?")
    h.wait_for_render(1)
    time.sleep(WARN + 2.5)
    check(page.js("document.getElementById('idle-dialog').hidden"), "no prompt while the video renders")
    check(state(page) == "thinking", "still waiting for the avatar")


def scenario_warmup(h: Harness, page: Page) -> None:
    print("scenario: 'Warming up' locks the page until every phrase clip exists")
    check(not page.js("document.getElementById('ask-btn').disabled"), "without pre-rendering the page is not locked")
    total = h.clips.total()
    h.renderer.schedule(FaceRef(path=h.concierge, key=DEFAULT_FACE_KEY))   # what the server does at start
    h.wait_for_render(total)
    page.call("Page.navigate", url=BASE + "/")                              # a traveler opens the page now
    page.wait_for("document.readyState === 'complete'", 15, "page load")
    page.wait_for(f"document.getElementById('status').textContent === 'Warming up... 0 of {total} ready'", 10, "warming up message")
    check(state(page) == "thinking", "the page shows a busy state while warming up")
    for button in ("ask-btn", "mic-btn", "face-upload-btn", "face-camera-btn"):
        check(bool(page.js(f"document.getElementById('{button}').disabled")), f"{button} is locked")
    ask(page, "Where is gate C22?")
    time.sleep(0.5)
    check(h.avatar.renders[total:] == [], "pressing Enter / Ask during warm-up sends nothing")

    for job in list(h.avatar._pending)[:3]:
        h.avatar.complete(job)
    page.wait_for(f"document.getElementById('status').textContent === 'Warming up... 3 of {total} ready'", 8, "progress")
    check(True, "progress follows the server: 3 of %d" % total)

    for job in list(h.avatar._pending):
        h.avatar.complete(job)
    page.wait_for("document.getElementById('status').textContent === 'Ready'", 8, "unlock")
    check(not page.js("document.getElementById('ask-btn').disabled"), "Ask unlocks when the last clip is done")
    check(not page.js("document.getElementById('face-upload-btn').disabled"), "the photo buttons unlock too")


SCENARIOS = {
    "load": scenario_load,
    "hold": scenario_hold,
    "fallback": scenario_fallback_on_failure,
    "timeout": scenario_fallback_on_timeout,
    "clear": scenario_clear_cancels_wait,
    "upload": scenario_face_upload,
    "noface": scenario_face_rejected,
    "camera": scenario_face_camera,
    "idle": scenario_idle_handover,
    "idle-empty": scenario_idle_ignored_when_nothing_to_reset,
    "idle-render": scenario_idle_paused_while_rendering,
    "warmup": scenario_warmup,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=list(SCENARIOS), help="run just these scenarios")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="kiosk-ui-check-"))
    harness, browser = Harness(tmp), None
    failures = 0
    try:
        harness.start()
        browser = Browser(tmp)
        for name in args.only or list(SCENARIOS):
            harness.reset()
            page = browser.open()
            try:
                SCENARIOS[name](harness, page)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        print("\nALL UI SCENARIOS PASSED" if not failures else f"\n{failures} scenario(s) FAILED")
        return 1 if failures else 0
    finally:
        if browser:
            browser.close()
        harness.stop()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
