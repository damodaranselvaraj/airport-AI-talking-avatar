"""Drive the real kiosk UI in headless Chrome/Edge with a FAKE microphone.

Proves the browser half of Stages 2-3 (MediaRecorder -> /api/transcribe -> editable
transcript -> Ask -> answer + citation -> audio element, or with the avatar on:
held answer -> talking-face video) against a RUNNING server:

    python scripts/browser_smoke.py [--base http://127.0.0.1:8000] [--say "Where is gate C22?"]

The fake mic plays a WAV synthesized by Kokoro. No new dependencies: it speaks
the Chrome DevTools Protocol over `websockets`, which is already installed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
PORT = 9333


class Page:
    def __init__(self, ws_url: str) -> None:
        self.ws = connect(ws_url, max_size=None)
        self._id = 0

    def call(self, method: str, **params):
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv(timeout=60))
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    def js(self, expr: str):
        res = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in res:
            raise RuntimeError(res["exceptionDetails"])
        return res["result"].get("value")

    def wait_for(self, expr: str, timeout: float, what: str):
        end = time.time() + timeout
        while time.time() < end:
            if self.js(expr):
                return
            time.sleep(0.3)
        raise TimeoutError(f"timed out waiting for {what}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--say", default="Where is gate C22?")
    ap.add_argument("--fix", nargs=2, metavar=("OLD", "NEW"), help="correct the transcript before asking")
    args = ap.parse_args()

    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if not browser:
        print("No Chrome/Edge found.", file=sys.stderr)
        return 2

    from app.text_to_speech.kokoro_service import KokoroSynthesizer

    tmp = Path(tempfile.mkdtemp(prefix="kiosk-browser-"))
    wav = tmp / "mic.wav"
    wav.write_bytes(KokoroSynthesizer().synthesize(args.say))
    proc = subprocess.Popen(
        [
            browser, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={tmp / 'profile'}",
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={wav}%noloop", "--autoplay-policy=no-user-gesture-required",
            "--no-first-run", "--disable-gpu", "about:blank",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=2))
                page_ws = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        else:
            print("Could not reach the browser's debugging port.", file=sys.stderr)
            return 2

        page = Page(page_ws)
        page.call("Page.enable")
        page.call("Page.navigate", url=args.base + "/")
        page.wait_for("document.readyState === 'complete'", 15, "page load")
        print("page loaded; mic button:", page.js("document.getElementById('mic-btn').textContent.trim()"))

        t0 = time.time()
        page.js("document.getElementById('mic-btn').click()")
        page.wait_for("document.getElementById('mic-btn').getAttribute('aria-pressed') === 'true'", 10, "recording to start")
        print("recording; status:", page.js("document.getElementById('status').textContent"))
        time.sleep(4.0)                                   # fake mic plays the clip once
        page.js("document.getElementById('mic-btn').click()")
        page.wait_for("document.getElementById('question').value.length > 0", 90, "transcript")
        transcript = page.js("document.getElementById('question').value")
        print(f"transcript in box after {time.time() - t0:.1f}s: {transcript!r}")
        print("hint:", page.js("document.getElementById('transcript-hint').textContent"))
        assert page.js("document.getElementById('answer-panel').hidden"), "answered before Ask was tapped!"
        print("OK: nothing answered before Ask")

        if args.fix:
            fixed = transcript.replace(*args.fix)
            page.js(f"(() => {{ const q = document.getElementById('question'); q.value = {json.dumps(fixed)}; q.dispatchEvent(new Event('input')); }})()")
            print("corrected to:", fixed)

        avatar_on = bool(page.js("!document.getElementById('face').hidden"))
        print("avatar on:", avatar_on)
        t1 = time.time()
        page.js("document.getElementById('ask-btn').click()")
        if avatar_on:
            page.wait_for("document.getElementById('status').textContent.startsWith('Preparing the avatar')", 30, "avatar wait message")
            print("status while waiting:", page.js("document.getElementById('status').textContent"))
            assert page.js("document.getElementById('answer-panel').hidden"), "text was shown before the video was ready!"
            print("OK: text and voice are held while the video renders")
        page.wait_for("!document.getElementById('answer-panel').hidden && document.getElementById('answer-text').textContent.length > 0", 200 if avatar_on else 60, "answer")
        print(f"answer after {time.time() - t1:.1f}s: {page.js('document.getElementById(\"answer-text\").textContent')!r}")
        print("citation:", page.js("document.getElementById('citation').hidden ? '(none)' : document.getElementById('citation').textContent"))
        page.wait_for("document.getElementById('player').src.startsWith('blob:') || document.getElementById('avatar-video').src.startsWith('blob:') || !document.getElementById('audio-note').hidden", 60, "media")
        if page.js("document.getElementById('avatar-video').src.startsWith('blob:')"):
            time.sleep(1.0)
            print("video element: blob loaded, duration", page.js("document.getElementById('avatar-video').duration"),
                  "size", page.js("document.getElementById('avatar-video').videoWidth + 'x' + document.getElementById('avatar-video').videoHeight"),
                  "currentTime", page.js("document.getElementById('avatar-video').currentTime"))
        else:
            print("audio element:", page.js("document.getElementById('player').src.startsWith('blob:') ? 'blob loaded, duration ' + document.getElementById('player').duration : document.getElementById('audio-note').textContent"))
        print("final status:", page.js("document.getElementById('status').textContent"))
        return 0
    finally:
        if proc.poll() is not None:
            print(f"NOTE: the browser process had already exited (code {proc.returncode})", file=sys.stderr)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
