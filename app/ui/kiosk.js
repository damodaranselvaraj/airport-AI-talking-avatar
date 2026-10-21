/* Dam AI Talking Avatar Kiosk - plain-JS client (Stages 1-3).
 *
 * Flow: type OR tap Speak -> POST /api/transcribe -> transcript lands in the
 * editable box -> traveler fixes it -> taps Ask -> POST /api/ask.
 *
 * With the avatar on, the approved text and the voice are HELD until the
 * talking-face video is ready, then text, voice and moving lips start together.
 * If the video fails or takes longer than the server's limit, the text appears
 * and the Kokoro voice plays over the still face instead.
 * With the avatar off (Stage 2), text shows at once and the voice follows.
 *
 * Fixed phrases (greeting, refusals, ...) come from pre-rendered videos of the default
 * concierge. At server start those are rendered, and until they exist the page shows
 * "Warming up... n of m ready" and takes no questions (GET /api/clips/status).
 *
 * One state machine drives the page: idle | listening | thinking | speaking | error
 * (data-state on <main>, plus the status line). No API keys exist in the browser.
 * All server text is inserted with textContent, never as markup.
 */
(() => {
  "use strict";

  const MAX_RECORD_MS = 15000;      // server cap is 20 s; stop a little earlier
  const REQUEST_TIMEOUT_MS = 60000;
  const AUDIO_POLL_MS = 400;
  const AUDIO_WAIT_MS = 30000;
  const VIDEO_POLL_MS = 700;
  const LATE_VIDEO_POLL_MS = 2000;  // background poll after the fallback was shown
  const VIDEO_GRACE_MS = 5000;      // client waits this much longer than the server's own limit
  const LOW_CONFIDENCE = 0.6;
  const WARMUP_POLL_MS = 2000;      // how often to ask the server how far the phrase clips are

  // Overwritten from GET /api/config so the page and the server agree on every timer.
  const cfg = {
    avatar_enabled: false,
    has_concierge: false,
    avatar_timeout_seconds: 180,
    face_idle_warn_seconds: 30,
    face_idle_countdown_seconds: 15,
    face_keepalive_seconds: 20,
    max_face_bytes: 5000000,
  };

  const $ = (id) => document.getElementById(id);
  const el = {
    kiosk: document.querySelector("main.kiosk"),
    form: $("ask-form"), question: $("question"), hint: $("transcript-hint"),
    ask: $("ask-btn"), mic: $("mic-btn"), micLabel: $("mic-label"), clear: $("clear-btn"),
    status: $("status"), panel: $("answer-panel"), text: $("answer-text"),
    citation: $("citation"), audioNote: $("audio-note"), replay: $("replay-btn"),
    player: $("player"),
    face: $("face"), still: $("face-still"), video: $("avatar-video"), faceNote: $("face-note"),
    facePanel: $("face-panel"), consent: $("face-consent"), uploadBtn: $("face-upload-btn"),
    cameraBtn: $("face-camera-btn"), resetBtn: $("face-reset-btn"), file: $("face-file"),
    camera: $("camera"), preview: $("camera-preview"), snap: $("camera-snap"), cancelCam: $("camera-cancel"),
    faceStatus: $("face-status"),
    dialog: $("idle-dialog"), idleDesc: $("idle-desc"), idleStay: $("idle-stay"),
  };

  const sessionId = (() => {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  })();

  let state = "idle";
  let busy = false;
  let warming = false;               // the concierge's phrase clips are still being rendered: input is locked
  let warmupText = "";               // the status line while warming
  let waitingAvatar = false;         // the answer is held while the video renders; Clear may cancel
  let fromVoice = false;
  let recorder = null;
  let recordTimer = null;
  let stream = null;
  let currentAudioUrl = null;
  let currentVideoUrl = null;
  let lastAudioUrl = null;           // server path of this turn's Kokoro audio (the fallback)
  let turnToken = 0;                 // invalidates in-flight polls when a new turn starts or Clear is tapped
  let configReady = Promise.resolve();
  let customFace = false;            // the traveler's own photo is the avatar (server holds it for this session only)
  let customStillUrl = null;         // blob: preview of that photo
  let uploadingFace = false;
  let cameraStream = null;
  let cameraTimer = null;
  let lastActivity = Date.now();
  let lastKeepAlive = 0;
  let warningAt = 0;                 // when the "Still there?" prompt appeared (0 = not showing)
  let focusBeforeWarning = null;

  /* ------------------------------------------------------- state machine */
  const STATE_TEXT = {
    idle: "Ready",
    listening: "Listening... tap Stop when you finish.",
    thinking: "Thinking...",
    speaking: "Speaking...",
    error: "Something went wrong. Please try again.",
  };

  function setState(next, message) {
    state = next;
    el.kiosk.dataset.state = next;
    el.status.textContent = message || (warming && next === "idle" ? warmupText : STATE_TEXT[next]);
    el.status.classList.toggle("error", next === "error");
  }

  function endSpeaking() {
    if (state === "speaking") {
      setState("idle");
      keepAlive(true);                               // the idle clock starts now; so does the photo's
    }
  }

  function setBusy(value) {
    busy = value;
    el.ask.disabled = value || warming;
    el.mic.disabled = warming || (value && !recorder);
    el.clear.disabled = value && !waitingAvatar;
    updateFaceButtons();
  }

  /* ----------------------------------------------------------- warm-up */
  // At server start the default concierge's fixed-phrase videos are rendered (about 10 minutes
  // the first time, then cached). Until they exist the page shows progress and takes no questions.
  function setWarming(value, status) {
    const text = value ? (status ? `Warming up... ${status.done} of ${status.total} ready` : "Warming up...") : "";
    const changed = value !== warming;
    warming = value;
    if (text !== warmupText) {                        // #status is aria-live: only speak when it changes
      warmupText = text;
      if (value) setState("thinking", text);
    }
    if (changed) {
      setBusy(busy);                                  // lock or unlock Ask, Speak and the photo buttons
      if (!value) setState("idle");
    }
  }

  // Ask the server once. Returns true while the clips are still warming.
  async function checkWarmup() {
    let status = null;
    let absent = false;
    try {
      const response = await api("/api/clips/status");
      if (response.status === 404) absent = true;     // avatar off or no clips: nothing to wait for
      else if (response.ok) status = await response.json();
    } catch (_) { /* the server is busy: keep what we last knew */ }
    const stillWarming = !absent && (status ? status.state === "warming" : warming);
    setWarming(stillWarming, status);                 // "ready" and "degraded" both unlock the page
    return stillWarming;
  }

  async function watchWarmup() {
    while (await checkWarmup()) await sleep(WARMUP_POLL_MS);
  }

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  async function api(path, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      return await fetch(path, { ...options, signal: controller.signal });
    } finally {
      clearTimeout(timer);
    }
  }

  async function errorMessage(response, fallback) {
    try {
      const body = await response.json();
      if (body && body.detail && body.detail.message) return body.detail.message;
    } catch (_) { /* not JSON */ }
    return fallback;
  }

  /* -------------------------------------------------------------- config */
  async function loadConfig() {
    try {
      const response = await api("/api/config");
      if (response.ok) Object.assign(cfg, await response.json());
    } catch (_) { /* keep the defaults: the page still works as a voice + text kiosk */ }
    if (cfg.avatar_enabled && cfg.has_concierge) {
      el.still.src = "/api/concierge";
      el.face.hidden = false;
      el.facePanel.hidden = false;
      if (await checkWarmup()) watchWarmup();         // not awaited: the page stays usable for reading
    }
  }

  /* ---------------------------------------------------------------- ask */
  async function ask() {
    await configReady;
    const text = el.question.value.trim();
    if (warming) return;                             // still preparing the concierge's videos
    if (!text || busy) {
      if (!text) setState("error", "Please type or say a question first.");
      return;
    }
    stopPlayback();
    const token = ++turnToken;
    setBusy(true);
    setState("thinking");
    try {
      const response = await api("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          text,
          input_type: fromVoice ? "voice" : "typed",
          language: "en",
        }),
      });
      if (!response.ok) {
        throw new Error(await errorMessage(response, "Sorry, I couldn't answer that."));
      }
      const answer = await response.json();
      if (token !== turnToken) return;               // Clear was tapped while we waited
      lastAudioUrl = answer.audio_url || null;
      if (answer.video_url && cfg.avatar_enabled) {
        await presentWithAvatar(answer, token);
      } else {
        showAnswer(answer);
        setState("idle");
        if (answer.audio_url) fetchAndPlay(answer.audio_url, token);
      }
    } catch (err) {
      if (token === turnToken) {
        setState("error", err.name === "AbortError" ? "That took too long. Please try again." : (err.message || "Something went wrong."));
      }
    } finally {
      fromVoice = false;
      el.hint.hidden = true;
      waitingAvatar = false;
      setBusy(false);
    }
  }

  function showAnswer(answer) {
    el.panel.hidden = false;
    el.panel.className = "panel answer " + answer.status;
    el.text.textContent = answer.text;
    el.audioNote.hidden = true;
    el.replay.hidden = true;
    if (answer.citations && answer.citations.length) {
      const names = [...new Set(answer.citations.map((c) => c.source.split("/").pop()))];
      el.citation.textContent = "Source: " + names.join(", ");
      el.citation.hidden = false;
    } else {
      el.citation.hidden = true;
    }
  }

  /* ------------------------------------------- avatar: hold until ready */
  async function presentWithAvatar(answer, token) {
    waitingAvatar = true;
    setBusy(true);                                   // refresh buttons: Clear stays usable while we wait
    const started = Date.now();
    const deadline = started + cfg.avatar_timeout_seconds * 1000 + VIDEO_GRACE_MS;
    const tick = () => {
      const seconds = Math.round((Date.now() - started) / 1000);
      setState("thinking", seconds >= 2 ? `Preparing the avatar... ${seconds} s` : "Preparing the avatar...");
    };
    tick();
    const ticker = setInterval(tick, 1000);
    let blob = null;
    let settled = false;                             // the server gave a final answer (video or failure)
    try {
      while (token === turnToken && Date.now() < deadline) {
        let response;
        try { response = await api(answer.video_url); } catch (_) { break; }
        if (token !== turnToken) return;
        if (response.status === 202) { await sleep(VIDEO_POLL_MS); continue; }
        settled = true;
        if (response.ok) blob = await response.blob();
        break;
      }
    } finally {
      clearInterval(ticker);
    }
    if (token !== turnToken) return;

    showAnswer(answer);                               // text, voice and video start together
    if (blob) {
      await playVideo(blob, token);
    } else {
      showNote("The avatar could not be shown this time, so you will hear the voice instead.");
      setState("idle");
      if (answer.audio_url) fetchAndPlay(answer.audio_url, token);
      // The render may still finish after our deadline: keep looking, quietly, for a while.
      if (!settled) {
        const giveUpAt = started + Math.max(cfg.avatar_timeout_seconds * 3, 120) * 1000;
        waitForLateVideo(answer.video_url, token, giveUpAt);
      }
    }
  }

  // Background poll after the fallback was shown. Same token guard as the main wait:
  // a new question, Clear or a reset bumps turnToken and this stops without touching the page.
  async function waitForLateVideo(url, token, giveUpAt) {
    while (token === turnToken && Date.now() < giveUpAt) {
      await sleep(LATE_VIDEO_POLL_MS);
      if (token !== turnToken) return;
      let response;
      try { response = await api(url); } catch (_) { continue; }
      if (token !== turnToken) return;
      if (response.status === 202) continue;
      if (!response.ok) return;                       // 404 / 410 / 5xx: no video is coming
      let blob;
      try { blob = await response.blob(); } catch (_) { return; }
      if (token !== turnToken) return;
      offerLateVideo(blob);
      return;
    }
  }

  // Keep the finished video (not auto-played: the Kokoro voice already spoke) until the next
  // turn or reset; stopPlayback() -> releaseVideo() revokes the blob URL then.
  function offerLateVideo(blob) {
    releaseVideo();
    currentVideoUrl = URL.createObjectURL(blob);
    el.video.src = currentVideoUrl;                   // stays hidden until Play answer is tapped
    el.replay.hidden = false;
    const note = "The avatar video is ready. Tap Play answer to watch it.";
    showNote(note);
    if (state === "idle") setState("idle", note);     // #status is aria-live: announce it
  }

  function showNote(message) {
    el.audioNote.textContent = message;
    el.audioNote.hidden = false;
  }

  async function playVideo(blob, token) {
    releaseVideo();
    currentVideoUrl = URL.createObjectURL(blob);
    el.video.src = currentVideoUrl;
    el.video.hidden = false;
    el.replay.hidden = false;
    try {
      await el.video.play();
    } catch (_) {
      if (token === turnToken) { showNote("Tap Play answer to watch and hear it."); setState("idle"); }
    }
  }

  function releaseVideo() {
    el.video.pause();
    el.video.hidden = true;
    el.video.removeAttribute("src");
    el.video.load();
    if (currentVideoUrl) { URL.revokeObjectURL(currentVideoUrl); currentVideoUrl = null; }
  }

  el.video.addEventListener("playing", () => setState("speaking"));
  el.video.addEventListener("ended", () => { el.video.hidden = true; endSpeaking(); });
  el.video.addEventListener("error", () => {
    if (!currentVideoUrl) return;                    // releaseVideo() clears src and fires this; ignore it
    el.video.hidden = true;
    showNote("The avatar video could not be played, so you will hear the voice instead.");
    setState("idle");
    if (lastAudioUrl) fetchAndPlay(lastAudioUrl, turnToken);
  });
  el.player.addEventListener("playing", () => setState("speaking"));
  el.player.addEventListener("ended", endSpeaking);

  /* -------------------------------------------------------------- audio */
  function stopPlayback() {
    turnToken += 1;
    el.player.pause();
    releaseVideo();
    if (currentAudioUrl) { URL.revokeObjectURL(currentAudioUrl); currentAudioUrl = null; }
    el.player.removeAttribute("src");
    el.replay.hidden = true;
    if (state === "speaking") setState("idle");
  }

  async function fetchAndPlay(url, token) {
    const deadline = Date.now() + AUDIO_WAIT_MS;
    while (token === turnToken && Date.now() < deadline) {
      let response;
      try { response = await api(url); } catch (_) { break; }
      if (token !== turnToken) return;
      if (response.status === 202) {
        await sleep(AUDIO_POLL_MS);
        continue;
      }
      if (!response.ok) break;
      const blob = await response.blob();
      if (token !== turnToken) return;
      currentAudioUrl = URL.createObjectURL(blob);
      el.player.src = currentAudioUrl;
      el.replay.hidden = false;
      try {
        await el.player.play();
      } catch (_) {
        showNote("Tap Play answer to hear it.");
      }
      return;
    }
    if (token === turnToken) {
      showNote("Audio is not available right now. The answer is shown above.");
    }
  }

  el.replay.addEventListener("click", () => {
    if (currentVideoUrl) {
      el.player.pause();                              // a late video carries the voice itself
      el.video.hidden = false;
      el.video.currentTime = 0;
      el.video.play().catch(() => {});
    } else {
      el.player.currentTime = 0;
      el.player.play().catch(() => {});
    }
  });

  /* ---------------------------------------------------------- microphone */
  function pickMimeType() {
    const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
    return candidates.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || "";
  }

  async function startRecording() {
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      setState("error", "Voice input isn't available on this device. Please type your question.");
      return;
    }
    stopPlayback();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (_) {
      setState("error", "I can't use the microphone. Please allow access, or type your question.");
      return;
    }
    const mimeType = pickMimeType();
    const chunks = [];
    recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    recorder.onstop = () => {
      const type = recorder.mimeType || mimeType || "audio/webm";
      releaseMic();
      uploadRecording(new Blob(chunks, { type }));
    };
    recorder.start();
    el.mic.setAttribute("aria-pressed", "true");
    el.micLabel.textContent = "Stop";
    el.ask.disabled = true;
    el.clear.disabled = true;
    setState("listening");
    recordTimer = setTimeout(stopRecording, MAX_RECORD_MS);
  }

  function stopRecording() {
    clearTimeout(recordTimer);
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }

  function releaseMic() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    recorder = null;
    el.mic.setAttribute("aria-pressed", "false");
    el.micLabel.textContent = "Speak";
  }

  async function uploadRecording(blob) {
    if (!blob.size) {
      setBusy(false);
      setState("error", "I didn't hear anything. Please try again, or type your question.");
      return;
    }
    setBusy(true);
    setState("thinking", "Working out what you said...");
    try {
      const form = new FormData();
      const ext = blob.type.includes("mp4") ? "mp4" : blob.type.includes("ogg") ? "ogg" : "webm";
      form.append("audio", blob, "question." + ext);
      const response = await api("/api/transcribe", { method: "POST", body: form });
      if (!response.ok) {
        throw new Error(await errorMessage(response, "Sorry, I couldn't process that recording."));
      }
      const result = await response.json();
      el.question.value = result.transcript;
      fromVoice = true;
      el.hint.textContent = result.confidence !== null && result.confidence < LOW_CONFIDENCE
        ? "I'm not sure I heard that right. Please check the words, fix anything wrong, then tap Ask."
        : "Please check the words, fix anything wrong, then tap Ask.";
      el.hint.hidden = false;
      setState("idle", "Check your question, then tap Ask.");
      el.question.focus();
    } catch (err) {
      setState("error", err.name === "AbortError" ? "That took too long. Please try again." : (err.message || "Something went wrong."));
    } finally {
      setBusy(false);
    }
  }

  el.mic.addEventListener("click", () => {
    if (recorder) stopRecording();
    else if (!busy && !warming) startRecording();
  });

  /* ------------------------------------------------------------ wiring */
  el.form.addEventListener("submit", (e) => { e.preventDefault(); ask(); });

  el.question.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
  });
  // Typing over a transcript is still the traveler's own words; keep the hint honest.
  el.question.addEventListener("input", () => { el.hint.hidden = true; });

  el.clear.addEventListener("click", () => {
    stopPlayback();                                   // also cancels a pending avatar wait
    waitingAvatar = false;
    setBusy(false);
    clearConversation();
    removeFace();                                     // Clear also removes the traveler's photo
    setState("idle");
    el.question.focus();
  });

  function clearConversation() {
    el.question.value = "";
    el.hint.hidden = true;
    el.panel.hidden = true;
    fromVoice = false;
  }

  if (!navigator.mediaDevices || !window.MediaRecorder) {
    el.mic.disabled = true;
    el.micLabel.textContent = "Speak (unavailable)";
  }

  /* ---------------------------------------------------------- own face */
  function faceMessage(message, isError = false) {
    el.faceStatus.textContent = message;
    el.faceStatus.classList.toggle("error", isError);
  }

  function updateFaceButtons() {
    const blocked = busy || warming || uploadingFace || !!recorder;
    el.uploadBtn.disabled = blocked;
    el.cameraBtn.disabled = blocked || !!cameraStream;
    el.resetBtn.disabled = blocked;
  }

  function consentGiven() {
    if (el.consent.checked) return true;
    faceMessage("Please tick the box to agree to use your photo first.", true);
    el.consent.focus();
    return false;
  }

  function setCustomFace(blob) {
    if (customStillUrl) URL.revokeObjectURL(customStillUrl);
    customStillUrl = URL.createObjectURL(blob);
    el.still.src = customStillUrl;
    customFace = true;
    el.resetBtn.hidden = false;
  }

  function clearFaceLocal() {
    customFace = false;
    if (customStillUrl) { URL.revokeObjectURL(customStillUrl); customStillUrl = null; }
    if (cfg.avatar_enabled && cfg.has_concierge) el.still.src = "/api/concierge";
    el.resetBtn.hidden = true;
    el.consent.checked = false;
    faceMessage("");
  }

  async function removeFace() {
    if (!customFace) return;
    const wasCustom = customFace;
    clearFaceLocal();                                 // the page stops showing it at once
    try {
      await api("/api/face", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch (_) { /* the server deletes it on its own after a minute */ }
    return wasCustom;
  }

  async function sendPhoto(blob, filename) {
    if (blob.size > cfg.max_face_bytes) {
      faceMessage(`That photo is too large. Please use one under ${Math.round(cfg.max_face_bytes / 1048576)} MB.`, true);
      return;
    }
    uploadingFace = true;
    updateFaceButtons();
    faceMessage("Checking your photo... this can take a moment.");
    setState("thinking", "Checking your photo...");
    try {
      const form = new FormData();
      form.append("image", blob, filename);
      form.append("session_id", sessionId);
      form.append("consent", "true");
      const response = await api("/api/face", { method: "POST", body: form });
      if (!response.ok) throw new Error(await errorMessage(response, "Sorry, I couldn't use that photo."));
      setCustomFace(blob);
      lastKeepAlive = Date.now();
      faceMessage("Done. Your photo is now the avatar. It is removed when you leave.");
      setState("idle");
    } catch (err) {
      const message = err.name === "AbortError" ? "That took too long. Please try again." : (err.message || "Something went wrong with that photo.");
      faceMessage(message, true);
      setState("error", message);
    } finally {
      uploadingFace = false;
      updateFaceButtons();
      markActivity();
    }
  }

  el.uploadBtn.addEventListener("click", () => {
    if (consentGiven()) el.file.click();
  });
  el.file.addEventListener("change", () => {
    const file = el.file.files && el.file.files[0];
    el.file.value = "";
    if (!file) return;
    if (!consentGiven()) return;
    sendPhoto(file, file.name);
  });
  el.resetBtn.addEventListener("click", async () => {
    await removeFace();
    faceMessage("Back to the default concierge.");
    markActivity();
  });

  async function openCamera() {
    if (!consentGiven()) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      faceMessage("This device has no camera I can use. Please upload a photo instead.", true);
      return;
    }
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    } catch (_) {
      faceMessage("I can't use the camera. Please allow access, or upload a photo instead.", true);
      return;
    }
    el.preview.srcObject = cameraStream;
    el.camera.hidden = false;
    faceMessage("Look at the camera, then tap Capture.");
    updateFaceButtons();
    el.snap.focus();
    cameraTimer = setTimeout(closeCamera, 60000);     // never leave the camera on unattended
  }

  function closeCamera() {
    clearTimeout(cameraTimer);
    if (cameraStream) cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
    el.preview.srcObject = null;
    el.camera.hidden = true;
    updateFaceButtons();
  }

  function capturePhoto() {
    const width = el.preview.videoWidth;
    const height = el.preview.videoHeight;
    if (!width || !height) { faceMessage("The camera isn't ready yet. Please try again.", true); return; }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(el.preview, 0, 0, width, height);
    closeCamera();
    canvas.toBlob((blob) => {
      if (blob) sendPhoto(blob, "camera.jpg");
      else faceMessage("I couldn't take that photo. Please try again.", true);
    }, "image/jpeg", 0.92);
  }

  el.cameraBtn.addEventListener("click", openCamera);
  el.snap.addEventListener("click", capturePhoto);
  el.cancelCam.addEventListener("click", () => { closeCamera(); faceMessage(""); });

  // Leaving or reloading the page must not strand the photo on the server.
  window.addEventListener("pagehide", () => {
    if (!customFace) return;
    try {
      fetch("/api/face", {
        method: "DELETE",
        keepalive: true,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch (_) { /* best effort */ }
  });

  /* ------------------------------------- activity, keep-alive and "Still there?" */
  function keepAlive(force = false) {
    if (!customFace) return;
    const now = Date.now();
    if (!force && now - lastKeepAlive < cfg.face_keepalive_seconds * 1000) return;
    lastKeepAlive = now;
    api("/api/face/keepalive", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    }).catch(() => {});
  }

  function markActivity() {
    lastActivity = Date.now();
    keepAlive();
  }

  function conversationShowing() {
    return !el.panel.hidden || el.question.value.trim() !== "";
  }

  function showWarning() {
    warningAt = Date.now();
    focusBeforeWarning = document.activeElement;
    updateWarningText();
    el.dialog.hidden = false;
    el.idleStay.focus();
  }

  function updateWarningText() {
    const left = Math.max(0, Math.ceil((cfg.face_idle_countdown_seconds * 1000 - (Date.now() - warningAt)) / 1000));
    const what = customFace ? "your photo will be removed and the kiosk will reset" : "the screen will be cleared";
    el.idleDesc.textContent = `Tap the button if you are still using this kiosk. Otherwise ${what} in ${left} seconds.`;
  }

  function dismissWarning(stay) {
    if (!warningAt) return;
    warningAt = 0;
    el.dialog.hidden = true;
    lastActivity = Date.now();
    if (focusBeforeWarning && focusBeforeWarning.focus) focusBeforeWarning.focus();
    focusBeforeWarning = null;
    if (stay) keepAlive(true);
  }

  async function idleReset() {
    warningAt = 0;
    el.dialog.hidden = true;
    focusBeforeWarning = null;
    stopPlayback();
    closeCamera();
    clearConversation();
    await removeFace();
    lastActivity = Date.now();
    setState("idle", "Reset. Ready for the next traveler.");
  }

  function idleTick() {
    const quiet = (state === "idle" || state === "error") && !busy && !recorder && !uploadingFace && !cameraStream;
    const now = Date.now();
    if (!quiet) {                                     // thinking, speaking, listening...: not idle
      if (warningAt) dismissWarning(false);
      lastActivity = now;
      return;
    }
    if (warningAt) {
      if (now - warningAt >= cfg.face_idle_countdown_seconds * 1000) idleReset();
      else updateWarningText();
      return;
    }
    if (now - lastActivity >= cfg.face_idle_warn_seconds * 1000 && (customFace || conversationShowing())) {
      showWarning();
    }
  }

  for (const type of ["pointerdown", "keydown", "touchstart"]) {
    document.addEventListener(type, (event) => {
      if (warningAt) {
        // Any interaction means "I'm still here"; Tab stays inside the prompt.
        if (type === "keydown" && event.key === "Tab") { event.preventDefault(); el.idleStay.focus(); }
        dismissWarning(true);
      } else {
        markActivity();
      }
    }, true);
  }
  el.idleStay.addEventListener("click", () => dismissWarning(true));
  setInterval(idleTick, 500);

  configReady = loadConfig();
})();
