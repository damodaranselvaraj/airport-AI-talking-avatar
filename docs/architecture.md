# Architecture

The authoritative architecture overview and diagram live in the root
[`README.md`](../README.md). This document is expanded in Task 18 with
component-level detail, sequence diagrams, and the per-turn data flow.

## Components (independent before wiring)

| Component | Module | Responsibility |
|---|---|---|
| Kiosk UI | `app/ui/` | Text + mic input, avatar display, conversation state |
| Orchestrator | `app/workflow/` | LangGraph graph, routing, retries, timeouts, contracts |
| Retrieval | `app/retrieval/` | Offline ingestion + `search_airport` |
| Speech-to-Text | `app/speech_to_text/` | Whisper transcription (CPU) |
| Text-to-Speech | `app/text_to_speech/` | Kokoro synthesis (CPU) |
| Avatar | `app/avatar/` | Ditto talking-head generation (GPU) |

Data contracts between components are defined in
[`app/workflow/contracts.py`](../app/workflow/contracts.py).

## Stage 1-2 runtime (as built)

```text
Browser (app/ui)                 FastAPI (app/main.py, app/api.py)
-----------------                -----------------------------------------------
Speak  -> MediaRecorder -------> POST /api/transcribe -> WhisperTranscriber (CPU, in memory)
          editable transcript <-  {transcript, confidence}      (workflow NOT called)
Ask    ------------------------> POST /api/ask -> KioskWorkflow (LangGraph)
          text + citation     <-                    route -> greeting | deny | retrieve
          (immediately)                             retrieve -> assess -> generate -> validate
                                                    (medium/weak or invalid -> safe fallback)
       -> background task: KokoroSynthesizer (CPU, lock) -> sessions/<turn>.wav (TTL)
Play   <- GET /api/audio/{turn_id}  202 pending | 200 wav | 410 failed
```

| Module | Role |
|---|---|
| `app/main.py` | `create_app(services)` factory; fail-loud startup; security headers; static UI last |
| `app/services.py` | `Services` container + `build_services()` (validate stores, load models) |
| `app/api.py` | `/api/ask`, `/api/transcribe`, `/api/audio/{id}`; no exception text in responses |
| `app/workflow/routing.py` | Rule-based router + approved wording (`config/policy.json`, `config/responses.json`) |
| `app/workflow/evidence.py` | strong/medium/weak from top rerank score (`config/evidence_thresholds.json`) |
| `app/workflow/answer.py` | gpt-4o JSON-mode grounded answer, temperature 0 |
| `app/workflow/validation.py` | `Validator` interface + `CitationOverlapValidator` (Jev slot) |
| `app/workflow/graph.py` | LangGraph graph, per-node timeouts, errors -> safe text-only result, `TurnTrace` |
| `app/retrieval/search.py` | `search_airport` adapter: passages + "was rerank applied?" |
| `app/speech_to_text/whisper_service.py` | faster-whisper `small.en` int8; PyAV decode in memory |
| `app/text_to_speech/kokoro_service.py` | Kokoro-82M pinned to CPU; `speakable()` clean-up |
| `app/audio_store.py` | Per-turn WAVs, server-generated IDs, TTL sweep, purge on start/stop |

### Failure behaviour

| Failure | Result |
|---|---|
| Retriever / LLM error or timeout | `ResponseStatus.ERROR`, safe text, not spoken |
| Cohere rerank down | fused order used, evidence capped at MEDIUM -> fallback |
| Answer fails validation | discarded; safe fallback spoken instead |
| Kokoro fails | caption still shown, audio endpoint returns 410 |
| Whisper: silence / undecodable / too long / too big | typed error code + friendly message, user can type |

## Stage 3 runtime (talking avatar)

```text
Browser                              FastAPI (.venv, Python 3.12)             Ditto worker (conda env "ditto", GPU)
-------                              ----------------------------             -------------------------------------
Ask -------------------------------> POST /api/ask -> workflow (as above)
       {text, audio_url, video_url} <-
       state=thinking, text+voice HELD
                                     background: Kokoro WAV (or cached WAV)
                                       fixed phrase + clip cached? -> video ready at once
                                       else AvatarService.render(LIVE) --queue--> JSON line {"op":"render"} ---> StreamSDK (warm)
Poll   ---------------------------> GET /api/video/{turn}  202 ...                  <--- {"ok":true} + mp4 (voice muxed in)
       200 video/mp4 <-------------  200 (or 410 -> text + Kokoro voice fallback)
reveal text + play video together; state=speaking -> idle
```

| Module | Role |
|---|---|
| `scripts/ditto_worker.py` | Warm worker: loads `StreamSDK` once; `render` and `check_face` jobs over JSON lines; Ditto's prints go to stderr |
| `app/avatar/protocol.py` | Stdlib-only protocol shared by the app and the worker |
| `app/avatar/worker_client.py` | Spawns/stops the worker, one job at a time, kills a hung one |
| `app/avatar/queue.py` | Priority queue (LIVE > FACE_CHECK > BACKGROUND), deadlines, cancel-by-face |
| `app/avatar/service.py` | `AvatarService`: dispatcher thread, restart with back-off, `render`, `check_face`, `cancel_face` |
| `app/avatar/faces.py` | Session-only traveler photos: random names, keep-alive, 60 s backstop, purge at start/stop |
| `app/avatar/images.py` | Decode, auto-rotate, drop metadata, re-encode to PNG, size limits |
| `app/avatar/clips.py` | Fixed-phrase clip cache + background renderer (concierge: kept; traveler: deleted with the photo) |
| `app/video_store.py` | Per-turn MP4s (TTL), plus shared cached clips that are served but never deleted |
| `app/api_faces.py` | `POST/DELETE /api/face`, `POST /api/face/keepalive` |
| `app/ui/kiosk.js` | State machine (idle / listening / thinking / speaking / error), hold-until-ready, face picker, "Still there?" |

### Stage 3 failure behaviour

| Failure | Result |
|---|---|
| Worker not running / not ready | Video request resolves to 410 at once; text + Kokoro voice shown with a notice; the worker restarts with back-off |
| Worker dies or hangs mid-job | That turn falls back (410); a hung job is killed after a hard timeout; next turn restarts it |
| Video slower than `AVATAR_TIMEOUT_SECONDS` | Page falls back to text + Kokoro voice over the still face |
| Queue full | Same fallback for that turn |
| Photo has no face / is unusable / too large | Refused with a plain message; nothing is kept |
| Photo deleted while its render is queued | The queued jobs are cancelled; the turn falls back |
| Server killed with photos on disk | Purged at the next start |
### Environment pins that matter

- `kokoro==0.9.4` with `misaki==0.9.4` (kokoro 0.3.4 fails to import against misaki 0.9).
- `av==12.3.0`: faster-whisper 1.1.0 calls `av.open(metadata_errors=...)`, removed in PyAV 19.
  Without the pin every upload is "undecodable".
- torch in `.venv` is the CPU build, so Kokoro cannot touch the GPU; Ditto keeps it.
