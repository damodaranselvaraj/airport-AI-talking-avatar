"""HTTP API for the kiosk UI. No secret or exception text ever appears in a response.

POST /api/transcribe  audio -> editable transcript (never calls the workflow)
POST /api/ask         text  -> approved answer + citation (+ audio_url to poll)
GET  /api/audio/{id}  202 pending | 200 audio/wav | 410 failed | 404 unknown
GET  /api/video/{id}  202 pending | 200 video/mp4 | 410 failed | 404 unknown
GET  /api/config      timer values and feature flags the page needs (no secrets)
GET  /api/concierge   the default concierge image (404 when the avatar is off)
GET  /api/clips/status  warm-up of the pre-rendered phrase clips (404 when there are none)

LangSmith tracing
-----------------
When LANGSMITH_TRACING=true each request produces one parent trace:

  ask_turn        (ask endpoint)
  ├── [LangGraph nodes – auto-traced by langchain]
  │   ├── guardrail / supervisor (gpt-4o-mini) / rag / answer (gpt-4o) / finalize
  ├── kokoro_tts  (child span, background task)
  └── ditto_render (child span, avatar-dispatch callback thread)

  transcribe_turn (transcribe endpoint)
  └── whisper_stt (child span)

All tracing helpers are no-ops when tracing is disabled, so the kiosk behaves
identically regardless of the LANGSMITH_TRACING setting.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.audio_store import AudioState
from app.avatar.models import DEFAULT_FACE_KEY, FaceRef, JobOutcome, Priority, QueueFull
from app.services import Services
from app.speech_to_text.errors import (
    AudioTooLargeError,
    EmptyAudioError,
    TranscriptionError,
)
from app.tracing import child_span, close_child_span, open_child_span, turn_trace
from app.workflow.contracts import (
    Citation,
    EvidenceTier,
    InputType,
    KioskRequest,
    ResponseStatus,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

_UPLOAD_CHUNK = 64 * 1024
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"


class AskBody(BaseModel):
    session_id: str = Field(..., pattern=SESSION_ID_PATTERN)
    text: str = Field(..., min_length=1, max_length=2000)
    input_type: InputType = InputType.TYPED
    language: str = Field("en", pattern=r"^[a-z]{2}(-[A-Za-z]{2})?$")

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class AskResponse(BaseModel):
    turn_id: str
    status: ResponseStatus
    text: str
    citations: list[Citation]
    spoken: bool
    evidence_tier: Optional[EvidenceTier] = None
    language: str
    audio_url: Optional[str] = None
    video_url: Optional[str] = None


class ClientConfig(BaseModel):
    """Everything the page needs to match the server's timers. Contains no secrets or paths."""

    avatar_enabled: bool
    has_concierge: bool
    avatar_timeout_seconds: float
    face_idle_warn_seconds: int
    face_idle_countdown_seconds: int
    face_keepalive_seconds: int
    max_face_bytes: int


class TranscribeResponse(BaseModel):
    transcript: str
    duration_s: float
    confidence: Optional[float] = None
    language: str


def _services(request: Request) -> Services:
    return request.app.state.services


def error_body(code: str, message: str) -> dict:
    return {"detail": {"code": code, "message": message}}


def render_audio(
    services: Services,
    turn_id: str,
    text: str,
    cached_wav: Optional[Path] = None,
    parent_run: Optional[Any] = None,
) -> bool:
    """Synthesize the approved text (or reuse its cached audio). Failure degrades to captions only.

    When ``parent_run`` is a live RunTree, a ``kokoro_tts`` child span is posted
    to LangSmith with text length, voice, and output audio size.
    """
    try:
        if cached_wav is not None and cached_wav.is_file():
            # Fixed phrase: Kokoro already spoke it; just copy the bytes.
            wav = cached_wav.read_bytes()
        else:
            # Live synthesis: record it as a child span.
            with child_span(
                parent_run,
                "kokoro_tts",
                "tool",
                {
                    "text_chars": len(text),
                    "voice": services.settings.kokoro_voice,
                    "lang": services.settings.kokoro_lang,
                },
            ) as span:
                wav = services.synthesizer.synthesize(text)  # type: ignore[union-attr]
                if span is not None:
                    try:
                        span.end(outputs={"audio_bytes": len(wav), "source": "live_synthesis"})
                    except Exception:
                        pass
        services.audio.put(turn_id, wav)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("tts failed for turn: %s", type(exc).__name__)
        services.audio.fail(turn_id)
        return False


def submit_video(
    services: Services,
    turn_id: str,
    face: FaceRef,
    parent_run: Optional[Any] = None,
) -> None:
    """Hand the turn's spoken audio and the face to Ditto. Any failure degrades to audio only.

    When ``parent_run`` is a live RunTree, a ``ditto_render`` child span is opened
    before the render job is queued and closed (from the avatar-dispatch callback
    thread) when the job finishes or fails.
    """
    video, avatar = services.video, services.avatar
    assert video is not None and avatar is not None
    _, wav = services.audio.get(turn_id)
    if wav is None or not wav.is_file():
        video.fail(turn_id)
        release_face(services, face)
        close_child_span(
            open_child_span(
                parent_run,
                "ditto_render",
                "tool",
                {"face_key": face.key, "audio_bytes": 0,
                 "deadline_s": services.settings.avatar_timeout_seconds},
            ),
            outputs={"status": "no_audio", "render_ms": None, "error_code": "no_audio"},
        )
        return

    audio_bytes = wav.stat().st_size if wav.is_file() else 0
    render_span = open_child_span(
        parent_run,
        "ditto_render",
        "tool",
        {
            "face_key": face.key,
            "audio_bytes": audio_bytes,
            "deadline_s": services.settings.avatar_timeout_seconds,
        },
    )
    render_started = time.perf_counter()

    def finished(outcome: JobOutcome) -> None:
        render_ms = round((time.perf_counter() - render_started) * 1000, 1)
        if not outcome.ok:
            logger.info("avatar video not produced: %s", outcome.status.value)
        video.finish(turn_id, outcome.ok)
        release_face(services, face)
        close_child_span(
            render_span,
            outputs={
                "status": outcome.status.value,
                "render_ms": render_ms,
                "error_code": outcome.error_code,
            },
        )

    try:
        avatar.render(
            turn_id,
            face.path,
            wav,
            video.path_for(turn_id),
            priority=Priority.LIVE,
            face_key=face.key,
            deadline_s=services.settings.avatar_timeout_seconds,
            on_done=finished,
        )
    except (QueueFull, RuntimeError, ValueError) as exc:
        logger.warning("avatar render not queued: %s", type(exc).__name__)
        video.fail(turn_id)
        release_face(services, face)
        close_child_span(
            render_span,
            outputs={
                "status": "not_queued",
                "render_ms": None,
                "error_code": type(exc).__name__,
            },
        )


def render_turn(
    services: Services,
    turn_id: str,
    text: str,
    face: Optional[FaceRef],
    cached_wav: Optional[Path] = None,
    parent_run: Optional[Any] = None,
) -> None:
    """Background task: speak the approved text, then (if wanted) animate the face to it.

    ``parent_run`` is the RunTree from the ask_turn trace (may be None when tracing
    is off). Passed through to render_audio and submit_video so their child spans
    attach to the correct parent via create_child().
    """
    spoken = render_audio(services, turn_id, text, cached_wav, parent_run)
    if face is None or services.video is None:
        return
    if not spoken:
        services.video.fail(turn_id)
        release_face(services, face)
        return
    submit_video(services, turn_id, face, parent_run)


def release_face(services: Services, face: FaceRef) -> None:
    """A turn using ``face`` is over: the traveler photo's idle clock restarts from now."""
    if services.faces is not None and face.key != DEFAULT_FACE_KEY:
        services.faces.turn_finished(face.key)


def concierge_face(services: Services) -> Optional[FaceRef]:
    if services.default_face is None:
        return None
    return FaceRef(path=services.default_face, key=DEFAULT_FACE_KEY)


def is_fixed_phrase(services: Services, status: ResponseStatus, text: str) -> bool:
    """Greetings, policy refusals, the safe fallback and guardrail blocks: approved wording, not an answer."""
    if status is not ResponseStatus.ANSWERED:
        return True
    return services.clips is not None and services.clips.is_phrase(text)


def face_for_turn(services: Services, session_id: str, status: ResponseStatus, text: str) -> Optional[FaceRef]:
    """The face that speaks this turn.

    Fixed phrases always use the default concierge (pre-rendered, or rendered live if the clip is
    missing), even when the traveler added a photo. Only a grounded answer uses the traveler's own
    photo; without one it uses the concierge too.
    """
    if services.avatar is None or services.video is None:
        return None
    if not is_fixed_phrase(services, status, text) and services.faces is not None:
        custom = services.faces.get(session_id)
        if custom is not None:
            return custom
    return concierge_face(services)


@router.post("/ask", response_model=AskResponse)
def ask(body: AskBody, request: Request, background: BackgroundTasks) -> AskResponse:
    services = _services(request)
    if services.faces is not None:
        services.faces.touch(body.session_id)  # asking is activity: keep the traveler's photo alive
    if len(body.text) > services.settings.max_question_chars:
        raise HTTPException(status_code=422, detail={"code": "too_long", "message": "Question is too long."})

    # ------------------------------------------------------------------ tracing
    # Open the top-level ask_turn trace. LangGraph/LangChain calls made inside
    # the with-block auto-attach as children via the ambient LangSmith context.
    # The RunTree stays valid after the context exits (id, trace_id, dotted_order
    # remain in memory). We capture it inside the block and pass it to the
    # background task so render_audio and submit_video can call create_child(),
    # which correctly inherits the parent's dotted_order for child-run validation.
    parent_run: Optional[Any] = None
    with turn_trace(
        "ask_turn",
        inputs={
            "session_id": body.session_id,
            "question_chars": len(body.text),
            "input_type": body.input_type.value,
            "language": body.language,
        },
    ) as run:
        # ---------------------------------------------------------- workflow
        outcome = services.workflow.run(
            KioskRequest(
                session_id=body.session_id,
                text=body.text,
                input_type=body.input_type,
                language=body.language,
            )
        )
        # Record workflow-level metadata on the trace (best-effort).
        if run is not None:
            try:
                run.add_metadata({
                    "route": outcome.trace.route.value if outcome.trace.route else None,
                    "evidence_tier": (
                        outcome.trace.evidence_tier.value
                        if outcome.trace.evidence_tier else None
                    ),
                    "status": outcome.result.status.value,
                    "workflow_latency_ms": outcome.trace.latency_ms,
                    "guardrail_reason": outcome.trace.guardrail_reason,
                    "error_stage": outcome.trace.error_stage,
                })
            except Exception:
                pass
        # Capture INSIDE the with-block so the RunTree is available for
        # background task child spans after the context exits.
        parent_run = run

    # ------------------------------------------------------------------ media
    result = outcome.result
    turn_id = services.audio.new_turn_id()
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    media = "text_only"
    reason = "not_spoken"
    if result.spoken and services.synthesizer is not None:
        services.audio.begin(turn_id)
        audio_url = f"/api/audio/{turn_id}"
        face = face_for_turn(services, body.session_id, result.status, result.text)
        cached_wav = services.clips.cached_wav(result.text) if services.clips is not None else None
        inline_audio = False
        media, reason = "audio_only", "no_avatar"
        if face is not None:
            clip = services.clips.lookup(face, result.text) if services.clips is not None else None
            if clip is not None:
                # A fixed phrase for this face: its video already exists, so serve it now,
                # together with its cached audio (a file copy), so text, voice and video are ready at once.
                services.video.ready_from_shared(turn_id, clip)  # type: ignore[union-attr]
                video_url = f"/api/video/{turn_id}"
                face = None
                media, reason = "cached_clip", "ok"
                inline_audio = cached_wav is not None
            elif result.status is ResponseStatus.GREETING:
                # Never run live inference for a greeting: reveal text and voice without video.
                face = None
                media, reason = "audio_only", "greeting_clip_missing"
                logger.warning("greeting clip missing; revealing text and voice without video")
            elif not services.avatar.ready:  # type: ignore[union-attr]
                face = None
                media, reason = "audio_only", "worker_not_ready"
                logger.info("avatar worker not ready; revealing text and voice without video")
                restart = getattr(services.avatar, "start", None)
                if restart is not None:
                    try:
                        restart()  # non-blocking: only spawns a worker when none is alive
                        logger.info("avatar worker restart requested")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("avatar worker restart failed: %s", type(exc).__name__)
            else:
                services.video.begin(turn_id)  # type: ignore[union-attr]
                if face.key != DEFAULT_FACE_KEY and services.faces is not None:
                    services.faces.turn_started(face.key)  # pause the photo's expiry while it renders
                video_url = f"/api/video/{turn_id}"
                media, reason = "live_render", "ok"
        if inline_audio:
            # Cached clip audio — no TTS span needed.
            render_audio(services, turn_id, result.text, cached_wav)
        else:
            # Pass the RunTree so TTS and Ditto render attach as children via create_child().
            background.add_task(
                render_turn, services, turn_id, result.text, face, cached_wav, parent_run
            )
    logger.info("ask turn=%s media=%s reason=%s", turn_id, media, reason)
    return AskResponse(
        turn_id=turn_id,
        status=result.status,
        text=result.text,
        citations=result.citations,
        spoken=result.spoken,
        evidence_tier=result.evidence_tier,
        language=result.language,
        audio_url=audio_url,
        video_url=video_url,
    )


@router.get("/config", response_model=ClientConfig)
def client_config(request: Request) -> ClientConfig:
    services = _services(request)
    s = services.settings
    return ClientConfig(
        avatar_enabled=services.avatar is not None,
        has_concierge=services.default_face is not None,
        avatar_timeout_seconds=s.avatar_timeout_seconds,
        face_idle_warn_seconds=s.face_idle_warn_seconds,
        face_idle_countdown_seconds=s.face_idle_countdown_seconds,
        # Often enough that a photo in use never reaches the server backstop.
        face_keepalive_seconds=max(int(s.face_ttl_backstop_seconds // 3), 1),
        max_face_bytes=s.max_face_bytes,
    )


@router.get("/clips/status")
def clips_status(request: Request):
    """Warm-up of the concierge's pre-rendered phrase clips: ``{state, done, total}``.

    ``state`` is ``warming`` (the kiosk waits), ``ready`` or ``degraded`` (some clips missing and
    nothing rendering: they render live instead). 404 when the avatar or clips are off.
    """
    status = _services(request).clips_status()
    if status is None:
        return JSONResponse(error_body("not_found", "No pre-rendered clips."), status_code=404)
    return JSONResponse(status, headers={"Cache-Control": "no-store"})


@router.get("/concierge")
def concierge(request: Request):
    services = _services(request)
    path = services.default_face
    if services.avatar is None or path is None or not path.is_file():
        return JSONResponse(error_body("not_found", "No concierge image."), status_code=404)
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-cache"})


@router.get("/audio/{turn_id}")
def audio(turn_id: str, request: Request):
    state, path = _services(request).audio.get(turn_id)
    headers = {"Cache-Control": "no-store"}
    if state is None:
        return JSONResponse(error_body("not_found", "Audio not found."), status_code=404, headers=headers)
    if state is AudioState.PENDING:
        return JSONResponse({"status": "pending"}, status_code=202, headers={**headers, "Retry-After": "1"})
    if state is AudioState.FAILED or path is None or not path.is_file():
        return JSONResponse(error_body("audio_failed", "Audio is unavailable."), status_code=410, headers=headers)
    return FileResponse(path, media_type="audio/wav", headers=headers)


@router.get("/video/{turn_id}")
def video(turn_id: str, request: Request):
    store = _services(request).video
    headers = {"Cache-Control": "no-store"}
    if store is None:
        return JSONResponse(error_body("not_found", "Video not found."), status_code=404, headers=headers)
    state, path = store.get(turn_id)
    if state is None:
        return JSONResponse(error_body("not_found", "Video not found."), status_code=404, headers=headers)
    if state is AudioState.PENDING:
        return JSONResponse({"status": "pending"}, status_code=202, headers={**headers, "Retry-After": "1"})
    if state is AudioState.FAILED or path is None or not path.is_file():
        return JSONResponse(error_body("video_failed", "Video is unavailable."), status_code=410, headers=headers)
    return FileResponse(path, media_type="video/mp4", headers=headers)


async def read_capped(upload: UploadFile, limit: int, error=AudioTooLargeError) -> bytes:
    """Read an upload but stop as soon as it exceeds ``limit`` bytes."""
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await upload.read(_UPLOAD_CHUNK)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise error()
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: Request, audio: UploadFile = File(...)):
    services = _services(request)
    if services.transcriber is None:
        return JSONResponse(error_body("stt_unavailable", "Voice input is unavailable."), status_code=503)

    # Open the transcribe_turn parent trace. The with-block wraps the entire
    # handler so the trace captures both success and error outcomes.
    with turn_trace(
        "transcribe_turn",
        inputs={"language": services.settings.default_language},
    ) as run:
        data = b""
        transcript = None
        outcome = "error"
        try:
            data = await read_capped(audio, services.settings.max_audio_bytes)
            if not data:
                raise EmptyAudioError()

            # Decoding + inference are CPU-bound: keep them off the event loop.
            from starlette.concurrency import run_in_threadpool

            with child_span(
                run,
                "whisper_stt",
                "tool",
                {"audio_bytes": len(data), "language": services.settings.default_language},
            ) as stt_span:
                transcript = await run_in_threadpool(services.transcriber.transcribe, data)
                if stt_span is not None:
                    try:
                        stt_span.end(outputs={
                            "transcript_chars": len(transcript.text),
                            "duration_s": transcript.duration_s,
                            "confidence": transcript.confidence,
                            "language": transcript.language,
                        })
                    except Exception:
                        pass

            outcome = "success"

        except TranscriptionError as exc:
            if run is not None:
                try:
                    run.add_metadata({"outcome": "transcription_error", "error_code": exc.code})
                except Exception:
                    pass
            return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status_code)
        except Exception as exc:  # noqa: BLE001
            logger.error("transcription crashed: %s", type(exc).__name__)
            if run is not None:
                try:
                    run.add_metadata({"outcome": "crashed", "error_type": type(exc).__name__})
                except Exception:
                    pass
            return JSONResponse(
                error_body("stt_failed", "Sorry, I couldn't process that recording."),
                status_code=500,
            )
        finally:
            await audio.close()

        # Attach final metadata to the trace before the context manager closes it.
        if run is not None and transcript is not None:
            try:
                run.add_metadata({
                    "outcome": outcome,
                    "audio_bytes": len(data),
                    "transcript_chars": len(transcript.text),
                    "duration_s": transcript.duration_s,
                    "confidence": transcript.confidence,
                })
            except Exception:
                pass

        return TranscribeResponse(
            transcript=transcript.text,
            duration_s=transcript.duration_s,
            confidence=transcript.confidence,
            language=transcript.language,
        )
