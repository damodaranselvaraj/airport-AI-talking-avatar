"""Service container: everything the HTTP layer needs, built once at startup.

Tests build a ``Services`` from fakes and pass it to ``create_app``; production
uses ``build_services`` which validates configuration, loads the models, and
fails loudly if anything is wrong.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Protocol

from app.audio_store import AudioStore
from app.avatar.clips import ClipCache, ClipRenderer
from app.avatar.faces import FaceStore
from app.avatar.models import DEFAULT_FACE_KEY, FaceCheck, FaceRef, JobOutcome, Priority
from app.security import secret_values
from app.video_store import VideoStore
from app.workflow.contracts import KioskRequest, Transcript, TurnOutcome

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)


class Workflow(Protocol):
    evidence_calibrated: bool

    def run(self, request: KioskRequest) -> TurnOutcome: ...


class Transcriber(Protocol):
    ready: bool

    def transcribe(self, data: bytes) -> Transcript: ...


class Synthesizer(Protocol):
    ready: bool

    def synthesize(self, text: str) -> bytes: ...


class AvatarRenderer(Protocol):
    """What the HTTP layer needs from the Ditto worker (real service or test fake)."""

    ready: bool

    def render(
        self,
        job_id: str,
        source: Path,
        audio: Path,
        output: Path,
        *,
        priority: Priority = Priority.LIVE,
        face_key: Optional[str] = None,
        deadline_s: float = 90.0,
        on_done: Callable[[JobOutcome], None],
    ) -> None: ...

    def check_face(self, source: Path, *, timeout_s: float = 30.0) -> FaceCheck: ...

    def cancel_face(self, face_key: str) -> int: ...

    def stop(self) -> None: ...


@dataclass
class Services:
    settings: "Settings"
    workflow: Workflow
    audio: AudioStore
    transcriber: Optional[Transcriber] = None
    synthesizer: Optional[Synthesizer] = None
    avatar: Optional[AvatarRenderer] = None
    video: Optional[VideoStore] = None
    default_face: Optional[Path] = None  # the concierge image; None means no video
    faces: Optional[FaceStore] = None   # traveler photos, session only
    clips: Optional[ClipCache] = None   # pre-rendered fixed-phrase videos
    clip_renderer: Optional[ClipRenderer] = None

    def __post_init__(self) -> None:
        if self.avatar is not None and self.video is None:
            self.video = VideoStore(self.settings.sessions_dir, self.settings.video_ttl_seconds)
        if self.avatar is not None and self.faces is None:
            self.faces = FaceStore(
                self.settings.faces_dir,
                self.settings.face_ttl_backstop_seconds,
                on_expire=self.forget_face,
            )

    def forget_face(self, face_key: str) -> None:
        """A traveler photo is gone (deleted, replaced or expired): drop its queued jobs."""
        if self.avatar is not None:
            self.avatar.cancel_face(face_key)

    def clips_status(self) -> Optional[dict]:
        """Warm-up state of the concierge's pre-rendered clips, or ``None`` when there are none.

        ``warming``   clips are still being queued or rendered (the kiosk waits)
        ``ready``     every clip exists
        ``degraded``  nothing is rendering but some clips are missing (e.g. the Ditto worker died
                      or pre-render is off): those phrases render live with the concierge face
        """
        if self.clips is None or self.clip_renderer is None or self.default_face is None:
            return None
        face = FaceRef(path=self.default_face, key=DEFAULT_FACE_KEY)
        total = self.clips.total()
        missing = len(self.clips.missing(face))
        if missing == 0:
            state = "ready"
        elif self.clip_renderer.busy:
            state = "warming"
        else:
            state = "degraded"
        return {"state": state, "done": total - missing, "total": total}

    @property
    def secrets(self) -> list[str]:
        return secret_values(self.settings)

    def health(self) -> dict:
        from app.tracing import tracing_active

        components = {
            "workflow": self.workflow is not None,
            "transcriber": bool(self.transcriber and self.transcriber.ready),
            "synthesizer": bool(self.synthesizer and self.synthesizer.ready),
            "evidence_calibrated": bool(getattr(self.workflow, "evidence_calibrated", False)),
            "langsmith": tracing_active(),
        }
        if self.avatar is not None:  # key appears only when the avatar is switched on
            components["avatar"] = bool(self.avatar.ready)
        clips = self.clips_status()
        if clips is not None:
            components["clips"] = clips["state"]  # warming | ready | degraded
        # Which workflow is running ("multi_agent"); absent for test fakes.
        graph_kind = getattr(self.workflow, "graph_kind", None)
        if graph_kind:
            components["agent_graph"] = graph_kind
        return components

    def close(self) -> None:
        if self.avatar is not None:
            self.avatar.stop()
        if self.faces is not None:
            self.faces.close()  # deletes every traveler photo
        if self.video is not None:
            self.video.close()
        self.audio.close()


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def find_concierge_image(settings: "Settings") -> Path:
    """The default concierge photo: ``CONCIERGE_IMAGE`` or the first image in ``config/concierge/``."""
    if settings.concierge_image:
        path = Path(settings.concierge_image).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"CONCIERGE_IMAGE does not exist: {path}")
        return path
    folder = Path(settings.concierge_dir).resolve()
    candidates = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES) if folder.is_dir() else []
    if not candidates:
        raise RuntimeError(
            f"AVATAR_ENABLED=true but there is no default concierge image. "
            f"Put a PNG or JPG of the concierge face in {folder} (see config/concierge/README.md)."
        )
    return candidates[0]


def build_services(settings: Optional["Settings"] = None) -> Services:
    """Validate configuration and build the real services. Raises on any failure."""
    from app.agents.graph import MultiAgentKioskWorkflow
    from app.agents.guardrail import GuardrailResponses
    from app.agents.supervisor import LLMIntentClassifier
    from app.agents.tools import configure_search
    from app.retrieval.retrievers import HybridRetriever
    from app.retrieval.search import AirportSearch
    from app.retrieval.startup import validate_startup
    from app.speech_to_text.whisper_service import WhisperTranscriber
    from app.text_to_speech.kokoro_service import KokoroSynthesizer
    from app.workflow.answer import OpenAIAnswerGenerator
    from app.workflow.evidence import EvidenceAssessor, EvidenceThresholds
    from app.workflow.routing import ResponseBook, Router
    from app.workflow.validation import CitationOverlapValidator
    from config.settings import get_settings

    settings = settings or get_settings()
    validate_startup(settings)  # Cohere probe, Chroma, dimensions, BM25 - fail loudly

    retriever = HybridRetriever.from_settings(settings)
    search = AirportSearch(retriever, top_k=settings.answer_passages)
    # Wire the module-level ``search_airport`` @tool too, so it is usable outside the workflow.
    configure_search(search)

    workflow = MultiAgentKioskWorkflow(
        router=Router.from_file(settings.policy_path, settings.default_language),
        responses=ResponseBook.from_file(settings.responses_path, settings.default_language),
        search=search,
        assessor=EvidenceAssessor(EvidenceThresholds.from_file(settings.evidence_thresholds_path)),
        generator=OpenAIAnswerGenerator(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            max_tokens=settings.answer_max_tokens,
            timeout=settings.node_timeout_seconds,
        ),
        validator=CitationOverlapValidator(),
        node_timeout_seconds=settings.node_timeout_seconds,
        classifier=LLMIntentClassifier(
            api_key=settings.openai_api_key,
            model=settings.openai_classifier_model,
            timeout=settings.node_timeout_seconds,
        ),
        guardrail_responses=GuardrailResponses.from_file(settings.responses_path, settings.default_language),
        max_question_chars=settings.max_question_chars,
    )

    transcriber = WhisperTranscriber(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        max_seconds=settings.max_audio_seconds,
        language=settings.default_language,
    )
    synthesizer = KokoroSynthesizer(settings.kokoro_voice, settings.kokoro_lang)
    logger.info("Loading speech models (first run downloads weights)...")
    transcriber.warm_up()
    synthesizer.warm_up()

    avatar: Optional[AvatarRenderer] = None
    default_face: Optional[Path] = None
    if settings.avatar_enabled:
        from app.avatar.service import AvatarService

        default_face = find_concierge_image(settings)  # fail loudly if it is missing
        avatar_service = AvatarService.from_settings(settings)
        logger.info("Starting the Ditto worker (loads models onto the GPU)...")
        if avatar_service.start(wait_seconds=settings.avatar_startup_timeout_seconds):
            logger.info("Ditto worker ready.")
        else:
            # Not fatal: turns fall back to text + audio until the worker is up.
            logger.warning("Ditto worker is not ready yet; turns fall back to text and audio.")
        avatar = avatar_service
    else:
        logger.warning("Avatar is OFF (AVATAR_ENABLED=false): answers are text and voice only, no video.")

    clips: Optional[ClipCache] = None
    clip_renderer: Optional[ClipRenderer] = None
    if avatar is not None and default_face is not None:
        from app.avatar.clips import file_digest, load_phrases

        clips = ClipCache(
            settings.clips_dir,
            load_phrases(settings.responses_path, settings.default_language),
            voice=settings.kokoro_voice,
            language=settings.default_language,
            default_digest=file_digest(default_face),
        )
        clip_renderer = ClipRenderer(clips, avatar, synthesizer)
        if settings.clips_prerender_on_start:
            clip_renderer.schedule(FaceRef(path=default_face, key=DEFAULT_FACE_KEY))

    return Services(
        settings=settings,
        workflow=workflow,
        audio=AudioStore(settings.sessions_dir, settings.audio_ttl_seconds),
        transcriber=transcriber,
        synthesizer=synthesizer,
        avatar=avatar,
        default_face=default_face,
        clips=clips,
        clip_renderer=clip_renderer,
    )
