"""Kokoro-82M text-to-speech on CPU.

Loaded once at startup and shared; synthesis calls are serialized with a lock
(the model is not thread-safe and the CPU is the bottleneck anyway). Pinned to
``device="cpu"`` so the GPU stays free for Ditto in Stage 3.
"""
from __future__ import annotations

import io
import logging
import re
import threading
import wave
from typing import TYPE_CHECKING, Optional

from app import i18n

if TYPE_CHECKING:
    from app.i18n import LangProfile

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
_RANGE_RE = re.compile(r"(?<=\d)\s*[\u2013\u2014-]\s*(?=\d)")
_DASH_RE = re.compile(r"\s*[\u2013\u2014]\s*")
_MARKUP_RE = re.compile(r"[*_`#>\[\]]")


def speakable(text: str) -> str:
    """Light, meaning-preserving clean-up so the voice reads the text naturally.

    The displayed caption is unchanged; only what is sent to Kokoro differs
    (dashes between numbers become 'to', stray markup is dropped).
    """
    spoken = _RANGE_RE.sub(" to ", text)
    spoken = _DASH_RE.sub(", ", spoken)
    spoken = _MARKUP_RE.sub("", spoken)
    return re.sub(r"\s+", " ", spoken).strip()


def pcm16_to_wav(samples, sample_rate: int = SAMPLE_RATE) -> bytes:
    """float32 [-1, 1] mono samples -> 16-bit PCM WAV bytes."""
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class KokoroSynthesizer:
    """Language-aware Kokoro TTS: one cached ``KPipeline`` per language.

    Pipelines are keyed by Kokoro lang_code (``a``/``h``/``e``) and built lazily.
    ``warm_up`` builds and exercises every supported language so the first real
    turn in any of them is fast (the accepted ~3x startup cost).
    """

    def __init__(
        self,
        profiles: dict[str, "LangProfile"],
        default_language: str = "en",
    ) -> None:
        self._profiles = dict(profiles)
        self._default = default_language
        self._pipelines: dict[str, object] = {}  # Kokoro lang_code -> KPipeline
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        """True once at least the default-language pipeline is built."""
        return i18n.kokoro_lang_code_for(self._default) in self._pipelines

    def _pipeline_for(self, language: str) -> object:
        """Build (once) and return the cached ``KPipeline`` for ``language``.

        Must be called under ``self._lock``.
        """
        lang_code = i18n.kokoro_lang_code_for(language)
        pipeline = self._pipelines.get(lang_code)
        if pipeline is None:
            from kokoro import KPipeline

            pipeline = KPipeline(
                lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", device="cpu"
            )
            self._pipelines[lang_code] = pipeline
        return pipeline

    def warm_up(self) -> None:
        """Build the model and run one tiny synthesis for every supported language."""
        for lang in i18n.SUPPORTED:
            self.synthesize("Ready.", lang)

    def synthesize(self, text: str, language: str = "en") -> bytes:
        import numpy as np

        spoken = speakable(text)
        if not spoken:
            raise ValueError("nothing to synthesize")
        voice = i18n.kokoro_voice_for(language)
        with self._lock:
            pipeline = self._pipeline_for(language)
            chunks = []
            for _gs, _ps, audio in pipeline(spoken, voice=voice):  # type: ignore[operator]
                if audio is not None:
                    chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise RuntimeError("Kokoro produced no audio")
        return pcm16_to_wav(np.concatenate(chunks))
