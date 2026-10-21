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
from typing import Optional

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
    def __init__(self, voice: str = "af_heart", lang_code: str = "a") -> None:
        self._voice = voice
        self._lang = lang_code
        self._pipeline: Optional[object] = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    def warm_up(self) -> None:
        """Load the model and run one tiny synthesis so the first real turn is fast."""
        with self._lock:
            if self._pipeline is None:
                from kokoro import KPipeline

                self._pipeline = KPipeline(lang_code=self._lang, device="cpu")
        self.synthesize("Ready.")

    def synthesize(self, text: str) -> bytes:
        import numpy as np

        spoken = speakable(text)
        if not spoken:
            raise ValueError("nothing to synthesize")
        with self._lock:
            if self._pipeline is None:
                from kokoro import KPipeline

                self._pipeline = KPipeline(lang_code=self._lang, device="cpu")
            chunks = []
            for _gs, _ps, audio in self._pipeline(spoken, voice=self._voice):  # type: ignore[operator]
                if audio is not None:
                    chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise RuntimeError("Kokoro produced no audio")
        return pcm16_to_wav(np.concatenate(chunks))
