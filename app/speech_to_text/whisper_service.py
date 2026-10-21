"""Whisper speech-to-text (faster-whisper, CPU).

Audio arrives as bytes from the browser (WebM/Opus from MediaRecorder, or WAV),
is decoded in memory by PyAV (bundled with faster-whisper) and is never written
to disk. Transcript text is returned to the caller and never logged.
"""
from __future__ import annotations

import io
import logging
import math
import threading
from typing import Optional

import numpy as np

from app.speech_to_text.errors import (
    AudioTooLargeError,
    EmptyAudioError,
    NoSpeechError,
    UndecodableAudioError,
)
from app.workflow.contracts import Transcript

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Biases decoding toward airport vocabulary (carriers, places, codes) from data/*.md.
DEFAULT_INITIAL_PROMPT = (
    "Dam International Airport, DIA. Gate C22, Terminal 2, Concourse C, Level 2, Skywalk. "
    "Horizon Air, Cedar Pacific, Sahara Jet, NoorAir, Pearl Wings, Sandpiper, DIA Express. "
    "Date Palm Caf\u00e9, prayer room, lost and found, security, check-in, baggage."
)
_MIN_RMS = 0.003           # below this the clip is effectively silence
_NO_SPEECH_PROB = 0.6      # drop segments Whisper itself flags as non-speech


class WhisperTranscriber:
    def __init__(
        self,
        model_name: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        max_seconds: float = 20.0,
        initial_prompt: str = DEFAULT_INITIAL_PROMPT,
        language: str = "en",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._max_seconds = max_seconds
        self._prompt = initial_prompt
        self._language = language
        self._model: Optional[object] = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_name, device=self._device, compute_type=self._compute_type
            )
        return self._model

    def warm_up(self) -> None:
        """Load the model and decode one second of silence so the first turn is fast."""
        with self._lock:
            model = self._load()
            segments, _ = model.transcribe(  # type: ignore[attr-defined]
                np.zeros(SAMPLE_RATE, dtype=np.float32), language=self._language, beam_size=1
            )
            list(segments)

    @staticmethod
    def decode(data: bytes) -> np.ndarray:
        """Decode container audio to mono float32 at 16 kHz, entirely in memory."""
        from faster_whisper.audio import decode_audio

        try:
            audio = decode_audio(io.BytesIO(data), sampling_rate=SAMPLE_RATE)
        except Exception as exc:  # PyAV raises many distinct error types
            # Type only: surfaces environment bugs (e.g. a PyAV/faster-whisper mismatch)
            # without logging anything derived from the traveler's audio.
            logger.warning("audio decode failed: %s", type(exc).__name__)
            raise UndecodableAudioError() from exc
        return np.asarray(audio, dtype=np.float32)

    def transcribe(self, data: bytes) -> Transcript:
        if not data:
            raise EmptyAudioError()
        audio = self.decode(data)
        duration = len(audio) / SAMPLE_RATE
        if duration <= 0.0:
            raise EmptyAudioError()
        if duration > self._max_seconds:
            raise AudioTooLargeError()
        if float(np.sqrt(np.mean(np.square(audio)))) < _MIN_RMS:
            raise NoSpeechError()

        with self._lock:
            model = self._load()
            segments, _info = model.transcribe(  # type: ignore[attr-defined]
                audio,
                language=self._language,
                beam_size=5,
                vad_filter=True,
                initial_prompt=self._prompt,
                condition_on_previous_text=False,
            )
            kept = [s for s in segments if s.no_speech_prob < _NO_SPEECH_PROB]

        text = " ".join(s.text.strip() for s in kept).strip()
        if not text:
            raise NoSpeechError()
        total = sum(max(s.end - s.start, 1e-3) for s in kept)
        mean_logprob = sum(s.avg_logprob * max(s.end - s.start, 1e-3) for s in kept) / total
        confidence = max(0.0, min(1.0, math.exp(mean_logprob)))
        return Transcript(
            text=text,
            duration_s=round(duration, 2),
            confidence=round(confidence, 3),
            language=self._language,
        )
