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

from app import i18n
from app.speech_to_text.errors import (
    AudioTooLargeError,
    EmptyAudioError,
    NoSpeechError,
    UndecodableAudioError,
)
from app.workflow.contracts import Transcript

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Per-language initial prompts. Beyond biasing toward airport vocabulary, the script
# of the prompt anchors the output script: a Devanagari prompt makes Whisper emit
# Devanagari for Hindi instead of romanizing it ("प्रार्थना कक्ष" not "prarthana kaksh").
_PROMPT_EN = (
    "Dam International Airport, DIA. Gate C22, Terminal 2, Concourse C, Level 2, Skywalk. "
    "Horizon Air, Cedar Pacific, Sahara Jet, NoorAir, Pearl Wings, Sandpiper, DIA Express. "
    "Date Palm Café, prayer room, lost and found, security, check-in, baggage."
)
_PROMPT_HI = (
    "गेट C22 टर्मिनल 2, Concourse C, लेवल 2 पर है। T2 सिक्योरिटी से लगभग 12 मिनट की पैदल दूरी। "
    "सबसे पास कैफ़े Date Palm Café है। प्रार्थना कक्ष, खोया-पाया, सामान, चेक-इन, सुरक्षा जाँच।"
)
_PROMPT_ES = (
    "La puerta C22 está en la Terminal 2, muelle C, nivel 2. Desde seguridad de T2 son unos "
    "12 minutos a pie. El café más cercano es Date Palm Café. Sala de oración, objetos perdidos, "
    "equipaje, facturación, control de seguridad."
)
# Language code -> native-script initial prompt. Used on the second pass once the
# spoken language is known, so each language transcribes in its own script.
INITIAL_PROMPTS = {"en": _PROMPT_EN, "hi": _PROMPT_HI, "es": _PROMPT_ES}
# Kept for backward compatibility with callers/tests that import the old name.
DEFAULT_INITIAL_PROMPT = _PROMPT_EN

_MIN_RMS = 0.003           # below this the clip is effectively silence
_NO_SPEECH_PROB = 0.6      # drop segments Whisper itself flags as non-speech


class WhisperTranscriber:
    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        max_seconds: float = 20.0,
        initial_prompt: str = DEFAULT_INITIAL_PROMPT,
        language: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._max_seconds = max_seconds
        # Retained for the back-compat `initial_prompt` constructor arg / tests; the
        # per-language prompt actually used on decode comes from INITIAL_PROMPTS.
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
                np.zeros(SAMPLE_RATE, dtype=np.float32), language=None, beam_size=1
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

        # Two-pass decode so each language transcribes in its OWN script:
        #   pass 1 - pin the language if a caller fixed one, else auto-detect it cheaply;
        #   pass 2 - transcribe pinned to that language WITH its native-script initial prompt.
        # The native-script prompt is what forces Devanagari for Hindi instead of a Latin
        # romanization ("प्रार्थना कक्ष कहाँ है?" not "prarthana kaksh kahan hai?").
        with self._lock:
            model = self._load()

            if self._language is not None:
                # Caller pinned a language: no detection pass needed.
                decode_lang = self._language
            else:
                # Pass 1: detect the spoken language (beam_size=1, no prompt bias).
                _detect_segments, detect_info = model.transcribe(
                    audio,
                    language=None,
                    task="transcribe",
                    beam_size=1,
                    vad_filter=True,
                    initial_prompt=None,
                    condition_on_previous_text=False,
                )
                list(_detect_segments)  # force the generator so detect_info is populated
                decode_lang = getattr(detect_info, "language", None) or i18n.DEFAULT

            # Pass 2: transcribe pinned to decode_lang with that language's native-script
            # prompt. Fall back to the English prompt for any language without its own.
            prompt = INITIAL_PROMPTS.get(decode_lang, _PROMPT_EN)
            segments, info = model.transcribe(
                audio,
                language=decode_lang,
                task="transcribe",  # transcribe; never translate
                beam_size=5,
                vad_filter=True,
                initial_prompt=prompt,
                condition_on_previous_text=False,
            )
            kept = [s for s in segments if s.no_speech_prob < _NO_SPEECH_PROB]
            text = " ".join(s.text.strip() for s in kept).strip()
            # The raw code actually decoded. The supported-set gate lives at /api/ask.
            detected = getattr(info, "language", None) or decode_lang

        if not text:
            raise NoSpeechError()
        total = sum(max(s.end - s.start, 1e-3) for s in kept)
        mean_logprob = sum(s.avg_logprob * max(s.end - s.start, 1e-3) for s in kept) / total
        confidence = max(0.0, min(1.0, math.exp(mean_logprob)))
        return Transcript(
            text=text,
            duration_s=round(duration, 2),
            confidence=round(confidence, 3),
            language=detected or i18n.DEFAULT,
        )
