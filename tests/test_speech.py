"""Tasks 7-8: speech services. Model-free tests run by default; real-model tests are `slow`."""
import io
import wave

import numpy as np
import pytest

from app.speech_to_text.errors import (
    AudioTooLargeError,
    EmptyAudioError,
    NoSpeechError,
    UndecodableAudioError,
)
from app import i18n
from app.speech_to_text.whisper_service import SAMPLE_RATE, WhisperTranscriber
from app.text_to_speech.kokoro_service import KokoroSynthesizer, pcm16_to_wav, speakable
from app.workflow.contracts import Transcript
from tests.fakes import silent_wav


def make_synth() -> KokoroSynthesizer:
    return KokoroSynthesizer(i18n._PROFILES, default_language="en")


def wav_bytes(samples: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    return pcm16_to_wav(samples, rate)


def wav_info(data: bytes):
    with wave.open(io.BytesIO(data)) as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()


# ------------------------------------------------------------- TTS helpers
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Open 05:00\u201322:00 daily.", "Open 05:00 to 22:00 daily."),
        ("Gates A1-A20", "Gates A1-A20"),
        ("It is **open** 24 hours.", "It is open 24 hours."),
        ("Walk past C12 \u2014 then turn right.", "Walk past C12, then turn right."),
        ("  spaced   out  ", "spaced out"),
    ],
)
def test_speakable(raw, expected):
    assert speakable(raw) == expected


def test_pcm16_to_wav_roundtrip_and_clipping():
    wav = pcm16_to_wav(np.array([0.0, 0.5, -0.5, 2.0, -2.0], dtype=np.float32))
    assert wav_info(wav) == (1, 2, 24000, 5)
    pcm = np.frombuffer(wav[44:], dtype="<i2")
    assert pcm[3] == 32767 and pcm[4] == -32767


def test_synthesizer_rejects_text_with_nothing_to_say():
    with pytest.raises(ValueError):
        make_synth().synthesize("** __ ``")


def test_speakable_preserves_devanagari_and_strips_markup():
    assert speakable("गेट **C22** है") == "गेट C22 है"


# ------------------------------------------------------------- STT guards (no model load)
@pytest.fixture
def stt():
    return WhisperTranscriber(max_seconds=2.0)


def test_empty_bytes(stt):
    with pytest.raises(EmptyAudioError):
        stt.transcribe(b"")


def test_garbage_bytes(stt):
    with pytest.raises(UndecodableAudioError):
        stt.transcribe(b"this is definitely not audio" * 50)


def test_silence_is_no_speech(stt):
    with pytest.raises(NoSpeechError):
        stt.transcribe(silent_wav(1.0, SAMPLE_RATE))
    assert not stt.ready  # rejected before any model was loaded


def test_overlong_audio_rejected_before_inference(stt):
    noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE * 3) * 0.1).astype(np.float32)
    with pytest.raises(AudioTooLargeError):
        stt.transcribe(wav_bytes(noise))
    assert not stt.ready


def test_decode_resamples_to_16k_mono():
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(24000) / 24000)).astype(np.float32)
    audio = WhisperTranscriber.decode(wav_bytes(tone, 24000))
    assert audio.dtype == np.float32 and abs(len(audio) - SAMPLE_RATE) < 200


def _webm_opus(samples: np.ndarray, rate: int = 48000) -> bytes:
    """Encode like a browser's MediaRecorder: Opus in a WebM container."""
    import av

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="webm") as out:
        stream = out.add_stream("libopus", rate=rate)
        stream.layout = "mono"
        pcm = (samples * 32767).astype(np.int16)
        frame_size = 960
        for pts, i in enumerate(range(0, len(pcm) - frame_size + 1, frame_size)):
            frame = av.AudioFrame.from_ndarray(pcm[i:i + frame_size][None, :], format="s16", layout="mono")
            frame.sample_rate = rate
            frame.pts = pts * frame_size
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return buf.getvalue()


def test_decodes_browser_style_webm_opus():
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)).astype(np.float32)
    data = _webm_opus(tone)
    assert data[:4] == b"\x1a\x45\xdf\xa3"  # EBML header = WebM/Matroska
    audio = WhisperTranscriber.decode(data)
    assert abs(len(audio) / SAMPLE_RATE - 1.0) < 0.1
    assert float(np.sqrt(np.mean(audio ** 2))) > 0.1


# ---------------------------------------------- Whisper auto-detect (no model load)
class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text
        self.no_speech_prob = 0.0
        self.start = 0.0
        self.end = 1.0
        self.avg_logprob = -0.2


class _FakeInfo:
    def __init__(self, language: str) -> None:
        self.language = language
        self.language_probability = 0.99


class _FakeWhisperModel:
    """Records the kwargs passed to transcribe and returns scripted segments + info."""

    def __init__(self, detected: str = "hi", segment_text: str = "गेट C22 कहाँ है") -> None:
        self._detected = detected
        self._segment_text = segment_text
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return [_FakeSegment(self._segment_text)], _FakeInfo(self._detected)


def _loaded_transcriber(model: _FakeWhisperModel, language=None) -> WhisperTranscriber:
    stt = WhisperTranscriber(max_seconds=20.0, language=language)
    stt._model = model  # inject the fake so no real model loads
    return stt


def _speechy_audio() -> np.ndarray:
    return (np.random.default_rng(1).standard_normal(SAMPLE_RATE) * 0.1).astype(np.float32)


def test_autodetect_two_pass_hindi_uses_native_script_prompt(monkeypatch):
    # Two-pass: pass 1 detects the language (no prompt, beam 1); pass 2 re-transcribes
    # pinned to the detected language WITH its native-script prompt, so Hindi comes back
    # in Devanagari instead of a Latin romanization.
    model = _FakeWhisperModel(detected="hi", segment_text="गेट C22 कहाँ है")
    stt = _loaded_transcriber(model, language=None)
    monkeypatch.setattr(WhisperTranscriber, "decode", staticmethod(lambda data: _speechy_audio()))
    result = stt.transcribe(b"audio-bytes")
    assert result.language == "hi"             # raw detected code, not mapped/relabeled
    assert result.text == "गेट C22 कहाँ है"
    assert len(model.calls) == 2               # one detect pass, one decode pass

    detect = model.calls[0]
    assert detect["language"] is None          # pass 1 auto-detects
    assert detect["initial_prompt"] is None    # no prompt bias during detection

    decode = model.calls[1]
    assert decode["language"] == "hi"          # pass 2 pins the detected language
    assert decode["task"] == "transcribe"      # never translate
    # The native-script Hindi prompt is what anchors Devanagari output.
    assert decode["initial_prompt"] is not None and "गेट" in decode["initial_prompt"]


def test_pinned_english_single_pass_passes_the_english_prompt(monkeypatch):
    # A pinned language skips the detection pass: one decode call with that language's prompt.
    model = _FakeWhisperModel(detected="en", segment_text="Where is gate C22")
    stt = _loaded_transcriber(model, language="en")
    monkeypatch.setattr(WhisperTranscriber, "decode", staticmethod(lambda data: _speechy_audio()))
    result = stt.transcribe(b"audio-bytes")
    assert result.language == "en"
    assert len(model.calls) == 1               # pinned => no detection pass
    kwargs = model.calls[0]
    assert kwargs["language"] == "en"
    assert kwargs["initial_prompt"] is not None and "Dam International Airport" in kwargs["initial_prompt"]


# ---------------------------------------------- Kokoro multi-pipeline (fake KPipeline)
class _FakePipeline:
    instances: list["_FakePipeline"] = []

    def __init__(self, lang_code=None, repo_id=None, device=None) -> None:
        self.lang_code = lang_code
        self.repo_id = repo_id
        self.device = device
        self.calls: list[str] = []
        _FakePipeline.instances.append(self)

    def __call__(self, text, voice=None):
        self.calls.append(voice)
        yield None, None, np.zeros(1200, dtype=np.float32)


def test_synthesize_selects_voice_and_lang_code_per_language(monkeypatch):
    _FakePipeline.instances = []
    monkeypatch.setitem(__import__("sys").modules, "kokoro", type("M", (), {"KPipeline": _FakePipeline}))
    synth = make_synth()
    synth.synthesize("Hola", "es")
    synth.synthesize("नमस्ते", "hi")
    by_code = {p.lang_code: p for p in _FakePipeline.instances}
    assert set(by_code) == {"e", "h"}
    assert by_code["e"].repo_id == "hexgrad/Kokoro-82M"
    assert by_code["e"].calls == [i18n.kokoro_voice_for("es")]
    assert by_code["h"].calls == [i18n.kokoro_voice_for("hi")]


def test_warm_up_builds_every_supported_language(monkeypatch):
    _FakePipeline.instances = []
    monkeypatch.setitem(__import__("sys").modules, "kokoro", type("M", (), {"KPipeline": _FakePipeline}))
    synth = make_synth()
    synth.warm_up()
    assert {p.lang_code for p in _FakePipeline.instances} == {"a", "h", "e"}
    assert synth.ready


# ------------------------------------------------------------- real models (opt-in: -m slow)
@pytest.mark.slow
def test_real_kokoro_synthesis_is_valid_cpu_audio():
    wav = make_synth().synthesize("Gate C22 is in Terminal 2, Concourse C, Level 2.")
    channels, width, rate, frames = wav_info(wav)
    assert (channels, width, rate) == (1, 2, 24000)
    assert 1.5 < frames / rate < 12


@pytest.mark.slow
def test_voice_roundtrip_kokoro_to_whisper():
    """TTS -> STT: the full voice path, models only."""
    wav = make_synth().synthesize("Where is gate C22?")
    result = WhisperTranscriber(model_name="small").transcribe(wav)
    text = result.text.lower()
    assert "gate" in text and "22" in text.replace("twenty-two", "22").replace("twenty two", "22")
    assert 0.0 < result.confidence <= 1.0
    assert 0.5 < result.duration_s < 10


@pytest.mark.slow
def test_real_kokoro_speaks_hindi_and_spanish_native_script():
    """Real Kokoro in Hindi and Spanish (requires espeak-ng on the host for lang_code h and e)."""
    for lang in ("hi", "es"):
        wav = make_synth().synthesize("नमस्ते" if lang == "hi" else "Hola", lang)
        assert wav_info(wav)[:3] == (1, 2, 24000)
