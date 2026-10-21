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
from app.speech_to_text.whisper_service import SAMPLE_RATE, WhisperTranscriber
from app.text_to_speech.kokoro_service import KokoroSynthesizer, pcm16_to_wav, speakable
from tests.fakes import silent_wav


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
        KokoroSynthesizer().synthesize("** __ ``")


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


# ------------------------------------------------------------- real models (opt-in: -m slow)
@pytest.mark.slow
def test_real_kokoro_synthesis_is_valid_cpu_audio():
    wav = KokoroSynthesizer().synthesize("Gate C22 is in Terminal 2, Concourse C, Level 2.")
    channels, width, rate, frames = wav_info(wav)
    assert (channels, width, rate) == (1, 2, 24000)
    assert 1.5 < frames / rate < 12


@pytest.mark.slow
def test_voice_roundtrip_kokoro_to_whisper():
    """TTS -> STT: the full voice path, models only."""
    wav = KokoroSynthesizer().synthesize("Where is gate C22?")
    result = WhisperTranscriber(model_name="small.en").transcribe(wav)
    text = result.text.lower()
    assert "gate" in text and "22" in text.replace("twenty-two", "22").replace("twenty two", "22")
    assert 0.0 < result.confidence <= 1.0
    assert 0.5 < result.duration_s < 10
