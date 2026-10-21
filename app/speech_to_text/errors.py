"""Typed speech-to-text failures with a stable, user-safe code.

``message`` is shown to the traveler, so it never contains exception text.
"""
from __future__ import annotations


class TranscriptionError(Exception):
    status_code = 400
    code = "transcription_failed"
    message = "Sorry, I couldn't process that recording. Please try again or type your question."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)
        if message:
            self.message = message


class EmptyAudioError(TranscriptionError):
    status_code = 400
    code = "empty_audio"
    message = "No audio was received. Please try recording again."


class AudioTooLargeError(TranscriptionError):
    status_code = 413
    code = "audio_too_large"
    message = "That recording is too long. Please ask a shorter question."


class UndecodableAudioError(TranscriptionError):
    status_code = 415
    code = "undecodable_audio"
    message = "I couldn't read that recording. Please try again or type your question."


class NoSpeechError(TranscriptionError):
    status_code = 422
    code = "no_speech"
    message = "I didn't hear anything. Please try again, or type your question."
