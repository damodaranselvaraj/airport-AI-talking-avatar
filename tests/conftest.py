"""Shared fixtures: fake-backed Services and a TestClient (no network, no models)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.audio_store import AudioStore
from app.main import create_app
from app.services import Services
from config.settings import Settings
from tests.agents.helpers import make_agent_workflow
from tests.fakes import FakeSynthesizer, FakeTranscriber

SECRETS = {
    "openai_api_key": "sk-test-OPENAI-secret-0123456789",
    "typesafe_api_key": "tsk-test-TYPESAFE-secret-0123456789",
    "cohere_api_key": "co-test-COHERE-secret-0123456789",
}


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads real Whisper/Kokoro models")
    config.addinivalue_line("markers", "live: needs network + real API keys")
    config.addinivalue_line("markers", "gpu: needs the Ditto conda env, checkpoints and a CUDA GPU")


def pytest_collection_modifyitems(config, items):
    """slow/live tests are opt-in: pytest -m slow  /  pytest -m live."""
    selected = config.getoption("-m") or ""
    for marker in ("slow", "live", "gpu"):
        if marker in selected:
            continue
        skip = pytest.mark.skip(reason=f"opt-in: run with -m {marker}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        sessions_dir=str(tmp_path / "sessions"),
        faces_dir=str(tmp_path / "faces"),
        max_audio_bytes=50_000,
        **SECRETS,
    )


@pytest.fixture
def build_services(settings):
    def _build(**overrides) -> Services:
        workflow, retriever, generator, _ = make_agent_workflow()
        # Fields that Services.__post_init__ reads (avatar, video, default_face) must go
        # through the constructor; anything else is attached afterwards.
        ctor = {
            "transcriber": overrides.pop("transcriber", FakeTranscriber()),
            "synthesizer": overrides.pop("synthesizer", FakeSynthesizer()),
        }
        for name in ("avatar", "video", "default_face", "faces", "clips", "clip_renderer"):
            if name in overrides:
                ctor[name] = overrides.pop(name)
        svc = Services(
            settings=settings,
            workflow=workflow,
            audio=AudioStore(settings.sessions_dir, settings.audio_ttl_seconds),
            **ctor,
        )
        svc.retriever, svc.generator = retriever, generator  # type: ignore[attr-defined]
        for k, v in overrides.items():
            setattr(svc, k, v)
        return svc

    return _build


@pytest.fixture
def client(build_services):
    svc = build_services()
    with TestClient(create_app(svc)) as c:
        c.services = svc  # type: ignore[attr-defined]
        yield c
