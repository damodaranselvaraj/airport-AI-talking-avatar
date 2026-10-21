"""Task 1 verification: config validates and /health responds."""
import importlib

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


def test_health_ok(build_services):
    with TestClient(create_app(build_services())) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["components"] == {
            "workflow": True,
            "transcriber": True,
            "synthesizer": True,
            "evidence_calibrated": True,
            "langsmith": False,
            "agent_graph": "multi_agent",
        }


def test_health_reports_missing_voice_components(build_services):
    svc = build_services(transcriber=None, synthesizer=None)
    with TestClient(create_app(svc)) as client:
        comps = client.get("/health").json()["components"]
        assert comps["transcriber"] is False and comps["synthesizer"] is False
        assert comps["workflow"] is True


def test_importing_main_builds_no_services():
    # `uvicorn app.main:app` must not load models or hit the network at import time.
    import app.main as main

    assert not hasattr(main.app.state, "services")


def test_missing_secret_fails_loudly(monkeypatch):
    # Clear required secrets and the cached settings, then expect a loud failure.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    # Prevent pydantic-settings from reading a local .env during this test.
    monkeypatch.setenv("PYDANTIC_SETTINGS_IGNORE_ENV_FILE", "1")

    import config.settings as settings_module

    importlib.reload(settings_module)
    settings_module.get_settings.cache_clear()

    with pytest.raises(Exception):
        settings_module.Settings(_env_file=None)


def test_health_reports_avatar_only_when_enabled(build_services):
    from tests.fakes import FakeAvatar

    with TestClient(create_app(build_services(avatar=FakeAvatar(ready=True)))) as client:
        assert client.get("/health").json()["components"]["avatar"] is True
    with TestClient(create_app(build_services(avatar=FakeAvatar(ready=False)))) as client:
        assert client.get("/health").json()["components"]["avatar"] is False


def test_closing_services_stops_the_avatar(build_services):
    from tests.fakes import FakeAvatar

    avatar = FakeAvatar()
    with TestClient(create_app(build_services(avatar=avatar))):
        assert not avatar.stopped
    assert avatar.stopped
