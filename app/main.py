"""FastAPI entrypoint for the Dam AI Talking Avatar Kiosk.

``create_app(services)`` lets tests inject fakes; ``app`` (used by
``uvicorn app.main:app``) builds the real services at startup and fails loudly
if configuration, retrieval stores, or speech models are not ready.

SECURITY: this server has no authentication and is meant for localhost. Bind to
127.0.0.1 (the uvicorn default) and add authentication before any network
exposure. All API keys stay server-side; no endpoint returns them.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import error_body, router as api_router
from app.api_faces import router as faces_router
from app.services import Services, build_services
from app.tracing import configure_tracing, tracing_active

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Show app logs (turn traces carry no question text) next to uvicorn's."""
    app_logger = logging.getLogger("app")
    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
        app_logger.addHandler(handler)
        app_logger.setLevel(logging.INFO)


_configure_logging()

UI_DIR = Path(__file__).resolve().parent / "ui"
CSP = (
    "default-src 'self'; media-src 'self' blob:; img-src 'self' data: blob:; "
    "style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
)


def create_app(services: Optional[Services] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Bootstrap LangSmith tracing before building services so the env vars
        # are in place when LangGraph compiles its graph (which registers callbacks).
        from config.settings import get_settings
        _settings = get_settings()
        configure_tracing(_settings)
        # Real startup: validate config + stores + models. Injected services skip this.
        svc = services or build_services(_settings)
        app.state.services = svc
        try:
            yield
        finally:
            svc.close()

    # Interactive docs/schema are developer tools; a kiosk has no reason to expose them.
    app = FastAPI(
        title="Dam AI Talking Avatar Kiosk",
        version="0.2.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Content-Security-Policy", CSP)
        response.headers.setdefault("Permissions-Policy", "microphone=(self), camera=(self)")
        if request.url.path.startswith("/api"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        # Never echo the submitted value back; report only which field was wrong.
        fields = sorted({str(e["loc"][-1]) for e in exc.errors() if e.get("loc")})
        return JSONResponse(
            error_body("invalid_request", f"Invalid request: {', '.join(fields) or 'body'}."),
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        logger.error("unhandled error: %s", type(exc).__name__)
        return JSONResponse(error_body("internal", "Something went wrong."), status_code=500)

    @app.get("/health")
    def health(request: Request) -> dict:
        """Liveness + component readiness. Contains no configuration or secrets."""
        components = request.app.state.services.health()
        return {"status": "ok", "service": "dam-ai-talking-avatar-kiosk", "components": components}

    app.include_router(api_router)
    app.include_router(faces_router)
    # Static UI last, so it never shadows /api or /health.
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app


app = create_app()
