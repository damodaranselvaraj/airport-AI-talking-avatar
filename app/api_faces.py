"""Traveler face endpoints. A photo lives for one session and never leaves this machine.

POST   /api/face            multipart image + session_id + consent=true -> sets the session's face
POST   /api/face/keepalive  {session_id} -> push the photo's expiry out
DELETE /api/face            {session_id} -> delete the photo now

No response contains the photo, its file name, or exception text.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api import SESSION_ID_PATTERN, error_body, read_capped
from app.avatar.images import BadImage, sanitize_photo
from app.avatar.models import FaceCheck
from app.services import Services

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# Shorter than the browser's 60 s request timeout; a check may wait behind a running render.
FACE_CHECK_TIMEOUT_SECONDS = 55.0


class PhotoTooLarge(Exception):
    """Upload exceeded ``max_face_bytes``."""


class SessionBody(BaseModel):
    session_id: str = Field(..., pattern=SESSION_ID_PATTERN)


def _services(request: Request) -> Services:
    return request.app.state.services


def _unavailable() -> JSONResponse:
    return JSONResponse(
        error_body("avatar_unavailable", "Using your own photo isn't available right now."),
        status_code=503,
    )


@router.post("/face")
async def upload_face(
    request: Request,
    image: UploadFile = File(...),
    session_id: str = Form(..., pattern=SESSION_ID_PATTERN),
    consent: str = Form(""),
):
    services = _services(request)
    faces, avatar = services.faces, services.avatar
    try:
        if faces is None or avatar is None:
            return _unavailable()
        if consent.strip().lower() != "true":
            return JSONResponse(
                error_body("consent_required", "Please confirm that you agree to use your photo."),
                status_code=422,
            )
        try:
            data = await read_capped(image, services.settings.max_face_bytes, PhotoTooLarge)
        except PhotoTooLarge:
            return JSONResponse(error_body("too_large", "That photo is too large."), status_code=413)
        try:
            png = await run_in_threadpool(sanitize_photo, data)
        except BadImage:
            return JSONResponse(
                error_body("bad_image", "That file isn't a usable photo. Please use a clear JPG or PNG."),
                status_code=422,
            )
        ref = faces.stage(png)
        activated = False
        try:
            verdict = await run_in_threadpool(avatar.check_face, ref.path, timeout_s=FACE_CHECK_TIMEOUT_SECONDS)
            if verdict is FaceCheck.NO_FACE:
                return JSONResponse(
                    error_body("no_face", "I couldn't find a clear face in that photo. Please try another one."),
                    status_code=422,
                )
            if verdict is not FaceCheck.OK:
                return _unavailable()
            replaced = faces.activate(session_id, ref)
            activated = True
            if replaced:
                services.forget_face(replaced)  # its queued jobs go with it
        finally:
            if not activated:
                faces.discard(ref)  # rejected, failed or cancelled: the photo must not linger
        return {"status": "ok", "expires_in_seconds": services.settings.face_ttl_backstop_seconds}
    except Exception as exc:  # noqa: BLE001 - never leak internals
        logger.error("face upload failed: %s", type(exc).__name__)
        return JSONResponse(error_body("face_failed", "Something went wrong with that photo."), status_code=500)
    finally:
        await image.close()


@router.post("/face/keepalive")
def keepalive(body: SessionBody, request: Request):
    faces = _services(request).faces
    return {"active": bool(faces and faces.touch(body.session_id))}


@router.delete("/face")
def delete_face(body: SessionBody, request: Request):
    services = _services(request)
    if services.faces is None:
        return {"deleted": False}
    key = services.faces.delete(body.session_id)
    if key:
        services.forget_face(key)  # queued renders for that photo are dropped
    return {"deleted": key is not None}
