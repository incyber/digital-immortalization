"""Building a person's Gaussian splat, over HTTP.

Three things this surface does that the training endpoints next door do not,
each of them a product requirement rather than a technical one.

*A refusal is an answer, not an error.* When the uploaded material cannot
produce a likeness, the route selector says so and names what is missing in
the family's own counts. That comes back as 200 with a refusal in it, because
"we need a video of at least 8 seconds where their face is visible, or at
least 3 clear photographs, and you have 2" is guidance somebody can act on,
and 400 Bad Request is not. The client shows one of those to a grieving
family; it must not be the second.

*The disclosure travels with the result.* Every response that reports a
finished build carries the sentence and the measured fraction behind it. There
is no shape of this API that hands back a likeness without them.

*Ownership is in the query.* Every handler resolves the signed-in tenant first
and every lookup is narrowed by it, exactly as gateway/tenancy.py requires, and
a build that is not yours is refused identically to one that does not exist.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.gateway.csrf import require_same_site_header
from avatar.gateway.models import Avatar
from avatar.gateway.tenancy import TenantError, assert_owned
from avatar.splat.build import SplatRefused
from avatar.splat.routes import RouteDecision
from avatar.splat.service import SplatService, SplatUnavailable


def _guidance(decision: RouteDecision) -> str:
    """The refusal as one sentence somebody can act on.

    Assembled from what the route selector itemised rather than written here,
    so the numbers in the sentence are the numbers that made the decision.
    """
    if not decision.missing:
        return decision.reasoning
    return f"We need {' — '.join(decision.missing)}."


def _refusal(decision: RouteDecision) -> dict:
    return {
        "status": "refused",
        "buildable": False,
        "reasoning": decision.reasoning,
        "missing": list(decision.missing),
        "guidance": _guidance(decision),
        # The factual trail support reads when a family disputes the refusal.
        "considered": list(decision.considered),
    }


def build_router(current_user, get_db, service: SplatService) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/api/photo-sets/{photo_set_id}/splat",
        dependencies=[Depends(require_same_site_header)],
        status_code=202,
    )
    async def start(
        photo_set_id: str,
        db: AsyncSession = Depends(get_db),  # noqa: B008 - FastAPI injection idiom
        user_id: str = Depends(current_user),
    ):
        """Start a build, or explain what is missing.

        Deliberately not gated on the photo set passing the training checks.
        Those demand fifteen usable images because that is what a LoRA needs;
        a splat can be reconstructed from a short clip with far fewer, and the
        route selector applies the floor that actually governs this build.
        """
        try:
            started = await service.start(db, photo_set_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SplatRefused as exc:
            # 200, not 202 and not 400. Nothing is being built, so the
            # accepted status would be a lie; and the request was not bad.
            # See the module docstring.
            return JSONResponse(status_code=200, content=_refusal(exc.decision))
        except SplatUnavailable as exc:
            # The set is theirs and real; what is wrong is its state, which
            # they can act on but not by uploading anything.
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return {
            "status": "building",
            "buildable": True,
            "job_id": started.job_id,
            "avatar_id": started.avatar_id,
            "route": started.decision.route.value,
            "reasoning": started.decision.reasoning,
            "considered": list(started.decision.considered),
        }

    @router.get("/api/splat-jobs/{job_id}")
    async def read(
        job_id: str,
        db: AsyncSession = Depends(get_db),  # noqa: B008 - FastAPI injection idiom
        user_id: str = Depends(current_user),
    ):
        try:
            return await service.read(db, job_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post(
        "/api/splat-jobs/{job_id}/cancel",
        dependencies=[Depends(require_same_site_header)],
    )
    async def cancel(
        job_id: str,
        db: AsyncSession = Depends(get_db),  # noqa: B008 - FastAPI injection idiom
        user_id: str = Depends(current_user),
    ):
        try:
            return await service.cancel(db, job_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/avatars/{avatar_id}/splat")
    async def read_avatar_splat(
        avatar_id: str,
        db: AsyncSession = Depends(get_db),  # noqa: B008 - FastAPI injection idiom
        user_id: str = Depends(current_user),
    ):
        """What was built for this person, and what must be said about it.

        Exists so the disclosure survives the page that started the build. A
        family who comes back a week later still sees how much of the likeness
        was generated, at the moment they see the likeness.
        """
        try:
            avatar: Avatar = await assert_owned(db, avatar_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        measured = avatar.splat_measured_fraction
        return {
            "avatar_id": avatar.id,
            "built": bool(avatar.splat_key),
            "splat_key": avatar.splat_key,
            "route": avatar.splat_route,
            "reasoning": avatar.splat_reasoning,
            "disclosure": avatar.splat_disclosure,
            "measured_fraction": measured,
            "generated_fraction": round(1.0 - measured, 2) if measured is not None else None,
            "concerns": json.loads(avatar.splat_concerns) if avatar.splat_concerns else [],
            "gaussians": avatar.splat_gaussians,
            "size_bytes": avatar.splat_size_bytes,
            "backend": avatar.splat_backend,
        }

    return router
