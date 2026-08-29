"""HTTP surface.

Deliberately small. Everything that matters about a session request happens in
consent.py and sessions.py; this layer only translates exceptions into status
codes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Cookie, Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.config import Settings, get_settings
from avatar.gateway.auth import (
    SESSION_TTL_SECONDS,
    AuthError,
    authenticate,
    issue_session,
    read_session,
    register,
)
from avatar.gateway.consent import ConsentError
from avatar.gateway.db import create_all, get_db
from avatar.gateway.dispatch import LocalProcessDispatcher
from avatar.gateway.routes_avatars import build_router as build_avatar_router
from avatar.gateway.routes_ingest import build_router
from avatar.gateway.sessions import open_session
from avatar.gateway.tenancy import TenantError
from avatar.storage.factory import build_store
from avatar.training.factory import build_runner


class SessionRequest(BaseModel):
    avatar_id: str


class Credentials(BaseModel):
    email: EmailStr
    password: str


# The cookie is httpOnly so page scripts cannot read it, sameSite=lax so it is
# not sent on cross-site requests, and secure in deployment. Storing the token
# where JavaScript can reach it would mean any injected script could take over
# an account holding a family's photographs.
SESSION_COOKIE = "avatar_session"


def create_app(cfg: Settings | None = None) -> FastAPI:
    settings = cfg or get_settings()

    dispatcher = LocalProcessDispatcher(settings)
    store = build_store(settings)
    runner = build_runner(settings, store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await create_all(settings)
        yield
        # Agents outlive a request but not the gateway. Leaving them running
        # would hold GPU capacity and microphones open after a restart.
        dispatcher.shutdown()

    app = FastAPI(title="Avatar gateway", lifespan=lifespan)

    # The web client is served from a different origin in development. Listed
    # explicitly rather than wildcarded, because these responses carry room
    # tokens and a wildcard would let any page request one.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def set_session_cookie(response: Response, token: str) -> None:
        # Lax when the web app and the gateway share a host, none when they do
        # not. A deployment that serves the site from one domain and this API
        # from another is cross-site by definition, and a lax cookie is simply
        # never sent - sign-in appears to succeed and every later request is
        # anonymous. Browsers only accept samesite=none over https, so it is
        # tied to the same flag that turns on Secure.
        response.set_cookie(
            SESSION_COOKIE,
            token,
            httponly=True,
            samesite="none" if settings.cookies_secure else "lax",
            secure=settings.cookies_secure,
            max_age=SESSION_TTL_SECONDS,
            path="/",
        )

    async def current_user(
        avatar_session: str | None = Cookie(default=None),
    ) -> str:
        """The signed-in tenant, or 401.

        Every authenticated route depends on this. It returns an id rather than
        a User so that handlers cannot accidentally hold a detached row across
        a request boundary.
        """
        user_id = read_session(settings, avatar_session or "")
        if user_id is None:
            raise HTTPException(status_code=401, detail="sign in required")
        return user_id

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/api/auth/register", status_code=201)
    async def register_account(
        body: Credentials,
        response: Response,
        db: AsyncSession = Depends(get_db),  # noqa: B008 - FastAPI injection idiom
    ):
        try:
            user = await register(db, body.email, body.password)
        except AuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        set_session_cookie(response, issue_session(settings, user.id))
        return {"id": user.id, "email": user.email}

    @app.post("/api/auth/login")
    async def login(
        body: Credentials,
        response: Response,
        db: AsyncSession = Depends(get_db),  # noqa: B008 - FastAPI injection idiom
    ):
        try:
            user = await authenticate(db, body.email, body.password)
        except AuthError as exc:
            # 401 with the module's single message; see auth.py on why it does
            # not say which half was wrong.
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        set_session_cookie(response, issue_session(settings, user.id))
        return {"id": user.id, "email": user.email}

    @app.post("/api/auth/logout")
    async def logout(response: Response):
        response.delete_cookie(SESSION_COOKIE, path="/")
        return {"ok": True}

    @app.get("/api/me")
    async def me(user_id: str = Depends(current_user)):
        return {"id": user_id}

    @app.post("/api/sessions")
    async def create_session(
        body: SessionRequest,
        db: AsyncSession = Depends(get_db),  # noqa: B008 - FastAPI injection idiom
        user_id: str = Depends(current_user),
    ):
        try:
            return await open_session(db, settings, body.avatar_id, user_id, dispatcher)
        except TenantError as exc:
            # 404, not 403. The caller is authenticated but this avatar is not
            # theirs, and saying "forbidden" would confirm it exists.
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ConsentError as exc:
            # 403 rather than 404 here: the avatar is theirs, it is real, and
            # the reason it cannot be called is permission they can act on.
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    app.include_router(build_avatar_router(settings, current_user, get_db, store))
    app.include_router(
        build_router(settings, current_user, get_db, store, runner)
    )

    return app


app = create_app()
