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
from avatar.gateway import demo
from avatar.gateway.auth import (
    SESSION_TTL_SECONDS,
    AuthError,
    authenticate,
    issue_session,
    read_session,
    register,
)
from avatar.gateway.consent import ConsentError
from avatar.gateway.db import create_all, get_db, session_scope
from avatar.gateway.dispatch import LocalProcessDispatcher
from avatar.gateway.routes_avatars import build_router as build_avatar_router
from avatar.gateway.routes_ingest import build_router, sweep_spool
from avatar.gateway.routes_splat import build_router as build_splat_router
from avatar.gateway.sessions import (
    assert_demo_mode_safe,
    assert_production_ready,
    open_session,
)
from avatar.gateway.tenancy import TenantError
from avatar.gateway.web import mount_web, resolve_web_root
from avatar.ingest.video_service import VideoIngestService
from avatar.splat.service import SplatService, build_splat_builder
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

    # Before anything is constructed. A deployment that reaches this line with
    # development credentials is one where anyone can forge a session cookie or
    # a room token, and refusing to boot is the only useful response.
    if settings.production:
        assert_production_ready(settings)

    dispatcher = LocalProcessDispatcher(settings)
    store = build_store(settings)
    runner = build_runner(settings, store)
    # A splat build outlives the request that starts it, so the service is
    # given the session source rather than a session: it writes the outcome
    # long after the response has gone.
    splat = SplatService(build_splat_builder(settings, store), store, get_db)
    # An uploaded clip is minutes of frame checks, and it used to run inside
    # the request that delivered it - see ingest/video_service.py. Given the
    # session source for the same reason the splat service is: it writes sixty
    # rows long after the response has gone.
    video_jobs = VideoIngestService(store, get_db)

    web_root = resolve_web_root(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await create_all(settings)

        # A clip spooled to the volume by a process that then died is a
        # quarter of a gigabyte nothing will ever read. Startup is exactly
        # when that has just happened, so it is where they are cleared.
        sweep_spool(settings)

        # The half of the demo-mode check that needs a database. Deliberately
        # here rather than in create_app: the tables do not exist until the
        # line above, and a check that runs before the thing it inspects is
        # the kind of dead safety control this project has already been bitten
        # by twice. Raising here stops the process before it serves anything.
        if settings.demo_mode:
            async with session_scope() as db:
                assert_demo_mode_safe(settings, await demo.real_account_emails(db))

        yield
        # Agents outlive a request but not the gateway. Leaving them running
        # would hold GPU capacity and microphones open after a restart.
        dispatcher.shutdown()
        # Same rule for splat builds: a gateway going down must not leave a
        # GPU optimising somebody's father with nothing watching it.
        await splat.shutdown()
        # And for clips being read into frames: the tasks are in this process
        # and nothing else will finish them.
        await video_jobs.shutdown()

    app = FastAPI(title="Avatar gateway", lifespan=lifespan)

    # The web client is served from a different origin in development. Listed
    # explicitly rather than wildcarded, because these responses carry room
    # tokens and a wildcard would let any page request one.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        # PATCH and DELETE are needed by the product's own endpoints - editing
        # an avatar and deleting a photo set - and neither is a CORS "simple"
        # method, so a browser preflights them. Omitting them meant that in a
        # cross-origin deployment, which is the one this app is being prepared
        # for, "delete these photographs" failed silently in the browser and
        # never reached this process at all.
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["*"],
    )

    def set_session_cookie(response: Response, token: str) -> None:
        # Lax, because this process serves the site as well as the API and the
        # browser is therefore always making a first-party request. That is the
        # whole reason the two were brought onto one origin: a cross-site
        # deployment needs samesite=none, Safari refuses third-party cookies
        # outright and Chrome refuses them in common configurations, and the
        # result was a sign-in that returned 200, lost its cookie, and bounced
        # back to the sign-in page forever.
        #
        # COOKIE_SAMESITE=none is still available for a split-origin
        # deployment. Browsers only accept it over https, so it is downgraded
        # rather than honoured when the cookie is not also Secure - a cookie
        # the browser silently discards is worse than a lax one, because the
        # failure looks like the server.
        samesite = settings.cookie_samesite.lower()
        if samesite == "none" and not settings.cookies_secure:
            samesite = "lax"

        response.set_cookie(
            SESSION_COOKIE,
            token,
            httponly=True,
            samesite=samesite,  # type: ignore[arg-type]
            secure=settings.cookies_secure,
            max_age=SESSION_TTL_SECONDS,
            path="/",
        )

    async def current_user(
        response: Response,
        avatar_session: str | None = Cookie(default=None),
        db: AsyncSession = Depends(get_db),  # noqa: B008 - FastAPI injection idiom
    ) -> str:
        """The signed-in tenant, or 401.

        Every authenticated route depends on this. It returns an id rather than
        a User so that handlers cannot accidentally hold a detached row across
        a request boundary.

        Demo mode changes exactly one thing here: a visitor with no valid
        session is given the shared demo account and its cookie instead of a
        401. It is done in this one function rather than in a sign-in endpoint
        the site would have to call, because that keeps the property "there is
        one place that decides who you are" - the same reason the consent gate
        and the tenancy check each live in one module.

        Note what it does not change. It hands back an ordinary user id, so
        ownership, consent and the synthetic-media declaration all apply to a
        demo visitor exactly as they apply to a customer.
        """
        user_id = read_session(settings, avatar_session or "")
        if user_id is not None:
            return user_id

        if settings.demo_mode:
            user = await demo.ensure(db)
            set_session_cookie(response, issue_session(settings, user.id))
            return user.id

        raise HTTPException(status_code=401, detail="sign in required")

    def refuse_credentials_in_demo_mode() -> None:
        """Demo mode has one account and it is not reachable by password.

        Registering would create a second tenant, which is the one thing the
        demo must not do: the shared account only stays shared because it is
        the only account, and the startup check that guards real customer data
        is written in exactly those terms. Refusing here is what keeps that
        invariant true at runtime rather than merely at boot.
        """
        if settings.demo_mode:
            raise HTTPException(
                status_code=409,
                detail=(
                    "this deployment is a shared demo and does not take accounts; "
                    "you are already signed in"
                ),
            )

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/api/config")
    async def public_config():
        """What the site must know before it renders anything.

        Only demo_mode today, and it is not a convenience: the interface has to
        state on every screen that this is a shared account whose contents
        anyone with the link can read, and it cannot state that unless it is
        told. See components/DemoBanner.tsx.
        """
        return {"demo_mode": settings.demo_mode}

    @app.post("/api/auth/register", status_code=201)
    async def register_account(
        body: Credentials,
        response: Response,
        db: AsyncSession = Depends(get_db),  # noqa: B008 - FastAPI injection idiom
    ):
        refuse_credentials_in_demo_mode()
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
        refuse_credentials_in_demo_mode()
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
        build_router(settings, current_user, get_db, store, runner, video_jobs)
    )
    app.include_router(build_splat_router(current_user, get_db, splat))

    # Last, and it has to be: the site is served by a catch-all, and FastAPI
    # matches in registration order, so mounting it any earlier would shadow
    # every route above. None means no build is present, which is how the tests
    # and the split-port development setup run - API only.
    if web_root is not None:
        mount_web(app, web_root)

    return app


app = create_app()
