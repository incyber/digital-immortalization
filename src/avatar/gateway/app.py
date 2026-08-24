"""HTTP surface.

Deliberately small. Everything that matters about a session request happens in
consent.py and sessions.py; this layer only translates exceptions into status
codes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.config import Settings, get_settings
from avatar.gateway.consent import ConsentError
from avatar.gateway.db import create_all, get_db
from avatar.gateway.sessions import open_session


class SessionRequest(BaseModel):
    avatar_id: str


def create_app(cfg: Settings | None = None) -> FastAPI:
    settings = cfg or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await create_all(settings)
        yield

    app = FastAPI(title="Avatar gateway", lifespan=lifespan)

    # The web client is served from a different origin in development. Listed
    # explicitly rather than wildcarded, because these responses carry tokens.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/api/sessions")
    async def create_session(body: SessionRequest, db: AsyncSession = Depends(get_db)):
        try:
            return await open_session(db, settings, body.avatar_id)
        except ConsentError as exc:
            # 403 rather than 404 even when no record exists: the avatar may be
            # real, and the reason it cannot be called is permission.
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    return app


app = create_app()
