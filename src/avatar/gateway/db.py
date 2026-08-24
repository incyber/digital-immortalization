"""Database wiring. Async engine, session factory, and schema creation."""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from avatar.config import Settings
from avatar.gateway.models import Base

_engine = None
_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine(cfg: Settings):
    global _engine, _factory
    _engine = create_async_engine(cfg.database_url, future=True)
    _factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def create_all(cfg: Settings) -> None:
    """Create tables directly.

    Alembic owns migrations from the first deployment onward; this exists so a
    fresh checkout and the test suite can stand a database up in one call.
    """
    engine = init_engine(cfg)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncIterator[AsyncSession]:
    assert _factory is not None, "init_engine must run before requests are served"
    async with _factory() as session:
        yield session
