"""Database wiring. Async engine, session factory, and schema creation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from loguru import logger
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
    """Bring the schema up to date: create missing tables, add missing columns.

    create_all alone adds tables and silently ignores columns, which is fine on
    a fresh checkout and wrong everywhere else. A column added to a model does
    not appear in a database that already exists, and the failure surfaces as a
    500 the first time somebody saves a record - which is exactly how it
    surfaced here, on a deployment whose data lives on a volume that survives
    every release.

    Only additive changes are made. A column present in the model and absent in
    the database is added; nothing is renamed, retyped or dropped, because
    guessing at those is how an automatic migration destroys data. Anything
    beyond adding a nullable column needs a real migration written by hand.
    """
    engine = init_engine(cfg)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


def _add_missing_columns(connection) -> None:
    from sqlalchemy import inspect, text
    from sqlalchemy.schema import CreateColumn

    inspector = inspect(connection)
    existing_tables = set(inspector.get_table_names())

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        present = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue
            # Rendered by the dialect rather than assembled by hand, so a type
            # this project uses and this function has never seen still comes
            # out correct.
            spec = CreateColumn(column).compile(connection.engine).string
            # A NOT NULL column cannot be added to a table with rows in it and
            # no default. Added as nullable instead: a wrong constraint is
            # recoverable, a failed start on a live deployment is not.
            spec = spec.replace(" NOT NULL", "")
            logger.info(f"adding missing column {table.name}.{column.name}")
            connection.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN {spec}'))


async def get_db() -> AsyncIterator[AsyncSession]:
    assert _factory is not None, "init_engine must run before requests are served"
    async with _factory() as session:
        yield session


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """A session for work that is not a request.

    get_db is a FastAPI dependency and only usable as one. Startup checks and
    background work need the same session with an ordinary `async with`, and
    writing `async for ... break` at those call sites hides a resource leak one
    edit away.
    """
    assert _factory is not None, "init_engine must run before a session is opened"
    async with _factory() as session:
        yield session
