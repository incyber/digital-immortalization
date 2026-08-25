import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from avatar.config import Settings
from avatar.gateway.models import Avatar, Base, ConsentRecord, ConsentStatus, User
from tests.gateway.helpers import set_status


@pytest.fixture
def cfg():
    return Settings(_env_file=None, database_url="sqlite+aiosqlite:///:memory:")


@pytest_asyncio.fixture
async def db(cfg):
    engine = create_async_engine(cfg.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def owner(db: AsyncSession):
    u = User(email="owner@example.com", password_hash="x")
    db.add(u)
    await db.commit()
    return u


@pytest_asyncio.fixture
async def avatar(db: AsyncSession, owner):
    a = Avatar(owner_id=owner.id, display_name="Test", profile_path="p.json")
    db.add(a)
    await db.flush()
    db.add(
        ConsentRecord(
            avatar_id=a.id,
            rights_holder_name="Executor",
            relationship_to_subject="daughter",
            jurisdiction="US-CA",
            status=ConsentStatus.PENDING,
        )
    )
    await db.commit()
    return a


@pytest_asyncio.fixture
async def avatar_without_consent(db: AsyncSession, owner):
    a = Avatar(owner_id=owner.id, display_name="Orphan", profile_path="p.json")
    db.add(a)
    await db.commit()
    return a


@pytest_asyncio.fixture
async def verified_avatar(db: AsyncSession, avatar):
    await set_status(db, avatar, "verified")
    return avatar


@pytest_asyncio.fixture
async def other_owner(db: AsyncSession):
    """A second tenant. Every isolation test needs somebody to be excluded."""
    u = User(email="stranger@example.com", password_hash="x")
    db.add(u)
    await db.commit()
    return u
