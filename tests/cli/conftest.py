import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from avatar.gateway.models import Avatar, Base, ConsentRecord, ConsentStatus, User


@pytest_asyncio.fixture
async def db_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.fixture(autouse=True)
def _cli_uses_the_test_db(monkeypatch, db_factory):
    # cmd_set/cmd_list build their own engine from get_settings() by default,
    # which is the right thing in production and the wrong thing in a test
    # that must not touch ./avatar.db. Repointed at an in-memory database that
    # dies with the test.
    import avatar.cli.consent as consent_cli

    monkeypatch.setattr(consent_cli, "_factory", lambda: db_factory)


@pytest_asyncio.fixture
async def avatar_without_evidence(db_factory):
    async with db_factory() as db:
        owner = User(email="family@example.com", password_hash="x")
        db.add(owner)
        await db.flush()
        avatar = Avatar(
            owner_id=owner.id,
            display_name="Tomás Duarte",
            country="US",
            biography="A bookbinder from Porto.",
        )
        db.add(avatar)
        await db.flush()
        db.add(
            ConsentRecord(
                avatar_id=avatar.id,
                rights_holder_name="Ana Chen",
                relationship_to_subject="daughter",
                jurisdiction="US-CA",
                status=ConsentStatus.PENDING,
            )
        )
        await db.commit()
        return avatar.id


@pytest_asyncio.fixture
async def avatar_with_evidence(db_factory):
    async with db_factory() as db:
        owner = User(email="family2@example.com", password_hash="x")
        db.add(owner)
        await db.flush()
        avatar = Avatar(
            owner_id=owner.id,
            display_name="Marguerite Chen",
            country="US",
            biography="A cellist from Vancouver.",
        )
        db.add(avatar)
        await db.flush()
        db.add(
            ConsentRecord(
                avatar_id=avatar.id,
                rights_holder_name="Ana Chen",
                relationship_to_subject="daughter",
                jurisdiction="US-CA",
                status=ConsentStatus.PENDING,
                evidence_s3_key=f"tenants/{owner.id}/avatars/{avatar.id}/evidence.pdf",
            )
        )
        await db.commit()
        return avatar.id
