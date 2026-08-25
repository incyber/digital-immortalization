import cv2
import numpy as np
import pytest
import pytest_asyncio

from avatar.gateway.models import Base, PhotoSetStatus, User
from avatar.gateway.tenancy import TenantError
from avatar.ingest.service import (
    MAX_UPLOADS,
    add_photo,
    create_photo_set,
    delete_photo_set,
    evaluate_set,
    get_photo_set,
)
from avatar.storage.local import LocalBlobStore


@pytest_asyncio.fixture
async def db(tmp_path):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session
    await engine.dispose()


@pytest.fixture
def store(tmp_path):
    return LocalBlobStore(tmp_path / "blobs")


@pytest_asyncio.fixture
async def owner(db):
    u = User(email="a@example.com", password_hash="x")
    db.add(u)
    await db.commit()
    return u


@pytest_asyncio.fixture
async def stranger(db):
    u = User(email="b@example.com", password_hash="x")
    db.add(u)
    await db.commit()
    return u


def an_image(w=1024, h=1024) -> bytes:
    rng = np.random.default_rng(1)
    frame = (np.full((h, w, 3), 130, np.uint8) + rng.integers(-30, 30, (h, w, 3))).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


async def test_photo_set_belongs_to_its_creator(db, owner, stranger):
    ps = await create_photo_set(db, owner.id)
    assert (await get_photo_set(db, ps.id, owner.id)).id == ps.id
    with pytest.raises(TenantError):
        await get_photo_set(db, ps.id, stranger.id)


async def test_a_stranger_cannot_add_photos(db, store, owner, stranger):
    ps = await create_photo_set(db, owner.id)
    with pytest.raises(TenantError):
        await add_photo(db, store, ps.id, stranger.id, "x.jpg", "image/jpeg", an_image())


async def test_stored_key_is_inside_the_owners_prefix(db, store, owner):
    ps = await create_photo_set(db, owner.id)
    photo = await add_photo(db, store, ps.id, owner.id, "x.jpg", "image/jpeg", an_image())
    assert photo.blob_key.startswith(f"tenants/{owner.id}/")


async def test_a_hostile_filename_cannot_escape_the_prefix(db, store, owner):
    ps = await create_photo_set(db, owner.id)
    photo = await add_photo(
        db, store, ps.id, owner.id, "../../../etc/passwd.jpg", "image/jpeg", an_image()
    )
    assert ".." not in photo.blob_key
    assert photo.blob_key.startswith(f"tenants/{owner.id}/")


async def test_unsupported_types_are_refused(db, store, owner):
    ps = await create_photo_set(db, owner.id)
    with pytest.raises(ValueError, match="unsupported"):
        await add_photo(db, store, ps.id, owner.id, "x.gif", "image/gif", an_image())


async def test_oversized_uploads_are_refused(db, store, owner):
    ps = await create_photo_set(db, owner.id)
    with pytest.raises(ValueError, match="larger than"):
        await add_photo(db, store, ps.id, owner.id, "x.jpg", "image/jpeg", b"x" * (26 * 1024 * 1024))


async def test_upload_count_is_capped(db, store, owner, monkeypatch):
    from avatar.ingest import service

    monkeypatch.setattr(service, "MAX_UPLOADS", 2)
    ps = await create_photo_set(db, owner.id)
    for i in range(2):
        await add_photo(db, store, ps.id, owner.id, f"{i}.jpg", "image/jpeg", an_image())
    with pytest.raises(ValueError, match="at most"):
        await add_photo(db, store, ps.id, owner.id, "3.jpg", "image/jpeg", an_image())


async def test_a_faceless_set_is_rejected(db, store, owner):
    ps = await create_photo_set(db, owner.id)
    for i in range(3):
        await add_photo(db, store, ps.id, owner.id, f"{i}.jpg", "image/jpeg", an_image())
    result = await evaluate_set(db, ps.id, owner.id)
    assert result.status is PhotoSetStatus.REJECTED
    assert result.usable_count == 0


async def test_delete_removes_the_images_from_storage(db, store, owner):
    ps = await create_photo_set(db, owner.id)
    await add_photo(db, store, ps.id, owner.id, "x.jpg", "image/jpeg", an_image())
    removed = await delete_photo_set(db, store, ps.id, owner.id)
    assert removed == 1
    with pytest.raises(TenantError):
        await get_photo_set(db, ps.id, owner.id)


async def test_a_stranger_cannot_delete_a_photo_set(db, store, owner, stranger):
    ps = await create_photo_set(db, owner.id)
    await add_photo(db, store, ps.id, owner.id, "x.jpg", "image/jpeg", an_image())
    with pytest.raises(TenantError):
        await delete_photo_set(db, store, ps.id, stranger.id)
    assert (await get_photo_set(db, ps.id, owner.id)).id == ps.id


async def test_max_uploads_is_above_the_accepted_maximum():
    from avatar.ingest.validate import MAX_ACCEPTED

    assert MAX_UPLOADS > MAX_ACCEPTED


async def test_revalidation_reruns_the_current_rules_on_stored_images(db, store, owner):
    # A validator fix must reach photographs already uploaded. Asking a family
    # to gather pictures of someone who has died a second time is not an option.
    from sqlalchemy import select

    from avatar.gateway.models import Photo
    from avatar.ingest.service import revalidate_set

    ps = await create_photo_set(db, owner.id)
    await add_photo(db, store, ps.id, owner.id, "x.jpg", "image/jpeg", an_image())

    photo = (await db.execute(select(Photo).where(Photo.photo_set_id == ps.id))).scalar_one()
    photo.accepted = True          # a stale verdict from older rules
    await db.commit()

    await revalidate_set(db, store, ps.id, owner.id)

    await db.refresh(photo)
    assert photo.accepted is False, "the current rules should reject a faceless image"


async def test_a_stranger_cannot_revalidate_a_photo_set(db, store, owner, stranger):
    from avatar.ingest.service import revalidate_set

    ps = await create_photo_set(db, owner.id)
    with pytest.raises(TenantError):
        await revalidate_set(db, store, ps.id, stranger.id)
