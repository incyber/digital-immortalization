"""Deleting a person, and meaning it.

The storage layer documented delete_tenant as satisfying a legal deletion
obligation and nothing called it. There was no way through the product to
remove a voice clone, a built likeness, an avatar or an account - only a photo
set. For a product holding a grieving family's photographs, "delete my father's
photographs" being partially true is worse than most bugs.

These tests defend the two rules that make the difference: blobs go before
rows, and a partial erasure is never reported as complete.
"""

import pytest
import pytest_asyncio

from avatar.gateway.erasure import Erasure, erase_account, erase_avatar
from avatar.gateway.models import Photo, PhotoSet
from avatar.gateway.tenancy import TenantError
from avatar.storage.keys import photo_key, voice_key


class RecordingStore:
    """A store that remembers what it was asked to remove.

    The order matters more than the count: a row deleted before its blobs
    leaves objects nothing references and nobody can find, which is the one
    outcome that cannot be corrected later.
    """

    def __init__(self, missing: set[str] | None = None, broken: set[str] | None = None):
        self.deleted: list[str] = []
        self.swept: list[str] = []
        self.missing = missing or set()
        self.broken = broken or set()

    async def delete(self, tenant_id: str, key: str) -> None:
        if key in self.broken:
            raise RuntimeError("bucket unreachable")
        if key in self.missing:
            raise FileNotFoundError(key)
        self.deleted.append(key)

    async def delete_tenant(self, tenant_id: str) -> int:
        self.swept.append(tenant_id)
        return 3


@pytest_asyncio.fixture
async def furnished(db, avatar, owner):
    """An avatar with photographs, a voice recording and built assets."""
    photo_set = PhotoSet(owner_id=owner.id, avatar_id=avatar.id)
    db.add(photo_set)
    await db.flush()

    for i in range(3):
        db.add(
            Photo(
                photo_set_id=photo_set.id,
                owner_id=owner.id,
                filename=f"photo-{i}.jpg",
                content_type="image/jpeg",
                size_bytes=1024,
                blob_key=photo_key(owner.id, photo_set.id, f"photo-{i}.jpg"),
                accepted=True,
            )
        )

    avatar.voice_key = voice_key(owner.id, avatar.id)
    avatar.assets_key = f"assets/avatars/{avatar.id}"
    avatar.photo_set_id = photo_set.id
    await db.commit()
    return avatar, owner, photo_set


@pytest.mark.asyncio
async def test_erasing_an_avatar_removes_photographs_voice_and_likeness(db, furnished):
    avatar, owner, _ = furnished
    store = RecordingStore()

    result = await erase_avatar(db, store, avatar.id, owner.id)

    assert result.complete
    assert result.avatars == 1
    assert result.photos == 3
    assert any("voice.wav" in k for k in store.deleted), "the voice clone must go"
    assert any("assets/avatars" in k for k in store.deleted), "the likeness must go"


@pytest.mark.asyncio
async def test_the_consent_record_goes_with_the_recreation(db, furnished):
    """It authorises a recreation. With no recreation, keeping a grieving
    family's names and relationships serves nobody."""
    avatar, owner, _ = furnished

    result = await erase_avatar(db, RecordingStore(), avatar.id, owner.id)

    assert result.consent_records == 1


@pytest.mark.asyncio
async def test_a_stranger_cannot_erase_someone_elses_avatar(db, furnished, other_owner):
    """The most destructive route in the product, so it is asserted here too."""
    avatar, _, _ = furnished

    with pytest.raises(TenantError):
        await erase_avatar(db, RecordingStore(), avatar.id, other_owner.id)


@pytest.mark.asyncio
async def test_an_object_already_gone_counts_as_removed(db, furnished):
    """A second attempt at a half-finished erasure must be able to complete."""
    avatar, owner, photo_set = furnished
    store = RecordingStore(missing={photo_key(owner.id, photo_set.id, "photo-1.jpg")})

    result = await erase_avatar(db, store, avatar.id, owner.id)

    assert result.complete


@pytest.mark.asyncio
async def test_an_object_that_refuses_to_go_is_reported_not_swallowed(db, furnished):
    """Reporting a partial erasure as done is the one outcome nobody re-checks."""
    avatar, owner, photo_set = furnished
    store = RecordingStore(broken={photo_key(owner.id, photo_set.id, "photo-2.jpg")})

    result = await erase_avatar(db, store, avatar.id, owner.id)

    assert not result.complete
    assert len(result.failures) == 1


@pytest.mark.asyncio
async def test_one_unreachable_object_does_not_strand_the_rest(db, furnished):
    """A family asked for their data to go; one bad key must not stop that."""
    avatar, owner, photo_set = furnished
    store = RecordingStore(broken={photo_key(owner.id, photo_set.id, "photo-0.jpg")})

    await erase_avatar(db, store, avatar.id, owner.id)

    assert any("photo-1" in k for k in store.deleted)
    assert any("photo-2" in k for k in store.deleted)


@pytest.mark.asyncio
async def test_erasing_an_account_sweeps_the_whole_storage_prefix(db, furnished):
    """The sweep catches whatever an interrupted earlier attempt orphaned."""
    _, owner, _ = furnished
    store = RecordingStore()

    result = await erase_account(db, store, owner.id)

    assert result.complete
    assert result.avatars >= 1
    assert store.swept == [owner.id]


def test_a_partial_erasure_is_never_reported_as_complete():
    assert Erasure().complete is True
    assert Erasure(failures=["a-key: bucket unreachable"]).complete is False
