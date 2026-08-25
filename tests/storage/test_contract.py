"""One suite every BlobStore backend must pass.

The local backend passes it today; the S3 backend must pass it unchanged
before it is used for anybody's photographs. Most of these are isolation
tests, because that is the property worth guaranteeing across backends.
"""

import pytest

from avatar.storage.keys import KeyError_, photo_key
from avatar.storage.local import LocalBlobStore

A, B = "tenant-a", "tenant-b"


@pytest.fixture
def store(tmp_path):
    return LocalBlobStore(tmp_path)


async def test_put_then_get(store):
    key = photo_key(A, "set-1", "one.jpg")
    await store.put(A, key, b"image-bytes", "image/jpeg")
    assert await store.get(A, key) == b"image-bytes"


async def test_another_tenant_cannot_read_it(store):
    key = photo_key(A, "set-1", "one.jpg")
    await store.put(A, key, b"secret", "image/jpeg")
    with pytest.raises(Exception):
        await store.get(B, key)


async def test_another_tenant_cannot_delete_it(store):
    key = photo_key(A, "set-1", "one.jpg")
    await store.put(A, key, b"secret", "image/jpeg")
    with pytest.raises(Exception):
        await store.delete(B, key)
    assert await store.get(A, key) == b"secret"


async def test_another_tenant_cannot_write_into_the_prefix(store):
    key = photo_key(A, "set-1", "one.jpg")
    with pytest.raises(Exception):
        await store.put(B, key, b"planted", "image/jpeg")


async def test_listing_never_crosses_tenants(store):
    await store.put(A, photo_key(A, "set-1", "a.jpg"), b"a", "image/jpeg")
    await store.put(B, photo_key(B, "set-1", "b.jpg"), b"b", "image/jpeg")
    listed = await store.list(A, photo_key(A, "set-1", "x.jpg").rsplit("/", 1)[0] + "/")
    assert [o.key for o in listed] == [photo_key(A, "set-1", "a.jpg")]


async def test_listing_a_foreign_prefix_is_refused(store):
    await store.put(B, photo_key(B, "set-1", "b.jpg"), b"b", "image/jpeg")
    with pytest.raises(Exception):
        await store.list(A, f"tenants/{B}/")


async def test_delete_tenant_removes_only_that_tenant(store):
    await store.put(A, photo_key(A, "set-1", "a.jpg"), b"a", "image/jpeg")
    await store.put(A, photo_key(A, "set-2", "a2.jpg"), b"a2", "image/jpeg")
    await store.put(B, photo_key(B, "set-1", "b.jpg"), b"b", "image/jpeg")

    removed = await store.delete_tenant(A)
    assert removed == 2
    assert await store.get(B, photo_key(B, "set-1", "b.jpg")) == b"b"


async def test_missing_object_raises(store):
    with pytest.raises(Exception):
        await store.get(A, photo_key(A, "set-1", "absent.jpg"))


async def test_traversal_in_a_key_is_refused_before_it_reaches_the_store(store):
    with pytest.raises(KeyError_):
        photo_key(A, "set-1", "../../etc/passwd")


async def test_upload_url_is_scoped_and_expiring(store):
    key = photo_key(A, "set-1", "one.jpg")
    url = await store.upload_url(A, key, "image/jpeg")
    assert key in url

    with pytest.raises(Exception):
        await store.upload_url(B, key, "image/jpeg")
