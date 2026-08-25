"""Object keys are the isolation boundary.

Every object in the store lives under a prefix owned by exactly one tenant. If
a key can be made to escape its prefix, tenancy is decoration. These tests are
written as attacks.
"""

import pytest

from avatar.storage.keys import KeyError_, photo_key, tenant_prefix


def test_prefix_contains_the_tenant():
    assert tenant_prefix("tenant-a").startswith("tenants/tenant-a/")


def test_two_tenants_never_share_a_prefix():
    a, b = tenant_prefix("tenant-a"), tenant_prefix("tenant-b")
    assert not a.startswith(b) and not b.startswith(a)


def test_a_prefix_is_not_a_prefix_of_a_longer_tenant_id():
    # "tenants/abc/" must not prefix-match "tenants/abcd/...". Without the
    # trailing separator a list operation would cross tenants.
    assert not tenant_prefix("abcd").startswith(tenant_prefix("abc"))


@pytest.mark.parametrize(
    "hostile",
    [
        "../other-tenant",
        "a/../../b",
        "..",
        "./.",
        "a/./b",
        "with space",
        "semi;colon",
        "null\x00byte",
        "",
        "   ",
        "a" * 200,
    ],
)
def test_hostile_tenant_ids_are_refused(hostile):
    with pytest.raises(KeyError_):
        tenant_prefix(hostile)


@pytest.mark.parametrize("hostile", ["../escape.jpg", "a/../../b.jpg", "", "sub/dir.jpg"])
def test_hostile_filenames_are_refused(hostile):
    with pytest.raises(KeyError_):
        photo_key("tenant-a", "set-1", hostile)


def test_photo_key_is_inside_the_tenant_prefix():
    key = photo_key("tenant-a", "set-1", "portrait.jpg")
    assert key.startswith(tenant_prefix("tenant-a"))
    assert key.endswith("portrait.jpg")


def test_photo_key_separates_photo_sets():
    a = photo_key("tenant-a", "set-1", "p.jpg")
    b = photo_key("tenant-a", "set-2", "p.jpg")
    assert a != b


def test_real_artefact_extensions_are_accepted():
    # Model weights are ".safetensors", which is longer than a naive
    # extension limit allows.
    from avatar.storage.keys import asset_key

    for name in ("identity.safetensors", "idle.mp4", "plates.npz", "assets.json"):
        assert asset_key("tenant-a", "avatar-1", name).endswith(name)


def test_voice_keys_are_inside_the_tenant_prefix():
    from avatar.storage.keys import tenant_prefix, voice_key

    key = voice_key("tenant-a", "avatar-1")
    assert key.startswith(tenant_prefix("tenant-a"))
    assert key.endswith("voice.wav")


def test_voice_keys_differ_between_avatars():
    from avatar.storage.keys import voice_key

    assert voice_key("tenant-a", "avatar-1") != voice_key("tenant-a", "avatar-2")


def test_a_hostile_avatar_id_cannot_reach_another_tenants_voice():
    from avatar.storage.keys import KeyError_, voice_key

    with pytest.raises(KeyError_):
        voice_key("tenant-a", "../tenant-b/avatars/x")
