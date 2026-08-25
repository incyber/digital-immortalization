"""Object key construction.

Keys are the isolation boundary in blob storage. There are no per-object
permissions in an object store worth relying on; what keeps one family's
photographs away from another is that every key is built here, from a
validated tenant id, and that nothing can construct a key that escapes its
prefix.

So this module is strict to the point of being annoying: identifiers must be
plain, short, and free of separators, and anything else raises rather than
being sanitised. Sanitising is how "../" becomes "" becomes a collision.
"""

from __future__ import annotations

import re

# Deliberately narrow. Tenant and set ids are ours - uuids - so nothing
# legitimate needs anything outside this.
_SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

# Filenames may carry a single dot for the extension and nothing else.
_SAFE_FILENAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}\.[a-zA-Z0-9]{1,10}$")

# Trailing slash is load-bearing: without it "tenants/abc" prefix-matches
# "tenants/abcd/secret.jpg" and a list operation crosses tenants.
_ROOT = "tenants"


class KeyError_(ValueError):
    """Raised for any identifier that cannot safely become part of a key.

    Named with a trailing underscore to avoid shadowing the builtin KeyError,
    which means something entirely different.
    """


def _check_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.match(value):
        raise KeyError_(
            f"{label} must be 1-64 characters of letters, digits, hyphen or "
            f"underscore, starting alphanumeric; got {value!r}"
        )
    return value


def tenant_prefix(tenant_id: str) -> str:
    """Everything belonging to one tenant lives under this, and nothing else does."""
    return f"{_ROOT}/{_check_id(tenant_id, 'tenant id')}/"


def photo_set_prefix(tenant_id: str, photo_set_id: str) -> str:
    return f"{tenant_prefix(tenant_id)}photos/{_check_id(photo_set_id, 'photo set id')}/"


def photo_key(tenant_id: str, photo_set_id: str, filename: str) -> str:
    """Key for one uploaded image.

    The filename is validated rather than cleaned. A caller passing "../x.jpg"
    has a bug or is attacking; either way the right answer is to refuse.
    """
    if not isinstance(filename, str) or not _SAFE_FILENAME.match(filename):
        raise KeyError_(
            f"filename must be a plain name with an extension and no path "
            f"separators; got {filename!r}"
        )
    return f"{photo_set_prefix(tenant_id, photo_set_id)}{filename}"


def asset_key(tenant_id: str, avatar_id: str, filename: str) -> str:
    """Key for a derived artefact - idle clip, plates, trained weights."""
    if not isinstance(filename, str) or not _SAFE_FILENAME.match(filename):
        raise KeyError_(f"filename is not a plain name; got {filename!r}")
    return f"{tenant_prefix(tenant_id)}avatars/{_check_id(avatar_id, 'avatar id')}/{filename}"


def belongs_to(key: str, tenant_id: str) -> bool:
    """Whether a key lies inside a tenant's prefix.

    Used as a last check before any read or delete, so that a key arriving from
    a database row that was somehow wrong cannot reach across tenants.
    """
    return key.startswith(tenant_prefix(tenant_id))
