"""Accounts and sessions.

Deliberately small. Registration, sign-in, and a signed cookie value. No
password reset, no email verification, no OAuth - each of those is a real
feature with its own failure modes, and stubbing them here would look like
they existed.

Two decisions worth stating, because both make the code look less helpful than
it could be:

  Failed sign-in returns one message whether the account is unknown or the
  password is wrong. For this product the login form would otherwise answer
  the question "has this person buried someone", which is not ours to answer.

  Authentication always runs a hash, even for an unknown email. Skipping it
  makes unknown accounts measurably faster to reject, which is the same
  disclosure by another route.
"""

from __future__ import annotations

import hmac
import json
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode
from hashlib import sha256

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.config import Settings
from avatar.gateway.models import User

# Argon2id at the library's defaults, which track current guidance.
_hasher = PasswordHasher()

# Long enough to matter, short enough not to push people towards reuse.
MIN_PASSWORD_LENGTH = 12

# One week. Long enough that a grieving family is not repeatedly logged out,
# short enough that a stolen cookie expires.
SESSION_TTL_SECONDS = 7 * 24 * 3600

# Every sign-in failure says this.
_SIGNIN_REFUSED = "email or password is incorrect"

# Hashed once at import and verified against when the email is unknown, so an
# unknown account costs the same time as a wrong password.
_DUMMY_HASH = _hasher.hash("a-password-that-is-never-anybodys")


class AuthError(ValueError):
    pass


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(digest: str, password: str) -> bool:
    """False on mismatch and on a malformed digest. Never raises."""
    try:
        return _hasher.verify(digest, password)
    except VerifyMismatchError:
        return False
    except Exception:  # noqa: BLE001 - a corrupt row must fail closed
        return False


def normalise_email(email: str) -> str:
    return email.strip().lower()


async def register(db: AsyncSession, email: str, password: str) -> User:
    """Create an account.

    Each account is a tenant: everything it uploads is scoped to it and
    reachable by nothing else.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")

    address = normalise_email(email)
    if not address or "@" not in address:
        raise AuthError("a valid email address is required")

    existing = (
        await db.execute(select(User).where(User.email == address))
    ).scalar_one_or_none()
    if existing is not None:
        raise AuthError("that email address is already registered")

    user = User(email=address, password_hash=hash_password(password))
    db.add(user)
    await db.commit()
    return user


async def authenticate(db: AsyncSession, email: str, password: str) -> User:
    """Return the account, or raise with a message that reveals nothing."""
    address = normalise_email(email)
    user = (
        await db.execute(select(User).where(User.email == address))
    ).scalar_one_or_none()

    if user is None:
        # Burn equivalent time so an unknown email is not faster to reject.
        verify_password(_DUMMY_HASH, password)
        raise AuthError(_SIGNIN_REFUSED)

    if not verify_password(user.password_hash, password):
        raise AuthError(_SIGNIN_REFUSED)

    return user


def _sign(secret: str, payload: bytes) -> str:
    return urlsafe_b64encode(hmac.new(secret.encode(), payload, sha256).digest()).decode()


def issue_session(cfg: Settings, user_id: str, ttl_seconds: int = SESSION_TTL_SECONDS) -> str:
    """A signed, expiring bearer of one user id.

    Signed rather than random-and-stored so the gateway stays stateless. The
    trade is that revocation needs a rotated secret or a deny list, neither of
    which exists yet - noted rather than pretended away.
    """
    payload = json.dumps({"sub": user_id, "exp": int(time.time()) + ttl_seconds}).encode()
    body = urlsafe_b64encode(payload).decode()
    return f"{body}.{_sign(cfg.session_secret, payload)}"


def read_session(cfg: Settings, token: str) -> str | None:
    """The user id carried by a valid, unexpired token, else None."""
    try:
        body, _, signature = token.rpartition(".")
        if not body or not signature:
            return None
        payload = urlsafe_b64decode(body.encode())

        # compare_digest rather than ==, so a forged signature cannot be
        # refined a byte at a time by timing the comparison.
        if not hmac.compare_digest(_sign(cfg.session_secret, payload), signature):
            return None

        claims = json.loads(payload)
        if int(claims.get("exp", 0)) <= time.time():
            return None
        return str(claims["sub"])
    except Exception:  # noqa: BLE001 - any malformed token is simply invalid
        return None
