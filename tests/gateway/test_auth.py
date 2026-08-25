"""Registration, sign-in, and what a session token is allowed to prove."""

import pytest

from avatar.gateway.auth import (
    AuthError,
    authenticate,
    hash_password,
    issue_session,
    read_session,
    register,
    verify_password,
)


def test_passwords_are_hashed_not_stored():
    digest = hash_password("correct horse battery staple")
    assert "correct horse" not in digest
    assert digest.startswith("$argon2")


def test_same_password_hashes_differently_each_time():
    # Per-hash salt. Identical digests would reveal which accounts share a
    # password.
    assert hash_password("hunter2") != hash_password("hunter2")


def test_verify_accepts_the_right_password():
    assert verify_password(hash_password("hunter2"), "hunter2")


def test_verify_rejects_the_wrong_one():
    assert not verify_password(hash_password("hunter2"), "hunter3")


def test_verify_on_a_corrupt_hash_is_false_not_an_exception():
    # A malformed row must fail closed, not 500.
    assert not verify_password("not-a-hash", "anything")


async def test_registration_creates_an_account(db):
    user = await register(db, "grieving@example.com", "a-long-enough-password")
    assert user.email == "grieving@example.com"
    assert user.password_hash != "a-long-enough-password"


async def test_email_is_normalised(db):
    await register(db, "Mixed.Case@Example.COM ", "a-long-enough-password")
    assert await authenticate(db, "mixed.case@example.com", "a-long-enough-password")


async def test_duplicate_registration_is_refused(db):
    await register(db, "dup@example.com", "a-long-enough-password")
    with pytest.raises(AuthError):
        await register(db, "dup@example.com", "a-long-enough-password")


async def test_short_passwords_are_refused(db):
    with pytest.raises(AuthError, match="at least"):
        await register(db, "short@example.com", "abc")


async def test_authenticate_rejects_a_wrong_password(db):
    await register(db, "user@example.com", "a-long-enough-password")
    with pytest.raises(AuthError):
        await authenticate(db, "user@example.com", "wrong-password-entirely")


async def test_authenticate_gives_the_same_error_for_unknown_and_wrong(db):
    # Distinguishable errors turn the login form into an account-enumeration
    # oracle, which for this product reveals who has buried someone.
    await register(db, "known@example.com", "a-long-enough-password")
    with pytest.raises(AuthError) as wrong:
        await authenticate(db, "known@example.com", "not-the-password")
    with pytest.raises(AuthError) as unknown:
        await authenticate(db, "nobody@example.com", "not-the-password")
    assert str(wrong.value) == str(unknown.value)


def test_session_token_round_trips(cfg):
    token = issue_session(cfg, "user-123")
    assert read_session(cfg, token) == "user-123"


def test_a_tampered_token_is_rejected(cfg):
    token = issue_session(cfg, "user-123")
    body, _, signature = token.rpartition(".")
    forged = body + "." + ("a" * len(signature))
    assert read_session(cfg, forged) is None


def test_a_token_signed_with_another_secret_is_rejected(cfg):
    from avatar.config import Settings

    other = Settings(_env_file=None, session_secret="a-completely-different-secret-value")
    assert read_session(cfg, issue_session(other, "user-123")) is None


def test_an_expired_token_is_rejected(cfg):
    assert read_session(cfg, issue_session(cfg, "user-123", ttl_seconds=-1)) is None
