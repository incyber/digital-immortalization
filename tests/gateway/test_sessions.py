import pytest

from avatar.gateway.consent import ConsentError
from avatar.gateway.sessions import mint_token, open_session


async def test_open_session_refuses_without_consent(db, cfg, avatar):
    with pytest.raises(ConsentError):
        await open_session(db, cfg, avatar.id)


async def test_open_session_writes_nothing_when_refused(db, cfg, avatar):
    from sqlalchemy import func, select

    from avatar.gateway.models import Session

    with pytest.raises(ConsentError):
        await open_session(db, cfg, avatar.id)
    count = (await db.execute(select(func.count()).select_from(Session))).scalar_one()
    assert count == 0, "a refused request must leave no session row behind"


async def test_open_session_returns_joinable_details(db, cfg, verified_avatar):
    out = await open_session(db, cfg, verified_avatar.id)
    assert out["url"].startswith("ws")
    assert out["room"].startswith("call-")
    assert len(out["token"].split(".")) == 3


async def test_token_is_scoped_to_one_room(cfg):
    import jwt

    token = mint_token(cfg, room="call-abc", identity="human-1", name="Visitor")
    claims = jwt.decode(token, cfg.livekit_api_secret, algorithms=["HS256"])
    assert claims["video"]["room"] == "call-abc"
    assert claims["video"]["roomJoin"] is True


async def test_each_session_gets_a_distinct_room(db, cfg, verified_avatar):
    a = await open_session(db, cfg, verified_avatar.id)
    b = await open_session(db, cfg, verified_avatar.id)
    assert a["room"] != b["room"]


async def test_dev_secret_is_rejected_outside_development(cfg):
    # The LiveKit dev server ships a 6-byte secret. It is fine on localhost and
    # a real weakness anywhere else, so shipping it to cloud must be a failure
    # rather than a warning nobody reads.
    import pytest

    from avatar.gateway.sessions import assert_production_ready

    with pytest.raises(ValueError, match="development secret"):
        assert_production_ready(cfg)


async def test_agent_is_dispatched_into_the_room(db, cfg, verified_avatar):
    from avatar.gateway.dispatch import NullDispatcher

    dispatcher = NullDispatcher()
    out = await open_session(db, cfg, verified_avatar.id, dispatcher)
    assert dispatcher.calls == [(out["room"], verified_avatar.id, verified_avatar.profile_path)]


async def test_no_agent_is_dispatched_when_consent_is_refused(db, cfg, avatar):
    from avatar.gateway.dispatch import NullDispatcher

    dispatcher = NullDispatcher()
    with pytest.raises(ConsentError):
        await open_session(db, cfg, avatar.id, dispatcher)
    assert dispatcher.calls == [], "a refused call must never put an agent in a room"


async def test_visitor_tokens_cannot_rewrite_the_synthetic_declaration(cfg):
    # The declaration says the stream is AI-generated. A visitor able to edit
    # participant metadata could remove it.
    import jwt

    token = mint_token(cfg, room="call-abc", identity="human-1", name="Visitor")
    claims = jwt.decode(token, cfg.livekit_api_secret, algorithms=["HS256"])
    assert not claims["video"].get("canUpdateOwnMetadata", False)


async def test_agent_tokens_may_publish_the_declaration(cfg):
    import jwt

    token = mint_token(
        cfg, room="call-abc", identity="avatar-x", name="Avatar", can_update_metadata=True
    )
    claims = jwt.decode(token, cfg.livekit_api_secret, algorithms=["HS256"])
    assert claims["video"]["canUpdateOwnMetadata"] is True
