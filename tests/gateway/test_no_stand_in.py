"""A call shows the person, or it does not happen.

There was a generated stand-in behind this once - a still photograph with a
mouth warped into it - so a call could run before anybody had uploaded
anything. It reached a customer, who saw it and reasonably concluded the
product did not work.

Somebody opening this is looking for a person who has died. An approximation
of them is not a lesser version of the product; it is a different and worse
thing, and no amount of "it is only a placeholder" survives the moment they
see it. These tests exist so nobody can put it back by accident.
"""

import pytest

from avatar.gateway.consent import NoLikeness, assert_has_likeness


@pytest.mark.asyncio
async def test_an_avatar_with_nothing_built_refuses_the_call(db, verified_avatar):
    """Consented and owned is not enough. There has to be something to show."""
    with pytest.raises(NoLikeness, match="no likeness"):
        await assert_has_likeness(db, verified_avatar.id)


@pytest.mark.asyncio
async def test_the_refusal_says_what_to_do_about_it(db, verified_avatar):
    """It is the one gate a customer can clear themselves, so it says how."""
    with pytest.raises(NoLikeness) as raised:
        await assert_has_likeness(db, verified_avatar.id)

    message = str(raised.value)
    assert "video" in message and "photographs" in message


@pytest.mark.asyncio
async def test_a_built_likeness_passes(db, verified_avatar):
    verified_avatar.assets_key = "assets/avatars/somewhere"
    await db.commit()

    await assert_has_likeness(db, verified_avatar.id)


@pytest.mark.asyncio
async def test_a_splat_alone_is_enough(db, verified_avatar):
    """The splat is the real likeness; plates are the older path."""
    verified_avatar.splat_key = "tenants/t/avatars/a/avatar.splat"
    await db.commit()

    await assert_has_likeness(db, verified_avatar.id)


def test_the_agent_refuses_to_build_a_stand_in_renderer():
    """The generated stand-in is gone from the renderer too, not just gated.

    Belt and braces on purpose: if the session gate is ever bypassed - a
    direct dispatch, a test harness, a future entry point - the renderer must
    still refuse rather than quietly invent a face.
    """
    from avatar.config import Settings
    from avatar.realtime.agent import NoLikeness as RendererNoLikeness
    from avatar.realtime.agent import build_renderer

    with pytest.raises(RendererNoLikeness):
        build_renderer(Settings(_env_file=None), assets_path=None)


def test_nothing_can_still_generate_a_stand_in():
    """The function that made one is no longer reachable from the agent."""
    from avatar.realtime import agent

    assert not hasattr(agent, "synthetic_assets")
