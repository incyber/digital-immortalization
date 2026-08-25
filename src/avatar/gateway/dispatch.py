"""Getting an agent into the room the gateway just created.

A room with a token and nobody in it is not a call. Something has to put the
avatar on the other end.

This implementation spawns a local process per session. That is correct for one
machine and wrong for a fleet: in cloud the same responsibility belongs to a
pool of pre-warmed workers, because process start plus model load is far more
than the latency budget allows and because GPU capacity has to be scheduled
rather than assumed. The interface here is deliberately the one a pool would
also satisfy, so that replacement does not reach into the gateway.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Protocol

from loguru import logger

from avatar.config import Settings


class AgentDispatcher(Protocol):
    async def dispatch(
        self,
        room: str,
        avatar_id: str,
        consent_record_id: str | None = None,
        rights_holder: str | None = None,
    ) -> None: ...


class LocalProcessDispatcher:
    """One subprocess per session. Development only."""

    def __init__(self, cfg: Settings):
        self._cfg = cfg
        self._processes: dict[str, asyncio.subprocess.Process] = {}

    async def dispatch(
        self,
        room: str,
        avatar_id: str,
        consent_record_id: str | None = None,
        rights_holder: str | None = None,
    ) -> None:
        from avatar.gateway.sessions import mint_token

        token = mint_token(
            self._cfg,
            room,
            identity=f"avatar-{avatar_id}",
            name="Avatar",
            # The agent publishes the synthetic-content declaration as
            # participant attributes, which the server refuses without this.
            can_update_metadata=True,
        )

        assets = Path(self._cfg.assets_dir) / "avatars" / avatar_id
        command = [
            sys.executable, "-m", "avatar.realtime.agent",
            "--room", room,
            "--token", token,
            "--avatar", avatar_id,
        ]
        if assets.exists():
            command += ["--assets", str(assets)]
        # Carried through so the session's synthetic-content declaration can
        # name the consent it was produced under.
        if consent_record_id:
            command += ["--consent-record", consent_record_id]
        if rights_holder:
            command += ["--rights-holder", rights_holder]

        logger.info(f"dispatching agent into {room}")
        # create_subprocess_exec rather than subprocess.Popen: spawning blocks,
        # and this runs inside the request that is about to return a token.
        self._processes[room] = await asyncio.create_subprocess_exec(*command)

    def shutdown(self) -> None:
        for room, proc in self._processes.items():
            if proc.returncode is None:
                logger.info(f"terminating agent for {room}")
                proc.terminate()
        self._processes.clear()


class NullDispatcher:
    """Records dispatches without starting anything. Used by tests, which care
    that dispatch happens after the consent gate and not that a process runs."""

    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []

    async def dispatch(
        self,
        room: str,
        avatar_id: str,
        consent_record_id: str | None = None,
        rights_holder: str | None = None,
    ) -> None:
        self.calls.append((room, avatar_id))
