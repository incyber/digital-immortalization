"""Cross-site request defence for bodyless mutations.

A POST with no JSON body and no custom header is a CORS "simple" request: the
browser sends it straight through, with no preflight and therefore no point
at which this server's CORS origin allowlist is ever consulted - that
allowlist is only checked before a non-simple request. A page on any other
origin can already reach /api/photo-sets or /api/photo-sets/{id}/train with a
bare `fetch(url, {method: "POST", credentials: "include"})`, riding whatever
session cookie the browser already holds for this site. That the paths
contain unguessable UUIDs is not a defence; it is the only thing standing
between this and an exploit today.

The fix does not depend on a secret value. It depends on a fact about
browsers: they never attach a custom header to a cross-origin request without
running a preflight first, and a preflight is exactly the point at which the
CORS origin allowlist (already configured, already correct) does its job.
Requiring this header is what forces that preflight to happen at all; the
header's value is incidental; the enforcement is the preflight.
"""

from __future__ import annotations

from fastapi import Header, HTTPException

# Read by the CORS middleware's preflight logic as a non-"simple" header,
# which is the entire mechanism - see the module docstring. The value is
# fixed rather than secret: it is not a credential, it exists so a request
# missing it (curl, a forged form, a bare cross-site fetch) is refused with a
# clear reason instead of an accident of a typo passing through.
REQUIRED_HEADER = "x-avatar-client"
REQUIRED_VALUE = "web"


async def require_same_site_header(x_avatar_client: str | None = Header(default=None)) -> None:
    """Depend on this from every state-changing route that takes no JSON body.

    A route whose body is a Pydantic model already forces a preflight because
    `content-type: application/json` is itself a non-"simple" header; this
    dependency exists only for the routes that have no such body.
    """
    if x_avatar_client != REQUIRED_VALUE:
        raise HTTPException(
            status_code=403,
            detail="missing or invalid client header; this endpoint cannot be called cross-site",
        )
