"""Serving the web app from this process.

The site and the API are one origin because the alternative did not work. With
the site on one domain and this API on another, every request the browser makes
is cross-site, so the session cookie has to be SameSite=None - which Safari
blocks outright and Chrome blocks in common configurations. The symptom was a
sign-in that returned 200, had its cookie thrown away, and bounced back to the
sign-in page forever. One origin makes the cookie first-party and deletes that
whole class of problem, and CORS and preflight with it.

What is served is a static export of the Next application: plain HTML, CSS and
JavaScript, no Node process. That is possible because every page in the app is
already a client component - there are no server actions, no server-side data
loading and no request-time rendering anywhere in it - so a second runtime
beside the gateway would have been a process to supervise, a port to keep
private and a restart to sequence, in exchange for nothing the browser can
tell apart.

Two rules govern the routing, and both are here rather than spread around:

  /api and /health belong to the gateway and are registered before this. A
  request under /api that reaches here is a route that does not exist, and it
  gets a 404 - never index.html, which would answer a missing endpoint with a
  page and turn a broken call into a mystery.

  Everything else is the site. An export writes one file per route, so a deep
  link is resolved against the export's own layout rather than by handing every
  path the same shell.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from avatar.config import Settings

# Where the build lands when it is packaged with the gateway. Written by
# `make web` / apps/web's export step; absent in a checkout that has never
# built the site, which is a gateway that serves the API only.
PACKAGED_WEB_ROOT = Path(__file__).resolve().parent / "webroot"

# Hashed filenames. The content of /_next/static/<hash>/... cannot change
# without the URL changing, so a year is honest and anything less is a
# needless round trip on every navigation.
_IMMUTABLE = "public, max-age=31536000, immutable"

# Everything else. An HTML file keeps its name across deploys, so a cached copy
# is a stale copy; must-revalidate rather than no-store so the browser can
# still skip the body when nothing changed.
_REVALIDATE = "no-cache, must-revalidate"


def resolve_web_root(cfg: Settings) -> Path | None:
    """The directory holding the built site, or None if there is not one.

    None is a normal answer, not a failure: the API alone is a valid way to run
    this process, and it is how the tests and the local split-port development
    setup run it.
    """
    candidate = Path(cfg.web_root).expanduser() if cfg.web_root else PACKAGED_WEB_ROOT
    return candidate if (candidate / "index.html").is_file() else None


def _resolve_file(root: Path, path: str) -> Path | None:
    """The exported file that answers a URL path, or None.

    Three shapes, in the order the export produces them: the literal asset,
    then `<route>.html`, then `<route>/index.html`. The last two are the two
    layouts Next writes depending on the trailingSlash setting, and accepting
    both means this file does not have to be changed if that setting is.
    """
    # A path that climbs out of the root is refused before it is touched.
    # Resolving first and comparing after is the only ordering that catches
    # "..", symlinks and encoded separators together.
    base = root.resolve()
    try:
        target = (base / path).resolve()
    except (OSError, ValueError):
        return None
    if target != base and base not in target.parents:
        return None

    if target.is_file():
        return target
    for candidate in (target.with_suffix(".html"), target / "index.html"):
        if candidate.is_file():
            return candidate
    return None


def _response(file: Path, status_code: int = 200) -> FileResponse:
    immutable = "_next/static" in file.as_posix()
    return FileResponse(
        file,
        status_code=status_code,
        headers={"cache-control": _IMMUTABLE if immutable else _REVALIDATE},
    )


def mount_web(app: FastAPI, root: Path) -> None:
    """Serve the export from `root` for every path the gateway did not claim.

    Must be called last. FastAPI matches routes in the order they were added,
    and the catch-all registered here would otherwise shadow the API.
    """

    @app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def site(path: str):
        # The gateway's own namespace. Reaching this line means no API route
        # matched, so the answer is that the endpoint does not exist - not a
        # page, which would be a 200 for a call that failed.
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="no such endpoint")

        file = _resolve_file(root, path)
        if file is not None:
            return _response(file)

        # The export's own not-found page, so a mistyped link still looks like
        # this product rather than like a web server.
        missing = root / "404.html"
        if missing.is_file():
            return _response(missing, status_code=404)
        raise HTTPException(status_code=404, detail="not found")
