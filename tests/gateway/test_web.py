"""Serving the site from the same process as the API.

The reason this exists is a cookie. Cross-origin, the session cookie has to be
SameSite=None, and browsers throw those away — so the tests that matter here
are the ones about which requests reach which half, because getting that wrong
puts the two back on separate origins in effect if not in name.
"""

import pytest
from fastapi.testclient import TestClient

from avatar.config import Settings
from avatar.gateway.app import create_app
from avatar.gateway.web import resolve_web_root


@pytest.fixture
def site(tmp_path):
    """A miniature export, laid out the way `next build` lays one out."""
    root = tmp_path / "site"
    (root / "_next" / "static" / "chunk").mkdir(parents=True)
    (root / "avatars").mkdir()
    (root / "call").mkdir()

    (root / "index.html").write_text("<html>home</html>")
    (root / "404.html").write_text("<html>not found</html>")
    (root / "avatars" / "index.html").write_text("<html>avatars</html>")
    (root / "call" / "index.html").write_text("<html>call</html>")
    (root / "_next" / "static" / "chunk" / "app.js").write_text("console.info(1)")

    # Something outside the root that a traversal would reach.
    (tmp_path / "secret.txt").write_text("not for the web")
    return root


@pytest.fixture
def client(site, tmp_path):
    cfg = Settings(
        _env_file=None,
        web_root=str(site),
        database_url=f"sqlite+aiosqlite:///{tmp_path}/web.db",
    )
    with TestClient(create_app(cfg)) as c:
        yield c


def test_the_root_serves_the_site(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "home" in response.text


def test_a_route_resolves_to_its_index_file(client):
    assert "avatars" in client.get("/avatars/").text
    # Without the trailing slash too: a link somebody typed is still a link.
    assert "avatars" in client.get("/avatars").text


def test_a_deep_link_into_the_call_page_works(client):
    """The whole reason a call is /call?avatar= rather than /call/<id>.

    An export has no file for a path segment it could not enumerate at build
    time. The query string is invisible to routing, so one file answers for
    every avatar — including on a cold load, which is what a shared link is.
    """
    response = client.get("/call/?avatar=99e1d0c8-0000-4000-8000-000000000000")
    assert response.status_code == 200
    assert "call" in response.text


def test_the_api_still_belongs_to_the_gateway(client):
    """The failure this prevents is quiet and expensive to debug.

    A catch-all registered over the API answers a real endpoint with a page.
    The browser gets 200 and HTML where it expected JSON, and the bug looks
    like it is in the client.
    """
    assert client.get("/api/config").json() == {"demo_mode": False}
    assert client.get("/api/me").status_code == 401
    assert client.get("/health").json() == {"ok": True}


def test_an_unknown_api_path_is_a_404_and_not_a_page(client):
    response = client.get("/api/there-is-no-such-endpoint")
    assert response.status_code == 404
    assert response.json()["detail"] == "no such endpoint"
    assert "html" not in response.headers["content-type"]


def test_an_unknown_page_gets_the_export_s_own_404(client):
    response = client.get("/somewhere-that-does-not-exist")
    assert response.status_code == 404
    assert "not found" in response.text


def test_hashed_assets_are_cached_and_pages_are_not(client):
    """Two different lifetimes, and using one policy for both is a real fault.

    A page cached hard is a deploy nobody sees. A hashed chunk revalidated on
    every navigation is a round trip for a file that cannot have changed.
    """
    asset = client.get("/_next/static/chunk/app.js")
    assert asset.status_code == 200
    assert "immutable" in asset.headers["cache-control"]

    page = client.get("/")
    assert "immutable" not in page.headers["cache-control"]


@pytest.mark.parametrize(
    "path",
    [
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/avatars/../../secret.txt",
        "/....//secret.txt",
        # An absolute path in the URL. pathlib's join REPLACES the base when
        # the right-hand side is absolute, so this is the one traversal shape
        # that does not look like a traversal.
        "//etc/passwd",
        "///Users/admin/.ssh/id_rsa",
    ],
)
def test_nothing_outside_the_site_can_be_read(client, path):
    """This process's working directory holds .env and the database."""
    response = client.get(path)
    assert response.status_code == 404
    assert "not for the web" not in response.text


def test_a_gateway_with_no_site_built_is_still_a_gateway(tmp_path):
    """API-only is a supported way to run this, not a broken deployment.

    It is how the tests run, and how the split-port development setup runs.
    A missing export must not be an error at boot.
    """
    cfg = Settings(
        _env_file=None,
        web_root=str(tmp_path / "nothing-here"),
        database_url=f"sqlite+aiosqlite:///{tmp_path}/bare.db",
    )
    assert resolve_web_root(cfg) is None

    with TestClient(create_app(cfg)) as client:
        assert client.get("/health").json() == {"ok": True}
        assert client.get("/").status_code == 404


def test_a_directory_without_an_index_is_not_served(client, site):
    (site / "empty").mkdir()
    assert client.get("/empty/").status_code == 404
