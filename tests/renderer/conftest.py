import pytest

from avatar.renderer.plates import synthetic_assets
from avatar.renderer.viseme import VisemeRenderer


@pytest.fixture
def tmp_assets():
    # Small and short: the contract suite cares about counts and timing, not
    # resolution, and a 128px loop keeps the whole suite well under a second.
    return synthetic_assets(size=(128, 128), fps=25, seconds=1.0)


def _implementations():
    yield pytest.param(lambda assets: VisemeRenderer(assets), id="viseme")
    # MuseTalkRenderer is added here in sub-project 2 and must pass this suite
    # unchanged. That is the whole point of the contract living in one place.


@pytest.fixture(params=list(_implementations()))
def renderer_factory(request):
    return request.param
