import pytest
import pytest_asyncio

from avatar.renderer.musetalk import MuseTalkRenderer
from avatar.renderer.plates import synthetic_assets
from avatar.renderer.viseme import VisemeRenderer
from tests.renderer.fake_service import FakeRendererService


@pytest.fixture
def tmp_assets():
    # Small and short: the contract suite cares about counts and timing, not
    # resolution, and a 128px loop keeps the whole suite well under a second.
    return synthetic_assets(size=(128, 128), fps=25, seconds=1.0)


@pytest_asyncio.fixture
async def fake_service():
    """The GPU service, stood in for. See fake_service.py for what it imitates."""
    service = await FakeRendererService(size=(128, 128)).start()
    try:
        yield service
    finally:
        await service.stop()


@pytest.fixture(params=["viseme", "musetalk"])
def renderer_factory(request, fake_service):
    """Both backends, behind one factory.

    MuseTalkRenderer passing this unchanged is the whole point of the contract
    living in one place: the call loop, turn timing and barge-in behave the
    same on a laptop and on a GPU, and only pixels differ.
    """
    if request.param == "viseme":
        return lambda assets: VisemeRenderer(assets)
    return lambda _assets: MuseTalkRenderer(fake_service.url, size=(128, 128))
