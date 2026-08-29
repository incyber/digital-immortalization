"""Packing the private half of a worker.

Two properties carry the whole design. The archive must be reproducible, or the
digest stops meaning "this code" and every build forces every endpoint to be
reconfigured. And the file list must be explicit, because this archive is
uploaded to storage that a GPU reads and executes - a glob eventually picks up
a stray .env.
"""

from pathlib import Path

import pytest

from avatar.gpu.bundle import BUNDLES, build

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", sorted(BUNDLES))
def test_every_bundle_builds_from_the_repository(name):
    bundle = build(name, ROOT)

    assert bundle.data
    assert len(bundle.sha256) == 64


@pytest.mark.parametrize("name", sorted(BUNDLES))
def test_the_same_sources_produce_the_same_bytes(name):
    """Not merely the same hash: the same archive."""
    assert build(name, ROOT).data == build(name, ROOT).data


def test_a_missing_file_fails_the_build_rather_than_shipping_a_gap(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        build("serverless", tmp_path)


def test_an_unknown_bundle_is_rejected():
    with pytest.raises(ValueError, match="unknown bundle"):
        build("wishful", ROOT)


def test_the_key_is_content_addressed():
    """A new bundle must not overwrite the one a running endpoint expects."""
    bundle = build("serverless", ROOT)

    assert bundle.sha256[:16] in bundle.key
    assert bundle.key.endswith(".tar.gz")


@pytest.mark.parametrize("name", sorted(BUNDLES))
def test_the_archive_holds_only_what_was_listed(name):
    import io
    import tarfile

    bundle = build(name, ROOT)
    with tarfile.open(fileobj=io.BytesIO(bundle.data), mode="r:gz") as tar:
        members = sorted(m.name for m in tar.getmembers())

    assert members == sorted(BUNDLES[name])


def test_every_bundle_has_the_entrypoint_the_bootstrap_will_exec():
    # The bootstrap execs APP_ENTRYPOINT, which defaults to handler.py. A
    # bundle without one produces a worker that starts and then stops.
    for name in BUNDLES:
        assert "handler.py" in BUNDLES[name]
