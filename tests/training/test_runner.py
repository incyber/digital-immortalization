"""One suite both training backends must satisfy."""

import pytest

from avatar.storage.local import LocalBlobStore
from avatar.training.base import JobState, TrainingRequest
from avatar.training.local import LocalTrainingRunner

TENANT = "tenant-a"


@pytest.fixture
def store(tmp_path):
    return LocalBlobStore(tmp_path)


@pytest.fixture
def runner(store):
    return LocalTrainingRunner(store, duration_s=0.2)


def a_request(keys=("tenants/tenant-a/photos/set-1/photo-000.jpg",)):
    return TrainingRequest(
        tenant_id=TENANT,
        photo_set_id="set-1",
        image_keys=list(keys),
        subject_name="Ada Lovelace",
    )


async def test_start_returns_immediately_with_an_id(runner):
    result = await runner.start(a_request())
    assert result.external_id
    assert not result.state.terminal, "start must not block until the run finishes"


async def test_a_run_reaches_succeeded(runner):
    started = await runner.start(a_request())
    result = await runner.wait(started.external_id)
    assert result.state is JobState.SUCCEEDED
    assert result.output_key


async def test_the_artefact_lands_in_the_tenants_prefix(runner, store):
    started = await runner.start(a_request())
    result = await runner.wait(started.external_id)
    assert result.output_key.startswith(f"tenants/{TENANT}/")
    assert await store.get(TENANT, result.output_key)


async def test_progress_advances(runner):
    started = await runner.start(a_request())
    first = await runner.poll(started.external_id)
    assert 0.0 <= first.progress < 1.0


async def test_an_empty_set_fails_rather_than_running(runner):
    result = await runner.start(a_request(keys=()))
    assert result.state is JobState.FAILED
    assert "no images" in result.error


async def test_cancel_stops_a_run(runner):
    started = await runner.start(a_request())
    await runner.cancel(started.external_id)
    assert (await runner.poll(started.external_id)).state is JobState.CANCELLED


async def test_cancelling_twice_is_safe(runner):
    started = await runner.start(a_request())
    await runner.cancel(started.external_id)
    await runner.cancel(started.external_id)


async def test_polling_an_unknown_run_fails_rather_than_raising(runner):
    assert (await runner.poll("no-such-run")).state is JobState.FAILED


async def test_trigger_word_avoids_the_real_name():
    # A real name collides with what the base model already knows about
    # anyone famous, and the likeness drifts to that instead of the photos.
    from avatar.training.replicate import _trigger_word

    assert _trigger_word("Ada Lovelace") != "Ada Lovelace"
    assert _trigger_word("Ada Lovelace").startswith("tok")
    assert _trigger_word("") == "toksubject"


def test_replicate_runner_requires_a_token(store):
    from avatar.training.replicate import ReplicateTrainingRunner

    with pytest.raises(ValueError, match="API token"):
        ReplicateTrainingRunner(api_token="", model_version="v", store=store)
