"""One suite, run against every splat backend, plus what the job may carry.

A backend that passes this is substitutable in the pipeline. The fake exists so
that everything around a GPU - the route, the refusal, the cost, the sentence a
family reads about how much of their father was invented - is finished and
defended before a GPU is ever rented.

The failure tests are the load-bearing ones. A splat build is minutes of GPU,
and the posture inherited from avatar/gpu/serverless.py is that nothing is
allocated between jobs: every exit from a build, including the ones nobody
plans for, must leave nothing pending and nothing running.
"""

import asyncio
import dataclasses

import pytest

from avatar.gpu.serverless import DEFAULT_EXECUTION_TIMEOUT_S, JobResult, JobState
from avatar.splat.build import (
    BYTES_PER_GAUSSIAN,
    MAX_MEASURED_ON_GENERATION,
    FakeSplatBackend,
    Quality,
    QualityReport,
    RunPodSplatBackend,
    SplatBackend,
    SplatBuilder,
    SplatBuildError,
    SplatJob,
    SplatRefused,
    SplatResult,
    cost_of,
    plan,
    report_for,
)
from avatar.splat.routes import Intake, Route, choose_route
from avatar.storage.keys import asset_key, photo_key, source_clip_key
from avatar.storage.local import LocalBlobStore

TENANT = "tenant-a"
OTHER_TENANT = "tenant-b"
SET = "set-1"
AVATAR = "avatar-1"


def photos(count: int, tenant: str = TENANT) -> tuple[str, ...]:
    return tuple(photo_key(tenant, SET, f"photo-{i:03d}.jpg") for i in range(count))


def photo_intake(count: int = 20, **overrides) -> Intake:
    base = {"tenant_id": TENANT, "photo_set_id": SET, "photo_keys": photos(count)}
    base.update(overrides)
    return Intake(**base)


def video_intake(**overrides) -> Intake:
    base = {
        "tenant_id": TENANT,
        "photo_set_id": SET,
        "photo_keys": photos(20),
        "video_key": source_clip_key(TENANT, SET),
        "video_seconds": 30.0,
        "video_frames": 45,
        "video_frames_with_face": 44,
    }
    base.update(overrides)
    return Intake(**base)


class StubServerlessClient:
    """Stands in for RunPod's API, with its three methods and its shapes.

    Injected rather than patched so the real backend's poll loop, timeout,
    cancel and output parsing are all exercised - the parts of a GPU path that
    only run on a bad day are the parts worth testing on a good one.
    """

    def __init__(self, *, states=(JobState.COMPLETED,), execution_ms=420_000, output=None):
        self._states = list(states)
        self._execution_ms = execution_ms
        self._output = output
        self.submitted: list[dict] = []
        self.cancelled: list[str] = []

    def submit(self, payload: dict) -> str:
        self.submitted.append(payload)
        return "runpod-000001"

    def status(self, job_id: str) -> JobResult:
        state = self._states.pop(0) if len(self._states) > 1 else self._states[0]
        return JobResult(
            id=job_id,
            state=state,
            output=self._output_for(state),
            execution_ms=self._execution_ms if state.terminal else 0,
        )

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)

    def _output_for(self, state: JobState) -> dict | None:
        if state is not JobState.COMPLETED:
            return None
        if self._output is not None:
            return self._output
        payload = self.submitted[-1]
        budget = payload["gaussian_budget"]
        return {
            "splat_key": payload["output_key"],
            "gaussians": budget,
            "bytes": budget * BYTES_PER_GAUSSIAN,
            "angular_coverage": 0.8,
        }


@pytest.fixture(params=["fake", "runpod"])
def backend(request):
    """Both backends, behind one fixture.

    The real backend passing this unchanged is the whole point: the routing,
    the refusal, the report and the cost behave identically on a laptop and on
    a rented GPU, and only the pixels differ.
    """
    if request.param == "fake":
        return FakeSplatBackend()
    return RunPodSplatBackend(client=StubServerlessClient(), poll_s=0.0)


@pytest.fixture
def builder(backend):
    return SplatBuilder(backend)


# --- the contract both backends satisfy -------------------------------------


def test_both_backends_satisfy_one_protocol(backend):
    assert isinstance(backend, SplatBackend)


async def test_a_video_build_reports_the_route_it_took(builder):
    result = await builder.build(video_intake(), AVATAR)
    assert result.route is Route.RECONSTRUCT


async def test_a_photograph_build_reports_the_route_it_took(builder):
    result = await builder.build(photo_intake(), AVATAR)
    assert result.route is Route.GENERATE


async def test_the_reasoning_survives_into_the_result(builder):
    """Support and the customer are shown this months after the build ran."""
    result = await builder.build(video_intake(), AVATAR)
    assert result.reasoning == choose_route(video_intake()).reasoning
    assert "camera" in result.reasoning


async def test_the_evidence_behind_the_choice_survives_into_the_result(builder):
    result = await builder.build(photo_intake(), AVATAR)
    assert "photographs accepted: 20" in " ".join(result.decision.considered)


async def test_a_generated_splat_states_how_much_of_it_was_invented(builder):
    result = await builder.build(photo_intake(), AVATAR)
    assert result.report.generated_fraction > 0
    assert "generated rather than photographed" in result.report.disclosure
    assert "not measured" in result.report.disclosure


async def test_a_reconstructed_splat_states_that_nothing_was_invented(builder):
    result = await builder.build(video_intake(), AVATAR)
    assert result.report.measured_fraction == 1.0
    assert "Nothing about the face was invented" in result.report.disclosure


async def test_the_splat_lands_inside_the_tenants_own_prefix(builder):
    result = await builder.build(photo_intake(), AVATAR)
    assert result.splat_key == asset_key(TENANT, AVATAR, "avatar.splat")


async def test_a_build_reports_size_time_and_cost(builder):
    result = await builder.build(photo_intake(), AVATAR)
    assert result.gaussian_count > 0
    assert result.size_bytes > 0
    assert result.build_seconds > 0
    assert result.cost_usd > 0


async def test_the_backend_that_built_it_is_named_on_the_result(builder, backend):
    result = await builder.build(photo_intake(), AVATAR)
    assert result.backend == backend.name


async def test_the_report_carries_what_the_source_material_was(builder):
    result = await builder.build(photo_intake(source_short_edge_px=1080), AVATAR)
    assert result.report.source_short_edge_px == 1080
    assert result.report.views_used == 20


async def test_nothing_is_pending_before_or_after_a_build(builder):
    assert builder.pending == ()
    await builder.build(photo_intake(), AVATAR)
    assert builder.pending == ()


async def test_an_unbuildable_set_never_reaches_the_gpu(builder, backend):
    """Refusal costs nothing. A build attempted on one photograph costs minutes."""
    with pytest.raises(SplatRefused):
        await builder.build(photo_intake(1), AVATAR)
    assert builder.pending == ()
    assert getattr(backend, "submitted", []) == []


async def test_the_refusal_carries_the_decision_the_customer_is_shown(builder):
    with pytest.raises(SplatRefused) as raised:
        await builder.build(photo_intake(0), AVATAR)
    assert raised.value.decision.route is Route.REFUSE
    assert "photographs" in str(raised.value)
    assert "video" in str(raised.value)


# --- the job contract -------------------------------------------------------


def test_source_assets_are_passed_as_keys_not_bytes():
    """A thirty-image set in the job contract is a thirty-image set in memory,
    in every log line and in every traceback for the length of the build."""
    job = plan(video_intake(), AVATAR)
    payload = job.payload()
    assert payload["photo_keys"] == list(photos(20))
    assert payload["video_key"] == source_clip_key(TENANT, SET)
    for value in payload.values():
        assert not isinstance(value, bytes | bytearray | memoryview)
        if isinstance(value, list):
            assert all(isinstance(item, str) for item in value)


def test_a_job_carrying_image_bytes_is_refused():
    with pytest.raises(TypeError, match="never as bytes"):
        SplatJob(
            tenant_id=TENANT,
            avatar_id=AVATAR,
            decision=choose_route(photo_intake()),
            quality=Quality.STANDARD,
            output_key=asset_key(TENANT, AVATAR, "avatar.splat"),
            photo_keys=(b"\xff\xd8\xff\xe0 a jpeg",),
        )


def test_a_job_cannot_reach_into_another_tenants_prefix():
    with pytest.raises(ValueError, match="outside tenant"):
        SplatJob(
            tenant_id=TENANT,
            avatar_id=AVATAR,
            decision=choose_route(photo_intake()),
            quality=Quality.STANDARD,
            output_key=asset_key(TENANT, AVATAR, "avatar.splat"),
            photo_keys=photos(1, tenant=OTHER_TENANT),
        )


def test_the_anchor_must_be_one_of_the_photographs_supplied():
    with pytest.raises(ValueError, match="anchor"):
        SplatJob(
            tenant_id=TENANT,
            avatar_id=AVATAR,
            decision=choose_route(photo_intake()),
            quality=Quality.STANDARD,
            output_key=asset_key(TENANT, AVATAR, "avatar.splat"),
            photo_keys=photos(3),
            anchor_key=photo_key(TENANT, SET, "somebody-else.jpg"),
        )


def test_a_generated_build_is_anchored_on_a_photograph():
    job = plan(photo_intake(), AVATAR)
    assert job.anchor_key == photos(20)[0]


def test_a_chosen_anchor_beats_the_default():
    """No metric picks the photograph that most looks like him. A family does."""
    chosen = photos(20)[7]
    job = plan(photo_intake(anchor_key=chosen), AVATAR)
    assert job.anchor_key == chosen


def test_a_generated_build_does_not_carry_a_video_it_did_not_use():
    job = plan(video_intake(video_seconds=2.0), AVATAR)
    assert job.route is Route.GENERATE
    assert job.video_key is None


def test_the_quality_target_reaches_the_worker():
    payload = plan(photo_intake(), AVATAR, quality=Quality.PREVIEW).payload()
    assert payload["gaussian_budget"] == Quality.PREVIEW.gaussian_budget
    assert payload["iterations"] == Quality.PREVIEW.iterations


def test_a_refused_decision_can_never_become_a_job():
    with pytest.raises(SplatRefused):
        SplatJob(
            tenant_id=TENANT,
            avatar_id=AVATAR,
            decision=choose_route(photo_intake(0)),
            quality=Quality.STANDARD,
            output_key=asset_key(TENANT, AVATAR, "avatar.splat"),
        )


# --- the quality report -----------------------------------------------------


def test_a_result_cannot_be_built_without_a_quality_report():
    """The one thing the plan requires never be discovered later."""
    with pytest.raises(TypeError):
        SplatResult(
            decision=choose_route(photo_intake()),
            splat_key="k",
            gaussian_count=1,
            size_bytes=1,
            build_seconds=1.0,
            cost_usd=0.1,
        )


def test_how_much_was_measured_is_derived_not_reported():
    """A backend cannot overstate it, because there is nowhere to state it."""
    names = {f.name for f in dataclasses.fields(QualityReport)}
    assert "measured_fraction" not in names
    assert "generated_fraction" not in names


@pytest.mark.parametrize("views", [1, 3, 20, 24, 1000])
def test_a_generated_splat_can_never_claim_to_be_fully_measured(views):
    """A photograph shows one hemisphere. The back of the head is invented
    however large the album is."""
    report = QualityReport(
        route=Route.GENERATE, views_used=views, views_with_face=views, angular_coverage=1.0
    )
    assert report.measured_fraction <= MAX_MEASURED_ON_GENERATION
    assert report.generated_fraction >= 1.0 - MAX_MEASURED_ON_GENERATION
    assert report.disclosure


def test_coverage_of_the_viewing_sphere_is_not_a_quality_score():
    """Generation covers every angle by construction. That is the trap: it
    looks complete precisely where nothing was measured."""
    generated = QualityReport(
        route=Route.GENERATE, views_used=20, views_with_face=20, angular_coverage=1.0
    )
    assert generated.angular_coverage == 1.0
    assert generated.measured_fraction < 1.0


def test_frames_with_no_face_in_them_are_reported():
    report = QualityReport(
        route=Route.RECONSTRUCT, views_used=45, views_with_face=40, angular_coverage=0.9
    )
    assert not report.face_in_every_view
    assert any("5 of the 45" in note for note in report.concerns)


def test_a_clean_reconstruction_has_nothing_to_flag():
    report = QualityReport(
        route=Route.RECONSTRUCT, views_used=45, views_with_face=45, angular_coverage=0.95
    )
    assert report.face_in_every_view
    assert report.concerns == ()


def test_source_material_below_the_pipelines_resolution_is_flagged():
    report = QualityReport(
        route=Route.GENERATE,
        views_used=20,
        views_with_face=20,
        angular_coverage=1.0,
        source_short_edge_px=320,
    )
    assert any("320px" in note for note in report.concerns)


def test_a_narrow_range_of_angles_is_flagged():
    report = QualityReport(
        route=Route.RECONSTRUCT, views_used=20, views_with_face=20, angular_coverage=0.2
    )
    assert any("narrow range of angles" in note for note in report.concerns)


def test_a_mostly_invented_likeness_says_a_video_would_fix_it():
    report = QualityReport(
        route=Route.GENERATE, views_used=3, views_with_face=3, angular_coverage=1.0
    )
    assert any("a video of the person" in note for note in report.concerns)


def test_a_refused_build_has_nothing_to_report_on():
    with pytest.raises(ValueError, match="refused build"):
        QualityReport(
            route=Route.REFUSE, views_used=1, views_with_face=1, angular_coverage=1.0
        )


def test_a_splat_built_from_no_views_is_not_a_splat():
    with pytest.raises(ValueError, match="no views"):
        QualityReport(
            route=Route.GENERATE, views_used=0, views_with_face=0, angular_coverage=1.0
        )


def test_the_report_reaches_the_result_as_warnings_too():
    job = plan(video_intake(video_frames_with_face=30), AVATAR)
    report = report_for(job)
    assert report.concerns
    assert report.views_with_face == 30


# --- cost and safety --------------------------------------------------------


def test_cost_is_measured_the_way_every_other_gpu_job_measures_it():
    """One number to change when the card or the provider does."""
    assert 0.21 < cost_of(1_800_000) < 0.23
    assert cost_of(0) == 0.0


@pytest.mark.parametrize("quality", list(Quality))
def test_every_wait_sits_under_the_platforms_own_ceiling(quality):
    """Waiting past the execution timeout is waiting for a job already killed."""
    assert 0 < quality.wait_s <= DEFAULT_EXECUTION_TIMEOUT_S


async def test_a_gpu_failure_leaves_no_job_pending_and_nothing_allocated():
    backend = FakeSplatBackend(fail_in="collect")
    builder = SplatBuilder(backend)

    with pytest.raises(SplatBuildError):
        await builder.build(photo_intake(), AVATAR)

    assert builder.pending == ()
    assert backend.cancelled == ["fake-000001"], "a failed build must stop its own job"


async def test_a_failure_before_anything_is_queued_cancels_nothing():
    backend = FakeSplatBackend(fail_in="submit")
    builder = SplatBuilder(backend)

    with pytest.raises(SplatBuildError):
        await builder.build(photo_intake(), AVATAR)

    assert builder.pending == ()
    assert backend.cancelled == []


async def test_a_cancelled_build_still_stops_the_job():
    """The case that leaks: the waiter goes away and the GPU does not."""

    class Abandoned(FakeSplatBackend):
        async def collect(self, external_id, job, *, wait_s):
            raise asyncio.CancelledError

    backend = Abandoned()
    builder = SplatBuilder(backend)

    with pytest.raises(asyncio.CancelledError):
        await builder.build(photo_intake(), AVATAR)

    assert builder.pending == ()
    assert backend.cancelled == ["fake-000001"]


async def test_a_cancel_that_fails_does_not_hide_the_failure_that_caused_it():
    class Stubborn(FakeSplatBackend):
        async def cancel(self, external_id):
            raise RuntimeError("the provider is down too")

    builder = SplatBuilder(Stubborn(fail_in="collect"))
    with pytest.raises(SplatBuildError, match="fail at collect"):
        await builder.build(photo_intake(), AVATAR)
    assert builder.pending == ()


async def test_a_build_that_outruns_its_bound_is_cancelled():
    client = StubServerlessClient(states=(JobState.IN_PROGRESS,))
    backend = RunPodSplatBackend(client=client, poll_s=0.0)
    job = plan(photo_intake(), AVATAR)

    with pytest.raises(SplatBuildError, match="did not finish"):
        await backend.collect("runpod-000001", job, wait_s=0.0)

    assert client.cancelled == ["runpod-000001"]


async def test_a_job_the_platform_failed_is_reported_as_a_failure():
    client = StubServerlessClient(states=(JobState.FAILED,))
    backend = RunPodSplatBackend(client=client, poll_s=0.0)
    job = plan(photo_intake(), AVATAR)

    with pytest.raises(SplatBuildError, match="FAILED"):
        await backend.collect("runpod-000001", job, wait_s=10.0)


async def test_a_job_that_finished_with_no_splat_is_a_failure():
    """Otherwise it is recorded as a success and discovered on the call."""
    client = StubServerlessClient(output={"gaussians": 100})
    backend = RunPodSplatBackend(client=client, poll_s=0.0)
    job = plan(photo_intake(), AVATAR)

    with pytest.raises(SplatBuildError, match="no splat"):
        await backend.collect("runpod-000001", job, wait_s=10.0)


async def test_a_worker_writing_somewhere_other_than_the_job_said_is_a_failure():
    client = StubServerlessClient(
        output={"splat_key": asset_key(OTHER_TENANT, AVATAR, "avatar.splat"), "gaussians": 10}
    )
    backend = RunPodSplatBackend(client=client, poll_s=0.0)
    job = plan(photo_intake(), AVATAR)

    with pytest.raises(SplatBuildError, match="rather than"):
        await backend.collect("runpod-000001", job, wait_s=10.0)


async def test_the_real_backend_waits_rather_than_assuming_the_first_answer():
    client = StubServerlessClient(
        states=(JobState.IN_QUEUE, JobState.IN_PROGRESS, JobState.COMPLETED)
    )
    backend = RunPodSplatBackend(client=client, poll_s=0.0)
    job = plan(video_intake(), AVATAR)

    external_id = await backend.submit(job)
    result = await backend.collect(external_id, job, wait_s=10.0)
    assert result.splat_key == job.output_key
    assert client.cancelled == []


# --- the fake, as a development backend -------------------------------------


async def test_the_fake_writes_an_artefact_when_it_is_given_a_store(tmp_path):
    """So the download and attach paths can be exercised with no GPU."""
    store = LocalBlobStore(tmp_path)
    builder = SplatBuilder(FakeSplatBackend(store))
    result = await builder.build(photo_intake(), AVATAR)
    assert await store.get(TENANT, result.splat_key)


async def test_the_fake_returns_instantly_while_reporting_a_realistic_build():
    builder = SplatBuilder(FakeSplatBackend())
    result = await builder.build(video_intake(), AVATAR, quality=Quality.STANDARD)
    assert result.build_seconds > 60, "a splat build is minutes of GPU, and says so"
    assert result.size_bytes == result.gaussian_count * BYTES_PER_GAUSSIAN


async def test_a_bigger_quality_target_costs_more_and_downloads_larger():
    builder = SplatBuilder(FakeSplatBackend())
    preview = await builder.build(video_intake(), AVATAR, quality=Quality.PREVIEW)
    high = await builder.build(video_intake(), AVATAR, quality=Quality.HIGH)
    assert high.size_bytes > preview.size_bytes
    assert high.cost_usd > preview.cost_usd
