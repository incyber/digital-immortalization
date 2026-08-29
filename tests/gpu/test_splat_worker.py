"""The splat worker, tested on a machine with no GPU.

There is no card here, so nothing below runs gsplat or TRELLIS. What is tested
is everything that decides whether a build is correct *before* the arithmetic
starts and *after* it finishes: which route a job reaches, that source assets
travel as keys rather than as a family's photographs, that a malformed job
comes back as a sentence instead of a traceback, that the file the browser
downloads is the size its Gaussian count implies, and - the one that matters
most - that no output this worker can produce lets a generated likeness be
reported as measured.

The geometry that can be checked without a GPU is checked: the coordinate
inversion that turns a turning head into an orbiting camera is linear algebra,
and it is asserted directly rather than trusted to a comment.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from avatar.gpu.serverless import JobResult, JobState
from avatar.splat.build import (
    MAX_MEASURED_ON_GENERATION,
    Quality,
    RunPodSplatBackend,
    SplatBuildError,
    SplatRefused,
    plan,
)
from avatar.splat.routes import Intake, Route

sys.path.insert(0, str(Path("infra/splatworker").resolve()))

import generate
import handler
import reconstruct

TENANT = "t1"
AVATAR = "a1"


# --------------------------------------------------------------------------
# helpers


def a_cloud(n=5, coverage=0.75, views=20, notes=()):
    rng = np.random.default_rng(0)
    return reconstruct.GaussianCloud(
        means=rng.normal(size=(n, 3)).astype("float32"),
        scales=rng.uniform(0.01, 0.05, size=(n, 3)).astype("float32"),
        quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), (n, 1)),
        colors=rng.uniform(0.0, 1.0, size=(n, 3)).astype("float32"),
        opacities=rng.uniform(0.1, 1.0, size=n).astype("float32"),
        angular_coverage=coverage,
        views_used=views,
        notes=tuple(notes),
    )


def a_video_intake():
    return Intake(
        tenant_id=TENANT,
        photo_set_id="set1",
        photo_keys=(f"tenants/{TENANT}/photos/set1/a.jpg",),
        video_key=f"tenants/{TENANT}/photos/set1/source.mp4",
        video_seconds=30.0,
        video_frames=60,
        video_frames_with_face=58,
        source_short_edge_px=1080,
    )


def a_photo_intake(count=6):
    return Intake(
        tenant_id=TENANT,
        photo_set_id="set1",
        photo_keys=tuple(f"tenants/{TENANT}/photos/set1/p{i}.jpg" for i in range(count)),
        source_short_edge_px=1080,
    )


@pytest.fixture
def offline(monkeypatch):
    """Every route stubbed and object storage removed.

    The worker's own arithmetic is not what these tests are about; where a job
    goes, what it is allowed to carry and what it may say afterwards is.
    """
    calls = {"reconstruct": [], "generate": [], "downloaded": [], "uploaded": []}

    monkeypatch.setattr(handler, "_client", lambda: object())
    monkeypatch.setattr(
        handler, "_download",
        lambda client, keys, workdir: (
            calls["downloaded"].extend(keys) or [workdir / f"src-{i}" for i in range(len(keys))]
        ),
    )
    monkeypatch.setattr(
        handler, "_upload",
        lambda client, key, data: calls["uploaded"].append((key, len(data))),
    )
    monkeypatch.setattr(
        reconstruct, "reconstruct",
        lambda video, **kw: (calls["reconstruct"].append((video, kw)) or a_cloud()),
    )
    monkeypatch.setattr(
        generate, "generate",
        lambda anchor, photos, **kw: (
            calls["generate"].append((anchor, photos, kw)) or a_cloud(coverage=1.0)
        ),
    )
    return calls


def run(job_payload):
    return handler.handler({"input": job_payload})


# --------------------------------------------------------------------------
# import-time safety


def test_the_handler_imports_with_no_cuda_and_no_weights():
    """The property the existing worker has, restated for this one.

    A worker whose module body needs a GPU crash-loops on a misconfigured
    endpoint, and the platform retries it at GPU rates. This module body must
    touch nothing heavier than the standard library.
    """
    assert callable(handler.handler)
    assert callable(reconstruct.reconstruct)
    assert callable(generate.generate)


def test_health_reports_what_is_missing_rather_than_raising():
    report = handler._health({})

    assert report["cuda"] is False
    assert report["trellis_weights"] is False
    assert report["storage_configured"] is False
    # Not "unused": absent. The same claim the other worker makes about
    # InsightFace, made about the rasterisers whose licences forbid this.
    assert report["restricted_modules_present"] == []


def test_a_worker_with_no_storage_configured_says_so():
    result = run({"task": "splat", "route": "reconstruct", "tenant_id": TENANT,
                  "output_key": f"tenants/{TENANT}/avatars/{AVATAR}/avatar.splat",
                  "video_key": f"tenants/{TENANT}/photos/set1/source.mp4",
                  "gaussian_budget": 1000, "iterations": 10})

    assert "S3_BUCKET" in result["error"]


# --------------------------------------------------------------------------
# route dispatch


def test_a_reconstruct_job_reaches_the_reconstruct_route(offline):
    job = plan(a_video_intake(), AVATAR)
    assert job.route is Route.RECONSTRUCT

    result = run(job.payload())

    assert "error" not in result
    assert len(offline["reconstruct"]) == 1
    assert offline["generate"] == []
    assert offline["downloaded"] == [job.video_key]


def test_a_generate_job_reaches_the_generate_route(offline):
    job = plan(a_photo_intake(), AVATAR)
    assert job.route is Route.GENERATE

    result = run(job.payload())

    assert "error" not in result
    assert len(offline["generate"]) == 1
    assert offline["reconstruct"] == []
    # The anchor is fetched first so its local path is unambiguous.
    assert offline["downloaded"][0] == job.anchor_key


def test_a_refused_decision_never_becomes_a_job():
    """Refusal happens before a GPU is asked for anything."""
    with pytest.raises(SplatRefused):
        plan(Intake(tenant_id=TENANT, photo_set_id="set1", photo_keys=()), AVATAR)


def test_a_refusal_that_somehow_reaches_the_worker_is_not_built(offline):
    """Belt and braces: the planner refuses, and so does the worker.

    A refusal arriving here means something bypassed the planner, and building
    it anyway would hand a family the likeness the system decided not to make.
    """
    result = run({"task": "splat", "route": "refuse", "tenant_id": TENANT,
                  "output_key": f"tenants/{TENANT}/avatars/{AVATAR}/avatar.splat",
                  "gaussian_budget": 1000, "iterations": 10})

    assert "refused" in result["error"]
    assert offline["reconstruct"] == [] and offline["generate"] == []


def test_an_unknown_route_is_refused(offline):
    assert "unknown route" in run(
        {"task": "splat", "route": "photogrammetry", "tenant_id": TENANT,
         "output_key": f"tenants/{TENANT}/avatars/{AVATAR}/avatar.splat",
         "gaussian_budget": 1, "iterations": 1}
    )["error"]


def test_an_unknown_task_is_refused():
    assert "unknown task" in run({"task": "wishful"})["error"]


# --------------------------------------------------------------------------
# payload handling


def test_photographs_may_not_arrive_as_bytes(offline):
    """The contract SplatJob enforces at one end, enforced at the other.

    Bytes here would mean a thirty-image set in the queue, in every retry and
    in every log line that prints a payload.
    """
    result = run({"task": "splat", "route": "generate", "tenant_id": TENANT,
                  "output_key": f"tenants/{TENANT}/avatars/{AVATAR}/avatar.splat",
                  "photo_keys": [b"\xff\xd8\xff\xe0 a real jpeg"],
                  "anchor_key": f"tenants/{TENANT}/photos/set1/p0.jpg",
                  "gaussian_budget": 1000, "iterations": 10})

    assert "bytes" in result["error"]
    assert "storage keys" in result["error"]


def test_a_key_outside_the_tenants_prefix_is_refused(offline):
    result = run({"task": "splat", "route": "reconstruct", "tenant_id": TENANT,
                  "output_key": f"tenants/{TENANT}/avatars/{AVATAR}/avatar.splat",
                  "video_key": "tenants/someone-else/photos/set1/source.mp4",
                  "gaussian_budget": 1000, "iterations": 10})

    assert "outside tenant" in result["error"]
    assert offline["reconstruct"] == []


def test_a_prefix_that_merely_starts_the_same_is_refused(offline):
    """tenants/t1 must not match tenants/t1x - the slash is load-bearing."""
    result = run({"task": "splat", "route": "reconstruct", "tenant_id": TENANT,
                  "output_key": f"tenants/{TENANT}/avatars/{AVATAR}/avatar.splat",
                  "video_key": "tenants/t1x/photos/set1/source.mp4",
                  "gaussian_budget": 1000, "iterations": 10})

    assert "outside tenant" in result["error"]


def test_a_traversal_in_a_key_is_refused(offline):
    result = run({"task": "splat", "route": "reconstruct", "tenant_id": TENANT,
                  "output_key": f"tenants/{TENANT}/avatars/{AVATAR}/avatar.splat",
                  "video_key": f"tenants/{TENANT}/../t2/photos/set1/source.mp4",
                  "gaussian_budget": 1000, "iterations": 10})

    assert "traversal" in result["error"]


def test_a_generated_build_whose_anchor_is_not_in_the_set_is_refused(offline):
    result = run({"task": "splat", "route": "generate", "tenant_id": TENANT,
                  "output_key": f"tenants/{TENANT}/avatars/{AVATAR}/avatar.splat",
                  "photo_keys": [f"tenants/{TENANT}/photos/set1/p0.jpg"],
                  "anchor_key": f"tenants/{TENANT}/photos/set1/other.jpg",
                  "gaussian_budget": 1000, "iterations": 10})

    assert "anchor" in result["error"]


@pytest.mark.parametrize("field", ["tenant_id", "output_key", "gaussian_budget", "iterations"])
def test_a_malformed_job_fails_with_a_sentence_not_a_stack_trace(offline, field):
    payload = plan(a_video_intake(), AVATAR).payload()
    payload.pop(field)

    result = run(payload)

    assert field in result["error"]
    assert "Traceback" not in result["error"]


def test_an_input_that_is_not_an_object_is_refused():
    assert "must be an object" in handler.handler({"input": ["a list"]})["error"]


def test_a_budget_of_zero_is_refused(offline):
    payload = plan(a_video_intake(), AVATAR).payload()
    payload["gaussian_budget"] = 0

    assert "greater than zero" in run(payload)["error"]


# --------------------------------------------------------------------------
# what the worker may say about a build


def test_the_worker_cannot_report_a_measured_fraction():
    """The guarantee this handler exists to not break.

    How much of a likeness came from a camera is derived from the route in
    avatar/splat/build.py, and there is deliberately no field here through
    which a worker could contribute to it.
    """
    assert not [name for name in handler.RESULT_FIELDS if "measur" in name]
    assert not [name for name in handler.RESULT_FIELDS if "fraction" in name]


def test_the_output_carries_nothing_beyond_the_declared_fields(offline):
    result = run(plan(a_video_intake(), AVATAR).payload())

    # task, job_id and total_seconds are added by the dispatcher, as in the
    # existing worker; everything else must come from RESULT_FIELDS.
    assert set(result) == set(handler.RESULT_FIELDS) | {"task", "job_id", "total_seconds"}


def test_a_generated_build_cannot_claim_to_be_measured(offline):
    """A worker reporting full coverage still cannot make the report say measured.

    The generate route legitimately reports angular_coverage 1.0 - the model
    invents every direction - so this is the exact case where a coverage
    figure could be mistaken for a quality claim. It is not: the report caps
    the measured fraction at MAX_MEASURED_ON_GENERATION regardless.
    """
    job = plan(a_photo_intake(count=30), AVATAR)
    output = run(job.payload())
    assert output["angular_coverage"] == 1.0

    result = RunPodSplatBackend(client=object())._result_from(
        JobResult(id="j", state=JobState.COMPLETED, execution_ms=120_000, output=output), job
    )

    assert result.report.measured_fraction <= MAX_MEASURED_ON_GENERATION
    assert result.report.generated_fraction >= 1.0 - MAX_MEASURED_ON_GENERATION
    assert "generated rather than photographed" in result.report.disclosure


def test_a_reconstructed_build_satisfies_the_result_contract(offline):
    job = plan(a_video_intake(), AVATAR)
    output = run(job.payload())

    result = RunPodSplatBackend(client=object())._result_from(
        JobResult(id="j", state=JobState.COMPLETED, execution_ms=300_000, output=output), job
    )

    assert result.splat_key == job.output_key
    assert result.gaussian_count == output["gaussians"]
    assert result.size_bytes == output["bytes"]
    assert result.report.measured_fraction == 1.0
    assert result.route is Route.RECONSTRUCT
    assert result.cost_usd > 0


def test_a_worker_that_wrote_somewhere_else_fails_the_build(offline, monkeypatch):
    """A completed job with the wrong artefact must not be recorded as success."""
    job = plan(a_video_intake(), AVATAR)
    output = dict(run(job.payload()), splat_key=f"tenants/{TENANT}/avatars/other/avatar.splat")

    with pytest.raises(SplatBuildError, match="rather than"):
        RunPodSplatBackend(client=object())._result_from(
            JobResult(id="j", state=JobState.COMPLETED, output=output), job
        )


def test_the_splat_is_written_to_the_key_the_job_named(offline):
    job = plan(a_video_intake(), AVATAR)
    result = run(job.payload())

    assert offline["uploaded"] == [(job.output_key, result["bytes"])]
    assert result["splat_key"] == job.output_key


@pytest.mark.parametrize("quality", list(Quality))
def test_the_budget_and_iterations_reaching_the_worker_are_the_quality_settings(
    offline, quality
):
    job = plan(a_video_intake(), AVATAR, quality=quality)

    run(job.payload())

    _, kwargs = offline["reconstruct"][0]
    assert kwargs["gaussian_budget"] == quality.gaussian_budget
    assert kwargs["iterations"] == quality.iterations


# --------------------------------------------------------------------------
# the file the customer downloads


def test_a_splat_is_thirty_two_bytes_per_gaussian():
    """The number avatar/splat/build.py sizes every download against."""
    data = reconstruct.write_splat(a_cloud(n=100))

    assert len(data) == 100 * reconstruct.BYTES_PER_GAUSSIAN
    assert reconstruct.BYTES_PER_GAUSSIAN == 32


def test_the_splat_is_ordered_by_falling_visual_importance():
    """What lets a viewer show a recognisable face after the first megabyte.

    Opacity is held equal across the cloud so the assertion is about the
    ordering rather than about how a float survives being quantised into the
    one byte the format allows for alpha.
    """
    import struct

    cloud = a_cloud(n=50)
    even = reconstruct.GaussianCloud(
        means=cloud.means, scales=cloud.scales, quats=cloud.quats,
        colors=cloud.colors, opacities=np.ones(50, dtype="float32"),
        angular_coverage=0.5, views_used=50,
    )
    data = reconstruct.write_splat(even)

    volumes = [
        float(np.prod(struct.unpack("<3f", data[i * 32 + 12: i * 32 + 24])))
        for i in range(50)
    ]

    assert volumes == sorted(volumes, reverse=True)


def test_an_inconsistent_cloud_is_refused_rather_than_truncated():
    cloud = a_cloud(n=10)
    broken = reconstruct.GaussianCloud(
        means=cloud.means, scales=cloud.scales, quats=cloud.quats,
        colors=cloud.colors, opacities=cloud.opacities[:5],
        angular_coverage=0.5, views_used=10,
    )

    with pytest.raises(reconstruct.ReconstructError, match="disagree"):
        reconstruct.write_splat(broken)


def test_the_handler_refuses_a_file_that_is_not_the_size_its_count_implies(
    offline, monkeypatch
):
    monkeypatch.setattr(reconstruct, "write_splat", lambda cloud: b"\x00" * 7)

    assert "size its Gaussian count implies" in run(
        plan(a_video_intake(), AVATAR).payload()
    )["error"]


# --------------------------------------------------------------------------
# the inversion, which is arithmetic and needs no GPU


def rotation_about_y(degrees):
    theta = np.radians(degrees)
    return np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)],
    ], dtype="float32")


def a_pose(index, rotation, image_size=64):
    view = np.eye(4, dtype="float32")
    view[:3, :3] = rotation
    view[:3, 3] = np.array([0.0, 0.0, 0.5], dtype="float32")
    return reconstruct.HeadPose(
        index=index,
        image=np.zeros((image_size, image_size, 3), dtype="uint8"),
        view=view,
        yaw=float(np.arctan2(-rotation[2][0], np.hypot(rotation[2][1], rotation[2][2]))),
        pitch=0.0,
        expression=0.0,
    )


def test_a_turning_head_becomes_an_orbiting_camera():
    """The claim reconstruct.py is built on, asserted rather than commented.

    A photograph in which the head has turned 30 degrees further than the
    anchor must produce a camera rotated by exactly that much about the fixed
    object - which is what makes the album's unseen angles addressable at all.
    """
    anchor = a_pose(0, np.eye(3, dtype="float32"))
    turned = a_pose(1, rotation_about_y(30))

    views = generate._relative_views(anchor, [turned], radius=2.0)

    assert np.allclose(views[0][:3, :3], rotation_about_y(30), atol=1e-5)
    assert np.allclose(views[0][:3, 3], [0.0, 0.0, 2.0])


def test_the_inversion_uses_only_relative_pose():
    """Both heads turned by the same absolute amount is no camera motion at all.

    This is why the alignment between MediaPipe's canonical head and TRELLIS's
    canonical cube never has to be solved: it cancels.
    """
    anchor = a_pose(0, rotation_about_y(20))
    same = a_pose(1, rotation_about_y(20))

    views = generate._relative_views(anchor, [same], radius=2.0)

    assert np.allclose(views[0][:3, :3], np.eye(3), atol=1e-5)


def test_coverage_counts_the_angles_a_camera_actually_reached():
    """Measured, not assumed. A clip of someone facing forward scores low."""
    still = [a_pose(i, np.eye(3, dtype="float32")) for i in range(40)]
    assert reconstruct._coverage(still) == round(1 / reconstruct.YAW_BINS, 2)

    turning = [a_pose(i, rotation_about_y(d)) for i, d in enumerate(range(-90, 91, 10))]
    assert reconstruct._coverage(turning) > reconstruct._coverage(still)


def test_no_views_is_no_coverage():
    assert reconstruct._coverage([]) == 0.0


def test_the_calmest_frames_are_kept_and_the_angles_are_not_collapsed():
    """Both filters at once: rigid frames, without losing the spread.

    Keeping only the calmest frames would bias towards whichever direction
    someone happened to be facing when they stopped talking.
    """
    poses = []
    for bucket, degrees in enumerate((-60, -20, 20, 60)):
        for i in range(10):
            pose = a_pose(bucket * 10 + i, rotation_about_y(degrees))
            poses.append(
                reconstruct.HeadPose(
                    index=pose.index, image=pose.image, view=pose.view,
                    yaw=pose.yaw, pitch=pose.pitch, expression=i / 10.0,
                )
            )

    kept = reconstruct._rigid_spread(poses)

    assert len(kept) == 4 * reconstruct.MAX_FRAMES_PER_BIN
    # The mid-word frames are the ones dropped, in every bucket.
    assert max(p.expression for p in kept) < 0.3
    assert len({round(p.yaw, 3) for p in kept}) == 4
