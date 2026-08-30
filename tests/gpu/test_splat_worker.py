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


# --------------------------------------------------------------------------
# pose refinement, which is also arithmetic and also needs no GPU
#
# gsplat is not installed here, so nothing below renders a Gaussian. What it
# does instead is drive the exact functions that sit inside the render loop -
# _axis_angle_to_rotation, _corrected_views, _pose_regulariser and
# _clamp_pose_correction - around a toy scene of known 3D points seen by a
# known camera, where the right answer exists and the wrong answer is
# measurable in degrees. A parameterisation that cannot recover a camera from
# the reprojection error of three hundred points will not recover one from a
# splat, and this is where that is found out.

torch = pytest.importorskip("torch")

TOY_IMAGE_EDGE = 512


def axis_angle_rotation(axis, degrees):
    """Rodrigues in numpy, written out rather than imported.

    Deliberately a second implementation: building the test's perturbation with
    the same function the code under test uses would make a sign error in that
    function invisible, because it would be applied twice and cancel.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(degrees)
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def rotation_error_degrees(a, b):
    """The angle of the rotation that takes a to b. The metric that matters."""
    relative = np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64).T
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))


def a_view(rotation, translation=(0.0, 0.0, 0.5)):
    """One world-to-camera matrix, batched as gsplat and _corrected_views want."""
    view = torch.eye(4)
    view[:3, :3] = torch.as_tensor(np.asarray(rotation), dtype=torch.float32)
    view[:3, 3] = torch.tensor(translation, dtype=torch.float32)
    return view[None]


def toy_points(seed=7):
    """A head-sized cloud around the origin of the head-fixed world.

    The size is not decoration. All of the optimiser's leverage on a rotation
    comes from parallax across the cloud's own depth, so a box far smaller or
    far larger than a head would make this test easier or harder than the
    problem it stands in for.
    """
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.uniform(-0.12, 0.12, size=(300, 3)), dtype=torch.float32)


def project(points, view):
    """Pixels, through the same intrinsic the poses were solved against."""
    k = reconstruct._intrinsics(TOY_IMAGE_EDGE, TOY_IMAGE_EDGE)
    camera = points @ view[0, :3, :3].T + view[0, :3, 3]
    focal = torch.tensor([float(k[0, 0]), float(k[1, 1])])
    centre = torch.tensor([float(k[0, 2]), float(k[1, 2])])
    return camera[:, :2] / camera[:, 2:3] * focal + centre


# What one view is actually stepped on a standard build: 15,000 iterations,
# thirty per cent of them warmup, spread over about thirty kept frames. The toy
# gets exactly that many and no more, so a result here is a statement about
# what the shipping settings can do rather than about what an unlimited
# optimiser could do given a scene this easy.
STANDARD_BUILD_ITERATIONS = 15_000
STANDARD_BUILD_VIEWS = 30
STEPS_ONE_VIEW_RECEIVES = 350


def refine(prior, points, observed, *, steps=STEPS_ONE_VIEW_RECEIVES, regularise=True):
    """Optimise only the pose correction against known pixels.

    The optimiser, the learning rates, the regulariser and the bound are all
    the production ones - only the loss is replaced, because the real one needs
    a rasteriser that is not installed here. Squared reprojection error stands
    in for it: same quantity, fewer Gaussians.
    """
    rotation_delta = torch.zeros(1, 3, requires_grad=True)
    translation_delta = torch.zeros(1, 3, requires_grad=True)
    optimizer = reconstruct._pose_optimiser(
        rotation_delta, translation_delta,
        *reconstruct._pose_learning_rates(
            iterations=STANDARD_BUILD_ITERATIONS,
            views=STANDARD_BUILD_VIEWS,
            warmup=int(STANDARD_BUILD_ITERATIONS * reconstruct.POSE_WARMUP_FRACTION),
        ),
    )

    for _ in range(steps):
        view = reconstruct._corrected_views(prior, rotation_delta, translation_delta)
        loss = ((project(points, view) - observed) ** 2).mean()
        if regularise:
            loss = loss + reconstruct._pose_regulariser(rotation_delta, translation_delta)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        reconstruct._clamp_pose_correction(rotation_delta, translation_delta)

    return rotation_delta.detach(), translation_delta.detach()


def a_toy_capture(prior_error_deg, axis=(0.3, 1.0, -0.2)):
    """A true camera, the pixels it saw, and a prior that is wrong by a known angle.

    The prior is composed on the same side the correction is, so the error is
    exactly representable: the test then measures whether the optimiser finds
    it, rather than whether the parameterisation happens to span it.
    """
    points = toy_points()
    truth = a_view(rotation_about_y(12))
    observed = project(points, truth).detach()

    wrong = torch.eye(4)
    wrong[:3, :3] = torch.as_tensor(
        axis_angle_rotation(axis, prior_error_deg), dtype=torch.float32
    )
    prior = (wrong[None] @ truth).float()
    return points, observed, prior, truth


def test_rodrigues_matches_a_second_implementation_of_itself():
    """The one piece of arithmetic a sign error would hide inside."""
    for degrees in (0.0, 1.0, 7.5, 45.0, 179.0):
        axis = np.array([0.3, 1.0, -0.2])
        expected = axis_angle_rotation(axis, degrees)
        omega = torch.tensor(
            (axis / np.linalg.norm(axis)) * np.radians(degrees), dtype=torch.float64
        )[None]

        got = reconstruct._axis_angle_to_rotation(omega)[0].numpy()

        assert np.allclose(got, expected, atol=1e-9)


def test_a_correction_of_zero_leaves_the_prior_exactly_where_it_was():
    """The property that makes MediaPipe's estimate an anchor rather than a guess.

    Every one of these parameters is initialised at zero, so the first
    iteration of a run must render through the prior unchanged - bit for bit,
    not to within a tolerance. A parameterisation that perturbs the pose before
    it has learned anything has already thrown away the prior.
    """
    prior = a_view(rotation_about_y(23))
    zero = torch.zeros(1, 3)

    corrected = reconstruct._corrected_views(prior, zero, zero)

    assert torch.equal(corrected, prior)
    assert float(reconstruct._pose_regulariser(zero, zero)) == 0.0


def test_the_first_backward_pass_at_the_prior_produces_numbers_and_not_nans():
    """Where the naive Rodrigues formula fails, silently and expensively.

    Dividing by the rotation angle is undefined at zero, and zero is where
    these parameters spend the whole warmup. A NaN here does not announce
    itself: it flows into the extrinsics, the render comes out empty, the loss
    stays finite and the build completes having learned nothing at all.
    """
    points, observed, prior, _ = a_toy_capture(prior_error_deg=3.0)
    rotation_delta = torch.zeros(1, 3, requires_grad=True)
    translation_delta = torch.zeros(1, 3, requires_grad=True)

    view = reconstruct._corrected_views(prior, rotation_delta, translation_delta)
    loss = ((project(points, view) - observed) ** 2).mean()
    loss = loss + reconstruct._pose_regulariser(rotation_delta, translation_delta)
    loss.backward()

    assert torch.isfinite(rotation_delta.grad).all()
    assert torch.isfinite(translation_delta.grad).all()
    # A prior that is wrong must actually push. A finite gradient of zero would
    # pass the check above and refine nothing.
    assert float(rotation_delta.grad.abs().sum()) > 0.0


def test_a_corrected_view_is_still_a_world_to_camera_matrix():
    """A drifting parameterisation shows up as a scene that quietly scales."""
    prior = a_view(rotation_about_y(-31))
    rotation_delta = torch.tensor([[0.04, -0.02, 0.01]])
    translation_delta = torch.tensor([[0.005, 0.002, -0.003]])

    view = reconstruct._corrected_views(prior, rotation_delta, translation_delta)
    rotation = view[0, :3, :3]

    assert torch.allclose(rotation @ rotation.T, torch.eye(3), atol=1e-6)
    assert float(torch.det(rotation)) == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(view[0, 3, :], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_the_correction_moves_a_wrongly_placed_camera_towards_the_truth():
    """The whole reason the poses were unfrozen, measured on a toy scene.

    Three degrees is the error MediaPipe actually makes, and at this focal
    length it is tens of pixels of disagreement between two views of the same
    cheekbone - which a splat can only explain by going translucent. If the
    correction cannot take that back out, unfreezing the poses has bought
    nothing and has cost six parameters a view.
    """
    points, observed, prior, truth = a_toy_capture(prior_error_deg=3.0)

    before_degrees = rotation_error_degrees(prior[0, :3, :3], truth[0, :3, :3])
    before_pixels = float(((project(points, prior) - observed) ** 2).mean())

    rotation_delta, translation_delta = refine(prior, points, observed)
    corrected = reconstruct._corrected_views(prior, rotation_delta, translation_delta)

    after_degrees = rotation_error_degrees(corrected[0, :3, :3], truth[0, :3, :3])
    after_pixels = float(((project(points, corrected) - observed) ** 2).mean())

    assert before_degrees == pytest.approx(3.0, abs=0.05)
    # Reduced, not merely changed: the failure this guards against is a
    # correction that fits the pixels by moving the camera further from where
    # the camera was. Measured at the shipping settings, three degrees comes
    # down to roughly two tenths of one.
    assert after_degrees < before_degrees / 10.0
    assert after_pixels < before_pixels / 50.0


@pytest.mark.parametrize("prior_error_deg", [0.5, 1.0, 3.0, 6.0])
def test_the_correction_reduces_the_pose_error_at_every_size_of_prior_error(
    prior_error_deg,
):
    """Including the small ones, where a refinement can do harm.

    A correction that helps a badly wrong prior and hurts a nearly right one is
    worse than no correction at all, because most captures are nearly right.
    """
    points, observed, prior, truth = a_toy_capture(prior_error_deg)

    rotation_delta, translation_delta = refine(prior, points, observed)
    corrected = reconstruct._corrected_views(prior, rotation_delta, translation_delta)

    before = rotation_error_degrees(prior[0, :3, :3], truth[0, :3, :3])
    after = rotation_error_degrees(corrected[0, :3, :3], truth[0, :3, :3])

    assert after < before


def test_a_pose_cannot_run_away_from_an_adversarially_wrong_prior():
    """A correction of thirty degrees is not a refinement, and is not allowed.

    A prior this wrong means something else is broken - the capture, or the
    assumption that a phone films at sixty-three degrees - and an optimiser
    permitted to accommodate it silently converts a diagnosable bad capture
    into an inexplicable bad avatar. The bound holds even when the data is
    begging the pose to keep going.
    """
    points, observed, prior, _ = a_toy_capture(prior_error_deg=60.0)

    rotation_delta, translation_delta = refine(prior, points, observed)

    correction_degrees = float(torch.rad2deg(torch.linalg.norm(rotation_delta, dim=-1)).max())
    correction_metres = float(torch.linalg.norm(translation_delta, dim=-1).max())

    assert correction_degrees <= reconstruct.MAX_POSE_ROTATION_DEG + 1e-4
    assert correction_metres <= reconstruct.MAX_POSE_TRANSLATION_M + 1e-6
    # And it did strain against the bound, so the diagnostic below is not
    # passing for the trivial reason that nothing moved.
    assert correction_degrees > reconstruct.SUSPECT_POSE_CORRECTION_DEG


def test_the_clamp_bounds_every_view_it_is_handed():
    """One badly landmarked frame must not be able to take its camera with it."""
    rotation_delta = torch.tensor([[3.0, -4.0, 12.0], [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    translation_delta = torch.tensor([[9.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])

    reconstruct._clamp_pose_correction(rotation_delta, translation_delta)

    degrees = torch.rad2deg(torch.linalg.norm(rotation_delta, dim=-1))
    metres = torch.linalg.norm(translation_delta, dim=-1)

    assert float(degrees.max()) <= reconstruct.MAX_POSE_ROTATION_DEG + 1e-4
    assert float(metres.max()) <= reconstruct.MAX_POSE_TRANSLATION_M + 1e-6
    # A correction already inside the ceiling is left exactly alone; the clamp
    # is a wall, not a shrinkage applied to every view on every step.
    assert float(degrees[2]) == pytest.approx(np.degrees(0.01), abs=1e-6)
    assert float(metres[2]) == pytest.approx(0.001, abs=1e-9)


def test_the_regulariser_is_free_at_the_prior_and_expensive_at_the_ceiling():
    """Why a large correction loses an argument with a small one.

    Expressed as a fraction of the hard ceiling so that one weight governs both
    rotation and translation, which is the only reason the two terms are
    comparable at all.
    """
    zero = torch.zeros(1, 3)
    small = torch.tensor([[np.radians(1.0), 0.0, 0.0]], dtype=torch.float32)
    at_ceiling = torch.tensor(
        [[np.radians(reconstruct.MAX_POSE_ROTATION_DEG), 0.0, 0.0]], dtype=torch.float32
    )

    free = float(reconstruct._pose_regulariser(zero, zero))
    cheap = float(reconstruct._pose_regulariser(small, zero))
    dear = float(reconstruct._pose_regulariser(at_ceiling, zero))

    assert free == 0.0
    assert cheap < dear / 50.0
    assert dear == pytest.approx(reconstruct.POSE_REGULARISATION, rel=1e-4)


def test_the_regulariser_pulls_an_unsupported_correction_back_to_the_prior():
    """With no pixels asking for it, a correction decays rather than persisting."""
    rotation_delta = torch.tensor([[0.08, 0.0, 0.0]], requires_grad=True)
    translation_delta = torch.tensor([[0.02, 0.0, 0.0]], requires_grad=True)
    optimizer = torch.optim.SGD([rotation_delta, translation_delta], lr=1e-2)

    started = float(torch.linalg.norm(rotation_delta.detach()))
    for _ in range(200):
        loss = reconstruct._pose_regulariser(rotation_delta, translation_delta)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    assert float(torch.linalg.norm(rotation_delta.detach())) < started
    assert float(torch.linalg.norm(translation_delta.detach())) < 0.02


# --------------------------------------------------------------------------
# what the pose refinement tells the people who have to explain a build


def a_params_dict(n=4):
    """The parameter names _cloud_from reads, with nothing interesting in them."""
    return {
        "means": torch.zeros(n, 3),
        "scales": torch.full((n, 3), float(np.log(0.01))),
        "quats": torch.tile(torch.tensor([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        "opacities": torch.zeros(n),
        "sh0": torch.zeros(n, 1, 3),
    }


def test_the_pose_correction_magnitudes_reach_the_cloud():
    """They are diagnostic, so they must survive the trip off the card.

    A family's failed avatar has to be explainable, and after the run the only
    remaining evidence of why the views disagreed is how far the optimiser had
    to move them.
    """
    poses = [a_pose(i, rotation_about_y(d)) for i, d in enumerate(range(-40, 41, 10))]
    rotation_delta = torch.tensor([[np.radians(2.0), 0.0, 0.0]] * len(poses))
    translation_delta = torch.tensor([[0.004, 0.0, 0.0]] * len(poses))

    correction = reconstruct._pose_correction_report(rotation_delta, translation_delta)
    cloud = reconstruct._cloud_from(a_params_dict(), poses, correction)

    assert cloud.pose_correction.mean_deg == pytest.approx(2.0, abs=1e-2)
    assert cloud.pose_correction.max_deg == pytest.approx(2.0, abs=1e-2)
    assert cloud.pose_correction.mean_mm == pytest.approx(4.0, abs=1e-2)
    assert cloud.pose_correction.views == len(poses)
    assert cloud.pose_correction.suspect is False


def test_a_small_correction_is_reported_and_a_large_one_is_called_out():
    """The difference between "MediaPipe was right" and "the capture was wrong".

    Both end up in the notes; only one of them says so in words a support agent
    can act on, because only one of them is a fault.
    """
    poses = [a_pose(i, rotation_about_y(d)) for i, d in enumerate(range(-40, 41, 10))]
    span = reconstruct._yaw_span_degrees(poses)

    modest = reconstruct.PoseCorrection(mean_deg=0.6, max_deg=1.1, mean_mm=3.0, views=9)
    alarming = reconstruct.PoseCorrection(mean_deg=7.4, max_deg=9.9, mean_mm=31.0, views=9)

    quiet = reconstruct._notes(
        poses, poses,
        a_cloud(coverage=0.25, views=9),
        heard=True,
    )
    assert not any("large correction" in note for note in quiet)

    for correction, expected in ((modest, False), (alarming, True)):
        cloud = reconstruct.GaussianCloud(
            **{**a_cloud(coverage=0.25, views=9).__dict__,
               "yaw_span_deg": span, "pose_correction": correction},
        )
        notes = reconstruct._notes(poses, poses, cloud, heard=True)

        assert any(f"{correction.mean_deg:.2f} degrees" in note for note in notes)
        assert any("large correction" in note for note in notes) is expected
        if expected:
            # It must name both suspects, because the family can only fix one
            # of them and support has to know which to ask about.
            assert any("63-degree camera" in note for note in notes)


def test_the_notes_the_worker_reports_reach_the_job_result(offline, monkeypatch):
    """A diagnostic nobody can read is not a diagnostic.

    handler.RESULT_FIELDS carries `notes` and nothing that could be mistaken
    for a quality claim, so this is the channel the pose magnitudes travel on.
    """
    monkeypatch.setattr(
        reconstruct, "reconstruct",
        lambda video, **kw: a_cloud(notes=("the camera poses were refined by 0.42 degrees",)),
    )

    result = run(plan(a_video_intake(), AVATAR).payload())

    assert "the camera poses were refined by 0.42 degrees" in result["notes"]


# --------------------------------------------------------------------------
# coverage: a frontal shell is a shape, not a failure


def a_pose_at_yaw(index, degrees, expression=0.0, speech=0.0):
    """A pose with its yaw stated outright.

    _yaw_pitch cannot express a yaw beyond a quarter turn, so a clip that
    straddles the wrap at pi has to be built directly rather than from a
    rotation matrix.
    """
    view = np.eye(4, dtype="float32")
    view[:3, 3] = np.array([0.0, 0.0, 0.5], dtype="float32")
    return reconstruct.HeadPose(
        index=index, image=np.zeros((8, 8, 3), dtype="uint8"), view=view,
        yaw=float(np.radians(degrees)), pitch=0.0,
        expression=expression, speech=speech,
    )


def test_a_forty_degree_traverse_is_reported_as_a_shape_and_not_as_a_shortfall():
    """What a person talking to a propped phone actually produces.

    Roughly a fifth of the circle, every time, however well they followed the
    instructions. A bare 0.22 next to no sentence is how a perfectly good
    head-and-shoulders avatar gets thrown away, so the note says which of the
    two things it is: complete at the front, empty behind the ears.
    """
    poses = [a_pose_at_yaw(i, d) for i, d in enumerate(range(-40, 41, 5))]

    coverage = reconstruct._coverage(poses)
    span = reconstruct._yaw_span_degrees(poses)

    assert 0.15 <= coverage <= 0.30
    assert span == pytest.approx(80.0, abs=0.5)

    cloud = reconstruct.GaussianCloud(
        **{**a_cloud(coverage=coverage, views=len(poses)).__dict__, "yaw_span_deg": span}
    )
    notes = reconstruct._notes(poses, poses, cloud, heard=True)
    text = " ".join(notes)

    assert "frontal shell" in text
    assert "80 degrees" in text
    assert "not a fault" in text
    # The number is still stated. Honest means explained, not hidden.
    assert str(coverage) in text


def test_a_head_that_really_did_turn_is_not_described_as_a_frontal_shell():
    poses = [a_pose_at_yaw(i, d) for i, d in enumerate(range(-90, 91, 10))]
    span = reconstruct._yaw_span_degrees(poses)

    cloud = reconstruct.GaussianCloud(
        **{**a_cloud(views=len(poses)).__dict__, "yaw_span_deg": span}
    )
    text = " ".join(reconstruct._notes(poses, poses, cloud, heard=True))

    assert span == pytest.approx(180.0, abs=0.5)
    assert "frontal shell" not in text
    assert "seen from the side" in text


def test_the_yaw_span_is_the_arc_the_views_occupy_and_survives_the_wrap():
    """A clip that happens to straddle pi has not swept the whole way round."""
    straddling = [a_pose_at_yaw(i, d) for i, d in enumerate((170.0, 175.0, -175.0, -170.0))]

    assert reconstruct._yaw_span_degrees(straddling) == pytest.approx(20.0, abs=0.5)
    assert reconstruct._yaw_span_degrees([]) == 0.0
    assert reconstruct._yaw_span_degrees([a_pose_at_yaw(0, 12.0)]) == 0.0


# --------------------------------------------------------------------------
# frame selection: the calmest frame of continuous speech is still mid-phoneme


def a_talking_bin(expressions, speech):
    """Several frames at one yaw, differing only in how mid-word they are."""
    return [
        a_pose_at_yaw(index, 0.0, expression=expression, speech=loudness)
        for index, (expression, loudness) in enumerate(zip(expressions, speech, strict=True))
    ]


def test_frames_from_the_quiet_seconds_beat_the_calmest_frames_of_speech():
    """The failure this defends against is a sharp forehead over a smeared mouth.

    Blendshapes alone pick the least-open mouth of a face that never stopped
    moving. The audio knows the difference between a closed mouth between two
    words and a closed mouth in a silence, and only the second one has lips
    that agree with every other frame.
    """
    poses = a_talking_bin(
        expressions=[0.10, 0.12, 0.14, 0.30, 0.31, 0.32],
        speech=[0.90, 0.90, 0.90, 0.00, 0.00, 0.00],
    )

    kept = reconstruct._rigid_spread(poses)

    assert sorted(p.index for p in kept) == [3, 4, 5]


def test_a_clip_with_no_audio_track_chooses_exactly_as_it_did_before():
    """Audio improves the choice; it is never a condition of making one."""
    poses = a_talking_bin(
        expressions=[0.10, 0.12, 0.14, 0.30, 0.31, 0.32],
        speech=[0.0] * 6,
    )

    kept = reconstruct._rigid_spread(poses)

    assert sorted(p.index for p in kept) == [0, 1, 2]


def test_the_loudness_envelope_finds_the_quiet_half_of_a_clip(monkeypatch):
    """Measured from the decoded samples, normalised against this clip's own loudest.

    An absolute threshold would be meaningless: the input is a phone microphone
    at an unknown distance, and the only answerable question is which parts of
    *this* recording are its quiet ones.
    """
    rate = reconstruct.AUDIO_SAMPLE_RATE
    time = np.arange(4 * rate) / rate
    tone = np.sin(2 * np.pi * 220.0 * time) * 0.5
    tone[int(2.0 * rate):] = 0.0  # the person stops talking two seconds in
    pcm = (tone * 32767).astype("<i2").tobytes()

    monkeypatch.setattr(reconstruct, "_decode_audio", lambda video: pcm)

    envelope = reconstruct._speech_envelope(Path("clip.mp4"), frame_count=8)

    assert envelope[0] == pytest.approx(1.0, abs=1e-6)
    assert envelope[7] == 0.0
    assert envelope[6] < 0.05
    assert envelope[1] > 0.9


@pytest.mark.parametrize(
    "pcm", [b"", b"\x00", (np.zeros(16_000, dtype="<i2")).tobytes()],
    ids=["no audio track", "a truncated sample", "digital silence"],
)
def test_an_unusable_audio_track_is_not_mistaken_for_a_quiet_moment(monkeypatch, pcm):
    """Nothing heard means nothing claimed.

    A track that is absent, unreadable or entirely silent carries no
    information about which frames are quiet, and must not be allowed to score
    every frame as quiet - which would silently disable the blendshape ordering
    it was added to improve.
    """
    monkeypatch.setattr(reconstruct, "_decode_audio", lambda video: pcm)

    assert reconstruct._speech_envelope(Path("clip.mp4"), frame_count=8) is None


def test_a_clip_whose_audio_could_not_be_decoded_says_so_in_the_notes():
    """Support should not have to guess why the mouth came out soft."""
    poses = [a_pose_at_yaw(i, d) for i, d in enumerate(range(-40, 41, 5))]
    cloud = reconstruct.GaussianCloud(
        **{**a_cloud(coverage=0.25, views=len(poses)).__dict__,
           "yaw_span_deg": reconstruct._yaw_span_degrees(poses)}
    )

    deaf = " ".join(reconstruct._notes(poses, poses, cloud, heard=False))
    heard = " ".join(reconstruct._notes(poses, poses, cloud, heard=True))

    assert "no usable audio" in deaf
    assert "no usable audio" not in heard


def test_ffmpeg_failing_is_a_silent_clip_and_not_a_failed_build(monkeypatch):
    """Every way the decode can go wrong ends in the same harmless place."""
    def explode(*args, **kwargs):
        raise OSError("ffmpeg is not installed")

    monkeypatch.setattr(reconstruct.subprocess, "run", explode)

    assert reconstruct._decode_audio(Path("clip.mp4")) == b""


# --------------------------------------------------------------------------
# the render loop itself, with a stand-in where the rasteriser would be
#
# The arithmetic above proves the pose parameterisation is correct. It cannot
# prove it is wired in, and the ways this wiring fails are silent: poses
# released during the warmup, pose tensors handed to the densification
# strategy, a correction that is computed and never applied. gsplat is not
# installed here, so `rasterization` and `MCMCStrategy` are replaced by
# stand-ins that are differentiable in everything the real ones are. Nothing
# below is a claim about image quality; it is a claim about the loop.


class FakeStrategy:
    """MCMC's interface, and a record of what was handed to it."""

    def __init__(self, cap_max, verbose=False):
        self.cap_max = cap_max
        self.optimised_shapes = []

    def initialize_state(self):
        return {}

    def _record(self, params, optimizers):
        for name, optimizer in optimizers.items():
            for group in optimizer.param_groups:
                for tensor in group["params"]:
                    self.optimised_shapes.append((name, tuple(tensor.shape)))

    def step_pre_backward(self, params, optimizers, state, step, info):
        self._record(params, optimizers)

    def step_post_backward(self, params, optimizers, state, step, info, lr=None):
        self._record(params, optimizers)


@pytest.fixture
def fake_gsplat(monkeypatch):
    """gsplat's two entry points, replaced by differentiable stand-ins."""
    import types

    seen = {"viewmats_needed_gradient": [], "strategy": None}

    def fake_rasterization(*, means, quats, scales, opacities, colors,
                           viewmats, Ks, width, height, sh_degree, packed):
        # Whether the extrinsics are part of the graph *is* the warmup gate, so
        # it is recorded rather than inferred.
        seen["viewmats_needed_gradient"].append(bool(viewmats.requires_grad))

        camera = means @ viewmats[0, :3, :3].T + viewmats[0, :3, 3]
        projected = (camera[:, :2] / camera[:, 2:3].clamp(min=1e-3)).mean()
        value = projected + colors.mean() + scales.mean() + opacities.mean() + quats.mean()
        rendered = value * torch.ones(1, height, width, 3)
        info = {"width": width, "height": height, "n_cameras": 1,
                "radii": torch.zeros(1, means.shape[0], dtype=torch.int32),
                "means2d": torch.zeros(1, means.shape[0], 2)}
        return rendered, torch.ones(1, height, width, 1), info

    def build_strategy(cap_max, verbose=False):
        seen["strategy"] = FakeStrategy(cap_max, verbose)
        return seen["strategy"]

    gsplat = types.ModuleType("gsplat")
    gsplat.rasterization = fake_rasterization
    strategy_module = types.ModuleType("gsplat.strategy")
    strategy_module.MCMCStrategy = build_strategy
    gsplat.strategy = strategy_module

    monkeypatch.setitem(sys.modules, "gsplat", gsplat)
    monkeypatch.setitem(sys.modules, "gsplat.strategy", strategy_module)
    return seen


def a_masked_view(index, degrees, edge=16):
    """One pose and one mask, small enough that the loop is the slow part."""
    pose = a_pose(index, rotation_about_y(degrees), image_size=edge)
    coloured = reconstruct.HeadPose(
        index=pose.index,
        image=np.full((edge, edge, 3), 128, dtype="uint8"),
        view=pose.view, yaw=pose.yaw, pitch=pose.pitch, expression=0.0,
    )
    return coloured, np.ones((edge, edge), dtype="float32")


def test_the_poses_are_frozen_for_the_warmup_and_learnable_afterwards(fake_gsplat):
    """Released together, the poses absorb error that belongs to the geometry.

    Six parameters per view move faster than a million Gaussians, so a pose let
    loose before there is any geometry to disagree with will happily explain
    the residual of a cloud that has not formed yet - and the run ends with a
    beautifully fitted set of wrong cameras.
    """
    iterations = 300
    poses, masks = zip(*[a_masked_view(i, d) for i, d in enumerate((-20, -10, 0, 10))],
                       strict=True)

    reconstruct.optimise(
        list(poses), list(masks),
        iterations=iterations, gaussian_budget=64, device="cpu",
    )

    gate = fake_gsplat["viewmats_needed_gradient"]
    warmup = max(len(poses), int(iterations * reconstruct.POSE_WARMUP_FRACTION))

    assert len(gate) == iterations
    assert not any(gate[:warmup])
    assert all(gate[warmup:])


def test_the_pose_parameters_are_kept_away_from_the_densification_strategy(fake_gsplat):
    """MCMCStrategy assumes every tensor it is given is one row per Gaussian.

    It relocates and duplicates them with the cloud. A (views, 3) tensor in
    that dictionary is not rejected - it is reindexed, and the poses become
    whatever the strategy last decided about the Gaussians.
    """
    poses, masks = zip(*[a_masked_view(i, d) for i, d in enumerate((-20, 0, 20))],
                       strict=True)

    cloud = reconstruct.optimise(
        list(poses), list(masks), iterations=20, gaussian_budget=64, device="cpu",
    )

    shapes = fake_gsplat["strategy"].optimised_shapes
    assert shapes, "the strategy was never stepped"
    assert {name for name, _ in shapes} == {
        "means", "scales", "quats", "opacities", "sh0"
    }
    # Every tensor the strategy can reach has one row per Gaussian, and three
    # views is not a count of Gaussians.
    assert all(shape[0] == cloud.count for _, shape in shapes)


def test_a_finished_optimisation_reports_how_far_it_moved_the_cameras(fake_gsplat):
    """The correction is applied, bounded, and carried out of the loop."""
    poses, masks = zip(*[a_masked_view(i, d) for i, d in enumerate((-20, -10, 0, 10))],
                       strict=True)

    cloud = reconstruct.optimise(
        list(poses), list(masks), iterations=300, gaussian_budget=64, device="cpu",
    )

    correction = cloud.pose_correction
    assert correction.views == len(poses)
    # It moved: a correction reported as exactly zero after three hundred
    # iterations means the optimiser stepped something else.
    assert correction.max_deg > 0.0
    assert correction.max_deg <= reconstruct.MAX_POSE_ROTATION_DEG
    assert correction.max_mm <= reconstruct.MAX_POSE_TRANSLATION_M * 1000 + 1e-6
    # And the measured coverage still describes the views it was given.
    assert cloud.yaw_span_deg == pytest.approx(30.0, abs=0.5)
    assert cloud.views_used == len(poses)


def test_a_run_too_short_to_warm_up_never_unfreezes_the_poses_at_all(
    fake_gsplat,
):
    """The prior is the answer until something measured disagrees with it.

    A fraction of a very short run rounds to nothing, which would release every
    camera on the first step against a cloud that is still the seed points. The
    warmup is floored at one pass over the views, so a run shorter than that
    refines no pose rather than refining them all against nothing.
    """
    poses, masks = zip(*[a_masked_view(i, d) for i, d in enumerate((-10, 0, 10))],
                       strict=True)

    cloud = reconstruct.optimise(
        list(poses), list(masks), iterations=1, gaussian_budget=64, device="cpu",
    )

    assert cloud.pose_correction.max_deg == 0.0
    assert cloud.pose_correction.max_mm == 0.0


def test_a_view_that_was_not_rendered_this_step_does_not_move():
    """Adam does not stand still on a zero gradient, and here it must.

    One view is rendered per step, so every other view's gradient is exactly
    zero - and plain Adam keeps spending the momentum from the last step that
    did move. Across thirty views that is roughly nine free steps of the last
    real direction before a pose is next rendered, so each camera travels about
    ten times as far as its learning rate claims. The learning rate here is
    chosen as a displacement budget, so it has to mean what it says.
    """
    rotation_delta = torch.zeros(3, 3, requires_grad=True)
    translation_delta = torch.zeros(3, 3, requires_grad=True)
    optimizer = reconstruct._pose_optimiser(rotation_delta, translation_delta, 1e-4, 1e-4)

    # One step in which only the middle view has an opinion.
    rotation_delta.grad = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    translation_delta.grad = torch.zeros(3, 3)
    optimizer.step()

    moved = rotation_delta.detach().clone()
    assert float(moved[1, 0]) != 0.0
    assert float(moved[0].abs().sum()) == 0.0

    # Three steps in which nobody does. Nothing may drift.
    for _ in range(3):
        rotation_delta.grad = torch.zeros(3, 3)
        translation_delta.grad = torch.zeros(3, 3)
        optimizer.step()

    assert torch.equal(rotation_delta.detach(), moved)


def test_one_step_moves_a_pose_by_about_its_learning_rate():
    """Which is what lets the learning rate be read as a budget in radians.

    Roughly three hundred and fifty steps reach a view over a standard run, so
    a step of one learning rate is about two degrees of reachable correction:
    the size of the error MediaPipe actually makes, and not the size of the
    error that would let the poses race the Gaussians for the same residual.
    """
    rotation_delta = torch.zeros(1, 3, requires_grad=True)
    translation_delta = torch.zeros(1, 3, requires_grad=True)
    rotation_lr, translation_lr = reconstruct._pose_learning_rates(
        iterations=15_000, views=30, warmup=4_500
    )
    optimizer = reconstruct._pose_optimiser(
        rotation_delta, translation_delta, rotation_lr, translation_lr
    )

    # The gradient magnitude is deliberately absurd: Adam's second moment makes
    # the step independent of it, which is why a bright clip and a dim one
    # refine at the same rate.
    rotation_delta.grad = torch.tensor([[4_000.0, 0.0, 0.0]])
    translation_delta.grad = torch.tensor([[0.0, -0.05, 0.0]])
    optimizer.step()

    assert float(rotation_delta.detach().abs().max()) == pytest.approx(
        rotation_lr, rel=0.01
    )
    assert float(translation_delta.detach().abs().max()) == pytest.approx(
        translation_lr, rel=0.01
    )


def test_a_prior_that_is_already_right_is_left_alone():
    """The property that decides whether unfreezing the poses is safe at all.

    Most views in a good capture are nearly correct, so a refinement that pays
    for its wins on the bad frames by injecting error into the good ones is a
    net loss - and would be invisible, because the loss it optimises would go
    down either way.
    """
    points = toy_points()
    truth = a_view(rotation_about_y(12))
    observed = project(points, truth).detach()

    rotation_delta, translation_delta = refine(truth, points, observed)
    corrected = reconstruct._corrected_views(truth, rotation_delta, translation_delta)

    assert rotation_error_degrees(corrected[0, :3, :3], truth[0, :3, :3]) < 1e-4
    assert float(torch.linalg.norm(corrected[0, :3, 3] - truth[0, :3, 3])) < 1e-5


def test_a_camera_may_traverse_its_whole_ceiling_over_a_run_and_no_further():
    """The learning rate is derived from the run, so the run may change.

    Fixing it as a constant would mean the refinement quietly stopped reaching
    far enough the day somebody raised or lowered Quality.iterations - and it
    would stop by producing a slightly blurrier avatar, which nobody would
    trace back to a number in this file.
    """
    for iterations, views in ((15_000, 30), (30_000, 30), (60_000, 12)):
        warmup = int(iterations * reconstruct.POSE_WARMUP_FRACTION)
        rotation_lr, translation_lr = reconstruct._pose_learning_rates(
            iterations, views, warmup
        )
        steps = (iterations - warmup) // views

        assert np.degrees(rotation_lr * steps) == pytest.approx(
            reconstruct.MAX_POSE_ROTATION_DEG, rel=0.02
        )
        assert translation_lr * steps == pytest.approx(
            reconstruct.MAX_POSE_TRANSLATION_M, rel=0.02
        )


def test_a_short_run_refines_more_slowly_rather_than_lurching():
    """A preview divides the same ceiling over a tenth of the steps.

    Left underived, that is a camera permitted to jump a third of a degree on
    one noisy image gradient. A preview that refines less is the right trade; a
    preview that shakes its cameras is not, because the customer is looking at
    the preview when they decide whether any of this works.
    """
    preview = reconstruct._pose_learning_rates(3_000, 30, 900)
    standard = reconstruct._pose_learning_rates(15_000, 30, 4_500)

    assert preview[0] == reconstruct.POSE_ROTATION_LR_MAX
    assert preview[1] == reconstruct.POSE_TRANSLATION_LR_MAX
    assert standard[0] <= reconstruct.POSE_ROTATION_LR_MAX
    # Which still leaves a preview able to reach the error MediaPipe makes.
    assert np.degrees(preview[0] * ((3_000 - 900) // 30)) > 2.0
