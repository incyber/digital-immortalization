"""What a splat rig has to be true for, tested with no GPU and no FLAME asset.

Every mesh here is synthetic - a triangle, a grid, a sphere - and every
Gaussian is a number. That is deliberate and it is the same posture the rest of
this project takes towards GPUs: the geometry either holds or it does not, and
whether it holds is decided by arithmetic that a laptop can check in a second.

Four of these tests are about one failure. A rigged splat tears when a Gaussian
that is not part of the face gets bound to the face anyway, or when the frame a
Gaussian rides on is not actually rigid. Tearing does not look like a bug; it
looks like the person's hair being dragged through their cheek, and the person
is somebody's dead father. So the static gate, the rigid-motion invariant, the
round trip and the off-edge gate are each asserted directly rather than being
left to follow from the others.
"""

import time

import numpy as np
import pytest

from avatar.motion.pose import CHANNELS, VISEME_COUNT, PoseFrame, channel_names
from avatar.splat.rig import (
    BIND_RADIUS_IN_EDGES,
    CHANNEL_MAP,
    FLAME_EXPRESSION_COUNT,
    FLAME_JAW_MAX_RAD,
    LIMITATIONS,
    MAX_BARYCENTRIC_EXCURSION,
    VISEME_MAP,
    VISEME_NAMES,
    Bindings,
    BindMode,
    ExpressionBasis,
    Fidelity,
    GaussianBinding,
    Gaussians,
    Mesh,
    RigError,
    bind,
    deform,
    mapping_table,
    pose_to_flame,
    quat_from_matrix,
    quat_multiply,
    quat_to_matrix,
    triangle_frames,
)

# One frame at the rate the motion system runs at. The per-frame budget the
# deform path is held to.
FRAME_S = 1.0 / 25.0


# ---------------------------------------------------------------- fixtures --


def rotation(axis, angle: float) -> np.ndarray:
    """Rodrigues, so the tests need no dependency numpy does not already have."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def moved(mesh: Mesh, matrix=None, translation=(0.0, 0.0, 0.0), scale: float = 1.0) -> Mesh:
    """The same mesh under a similarity transform. Same topology, by construction."""
    matrix = np.eye(3) if matrix is None else matrix
    vertices = (mesh.vertices.astype(np.float64) @ matrix.T) * scale + np.asarray(translation)
    return Mesh(vertices, mesh.faces)


def one_triangle() -> Mesh:
    return Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0, 1, 2]]),
    )


def grid_mesh(n: int = 9, size: float = 1.0) -> Mesh:
    """A flat n-by-n grid. Closed enough for binding, trivial to reason about."""
    axis = np.linspace(-size, size, n)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack([xx.ravel(), yy.ravel(), np.zeros(n * n)], axis=-1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces.append([a, a + 1, a + n + 1])
            faces.append([a, a + n + 1, a + n])
    return Mesh(vertices, np.array(faces))


def sphere_mesh(rings: int = 24, radius: float = 0.12) -> Mesh:
    """A closed-ish surface with triangles facing every direction.

    Used wherever a test needs the frame to span the whole rotation group -
    including the half-turns where a careless quaternion extraction silently
    returns the wrong rotation.

    The poles are trimmed off so that no triangle is degenerate. Degenerate
    triangles are a real thing on a face mesh, but they are their own test
    rather than a confound in every other one.
    """
    u = np.linspace(0.05 * np.pi, 0.95 * np.pi, rings)
    v = np.linspace(0.0, 2.0 * np.pi, rings)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    vertices = (
        np.stack([np.sin(uu) * np.cos(vv), np.sin(uu) * np.sin(vv), np.cos(uu)], axis=-1).reshape(
            -1, 3
        )
        * radius
    )
    faces = []
    for i in range(rings - 1):
        for j in range(rings - 1):
            a = i * rings + j
            faces.append([a, a + 1, a + rings + 1])
            faces.append([a, a + rings + 1, a + rings])
    return Mesh(vertices, np.array(faces))


def cloud(positions, seed: int = 0) -> Gaussians:
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    rotations = rng.normal(size=(len(positions), 4))
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    scales = rng.uniform(0.01, 0.05, size=(len(positions), 3))
    return Gaussians(positions, rotations, scales)


def near_surface(mesh: Mesh, count: int, seed: int = 0, standoff: float = 0.01) -> Gaussians:
    """A cloud sitting just off the mesh, the way skin Gaussians do."""
    rng = np.random.default_rng(seed)
    v0, v1, v2 = mesh.corners()
    picked = rng.integers(0, mesh.triangle_count, size=count)
    weights = rng.dirichlet(np.ones(3), size=count)
    _, _, normal, _ = triangle_frames(v0[picked], v1[picked], v2[picked])
    surface = (
        weights[:, 0:1] * v0[picked] + weights[:, 1:2] * v1[picked] + weights[:, 2:3] * v2[picked]
    )
    offset = rng.uniform(-standoff, standoff, size=(count, 1))
    return cloud(surface + offset * normal, seed=seed + 1)


def rotations_of(transforms) -> np.ndarray:
    """Rotations as matrices. Quaternions carry a sign; rotations do not."""
    return quat_to_matrix(transforms.rotations)


# -------------------------------------------------- a Gaussian rides its triangle --


@pytest.mark.parametrize(
    "name, matrix, translation, scale",
    [
        ("still", None, (0.0, 0.0, 0.0), 1.0),
        ("translated", None, (3.0, -2.0, 0.5), 1.0),
        ("rotated", rotation((0, 0, 1), 0.9), (0.0, 0.0, 0.0), 1.0),
        ("tumbled", rotation((1, 2, 3), 2.4), (0.0, 0.0, 0.0), 1.0),
        ("scaled", None, (0.0, 0.0, 0.0), 2.5),
        ("everything", rotation((1, -1, 2), 1.3), (0.4, 0.1, -0.7), 0.6),
    ],
)
def test_a_gaussian_stays_on_its_triangle(name, matrix, translation, scale):
    """The whole premise: move the triangle, and the Gaussian goes with it.

    Asserted against the closed-form answer rather than against a golden file.
    Under a similarity transform every bound Gaussian must land at exactly
    s.R.p + t, turn by exactly R, and grow by exactly s - not approximately, and
    not "somewhere on the triangle". Anything looser would pass while the face
    slid across itself.
    """
    mesh = one_triangle()
    gaussians = cloud([[0.25, 0.25, 0.05], [0.5, 0.2, -0.03], [0.1, 0.6, 0.0]])
    bindings = bind(gaussians, mesh)
    assert bindings.static_count == 0, f"{name}: these sit on the triangle and must bind"

    posed = deform(bindings, moved(mesh, matrix, translation, scale))

    r = np.eye(3) if matrix is None else matrix
    expected = (gaussians.positions.astype(np.float64) @ r.T) * scale + np.asarray(translation)
    assert np.allclose(posed.positions, expected, atol=1e-5)
    assert np.allclose(posed.scales, gaussians.scales * scale, atol=1e-6)
    assert np.allclose(rotations_of(posed), r @ quat_to_matrix(gaussians.rotations), atol=1e-5)


def test_a_gaussian_keeps_its_standoff_when_the_triangle_grows():
    """The normal offset is stored in triangle-scale units, so it scales too.

    Stored in metres instead, a face that got bigger would swallow its own
    Gaussians: the skin layer would sink below the surface it is meant to sit
    on, and the splat would render inside-out in the places that stretched most.
    """
    mesh = one_triangle()
    gaussians = cloud([[0.3, 0.3, 0.1]])
    bindings = bind(gaussians, mesh)

    posed = deform(bindings, moved(mesh, scale=3.0))
    assert posed.positions[0, 2] == pytest.approx(0.3, abs=1e-5)


# ------------------------------------------------------------- the tearing bug --


def test_a_gaussian_far_from_the_mesh_is_static_and_does_not_move():
    """The tearing bug, asserted head on.

    A Gaussian ten units away from the mesh is hair, a collar, or a piece of
    the room. There is always a nearest triangle, and binding to it is what
    drags the wall across the cheek when the jaw opens. It must be STATIC, and
    STATIC must mean it does not move at all - not that it moves less.
    """
    mesh = grid_mesh()
    gaussians = cloud([[0.0, 0.0, 0.005], [0.0, 0.0, 10.0]], seed=7)
    bindings = bind(gaussians, mesh)

    assert bindings[0].mode is BindMode.BOUND
    assert bindings[1].mode is BindMode.STATIC
    assert bindings.static_count == 1

    # A violent pose: the mesh turns a quarter turn, moves and doubles in size.
    posed = deform(bindings, moved(mesh, rotation((1, 0, 0), np.pi / 2), (5.0, 5.0, 5.0), 2.0))

    assert np.allclose(posed.positions[1], gaussians.positions[1], atol=1e-6)
    assert np.allclose(posed.scales[1], gaussians.scales[1], atol=1e-6)
    assert np.allclose(posed.rotations[1], gaussians.rotations[1], atol=1e-6)

    # And the test must not be passing because nothing moved. The bound one did.
    assert np.linalg.norm(posed.positions[0] - gaussians.positions[0]) > 1.0


def test_a_gaussian_off_the_edge_of_the_mesh_is_static():
    """The second gate, which is the one that is easy to leave out.

    This Gaussian is well inside the bind radius - three units from a mesh whose
    radius is over four - but it is off the *edge*, not above the surface. Bind
    it and its barycentric coordinates become a three-to-one lever: every
    degree the triangle turns swings it three times as far. That is the same
    tear as the far case, arriving through a different door.
    """
    mesh = one_triangle()
    radius = BIND_RADIUS_IN_EDGES * mesh.median_edge_length()
    off_edge = np.array([[-3.0, -0.0, 0.0]])
    assert np.linalg.norm(off_edge) < radius, "the point must pass the distance gate"

    bindings = bind(cloud(off_edge), mesh)
    assert bindings[0].mode is BindMode.STATIC
    excursion = max(0.0, -min(bindings[0].barycentric))
    assert excursion > MAX_BARYCENTRIC_EXCURSION


def test_the_bind_radius_is_the_knob_that_decides():
    """Same geometry, two radii, two answers. The gate is a parameter, not a mood."""
    mesh = grid_mesh()
    gaussians = cloud([[0.0, 0.0, 0.4]])

    assert bind(gaussians, mesh, max_distance=0.1)[0].mode is BindMode.STATIC
    assert bind(gaussians, mesh, max_distance=1.0)[0].mode is BindMode.BOUND


def test_hair_and_collar_do_not_follow_a_face_that_moves():
    """The realistic shape of the failure, not just its unit case.

    A face-shaped cloud with a shell of debris standing off it. After a large
    head rotation the face must have moved and the debris must be exactly where
    it was - which is the visible, honest, documented behaviour, and the one
    thing that is not a grotesque.
    """
    mesh = sphere_mesh()
    skin = near_surface(mesh, 400, seed=3, standoff=0.004)
    debris = cloud(np.random.default_rng(11).normal(size=(80, 3)) * 0.4 + 0.6, seed=12)
    gaussians = Gaussians(
        np.vstack([skin.positions, debris.positions]),
        np.vstack([skin.rotations, debris.rotations]),
        np.vstack([skin.scales, debris.scales]),
    )
    bindings = bind(gaussians, mesh)

    assert bindings.mode[:400].max() == BindMode.BOUND, "skin must bind"
    assert bindings.mode[400:].min() == BindMode.STATIC, "debris must not"

    posed = deform(bindings, moved(mesh, rotation((0, 1, 0), 0.6)))
    assert np.abs(posed.positions[400:] - gaussians.positions[400:]).max() < 1e-6
    assert np.linalg.norm(posed.positions[:400] - gaussians.positions[:400], axis=1).mean() > 0.01


# ------------------------------------------------------- rigidity and round trip --


def test_rigid_mesh_motion_preserves_every_distance_between_gaussians():
    """The invariant a wrong rotation breaks without looking wrong.

    If the triangle frame is not genuinely rigid - built from a smoothed normal,
    say, or normalised in the wrong order - individual Gaussians still land
    somewhere plausible and the cloud quietly shears. Pairwise distance is the
    measurement that catches it: under a rigid motion of the mesh, every
    distance between two Gaussians must be exactly what it was.
    """
    mesh = sphere_mesh()
    gaussians = near_surface(mesh, 300, seed=5, standoff=0.004)
    bindings = bind(gaussians, mesh)
    assert bindings.static_count == 0

    rest = deform(bindings, mesh)
    matrix = rotation((0.3, -1.0, 0.7), 1.9)
    posed = deform(bindings, moved(mesh, matrix, (0.7, -0.2, 1.5)))

    before = np.linalg.norm(rest.positions[:, None, :] - rest.positions[None, :, :], axis=-1)
    after = np.linalg.norm(posed.positions[:, None, :] - posed.positions[None, :, :], axis=-1)
    assert np.abs(before - after).max() < 1e-5
    assert before.max() > 0.1, "a cloud with no extent would pass this vacuously"

    # And the rotation must have been applied, not merely have preserved lengths.
    expected = gaussians.positions.astype(np.float64) @ matrix.T + np.array([0.7, -0.2, 1.5])
    assert np.allclose(posed.positions, expected, atol=1e-5)
    assert np.allclose(rotations_of(posed), matrix @ quat_to_matrix(gaussians.rotations), atol=1e-4)
    assert np.allclose(posed.scales, gaussians.scales, atol=1e-6)


def test_binding_then_deforming_an_unposed_mesh_is_the_identity():
    """The round trip. If neutral is not neutral, nothing downstream can be.

    A rig that shifts the splat by half a millimetre at rest is a rig that has
    changed the likeness before the first frame of animation, and no amount of
    correct motion afterwards puts the face back.
    """
    mesh = sphere_mesh()
    gaussians = near_surface(mesh, 500, seed=9, standoff=0.006)
    bindings = bind(gaussians, mesh)
    posed = deform(bindings, mesh)

    assert np.abs(posed.positions - gaussians.positions).max() < 1e-6
    assert np.abs(posed.scales - gaussians.scales).max() < 1e-6
    assert np.abs(rotations_of(posed) - quat_to_matrix(gaussians.rotations)).max() < 1e-5


def test_round_trip_holds_for_static_gaussians_too():
    mesh = grid_mesh()
    gaussians = cloud(np.random.default_rng(4).normal(size=(200, 3)) * 2.0, seed=4)
    bindings = bind(gaussians, mesh)
    assert 0 < bindings.static_count < len(gaussians), "this test needs both kinds"

    posed = deform(bindings, mesh)
    assert np.abs(posed.positions - gaussians.positions).max() < 1e-5


def test_deformation_never_produces_a_gaussian_at_infinity():
    """A NaN position renders as a full-screen smear, not as an error.

    Degenerate triangles are real - a face mesh has collapsed faces around the
    eye contours - so the arithmetic has to survive one rather than propagate it.
    """
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]
    )
    mesh = Mesh(vertices, np.array([[0, 1, 2], [3, 4, 3]]))
    gaussians = cloud([[0.4, 0.4, 0.02], [0.5, 0.5, 0.001]])
    bindings = bind(gaussians, mesh)

    assert (bindings.triangle != 1).all(), "a collapsed triangle must never win the search"
    posed = deform(bindings, moved(mesh, rotation((1, 1, 0), 0.8), (1.0, 0.0, 0.0), 1.7))
    assert np.isfinite(posed.positions).all()
    assert np.isfinite(posed.rotations).all()
    assert np.isfinite(posed.scales).all()


# ------------------------------------------------------------------- the budget --


def test_deforming_a_hundred_thousand_gaussians_fits_inside_a_frame():
    """The one function here with a clock on it.

    The number that matters is not the margin against the threshold below - it
    is the three orders of magnitude between this and a Python loop over
    Gaussians. A regression that de-vectorises `deform` does not make it
    twenty percent slower; it makes it take a minute. The threshold is half a
    frame so the test still says something on a machine slower than the one it
    was written on, and the fastest of several runs is used because a shared
    core's median measures the machine rather than the code.
    """
    mesh = grid_mesh(21, size=0.15)
    rng = np.random.default_rng(21)
    count = 100_000
    positions = np.column_stack(
        [
            rng.uniform(-0.14, 0.14, count),
            rng.uniform(-0.14, 0.14, count),
            rng.uniform(-0.004, 0.004, count),
        ]
    )
    bindings = bind(cloud(positions, seed=22), mesh)
    assert bindings.bound_count > 99_000, "the timing must measure the bound path"

    posed = moved(mesh, rotation((0, 1, 0), 0.2), (0.01, 0.0, 0.0))
    deform(bindings, posed)  # warm up: the first touch pays for page faults

    best = min(_timed(deform, bindings, posed) for _ in range(5))
    assert best < 0.5 * FRAME_S, f"deform of {count} gaussians took {best * 1000:.1f}ms"


def _timed(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


# ------------------------------------------------------- shape of the artefact --


def test_bindings_survive_serialisation_unchanged():
    """Bindings are computed once and read forever, so they have to persist.

    Re-deriving them on load would mean a nearest-triangle search against a
    million Gaussians before the first frame of every call.
    """
    mesh = sphere_mesh()
    gaussians = near_surface(mesh, 250, seed=13)
    bindings = bind(gaussians, mesh)

    restored = Bindings.from_npz_bytes(bindings.to_npz_bytes())
    assert len(restored) == len(bindings)
    assert np.array_equal(restored.triangle, bindings.triangle)
    assert np.array_equal(restored.mode, bindings.mode)

    posed = moved(mesh, rotation((1, 0, 1), 0.5), (0.1, 0.2, 0.3))
    assert np.array_equal(deform(restored, posed).positions, deform(bindings, posed).positions)


def test_one_binding_reads_back_as_a_record():
    mesh = one_triangle()
    bindings = bind(cloud([[0.3, 0.3, 0.02]]), mesh)
    one = bindings[0]

    assert isinstance(one, GaussianBinding)
    assert one.triangle == 0
    assert sum(one.barycentric) == pytest.approx(1.0, abs=1e-5)
    assert GaussianBinding.from_dict(one.to_dict()) == one


def test_deforming_against_the_wrong_topology_is_an_error_not_a_face():
    """A wrong face is much harder to notice than a raised exception."""
    mesh = grid_mesh()
    bindings = bind(near_surface(mesh, 50, seed=2), mesh)
    with pytest.raises(RigError, match="topology"):
        deform(bindings, one_triangle())


def test_an_empty_splat_binds_and_deforms_to_nothing():
    mesh = one_triangle()
    bindings = bind(Gaussians(np.zeros((0, 3)), np.zeros((0, 4)), np.zeros((0, 3))), mesh)
    assert len(bindings) == 0
    assert len(deform(bindings, mesh)) == 0


@pytest.mark.parametrize(
    "vertices, faces, message",
    [
        (np.zeros((3, 2)), np.array([[0, 1, 2]]), "vertices"),
        (np.zeros((3, 3)), np.array([[0, 1]]), "faces"),
        (np.zeros((3, 3)), np.zeros((0, 3)), "no triangles"),
        (np.zeros((3, 3)), np.array([[0, 1, 9]]), "does not exist"),
    ],
)
def test_a_malformed_mesh_is_refused(vertices, faces, message):
    with pytest.raises(RigError, match=message):
        Mesh(vertices, faces)


def test_gaussian_arrays_must_agree_on_how_many_there_are():
    with pytest.raises(RigError, match="rotations"):
        Gaussians(np.zeros((4, 3)), np.zeros((3, 4)), np.zeros((4, 3)))


# ------------------------------------------------------------ quaternion helpers --


def test_quaternion_and_matrix_are_inverses_including_at_a_half_turn():
    """Half-turns are where the cheap quaternion extraction silently lies.

    They are also common on a face mesh, where the triangles on the two sides
    of a head are nearly each other's mirror, so the safe formula is not
    defensive programming - it is the case that actually occurs.
    """
    rng = np.random.default_rng(17)
    quaternions = rng.normal(size=(500, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    half_turns = np.array(
        [[0.0, 1, 0, 0], [0.0, 0, 1, 0], [0.0, 0, 0, 1], [1.0, 0, 0, 0]], dtype=np.float64
    )
    quaternions = np.vstack([quaternions, half_turns])

    matrices = quat_to_matrix(quaternions)
    assert np.abs(quat_to_matrix(quat_from_matrix(matrices)) - matrices).max() < 1e-5


def test_quaternion_multiplication_composes_rotations_in_the_right_order():
    a = quat_from_matrix(rotation((0, 0, 1), 0.7)[None])
    b = quat_from_matrix(rotation((1, 0, 0), 1.1)[None])
    composed = quat_to_matrix(quat_multiply(a, b))[0]
    assert np.allclose(composed, rotation((0, 0, 1), 0.7) @ rotation((1, 0, 0), 1.1), atol=1e-5)


def test_the_triangle_frame_is_orthonormal_and_scales_with_the_triangle():
    mesh = sphere_mesh(12)
    v0, v1, v2 = mesh.corners()
    tangent, bitangent, normal, scale = triangle_frames(v0, v1, v2)

    for axis in (tangent, bitangent, normal):
        assert np.abs(np.linalg.norm(axis, axis=1) - 1.0).max() < 1e-4
    assert np.abs((tangent * bitangent).sum(1)).max() < 1e-4
    assert np.abs((tangent * normal).sum(1)).max() < 1e-4
    assert np.abs(np.cross(tangent, bitangent) - normal).max() < 1e-4

    _, _, _, doubled = triangle_frames(v0 * 2.0, v1 * 2.0, v2 * 2.0)
    assert np.allclose(doubled, scale * 2.0, rtol=1e-4)


# ------------------------------------------------------- PoseFrame -> FLAME 2023 --


def test_every_motion_channel_is_accounted_for():
    """The totality test. Add a channel to pose.py and this fails.

    A mapping that silently drops a channel is worse than one that refuses it:
    the face animates, it looks nearly right, and the reason it is not right is
    invisible in every frame of output. So every one of pose.py's channels must
    appear in the table, and must be classified as exactly one of mapped,
    approximated, or impossible.
    """
    assert set(CHANNEL_MAP) == set(channel_names())
    assert len(CHANNEL_MAP) == len(CHANNELS)
    for name, mapping in CHANNEL_MAP.items():
        assert mapping.channel == name
        assert isinstance(mapping.fidelity, Fidelity)
        assert mapping.note.strip(), f"{name} is classified but not explained"
        if mapping.fidelity is Fidelity.UNMAPPABLE:
            assert "nothing in FLAME" in mapping.target
        else:
            assert mapping.target.strip()


def test_the_mapping_partitions_the_channels_with_nothing_left_over():
    """Mapped, approximated and impossible must cover the channels exactly once."""
    frame = PoseFrame(**{c.name: 0.1 for c in CHANNELS})
    params = pose_to_flame(frame)

    approximated = set(params.approximate)
    impossible = set(params.unmapped)
    exact = {n for n, m in CHANNEL_MAP.items() if m.fidelity is Fidelity.EXACT}

    assert exact | approximated | impossible == set(channel_names())
    assert not (exact & approximated)
    assert not (exact & impossible)
    assert not (approximated & impossible)


def test_channels_with_no_flame_equivalent_are_handed_back_rather_than_dropped():
    """FLAME is a head and a neck. The body is somebody else's problem, by value.

    Carried in the return value rather than logged, so that losing breath takes
    a deliberate act by whoever consumes this instead of happening by omission.
    """
    frame = PoseFrame(torso_lean=0.5, torso_yaw=-0.2, shoulder_raise=0.75, breath=0.4)
    params = pose_to_flame(frame)

    assert params.unmapped == {
        "torso_lean": pytest.approx(0.5),
        "torso_yaw": pytest.approx(-0.2),
        "shoulder_raise": pytest.approx(0.75),
        "breath": pytest.approx(0.4),
    }
    # And none of it leaked into the head.
    assert np.allclose(params.neck_pose, 0.0)
    assert np.allclose(params.global_pose, 0.0)


def test_head_and_gaze_are_carried_exactly():
    frame = PoseFrame(
        head_yaw=0.21, head_pitch=-0.13, head_roll=0.07, gaze_yaw=-0.3, gaze_pitch=0.11
    )
    params = pose_to_flame(frame)

    assert params.neck_pose == pytest.approx([-0.13, 0.21, 0.07], abs=1e-6)
    assert params.eye_pose == pytest.approx([0.11, -0.3, 0.0, 0.11, -0.3, 0.0], abs=1e-6)
    # Both eyes get the same rotation: the motion system produces no vergence.
    assert params.eye_pose[:3] == pytest.approx(params.eye_pose[3:])


def test_the_jaw_takes_the_larger_of_speech_and_expression_never_the_sum():
    """Adding them drives a real joint through the chin.

    jaw_open and the visemes describe the same hinge from two directions. A
    wide vowel arriving on a frame where the jaw was already open would, summed,
    ask FLAME for more rotation than a jaw has.
    """
    visemes = [0.0] * VISEME_COUNT
    visemes[VISEME_NAMES.index("aa")] = 1.0
    open_vowel = VISEME_MAP[VISEME_NAMES.index("aa")].jaw

    both = pose_to_flame(PoseFrame(jaw_open=0.5, visemes=tuple(visemes)))
    assert both.jaw_pose[0] == pytest.approx(open_vowel * FLAME_JAW_MAX_RAD, abs=1e-6)

    wider = pose_to_flame(PoseFrame(jaw_open=1.0, visemes=tuple(visemes)))
    assert wider.jaw_pose[0] == pytest.approx(FLAME_JAW_MAX_RAD, abs=1e-6)
    assert wider.jaw_pose[0] <= FLAME_JAW_MAX_RAD + 1e-6


def test_a_closed_mouth_leaves_the_jaw_alone():
    params = pose_to_flame(PoseFrame())
    assert np.allclose(params.jaw_pose, 0.0)
    assert np.allclose(params.expression, 0.0)


def test_the_expression_basis_is_symmetric_and_says_so():
    """One eyebrow is not representable, and the loss is halved rather than hidden.

    FLAME's expression space is global PCA with no notion of a side, so a
    single-sided raise becomes a symmetric one at half amplitude. Asserted
    because it is a real limitation that somebody will otherwise rediscover as
    a bug report about a face that cannot raise an eyebrow.
    """
    one_side = pose_to_flame(PoseFrame(brow_inner_l=1.0))
    both_sides = pose_to_flame(PoseFrame(brow_inner_l=1.0, brow_inner_r=1.0))
    mirrored = pose_to_flame(PoseFrame(brow_inner_r=1.0))

    assert np.allclose(one_side.expression, mirrored.expression)
    assert np.allclose(one_side.expression * 2.0, both_sides.expression)
    assert "brow_inner_l" in one_side.approximate


def test_a_placeholder_basis_never_claims_to_be_fitted():
    """The honesty flag. A meaningless expression must not read as a real one.

    The placeholder exists so this path is testable with no FLAME asset. It
    produces numbers of the right shape and no geometric meaning whatsoever,
    and the only thing stopping that being mistaken for a fitted rig is this
    flag travelling with the result.
    """
    assert not ExpressionBasis.placeholder().fitted
    assert not pose_to_flame(PoseFrame(blink=1.0)).basis_fitted

    fitted = ExpressionBasis(ExpressionBasis.placeholder().directions, fitted=True)
    assert pose_to_flame(PoseFrame(blink=1.0), fitted).basis_fitted


def test_a_basis_missing_a_direction_produces_stillness_not_a_crash():
    """A capture where they never pressed their lips has no lip-press direction.

    The honest result is a face that does not press its lips, not an exception
    on the frame that asked for it - the alternative is an avatar that crashes
    mid-sentence because of something that was never photographed.
    """
    sparse = ExpressionBasis({"smile": np.ones(FLAME_EXPRESSION_COUNT)}, fitted=True)
    params = pose_to_flame(PoseFrame(mouth_press=1.0, mouth_smile_l=1.0), sparse)
    assert np.allclose(params.expression, 0.5)


def test_the_expression_vector_is_the_size_flame_expects():
    params = pose_to_flame(PoseFrame(blink=1.0, mouth_smile_l=0.4))
    assert params.expression.shape == (FLAME_EXPRESSION_COUNT,)
    assert params.full_pose.shape == (15,)
    assert np.allclose(params.full_pose[:3], params.global_pose)
    assert np.allclose(params.full_pose[3:6], params.neck_pose)
    assert np.allclose(params.full_pose[6:9], params.jaw_pose)
    assert np.allclose(params.full_pose[9:], params.eye_pose)


def test_the_viseme_table_matches_the_motion_system():
    assert len(VISEME_MAP) == VISEME_COUNT
    assert tuple(m.name for m in VISEME_MAP) == VISEME_NAMES
    assert len(set(VISEME_NAMES)) == VISEME_COUNT
    for mapping in VISEME_MAP:
        assert 0.0 <= mapping.jaw <= 1.0
        assert isinstance(mapping.fidelity, Fidelity)
    # The one FLAME genuinely cannot show, said out loud rather than faked.
    assert VISEME_MAP[VISEME_NAMES.index("TH")].fidelity is Fidelity.UNMAPPABLE


def test_the_mapping_table_can_be_printed_for_review():
    """The most likely failure of this mapping is a wrong assumption about it."""
    rows = mapping_table()
    assert len(rows) == len(CHANNELS) + VISEME_COUNT
    assert {row[2] for row in rows} <= {f.value for f in Fidelity}
    assert all(row[0] and row[1] for row in rows)


def test_a_clamped_frame_maps_without_leaving_flame_ranges():
    """Whatever the director does, the jaw stays inside a jaw."""
    extreme = PoseFrame(**{c.name: c.high for c in CHANNELS}, visemes=(1.0,) * VISEME_COUNT)
    params = pose_to_flame(extreme.clamped())
    assert 0.0 <= params.jaw_pose[0] <= FLAME_JAW_MAX_RAD + 1e-6
    assert np.isfinite(params.expression).all()


# ------------------------------------------------------------------ disclosure --


def test_the_limitations_are_stated_rather_than_discovered():
    """The same posture as QualityReport.disclosure: say it before they see it."""
    assert len(LIMITATIONS) >= 5
    joined = " ".join(LIMITATIONS).lower()
    for subject in ("hair", "static", "tongue", "breath", "camera"):
        assert subject in joined, f"the limitations do not mention {subject}"
    assert all(len(text) > 80 and text.endswith(".") for text in LIMITATIONS)


def test_the_viseme_map_covers_exactly_the_shapes_the_motion_system_produces():
    """One definition of the mouth shapes, not two that happen to agree.

    These names lived in both pose.py and this module until the duplicate was
    removed. The copies matched, which is luck rather than correctness: a
    permuted viseme set makes the wrong mouth shape for every sound and is
    subtle enough to survive a demo. If a shape is ever added, renamed or
    reordered, this fails rather than drifting.
    """
    from avatar.motion.pose import VISEME_COUNT, VISEME_NAMES
    from avatar.splat.rig import VISEME_MAP

    assert len(VISEME_MAP) == VISEME_COUNT
    assert tuple(m.name for m in VISEME_MAP) == VISEME_NAMES


def test_silence_closes_the_mouth_and_the_open_vowel_opens_it():
    """The two shapes whose meaning is unambiguous, as an ordering sanity check.

    If the table were reversed or rotated, these two are where it shows.
    """
    from avatar.splat.rig import VISEME_MAP

    by_name = {m.name: m for m in VISEME_MAP}

    assert by_name["sil"].jaw == 0.0
    assert by_name["aa"].jaw > by_name["ih"].jaw
    assert by_name["PP"].jaw < by_name["aa"].jaw
