"""Making a photoreal splat animatable: binding every Gaussian to a face.

A Gaussian splat of a person is the best likeness this pipeline can produce and
it is completely inert. It is a few hundred thousand coloured, oriented,
anisotropic blobs floating in space with no notion that some of them are a lip
and some of them are a cheek. Nothing in the artefact knows that opening a jaw
should move one and not the other.

This module supplies the missing knowledge, and it does it the way the
literature settled on: **bind each Gaussian to the nearest triangle of a
parametric face mesh, and let it ride.** Store, once, at build time, where the
Gaussian sits relative to that triangle - barycentric coordinates in the
triangle's plane, a standoff along its normal, and its rotation and scale
expressed in the triangle's own frame. Then on every frame, pose the mesh from
the motion system, rebuild each triangle's frame, and put the Gaussian back
where its stored coordinates say it belongs. The mesh is small and cheap to
pose; the splat is large and never re-optimised. That asymmetry is the whole
trick.

**Provenance, because this project has been bitten by licences before.**
GaussianAvatars (Toyota Motor Europe) and SplattingAvatar both demonstrate this
technique and both are CC-BY-NC-SA: *no code from either is used here, and
neither was consulted beyond its published method description*. The rigging
below is written from the geometry - barycentric interpolation, a Gram-matrix
plane projection, a Gram-Schmidt triangle frame, Ericson's point-triangle
closest-point regions - all of which predate both papers by decades. The mesh
is FLAME 2023, which is CC-BY and may ship. FLAME 2017/2019/2020 and the FLAME
*texture* model may not, and nothing here depends on them.

**The failure this module exists to prevent is tearing.** A splat of a real
person contains a great deal that is not face: hair standing off the scalp, a
shirt collar, spectacle frames, background fragments the optimiser never
cleaned up. Bind those to "the nearest triangle" and every one of them acquires
a lever arm to the jaw. The result is the characteristic splat-avatar failure -
a face that talks and drags its hair, its collar and a piece of the wall along
with it. So binding is gated twice, and anything that fails either gate is
marked STATIC and simply does not move. A stiff avatar is a defect. A tearing
one is unusable, and looks like a desecration of the person it depicts.

Nothing here needs a GPU, a FLAME asset, or a trained model. Binding is
arithmetic on a mesh and a point cloud; deformation is a fixed sequence of
array operations. Both are tested on synthetic meshes, which is what lets the
rig be finished and defended before a single GPU-minute is spent.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

import numpy as np

from avatar.motion.pose import (
    VISEME_COUNT,
    VISEME_NAMES,
    PoseFrame,
    channel_names,
)

# Everything is stored and computed in float32. A head is about 0.2m across, so
# float32 resolves it to roughly 10 nanometres - four orders of magnitude finer
# than anything a camera contributed. Doubling the width would double the
# memory traffic of the one function in here that runs every frame.
DTYPE = np.float32

_TINY = 1e-12

# How far a Gaussian may sit from the mesh and still be bound to it, expressed
# in median edge lengths of that mesh rather than in metres, so the same number
# is right for a FLAME head (~5mm edges) and for a synthetic test cube.
#
# Four edges is roughly 2cm on FLAME. Skin, lips, eyelids and the near side of
# an eyebrow are inside it; hair volume, a collar and anything the splat
# optimiser left floating are outside. This is the first tearing gate and it is
# deliberately conservative: a Gaussian wrongly left STATIC costs a little
# stiffness in one place, and a Gaussian wrongly bound costs the whole face.
BIND_RADIUS_IN_EDGES = 4.0

# The second tearing gate, and the less obvious one. Barycentric coordinates
# are stored unclamped - from the perpendicular projection onto the triangle's
# plane - so that a Gaussian returns exactly where it started when the mesh is
# unposed. That is exact and well behaved while the projection lands inside the
# triangle. It becomes a lever the moment it does not: coordinates of (-3, 2, 2)
# describe a point three triangle-widths off the edge, and every rotation of
# that triangle multiplies by three.
#
# A projection lands outside its own nearest triangle mainly at a hole in the
# mesh (FLAME is open at the neck and inside the mouth) or in a sharp crease.
# Those are precisely the places where a bound Gaussian tears. One triangle
# width of tolerance covers honest numerical slop at a shared edge; past it,
# STATIC.
MAX_BARYCENTRIC_EXCURSION = 1.0

# Triangles considered per Gaussian during binding. The exact nearest triangle
# is found by evaluating the true point-triangle distance against the K
# triangles with the nearest centroids; on a mesh as uniform as FLAME, sixteen
# is far more than enough for the true nearest to be among them. When a mesh
# has K triangles or fewer - every mesh in the test suite - all of them are
# evaluated and the result is exact by construction.
CANDIDATE_TRIANGLES = 16

# Peak size of the intermediate distance matrix during binding, in float32
# elements. Binding is a build-time one-off, so this trades a little speed for
# not allocating a gigabyte on a mesh with ten thousand faces.
_CHUNK_ELEMENTS = 8_000_000


class RigError(ValueError):
    """The rig was handed something it cannot bind or deform."""


class BindMode(IntEnum):
    """Whether a Gaussian follows the mesh or is nailed down.

    Stored as an integer rather than a string because there is one per
    Gaussian and there may be two million of them.
    """

    BOUND = 0
    STATIC = 1


@dataclass(frozen=True)
class Mesh:
    """A triangle mesh, in the one form the rig needs it.

    Deliberately not a FLAME type. The rig knows about vertices and faces; what
    produced them - FLAME 2023, a scan, a unit test's single triangle - is not
    its business, and keeping it that way is what lets the whole of this module
    be tested with no licensed asset in the repository.
    """

    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3) integer indices

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices", np.ascontiguousarray(self.vertices, dtype=DTYPE))
        object.__setattr__(self, "faces", np.ascontiguousarray(self.faces, dtype=np.int32))
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise RigError(f"vertices must be (V, 3); got {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise RigError(f"faces must be (F, 3); got {self.faces.shape}")
        if len(self.faces) == 0:
            raise RigError("a mesh with no triangles cannot carry a binding")
        if self.faces.min() < 0 or self.faces.max() >= len(self.vertices):
            raise RigError("a face indexes a vertex that does not exist")

    @property
    def triangle_count(self) -> int:
        return len(self.faces)

    def corners(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The three corners of every triangle, each (F, 3)."""
        v = self.vertices[self.faces]
        return v[:, 0], v[:, 1], v[:, 2]

    def centroids(self) -> np.ndarray:
        return self.vertices[self.faces].mean(axis=1)

    def areas(self) -> np.ndarray:
        a, b, c = self.corners()
        return 0.5 * _norm(np.cross(b - a, c - a))

    def median_edge_length(self) -> float:
        """The length scale of this mesh, robust to a few degenerate faces.

        Median rather than mean because a mesh with a handful of collapsed
        triangles - which FLAME has around the eye contours - would otherwise
        report a shorter scale than it has, and tighten the bind radius on the
        whole face because of six triangles nobody sees.
        """
        a, b, c = self.corners()
        lengths = np.concatenate([_norm(b - a), _norm(c - b), _norm(a - c)])
        return float(np.median(lengths))

    def same_topology_as(self, other: Mesh) -> bool:
        return self.faces.shape == other.faces.shape and bool(np.array_equal(self.faces, other.faces))


@dataclass(frozen=True)
class Gaussians:
    """The subset of a splat that rigging touches.

    Colour, opacity and spherical harmonics are absent on purpose. This module
    moves Gaussians; it never changes what they look like, and a type that
    could not carry appearance is the cheapest way to guarantee that. It is
    also a statement of a real limitation - see LIMITATIONS - because appearance
    that should change with expression is exactly what a rigged splat cannot do.

    Rotations are unit quaternions in (w, x, y, z) order. Scales are the three
    axis lengths of the Gaussian in its own local frame.
    """

    positions: np.ndarray  # (N, 3)
    rotations: np.ndarray  # (N, 4) wxyz
    scales: np.ndarray  # (N, 3)

    def __post_init__(self) -> None:
        pos = np.ascontiguousarray(self.positions, dtype=DTYPE)
        rot = np.ascontiguousarray(self.rotations, dtype=DTYPE)
        scl = np.ascontiguousarray(self.scales, dtype=DTYPE)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise RigError(f"positions must be (N, 3); got {pos.shape}")
        if rot.shape != (len(pos), 4):
            raise RigError(f"rotations must be (N, 4) wxyz; got {rot.shape}")
        if scl.shape != (len(pos), 3):
            raise RigError(f"scales must be (N, 3); got {scl.shape}")
        object.__setattr__(self, "positions", pos)
        object.__setattr__(self, "rotations", _normalise_quat(rot))
        object.__setattr__(self, "scales", scl)

    def __len__(self) -> int:
        return len(self.positions)


@dataclass(frozen=True)
class Transforms:
    """Where every Gaussian is, this frame. What a renderer is handed.

    Same three arrays as `Gaussians` and a different meaning: those are the
    artefact as it was built, these are the artefact as it is posed. Kept as a
    separate type so that nothing can accidentally write a posed frame back
    over the rest pose it was derived from.
    """

    positions: np.ndarray  # (N, 3)
    rotations: np.ndarray  # (N, 4) wxyz
    scales: np.ndarray  # (N, 3)

    def __len__(self) -> int:
        return len(self.positions)

    def as_gaussians(self) -> Gaussians:
        return Gaussians(self.positions, self.rotations, self.scales)


@dataclass(frozen=True)
class GaussianBinding:
    """One Gaussian's attachment to one triangle.

    Computed once, at build time, and read on every frame for the rest of the
    avatar's life - which is why it is a plain record of numbers and why the
    set of them serialises. Re-deriving these on load would mean running the
    nearest-triangle search against a million Gaussians before the first frame.

    `barycentric` are the coordinates of the Gaussian's perpendicular
    projection onto the triangle's plane and are *not* clamped into the
    triangle; they sum to one and may run slightly negative for a Gaussian
    beside an edge. That is what makes the round trip exact.

    `normal_offset` is the standoff along the triangle normal measured in units
    of the triangle's own scale, not in metres. Stored that way so a triangle
    that grows carries its Gaussians outwards with it instead of letting them
    sink into the surface.

    `local_rotation` and `local_scale` are the Gaussian's orientation and size
    expressed in the triangle's frame, so both follow the triangle rather than
    being reapplied in world space.
    """

    triangle: int
    barycentric: tuple[float, float, float]
    normal_offset: float
    local_rotation: tuple[float, float, float, float]  # wxyz
    local_scale: tuple[float, float, float]
    mode: BindMode = BindMode.BOUND

    def to_dict(self) -> dict:
        return {
            "triangle": int(self.triangle),
            "barycentric": [float(x) for x in self.barycentric],
            "normal_offset": float(self.normal_offset),
            "local_rotation": [float(x) for x in self.local_rotation],
            "local_scale": [float(x) for x in self.local_scale],
            "mode": int(self.mode),
        }

    @classmethod
    def from_dict(cls, data: dict) -> GaussianBinding:
        return cls(
            triangle=int(data["triangle"]),
            barycentric=tuple(float(x) for x in data["barycentric"]),  # type: ignore[arg-type]
            normal_offset=float(data["normal_offset"]),
            local_rotation=tuple(float(x) for x in data["local_rotation"]),  # type: ignore[arg-type]
            local_scale=tuple(float(x) for x in data["local_scale"]),  # type: ignore[arg-type]
            mode=BindMode(int(data["mode"])),
        )


@dataclass(frozen=True)
class Bindings:
    """Every Gaussian's attachment, held as arrays rather than as objects.

    Struct-of-arrays, not array-of-structs, and that is the load-bearing
    decision in this file. `deform` runs at least twenty-five times a second
    against up to two million Gaussians; a Python-level loop over per-Gaussian
    objects is three orders of magnitude too slow and no amount of care
    elsewhere recovers it. `GaussianBinding` exists as a readable view onto one
    row, for inspection and for tests, and is never touched on the hot path.

    Rest values are carried only for the STATIC Gaussians. Storing them for
    every Gaussian would make the rig artefact larger than the splat it rigs.
    """

    triangle: np.ndarray  # (N,) int32 - valid index even for STATIC rows
    barycentric: np.ndarray  # (N, 3) float32
    normal_offset: np.ndarray  # (N,) float32, in triangle-scale units
    local_rotation: np.ndarray  # (N, 4) float32 wxyz
    local_scale: np.ndarray  # (N, 3) float32
    mode: np.ndarray  # (N,) uint8, BindMode

    static_index: np.ndarray  # (S,) int32 - which Gaussians never move
    static_position: np.ndarray  # (S, 3)
    static_rotation: np.ndarray  # (S, 4)
    static_scale: np.ndarray  # (S, 3)

    # How far each Gaussian sat from the mesh when it was bound, in metres.
    # Kept because it is the single most useful diagnostic when an avatar
    # looks stiff or looks torn, and because it costs four bytes.
    distance: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=DTYPE))

    def __len__(self) -> int:
        return len(self.triangle)

    @property
    def static_mask(self) -> np.ndarray:
        return self.mode == BindMode.STATIC

    @property
    def static_count(self) -> int:
        return int(self.static_mask.sum())

    @property
    def bound_count(self) -> int:
        return len(self) - self.static_count

    @property
    def static_fraction(self) -> float:
        return self.static_count / len(self) if len(self) else 0.0

    def __getitem__(self, i: int) -> GaussianBinding:
        """One row, as a readable record. Never used on the hot path."""
        return GaussianBinding(
            triangle=int(self.triangle[i]),
            barycentric=tuple(float(x) for x in self.barycentric[i]),  # type: ignore[arg-type]
            normal_offset=float(self.normal_offset[i]),
            local_rotation=tuple(float(x) for x in self.local_rotation[i]),  # type: ignore[arg-type]
            local_scale=tuple(float(x) for x in self.local_scale[i]),  # type: ignore[arg-type]
            mode=BindMode(int(self.mode[i])),
        )

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "triangle": self.triangle,
            "barycentric": self.barycentric,
            "normal_offset": self.normal_offset,
            "local_rotation": self.local_rotation,
            "local_scale": self.local_scale,
            "mode": self.mode,
            "static_index": self.static_index,
            "static_position": self.static_position,
            "static_rotation": self.static_rotation,
            "static_scale": self.static_scale,
            "distance": self.distance,
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> Bindings:
        return cls(**{name: np.asarray(arrays[name]) for name in cls.__dataclass_fields__})

    def to_npz_bytes(self) -> bytes:
        """The whole rig as one blob, for the object store beside the .splat.

        Bytes rather than a path because every other artefact in this system
        travels through BlobStore by key, and a rig that needed a filesystem
        would be the one thing in the pipeline that could not.
        """
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **self.to_arrays())
        return buffer.getvalue()

    @classmethod
    def from_npz_bytes(cls, blob: bytes) -> Bindings:
        with np.load(io.BytesIO(blob)) as data:
            return cls.from_arrays({name: data[name] for name in cls.__dataclass_fields__})


# --------------------------------------------------------------------------
# Small vectorised geometry. All of it operates on trailing-axis 3-vectors so
# the same functions serve the (N,) hot path and the (N, K) candidate search.
# --------------------------------------------------------------------------


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("...i,...i->...", a, b)


def _norm(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(_dot(a, a), 0.0))


def _safe(a: np.ndarray) -> np.ndarray:
    """Denominators, with zero replaced rather than divided by.

    Degenerate triangles are real - FLAME has collapsed faces around the eye
    contours - and a NaN position propagates through a renderer as a Gaussian
    at infinity, which is visible as a full-screen smear rather than as an
    error.
    """
    return np.where(np.abs(a) < _TINY, _TINY, a)


def _normalise_quat(q: np.ndarray) -> np.ndarray:
    return (q / _safe(_norm(q))[..., None]).astype(DTYPE, copy=False)


def quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product, (w, x, y, z), applied over the trailing axis."""
    w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w2, x2, y2, z2 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    ).astype(DTYPE, copy=False)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3). Columns are the rotated basis vectors."""
    q = _normalise_quat(np.asarray(q, dtype=DTYPE))
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    ).astype(DTYPE, copy=False)


def quat_from_basis(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """The quaternion of the orthonormal frame whose columns are x, y and z.

    Shepperd's method, arranged so that numpy does the work in whole-array
    operations rather than branching per element. Each of the four quaternion
    components can be recovered from the matrix diagonal up to sign; the one
    that is largest in magnitude is the numerically safe anchor, and the other
    three come from the off-diagonal sums and differences divided by it. All
    four magnitudes are computed - four square roots over the array - and the
    anchor is chosen with `where`, which costs a few percent of what evaluating
    four complete candidate quaternions costs and about a thousandth of what a
    Python loop would.

    Anchoring on the largest is what makes this correct at a half-turn, where
    the naive "recover the signs from m21-m12" formula silently returns the
    wrong rotation because those differences vanish. Half-turns are common on a
    face mesh - the triangles on the two sides of a head are close to being
    each other's mirror - so this is not a theoretical case.

    The sign of the result is arbitrary: q and -q are the same rotation, so
    callers must compare rotations as matrices or by their action on a vector,
    never component by component.
    """
    m00, m10, m20 = x[..., 0], x[..., 1], x[..., 2]
    m01, m11, m21 = y[..., 0], y[..., 1], y[..., 2]
    m02, m12, m22 = z[..., 0], z[..., 1], z[..., 2]

    # Twice the magnitude of each component, from the diagonal.
    mag_w = np.sqrt(np.maximum(1.0 + m00 + m11 + m22, 0.0))
    mag_x = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0.0))
    mag_y = np.sqrt(np.maximum(1.0 - m00 + m11 - m22, 0.0))
    mag_z = np.sqrt(np.maximum(1.0 - m00 - m11 + m22, 0.0))

    # The off-diagonals, which are four times the pairwise products.
    wx, wy, wz = m21 - m12, m02 - m20, m10 - m01
    xy, xz, yz = m01 + m10, m02 + m20, m12 + m21

    anchor_w = (mag_w >= mag_x) & (mag_w >= mag_y) & (mag_w >= mag_z)
    anchor_x = ~anchor_w & (mag_x >= mag_y) & (mag_x >= mag_z)
    anchor_y = ~anchor_w & ~anchor_x & (mag_y >= mag_z)
    largest = np.where(anchor_w, mag_w, np.where(anchor_x, mag_x, np.where(anchor_y, mag_y, mag_z)))
    inv = 0.5 / np.maximum(largest, _TINY)

    w = np.where(anchor_w, 0.5 * mag_w, np.where(anchor_x, wx * inv,
                 np.where(anchor_y, wy * inv, wz * inv)))
    qx = np.where(anchor_w, wx * inv, np.where(anchor_x, 0.5 * mag_x,
                  np.where(anchor_y, xy * inv, xz * inv)))
    qy = np.where(anchor_w, wy * inv, np.where(anchor_x, xy * inv,
                  np.where(anchor_y, 0.5 * mag_y, yz * inv)))
    qz = np.where(anchor_w, wz * inv, np.where(anchor_x, xz * inv,
                  np.where(anchor_y, yz * inv, 0.5 * mag_z)))
    return _normalise_quat(np.stack([w, qx, qy, qz], axis=-1))


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) wxyz. The columns are the frame."""
    m = np.asarray(m, dtype=DTYPE)
    return quat_from_basis(m[..., 0], m[..., 1], m[..., 2])


def triangle_frames(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each triangle's orthonormal frame - tangent, bitangent, normal - and its scale.

    Returned as three separate axis arrays rather than as a stacked (..., 3, 3)
    matrix, because everything downstream wants the axes and a stacked matrix
    would only be sliced apart again along its slowest stride. On the per-frame
    path that restriding costs more than the frame construction itself.

    The frame is Gram-Schmidt on the first edge and the face normal: x along
    v0->v1, z along the normal, y completing a right-handed set. It is defined
    by the triangle's corners alone, so it rotates exactly as the triangle
    rotates and is unchanged by moving the mesh rigidly - which is what makes a
    rigid mesh motion move every bound Gaussian rigidly, with the distances
    between them preserved. A frame built any other way - from a vertex normal,
    say, or from a smoothed normal field - quietly breaks that property, and
    the break shows up as a face that shears rather than turns.

    The scale is the square root of twice the area: a length, one that scales
    linearly under uniform scaling of the mesh, and one that depends on no
    particular edge. Isotropic on purpose - a Gaussian stretched by the
    anisotropy of its own triangle looks like a smear, and triangle anisotropy
    on a face mesh is an artefact of tessellation rather than of anatomy.
    """
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    twice_area = _norm(cross)
    normal = cross / _safe(twice_area)[..., None]
    tangent = e1 / _safe(_norm(e1))[..., None]
    bitangent = np.cross(normal, tangent)
    scale = np.sqrt(np.maximum(twice_area, _TINY))
    return (
        tangent.astype(DTYPE, copy=False),
        bitangent.astype(DTYPE, copy=False),
        normal.astype(DTYPE, copy=False),
        scale.astype(DTYPE, copy=False),
    )


def _closest_barycentric(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closest point on each triangle to each point: barycentrics and sq. distance.

    Ericson's seven-region decomposition (three vertices, three edges, the
    interior), evaluated for every region and selected rather than branched.
    Used only to *choose* a triangle; the coordinates actually stored come from
    the unclamped plane projection afterwards.
    """
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    inv = 1.0 / _safe(va + vb + vc)
    v_in, w_in = vb * inv, vc * inv
    u_in = 1.0 - v_in - w_in

    t_ab = np.clip(d1 / _safe(d1 - d3), 0.0, 1.0)
    t_ac = np.clip(d2 / _safe(d2 - d6), 0.0, 1.0)
    t_bc = np.clip((d4 - d3) / _safe((d4 - d3) + (d5 - d6)), 0.0, 1.0)

    zero, one = np.zeros_like(d1), np.ones_like(d1)
    conditions = [
        (d1 <= 0) & (d2 <= 0),  # vertex A
        (d3 >= 0) & (d4 <= d3),  # vertex B
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),  # edge AB
        (d6 >= 0) & (d5 <= d6),  # vertex C
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),  # edge AC
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),  # edge BC
    ]
    u = np.select(conditions, [one, zero, 1.0 - t_ab, zero, 1.0 - t_ac, zero], default=u_in)
    v = np.select(conditions, [zero, one, t_ab, zero, zero, 1.0 - t_bc], default=v_in)
    w = np.select(conditions, [zero, zero, zero, one, t_ac, t_bc], default=w_in)

    bary = np.stack([u, v, w], axis=-1)
    closest = bary[..., 0:1] * a + bary[..., 1:2] * b + bary[..., 2:3] * c
    delta = p - closest
    return bary.astype(DTYPE, copy=False), _dot(delta, delta)


def _nearest_triangles(points: np.ndarray, mesh: Mesh, candidates: int) -> np.ndarray:
    """Index of the triangle each point is genuinely closest to.

    Two stages: shortlist by centroid distance with a BLAS matrix product, then
    evaluate the true point-triangle distance against the shortlist only. When
    the shortlist is the whole mesh - any mesh with `candidates` faces or fewer,
    which is every mesh in the test suite - the answer is exact.

    On a bigger mesh it is exact in practice and approximate in principle: a
    triangle whose centroid is far while its nearest edge is close could in
    theory be missed. On a face mesh, whose triangles are small and about the
    same size, sixteen candidates is a wide margin. On a mesh with a few huge
    triangles it would not be, and that is a real caveat rather than a
    theoretical one - `candidates` is a parameter for that reason.
    """
    corners = mesh.vertices[mesh.faces]  # (F, 3, 3)
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    degenerate = mesh.areas() <= _TINY

    faces = mesh.triangle_count
    k = min(candidates, faces)
    centroids = mesh.centroids()
    centroid_sq = _dot(centroids, centroids)

    chunk = max(1, min(len(points), _CHUNK_ELEMENTS // max(faces, 1)))
    best = np.zeros(len(points), dtype=np.int32)

    for start in range(0, len(points), chunk):
        block = points[start : start + chunk]
        if k == faces:
            shortlist = np.broadcast_to(np.arange(faces, dtype=np.int32), (len(block), faces))
        else:
            # |p - c|^2 without materialising the difference: the cross term is
            # one matrix product, which is where the speed comes from.
            dist2 = centroid_sq[None, :] - 2.0 * (block @ centroids.T)
            shortlist = np.argpartition(dist2, k - 1, axis=1)[:, :k].astype(np.int32)

        _, sq = _closest_barycentric(
            block[:, None, :], a[shortlist], b[shortlist], c[shortlist]
        )
        # A collapsed triangle has no well-defined frame, so it may not win.
        sq = np.where(degenerate[shortlist], np.inf, sq)
        best[start : start + chunk] = shortlist[np.arange(len(block)), np.argmin(sq, axis=1)]

    return best


def bind(
    gaussians: Gaussians,
    mesh: Mesh,
    *,
    max_distance: float | None = None,
    candidates: int = CANDIDATE_TRIANGLES,
    max_excursion: float = MAX_BARYCENTRIC_EXCURSION,
) -> Bindings:
    """Attach every Gaussian to the mesh it should follow, or nail it down.

    `mesh` is the rest mesh: the neutral-expression FLAME fit that the splat
    was optimised alongside. Bind against a posed mesh and every Gaussian
    inherits that pose as its own idea of neutral.

    Two gates decide BOUND against STATIC, and both exist to stop the same
    failure. A Gaussian further than `max_distance` from the surface is not
    part of the face - it is hair, a collar, a spectacle frame, a fragment of
    the room - and binding it to whichever triangle happens to be nearest is
    what drags a wall across a cheek when the jaw opens. A Gaussian whose plane
    projection lands more than `max_excursion` triangle-widths outside its own
    triangle is sitting off the edge of the mesh, at the neck opening or inside
    the mouth, where its stored coordinates would act as a lever arm.

    Erring towards STATIC is deliberate. Too many static Gaussians is a face
    that under-articulates in a few places. Too few is a face that tears, and
    tearing is not a degraded likeness of a dead person - it is a grotesque.
    """
    count = len(gaussians)
    empty_i = np.zeros(0, dtype=np.int32)
    if count == 0:
        return Bindings(
            triangle=empty_i,
            barycentric=np.zeros((0, 3), dtype=DTYPE),
            normal_offset=np.zeros(0, dtype=DTYPE),
            local_rotation=np.zeros((0, 4), dtype=DTYPE),
            local_scale=np.zeros((0, 3), dtype=DTYPE),
            mode=np.zeros(0, dtype=np.uint8),
            static_index=empty_i,
            static_position=np.zeros((0, 3), dtype=DTYPE),
            static_rotation=np.zeros((0, 4), dtype=DTYPE),
            static_scale=np.zeros((0, 3), dtype=DTYPE),
            distance=np.zeros(0, dtype=DTYPE),
        )

    if max_distance is None:
        max_distance = BIND_RADIUS_IN_EDGES * mesh.median_edge_length()

    points = gaussians.positions
    chosen = _nearest_triangles(points, mesh, candidates)

    faces = mesh.faces[chosen]
    v0 = mesh.vertices[faces[:, 0]]
    v1 = mesh.vertices[faces[:, 1]]
    v2 = mesh.vertices[faces[:, 2]]
    tangent, bitangent, normal, tri_scale = triangle_frames(v0, v1, v2)

    # Barycentrics of the perpendicular projection onto the triangle's plane,
    # solved in the (e1, e2) basis via the 2x2 Gram matrix. Unclamped, so the
    # stored coordinates reproduce the Gaussian exactly rather than snapping it
    # to the triangle. The normal component drops out of the right-hand side
    # because e1 and e2 are both perpendicular to it.
    e1, e2 = v1 - v0, v2 - v0
    delta = points - v0
    g11, g12, g22 = _dot(e1, e1), _dot(e1, e2), _dot(e2, e2)
    b1, b2 = _dot(delta, e1), _dot(delta, e2)
    det = _safe(g11 * g22 - g12 * g12)
    v = (g22 * b1 - g12 * b2) / det
    w = (g11 * b2 - g12 * b1) / det
    u = 1.0 - v - w
    barycentric = np.stack([u, v, w], axis=-1).astype(DTYPE, copy=False)

    signed = _dot(delta, normal)

    # The distance that gates binding is to the closest point *on* the
    # triangle, not to its plane: a Gaussian a millimetre above the plane but
    # ten centimetres beyond the edge is ten centimetres away, and treating it
    # as a millimetre away is exactly how a piece of the background ends up
    # welded to a cheek.
    _, sq_clamped = _closest_barycentric(points, v0, v1, v2)
    distance = np.sqrt(np.maximum(sq_clamped, 0.0)).astype(DTYPE, copy=False)

    normal_offset = (signed / _safe(tri_scale)).astype(DTYPE, copy=False)

    # Rotation and scale, expressed in the triangle's frame. The frame is
    # orthonormal, so its inverse is its transpose and the local rotation is
    # just the triangle's rotation undone.
    tri_quat = quat_from_basis(tangent, bitangent, normal)
    local_rotation = quat_multiply(quat_conjugate(tri_quat), gaussians.rotations)
    local_scale = (gaussians.scales / _safe(tri_scale)[:, None]).astype(DTYPE, copy=False)

    excursion = np.maximum(0.0, -barycentric.min(axis=1))
    too_far = distance > max_distance
    too_far |= ~np.isfinite(distance)
    off_edge = excursion > max_excursion
    degenerate = mesh.areas()[chosen] <= _TINY
    static = too_far | off_edge | degenerate

    mode = np.where(static, BindMode.STATIC, BindMode.BOUND).astype(np.uint8)
    static_index = np.flatnonzero(static).astype(np.int32)

    return Bindings(
        triangle=chosen.astype(np.int32),
        barycentric=barycentric,
        normal_offset=normal_offset,
        local_rotation=local_rotation,
        local_scale=local_scale,
        mode=mode,
        static_index=static_index,
        static_position=points[static_index].copy(),
        static_rotation=gaussians.rotations[static_index].copy(),
        static_scale=gaussians.scales[static_index].copy(),
        distance=distance,
    )


def deform(bindings: Bindings, mesh: Mesh) -> Transforms:
    """Where every Gaussian goes, given the mesh as it is posed this frame.

    This is the per-frame function and the only thing in this module with a
    real time budget: at 25fps a frame is 40ms, shared with the renderer, so
    this needs to be a small fraction of that against a million Gaussians. It
    is therefore a fixed sequence of whole-array operations with no Python loop
    over Gaussians anywhere - not a loop that is fast enough, none at all. The
    STATIC rows are computed along with the rest and then overwritten, because
    one scatter is cheaper than the two fancy-index copies that skipping them
    would cost.

    Measured at about 10ms for 100,000 Gaussians on one unloaded laptop core -
    a quarter of a frame - so a PREVIEW splat at 200,000 fits comfortably and a
    STANDARD one at 800,000 does not, at roughly 80ms. That is the honest
    ceiling of the CPU path and the reason the shipping renderer does this
    arithmetic on the GPU: the operations here are per-Gaussian and independent,
    which is the shape a shader wants. This implementation is the reference the
    shader is checked against, and the one the whole pipeline is developed on
    while no GPU is in the room.

    `mesh` must have the same topology as the mesh that was bound. Deforming
    against a differently-triangulated mesh would index a triangle that means
    something else, which produces a face rather than an error - and a wrong
    face is much harder to notice than a raised exception.
    """
    if len(bindings) and int(bindings.triangle.max()) >= mesh.triangle_count:
        raise RigError(
            "these bindings reference a triangle this mesh does not have: the posed "
            "mesh must have the same topology as the mesh that was bound"
        )
    if len(bindings) == 0:
        return Transforms(
            positions=np.zeros((0, 3), dtype=DTYPE),
            rotations=np.zeros((0, 4), dtype=DTYPE),
            scales=np.zeros((0, 3), dtype=DTYPE),
        )

    # Three gathers of contiguous (N, 3) arrays rather than one gather of
    # (N, 3, 3) that is then sliced. They cost the same to fetch, and every
    # arithmetic operation afterwards runs on a contiguous buffer instead of
    # one with a nine-float stride. On the hot path that is worth about a fifth
    # of the whole frame.
    faces = mesh.faces[bindings.triangle]
    v0 = mesh.vertices[faces[:, 0]]
    v1 = mesh.vertices[faces[:, 1]]
    v2 = mesh.vertices[faces[:, 2]]
    tangent, bitangent, normal, tri_scale = triangle_frames(v0, v1, v2)

    bary = bindings.barycentric
    positions = bary[:, 0:1] * v0 + bary[:, 1:2] * v1 + bary[:, 2:3] * v2
    positions += (bindings.normal_offset * tri_scale)[:, None] * normal

    rotations = quat_multiply(
        quat_from_basis(tangent, bitangent, normal), bindings.local_rotation
    )
    scales = bindings.local_scale * tri_scale[:, None]

    index = bindings.static_index
    if index.size:
        positions[index] = bindings.static_position
        rotations[index] = bindings.static_rotation
        scales[index] = bindings.static_scale

    return Transforms(
        positions=positions.astype(DTYPE, copy=False),
        rotations=rotations.astype(DTYPE, copy=False),
        scales=scales.astype(DTYPE, copy=False),
    )


# --------------------------------------------------------------------------
# PoseFrame -> FLAME 2023 parameters.
#
# The motion system speaks in human terms: a brow rises, a jaw opens, a torso
# leans. FLAME speaks in a hundred anonymous PCA coefficients and four small
# joint rotations. The translation between them is partly exact, largely
# approximate, and in four places impossible - and this section says which is
# which for every single channel rather than producing a plausible number for
# all of them.
# --------------------------------------------------------------------------

# FLAME 2023's expression space. One hundred coefficients on a PCA basis built
# from expression scans; the components have no individual meaning and are not
# separable by side of the face.
FLAME_EXPRESSION_COUNT = 100

# The jaw joint's rotation at a fully open mouth, in radians. About 24 degrees,
# which is a wide speaking jaw rather than a yawn. FLAME's jaw is a real
# revolute joint, so this is a genuine angle rather than a blend weight.
FLAME_JAW_MAX_RAD = 0.42

# Eyeball rotation, radians per unit of gaze. pose.py's gaze channels are
# already in radians and already head-relative, exactly as FLAME's eye joints
# are, so this is one.
FLAME_GAZE_GAIN = 1.0


class Fidelity(StrEnum):
    """How true a channel's translation into FLAME is.

    Three values rather than a confidence number because they are three
    different kinds of statement and averaging them would be meaningless.
    """

    # The channel and the FLAME parameter are the same physical quantity. Head
    # yaw in radians is a neck joint rotation in radians.
    EXACT = "exact"
    # FLAME can express something like this, but not this. Usually because the
    # expression basis is global PCA: it has no notion of one eyebrow.
    APPROXIMATED = "approximated"
    # FLAME has no such degree of freedom at all. FLAME is a head and a neck.
    # It has no torso, no shoulders and no lungs.
    UNMAPPABLE = "unmappable"


@dataclass(frozen=True)
class ChannelMap:
    channel: str
    target: str
    fidelity: Fidelity
    note: str


# Every one of pose.py's twenty continuous channels, and what becomes of it.
# The module refuses to import if this table and pose.py disagree - see the
# assertion at the foot of the file - so a channel added to the motion system
# cannot silently arrive here and be dropped.
CHANNEL_MAP: dict[str, ChannelMap] = {
    m.channel: m
    for m in (
        ChannelMap(
            "head_yaw",
            "neck_pose[1]",
            Fidelity.EXACT,
            "FLAME's neck joint is an axis-angle rotation in radians and so is this "
            "channel. The axis assignment (x pitch, y yaw, z roll) is FLAME's stated "
            "convention and must be confirmed against the actual 2023 asset: a sign "
            "flip here is the classic first-integration bug and shows as a head that "
            "turns away from what it is looking at.",
        ),
        ChannelMap("head_pitch", "neck_pose[0]", Fidelity.EXACT, "As head_yaw, about x."),
        ChannelMap("head_roll", "neck_pose[2]", Fidelity.EXACT, "As head_yaw, about z."),
        ChannelMap(
            "gaze_yaw",
            "eye_pose[1] and eye_pose[4]",
            Fidelity.EXACT,
            "Both eyeballs get the same rotation, which is correct for anything more "
            "than a metre away and wrong for anything nearer: FLAME has the degrees of "
            "freedom for vergence, and the motion system does not produce it. That is "
            "a gap in the motion system, not in the mapping.",
        ),
        ChannelMap("gaze_pitch", "eye_pose[0] and eye_pose[3]", Fidelity.EXACT, "As gaze_yaw."),
        ChannelMap(
            "blink",
            "expression, along a fitted lid-closure direction",
            Fidelity.APPROXIMATED,
            "FLAME has no eyelid joint. Lid closure exists in the expression PCA only "
            "to the extent the scan corpus blinked, and it is one of the weakest "
            "directions in the basis. The direction used here must be fitted from "
            "frames of this person actually blinking; without that fit the lids barely "
            "move. This is the single largest known weakness of the mapping, because a "
            "face that does not blink properly reads as dead within two seconds.",
        ),
        ChannelMap(
            "lid_upper_l",
            "expression, along a fitted lid-raise direction",
            Fidelity.APPROXIMATED,
            "Same basis problem as blink, and additionally not separable by side: the "
            "expression basis is global, so the left and right lid channels are "
            "averaged and applied symmetrically. Asymmetric lids are not representable.",
        ),
        ChannelMap("lid_upper_r", "expression, along a fitted lid-raise direction",
                   Fidelity.APPROXIMATED, "See lid_upper_l; averaged with it."),
        ChannelMap(
            "brow_inner_l",
            "expression, along a fitted inner-brow-raise direction",
            Fidelity.APPROXIMATED,
            "Brow motion is well represented in FLAME's expression space - the scan "
            "corpus contains plenty of it - so the shape is good. What is lost is the "
            "side: a single-sided raise, which is most of what an eyebrow is for "
            "conversationally, cannot be expressed and is averaged away.",
        ),
        ChannelMap("brow_inner_r", "expression, along a fitted inner-brow-raise direction",
                   Fidelity.APPROXIMATED, "See brow_inner_l; averaged with it."),
        ChannelMap("brow_outer_l", "expression, along a fitted outer-brow-raise direction",
                   Fidelity.APPROXIMATED, "See brow_inner_l."),
        ChannelMap("brow_outer_r", "expression, along a fitted outer-brow-raise direction",
                   Fidelity.APPROXIMATED, "See brow_inner_l; averaged with brow_outer_l."),
        ChannelMap(
            "jaw_open",
            "jaw_pose[0]",
            Fidelity.EXACT,
            "The one channel FLAME models as an actual joint. 0..1 maps onto 0.."
            f"{FLAME_JAW_MAX_RAD} radians of rotation about the jaw hinge. Combined with "
            "the visemes by maximum rather than by sum, because a vowel and an open jaw "
            "arriving together would otherwise drive the joint past its stop.",
        ),
        ChannelMap(
            "mouth_smile_l",
            "expression, along a fitted smile direction",
            Fidelity.APPROXIMATED,
            "Lip-corner pull is present in the expression basis. Its asymmetry is not, "
            "so a wry half-smile - which is most of what this channel is for - becomes "
            "a symmetric one at half the amplitude.",
        ),
        ChannelMap("mouth_smile_r", "expression, along a fitted smile direction",
                   Fidelity.APPROXIMATED, "See mouth_smile_l; averaged with it."),
        ChannelMap(
            "mouth_press",
            "expression, along a fitted lip-press direction",
            Fidelity.APPROXIMATED,
            "Lip compression and lip roll are shallow directions in the expression "
            "basis and interact with the jaw. Reads as a thinner mouth rather than as "
            "pressed lips.",
        ),
        ChannelMap(
            "torso_lean",
            "nothing in FLAME",
            Fidelity.UNMAPPABLE,
            "FLAME is a head and a neck. There is no torso to lean. Carried out in "
            "FlameParams.unmapped for the body layer, which owns the root transform of "
            "the whole splat.",
        ),
        ChannelMap("torso_yaw", "nothing in FLAME", Fidelity.UNMAPPABLE,
                   "See torso_lean. Note this is *not* head_yaw and must not be folded "
                   "into the neck: doing so turns a body that turns into a head that "
                   "turns twice as far."),
        ChannelMap("shoulder_raise", "nothing in FLAME", Fidelity.UNMAPPABLE,
                   "FLAME's mesh ends at the neck. There are no shoulders."),
        ChannelMap(
            "breath",
            "nothing in FLAME",
            Fidelity.UNMAPPABLE,
            "Breathing is a chest and a shoulder line. It is also the cheapest signal "
            "that a likeness is alive, which makes losing it here consequential rather "
            "than tidy: the body layer must implement it or the avatar reads as a "
            "photograph with a moving mouth.",
        ),
    )
}

# The names come from pose.py, which owns them. They used to be duplicated
# here as this module's reading of what a fifteen-slot vector conventionally
# means - a guess that happened to be right, which is not the same as being
# correct. A permuted viseme set is a mouth making the wrong shape for every
# sound, and subtle enough to survive a demo, so it gets one definition.

@dataclass(frozen=True)
class VisemeMap:
    """One mouth shape: how far it opens the jaw, and what it does to the lips.

    The split is not cosmetic. Jaw aperture is a joint rotation FLAME models
    exactly; lip shape is expression PCA, where lip *protrusion* - which is the
    whole difference between /u/ and /i/ at the same aperture - is one of the
    poorest-covered directions in the basis.
    """

    name: str
    jaw: float
    direction: str
    fidelity: Fidelity
    note: str = ""


VISEME_MAP: tuple[VisemeMap, ...] = (
    VisemeMap("sil", 0.00, "", Fidelity.EXACT, "Rest. Nothing to express."),
    VisemeMap("PP", 0.00, "viseme_PP", Fidelity.APPROXIMATED, "Bilabial closure. The lips "
              "meeting is present in the basis but soft; plosive release is not modelled."),
    VisemeMap("FF", 0.05, "viseme_FF", Fidelity.APPROXIMATED, "Labiodental. The lower lip "
              "tucking under the teeth needs geometry FLAME does not separate from the jaw."),
    VisemeMap("TH", 0.15, "viseme_TH", Fidelity.UNMAPPABLE, "Interdental. FLAME 2023 has no "
              "tongue. The tongue between the teeth cannot be shown at all, and the shape "
              "falls back to an open jaw, which is what every tongue-less rig does."),
    VisemeMap("DD", 0.12, "viseme_DD", Fidelity.APPROXIMATED, "Alveolar. Tongue-driven; only "
              "the jaw and lip aperture survive."),
    VisemeMap("kk", 0.18, "viseme_kk", Fidelity.APPROXIMATED, "Velar. Almost entirely tongue, "
              "almost entirely invisible - which is why it approximates well."),
    VisemeMap("CH", 0.14, "viseme_CH", Fidelity.APPROXIMATED, "Postalveolar, with lip "
              "rounding. The rounding is the weak part."),
    VisemeMap("SS", 0.08, "viseme_SS", Fidelity.APPROXIMATED, "Sibilant. Narrow aperture, "
              "spread lips."),
    VisemeMap("nn", 0.10, "viseme_nn", Fidelity.APPROXIMATED, "Nasal. Tongue-driven."),
    VisemeMap("RR", 0.16, "viseme_RR", Fidelity.APPROXIMATED, "Rhotic. Lip rounding plus a "
              "tongue shape that is not available."),
    VisemeMap("aa", 0.75, "viseme_aa", Fidelity.EXACT, "Open vowel. Almost pure jaw, which "
              "FLAME models as a joint."),
    VisemeMap("E", 0.40, "viseme_E", Fidelity.APPROXIMATED, "Mid front vowel; spread lips."),
    VisemeMap("ih", 0.25, "viseme_ih", Fidelity.APPROXIMATED, "Close front vowel."),
    VisemeMap("oh", 0.45, "viseme_oh", Fidelity.APPROXIMATED, "Rounded back vowel. Protrusion "
              "is under-represented in the expression basis, so this reads as open rather "
              "than as rounded."),
    VisemeMap("ou", 0.20, "viseme_ou", Fidelity.APPROXIMATED, "Close rounded vowel. The worst "
              "case for the same reason as oh: aperture is right, protrusion is not."),
)


@dataclass(frozen=True)
class ExpressionBasis:
    """Named directions through FLAME's hundred anonymous expression coefficients.

    FLAME's expression components have no individual meaning - component 7 is
    not "smile" - so anything that wants to drive FLAME in human terms needs a
    set of direction vectors saying what "smile" is in that space. Those
    directions are *fitted*, per subject where possible, from frames of that
    person's own capture. They are not a constant of the model and this module
    does not ship them.

    `placeholder()` exists so that the whole path from PoseFrame to parameters
    is exercisable, and tested, with no FLAME asset present. It assigns each
    named direction its own reserved coefficient with unit weight. The plumbing
    is real; the geometry it would produce is meaningless, and `fitted` says so
    so that nothing downstream can mistake one for the other.
    """

    directions: dict[str, np.ndarray]
    fitted: bool = False

    @classmethod
    def placeholder(cls) -> ExpressionBasis:
        names = [
            "lid_close", "lid_raise", "brow_inner_raise", "brow_outer_raise",
            "smile", "lip_press",
        ] + [m.direction for m in VISEME_MAP if m.direction]
        directions = {}
        for i, name in enumerate(names):
            vector = np.zeros(FLAME_EXPRESSION_COUNT, dtype=DTYPE)
            vector[i % FLAME_EXPRESSION_COUNT] = 1.0
            directions[name] = vector
        return cls(directions=directions, fitted=False)

    def direction(self, name: str) -> np.ndarray:
        """The named direction, or zero. A missing direction moves nothing.

        Zero rather than an exception: a basis fitted from a capture where the
        person never pressed their lips genuinely has no lip-press direction,
        and the honest result is a face that does not press its lips, not a
        crash on the frame where it was asked to.
        """
        found = self.directions.get(name)
        if found is None:
            return np.zeros(FLAME_EXPRESSION_COUNT, dtype=DTYPE)
        return np.asarray(found, dtype=DTYPE)


@dataclass(frozen=True)
class FlameParams:
    """One frame of FLAME 2023, plus an honest account of what did not fit.

    `unmapped` and `approximate` are part of the value rather than a log line.
    Four of the motion system's channels describe a body FLAME does not have;
    they are carried out of here so the body layer consumes them, and so that
    the only way to lose them is to explicitly ignore a field.
    """

    expression: np.ndarray  # (100,)
    neck_pose: np.ndarray  # (3,) axis-angle radians
    jaw_pose: np.ndarray  # (3,)
    eye_pose: np.ndarray  # (6,) left xyz then right xyz
    global_pose: np.ndarray  # (3,)
    unmapped: dict[str, float]
    approximate: tuple[str, ...]
    basis_fitted: bool = False

    @property
    def full_pose(self) -> np.ndarray:
        """FLAME's fifteen-element pose vector, in its own order."""
        return np.concatenate(
            [self.global_pose, self.neck_pose, self.jaw_pose, self.eye_pose]
        ).astype(DTYPE, copy=False)


def pose_to_flame(frame: PoseFrame, basis: ExpressionBasis | None = None) -> FlameParams:
    """Translate one frame of the motion system into FLAME 2023 parameters.

    Head pose lands on the neck joint and not on the global rotation: the
    global rotation is where the body layer puts the whole person, and putting
    a head nod there would nod the shoulders too.

    Everything that reaches the expression vector is a weighted sum of fitted
    directions, and the sum is *not* normalised. FLAME's expression space is
    linear and its coefficients are unbounded in principle; in practice driving
    it far outside the range the scan corpus covered produces a face that is
    not a face. Bounding that is the fitting stage's job, not this function's,
    because the safe range depends on the directions that were fitted.
    """
    basis = basis if basis is not None else ExpressionBasis.placeholder()
    expression = np.zeros(FLAME_EXPRESSION_COUNT, dtype=DTYPE)

    # Head. Exact: radians in, radians out.
    neck_pose = np.array([frame.head_pitch, frame.head_yaw, frame.head_roll], dtype=DTYPE)

    # Gaze. Exact in kind, and applied identically to both eyeballs - the
    # motion system produces no vergence, so neither does this.
    eye = np.array([frame.gaze_pitch * FLAME_GAZE_GAIN, frame.gaze_yaw * FLAME_GAZE_GAIN, 0.0])
    eye_pose = np.concatenate([eye, eye]).astype(DTYPE)

    # Jaw. The one joint FLAME and the motion system agree about. Visemes and
    # the explicit jaw channel are combined by maximum, never by sum: both
    # describe the same hinge, and adding them drives it through the chin.
    viseme_jaw = 0.0
    for weight, mapping in zip(frame.visemes, VISEME_MAP, strict=True):
        viseme_jaw = max(viseme_jaw, weight * mapping.jaw)
        if mapping.direction:
            expression += basis.direction(mapping.direction) * float(weight)
    jaw = max(float(frame.jaw_open), viseme_jaw)
    jaw_pose = np.array([jaw * FLAME_JAW_MAX_RAD, 0.0, 0.0], dtype=DTYPE)

    # Everything else is expression PCA, and every one of these is symmetric
    # because the basis is: the paired channels are averaged rather than
    # applied per side. Averaging is the honest lossy choice - taking the
    # larger side would exaggerate, taking the left would be arbitrary.
    lid_raise = 0.5 * (frame.lid_upper_l + frame.lid_upper_r)
    brow_inner = 0.5 * (frame.brow_inner_l + frame.brow_inner_r)
    brow_outer = 0.5 * (frame.brow_outer_l + frame.brow_outer_r)
    smile = 0.5 * (frame.mouth_smile_l + frame.mouth_smile_r)

    expression += basis.direction("lid_close") * float(frame.blink)
    expression += basis.direction("lid_raise") * float(lid_raise)
    expression += basis.direction("brow_inner_raise") * float(brow_inner)
    expression += basis.direction("brow_outer_raise") * float(brow_outer)
    expression += basis.direction("smile") * float(smile)
    expression += basis.direction("lip_press") * float(frame.mouth_press)

    unmapped = {
        name: float(getattr(frame, name))
        for name, mapping in CHANNEL_MAP.items()
        if mapping.fidelity is Fidelity.UNMAPPABLE
    }
    approximate = tuple(
        name for name, mapping in CHANNEL_MAP.items()
        if mapping.fidelity is Fidelity.APPROXIMATED
    )

    return FlameParams(
        expression=expression,
        neck_pose=neck_pose,
        jaw_pose=jaw_pose,
        eye_pose=eye_pose,
        # Left at zero on purpose: the root transform belongs to the body
        # layer, which is also where torso_lean and torso_yaw are consumed.
        global_pose=np.zeros(3, dtype=DTYPE),
        unmapped=unmapped,
        approximate=approximate,
        basis_fitted=basis.fitted,
    )


def mapping_table() -> list[tuple[str, str, str]]:
    """Channel, FLAME target, fidelity - for documentation and for review.

    Exists so the table can be printed rather than read out of source, because
    the thing most likely to go wrong with this mapping is that somebody
    assumes a channel is doing something it is not.
    """
    rows = [(m.channel, m.target, m.fidelity.value) for m in CHANNEL_MAP.values()]
    rows += [
        (f"viseme[{i}] {m.name}", f"jaw_pose[0] x{m.jaw} + {m.direction or 'nothing'}",
         m.fidelity.value)
        for i, m in enumerate(VISEME_MAP)
    ]
    return rows


# What a rigged splat cannot do. Stated here, in the module that would
# otherwise be blamed for it, and in the same register as
# QualityReport.disclosure: a family should be told these before they see them.
LIMITATIONS: tuple[str, ...] = (
    (
        "A splat can only show expressions a camera saw. Gaussian colour and opacity are "
        "baked at build time and this rig never touches them, so a smile is the geometry "
        "of a smile lit by whatever light was on the face when it was still. If nobody "
        "photographed them laughing, the avatar's laugh will be their neutral face with "
        "its mouth open."
    ),
    (
        "Hair does not follow the face. Hair standing off the scalp binds nowhere and is "
        "marked STATIC by design, because binding it is what makes splat avatars tear. It "
        "will therefore sit still while the head turns underneath it - visible, wrong, and "
        "far less wrong than the alternative. Hair needs its own simulation or its own "
        "coarse cage; neither is in this module."
    ),
    (
        "Anything STATIC stays put: collar, spectacle frames, earrings, background "
        "fragments the optimiser left behind, and any Gaussian more than the bind radius "
        "from the mesh. On a clean capture this is a few percent. On a noisy one it can be "
        "much more, and a high static fraction is the number to look at when an avatar "
        "reads as stiff."
    ),
    (
        "FLAME 2023 has no tongue and no teeth. Interdental sounds cannot be shown at all "
        "and are approximated by jaw aperture, which is what every tongue-less rig does "
        "and what every viewer eventually notices."
    ),
    (
        "The expression basis is global and symmetric. A raised single eyebrow, a wry "
        "one-sided smile and an asymmetric blink are not representable and are averaged "
        "into symmetric versions at reduced amplitude - which removes a good deal of what "
        "makes one person's face theirs."
    ),
    (
        "Four of the motion system's twenty channels - torso lean, torso yaw, shoulder "
        "raise and breath - have no FLAME equivalent whatsoever. They are handed back in "
        "FlameParams.unmapped and are the body layer's problem. If nothing consumes them "
        "the avatar does not breathe, and a likeness that does not breathe reads as a "
        "photograph however good the face is."
    ),
    (
        "The rig deforms; it does not re-optimise. Skin that should slide over bone, a lip "
        "that should thin as it stretches, and a wrinkle that should deepen are all "
        "outside what a fixed barycentric attachment can express."
    ),
)


# The motion system's channel table is the contract. If a channel is added to
# pose.py and not handled here, this module fails to import rather than
# silently animating nineteen of twenty things.
assert set(CHANNEL_MAP) == set(channel_names()), (
    "the FLAME mapping and pose.py have diverged: "
    f"{set(channel_names()) ^ set(CHANNEL_MAP)}"
)
assert len(VISEME_MAP) == VISEME_COUNT, "the viseme mapping and pose.py have diverged"
assert tuple(m.name for m in VISEME_MAP) == VISEME_NAMES
