"""Regular-tetrahedron geometry primitives and dense cluster generators.

This module provides the geometric substrate for spectral analysis of
tetrahedral point clouds:

* :func:`regular_tetrahedron` -- the canonical unit-edge regular tetrahedron.
* :func:`build_honeycomb` -- the tetrahedral-octahedral (FCC) honeycomb, an
  *interlocking* arrangement in which tetrahedra share vertices and edges.
* :func:`build_dense_packing` -- a genuine non-overlapping packing produced by
  an adaptive-shrinking-cell (ASC) Monte Carlo search, with the achieved
  packing fraction measured rather than assumed.
* :func:`sat_overlap_depth` -- an exact separating-axis overlap test for pairs
  of tetrahedra, used to certify that packings are physically valid.

All coordinates are ``float64``.  Edge length is normalised to 1 throughout, so
the "unit distance" used to build graphs is literally ``1.0``.

Geometric conventions
---------------------
A *cloud* is described by :class:`TetraCloud`, which stores the tetrahedra as
an ``(n_tetra, 4, 3)`` array of vertex coordinates.  Vertex de-duplication is a
graph-construction concern and lives in :mod:`tetra_spectral`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

__all__ = [
    "TetraCloud",
    "regular_tetrahedron",
    "tetrahedron_volume",
    "TETRA_EDGES",
    "TETRA_FACES",
    "sat_overlap_depth",
    "build_honeycomb",
    "build_dense_packing",
    "PackingResult",
]

LOGGER = logging.getLogger(__name__)

F64 = np.float64
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: Volume of a regular tetrahedron of edge length 1.
UNIT_TETRA_VOLUME: float = 1.0 / (6.0 * math.sqrt(2.0))

#: Circumradius of a regular tetrahedron of edge length 1.
UNIT_TETRA_CIRCUMRADIUS: float = math.sqrt(3.0 / 8.0)

#: The six vertex-index pairs spanning the edges of a tetrahedron.
TETRA_EDGES: IntArray = np.array(
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int64
)

#: The four vertex-index triples spanning the faces of a tetrahedron.
TETRA_FACES: IntArray = np.array(
    [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)], dtype=np.int64
)


# --------------------------------------------------------------------------- #
# Canonical shape
# --------------------------------------------------------------------------- #
def regular_tetrahedron(edge: float = 1.0) -> FloatArray:
    """Return the 4 vertices of a regular tetrahedron centred at the origin.

    The construction takes four alternating corners of a cube, which is the
    numerically cleanest way to obtain an exactly regular tetrahedron: every
    pair of the selected corners is separated by a face diagonal of the cube,
    so all six edges are equal by construction rather than by optimisation.

    Parameters
    ----------
    edge:
        Desired edge length.  Must be strictly positive.

    Returns
    -------
    ndarray, shape (4, 3), dtype float64
        Vertices whose centroid is the origin and whose six pairwise distances
        all equal ``edge``.
    """
    if not np.isfinite(edge) or edge <= 0.0:
        raise ValueError(f"edge must be a positive finite float, got {edge!r}")

    cube_corners = np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
        dtype=F64,
    )
    # Face diagonal of the (side-2) cube is 2*sqrt(2); rescale to `edge`.
    vertices = cube_corners * (edge / (2.0 * math.sqrt(2.0)))
    return np.ascontiguousarray(vertices, dtype=F64)


def tetrahedron_volume(vertices: FloatArray) -> float:
    """Signed-magnitude volume of a tetrahedron given its 4 vertices."""
    v = np.asarray(vertices, dtype=F64)
    if v.shape != (4, 3):
        raise ValueError(f"expected shape (4, 3), got {v.shape}")
    return float(abs(np.linalg.det(v[1:] - v[0])) / 6.0)


def edge_lengths(tetra: FloatArray) -> FloatArray:
    """Return the six edge lengths for each tetrahedron in ``(n, 4, 3)``."""
    t = np.asarray(tetra, dtype=F64)
    if t.ndim != 3 or t.shape[1:] != (4, 3):
        raise ValueError(f"expected shape (n, 4, 3), got {t.shape}")
    i, j = TETRA_EDGES[:, 0], TETRA_EDGES[:, 1]
    return np.linalg.norm(t[:, i, :] - t[:, j, :], axis=2)


# --------------------------------------------------------------------------- #
# Cloud container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TetraCloud:
    """An immutable collection of regular tetrahedra.

    Attributes
    ----------
    tetrahedra:
        ``(n_tetra, 4, 3)`` float64 vertex coordinates.
    lattice:
        Optional ``(3, 3)`` array whose *rows* are the generating lattice
        vectors of the underlying periodic cell, when one exists.
    tetra_per_cell:
        Number of tetrahedra in the periodic unit cell, when applicable.
    provenance:
        Free-form metadata describing how the cloud was generated.
    """

    tetrahedra: FloatArray
    lattice: FloatArray | None = None
    tetra_per_cell: int | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = np.ascontiguousarray(self.tetrahedra, dtype=F64)
        if t.ndim != 3 or t.shape[1:] != (4, 3):
            raise ValueError(f"tetrahedra must have shape (n, 4, 3), got {t.shape}")
        if t.shape[0] == 0:
            raise ValueError("cloud must contain at least one tetrahedron")
        if not np.all(np.isfinite(t)):
            raise ValueError("tetrahedra contain non-finite coordinates")
        object.__setattr__(self, "tetrahedra", t)

        if self.lattice is not None:
            lat = np.ascontiguousarray(self.lattice, dtype=F64)
            if lat.shape != (3, 3):
                raise ValueError(f"lattice must have shape (3, 3), got {lat.shape}")
            object.__setattr__(self, "lattice", lat)

    @property
    def n_tetrahedra(self) -> int:
        return int(self.tetrahedra.shape[0])

    @property
    def raw_points(self) -> FloatArray:
        """All ``4 * n_tetra`` vertices, with duplicates, as ``(m, 3)``."""
        return self.tetrahedra.reshape(-1, 3)

    @property
    def occupied_volume(self) -> float:
        """Total volume of the constituent tetrahedra."""
        return self.n_tetrahedra * UNIT_TETRA_VOLUME

    def max_edge_deviation(self, edge: float = 1.0) -> float:
        """Largest absolute deviation of any edge from ``edge``."""
        return float(np.max(np.abs(edge_lengths(self.tetrahedra) - edge)))

    def assert_regular(self, edge: float = 1.0, atol: float = 1e-9) -> None:
        """Raise if any tetrahedron is not regular with the requested edge."""
        dev = self.max_edge_deviation(edge)
        if dev > atol:
            raise AssertionError(
                f"cloud contains irregular tetrahedra: max edge deviation "
                f"{dev:.3e} exceeds atol={atol:.3e}"
            )


# --------------------------------------------------------------------------- #
# Exact overlap test (separating axis theorem)
# --------------------------------------------------------------------------- #
def _candidate_axes(a: FloatArray, b: FloatArray) -> FloatArray:
    """Build SAT candidate axes for batched tetrahedron pairs.

    For two convex polytopes it is sufficient (and necessary) to test the face
    normals of each body plus the cross products of every pair of edge
    directions.  For tetrahedra that is ``4 + 4 + 36 = 44`` axes.

    Parameters
    ----------
    a, b:
        ``(m, 4, 3)`` batches of tetrahedron vertices.

    Returns
    -------
    ndarray, shape (m, 44, 3)
        Un-normalised candidate separating axes.
    """
    m = a.shape[0]

    def face_normals(t: FloatArray) -> FloatArray:
        f = TETRA_FACES
        p0 = t[:, f[:, 0], :]
        p1 = t[:, f[:, 1], :]
        p2 = t[:, f[:, 2], :]
        return np.cross(p1 - p0, p2 - p0)

    def edge_dirs(t: FloatArray) -> FloatArray:
        i, j = TETRA_EDGES[:, 0], TETRA_EDGES[:, 1]
        return t[:, j, :] - t[:, i, :]

    ea = edge_dirs(a)  # (m, 6, 3)
    eb = edge_dirs(b)  # (m, 6, 3)
    cross = np.cross(ea[:, :, None, :], eb[:, None, :, :]).reshape(m, 36, 3)

    return np.concatenate([face_normals(a), face_normals(b), cross], axis=1)


def sat_overlap_depth(a: FloatArray, b: FloatArray) -> FloatArray:
    """Exact overlap depth for batched tetrahedron pairs.

    Uses the separating axis theorem, which is exact for convex polytopes.

    Parameters
    ----------
    a, b:
        ``(m, 4, 3)`` float64 batches of tetrahedron vertices.

    Returns
    -------
    ndarray, shape (m,), dtype float64
        The minimum projection overlap across all candidate axes.  A value
        ``<= 0`` certifies that the pair is disjoint (a separating axis
        exists); a positive value is the penetration depth along the least
        overlapping axis.
    """
    a = np.ascontiguousarray(a, dtype=F64)
    b = np.ascontiguousarray(b, dtype=F64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (4, 3):
        raise ValueError(
            f"a and b must both have shape (m, 4, 3); got {a.shape} and {b.shape}"
        )
    if a.shape[0] == 0:
        return np.zeros(0, dtype=F64)

    axes = _candidate_axes(a, b)  # (m, 44, 3)
    norms = np.linalg.norm(axes, axis=2)

    # Degenerate axes (parallel edges, zero-area faces) carry no information.
    # Guard the division, then neutralise them so they can never *declare*
    # separation -- only well-conditioned axes may do that.
    valid = norms > 1e-12
    safe = np.where(valid, norms, 1.0)
    axes = axes / safe[:, :, None]

    proj_a = np.einsum("mad,mvd->mav", axes, a)  # (m, 44, 4)
    proj_b = np.einsum("mad,mvd->mav", axes, b)

    overlap = np.minimum(proj_a.max(axis=2), proj_b.max(axis=2)) - np.maximum(
        proj_a.min(axis=2), proj_b.min(axis=2)
    )
    overlap = np.where(valid, overlap, np.inf)
    return np.asarray(overlap.min(axis=1), dtype=F64)


def find_overlapping_pairs(
    tetrahedra: FloatArray,
    *,
    tolerance: float = 1e-9,
    chunk_size: int = 200_000,
) -> IntArray:
    """Return index pairs of tetrahedra that genuinely interpenetrate.

    A broad phase using circumsphere proximity (via a KD-tree on centroids)
    reduces the candidate set to O(n) pairs before the exact SAT narrow phase
    runs in vectorised chunks.

    Parameters
    ----------
    tetrahedra:
        ``(n, 4, 3)`` vertex coordinates.
    tolerance:
        Penetration depths at or below this value are treated as contact
        rather than overlap, absorbing floating-point noise on face-to-face
        and vertex-sharing configurations.
    chunk_size:
        Maximum number of candidate pairs evaluated per vectorised batch.

    Returns
    -------
    ndarray, shape (k, 2), dtype int64
        Sorted, unique index pairs whose overlap depth exceeds ``tolerance``.
    """
    t = np.ascontiguousarray(tetrahedra, dtype=F64)
    centroids = t.mean(axis=1)
    tree = cKDTree(centroids)
    candidates = tree.query_pairs(
        r=2.0 * UNIT_TETRA_CIRCUMRADIUS, output_type="ndarray"
    )
    if candidates.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    hits: list[IntArray] = []
    for start in range(0, candidates.shape[0], chunk_size):
        block = candidates[start : start + chunk_size]
        depth = sat_overlap_depth(t[block[:, 0]], t[block[:, 1]])
        bad = block[depth > tolerance]
        if bad.size:
            hits.append(bad)

    if not hits:
        return np.zeros((0, 2), dtype=np.int64)
    return np.ascontiguousarray(np.concatenate(hits, axis=0), dtype=np.int64)


# --------------------------------------------------------------------------- #
# Backend 1: tetrahedral-octahedral (FCC) honeycomb
# --------------------------------------------------------------------------- #
def _fcc_sites(radius: float) -> FloatArray:
    """FCC lattice sites with nearest-neighbour distance 1 inside a ball.

    The FCC lattice is realised as ``(i, j, k) / sqrt(2)`` restricted to
    ``i + j + k`` even, which places nearest neighbours at distance exactly 1.
    A spherical cut-off is used because a ball is invariant under the full
    octahedral group :math:`O_h`, so the resulting cluster inherits the point
    symmetry whose irreducible representations drive eigenvalue degeneracy.
    """
    scale = 1.0 / math.sqrt(2.0)
    n = int(math.ceil(radius * math.sqrt(2.0))) + 1
    rng = np.arange(-n, n + 1, dtype=np.int64)
    i, j, k = np.meshgrid(rng, rng, rng, indexing="ij")
    parity = (i + j + k) % 2 == 0
    pts = np.stack([i[parity], j[parity], k[parity]], axis=1).astype(F64) * scale
    inside = np.linalg.norm(pts, axis=1) <= radius + 1e-12
    pts = pts[inside]
    order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
    return np.ascontiguousarray(pts[order], dtype=F64)


def _unit_cliques_of_size_four(points: FloatArray, tol: float = 1e-9) -> IntArray:
    """Enumerate all 4-cliques of the unit-distance graph on ``points``.

    In the FCC nearest-neighbour graph every 4-clique has all six pairwise
    distances equal to 1, hence *is* a regular unit tetrahedron; and no clique
    of size 5 exists (an octahedron's antipodal vertices are sqrt(2) apart), so
    the maximal cliques of size 4 are exactly the tetrahedral cells of the
    honeycomb.
    """
    import networkx as nx

    tree = cKDTree(points)
    pairs = tree.query_pairs(r=1.0 + tol, output_type="ndarray")
    if pairs.size:
        d = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
        pairs = pairs[np.abs(d - 1.0) <= tol]

    graph = nx.Graph()
    graph.add_nodes_from(range(points.shape[0]))
    graph.add_edges_from(map(tuple, pairs))

    cliques = [sorted(c) for c in nx.find_cliques(graph) if len(c) == 4]
    if not cliques:
        return np.zeros((0, 4), dtype=np.int64)
    out = np.array(sorted(cliques), dtype=np.int64)
    return np.ascontiguousarray(out)


def build_honeycomb(min_tetrahedra: int = 1000, *, max_radius: float = 40.0) -> TetraCloud:
    """Build a tetrahedral-octahedral honeycomb cluster.

    The honeycomb is the space-filling arrangement of regular tetrahedra and
    regular octahedra on the FCC lattice.  Tetrahedra meet face-to-face and
    share vertices and edges, which is precisely what makes the merged
    unit-distance graph connected and richly symmetric.  Tetrahedra alone
    occupy exactly 1/3 of space here (two tetrahedra and one octahedron per
    lattice site, with the octahedron four times the tetrahedron's volume).

    The cluster radius grows until at least ``min_tetrahedra`` cells exist.

    Parameters
    ----------
    min_tetrahedra:
        Minimum number of tetrahedra the returned cloud must contain.
    max_radius:
        Safety bound on the cluster radius, in units of the edge length.

    Returns
    -------
    TetraCloud
    """
    if min_tetrahedra < 1:
        raise ValueError("min_tetrahedra must be >= 1")

    radius = 2.0
    while radius <= max_radius:
        sites = _fcc_sites(radius)
        if sites.shape[0] >= 4:
            simplices = _unit_cliques_of_size_four(sites)
            if simplices.shape[0] >= min_tetrahedra:
                cloud = TetraCloud(
                    tetrahedra=sites[simplices],
                    provenance={
                        "backend": "honeycomb",
                        "lattice": "face-centred cubic",
                        "cluster_radius": float(radius),
                        "n_lattice_sites": int(sites.shape[0]),
                        "point_group": "O_h",
                        "tetrahedral_volume_fraction": 1.0 / 3.0,
                        "space_filling_with": "regular octahedra",
                    },
                )
                cloud.assert_regular(edge=1.0, atol=1e-9)
                LOGGER.info(
                    "honeycomb: radius=%.2f, sites=%d, tetrahedra=%d",
                    radius,
                    sites.shape[0],
                    cloud.n_tetrahedra,
                )
                return cloud
        radius += 1.0

    raise RuntimeError(
        f"could not reach {min_tetrahedra} tetrahedra within max_radius={max_radius}"
    )


# --------------------------------------------------------------------------- #
# Backend 2: dense packing via adaptive shrinking cell Monte Carlo
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PackingResult:
    """Outcome of an adaptive-shrinking-cell packing search."""

    lattice: FloatArray
    centres: FloatArray
    rotations: FloatArray
    packing_fraction: float
    accepted_moves: int
    attempted_moves: int
    n_particles: int

    @property
    def acceptance_ratio(self) -> float:
        return self.accepted_moves / max(self.attempted_moves, 1)


def _quat_to_matrix(q: FloatArray) -> FloatArray:
    """Convert a batch of unit quaternions ``(n, 4)`` to rotations ``(n, 3, 3)``."""
    q = np.asarray(q, dtype=F64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
        ],
        axis=1,
    )


def _random_quaternions(rng: np.random.Generator, n: int) -> FloatArray:
    """Uniformly distributed unit quaternions (Shoemake's method)."""
    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    s1, s2 = np.sqrt(1.0 - u1), np.sqrt(u1)
    return np.stack(
        [
            s1 * np.sin(2 * np.pi * u2),
            s1 * np.cos(2 * np.pi * u2),
            s2 * np.sin(2 * np.pi * u3),
            s2 * np.cos(2 * np.pi * u3),
        ],
        axis=1,
    ).astype(F64)


def _perturbation_quaternion(rng: np.random.Generator, magnitude: float) -> FloatArray:
    """A quaternion representing a small random rotation of the given scale."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.normal() * magnitude
    half = 0.5 * angle
    return np.array(
        [math.cos(half), *(math.sin(half) * axis)], dtype=F64
    )


def _quat_multiply(a: FloatArray, b: FloatArray) -> FloatArray:
    """Hamilton product of two quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=F64,
    )


def _lattice_widths(lattice: FloatArray) -> FloatArray:
    """Perpendicular width of the cell along each lattice direction."""
    volume = abs(float(np.linalg.det(lattice)))
    widths = np.empty(3, dtype=F64)
    for axis in range(3):
        other = lattice[[i for i in range(3) if i != axis]]
        area = np.linalg.norm(np.cross(other[0], other[1]))
        widths[axis] = volume / max(area, 1e-300)
    return widths


#: Largest per-axis periodic image span the overlap test will enumerate.  A
#: configuration needing more than this is rejected rather than under-tested.
MAX_IMAGE_SPAN: int = 6


def _image_offsets(lattice: FloatArray) -> FloatArray | None:
    """Integer lattice translations that can bring two tetrahedra into contact.

    Two unit tetrahedra can only interact if their centroids lie within twice
    the circumradius, so the search range along each lattice direction is set
    by that reach divided by the cell's perpendicular width.  This is what
    keeps the packing valid even after the cell shrinks below the particle
    diameter, where a naive 3x3x3 minimum-image scheme silently misses
    collisions.

    Returns ``None`` when the cell is so anisotropic that the required span
    exceeds :data:`MAX_IMAGE_SPAN`; callers must treat that as "cannot certify"
    and reject the configuration.
    """
    reach = 2.0 * UNIT_TETRA_CIRCUMRADIUS
    widths = _lattice_widths(lattice)
    if not np.all(np.isfinite(widths)) or np.any(widths <= 1e-12):
        return None
    spans = np.maximum(np.ceil(reach / widths), 1.0).astype(np.int64)
    if np.any(spans > MAX_IMAGE_SPAN):
        return None
    grids = [np.arange(-s, s + 1, dtype=np.int64) for s in spans]
    a, b, c = np.meshgrid(*grids, indexing="ij")
    return np.stack([a.ravel(), b.ravel(), c.ravel()], axis=1).astype(F64)


def _cell_vertices(
    base: FloatArray, lattice: FloatArray, fractional: FloatArray, quats: FloatArray
) -> FloatArray:
    """Vertices ``(n, 4, 3)`` of every particle in the fundamental cell."""
    rot = _quat_to_matrix(quats)  # (n, 3, 3)
    oriented = np.einsum("nij,vj->nvi", rot, base)
    centres = fractional @ lattice
    return oriented + centres[:, None, :]


def _configuration_is_valid(
    verts: FloatArray, lattice: FloatArray, tolerance: float = 1e-12
) -> bool:
    """True when no particle overlaps another, including periodic images.

    The broad phase compares centroid separations against the circumsphere
    diameter, which is a strict necessary condition for contact; only the
    surviving candidates reach the exact SAT narrow phase.
    """
    offsets = _image_offsets(lattice)
    if offsets is None:
        return False

    n = verts.shape[0]
    shifts = offsets @ lattice  # (n_img, 3)
    n_img = shifts.shape[0]
    centres = verts.mean(axis=1)  # (n, 3)

    # The offset grid is symmetric, so image `k` and image `n_img - 1 - k` are
    # negatives of one another.  Pair (i, j, k) therefore duplicates
    # (j, i, n_img - 1 - k); keeping only the upper half of the image range
    # (plus j > i within the identity image) visits each physical pair once,
    # including a particle against its own periodic images.
    identity = int(np.argmin(np.linalg.norm(shifts, axis=1)))

    img_idx = np.repeat(np.arange(n_img, dtype=np.int64), n * n)
    i_idx = np.tile(np.repeat(np.arange(n, dtype=np.int64), n), n_img)
    j_idx = np.tile(np.arange(n, dtype=np.int64), n_img * n)

    keep = (img_idx > identity) | ((img_idx == identity) & (j_idx > i_idx))
    img_idx, i_idx, j_idx = img_idx[keep], i_idx[keep], j_idx[keep]

    separation = np.linalg.norm(
        centres[i_idx] - (centres[j_idx] + shifts[img_idx]), axis=1
    )
    near = separation < 2.0 * UNIT_TETRA_CIRCUMRADIUS
    if not np.any(near):
        return True

    img_idx, i_idx, j_idx = img_idx[near], i_idx[near], j_idx[near]
    shifted = verts[j_idx] + shifts[img_idx][:, None, :]
    depth = sat_overlap_depth(verts[i_idx], shifted)
    return bool(np.all(depth <= tolerance))


class _AdaptiveStep:
    """A Monte Carlo step size that self-tunes toward a target acceptance rate.

    Hard-particle acceptance collapses as the cell densifies, so fixed move
    amplitudes either stall early (too large) or crawl (too small).  Rescaling
    toward a ~30% acceptance rate keeps the search productive across the whole
    density range.
    """

    __slots__ = (
        "value",
        "_initial",
        "_target",
        "_lo",
        "_hi",
        "_interval",
        "_accepted",
        "_tried",
    )

    def __init__(
        self,
        value: float,
        *,
        target: float = 0.3,
        bounds: tuple[float, float] = (1e-6, 1.0),
        interval: int = 40,
    ) -> None:
        self.value = float(value)
        self._initial = float(value)
        self._target = float(target)
        self._lo, self._hi = bounds
        self._interval = int(interval)
        self._accepted = 0
        self._tried = 0

    def reset(self) -> None:
        """Restore the initial amplitude, re-opening a jammed search."""
        self.value = self._initial
        self._accepted = 0
        self._tried = 0

    def record(self, accepted: bool) -> None:
        self._tried += 1
        self._accepted += int(accepted)
        if self._tried < self._interval:
            return
        rate = self._accepted / self._tried
        if rate > self._target:
            self.value = min(self.value * 1.15, self._hi)
        else:
            self.value = max(self.value * 0.85, self._lo)
        self._accepted = 0
        self._tried = 0


def _matrix_to_quaternion(rot: FloatArray) -> FloatArray:
    """Convert a proper rotation matrix to a unit quaternion (w, x, y, z)."""
    r = np.asarray(rot, dtype=F64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s],
            dtype=F64,
        )
    else:
        axis = int(np.argmax(np.diag(r)))
        if axis == 0:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = np.array(
                [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s],
                dtype=F64,
            )
        elif axis == 1:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = np.array(
                [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s],
                dtype=F64,
            )
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = np.array(
                [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s],
                dtype=F64,
            )
    return q / np.linalg.norm(q)


def _dimer_partner() -> tuple[FloatArray, FloatArray]:
    """Rotation and offset placing a second tetrahedron face-to-face on the base.

    Reflecting the apex of the canonical tetrahedron through the plane of the
    opposite face produces the triangular bipyramid ("dimer") that underlies
    every known high-density tetrahedron packing.  A regular tetrahedron is
    achiral, so this mirror image is also reachable by a *proper* rotation
    combined with an odd permutation of the shared face's vertices -- which is
    what lets the partner be stored as an ordinary orientation quaternion.

    Returns
    -------
    (quaternion, offset)
        Unit quaternion of shape ``(4,)`` and centroid offset of shape ``(3,)``,
        both expressed in the base tetrahedron's own frame.
    """
    v = regular_tetrahedron(1.0)
    normal = np.cross(v[2] - v[1], v[3] - v[1])
    normal /= np.linalg.norm(normal)
    apex = v[0] - 2.0 * float(np.dot(v[0] - v[1], normal)) * normal

    # Swapping two vertices of the shared face turns the reflection into a
    # proper rotation without moving the point set.
    target = np.array([apex, v[1], v[3], v[2]], dtype=F64)
    offset = target.mean(axis=0)
    centred = target - offset

    u, _, vt = np.linalg.svd(v.T @ centred)
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(vt.T @ u.T)))])
    rotation = vt.T @ correction @ u.T

    residual = float(np.linalg.norm((rotation @ v.T).T - centred))
    if residual > 1e-9 or np.linalg.det(rotation) < 0.0:
        raise RuntimeError(f"dimer construction failed (residual {residual:.3e})")

    return _matrix_to_quaternion(rotation), np.ascontiguousarray(offset, dtype=F64)


def _seed_dimers(
    n_particles: int, lattice: FloatArray, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """Initial fractional centres and orientations built from dimer pairs.

    Tetrahedra are placed as face-sharing pairs at random positions and
    orientations.  An odd ``n_particles`` leaves one unpaired tetrahedron.
    """
    partner_q, partner_offset = _dimer_partner()
    inverse = np.linalg.inv(lattice)

    fractional = np.empty((n_particles, 3), dtype=F64)
    quats = np.empty((n_particles, 4), dtype=F64)

    for start in range(0, n_particles, 2):
        anchor_f = rng.random(3).astype(F64)
        anchor_q = _random_quaternions(rng, 1)[0]
        fractional[start] = anchor_f
        quats[start] = anchor_q

        if start + 1 < n_particles:
            rot = _quat_to_matrix(anchor_q[None])[0]
            shift = rot @ partner_offset
            fractional[start + 1] = anchor_f + shift @ inverse
            combined = _quat_multiply(anchor_q, partner_q)
            quats[start + 1] = combined / np.linalg.norm(combined)

    return fractional, quats


def _initial_lattice(
    n_particles: int, target_fraction: float, rng: np.random.Generator
) -> FloatArray:
    """A near-cubic cell sized to hold ``n_particles`` at ``target_fraction``."""
    volume = n_particles * UNIT_TETRA_VOLUME / target_fraction
    side = volume ** (1.0 / 3.0)
    jitter = np.eye(3, dtype=F64) + rng.normal(scale=0.02, size=(3, 3))
    return np.ascontiguousarray(side * jitter, dtype=F64)


def _asc_search(
    n_particles: int,
    cycles: int,
    seed: int,
    *,
    initial_fraction: float = 0.15,
    lattice_moves_per_cycle: int = 4,
    compression_bias: float = 0.004,
    reheat_interval: int = 250,
    seed_dimers: bool = False,
) -> PackingResult:
    """Adaptive shrinking cell Monte Carlo for hard regular tetrahedra.

    Each cycle attempts ``n_particles`` translation or rotation moves followed
    by ``lattice_moves_per_cycle`` volume-reducing cell deformations.  Any move
    that produces an overlap is rejected outright (hard-particle dynamics), so
    every accepted state -- including the final one -- is a certified valid
    packing.  Step sizes self-tune toward ~30% acceptance.  The achieved
    packing fraction is *measured*, never assumed.

    ``seed_dimers`` initialises the cell with face-sharing tetrahedron pairs,
    the motif underlying the known optima.  It defaults to off because it
    measurably does *not* help here: across three seeds it reached 0.42-0.52
    versus 0.49-0.72 for random initialisation, since reheating disassembles
    the seeded pairs long before the cell approaches jamming.  Run-to-run seed
    variance dominates, which is why :func:`build_dense_packing` restarts.
    """
    rng = np.random.default_rng(seed)
    base = regular_tetrahedron(1.0)

    lattice = _initial_lattice(n_particles, initial_fraction, rng)
    if seed_dimers:
        fractional, quats = _seed_dimers(n_particles, lattice, rng)
    else:
        fractional = rng.random((n_particles, 3)).astype(F64)
        quats = _random_quaternions(rng, n_particles)

    for _ in range(60):
        if _configuration_is_valid(_cell_vertices(base, lattice, fractional, quats), lattice):
            break
        lattice = lattice * 1.15
    else:
        raise RuntimeError("failed to find a valid initial packing configuration")

    trans_step = _AdaptiveStep(0.08, bounds=(1e-5, 0.5))
    rot_step = _AdaptiveStep(0.30, bounds=(1e-4, math.pi))
    shear_step = _AdaptiveStep(0.02, bounds=(1e-6, 0.2))
    squeeze_step = _AdaptiveStep(compression_bias, bounds=(1e-7, 0.05))

    identity3 = np.eye(3, dtype=F64)
    accepted = 0
    attempted = 0
    log_every = max(cycles // 10, 1)

    steps = (trans_step, rot_step, shear_step, squeeze_step)

    for cycle in range(cycles):
        # Hard-particle acceptance decays to zero once the configuration jams,
        # dragging every amplitude to its floor and freezing the search.
        # Periodically restoring the initial amplitudes lets particles hop
        # between voids, in the spirit of basin hopping; the lattice is never
        # relaxed, so density remains monotone across reheats.
        if reheat_interval > 0 and cycle > 0 and cycle % reheat_interval == 0:
            for step in steps:
                step.reset()

        for _ in range(n_particles):
            attempted += 1
            idx = int(rng.integers(n_particles))
            old_f = fractional[idx].copy()
            old_q = quats[idx].copy()

            # Translation and rotation are proposed separately so each step
            # size adapts on its own acceptance statistics.
            rotating = bool(rng.random() < 0.5)
            if rotating:
                trial_q = _quat_multiply(
                    _perturbation_quaternion(rng, rot_step.value), old_q
                )
                quats[idx] = trial_q / np.linalg.norm(trial_q)
            else:
                fractional[idx] = old_f + rng.normal(scale=trans_step.value, size=3)

            ok = _configuration_is_valid(
                _cell_vertices(base, lattice, fractional, quats), lattice
            )
            if ok:
                accepted += 1
            else:
                fractional[idx] = old_f
                quats[idx] = old_q
            (rot_step if rotating else trans_step).record(ok)

        for _ in range(lattice_moves_per_cycle):
            attempted += 1
            # Pure shear: symmetric *and traceless*, so it explores cell shape
            # without changing volume.  Volume reduction is then applied
            # explicitly, which makes every accepted lattice move strictly
            # densifying -- the cell can never random-walk back outward.
            shear = rng.normal(scale=shear_step.value, size=(3, 3))
            shear = 0.5 * (shear + shear.T)
            shear -= (np.trace(shear) / 3.0) * identity3
            candidate = lattice @ (identity3 + shear)

            det_now = abs(float(np.linalg.det(lattice)))
            det_try = abs(float(np.linalg.det(candidate)))
            ok = False
            if det_try > 1e-12 and det_now > 1e-12:
                target = det_now * (1.0 - squeeze_step.value)
                candidate = candidate * (target / det_try) ** (1.0 / 3.0)
                ok = _configuration_is_valid(
                    _cell_vertices(base, candidate, fractional, quats), candidate
                )
            if ok:
                lattice = candidate
                accepted += 1
            shear_step.record(ok)
            squeeze_step.record(ok)

        if cycle % log_every == 0:
            phi = n_particles * UNIT_TETRA_VOLUME / abs(float(np.linalg.det(lattice)))
            LOGGER.debug(
                "ASC cycle %d/%d: phi=%.4f trans=%.4f rot=%.4f shear=%.5f squeeze=%.6f",
                cycle,
                cycles,
                phi,
                trans_step.value,
                rot_step.value,
                shear_step.value,
                squeeze_step.value,
            )

    volume = abs(float(np.linalg.det(lattice)))
    return PackingResult(
        lattice=lattice,
        centres=fractional @ lattice,
        rotations=_quat_to_matrix(quats),
        packing_fraction=n_particles * UNIT_TETRA_VOLUME / volume,
        accepted_moves=accepted,
        attempted_moves=attempted,
        n_particles=n_particles,
    )


def build_dense_packing(
    min_tetrahedra: int = 1000,
    *,
    n_particles: int = 4,
    cycles: int = 1500,
    restarts: int = 4,
    seed: int = 20250805,
    max_replicas: int = 30,
) -> TetraCloud:
    """Build a dense, certified non-overlapping packing of regular tetrahedra.

    An adaptive-shrinking-cell search densifies a periodic cell holding
    ``n_particles`` tetrahedra; the optimised cell is then tiled until at least
    ``min_tetrahedra`` tetrahedra are present.

    The packing fraction reported in ``provenance`` is the value actually
    achieved by the search.  Outcomes vary substantially between random seeds
    -- roughly 0.45 to 0.75 at 1500 cycles -- so several independent restarts
    are run and the densest is kept.

    For reference, the best packing of regular tetrahedra known in the
    literature is the Chen-Engel-Glotzer dimer double lattice at 4000/4671
    (~0.856347).  This short stochastic search does not reproduce that optimum
    and makes no claim to; approaching it requires far longer runs and larger
    cells than a demonstration script should default to.  Nothing here reports
    a density that was not measured.

    Parameters
    ----------
    min_tetrahedra:
        Minimum tetrahedron count in the returned cloud.
    n_particles:
        Tetrahedra per periodic cell.  Four is the cell size of the known dimer
        double-lattice optima.
    cycles:
        Monte Carlo cycles per restart.  Higher values densify further, with
        diminishing returns once the configuration jams.
    restarts:
        Independent searches to run; the densest result is kept.  Seed variance
        is the dominant factor in the achieved density, so this matters more
        than raising ``cycles`` alone.
    seed:
        Base seed; restart ``i`` uses ``seed + i``.  Runs are reproducible.
    max_replicas:
        Safety bound on tiling repetitions per lattice direction.

    Returns
    -------
    TetraCloud
    """
    if min_tetrahedra < 1:
        raise ValueError("min_tetrahedra must be >= 1")
    if n_particles < 1:
        raise ValueError("n_particles must be >= 1")
    if cycles < 1:
        raise ValueError("cycles must be >= 1")
    if restarts < 1:
        raise ValueError("restarts must be >= 1")

    attempts: list[PackingResult] = []
    for index in range(restarts):
        attempt = _asc_search(n_particles, cycles, seed + index)
        attempts.append(attempt)
        LOGGER.info(
            "ASC restart %d/%d: packing fraction %.6f (acceptance %.3f)",
            index + 1,
            restarts,
            attempt.packing_fraction,
            attempt.acceptance_ratio,
        )

    result = max(attempts, key=lambda item: item.packing_fraction)
    fractions = np.array([a.packing_fraction for a in attempts], dtype=F64)
    LOGGER.info(
        "ASC best of %d: packing fraction %.6f (%.1f%% of the 0.856347 literature "
        "record); restart spread %.4f-%.4f",
        restarts,
        result.packing_fraction,
        100.0 * result.packing_fraction / (4000.0 / 4671.0),
        fractions.min(),
        fractions.max(),
    )

    reps = 1
    while reps <= max_replicas and n_particles * reps**3 < min_tetrahedra:
        reps += 1
    if n_particles * reps**3 < min_tetrahedra:
        raise RuntimeError(
            f"cannot reach {min_tetrahedra} tetrahedra within max_replicas={max_replicas}"
        )

    base = regular_tetrahedron(1.0)
    oriented = np.einsum("nij,vj->nvi", result.rotations, base) + result.centres[:, None, :]

    grid = np.arange(reps, dtype=F64)
    a, b, c = np.meshgrid(grid, grid, grid, indexing="ij")
    translations = (
        np.stack([a.ravel(), b.ravel(), c.ravel()], axis=1) @ result.lattice
    )  # (reps^3, 3)

    tiled = (oriented[None, :, :, :] + translations[:, None, None, :]).reshape(-1, 4, 3)

    cloud = TetraCloud(
        tetrahedra=tiled,
        lattice=result.lattice,
        tetra_per_cell=n_particles,
        provenance={
            "backend": "dense_packing",
            "method": "adaptive shrinking cell Monte Carlo (hard particles)",
            "measured_packing_fraction": result.packing_fraction,
            "literature_record_packing_fraction": 4000.0 / 4671.0,
            "asc_cycles": cycles,
            "asc_restarts": restarts,
            "asc_restart_spread": (float(fractions.min()), float(fractions.max())),
            "asc_seed": seed,
            "asc_acceptance_ratio": result.acceptance_ratio,
            "particles_per_cell": n_particles,
            "replicas_per_axis": reps,
        },
    )
    cloud.assert_regular(edge=1.0, atol=1e-9)
    LOGGER.info("packing: %d tetrahedra from %d^3 replicas", cloud.n_tetrahedra, reps)
    return cloud
