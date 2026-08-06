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
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

import tetra_fastsat

__all__ = [
    "TetraCloud",
    "regular_tetrahedron",
    "tetrahedron_volume",
    "TETRA_EDGES",
    "TETRA_FACES",
    "sat_overlap_depth",
    "sat_disjoint",
    "build_honeycomb",
    "build_ceg_packing",
    "build_n3_packing",
    "build_n3_cluster",
    "double_lattice_cell",
    "refine_double_lattice",
    "N2_PACKING_FRACTION",
    "n3_unit_cell",
    "N3_PACKING_FRACTION",
    "ceg_unit_cell",
    "ceg_family_vectors",
    "ceg_packing_fraction",
    "ceg_in_restricted_space",
    "CEG_FAMILY_PRESETS",
    "CEG_PRESET_FRACTIONS",
    "build_dense_packing",
    "build_motif",
    "MOTIF_NAMES",
    "CEG_PACKING_FRACTION",
    "PackingResult",
]

LOGGER = logging.getLogger(__name__)

#: Set False to force the array implementation of the overlap test.
_USE_FAST_SAT: bool = True

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
def _cross3(u: FloatArray, v: FloatArray) -> FloatArray:
    """Cross product over the last axis, without ``np.cross``'s dispatch cost.

    ``np.cross`` spends most of its time in ``moveaxis`` and axis normalisation,
    which is pure overhead for fixed three-component vectors and showed up as
    15% of the search's runtime.
    """
    out = np.empty(np.broadcast_shapes(u.shape, v.shape), dtype=F64)
    out[..., 0] = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    out[..., 1] = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    out[..., 2] = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return out


def _face_axes(a: FloatArray, b: FloatArray) -> FloatArray:
    """The eight face normals of a batched tetrahedron pair, shape ``(m, 8, 3)``."""
    f = TETRA_FACES
    parts = []
    for t in (a, b):
        p0, p1, p2 = t[:, f[:, 0], :], t[:, f[:, 1], :], t[:, f[:, 2], :]
        parts.append(_cross3(p1 - p0, p2 - p0))
    return np.concatenate(parts, axis=1)


def _edge_axes(a: FloatArray, b: FloatArray) -> FloatArray:
    """The 36 edge-pair cross products, shape ``(m, 36, 3)``."""
    i, j = TETRA_EDGES[:, 0], TETRA_EDGES[:, 1]
    ea = a[:, j, :] - a[:, i, :]
    eb = b[:, j, :] - b[:, i, :]
    return _cross3(ea[:, :, None, :], eb[:, None, :, :]).reshape(a.shape[0], 36, 3)


def _min_projection_gap(axes: FloatArray, a: FloatArray, b: FloatArray) -> FloatArray:
    """Smallest projection overlap over the given axes, ``(m,)``.

    Degenerate axes -- parallel edges, zero-area faces -- carry no information
    and are neutralised so they can never *declare* separation.
    """
    norms = np.sqrt(np.einsum("mkd,mkd->mk", axes, axes))
    valid = norms > 1e-12
    axes = axes / np.where(valid, norms, 1.0)[:, :, None]

    proj_a = np.matmul(axes, a.transpose(0, 2, 1))  # (m, K, 4)
    proj_b = np.matmul(axes, b.transpose(0, 2, 1))
    overlap = np.minimum(proj_a.max(axis=2), proj_b.max(axis=2)) - np.maximum(
        proj_a.min(axis=2), proj_b.min(axis=2)
    )
    return np.where(valid, overlap, np.inf).min(axis=1)


def _has_separating_axis(
    axes: FloatArray, a: FloatArray, b: FloatArray, tolerance: float
) -> NDArray[np.bool_]:
    """Whether any of ``axes`` separates each pair.

    The axes are deliberately *not* normalised.  A normalised overlap is below
    ``tolerance`` exactly when the raw overlap is below ``tolerance * ||axis||``,
    so scaling the threshold instead of the axes gives the identical verdict
    while skipping a division across the whole ``(m, K, 3)`` array.
    """
    square = np.einsum("mkd,mkd->mk", axes, axes)
    valid = square > 1e-24

    proj_a = np.matmul(axes, a.transpose(0, 2, 1))  # (m, K, 4)
    proj_b = np.matmul(axes, b.transpose(0, 2, 1))
    overlap = np.minimum(proj_a.max(axis=2), proj_b.max(axis=2)) - np.maximum(
        proj_a.min(axis=2), proj_b.min(axis=2)
    )
    threshold = tolerance * np.sqrt(square) if tolerance else 0.0
    return np.any(valid & (overlap <= threshold), axis=1)


def sat_disjoint(a: FloatArray, b: FloatArray, tolerance: float = 1e-12) -> NDArray[np.bool_]:
    """Whether each batched tetrahedron pair is disjoint, testing in two stages.

    A separating axis proves disjointness, so the 8 face normals are tried
    first and only the pairs that survive them need the 36 edge-pair crosses.
    Most pairs separate on a face normal, so the expensive stage runs on a
    small remainder.  The verdict is identical to thresholding
    :func:`sat_overlap_depth`, since the minimum over all 44 axes is the
    minimum of the two stages.

    Parameters
    ----------
    a, b:
        ``(m, 4, 3)`` float64 batches of tetrahedron vertices.
    tolerance:
        Overlap depths at or below this count as contact rather than overlap.

    Returns
    -------
    ndarray of bool, shape (m,)
    """
    a = np.ascontiguousarray(a, dtype=F64)
    b = np.ascontiguousarray(b, dtype=F64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (4, 3):
        raise ValueError(
            f"a and b must both have shape (m, 4, 3); got {a.shape} and {b.shape}"
        )
    if a.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    disjoint = _has_separating_axis(_face_axes(a, b), a, b, tolerance)
    remaining = ~disjoint
    if np.any(remaining):
        left, right = a[remaining], b[remaining]
        disjoint[remaining] = _has_separating_axis(
            _edge_axes(left, right), left, right, tolerance
        )
    return disjoint


def sat_overlap_depth(a: FloatArray, b: FloatArray) -> FloatArray:
    """Exact overlap depth for batched tetrahedron pairs.

    Uses the separating axis theorem, which is exact for convex polytopes: for
    two convex bodies it suffices to test the face normals of each plus the
    cross products of every pair of edge directions, so ``4 + 4 + 36 = 44``
    axes for tetrahedra.

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

    See Also
    --------
    sat_disjoint : faster when only the yes/no verdict is needed.
    """
    a = np.ascontiguousarray(a, dtype=F64)
    b = np.ascontiguousarray(b, dtype=F64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (4, 3):
        raise ValueError(
            f"a and b must both have shape (m, 4, 3); got {a.shape} and {b.shape}"
        )
    if a.shape[0] == 0:
        return np.zeros(0, dtype=F64)

    axes = np.concatenate([_face_axes(a, b), _edge_axes(a, b)], axis=1)
    return np.asarray(_min_projection_gap(axes, a, b), dtype=F64)


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
# Backend 2: the Chen-Engel-Glotzer double dimer family (exact, analytic)
# --------------------------------------------------------------------------- #
#: Densest known packing fraction of regular tetrahedra, 4000/4671.
CEG_PACKING_FRACTION: float = 4000.0 / 4671.0

#: Vertices of the positive dimer +F2 in the paper's coordinates (Definition 1).
#: ``p, q, r`` span the face shared by the two tetrahedra; the tetrahedron edge
#: is ``3 * sqrt(2)`` and each tetrahedron has volume 9.
CEG_DIMER_VERTICES: dict[str, tuple[int, int, int]] = {
    "o": (2, 2, 2),
    "p": (2, -1, -1),
    "q": (-1, 2, -1),
    "r": (-1, -1, 2),
    "s": (-2, -2, -2),
}

#: Named points of the three-parameter family, as exact ``(u, v, w)`` rationals.
#:
#: ``optimal`` / ``optimal-mirror``
#:     The two maximal-density points, phi = 4000/4671 (Theorem 1).  They are
#:     related by the crystallographic isometry T of eq. (14).
#: ``kallus-elser-gravel``
#:     The origin, phi = 100/117.  The whole line ``(0, 0, w)`` is the
#:     one-parameter family of Kallus, Elser & Gravel.
#: ``torquato-jiao``
#:     Densest point of the Torquato-Jiao two-parameter family (the plane
#:     ``5u = -2v``), phi = 12250/14319.
#: ``densest-connected``
#:     The densest member whose unit-distance graph is *connected*, phi =
#:     125/146.  Inter-dimer contacts exist only on the central plane ``u = 0``
#:     -- exactly where the vertex-to-edge incidence condition ``H_{a-b}`` of
#:     eq. (10) holds -- so maximising density subject to connectivity means
#:     maximising ``|v|`` on that plane.  Adding the two binding constraints of
#:     ``P''``, ``2v - w <= 33/320`` and ``v + w <= 3/64``, gives ``v <= 1/20``.
#:     This point is the paper's C3+cen entry.
CEG_FAMILY_PRESETS: dict[str, tuple[Fraction, Fraction, Fraction]] = {
    "optimal": (Fraction(3, 160), Fraction(3, 64), Fraction(0)),
    "optimal-mirror": (Fraction(-3, 160), Fraction(-3, 64), Fraction(0)),
    "kallus-elser-gravel": (Fraction(0), Fraction(0), Fraction(0)),
    "torquato-jiao": (Fraction(3, 140), Fraction(-3, 56), Fraction(-3, 448)),
    "densest-connected": (Fraction(0), Fraction(1, 20), Fraction(-1, 320)),
}

#: Exact packing fraction at each preset, for cross-checking the construction.
CEG_PRESET_FRACTIONS: dict[str, Fraction] = {
    "optimal": Fraction(4000, 4671),
    "optimal-mirror": Fraction(4000, 4671),
    "kallus-elser-gravel": Fraction(100, 117),
    "torquato-jiao": Fraction(12250, 14319),
    "densest-connected": Fraction(125, 146),
}


def ceg_family_vectors(
    u: Fraction, v: Fraction, w: Fraction
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Lattice vectors ``a, b, c`` and offset ``d`` of the double dimer family.

    This is equation (6) of Chen, Engel & Glotzer: the three-parameter linear
    space of double dimer configurations that satisfy the nine linear incidence
    conditions of their Lemma 2.  Coordinates are the paper's own, in which the
    tetrahedron edge is ``3 * sqrt(2)``.

    Parameters
    ----------
    u, v, w:
        Family parameters.  Passing :class:`~fractions.Fraction` keeps the
        construction exact up to the final float conversion.

    Returns
    -------
    (a, b, c, d)
        Four ``(3,)`` float64 vectors.
    """
    half = Fraction(1, 2)
    rows = (
        (Fraction(27, 10) + u, Fraction(21, 20) - v, Fraction(-3, 20) + 2 * u + v),
        (Fraction(-3, 10) - u, Fraction(51, 20) + v, Fraction(27, 20) - 2 * u - v),
        (
            Fraction(129, 160) - u + 2 * v + 2 * w,
            Fraction(-237, 320) + half * u - v + 3 * w,
            Fraction(753, 320) + half * u - v + w,
        ),
        (Fraction(1, 10) + u, Fraction(-1, 20) + u + v, Fraction(-1, 20) + u - v),
    )
    return tuple(  # type: ignore[return-value]
        np.array([float(component) for component in row], dtype=F64) for row in rows
    )


def ceg_packing_fraction(u: Fraction, v: Fraction, w: Fraction) -> Fraction:
    """Exact packing fraction of the family member at ``(u, v, w)``.

    From equation (11), ``V = (9/25)(117 + 60u^2 - 80uv - 80v^2)`` and
    ``phi = 2U/V = 36/V``, giving

    ``phi = 100 / (117 + 60u^2 - 80uv - 80v^2)``.

    The result is independent of ``w``: that direction is a pure lattice shear,
    so the whole line ``(u, v, w)`` has constant density.  The quadratic form is
    a hyperbolic paraboloid with a saddle at the origin, which is why the
    extrema are attained on the boundary of the restricted space rather than at
    an interior stationary point.
    """
    denominator = Fraction(117) + 60 * u * u - 80 * u * v - 80 * v * v
    if denominator <= 0:
        raise ValueError(f"degenerate family parameters: V <= 0 at ({u}, {v}, {w})")
    return Fraction(100) / denominator


def ceg_in_restricted_space(u: Fraction, v: Fraction, w: Fraction) -> bool:
    """Whether ``(u, v, w)`` lies in the restricted space ``P''`` of Lemma 3.

    ``P''`` is the intersection of the four half-spaces of equation (9), each
    coming from an edge-to-edge or vertex-to-face incidence condition:

    * ``H_{b+c}``:  ``+u/2 + 2v - w <= 33/320``
    * ``H_{c+a}``:  ``-u/2 - 2v - w <= 33/320``
    * ``H_{b-c}``:  ``-v + w <= 3/64``
    * ``H_{c-a}``:  ``+v + w <= 3/64``

    Every configuration in ``P''`` is proved to be a genuine packing.  Points
    outside it are not necessarily overlapping -- the conditions are sufficient,
    not necessary -- so :func:`ceg_unit_cell` runs the separating-axis test
    regardless rather than relying on this predicate.
    """
    half = Fraction(1, 2)
    limit_a, limit_b = Fraction(33, 320), Fraction(3, 64)
    return (
        half * u + 2 * v - w <= limit_a
        and -half * u - 2 * v - w <= limit_a
        and -v + w <= limit_b
        and v + w <= limit_b
    )


def ceg_unit_cell(
    u: Fraction | None = None,
    v: Fraction | None = None,
    w: Fraction | None = None,
    *,
    variant: str = "optimal",
) -> tuple[FloatArray, FloatArray]:
    """Return the four tetrahedra and lattice of a CEG double dimer packing.

    A *dimer* is two regular tetrahedra sharing a face, forming a triangular
    dipyramid; the positive dimer ``+F2`` has vertices ``o, p, q, r, s`` with
    ``p, q, r`` spanning the shared face, and the negative dimer ``-F2`` is its
    inversion.  Positive dimers occupy the even sublattice

    ``L+ = {n_a a + n_b b + n_c c : n_a + n_b + n_c = 0 mod 2}``

    spanned by ``a + b``, ``b + c``, ``c + a``; negative dimers occupy the odd
    coset ``L- = L+ + (d + a)``.  One unit cell therefore holds one positive and
    one negative dimer -- four tetrahedra of total volume ``2U = 36`` -- giving
    ``phi = 36/V``.

    Coordinates are rescaled from the paper's edge length of ``3 * sqrt(2)`` to
    the unit edge used throughout this module.

    Parameters
    ----------
    u, v, w:
        Explicit family parameters.  Supply all three, or none to use
        ``variant``.
    variant:
        Name from :data:`CEG_FAMILY_PRESETS`.  Ignored when ``u, v, w`` are
        given.

    Returns
    -------
    (tetrahedra, lattice)
        ``(4, 4, 3)`` vertex coordinates and the ``(3, 3)`` lattice spanning
        ``L+``.
    """
    supplied = [p for p in (u, v, w) if p is not None]
    if supplied and len(supplied) != 3:
        raise ValueError("supply all three of u, v, w, or none of them")
    if not supplied:
        if variant not in CEG_FAMILY_PRESETS:
            raise ValueError(
                f"unknown variant {variant!r}; expected one of "
                f"{tuple(CEG_FAMILY_PRESETS)}"
            )
        u, v, w = CEG_FAMILY_PRESETS[variant]

    params = tuple(Fraction(p) for p in (u, v, w))  # type: ignore[arg-type]
    a, b, c, d = ceg_family_vectors(*params)
    vertices = {
        name: np.array(coords, dtype=F64) for name, coords in CEG_DIMER_VERTICES.items()
    }
    positive = np.array(
        [
            [vertices["o"], vertices["p"], vertices["q"], vertices["r"]],
            [vertices["s"], vertices["p"], vertices["q"], vertices["r"]],
        ],
        dtype=F64,
    )
    negative = -positive + (d + a)

    scale = 1.0 / (3.0 * math.sqrt(2.0))
    tetrahedra = np.concatenate([positive, negative]) * scale
    lattice = np.array([a + b, b + c, c + a], dtype=F64) * scale
    return np.ascontiguousarray(tetrahedra), np.ascontiguousarray(lattice)


def ceg_exact_vertices(
    u: Fraction, v: Fraction, w: Fraction, reps: int = 2
) -> list[list[Fraction]]:
    """Vertices of a tiled CEG packing in *exact rational* paper coordinates.

    Coordinates are left in the paper's frame, where the tetrahedron edge is
    ``3 * sqrt(2)`` and every lattice, offset and dimer component is rational.
    The unit-edge rescaling used elsewhere multiplies everything by a single
    irrational factor, which is a global similarity and cannot change the rank
    of the Z-module the coordinates generate -- so the rational frame is both
    equivalent and exactly representable.

    Parameters
    ----------
    u, v, w:
        Family parameters.
    reps:
        Tiling repetitions per lattice direction.

    Returns
    -------
    list of [Fraction, Fraction, Fraction]
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")

    half = Fraction(1, 2)
    a = (Fraction(27, 10) + u, Fraction(21, 20) - v, Fraction(-3, 20) + 2 * u + v)
    b = (Fraction(-3, 10) - u, Fraction(51, 20) + v, Fraction(27, 20) - 2 * u - v)
    c = (
        Fraction(129, 160) - u + 2 * v + 2 * w,
        Fraction(-237, 320) + half * u - v + 3 * w,
        Fraction(753, 320) + half * u - v + w,
    )
    d = (Fraction(1, 10) + u, Fraction(-1, 20) + u + v, Fraction(-1, 20) + u - v)

    def add(p, q):
        return tuple(x + y for x, y in zip(p, q))

    def scale(k, p):
        return tuple(Fraction(k) * x for x in p)

    positive = [
        [Fraction(x) for x in CEG_DIMER_VERTICES[name]]
        for name in ("o", "p", "q", "r", "s")
    ]
    shift = add(d, a)
    negative = [[shift[i] - vertex[i] for i in range(3)] for vertex in positive]

    lattice = (add(a, b), add(b, c), add(c, a))

    vertices: list[list[Fraction]] = []
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                offset = add(add(scale(i, lattice[0]), scale(j, lattice[1])), scale(k, lattice[2]))
                for vertex in positive + negative:
                    vertices.append([vertex[n] + offset[n] for n in range(3)])
    return vertices


def build_ceg_packing(
    min_tetrahedra: int = 1000,
    *,
    variant: str = "optimal",
    u: Fraction | None = None,
    v: Fraction | None = None,
    w: Fraction | None = None,
    max_replicas: int = 40,
) -> TetraCloud:
    """Build a Chen-Engel-Glotzer double dimer packing and certify it.

    The default ``variant="optimal"`` is the densest packing of regular
    tetrahedra known, ``phi = 4000/4671 ~ 0.856347`` -- an exact analytic
    construction, not a stochastic search result.

    Every cell is verified rather than trusted: the tetrahedra are checked to be
    regular with unit edge, the geometric packing fraction is checked against the
    closed form of :func:`ceg_packing_fraction`, and the separating-axis test is
    run over all periodic images to certify that nothing interpenetrates.

    Parameters
    ----------
    min_tetrahedra:
        Minimum tetrahedron count in the returned cloud.
    variant:
        Name from :data:`CEG_FAMILY_PRESETS`.
    u, v, w:
        Explicit family parameters, overriding ``variant`` when all are given.
    max_replicas:
        Safety bound on tiling repetitions per lattice direction.

    Returns
    -------
    TetraCloud
    """
    if min_tetrahedra < 1:
        raise ValueError("min_tetrahedra must be >= 1")

    if u is None and v is None and w is None:
        if variant not in CEG_FAMILY_PRESETS:
            raise ValueError(
                f"unknown variant {variant!r}; expected one of "
                f"{tuple(CEG_FAMILY_PRESETS)}"
            )
        params = CEG_FAMILY_PRESETS[variant]
        label = variant
    else:
        if None in (u, v, w):
            raise ValueError("supply all three of u, v, w, or none of them")
        params = (Fraction(u), Fraction(v), Fraction(w))  # type: ignore[arg-type]
        label = "custom"

    cell, lattice = ceg_unit_cell(*params)
    per_cell = int(cell.shape[0])

    volume = abs(float(np.linalg.det(lattice)))
    phi = per_cell * UNIT_TETRA_VOLUME / volume
    expected = float(ceg_packing_fraction(*params))
    if abs(phi - expected) > 1e-12:
        raise AssertionError(
            f"CEG cell reproduces phi={phi:.15f}, closed form gives {expected:.15f}"
        )
    if not _configuration_is_valid(cell, lattice):
        raise AssertionError(
            f"CEG cell at (u, v, w) = {params} failed the separating-axis overlap "
            "test; the point lies outside the region where the construction packs"
        )

    reps = 1
    while reps <= max_replicas and per_cell * reps**3 < min_tetrahedra:
        reps += 1
    if per_cell * reps**3 < min_tetrahedra:
        raise RuntimeError(
            f"cannot reach {min_tetrahedra} tetrahedra within max_replicas={max_replicas}"
        )

    grid = np.arange(reps, dtype=F64)
    i, j, k = np.meshgrid(grid, grid, grid, indexing="ij")
    translations = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) @ lattice
    tiled = (cell[None, :, :, :] + translations[:, None, None, :]).reshape(-1, 4, 3)

    cloud = TetraCloud(
        tetrahedra=tiled,
        lattice=lattice,
        tetra_per_cell=per_cell,
        provenance={
            "backend": "ceg",
            "source": (
                "Chen, Engel & Glotzer, Discrete Comput. Geom. 44, 253 (2010); "
                "double dimer family eq. (6), optimum Theorem 1"
            ),
            "variant": label,
            "family_parameters_uvw": tuple(str(p) for p in params),
            "packing_fraction": phi,
            "packing_fraction_exact": str(ceg_packing_fraction(*params)),
            "in_restricted_space": ceg_in_restricted_space(*params),
            "unit_cell_volume": volume,
            "structure": "double dimer lattice, 1 positive + 1 negative dimer per cell",
            "space_group": "P-1",
            "tetrahedra_per_cell": per_cell,
            "replicas_per_axis": reps,
            "overlap_certified": True,
        },
    )
    cloud.assert_regular(edge=1.0, atol=1e-12)
    LOGGER.info(
        "CEG packing (%s): phi=%.12f = %s, %d tetrahedra from %d^3 cells",
        label,
        phi,
        ceg_packing_fraction(*params),
        cloud.n_tetrahedra,
        reps,
    )
    return cloud


# --------------------------------------------------------------------------- #
# Backend 3: three-fold screw-symmetric packings (the N = 3 phase)
# --------------------------------------------------------------------------- #
#: Analytical density of the N = 3 phase, Table I of Chen, Engel & Glotzer.
N3_TARGET_FRACTION: Fraction = Fraction(2, 3)

#: The 120 degree rotation about z shared by every three-fold construction.
_ROT120: FloatArray = np.array(
    [
        [math.cos(2.0 * math.pi / 3.0), -math.sin(2.0 * math.pi / 3.0), 0.0],
        [math.sin(2.0 * math.pi / 3.0), math.cos(2.0 * math.pi / 3.0), 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=F64,
)

#: Screw indices: 0 gives the point group P3, 1 and 2 the screw groups P3_1
#: and P3_2.  The generator rotates by 120 degrees about z and translates by
#: ``index * c / 3`` along it, so its cube is always a lattice translation.
P3_SCREW_INDICES: tuple[int, ...] = (0, 1, 2)


def p3_cell(
    a: float,
    c: float,
    fx: float,
    fy: float,
    quat: FloatArray,
    screw: int,
) -> tuple[FloatArray, FloatArray]:
    """Three tetrahedra related by a three-fold screw, and their lattice.

    A three-fold rotation must map the lattice to itself, which forces a
    hexagonal cell: two basal vectors of equal length ``a`` at 120 degrees, and
    ``c`` along the rotation axis.  The generator is

    ``S(p) = R_z(120 deg) p + (0, 0, screw * c / 3)``

    so that ``S^3`` is translation by ``screw * c``, a lattice vector, and the
    three tetrahedra of the cell are ``T, S(T), S^2(T)``.

    This is Table I's N = 3 motif -- "3 monomers, three-fold symmetric" -- and
    reduces the packing problem from four free rigid bodies to seven numbers:
    ``a``, ``c``, the in-plane offset ``(x, y)``, and three orientation degrees
    of freedom.  Translation along the axis is a gauge freedom (it can be
    absorbed into the choice of screw origin) and is fixed to zero.

    Parameters
    ----------
    a, c:
        Hexagonal cell parameters.  Both must be positive.
    fx, fy:
        In-plane offset of the generating tetrahedron's centroid, in
        *fractional* coordinates along the two basal vectors.  Fractional
        rather than Cartesian because the three tetrahedra are separated by
        their distance from the rotation axis, which scales with ``a``: with a
        Cartesian offset, growing the cell would leave them overlapping on the
        axis no matter how large it became.
    quat:
        Unit quaternion giving the generating tetrahedron's orientation.
    screw:
        0, 1 or 2, selecting P3, P3_1 or P3_2.

    Returns
    -------
    (tetrahedra, lattice)
        ``(3, 4, 3)`` vertex coordinates and the ``(3, 3)`` hexagonal lattice.
    """
    if not (a > 0.0 and c > 0.0):
        raise ValueError(f"cell parameters must be positive, got a={a}, c={c}")
    if screw not in P3_SCREW_INDICES:
        raise ValueError(f"screw must be one of {P3_SCREW_INDICES}, got {screw}")

    lattice = np.array(
        [
            [a, 0.0, 0.0],
            [-0.5 * a, 0.5 * math.sqrt(3.0) * a, 0.0],
            [0.0, 0.0, c],
        ],
        dtype=F64,
    )

    rotation = _ROT120
    translation = np.array([0.0, 0.0, screw * c / 3.0], dtype=F64)

    base = (_quat_to_matrix(np.asarray(quat, dtype=F64)[None])[0] @ regular_tetrahedron(1.0).T).T
    base = base + fx * lattice[0] + fy * lattice[1]

    tetrahedra = np.empty((3, 4, 3), dtype=F64)
    current = base
    for index in range(3):
        tetrahedra[index] = current
        current = (rotation @ current.T).T + translation

    # Each image may be moved back by a whole lattice vector without changing
    # the periodic packing; keeping every tetrahedron near the cell is what
    # makes the periodic overlap test's image range valid.
    inverse = np.linalg.inv(lattice)
    centroids = tetrahedra.mean(axis=1)
    tetrahedra = tetrahedra - (np.floor(centroids @ inverse) @ lattice)[:, None, :]
    return np.ascontiguousarray(tetrahedra), np.ascontiguousarray(lattice)


#: Largest cluster radius the trimer search will consider.  A trimer wider than
#: this is not a compact motif, and its periodic images interleave so heavily
#: that the configuration cannot be certified.
MAX_TRIMER_RADIUS: float = 2.0


#: Height of a unit tetrahedron measured along its own three-fold axis.
C3_TETRA_HEIGHT: float = math.sqrt(2.0 / 3.0)

#: Circumradius of the base triangle when the three-fold axis is vertical.
C3_TETRA_BASE_RADIUS: float = 1.0 / math.sqrt(3.0)

#: Fractional positions of the three distinct three-fold axes of a hexagonal
#: cell -- Wyckoff sites 1a, 1b and 1c of space group P3.
P3_AXIS_SITES: tuple[tuple[Fraction, Fraction], ...] = (
    (Fraction(0), Fraction(0)),
    (Fraction(1, 3), Fraction(2, 3)),
    (Fraction(2, 3), Fraction(1, 3)),
)


def c3_aligned_tetrahedron(azimuth: float, sign: int) -> FloatArray:
    """A unit tetrahedron whose own three-fold axis is the z axis.

    Such a tetrahedron is mapped to *itself* by a 120 degree rotation about z,
    which is what lets three separate monomers each sit on their own axis and
    still leave the crystal three-fold symmetric.

    Parameters
    ----------
    azimuth:
        Rotation about the axis.  The body has C3 symmetry, so only the range
        ``[0, 2*pi/3)`` is distinct.
    sign:
        ``+1`` points the apex along ``+z``, ``-1`` along ``-z``.

    Returns
    -------
    ndarray, shape (4, 3)
        Vertices, centroid at the origin.
    """
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign!r}")

    height, radius = C3_TETRA_HEIGHT, C3_TETRA_BASE_RADIUS
    vertices = np.empty((4, 3), dtype=F64)
    vertices[0] = (0.0, 0.0, 0.75 * height)
    for k in range(3):
        angle = azimuth + 2.0 * math.pi * k / 3.0
        vertices[k + 1] = (
            radius * math.cos(angle),
            radius * math.sin(angle),
            -0.25 * height,
        )
    vertices[:, 2] *= sign
    return vertices


def p3_columns_cell(
    a: float,
    c: float,
    heights: FloatArray,
    azimuths: FloatArray,
    signs: tuple[int, int, int],
) -> tuple[FloatArray, FloatArray]:
    """Three monomers, one on each three-fold axis of a hexagonal cell.

    This is the reading of Table I's "3 monomers, three-fold symmetric" in which
    the three-fold rotation maps every tetrahedron to *itself* rather than
    permuting them.  It is disjoint from :func:`p3_cell`, where the three form a
    single orbit, and it is the only remaining possibility once two others are
    ruled out by volume arithmetic: all three monomers on one shared axis needs
    a cell too narrow for the columns to clear each other, and a rhombohedral
    R-centred cell reduces to a lattice packing, capped at 18/49.

    Parameters
    ----------
    a, c:
        Hexagonal cell parameters.
    heights:
        Axial offset of each monomer, in fractions of ``c``.  A common shift is
        a global translation, so the first is conventionally held at zero.
    azimuths:
        Rotation of each monomer about its own axis.
    signs:
        Apex direction of each monomer, ``+1`` or ``-1``.

    Returns
    -------
    (tetrahedra, lattice)
        ``(3, 4, 3)`` vertex coordinates and the ``(3, 3)`` hexagonal lattice.
    """
    if not (a > 0.0 and c > 0.0):
        raise ValueError(f"cell parameters must be positive, got a={a}, c={c}")

    lattice = np.array(
        [
            [a, 0.0, 0.0],
            [-0.5 * a, 0.5 * math.sqrt(3.0) * a, 0.0],
            [0.0, 0.0, c],
        ],
        dtype=F64,
    )

    tetrahedra = np.empty((3, 4, 3), dtype=F64)
    for index, (fx, fy) in enumerate(P3_AXIS_SITES):
        body = c3_aligned_tetrahedron(float(azimuths[index]), signs[index])
        origin = float(fx) * lattice[0] + float(fy) * lattice[1]
        origin = origin + float(heights[index]) * lattice[2]
        tetrahedra[index] = body + origin

    inverse = np.linalg.inv(lattice)
    centroids = tetrahedra.mean(axis=1)
    tetrahedra = tetrahedra - (np.floor(centroids @ inverse) @ lattice)[:, None, :]
    return np.ascontiguousarray(tetrahedra), np.ascontiguousarray(lattice)


@dataclass(frozen=True)
class P3ColumnsResult:
    """Outcome of a three-axis monomer search."""

    a: float
    c: float
    heights: FloatArray
    azimuths: FloatArray
    signs: tuple[int, int, int]
    packing_fraction: float

    @property
    def cell(self) -> tuple[FloatArray, FloatArray]:
        return p3_columns_cell(self.a, self.c, self.heights, self.azimuths, self.signs)


def _p3_columns_search(
    cycles: int,
    seed: int,
    signs: tuple[int, int, int],
    *,
    initial_fraction: float = 0.15,
    compression: float = 0.004,
) -> P3ColumnsResult:
    """Monte Carlo over the seven free numbers of the three-axis ansatz.

    Free: the cell ``(a, c)``, two relative axial offsets (the third is a global
    translation), and three azimuths.  The apex directions are discrete and
    enumerated by the caller.
    """
    rng = np.random.default_rng(seed)

    volume = 3.0 * UNIT_TETRA_VOLUME / initial_fraction
    a = (2.0 * volume / math.sqrt(3.0)) ** (1.0 / 3.0)
    c = volume / (0.5 * math.sqrt(3.0) * a * a)
    heights = np.array([0.0, rng.random(), rng.random()], dtype=F64)
    azimuths = rng.random(3).astype(F64) * (2.0 * math.pi / 3.0)

    for _ in range(80):
        verts, lattice = p3_columns_cell(a, c, heights, azimuths, signs)
        if _configuration_is_valid(verts, lattice):
            break
        a, c = a * 1.12, c * 1.12
    else:
        raise RuntimeError("failed to find a valid initial three-axis configuration")

    height_step = _AdaptiveStep(0.05, bounds=(1e-7, 0.5))
    azimuth_step = _AdaptiveStep(0.2, bounds=(1e-6, 2.0 * math.pi / 3.0))
    cell_step = _AdaptiveStep(0.02, bounds=(1e-6, 0.2))

    for _ in range(cycles):
        for _ in range(3):
            index = int(rng.integers(3))
            old_h, old_az = heights[index], azimuths[index]
            if rng.random() < 0.5 and index > 0:
                heights[index] = (old_h + float(rng.normal(scale=height_step.value))) % 1.0
                step = height_step
            else:
                azimuths[index] = old_az + float(rng.normal(scale=azimuth_step.value))
                step = azimuth_step
            verts, lattice = p3_columns_cell(a, c, heights, azimuths, signs)
            ok = _configuration_is_valid(verts, lattice)
            if not ok:
                heights[index], azimuths[index] = old_h, old_az
            step.record(ok)

        for _ in range(3):
            ratio = math.exp(rng.normal(scale=cell_step.value))
            shrink = (1.0 - compression) ** (1.0 / 3.0)
            trial_a = a * ratio * shrink
            trial_c = c * shrink**3 / ratio**2
            ok = trial_a > 1e-6 and trial_c > 1e-6
            if ok:
                verts, lattice = p3_columns_cell(trial_a, trial_c, heights, azimuths, signs)
                ok = _configuration_is_valid(verts, lattice)
            if ok:
                a, c = trial_a, trial_c
            cell_step.record(ok)

    return P3ColumnsResult(
        a=a, c=c, heights=heights.copy(), azimuths=azimuths.copy(), signs=signs,
        packing_fraction=p3_packing_fraction(a, c),
    )


#: Exact density of the N = 3 phase, Chen, Engel & Glotzer Table I.
N3_PACKING_FRACTION: Fraction = Fraction(2, 3)

#: Exact hexagonal cell of the N = 3 phase.  ``c`` is the tetrahedron's own
#: height along its three-fold axis, so each column is exactly one tetrahedron
#: tall; ``a`` is the height of a unit equilateral triangle.  Together they give
#: ``V = (sqrt3/2) a^2 c = 3 sqrt2 / 8``, and ``phi = 3 V_tet / V = 2/3`` exactly.
N3_CELL_A: float = math.sqrt(3.0) / 2.0
N3_CELL_C: float = math.sqrt(2.0 / 3.0)

#: Axial offsets, azimuths and apex directions of the three monomers.  One of
#: 24 symmetry-equivalent solutions; the others differ by relabelling the axes,
#: shifting z, or reflecting.
N3_HEIGHTS: tuple[Fraction, Fraction, Fraction] = (
    Fraction(0), Fraction(1, 2), Fraction(0),
)
N3_AZIMUTH_THIRDS: tuple[int, int, int] = (0, 1, 1)  # multiples of pi/3
N3_SIGNS: tuple[int, int, int] = (1, 1, -1)


def n3_unit_cell() -> tuple[FloatArray, FloatArray]:
    """The three tetrahedra and lattice of the N = 3 phase, ``phi = 2/3``.

    Table I of Chen, Engel & Glotzer records an N = 3 phase at an analytical
    density of exactly 2/3, described as "3 monomers, three-fold symmetric", but
    does not publish its coordinates and the referenced data file is no longer
    online.  This is that structure, recovered by searching the ansatz in which
    the three-fold rotation maps each monomer to *itself*: one tetrahedron on
    each of the hexagonal cell's three distinct three-fold axes, its own C3 axis
    aligned with the crystal's.

    Two other readings are excluded by volume arithmetic alone.  All three
    monomers on a single shared axis needs ``c >= 3 * 0.8165``, forcing
    ``a <= 0.5`` and putting the columns 0.29 apart against a base radius of
    0.577.  A rhombohedral R-centred cell puts one tetrahedron in the primitive
    cell, which is a lattice packing and capped at 18/49.

    Returns
    -------
    (tetrahedra, lattice)
        ``(3, 4, 3)`` vertex coordinates and the ``(3, 3)`` hexagonal lattice.
    """
    heights = np.array([float(h) for h in N3_HEIGHTS], dtype=F64)
    azimuths = np.array(
        [k * math.pi / 3.0 for k in N3_AZIMUTH_THIRDS], dtype=F64
    )
    return p3_columns_cell(N3_CELL_A, N3_CELL_C, heights, azimuths, N3_SIGNS)


def build_n3_packing(min_tetrahedra: int = 1000, *, max_replicas: int = 40) -> TetraCloud:
    """Build the N = 3 phase at exactly ``phi = 2/3`` and certify it.

    The cell is verified on construction rather than trusted: the tetrahedra are
    checked regular with unit edge, the density is checked against 2/3 exactly,
    and the separating-axis test is run over all periodic images.

    Parameters
    ----------
    min_tetrahedra:
        Minimum tetrahedron count in the returned cloud.
    max_replicas:
        Safety bound on tiling repetitions per lattice direction.

    Returns
    -------
    TetraCloud
    """
    if min_tetrahedra < 1:
        raise ValueError("min_tetrahedra must be >= 1")

    cell, lattice = n3_unit_cell()
    per_cell = int(cell.shape[0])

    volume = abs(float(np.linalg.det(lattice)))
    phi = per_cell * UNIT_TETRA_VOLUME / volume
    if abs(phi - float(N3_PACKING_FRACTION)) > 1e-14:
        raise AssertionError(f"N=3 cell gives phi={phi:.15f}, expected 2/3")
    if not _configuration_is_valid(cell, lattice, tolerance=1e-9):
        raise AssertionError("N=3 unit cell failed the separating-axis overlap test")

    reps = 1
    while reps <= max_replicas and per_cell * reps**3 < min_tetrahedra:
        reps += 1
    if per_cell * reps**3 < min_tetrahedra:
        raise RuntimeError(
            f"cannot reach {min_tetrahedra} tetrahedra within max_replicas={max_replicas}"
        )

    grid = np.arange(reps, dtype=F64)
    i, j, k = np.meshgrid(grid, grid, grid, indexing="ij")
    translations = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) @ lattice
    tiled = (cell[None, :, :, :] + translations[:, None, None, :]).reshape(-1, 4, 3)

    overlaps = find_overlapping_pairs(tiled, tolerance=1e-9)
    if overlaps.shape[0]:
        raise AssertionError(
            f"N=3 packing has {overlaps.shape[0]} overlapping pairs after tiling"
        )

    cloud = TetraCloud(
        tetrahedra=tiled,
        lattice=lattice,
        tetra_per_cell=per_cell,
        provenance={
            "backend": "n3",
            "target": "Chen, Engel & Glotzer Table I, N = 3 phase",
            "packing_fraction": phi,
            "packing_fraction_exact": "2/3",
            "cell_a": "sqrt(3)/2",
            "cell_c": "sqrt(2/3)  (the tetrahedron's own height)",
            "unit_cell_volume": volume,
            "structure": (
                "3 monomers, one on each three-fold axis (Wyckoff 1a/1b/1c); "
                "each mapped to itself by the rotation"
            ),
            "space_group": "P3",
            "tetrahedra_per_cell": per_cell,
            "replicas_per_axis": reps,
            "overlap_certified": True,
            "note": "recovered by search; the published coordinates are unavailable",
        },
    )
    cloud.assert_regular(edge=1.0, atol=1e-12)
    LOGGER.info(
        "N=3 packing: phi=%.15f (2/3), %d tetrahedra from %d^3 cells",
        phi, cloud.n_tetrahedra, reps,
    )
    return cloud


def build_n3_cluster(
    radius: float = 4.0, layers: int = 5, *, max_cells: int = 40
) -> TetraCloud:
    """A cluster of the N = 3 phase that keeps its three-fold symmetry.

    Selection is by each *tetrahedron's* distance from the Wyckoff 1a axis, not
    by whole unit cells.  That distinction matters: rotation carries the 1b
    monomer of cell ``(0, 0)`` into cell ``(-1, -1)``, so a cell-based cut is
    not invariant even though the cell *positions* are.  Filtering individual
    tetrahedra radially is exactly invariant, because the rotation preserves
    distance from its own axis.

    This is the analogue of the spherical cut used for the FCC honeycomb, where
    a ball is invariant under :math:`O_h`.  Cutting a symmetric structure with
    an asymmetric boundary destroys precisely the degeneracies one is looking
    for.

    An odd ``layers`` is preferable: with an even count the sixteen networks
    pair up into isomorphic partners and every multiplicity doubles, which is a
    property of the finite cluster rather than of the crystal.

    Parameters
    ----------
    radius:
        Basal cut-off from the three-fold axis.
    layers:
        Number of cells stacked along the axis.
    max_cells:
        Safety bound on the basal cell range searched.

    Returns
    -------
    TetraCloud
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius!r}")
    if layers < 1:
        raise ValueError(f"layers must be >= 1, got {layers!r}")

    cell, lattice = n3_unit_cell()
    span = min(int(math.ceil(radius / min(N3_CELL_A, N3_CELL_C))) + 2, max_cells)

    grid = [
        (i, j, k)
        for i in range(-span, span + 1)
        for j in range(-span, span + 1)
        for k in range(layers)
    ]
    translations = np.array(grid, dtype=F64) @ lattice
    tiled = (cell[None, :, :, :] + translations[:, None, None, :]).reshape(-1, 4, 3)

    centroids = tiled.mean(axis=1)
    keep = np.hypot(centroids[:, 0], centroids[:, 1]) <= radius + 1e-9
    tetrahedra = tiled[keep]
    if tetrahedra.shape[0] == 0:
        raise ValueError(f"radius {radius} selects no tetrahedra")

    cloud = TetraCloud(
        tetrahedra=tetrahedra,
        lattice=lattice,
        tetra_per_cell=int(cell.shape[0]),
        provenance={
            "backend": "n3-cluster",
            "packing_fraction_exact": "2/3",
            "cut": "tetrahedra within `radius` of the Wyckoff 1a three-fold axis",
            "point_group": "C3",
            "radius": float(radius),
            "layers": int(layers),
            "layer_parity": "odd" if layers % 2 else "even (multiplicities double)",
        },
    )
    cloud.assert_regular(edge=1.0, atol=1e-12)
    return cloud


#: Analytical density of the N = 2 phase, Chen, Engel & Glotzer Table I:
#: ``phi_2 = 9 / (139 - 40 sqrt(10))``.
N2_PACKING_FRACTION: float = 9.0 / (139.0 - 40.0 * math.sqrt(10.0))


def double_lattice_cell(
    lattice: FloatArray, offset: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Two tetrahedra related by a point inversion, on a free lattice.

    Kuperberg's *double lattice*: the packing is the union of ``T + L`` and
    ``-T + d + L``.  Inversion through ``d/2`` exchanges the two, so the space
    group acts transitively on the tetrahedra -- which is exactly how Table I
    describes the N = 2 phase, "2 monomers, transitive".

    The generating tetrahedron is held in its canonical orientation; a general
    ``3 x 3`` lattice can present any orientation relative to it, so nothing is
    lost and three redundant rotational degrees of freedom are avoided.

    Parameters
    ----------
    lattice:
        ``(3, 3)`` lattice with vectors as rows.
    offset:
        ``(3,)`` translation applied to the inverted copy.

    Returns
    -------
    (tetrahedra, lattice)
        ``(2, 4, 3)`` vertex coordinates and the lattice, unchanged.
    """
    base = regular_tetrahedron(1.0)
    shift = np.asarray(offset, dtype=F64)
    cell = np.stack([base, -base + shift])

    inverse = np.linalg.inv(lattice)
    centroids = cell.mean(axis=1)
    cell = cell - (np.floor(centroids @ inverse) @ lattice)[:, None, :]
    return np.ascontiguousarray(cell), np.ascontiguousarray(lattice, dtype=F64)


@dataclass(frozen=True)
class DoubleLatticeResult:
    """Outcome of a monomer double-lattice search."""

    lattice: FloatArray
    offset: FloatArray
    packing_fraction: float

    @property
    def cell(self) -> tuple[FloatArray, FloatArray]:
        return double_lattice_cell(self.lattice, self.offset)


def _double_lattice_search(
    cycles: int,
    seed: int,
    *,
    initial_fraction: float = 0.15,
    compression: float = 0.004,
) -> DoubleLatticeResult:
    """Monte Carlo over the twelve numbers of the monomer double lattice."""
    rng = np.random.default_rng(seed)

    # The offset is a structural parameter, not a fraction of the cell: if the
    # two tetrahedra overlap each other, no amount of lattice expansion
    # separates them.  Resolve the pair first, then the periodic images.
    offset = rng.normal(scale=0.5, size=3).astype(F64)
    if np.linalg.norm(offset) < 1e-6:
        offset = np.array([1.0, 0.0, 0.0], dtype=F64)
    base = regular_tetrahedron(1.0)
    for _ in range(80):
        pair = np.stack([base, -base + offset])
        if sat_overlap_depth(pair[0:1], pair[1:2])[0] <= 1e-12:
            break
        offset = offset * 1.15
    else:
        raise RuntimeError("failed to separate the two tetrahedra of the double lattice")

    lattice = _initial_lattice(2, initial_fraction, rng)
    for _ in range(80):
        if _configuration_is_valid(*double_lattice_cell(lattice, offset)):
            break
        lattice = lattice * 1.12
    else:
        raise RuntimeError("failed to find a valid initial double-lattice configuration")

    offset_step = _AdaptiveStep(0.06, bounds=(1e-9, 1.0))
    shear_step = _AdaptiveStep(0.02, bounds=(1e-9, 0.2))
    # The compression magnitude must adapt too.  Held fixed, a 0.4% volume drop
    # per accepted move is far coarser than the remaining slack once the packing
    # approaches jamming, and the search stalls well short of the optimum.
    squeeze_step = _AdaptiveStep(compression, bounds=(1e-10, 0.05))
    identity3 = np.eye(3, dtype=F64)

    for _ in range(cycles):
        for _ in range(3):
            old = offset.copy()
            offset = offset + rng.normal(scale=offset_step.value, size=3)
            ok = _configuration_is_valid(*double_lattice_cell(lattice, offset))
            if not ok:
                offset = old
            offset_step.record(ok)

        for _ in range(3):
            shear = rng.normal(scale=shear_step.value, size=(3, 3))
            shear = 0.5 * (shear + shear.T)
            shear -= (np.trace(shear) / 3.0) * identity3
            candidate = lattice @ (identity3 + shear)
            det_now = abs(float(np.linalg.det(lattice)))
            det_try = abs(float(np.linalg.det(candidate)))
            ok = False
            if det_try > 1e-12 and det_now > 1e-12:
                candidate = candidate * (
                    det_now * (1.0 - squeeze_step.value) / det_try
                ) ** (1.0 / 3.0)
                ok = _configuration_is_valid(*double_lattice_cell(candidate, offset))
            if ok:
                lattice = candidate
            shear_step.record(ok)
            squeeze_step.record(ok)

    volume = abs(float(np.linalg.det(lattice)))
    return DoubleLatticeResult(
        lattice=lattice,
        offset=offset,
        packing_fraction=2.0 * UNIT_TETRA_VOLUME / volume,
    )


#: Tetrahedron with integer vertices (alternating cube corners), edge ``2*sqrt(2)``
#: and volume ``8/3``.  In this frame the N = 2 target determinant is
#: ``16(139 - 40 sqrt(10))/27``, which lies in ``Q(sqrt(10))`` with no ``sqrt(2)``
#: -- so lattice entries are recognisable as ``p + q sqrt(10)``.  The unit-edge
#: frame instead gives ``(139 sqrt2 - 80 sqrt5)/54``, mixing radicals and hiding
#: the structure.
INTEGER_TETRAHEDRON: FloatArray = np.array(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
    dtype=F64,
)


def _double_lattice_isobaric(
    cycles: int,
    seed: int,
    *,
    initial_fraction: float = 0.25,
    pressure_start: float = 1.0e2,
    pressure_end: float = 1.0e8,
    initial: tuple[FloatArray, FloatArray] | None = None,
) -> DoubleLatticeResult:
    """Isobaric Monte Carlo for the monomer double lattice.

    The other searches here compress monotonically: a lattice move is accepted
    only if it shrinks the cell.  That is a greedy descent, and it cannot leave a
    jammed configuration once every densifying move is blocked -- which is why
    they plateau short of the known optima.

    This instead runs at finite pressure, as the original Monte Carlo
    compressions do.  A volume-*increasing* move is accepted with probability
    ``exp(-P dV)``, so the search can back out of a jam, while the pressure ramps
    geometrically so the configuration is squeezed ever harder.  Overlapping
    moves are still rejected outright, so every accepted state remains a
    certified packing; the earlier runaway, where expansion was accepted
    unconditionally and the cell drifted apart, cannot happen because expansion
    now costs.

    Parameters
    ----------
    cycles:
        Monte Carlo cycles.
    seed:
        Reproducible seeding.
    initial_fraction:
        Starting density when no ``initial`` state is supplied.
    pressure_start, pressure_end:
        Geometric pressure ramp, in units where volume is measured with unit
        tetrahedron edges.
    initial:
        Optional ``(lattice, offset)`` to refine instead of starting cold.

    Returns
    -------
    DoubleLatticeResult
    """
    rng = np.random.default_rng(seed)
    base = regular_tetrahedron(1.0)

    if initial is not None:
        lattice = np.array(initial[0], dtype=F64, copy=True)
        offset = np.array(initial[1], dtype=F64, copy=True)
    else:
        offset = rng.normal(scale=0.5, size=3).astype(F64)
        if np.linalg.norm(offset) < 1e-6:
            offset = np.array([1.0, 0.0, 0.0], dtype=F64)
        for _ in range(80):
            pair = np.stack([base, -base + offset])
            if sat_overlap_depth(pair[0:1], pair[1:2])[0] <= 1e-12:
                break
            offset = offset * 1.15
        lattice = _initial_lattice(2, initial_fraction, rng)

    for _ in range(80):
        if _configuration_is_valid(*double_lattice_cell(lattice, offset)):
            break
        lattice = lattice * 1.1
    else:
        raise RuntimeError("failed to find a valid initial double-lattice configuration")

    offset_step = _AdaptiveStep(0.05, bounds=(1e-10, 1.0))
    strain_step = _AdaptiveStep(0.01, bounds=(1e-10, 0.15))
    identity3 = np.eye(3, dtype=F64)
    volume = abs(float(np.linalg.det(lattice)))
    best_lattice, best_offset, best_volume = lattice.copy(), offset.copy(), volume

    for cycle in range(cycles):
        fraction = cycle / max(cycles - 1, 1)
        pressure = pressure_start * (pressure_end / pressure_start) ** fraction

        for _ in range(3):
            trial = offset + rng.normal(scale=offset_step.value, size=3)
            ok = _configuration_is_valid(*double_lattice_cell(lattice, trial))
            if ok:
                offset = trial
            offset_step.record(ok)

        for _ in range(3):
            strain = rng.normal(scale=strain_step.value, size=(3, 3))
            strain = 0.5 * (strain + strain.T)
            candidate = lattice @ (identity3 + strain)
            trial_volume = abs(float(np.linalg.det(candidate)))
            ok = False
            if trial_volume > 1e-12:
                delta = trial_volume - volume
                # Metropolis on the pressure-volume work; shrinking is free.
                if delta <= 0.0 or rng.random() < math.exp(-pressure * delta):
                    ok = _configuration_is_valid(*double_lattice_cell(candidate, offset))
            if ok:
                lattice, volume = candidate, trial_volume
                if volume < best_volume:
                    best_lattice, best_offset, best_volume = (
                        lattice.copy(), offset.copy(), volume,
                    )
            strain_step.record(ok)

    return DoubleLatticeResult(
        lattice=best_lattice,
        offset=best_offset,
        packing_fraction=2.0 * UNIT_TETRA_VOLUME / best_volume,
    )


def _contact_candidates(
    lattice: FloatArray, offset: FloatArray, cutoff: float
) -> list[tuple[int, int, FloatArray]]:
    """Neighbour pairs close enough to constrain the optimum."""
    cell, _ = double_lattice_cell(lattice, offset)
    inverse = np.linalg.inv(lattice)
    reach = 2.0 * UNIT_TETRA_CIRCUMRADIUS
    centres = cell.mean(axis=1)

    candidates: list[tuple[int, int, FloatArray]] = []
    for i in range(2):
        for j in range(i, 2):
            delta = centres[i] - centres[j]
            base = np.round(delta @ inverse)
            for a in range(-3, 4):
                for b in range(-3, 4):
                    for c in range(-3, 4):
                        shift = base + np.array([a, b, c], dtype=F64)
                        if i == j and not (
                            shift[0] > 0
                            or (shift[0] == 0 and (shift[1] > 0 or (shift[1] == 0 and shift[2] > 0)))
                        ):
                            continue
                        if np.linalg.norm(delta - shift @ lattice) < reach + cutoff:
                            candidates.append((i, j, shift.copy()))
    return candidates


def refine_double_lattice(
    lattice: FloatArray,
    offset: FloatArray,
    *,
    rounds: int = 8,
    cutoff: float = 0.25,
    margin: float = 0.0,
    maxiter: int = 2000,
) -> DoubleLatticeResult:
    """Polish a double lattice to the true local optimum under hard contacts.

    Monte Carlo cannot finish this job.  Near jamming the accessible moves are
    smaller than any sensible step size, and the search plateaus: on the N = 2
    phase it stalls around 0.7155, which is 99.4% of the known optimum and looks
    converged from the inside.  Treating it instead as a constrained programme --
    minimise the cell volume subject to every contact depth staying at or below
    zero -- closes the remaining gap completely, reaching 0.99999999 of
    ``N2_PACKING_FRACTION``.

    The active set is rebuilt each round because which images are in contact
    changes as the cell shrinks.  With ``margin = 0`` the optimiser sits exactly
    on the constraint boundary, so rounds that land microscopically outside are
    discarded and the best *certified* configuration is returned.

    Parameters
    ----------
    lattice, offset:
        Starting configuration, typically from :func:`_double_lattice_search`.
    rounds:
        Active-set rebuild iterations.
    cutoff:
        Extra reach when collecting candidate contacts.
    margin:
        Safety gap held at each contact.  Any positive value costs density
        directly -- ``1e-6`` leaves the result about 6e-6 short.
    maxiter:
        Iteration cap for each SLSQP solve.

    Returns
    -------
    DoubleLatticeResult
        The densest configuration that passed the periodic overlap test.
    """
    from scipy.optimize import minimize

    current_lattice = np.array(lattice, dtype=F64, copy=True)
    current_offset = np.array(offset, dtype=F64, copy=True)

    best_lattice = current_lattice.copy()
    best_offset = current_offset.copy()
    best_volume = abs(float(np.linalg.det(best_lattice)))

    def split(vector: FloatArray) -> tuple[FloatArray, FloatArray]:
        return vector[:9].reshape(3, 3), vector[9:12]

    for _ in range(max(rounds, 1)):
        candidates = _contact_candidates(current_lattice, current_offset, cutoff)
        if not candidates:
            break

        def depths(vector: FloatArray, candidates=candidates) -> FloatArray:
            lat, off = split(vector)
            cell, _ = double_lattice_cell(lat, off)
            left = np.array([cell[i] for i, _j, _n in candidates])
            right = np.array([cell[j] + n @ lat for _i, j, n in candidates])
            return sat_overlap_depth(left, right)

        result = minimize(
            lambda vector: abs(float(np.linalg.det(split(vector)[0]))),
            np.concatenate([current_lattice.ravel(), current_offset]),
            constraints=[{"type": "ineq", "fun": lambda v: -depths(v) - margin}],
            method="SLSQP",
            options={"maxiter": maxiter, "ftol": 1e-16, "eps": 1e-10},
        )
        trial_lattice, trial_offset = split(result.x)
        volume = abs(float(np.linalg.det(trial_lattice)))
        if volume < 1e-12:
            break

        if _configuration_is_valid(
            *double_lattice_cell(trial_lattice, trial_offset), tolerance=1e-11
        ):
            current_lattice, current_offset = trial_lattice, trial_offset
            if volume < best_volume:
                best_lattice, best_offset, best_volume = (
                    trial_lattice.copy(), trial_offset.copy(), volume,
                )
        else:
            # Sitting a hair outside the feasible set: back off and retry.
            current_lattice = trial_lattice * 1.0000002
            current_offset = trial_offset

    return DoubleLatticeResult(
        lattice=best_lattice,
        offset=best_offset,
        packing_fraction=2.0 * UNIT_TETRA_VOLUME / best_volume,
    )


def trimer_motif(radius: float, quat: FloatArray) -> FloatArray:
    """A three-fold symmetric cluster of three tetrahedra, as a rigid body.

    Three tetrahedra are related by a 120 degree rotation about the z axis, the
    generating one sitting at distance ``radius`` from it.  Unlike
    :func:`p3_cell` this imposes the symmetry on the *cluster* only, leaving the
    lattice that packs it completely free.

    That distinction matters.  A three-fold rotation of the whole crystal forces
    the lattice to be hexagonal, since the rotation must map the lattice to
    itself.  Table I's "3 monomers, three-fold symmetric" plausibly describes
    the motif rather than the crystal, in which case demanding a hexagonal cell
    is an extra constraint the true packing need not satisfy.

    Parameters
    ----------
    radius:
        Distance of the generating tetrahedron's centroid from the axis.
    quat:
        Unit quaternion giving its orientation.

    Returns
    -------
    ndarray, shape (3, 4, 3)
    """
    if radius < 0.0 or not math.isfinite(radius):
        raise ValueError(f"radius must be finite and non-negative, got {radius!r}")

    rotation = _ROT120
    base = (_quat_to_matrix(np.asarray(quat, dtype=F64)[None])[0] @ regular_tetrahedron(1.0).T).T
    base = base + np.array([radius, 0.0, 0.0], dtype=F64)

    motif = np.empty((3, 4, 3), dtype=F64)
    current = base
    for index in range(3):
        motif[index] = current
        current = (rotation @ current.T).T
    return np.ascontiguousarray(motif)


@dataclass(frozen=True)
class TrimerResult:
    """Outcome of a free-lattice three-fold trimer search."""

    lattice: FloatArray
    radius: float
    quat: FloatArray
    packing_fraction: float

    @property
    def cell(self) -> tuple[FloatArray, FloatArray]:
        return trimer_motif(self.radius, self.quat), self.lattice


def _trimer_search(
    cycles: int,
    seed: int,
    *,
    initial_fraction: float = 0.15,
    compression: float = 0.004,
) -> TrimerResult:
    """Pack one three-fold symmetric trimer per cell on a free triclinic lattice.

    Ten degrees of freedom: the cluster's radius and orientation, and the six
    independent components of the lattice.  Every accepted state is checked for
    overlaps, so the result is always a certified packing.
    """
    rng = np.random.default_rng(seed)

    radius = 0.35 + 0.5 * float(rng.random())
    quat = _random_quaternions(rng, 1)[0]

    # The trimer's three tetrahedra can overlap *each other*, and that is
    # governed by the radius alone -- no amount of lattice expansion separates
    # them.  Resolve the cluster before worrying about its periodic images.
    for _ in range(80):
        motif = trimer_motif(radius, quat)
        pairs = np.array([(0, 1), (0, 2), (1, 2)], dtype=np.int64)
        if np.all(sat_overlap_depth(motif[pairs[:, 0]], motif[pairs[:, 1]]) <= 1e-12):
            break
        radius *= 1.15
    else:
        raise RuntimeError("failed to separate the three tetrahedra of the trimer")

    lattice = _initial_lattice(3, initial_fraction, rng)
    for _ in range(80):
        if _configuration_is_valid(trimer_motif(radius, quat), lattice):
            break
        lattice = lattice * 1.12
    else:
        raise RuntimeError("failed to find a valid initial trimer configuration")

    radius_step = _AdaptiveStep(0.05, bounds=(1e-7, 0.5))
    rot_step = _AdaptiveStep(0.30, bounds=(1e-4, math.pi))
    shear_step = _AdaptiveStep(0.02, bounds=(1e-6, 0.2))
    identity3 = np.eye(3, dtype=F64)

    for _ in range(cycles):
        for _ in range(3):
            old_radius, old_quat = radius, quat.copy()
            if rng.random() < 0.5:
                trial = _quat_multiply(_perturbation_quaternion(rng, rot_step.value), quat)
                quat = trial / np.linalg.norm(trial)
                step = rot_step
            else:
                # Cap the radius: beyond this the three tetrahedra are no
                # longer a cluster in any meaningful sense, and the "three-fold
                # motif" reading of Table I stops applying.
                radius = min(
                    abs(radius + float(rng.normal(scale=radius_step.value))),
                    MAX_TRIMER_RADIUS,
                )
                step = radius_step
            ok = _configuration_is_valid(trimer_motif(radius, quat), lattice)
            if not ok:
                radius, quat = old_radius, old_quat
            step.record(ok)

        for _ in range(3):
            shear = rng.normal(scale=shear_step.value, size=(3, 3))
            shear = 0.5 * (shear + shear.T)
            shear -= (np.trace(shear) / 3.0) * identity3
            candidate = lattice @ (identity3 + shear)
            det_now = abs(float(np.linalg.det(lattice)))
            det_try = abs(float(np.linalg.det(candidate)))
            ok = False
            if det_try > 1e-12 and det_now > 1e-12:
                candidate = candidate * (
                    det_now * (1.0 - compression) / det_try
                ) ** (1.0 / 3.0)
                ok = _configuration_is_valid(trimer_motif(radius, quat), candidate)
            if ok:
                lattice = candidate
            shear_step.record(ok)

    volume = abs(float(np.linalg.det(lattice)))
    return TrimerResult(
        lattice=lattice,
        radius=radius,
        quat=quat,
        packing_fraction=3.0 * UNIT_TETRA_VOLUME / volume,
    )


@dataclass(frozen=True)
class P3Result:
    """Outcome of a three-fold screw-symmetric packing search."""

    a: float
    c: float
    fx: float
    fy: float
    quat: FloatArray
    screw: int
    packing_fraction: float
    accepted_moves: int
    attempted_moves: int

    @property
    def cell(self) -> tuple[FloatArray, FloatArray]:
        return p3_cell(self.a, self.c, self.fx, self.fy, self.quat, self.screw)


def p3_packing_fraction(a: float, c: float) -> float:
    """Density of a three-tetrahedron hexagonal cell of parameters ``a``, ``c``."""
    volume = 0.5 * math.sqrt(3.0) * a * a * c
    return 3.0 * UNIT_TETRA_VOLUME / volume


def _p3_search(
    cycles: int,
    seed: int,
    screw: int,
    *,
    initial_fraction: float = 0.15,
    compression: float = 0.004,
) -> P3Result:
    """Adaptive shrinking cell Monte Carlo restricted to a three-fold screw.

    The symmetry is imposed on the *parametrisation*, not enforced afterwards,
    so every configuration visited is exactly three-fold symmetric and the
    search explores seven numbers instead of a 4-body configuration space.  As
    elsewhere, a move producing any overlap is rejected outright, so every
    accepted state is a certified packing.
    """
    rng = np.random.default_rng(seed)

    # Start dilute at the requested fraction, with a roughly isotropic cell.
    volume = 3.0 * UNIT_TETRA_VOLUME / initial_fraction
    a = (2.0 * volume / math.sqrt(3.0)) ** (1.0 / 3.0)
    c = volume / (0.5 * math.sqrt(3.0) * a * a)
    fx, fy = float(rng.random()), float(rng.random())
    quat = _random_quaternions(rng, 1)[0]

    for _ in range(80):
        verts, lattice = p3_cell(a, c, fx, fy, quat, screw)
        if _configuration_is_valid(verts, lattice):
            break
        a, c = a * 1.12, c * 1.12
    else:
        raise RuntimeError("failed to find a valid initial P3 configuration")

    move_step = _AdaptiveStep(0.06, bounds=(1e-7, 0.5))
    rot_step = _AdaptiveStep(0.30, bounds=(1e-4, math.pi))
    cell_step = _AdaptiveStep(0.02, bounds=(1e-6, 0.2))

    accepted = attempted = 0
    for cycle in range(cycles):
        for _ in range(3):
            attempted += 1
            old = (fx, fy, quat.copy())
            if rng.random() < 0.5:
                trial = _quat_multiply(_perturbation_quaternion(rng, rot_step.value), quat)
                quat = trial / np.linalg.norm(trial)
                step = rot_step
            else:
                fx = (fx + float(rng.normal(scale=move_step.value))) % 1.0
                fy = (fy + float(rng.normal(scale=move_step.value))) % 1.0
                step = move_step

            verts, lattice = p3_cell(a, c, fx, fy, quat, screw)
            ok = _configuration_is_valid(verts, lattice)
            if ok:
                accepted += 1
            else:
                fx, fy, quat = old
            step.record(ok)

        for _ in range(3):
            attempted += 1
            # Reshape the cell at fixed volume, then compress: as in the
            # general search, every accepted cell move is strictly densifying.
            ratio = math.exp(rng.normal(scale=cell_step.value))
            shrink = (1.0 - compression) ** (1.0 / 3.0)
            trial_a = a * ratio * shrink
            trial_c = c * shrink**3 / ratio**2

            ok = trial_a > 1e-6 and trial_c > 1e-6
            if ok:
                verts, lattice = p3_cell(trial_a, trial_c, fx, fy, quat, screw)
                ok = _configuration_is_valid(verts, lattice)
            if ok:
                a, c = trial_a, trial_c
                accepted += 1
            cell_step.record(ok)

        if cycle % max(cycles // 8, 1) == 0:
            LOGGER.debug(
                "P3(%d) cycle %d/%d: phi=%.4f a=%.4f c=%.4f",
                screw, cycle, cycles, p3_packing_fraction(a, c), a, c,
            )

    return P3Result(
        a=a, c=c, fx=fx, fy=fy, quat=quat, screw=screw,
        packing_fraction=p3_packing_fraction(a, c),
        accepted_moves=accepted, attempted_moves=attempted,
    )


def build_p3_packing(
    min_tetrahedra: int = 1000,
    *,
    screw: int | None = None,
    cycles: int = 3000,
    restarts: int = 8,
    seed: int = 20250805,
    max_replicas: int = 40,
) -> TetraCloud:
    """Build the densest three-fold screw-symmetric packing the search finds.

    Table I of Chen, Engel & Glotzer reports an N = 3 phase at an analytical
    density of exactly 2/3, described as "3 monomers, three-fold symmetric".
    The paper does not publish its coordinates, so this searches the symmetric
    family directly rather than reproducing them.

    Parameters
    ----------
    min_tetrahedra:
        Minimum tetrahedron count in the returned cloud.
    screw:
        Screw index to search, or ``None`` to try all of P3, P3_1 and P3_2 and
        keep the densest.
    cycles, restarts, seed:
        Monte Carlo effort and reproducible seeding.
    max_replicas:
        Safety bound on tiling repetitions per lattice direction.

    Returns
    -------
    TetraCloud
    """
    if min_tetrahedra < 1:
        raise ValueError("min_tetrahedra must be >= 1")
    if cycles < 1 or restarts < 1:
        raise ValueError("cycles and restarts must be >= 1")

    indices = P3_SCREW_INDICES if screw is None else (screw,)
    attempts: list[P3Result] = []
    for index in indices:
        for restart in range(restarts):
            attempts.append(_p3_search(cycles, seed + 1000 * index + restart, index))
    best = max(attempts, key=lambda r: r.packing_fraction)

    by_screw = {
        index: max(r.packing_fraction for r in attempts if r.screw == index)
        for index in indices
    }
    LOGGER.info(
        "P3 search: best phi=%.9f (target 2/3=%.9f, ratio %.4f); per screw index %s",
        best.packing_fraction, 2 / 3, best.packing_fraction / (2 / 3),
        {k: round(v, 6) for k, v in by_screw.items()},
    )

    cell, lattice = best.cell
    per_cell = int(cell.shape[0])
    reps = 1
    while reps <= max_replicas and per_cell * reps**3 < min_tetrahedra:
        reps += 1
    if per_cell * reps**3 < min_tetrahedra:
        raise RuntimeError(
            f"cannot reach {min_tetrahedra} tetrahedra within max_replicas={max_replicas}"
        )

    grid = np.arange(reps, dtype=F64)
    i, j, k = np.meshgrid(grid, grid, grid, indexing="ij")
    translations = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) @ lattice
    tiled = (cell[None, :, :, :] + translations[:, None, None, :]).reshape(-1, 4, 3)

    overlaps = find_overlapping_pairs(tiled)
    if overlaps.shape[0]:
        depth = float(sat_overlap_depth(tiled[overlaps[:, 0]], tiled[overlaps[:, 1]]).max())
        raise AssertionError(
            f"P3 search produced {overlaps.shape[0]} overlapping pairs "
            f"(max penetration {depth:.6f}); the result is not a packing"
        )

    cloud = TetraCloud(
        tetrahedra=tiled,
        lattice=lattice,
        tetra_per_cell=per_cell,
        provenance={
            "backend": "p3",
            "target": "Chen, Engel & Glotzer Table I, N = 3 phase, phi = 2/3",
            "measured_packing_fraction": best.packing_fraction,
            "target_packing_fraction": float(N3_TARGET_FRACTION),
            "screw_index": best.screw,
            "space_group": {0: "P3", 1: "P3_1", 2: "P3_2"}[best.screw],
            "cell_a": best.a,
            "cell_c": best.c,
            "best_by_screw_index": by_screw,
            "asc_cycles": cycles,
            "asc_restarts": restarts,
            "replicas_per_axis": reps,
        },
    )
    cloud.assert_regular(edge=1.0, atol=1e-9)
    return cloud


# --------------------------------------------------------------------------- #
# Backend 4: dense packing via adaptive shrinking cell Monte Carlo
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PackingResult:
    """Outcome of an adaptive-shrinking-cell packing search."""

    lattice: FloatArray
    cell_tetrahedra: FloatArray
    packing_fraction: float
    accepted_moves: int
    attempted_moves: int
    n_particles: int
    motif: str

    @property
    def acceptance_ratio(self) -> float:
        return self.accepted_moves / max(self.attempted_moves, 1)

    @property
    def n_tetrahedra_per_cell(self) -> int:
        return int(self.cell_tetrahedra.shape[0])


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


#: Largest per-axis periodic image span the overlap test will enumerate.  A
#: configuration needing more than this is rejected rather than under-tested.
#: With per-pair image boxes the span depends only on the lattice, so only a
#: genuinely degenerate cell can exceed it.
MAX_IMAGE_SPAN: int = 6


def _cell_vertices(
    motif: FloatArray, lattice: FloatArray, fractional: FloatArray, quats: FloatArray
) -> FloatArray:
    """Vertices of every tetrahedron in the fundamental cell.

    Each of the ``n_particles`` rigid bodies carries the whole motif, so the
    returned array holds ``n_particles * motif_size`` tetrahedra.  Collision
    detection then treats them all as independent convex bodies; tetrahedra
    belonging to the same motif touch face-to-face at zero depth and so pass
    the overlap test unchanged.
    """
    rot = _quat_to_matrix(quats)  # (P, 3, 3)
    oriented = np.einsum("pij,mvj->pmvi", rot, motif)  # (P, M, 4, 3)
    centres = fractional @ lattice  # (P, 3)
    return (oriented + centres[:, None, None, :]).reshape(-1, 4, 3)


_BOX_CACHE: dict[tuple[int, int, int], FloatArray] = {}


def _integer_box(spans: tuple[int, int, int]) -> FloatArray:
    """Cached ``(-s..s)^3`` integer grid; rebuilding it per call dominated cost."""
    box = _BOX_CACHE.get(spans)
    if box is None:
        grids = [np.arange(-s, s + 1, dtype=np.int64) for s in spans]
        i, j, k = np.meshgrid(*grids, indexing="ij")
        box = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1).astype(F64)
        _BOX_CACHE[spans] = box
    return box


def _configuration_is_valid(
    verts: FloatArray, lattice: FloatArray, tolerance: float = 1e-12
) -> bool:
    """True when no tetrahedron overlaps another, including periodic images.

    Each *pair* gets its own image box, centred on the integer shift that
    actually brings the two together, rather than one global box sized to cover
    the whole cell.  Two consequences:

    * the box depends only on the lattice, never on how far apart the cell's
      contents are, so a cell holding widely separated tetrahedra is handled
      correctly instead of needing an enormous global range;
    * the candidate count collapses -- for three tetrahedra, from ~730 global
      images times 9 ordered pairs down to 6 unordered pairs times ~125.

    A pair can only touch if its centroids lie within twice the circumradius.
    Writing the required shift as ``n``, the fractional offset obeys
    ``|frac_k - n_k| <= reach * ||inv[:, k]||``, and since the box is centred on
    ``round(frac)`` an extra half cell covers the rounding.

    Dispatches to the compiled kernel in :mod:`tetra_fastsat` when numba is
    installed, falling back to the array implementation otherwise.  The two are
    tested to agree.
    """
    setup = _periodic_setup(verts, lattice)
    if setup is None:
        return False
    inverse, spans, reach = setup

    if _USE_FAST_SAT and tetra_fastsat.HAVE_NUMBA:
        return tetra_fastsat.configuration_is_valid(
            verts, lattice, inverse, spans, tolerance, reach
        )
    return _configuration_is_valid_numpy(verts, lattice, inverse, spans, reach, tolerance)


def _periodic_setup(
    verts: FloatArray, lattice: FloatArray
) -> tuple[FloatArray, tuple[int, int, int], float] | None:
    """Inverse lattice, per-axis image spans and contact reach, or ``None``.

    ``None`` means the configuration cannot be certified -- a singular or
    extremely thin cell -- and callers must treat that as a rejection rather
    than as a pass.
    """
    if verts.shape[0] < 1:
        return None
    try:
        inverse = np.linalg.inv(lattice)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(inverse)):
        return None

    reach = 2.0 * UNIT_TETRA_CIRCUMRADIUS
    extent = reach * np.linalg.norm(inverse, axis=0) + 0.5
    if not np.all(np.isfinite(extent)):
        return None
    spans = np.ceil(extent).astype(np.int64)
    if np.any(spans > MAX_IMAGE_SPAN):
        return None
    return inverse, (int(spans[0]), int(spans[1]), int(spans[2])), reach


_BOX_CACHE: dict[tuple[int, int, int], FloatArray] = {}


def _integer_box(spans: tuple[int, int, int]) -> FloatArray:
    """Cached ``(-s..s)^3`` integer grid; rebuilding it per call dominated cost."""
    box = _BOX_CACHE.get(spans)
    if box is None:
        grids = [np.arange(-s, s + 1, dtype=np.int64) for s in spans]
        i, j, k = np.meshgrid(*grids, indexing="ij")
        box = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1).astype(F64)
        _BOX_CACHE[spans] = box
    return box


def _configuration_is_valid_numpy(
    verts: FloatArray,
    lattice: FloatArray,
    inverse: FloatArray,
    spans: tuple[int, int, int],
    reach: float,
    tolerance: float,
) -> bool:
    """Array implementation of the periodic overlap test."""
    box = _integer_box(spans)
    n = verts.shape[0]

    centres = verts.mean(axis=1)
    rows, cols = np.triu_indices(n)
    delta = centres[rows] - centres[cols]                             # (P, 3)
    shifts = np.round(delta @ inverse)[:, None, :] + box[None, :, :]  # (P, B, 3)
    separation = delta[:, None, :] - shifts @ lattice                 # (P, B, 3)
    near = np.linalg.norm(separation, axis=2) < reach

    # A particle against its own images: n and -n describe the same pair, and
    # the zero shift is the particle against itself.  Keep the lexicographically
    # positive half.
    first, second, third = shifts[..., 0], shifts[..., 1], shifts[..., 2]
    positive = (first > 0) | (
        (first == 0) & ((second > 0) | ((second == 0) & (third > 0)))
    )
    near &= np.where((rows == cols)[:, None], positive, True)
    if not np.any(near):
        return True

    pair_idx, box_idx = np.nonzero(near)
    left = verts[rows[pair_idx]]
    right = verts[cols[pair_idx]] + (shifts[pair_idx, box_idx] @ lattice)[:, None, :]
    return bool(np.all(sat_disjoint(left, right, tolerance)))


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


#: Rigid bodies the packing search can move as a unit.
MOTIF_NAMES: tuple[str, ...] = ("single", "dimer")


def build_motif(name: str) -> FloatArray:
    """Return the tetrahedra of a rigid packing motif in its own body frame.

    ``"single"``
        One free tetrahedron -- the unconstrained search.
    ``"dimer"``
        Two tetrahedra fused face-to-face into a triangular dipyramid, moved
        and rotated as one rigid body.  Every known high-density packing of
        regular tetrahedra is a *dimer* crystal, so constraining the search
        unit this way halves the degrees of freedom while keeping the known
        optima inside the search space: four free tetrahedra per cell become
        two rigid dimers.

    The motif is centred on its own centroid so that orientation moves rotate
    it about its centre rather than swinging it through space.

    Returns
    -------
    ndarray, shape (n_tetrahedra_in_motif, 4, 3)
    """
    base = regular_tetrahedron(1.0)
    if name == "single":
        return np.ascontiguousarray(base[None], dtype=F64)
    if name == "dimer":
        quat, offset = _dimer_partner()
        partner = (_quat_to_matrix(quat[None])[0] @ base.T).T + offset
        motif = np.stack([base, partner])
        return np.ascontiguousarray(motif - motif.reshape(-1, 3).mean(axis=0), dtype=F64)
    raise ValueError(f"unknown motif {name!r}; expected one of {MOTIF_NAMES}")


def _initial_lattice(
    n_tetrahedra: int, target_fraction: float, rng: np.random.Generator
) -> FloatArray:
    """A near-cubic cell sized to hold ``n_tetrahedra`` at ``target_fraction``."""
    volume = n_tetrahedra * UNIT_TETRA_VOLUME / target_fraction
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
    motif: str = "dimer",
) -> PackingResult:
    """Adaptive shrinking cell Monte Carlo for hard regular tetrahedra.

    Each cycle attempts ``n_particles`` translation or rotation moves followed
    by ``lattice_moves_per_cycle`` volume-reducing cell deformations.  Any move
    that produces an overlap is rejected outright (hard-particle dynamics), so
    every accepted state -- including the final one -- is a certified valid
    packing.  Step sizes self-tune toward ~30% acceptance.  The achieved
    packing fraction is *measured*, never assumed.

    ``motif`` selects the rigid body being packed.  Merely *seeding* random
    tetrahedra as dimer pairs does not help -- measured across three seeds it
    reached 0.42-0.52 versus 0.49-0.72 for random initialisation, because
    reheating disassembles the pairs long before jamming.  Making the dimer a
    genuine rigid constraint is a different matter: it halves the degrees of
    freedom and keeps the search inside the family containing the known optima.
    """
    rng = np.random.default_rng(seed)
    shape = build_motif(motif)
    motif_size = int(shape.shape[0])
    n_tetrahedra = n_particles * motif_size

    lattice = _initial_lattice(n_tetrahedra, initial_fraction, rng)
    fractional = rng.random((n_particles, 3)).astype(F64)
    quats = _random_quaternions(rng, n_particles)

    for _ in range(60):
        if _configuration_is_valid(_cell_vertices(shape, lattice, fractional, quats), lattice):
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
                # Wrap back into the cell.  Translating a particle by a whole
                # lattice vector leaves the periodic packing identical, but an
                # unwrapped coordinate random-walks out of the box, and the
                # periodic image search is only valid for particles inside it --
                # a drifted particle's true neighbours are never tested and
                # overlaps go unnoticed.
                fractional[idx] = np.mod(
                    old_f + rng.normal(scale=trans_step.value, size=3), 1.0
                )

            ok = _configuration_is_valid(
                _cell_vertices(shape, lattice, fractional, quats), lattice
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
                    _cell_vertices(shape, candidate, fractional, quats), candidate
                )
            if ok:
                lattice = candidate
                accepted += 1
            shear_step.record(ok)
            squeeze_step.record(ok)

        if cycle % log_every == 0:
            phi = n_tetrahedra * UNIT_TETRA_VOLUME / abs(float(np.linalg.det(lattice)))
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
        cell_tetrahedra=_cell_vertices(shape, lattice, fractional, quats),
        packing_fraction=n_tetrahedra * UNIT_TETRA_VOLUME / volume,
        accepted_moves=accepted,
        attempted_moves=attempted,
        n_particles=n_particles,
        motif=motif,
    )


def build_dense_packing(
    min_tetrahedra: int = 1000,
    *,
    n_particles: int = 2,
    motif: str = "dimer",
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
        attempt = _asc_search(n_particles, cycles, seed + index, motif=motif)
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

    per_cell = result.n_tetrahedra_per_cell
    reps = 1
    while reps <= max_replicas and per_cell * reps**3 < min_tetrahedra:
        reps += 1
    if per_cell * reps**3 < min_tetrahedra:
        raise RuntimeError(
            f"cannot reach {min_tetrahedra} tetrahedra within max_replicas={max_replicas}"
        )

    oriented = result.cell_tetrahedra

    grid = np.arange(reps, dtype=F64)
    a, b, c = np.meshgrid(grid, grid, grid, indexing="ij")
    translations = (
        np.stack([a.ravel(), b.ravel(), c.ravel()], axis=1) @ result.lattice
    )  # (reps^3, 3)

    tiled = (oriented[None, :, :, :] + translations[:, None, None, :]).reshape(-1, 4, 3)

    # Independent end-to-end check on the tiled result.  The periodic test used
    # during the search reasons about image ranges; this one simply looks at
    # every nearby pair of the assembled cloud, so a bug in that reasoning
    # cannot hide here.
    overlaps = find_overlapping_pairs(tiled)
    if overlaps.shape[0]:
        depth = float(sat_overlap_depth(tiled[overlaps[:, 0]], tiled[overlaps[:, 1]]).max())
        raise AssertionError(
            f"ASC produced {overlaps.shape[0]} overlapping tetrahedron pairs "
            f"(max penetration {depth:.6f}); the search result is not a packing"
        )

    cloud = TetraCloud(
        tetrahedra=tiled,
        lattice=result.lattice,
        tetra_per_cell=per_cell,
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
            "motif": motif,
            "particles_per_cell": n_particles,
            "tetrahedra_per_cell": per_cell,
            "replicas_per_axis": reps,
        },
    )
    cloud.assert_regular(edge=1.0, atol=1e-9)
    LOGGER.info("packing: %d tetrahedra from %d^3 replicas", cloud.n_tetrahedra, reps)
    return cloud
