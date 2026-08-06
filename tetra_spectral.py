"""Graph-Laplacian construction and spectral analysis for tetrahedral clouds.

The pipeline is:

1. :func:`merge_vertices` -- collapse coincident tetrahedron vertices into a
   single point set using a strict Euclidean tolerance.
2. :func:`build_unit_distance_graph` -- connect two merged vertices when their
   separation equals the tetrahedron edge length, i.e. when they span a
   physical edge of some tetrahedron.
3. :func:`graph_laplacian` -- assemble the sparse combinatorial Laplacian
   ``L = D - A``.
4. :func:`smallest_eigenvalues` -- extract the low end of the spectrum with
   ARPACK, with layered fallbacks for the cases where ARPACK struggles.
5. :func:`analyse_degeneracies` -- group eigenvalues into degenerate levels and
   quantify the spacing distribution.

Degeneracy is the observable of interest: a cluster carrying a non-abelian
point group has eigenspaces that transform as irreducible representations of
that group, so eigenvalue multiplicities are forced to match irrep dimensions.
For the octahedral group :math:`O_h` those dimensions are 1, 1, 2, 3 and 3, so
a spectrum dominated by multiplicities in ``{1, 2, 3}`` (and their sums, where
distinct irreps accidentally coincide) is the algebraic fingerprint of that
symmetry rather than of anything exotic.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix, diags, identity, issparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigsh
from scipy.spatial import cKDTree

__all__ = [
    "MergedVertices",
    "GraphBundle",
    "Spectrum",
    "DegeneracyReport",
    "merge_vertices",
    "build_unit_distance_graph",
    "graph_laplacian",
    "smallest_eigenvalues",
    "analyse_degeneracies",
    "periodic_graph_components",
    "PeriodicGraphReport",
    "PeriodicComponent",
    "sublattice_index",
    "sublattice_rank",
    "POISSON_RATIO",
    "GOE_RATIO",
]

LOGGER = logging.getLogger(__name__)

F64 = np.float64
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: Mean adjacent-gap ratio for uncorrelated (Poisson) spectra.
POISSON_RATIO: float = 2.0 * math.log(2.0) - 1.0  # ~= 0.386294

#: Mean adjacent-gap ratio for the Gaussian Orthogonal Ensemble.
GOE_RATIO: float = 0.535898


# --------------------------------------------------------------------------- #
# Vertex merging
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MergedVertices:
    """Result of collapsing coincident vertices.

    Attributes
    ----------
    points:
        ``(n_unique, 3)`` representative coordinates (cluster centroids).
    labels:
        ``(n_raw,)`` mapping from each input vertex to its merged index.
    cluster_sizes:
        ``(n_unique,)`` number of raw vertices merged into each point.
    max_cluster_radius:
        Largest distance from any raw vertex to its cluster centroid, a direct
        measure of how much chaining the tolerance permitted.
    """

    points: FloatArray
    labels: IntArray
    cluster_sizes: IntArray
    max_cluster_radius: float

    @property
    def n_unique(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_raw(self) -> int:
        return int(self.labels.shape[0])


def merge_vertices(points: FloatArray, *, atol: float = 1e-5) -> MergedVertices:
    """Collapse vertices lying within ``atol`` of one another.

    Merging is transitive (single-linkage): points are joined when they fall
    within ``atol``, and connected components of that proximity graph become
    single vertices.  This is the metric-space analogue of ``np.isclose`` with
    ``rtol=0, atol=atol`` applied to every coordinate pair, but it runs in
    ``O(n log n)`` via a KD-tree instead of ``O(n^2)``.

    Because single-linkage can in principle chain, the largest cluster radius
    is measured and reported; a radius far exceeding ``atol`` means the
    tolerance was too loose for the geometry and distinct vertices were fused.

    Parameters
    ----------
    points:
        ``(n, 3)`` coordinates, typically ``TetraCloud.raw_points``.
    atol:
        Absolute merge tolerance.  Must be positive and far smaller than the
        smallest genuine inter-vertex distance.

    Returns
    -------
    MergedVertices
    """
    pts = np.ascontiguousarray(points, dtype=F64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (n, 3), got {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError("cannot merge an empty point set")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points contain non-finite coordinates")
    if not (atol > 0.0) or not math.isfinite(atol):
        raise ValueError(f"atol must be a positive finite float, got {atol!r}")

    n = pts.shape[0]
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=atol, output_type="ndarray")

    if pairs.size:
        data = np.ones(pairs.shape[0], dtype=np.int8)
        proximity = coo_matrix(
            (data, (pairs[:, 0], pairs[:, 1])), shape=(n, n), dtype=np.int8
        ).tocsr()
    else:
        proximity = csr_matrix((n, n), dtype=np.int8)

    n_unique, labels = connected_components(proximity, directed=False)
    labels = labels.astype(np.int64, copy=False)

    sums = np.zeros((n_unique, 3), dtype=F64)
    np.add.at(sums, labels, pts)
    sizes = np.bincount(labels, minlength=n_unique).astype(np.int64)
    centroids = sums / sizes[:, None]

    radii = np.linalg.norm(pts - centroids[labels], axis=1)
    max_radius = float(radii.max()) if radii.size else 0.0

    if max_radius > 10.0 * atol:
        LOGGER.warning(
            "vertex merging chained: max cluster radius %.3e exceeds 10x atol "
            "(%.3e); consider tightening --merge-atol",
            max_radius,
            atol,
        )

    LOGGER.info(
        "merged %d raw vertices into %d unique points (max cluster radius %.3e)",
        n,
        n_unique,
        max_radius,
    )
    return MergedVertices(
        points=np.ascontiguousarray(centroids, dtype=F64),
        labels=labels,
        cluster_sizes=sizes,
        max_cluster_radius=max_radius,
    )


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GraphBundle:
    """A unit-distance graph in both NetworkX and sparse-matrix form."""

    graph: nx.Graph
    adjacency: csr_matrix
    edges: IntArray
    n_components: int
    component_labels: IntArray

    @property
    def n_nodes(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def degrees(self) -> IntArray:
        return np.asarray(self.adjacency.sum(axis=1)).ravel().astype(np.int64)

    @property
    def density(self) -> float:
        n = self.n_nodes
        return (2.0 * self.n_edges) / (n * (n - 1)) if n > 1 else 0.0


def build_unit_distance_graph(
    points: FloatArray,
    *,
    distance: float = 1.0,
    atol: float = 1e-6,
) -> GraphBundle:
    """Connect vertices separated by exactly ``distance``.

    An edge is created iff ``|d(u, v) - distance| <= atol``.  For a cloud of
    unit-edge tetrahedra with merged vertices, this reproduces exactly the
    union of the tetrahedra's physical edges (plus any unit-distance
    coincidences the arrangement happens to create, which for the FCC
    honeycomb are themselves honeycomb edges).

    The graph is unweighted and undirected; the adjacency matrix is symmetric
    with a zero diagonal.

    Parameters
    ----------
    points:
        ``(n, 3)`` merged vertex coordinates.
    distance:
        Target separation defining an edge.
    atol:
        Absolute tolerance on the separation.

    Returns
    -------
    GraphBundle
    """
    pts = np.ascontiguousarray(points, dtype=F64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (n, 3), got {pts.shape}")
    if distance <= 0.0 or not math.isfinite(distance):
        raise ValueError(f"distance must be positive and finite, got {distance!r}")
    if atol < 0.0 or not math.isfinite(atol):
        raise ValueError(f"atol must be non-negative and finite, got {atol!r}")
    if atol >= distance:
        raise ValueError("atol must be smaller than distance")

    n = pts.shape[0]
    tree = cKDTree(pts)
    candidates = tree.query_pairs(r=distance + atol, output_type="ndarray")

    if candidates.size:
        sep = np.linalg.norm(pts[candidates[:, 0]] - pts[candidates[:, 1]], axis=1)
        edges = candidates[np.abs(sep - distance) <= atol]
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
    edges = np.ascontiguousarray(edges, dtype=np.int64)

    if edges.size:
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.ones(rows.shape[0], dtype=F64)
        adjacency = coo_matrix((data, (rows, cols)), shape=(n, n), dtype=F64).tocsr()
    else:
        adjacency = csr_matrix((n, n), dtype=F64)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(map(tuple, edges))

    n_components, component_labels = connected_components(adjacency, directed=False)

    LOGGER.info(
        "unit-distance graph: %d nodes, %d edges, %d connected component(s)",
        n,
        edges.shape[0],
        n_components,
    )
    return GraphBundle(
        graph=graph,
        adjacency=adjacency,
        edges=edges,
        n_components=int(n_components),
        component_labels=component_labels.astype(np.int64, copy=False),
    )


def graph_laplacian(adjacency: csr_matrix) -> csr_matrix:
    """Assemble the combinatorial Laplacian ``L = D - A`` in sparse form.

    ``L`` is symmetric positive semi-definite; its kernel dimension equals the
    number of connected components of the graph.
    """
    if not issparse(adjacency):
        raise TypeError("adjacency must be a scipy.sparse matrix")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square, got {adjacency.shape}")

    degrees = np.asarray(adjacency.sum(axis=1), dtype=F64).ravel()
    laplacian = (diags(degrees, dtype=F64) - adjacency).tocsr()
    laplacian.eliminate_zeros()
    return laplacian


# --------------------------------------------------------------------------- #
# Spectrum
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Spectrum:
    """Computed low-end eigenvalues of a graph Laplacian."""

    eigenvalues: FloatArray
    k_requested: int
    method: Literal[
        "shift-invert", "arpack-sa", "dense", "arpack-partial", "block-diagonal"
    ]
    matrix_dimension: int
    n_components: int
    zero_tolerance: float
    residual_norm: float | None = None

    @property
    def k_returned(self) -> int:
        return int(self.eigenvalues.shape[0])

    @property
    def zero_multiplicity(self) -> int:
        """Number of eigenvalues indistinguishable from zero."""
        return int(np.count_nonzero(np.abs(self.eigenvalues) <= self.zero_tolerance))

    @property
    def lambda_0(self) -> float:
        return float(self.eigenvalues[0])

    @property
    def algebraic_connectivity(self) -> float:
        """The Fiedler value: the second-smallest eigenvalue.

        This is zero exactly when the graph is disconnected.
        """
        if self.k_returned < 2:
            raise ValueError("need at least two eigenvalues for algebraic connectivity")
        return float(self.eigenvalues[1])

    @property
    def first_positive(self) -> float | None:
        """Smallest strictly positive eigenvalue, or ``None`` if none computed.

        For a connected graph this coincides with the Fiedler value.  For a
        disconnected graph the Fiedler value is 0 and this instead reports the
        strongest connectivity present within any single component.
        """
        positive = self.eigenvalues[self.eigenvalues > self.zero_tolerance]
        return float(positive[0]) if positive.size else None


def _zero_tolerance(laplacian: csr_matrix) -> float:
    """A scale-aware threshold separating numerical zero from a real eigenvalue.

    Eigenvalues of ``L`` are bounded above by twice the maximum degree, so that
    bound sets the natural scale against which double-precision round-off from
    the eigensolver should be judged.
    """
    max_degree = float(laplacian.diagonal().max()) if laplacian.shape[0] else 1.0
    return max(1e-9, 1e-10 * max(2.0 * max_degree, 1.0) * laplacian.shape[0] ** 0.5)


def _block_diagonal_spectrum(
    laplacian: csr_matrix,
    k: int,
    component_labels: IntArray,
    n_components: int,
    solve_block,
) -> FloatArray:
    """Smallest ``k`` eigenvalues of a Laplacian with several components.

    A disconnected graph has a block-diagonal Laplacian, so its spectrum is the
    union of the blocks' spectra.  Solving block by block is both exact and far
    more reliable than aiming ARPACK at the whole matrix: the kernel of a graph
    with many components is massively degenerate, and Krylov methods cannot
    resolve a degeneracy of that order -- they converge to an arbitrary subset
    of the invariant subspace and silently return eigenvalues that are not the
    smallest.

    The matrix is permuted once so components occupy contiguous index ranges,
    after which each block is a cheap slice.
    """
    order = np.argsort(component_labels, kind="stable")
    permuted = laplacian[order][:, order]
    sorted_labels = component_labels[order]
    starts = np.searchsorted(sorted_labels, np.arange(n_components + 1))

    collected: list[FloatArray] = []
    for index in range(n_components):
        lo, hi = int(starts[index]), int(starts[index + 1])
        size = hi - lo
        if size <= 0:
            continue
        block = permuted[lo:hi, lo:hi].tocsr()
        collected.append(solve_block(block, min(k, size)))

    return np.sort(np.concatenate(collected))[:k]


def _single_block_eigenvalues(
    laplacian: csr_matrix,
    k_eff: int,
    *,
    dense_threshold: int,
    tol: float,
    maxiter: int | None,
) -> tuple[FloatArray, str]:
    """Smallest ``k_eff`` eigenvalues of one Laplacian block, and the path used.

    Strategy, in order of preference:

    1. **Dense LAPACK** when the block is small enough that ``O(n^3)`` is
       affordable -- exact and unconditionally reliable.
    2. **Shift-invert ARPACK** at ``sigma`` slightly below zero.  ``L`` is
       singular by construction, so a shift of exactly zero would factorise a
       singular matrix; shifting just below the kernel keeps the factorisation
       well posed while still targeting the low end.
    3. **Direct ARPACK** in ``which="SA"`` mode, which needs no factorisation
       but converges slowly on clustered low-end spectra.

    A partial ARPACK result is preferred over raising: if convergence stalls,
    the eigenvalues that did converge are returned and the path is reported as
    ``"arpack-partial"`` so the caller can see the degradation.
    """
    n = laplacian.shape[0]
    ztol = _zero_tolerance(laplacian)

    def sorted_values(values: FloatArray) -> FloatArray:
        return np.sort(np.real(np.asarray(values, dtype=F64)))

    if n <= dense_threshold:
        LOGGER.debug("block: dense LAPACK on %d x %d", n, n)
        dense = np.asarray(laplacian.todense(), dtype=F64)
        dense = 0.5 * (dense + dense.T)  # enforce exact symmetry
        return sorted_values(np.linalg.eigvalsh(dense)[:k_eff]), "dense"

    if k_eff >= n:
        raise ValueError(
            f"ARPACK requires k < n; got k={k_eff}, n={n}. Raise dense_threshold "
            "to use dense LAPACK for this matrix."
        )

    ncv = min(n, max(2 * k_eff + 1, 20))
    iterations = maxiter if maxiter is not None else 10 * n
    sigma = -max(1e-5, ztol * 10.0)

    try:
        LOGGER.info(
            "spectrum: ARPACK shift-invert (sigma=%.2e, k=%d, ncv=%d)", sigma, k_eff, ncv
        )
        values = eigsh(
            laplacian.astype(F64),
            k=k_eff,
            sigma=sigma,
            which="LM",
            mode="normal",
            tol=tol,
            maxiter=iterations,
            ncv=ncv,
            return_eigenvectors=False,
        )
        return sorted_values(values), "shift-invert"
    except (ArpackNoConvergence, ArpackError, RuntimeError, ValueError) as exc:
        LOGGER.warning(
            "shift-invert failed (%s: %s); trying direct ARPACK",
            type(exc).__name__,
            exc,
        )

    try:
        LOGGER.info("spectrum: ARPACK which='SA' (k=%d, ncv=%d)", k_eff, ncv)
        values = eigsh(
            laplacian.astype(F64),
            k=k_eff,
            which="SA",
            tol=tol if tol > 0.0 else 1e-10,
            maxiter=iterations,
            ncv=ncv,
            return_eigenvectors=False,
        )
        return sorted_values(values), "arpack-sa"
    except ArpackNoConvergence as exc:
        converged = np.asarray(exc.eigenvalues, dtype=F64)
        if converged.size:
            LOGGER.warning(
                "ARPACK returned %d of %d eigenvalues before stalling; "
                "reporting the converged subset",
                converged.size,
                k_eff,
            )
            return sorted_values(converged), "arpack-partial"
        LOGGER.warning("ARPACK returned no converged eigenvalues; falling back to dense")
    except (ArpackError, RuntimeError, ValueError) as exc:
        LOGGER.warning(
            "direct ARPACK failed (%s: %s); falling back to dense",
            type(exc).__name__,
            exc,
        )

    LOGGER.warning(
        "spectrum: falling back to dense LAPACK on a %d x %d matrix; this needs "
        "roughly %.1f GiB",
        n,
        n,
        (n * n * 8) / 2**30,
    )
    dense = np.asarray(laplacian.todense(), dtype=F64)
    dense = 0.5 * (dense + dense.T)
    return sorted_values(np.linalg.eigvalsh(dense)[:k_eff]), "dense"


def smallest_eigenvalues(
    laplacian: csr_matrix,
    k: int = 50,
    *,
    dense_threshold: int = 2000,
    tol: float = 0.0,
    maxiter: int | None = None,
    n_components: int | None = None,
    component_labels: IntArray | None = None,
) -> Spectrum:
    """Compute the ``k`` algebraically smallest eigenvalues of ``L``.

    When ``component_labels`` identifies more than one connected component the
    Laplacian is block diagonal, and the spectrum is assembled from the blocks'
    spectra.  That path is not merely an optimisation -- it is a correctness
    requirement.  A graph with many components has a kernel whose dimension
    equals that component count, and no Krylov method can resolve a degeneracy
    of that order: ARPACK converges to an arbitrary subset of the invariant
    subspace and silently returns eigenvalues that are *not* the smallest.  On
    the Chen-Engel-Glotzer packing (686 components) whole-matrix shift-invert
    reports 53 zeros and 7 threes instead of 60 zeros; the block path returns
    the exact answer.

    Parameters
    ----------
    laplacian:
        Symmetric PSD sparse Laplacian.
    k:
        Number of eigenvalues requested.  Clamped to the matrix dimension.
    dense_threshold:
        Blocks with dimension at or below this use dense LAPACK directly.
    tol:
        ARPACK relative tolerance; ``0`` requests machine precision.
    maxiter:
        ARPACK iteration cap.  ``None`` selects a generous default.
    n_components:
        Known number of connected components, recorded for cross-checking the
        multiplicity of the zero eigenvalue.
    component_labels:
        Per-vertex component index.  Supplying it together with
        ``n_components > 1`` enables the block-diagonal path.

    Returns
    -------
    Spectrum
    """
    if not issparse(laplacian):
        raise TypeError("laplacian must be a scipy.sparse matrix")
    if laplacian.shape[0] != laplacian.shape[1]:
        raise ValueError(f"laplacian must be square, got {laplacian.shape}")
    n = laplacian.shape[0]
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n < 1:
        raise ValueError("laplacian must be non-empty")

    k_eff = min(k, n)
    ztol = _zero_tolerance(laplacian)

    if component_labels is not None and n_components is not None and n_components > 1:
        LOGGER.info("spectrum: block-diagonal over %d connected components", n_components)

        def solve_block(block: csr_matrix, kb: int) -> FloatArray:
            return _single_block_eigenvalues(
                block, kb, dense_threshold=dense_threshold, tol=tol, maxiter=maxiter
            )[0]

        values = _block_diagonal_spectrum(
            laplacian,
            k_eff,
            np.asarray(component_labels, dtype=np.int64),
            int(n_components),
            solve_block,
        )
        method = "block-diagonal"
    else:
        values, method = _single_block_eigenvalues(
            laplacian, k_eff, dense_threshold=dense_threshold, tol=tol, maxiter=maxiter
        )

    values = np.sort(np.real(np.asarray(values, dtype=F64)))
    # Eigenvalues of a PSD operator cannot be negative; clip round-off.
    values[np.abs(values) <= ztol] = 0.0
    return Spectrum(
        eigenvalues=np.ascontiguousarray(values, dtype=F64),
        k_requested=k,
        method=method,  # type: ignore[arg-type]
        matrix_dimension=n,
        n_components=int(n_components) if n_components is not None else -1,
        zero_tolerance=ztol,
    )


# --------------------------------------------------------------------------- #
# Degeneracy and spacing analysis
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DegeneracyReport:
    """Degenerate-level decomposition of a spectrum."""

    levels: FloatArray
    multiplicities: IntArray
    level_spacings: FloatArray
    raw_spacings: FloatArray
    degeneracy_tolerance: float
    gap_ratio_mean: float | None
    multiplicity_histogram: dict[int, int] = field(default_factory=dict)

    @property
    def n_levels(self) -> int:
        return int(self.levels.shape[0])

    @property
    def max_multiplicity(self) -> int:
        return int(self.multiplicities.max()) if self.multiplicities.size else 0

    @property
    def degenerate_fraction(self) -> float:
        """Fraction of eigenvalues belonging to a level of multiplicity > 1."""
        if not self.multiplicities.size:
            return 0.0
        total = int(self.multiplicities.sum())
        degenerate = int(self.multiplicities[self.multiplicities > 1].sum())
        return degenerate / total


#: Minimum number of adjacent-gap ratios before their mean is reported.  The
#: statistic is a distributional one; quoting it from two or three spacings
#: invites reading a symmetry class out of pure noise.
MIN_GAP_RATIO_SAMPLES: int = 8


def analyse_degeneracies(
    eigenvalues: FloatArray,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-9,
    min_ratio_samples: int = MIN_GAP_RATIO_SAMPLES,
) -> DegeneracyReport:
    """Group eigenvalues into degenerate levels and characterise the spacings.

    Two consecutive eigenvalues join the same level when their separation
    falls within ``atol + rtol * |lambda|``.  Because eigenvalue error scales
    with the magnitude of the eigenvalue, a purely absolute threshold would
    over-merge at the top of the spectrum and under-merge at the bottom.

    The adjacent-gap ratio ``r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})`` is
    computed over *distinct* levels.  Its mean discriminates uncorrelated
    spectra (:data:`POISSON_RATIO`, ~0.386) from level-repelling ones
    (:data:`GOE_RATIO`, ~0.536).  Highly symmetric operators sit near the
    Poisson value because independent symmetry sectors superpose without
    interacting -- itself a symmetry signature.

    Parameters
    ----------
    eigenvalues:
        Sorted (or sortable) real eigenvalues.
    rtol, atol:
        Relative and absolute degeneracy tolerances.

    Returns
    -------
    DegeneracyReport
    """
    vals = np.sort(np.asarray(eigenvalues, dtype=F64).ravel())
    if vals.size == 0:
        raise ValueError("cannot analyse an empty spectrum")

    raw_spacings = np.diff(vals) if vals.size > 1 else np.zeros(0, dtype=F64)

    tolerances = atol + rtol * np.abs(vals[:-1])
    new_level = np.concatenate([[True], raw_spacings > tolerances])
    level_index = np.cumsum(new_level) - 1

    n_levels = int(level_index[-1]) + 1
    multiplicities = np.bincount(level_index, minlength=n_levels).astype(np.int64)
    sums = np.bincount(level_index, weights=vals, minlength=n_levels)
    levels = np.ascontiguousarray(sums / multiplicities, dtype=F64)

    level_spacings = np.diff(levels) if levels.size > 1 else np.zeros(0, dtype=F64)

    gap_ratio_mean: float | None = None
    if level_spacings.size >= 2:
        s1, s2 = level_spacings[:-1], level_spacings[1:]
        hi = np.maximum(s1, s2)
        usable = hi > 0.0
        ratios = np.minimum(s1, s2)[usable] / hi[usable]
        if ratios.size >= min_ratio_samples:
            gap_ratio_mean = float(ratios.mean())
        elif ratios.size:
            LOGGER.info(
                "adjacent-gap ratio suppressed: %d sample(s), need %d for the mean "
                "to carry any distributional meaning",
                ratios.size,
                min_ratio_samples,
            )

    counts = np.bincount(multiplicities)
    histogram = {int(m): int(c) for m, c in enumerate(counts) if c and m > 0}

    return DegeneracyReport(
        levels=levels,
        multiplicities=multiplicities,
        level_spacings=np.ascontiguousarray(level_spacings, dtype=F64),
        raw_spacings=np.ascontiguousarray(raw_spacings, dtype=F64),
        degeneracy_tolerance=float(atol),
        gap_ratio_mean=gap_ratio_mean,
        multiplicity_histogram=histogram,
    )


# --------------------------------------------------------------------------- #
# Tabular views
# --------------------------------------------------------------------------- #
def spectrum_frame(spectrum: Spectrum, limit: int | None = None) -> pd.DataFrame:
    """Per-eigenvalue table: value, consecutive spacing, and zero flag."""
    vals = spectrum.eigenvalues if limit is None else spectrum.eigenvalues[:limit]
    spacings = np.concatenate([[np.nan], np.diff(vals)]) if vals.size else np.zeros(0)
    return pd.DataFrame(
        {
            "n": np.arange(vals.size, dtype=np.int64),
            "lambda_n": vals,
            "delta_lambda_n": spacings,
            "is_zero": np.abs(vals) <= spectrum.zero_tolerance,
        }
    )


def level_frame(report: DegeneracyReport, limit: int | None = None) -> pd.DataFrame:
    """Per-level table: distinct eigenvalue, multiplicity, and level spacing."""
    levels = report.levels if limit is None else report.levels[:limit]
    mult = report.multiplicities[: levels.size]
    spacings = np.concatenate([[np.nan], np.diff(levels)]) if levels.size else np.zeros(0)
    return pd.DataFrame(
        {
            "level": np.arange(levels.size, dtype=np.int64),
            "lambda": levels,
            "multiplicity": mult,
            "spacing_to_previous": spacings,
        }
    )


def multiplicity_frame(report: DegeneracyReport) -> pd.DataFrame:
    """Histogram of level multiplicities, annotated with O_h irrep dimensions.

    The octahedral group has irreducible representations of dimension 1, 1, 2,
    3 and 3.  A multiplicity matching one of those is consistent with a single
    irrep; larger values indicate either an accidental coincidence of distinct
    irreps or a symmetry group larger than :math:`O_h`.
    """
    oh_dims = {1, 2, 3}
    rows = []
    total_levels = max(report.n_levels, 1)
    for mult, count in sorted(report.multiplicity_histogram.items()):
        rows.append(
            {
                "multiplicity": mult,
                "n_levels": count,
                "n_eigenvalues": mult * count,
                "fraction_of_levels": count / total_levels,
                "matches_O_h_irrep_dim": mult in oh_dims,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Periodic (crystal) graph analysis
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PeriodicComponent:
    """One connected piece of the quotient graph, and what it unrolls into."""

    orbits: tuple[int, ...]
    n_cycles: int
    sublattice_basis: tuple[tuple[int, int, int], ...] | None
    index: int | None
    rank: int

    @property
    def is_finite(self) -> bool:
        """Whether this piece unrolls into finitely many infinite networks."""
        return self.index is not None

    @property
    def dimensionality(self) -> int:
        """How far a single connected piece actually extends through space.

        The rank of the cycle-voltage lattice *is* the dimensionality of the
        component: rank 0 means it closes up into a bounded cluster (CEG's
        isolated dimers), rank 1 an infinite chain, rank 2 an infinite sheet,
        rank 3 a network filling all three directions.  Only rank 3 gives a
        finite component count; anything lower repeats forever transverse to
        itself.
        """
        return self.rank


@dataclass(frozen=True)
class PeriodicGraphReport:
    """Connected components of the *infinite* periodic unit-distance graph."""

    n_orbits: int
    components: tuple[PeriodicComponent, ...]

    @property
    def n_infinite_components(self) -> int | None:
        """Total number of infinite networks, or ``None`` if any piece is unbounded."""
        if any(not c.is_finite for c in self.components):
            return None
        return sum(int(c.index) for c in self.components)


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return (abs(a), 1 if a >= 0 else -1, 0)
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def _hermite_basis(vectors) -> list[list[int] | None]:
    """Column-echelon basis of the subgroup of ``Z^3`` generated by ``vectors``.

    Exact integer arithmetic throughout -- no symbolic dependency, and no
    floating-point determinant that could round a singular matrix into an
    invertible one.  Slot ``k`` holds the basis vector whose leading nonzero
    entry is in column ``k``, or ``None`` when the subgroup has no such vector.
    """
    basis: list[list[int] | None] = [None, None, None]
    for vector in vectors:
        current = [int(c) for c in vector]
        for column in range(3):
            if current[column] == 0:
                continue
            existing = basis[column]
            if existing is None:
                basis[column] = current
                current = [0, 0, 0]
                break
            gcd, x, y = _extended_gcd(existing[column], current[column])
            combined = [x * existing[k] + y * current[k] for k in range(3)]
            reduced = [
                (existing[column] // gcd) * current[k]
                - (current[column] // gcd) * existing[k]
                for k in range(3)
            ]
            basis[column], current = combined, reduced

    return basis


def sublattice_index(vectors) -> tuple[int | None, tuple[tuple[int, int, int], ...] | None]:
    """Index of the subgroup of ``Z^3`` generated by integer ``vectors``.

    Returns ``(None, None)`` when the generators do not span -- the subgroup
    then has infinite index, meaning that piece of the graph unrolls into
    infinitely many components rather than finitely many.  Use
    :func:`sublattice_rank` when the *reason* matters.
    """
    basis = _hermite_basis(vectors)
    if any(row is None for row in basis):
        return None, None
    index = abs(basis[0][0] * basis[1][1] * basis[2][2])  # type: ignore[index]
    return index, tuple(tuple(row) for row in basis)  # type: ignore[arg-type]


def sublattice_rank(vectors) -> int:
    """Rank of the subgroup of ``Z^3`` generated by integer ``vectors``."""
    return sum(1 for row in _hermite_basis(vectors) if row is not None)


def periodic_graph_components(
    cell: FloatArray,
    lattice: FloatArray,
    *,
    distance: float = 1.0,
    atol: float = 1e-8,
    span: int = 3,
) -> PeriodicGraphReport:
    """Count the connected components of the *infinite* periodic graph.

    A finite block cannot answer this: its component count mixes genuine
    networks with pieces the boundary happened to sever.  The quotient graph
    does answer it exactly.  Reduce the structure to one unit cell, so vertices
    become orbits under lattice translation; label each edge with the
    translation it crosses (its *voltage*); then a quotient component unrolls
    into exactly ``[Z^3 : L]`` infinite networks, where ``L`` is the subgroup
    generated by the voltages around its cycles.  Intuitively, a cycle whose
    voltages only ever sum to even translations can never reach an odd cell, so
    the odd cells must hold a separate copy.

    Parameters
    ----------
    cell:
        ``(n, 4, 3)`` tetrahedra of one unit cell.
    lattice:
        ``(3, 3)`` lattice with vectors as rows.
    distance, atol:
        Edge criterion, matching :func:`build_unit_distance_graph`.
    span:
        Range of lattice translations searched for edges.

    Returns
    -------
    PeriodicGraphReport
    """
    if cell.ndim != 3 or cell.shape[1:] != (4, 3):
        raise ValueError(f"cell must have shape (n, 4, 3); got {cell.shape}")
    if lattice.shape != (3, 3):
        raise ValueError(f"lattice must have shape (3, 3); got {lattice.shape}")
    if span < 1:
        raise ValueError(f"span must be at least 1; got {span}")

    inverse = np.linalg.inv(lattice)
    fractional = cell.reshape(-1, 3) @ inverse
    fractional -= np.floor(fractional)

    orbits: list[NDArray[np.float64]] = []
    for point in fractional:
        for existing in orbits:
            offset = point - existing
            offset -= np.round(offset)
            if np.linalg.norm(offset @ lattice) < atol:
                break
        else:
            orbits.append(point)
    positions = np.array(orbits) @ lattice

    neighbours: dict[int, list[tuple[int, NDArray[np.int64]]]] = {
        i: [] for i in range(len(orbits))
    }
    shifts = np.array(
        list(itertools.product(range(-span, span + 1), repeat=3)), dtype=np.int64
    )
    translations = shifts.astype(F64) @ lattice
    for u in range(len(orbits)):
        for v in range(len(orbits)):
            separations = np.linalg.norm(
                positions[u] - (positions[v] + translations), axis=1
            )
            hits = np.flatnonzero(np.abs(separations - distance) < atol)
            for hit in hits:
                shift = shifts[hit]
                if u == v and not shift.any():
                    continue
                neighbours[u].append((v, shift))

    potential: dict[int, NDArray[np.int64]] = {}
    components: list[PeriodicComponent] = []
    for start in range(len(orbits)):
        if start in potential:
            continue
        potential[start] = np.zeros(3, dtype=np.int64)
        members = [start]
        stack = [start]
        cycles: list[NDArray[np.int64]] = []
        while stack:
            u = stack.pop()
            for v, shift in neighbours[u]:
                if v not in potential:
                    potential[v] = potential[u] + shift
                    members.append(v)
                    stack.append(v)
                else:
                    cycles.append(potential[u] + shift - potential[v])
        voltages = [c for c in cycles if c.any()]
        index, basis = sublattice_index(voltages)
        components.append(
            PeriodicComponent(
                orbits=tuple(sorted(members)),
                n_cycles=len(cycles),
                sublattice_basis=basis,
                index=index,
                rank=sublattice_rank(voltages),
            )
        )

    return PeriodicGraphReport(n_orbits=len(orbits), components=tuple(components))
