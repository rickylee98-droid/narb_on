"""Community detection as a failure of the Ramanujan bound.

The prediction, and where it comes from
---------------------------------------
`selberg.py` establishes a dichotomy for the poles of the Ihara zeta function.
Each adjacency eigenvalue ``lambda`` of a ``(q+1)``-regular graph contributes two
non-backtracking eigenvalues, the roots of ``x^2 - lambda x + q``, and

* if ``|lambda| <= 2 sqrt q`` the roots are a conjugate pair of modulus **exactly**
  ``sqrt q`` -- they sit on a circle;
* if ``|lambda| > 2 sqrt q`` they are real and the larger strictly **escapes** it.

So "Ramanujan" means "nothing outside the circle", and a graph that fails the
bound is a graph with an eigenvalue outside it.  The content of this module is
that the escaping eigenvalue is not a defect but a *signal*: it is what planted
structure looks like spectrally, and its distance outside the circle measures how
much structure there is.

That converts a verified statement about zeta functions into a detector with a
threshold that is predicted rather than tuned.  For a sparse graph of average
degree ``c``, the bulk of the non-backtracking spectrum fills the disc of radius
``sqrt(c - 1)``, and a planted partition is visible exactly when it pushes an
eigenvalue past that radius.  For the two-group stochastic block model with
in-group and out-group parameters ``a`` and ``b``, average degree
``c = (a+b)/2`` and assortativity ``eps = (a-b)/(a+b)``, the second
non-backtracking eigenvalue sits near ``c * eps``, giving

    detectable  <=>  c * eps > sqrt(c - 1)  <=>  eps > sqrt(c - 1) / c ,

which is the Kesten--Stigum threshold.  Below it no algorithm is believed to do
better than chance; above it, the eigenvector of the escaping eigenvalue
recovers the partition.

Why non-backtracking rather than the adjacency matrix
-----------------------------------------------------
On a *sparse* graph the adjacency spectrum is ruined by degree fluctuations: a
vertex of unusually high degree produces its own large eigenvalue, localised on
that vertex and its neighbours, carrying no information about the partition.
Those spurious eigenvalues drown the signal well before the theoretical
threshold.

The non-backtracking operator does not have them, and the reason is the same
reason it appears in the zeta function at all: a walk that cannot immediately
reverse cannot linger on a single high-degree vertex.  The dichotomy above is the
precise statement -- the bulk is confined to a circle whatever the degree
distribution does, so anything outside is structure.

This module makes that comparison quantitative rather than rhetorical: both
detectors are run on the same graphs across the threshold, and the accuracy of
each is measured against the planted truth.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

import selberg

__all__ = [
    "sparse_hashimoto",
    "nonbacktracking_spectrum",
    "bulk_radius",
    "kesten_stigum_threshold",
    "empirical_bulk_edge",
    "finite_size_threshold",
    "stochastic_block_model",
    "DetectionResult",
    "detect_communities",
    "adjacency_partition",
    "overlap",
    "SweepPoint",
    "threshold_sweep",
]

LOGGER = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


# --------------------------------------------------------------------------- #
# The operator, sparse and floating point
# --------------------------------------------------------------------------- #
def sparse_hashimoto(graph: nx.Graph) -> tuple[csr_matrix, tuple[tuple[int, int], ...]]:
    """The non-backtracking operator as a sparse float matrix.

    `selberg.hashimoto_operator` builds the same matrix in exact Python integers,
    which is what the trace identities there require and what makes it unusable
    beyond a few hundred directed edges.  Eigenvalues do not need exact
    arithmetic, so this route trades exactness for the ability to run on graphs
    with hundreds of thousands of directed edges.

    Refereed against the exact operator on small graphs.
    """
    # Reuse selberg's enumeration rather than repeating its convention: edges are
    # canonicalised to (min, max), sorted lexicographically, and each emits the
    # forward orientation at an even index and the reverse at the next odd one.
    # Matching it exactly is what lets the two operators be compared entrywise
    # instead of only up to a permutation.
    edges = list(selberg.directed_edges(graph))
    index = {edge: i for i, edge in enumerate(edges)}
    rows: list[int] = []
    cols: list[int] = []
    for i, (u, v) in enumerate(edges):
        for w in graph.neighbors(v):
            if w == u:
                continue
            rows.append(i)
            cols.append(index[(v, w)])
    size = len(edges)
    data = np.ones(len(rows), dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(size, size)), tuple(edges)


def nonbacktracking_spectrum(
    graph: nx.Graph, count: int = 6
) -> NDArray[np.complex128]:
    """The ``count`` largest-modulus eigenvalues of the non-backtracking operator.

    ``B`` is not symmetric, so the eigenvalues are complex and a general sparse
    solver is required; ARPACK on the sparse operator costs a matrix-vector
    product per iteration, which is linear in the number of edges.
    """
    operator, edges = sparse_hashimoto(graph)
    size = operator.shape[0]
    if size <= max(count + 2, 20):
        dense = operator.toarray()
        values = np.linalg.eigvals(dense)
    else:
        values = eigs(
            operator, k=min(count, size - 2), which="LM", return_eigenvectors=False
        )
    order = np.argsort(-np.abs(values))
    return values[order][:count]


def bulk_radius(graph: nx.Graph) -> float:
    """``sqrt(c - 1)`` for average degree ``c``: the circle the bulk sits on.

    For a regular graph this is ``sqrt q`` exactly, which is where
    `selberg.spectral_radii` puts every non-trivial pole of a Ramanujan graph.
    For an irregular sparse graph the same radius describes the bulk, with ``c``
    the mean degree.
    """
    degrees = [d for _, d in graph.degree()]
    if not degrees:
        raise ValueError("graph has no vertices")
    mean = float(np.mean(degrees))
    if mean <= 1.0:
        raise ValueError(f"mean degree {mean} is too small for a bulk radius")
    return math.sqrt(mean - 1.0)


def kesten_stigum_threshold(mean_degree: float) -> float:
    """Assortativity at which the signal eigenvalue reaches the bulk radius.

    The second non-backtracking eigenvalue of a two-group block model sits near
    ``c * eps``, and the bulk edge is at ``sqrt(c - 1)``, so the crossing is at
    ``eps = sqrt(c - 1) / c``.  This is a prediction with no fitted constant.
    """
    if mean_degree <= 1.0:
        raise ValueError(f"mean degree must exceed 1; got {mean_degree}")
    return math.sqrt(mean_degree - 1.0) / mean_degree


def empirical_bulk_edge(
    size: int, mean_degree: float, *, samples: int = 3, seed: int = 0
) -> float:
    """Measure where the bulk actually ends, at finite ``size``, with no signal.

    The radius ``sqrt(c-1)`` is asymptotic.  At finite size the bulk edge sits
    above it -- at ``c = 5``, ``n = 4000`` the measured value is about ``2.27``
    against an asymptotic ``2.00`` -- because the spectrum of a finite sparse
    graph has fluctuations the limit law does not.

    Comparing a signal eigenvalue against the asymptotic radius therefore reports
    "escape" for a graph with no structure whatever.  This function supplies the
    honest null: the same statistic on the same size and degree at zero
    assortativity, which is what a detection claim must actually beat.
    """
    values = []
    for sample in range(samples):
        graph, _ = stochastic_block_model(
            size, mean_degree, 0.0, seed=seed + 7919 * sample
        )
        spectrum = nonbacktracking_spectrum(graph, count=4)
        if len(spectrum) > 1:
            values.append(float(np.abs(spectrum[1])))
    if not values:
        raise ValueError("no sample produced a second eigenvalue")
    return float(np.mean(values))


def finite_size_threshold(
    size: int, mean_degree: float, *, samples: int = 3, seed: int = 0
) -> float:
    """Assortativity at which ``c * eps`` reaches the *measured* bulk edge.

    The asymptotic Kesten--Stigum value ``sqrt(c-1)/c`` is the ``size -> infinity``
    limit of this.  At ``c = 5``, ``n = 4000`` the asymptotic threshold is
    ``0.400`` while this returns about ``0.454``, and detection is observed to
    begin at the latter -- so the finite-size version is the one that predicts
    the experiment.
    """
    return empirical_bulk_edge(size, mean_degree, samples=samples, seed=seed) / mean_degree


# --------------------------------------------------------------------------- #
# The test problem
# --------------------------------------------------------------------------- #
def stochastic_block_model(
    size: int,
    mean_degree: float,
    assortativity: float,
    *,
    seed: int = 0,
) -> tuple[nx.Graph, NDArray[np.int_]]:
    """Two equal planted groups, with a given mean degree and signal strength.

    ``assortativity`` is ``eps = (a - b)/(a + b)``, so ``eps = 0`` is a pure
    Erdos--Renyi graph with no structure at all and ``eps = 1`` disconnects the
    groups entirely.  Returns the graph together with the planted labels.

    Only the largest connected component is returned: the block model at sparse
    average degree always throws off small components and isolated vertices,
    which carry no signal and which the detection threshold statement is not
    about.
    """
    if size % 2:
        raise ValueError(f"size must be even to split into two groups; got {size}")
    if not 0.0 <= assortativity <= 1.0:
        raise ValueError(f"assortativity must lie in [0, 1]; got {assortativity}")
    inside = mean_degree * (1.0 + assortativity)
    between = mean_degree * (1.0 - assortativity)
    rng = np.random.default_rng(seed)
    labels = np.zeros(size, dtype=int)
    labels[size // 2 :] = 1
    graph = nx.Graph()
    graph.add_nodes_from(range(size))
    # Sample each unordered pair once, at the probability its label pair dictates.
    # The block list is written out rather than generated by a nested loop: the
    # obvious `for b in (a, 1)` visits (1,1) twice, which samples group one's
    # internal edges at double rate.  That leaves the two groups with different
    # densities -- structure that was never asked for -- and raises the mean
    # degree above the requested value, moving the very threshold being tested.
    for group_a, group_b in ((0, 0), (0, 1), (1, 1)):
        rate = inside if group_a == group_b else between
        probability = rate / size
        first = np.nonzero(labels == group_a)[0]
        second = np.nonzero(labels == group_b)[0]
        if group_a == group_b:
            pairs = np.array(
                [(u, v) for i, u in enumerate(first) for v in first[i + 1 :]]
            )
        else:
            pairs = np.array([(u, v) for u in first for v in second])
        if pairs.size == 0:
            continue
        keep = rng.random(len(pairs)) < probability
        graph.add_edges_from(map(tuple, pairs[keep]))
    component = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(component).copy()
    nodes = sorted(subgraph)
    relabelled = nx.relabel_nodes(subgraph, {n: i for i, n in enumerate(nodes)})
    return relabelled, labels[nodes]


def overlap(predicted: Sequence[int], truth: Sequence[int]) -> float:
    """Accuracy above chance, normalised so that random guessing scores zero.

    Two labels can be swapped without changing a partition, so the raw agreement
    is maximised over the relabelling; and a coin flip already scores ``1/2``, so
    the result is rescaled to put chance at ``0`` and perfection at ``1``.
    """
    predicted = np.asarray(predicted)
    truth = np.asarray(truth)
    if predicted.shape != truth.shape:
        raise ValueError(
            f"shape mismatch: {predicted.shape} predicted against {truth.shape}"
        )
    agree = float(np.mean(predicted == truth))
    best = max(agree, 1.0 - agree)
    return max(0.0, 2.0 * best - 1.0)


# --------------------------------------------------------------------------- #
# The detectors
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DetectionResult:
    """One graph, one detector, with the spectral quantity that drove it."""

    labels: NDArray[np.int_]
    signal_eigenvalue: float
    bulk_radius: float

    @property
    def escapes_bulk(self) -> bool:
        """Whether an eigenvalue left the circle -- the Ramanujan bound failing."""
        return self.signal_eigenvalue > self.bulk_radius

    @property
    def margin(self) -> float:
        """How far outside the circle, in units of the radius."""
        return self.signal_eigenvalue / self.bulk_radius - 1.0


def detect_communities(graph: nx.Graph) -> DetectionResult:
    """Partition by the eigenvector of the escaping non-backtracking eigenvalue.

    The largest eigenvalue of ``B`` is the Perron root near ``c - 1`` and carries
    no partition information; the *second* is the one that either sits on the
    bulk circle -- no structure -- or escapes it, in which case its eigenvector
    separates the groups.

    The eigenvector lives on directed edges, so it is pushed down to vertices by
    summing over the edges pointing into each one.
    """
    operator, edges = sparse_hashimoto(graph)
    size = operator.shape[0]
    if size < 10:
        raise ValueError(f"graph is too small to partition spectrally: {size} arcs")
    if size <= 40:
        values, vectors = np.linalg.eig(operator.toarray())
    else:
        values, vectors = eigs(operator, k=min(4, size - 2), which="LM")
    order = np.argsort(-np.abs(values))
    values, vectors = values[order], vectors[:, order]

    second = vectors[:, 1] if vectors.shape[1] > 1 else vectors[:, 0]
    scores = np.zeros(graph.number_of_nodes(), dtype=np.float64)
    for weight, (_, head) in zip(np.real(second), edges):
        scores[head] += float(weight)
    labels = (scores > np.median(scores)).astype(int)
    return DetectionResult(
        labels=labels,
        signal_eigenvalue=float(np.abs(values[1])) if len(values) > 1 else 0.0,
        bulk_radius=bulk_radius(graph),
    )


def adjacency_partition(graph: nx.Graph) -> DetectionResult:
    """The classical comparison: partition by the second adjacency eigenvector.

    Included so that the claim "non-backtracking is better in the sparse regime"
    is measured rather than asserted.  On a sparse graph this detector is
    expected to be defeated by degree fluctuations well before the threshold,
    because a high-degree vertex produces a large eigenvalue whose eigenvector is
    localised on it and says nothing about the partition.
    """
    nodes = sorted(graph)
    matrix = nx.to_scipy_sparse_array(graph, nodelist=nodes, dtype=np.float64)
    size = matrix.shape[0]
    if size <= 20:
        values, vectors = np.linalg.eigh(matrix.toarray())
        order = np.argsort(-values)
        values, vectors = values[order], vectors[:, order]
    else:
        from scipy.sparse.linalg import eigsh

        values, vectors = eigsh(matrix, k=min(4, size - 2), which="LA")
        order = np.argsort(-values)
        values, vectors = values[order], vectors[:, order]
    second = np.real(vectors[:, 1])
    labels = (second > np.median(second)).astype(int)
    return DetectionResult(
        labels=labels,
        signal_eigenvalue=float(values[1]),
        bulk_radius=bulk_radius(graph),
    )


# --------------------------------------------------------------------------- #
# The experiment
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SweepPoint:
    """One assortativity, averaged over graph samples."""

    assortativity: float
    mean_degree: float
    threshold: float
    nonbacktracking_overlap: float
    adjacency_overlap: float
    signal_eigenvalue: float
    bulk_radius: float
    samples: int

    @property
    def above_threshold(self) -> bool:
        return self.assortativity > self.threshold

    @property
    def escapes_bulk(self) -> bool:
        return self.signal_eigenvalue > self.bulk_radius


def threshold_sweep(
    assortativities: Sequence[float],
    *,
    size: int = 2000,
    mean_degree: float = 5.0,
    samples: int = 3,
    seed: int = 0,
) -> list[SweepPoint]:
    """Run both detectors across the predicted threshold.

    The prediction under test is that the non-backtracking overlap departs from
    zero at ``eps = sqrt(c-1)/c`` and not before, and that the escape of the
    second eigenvalue from the bulk circle happens at the same place -- the two
    being the same event seen from the algorithm and from the spectrum.
    """
    threshold = kesten_stigum_threshold(mean_degree)
    points: list[SweepPoint] = []
    for assortativity in assortativities:
        nb_scores, adjacency_scores, signals, radii = [], [], [], []
        for sample in range(samples):
            graph, truth = stochastic_block_model(
                size, mean_degree, assortativity, seed=seed + 1000 * sample
            )
            nb = detect_communities(graph)
            adjacency = adjacency_partition(graph)
            nb_scores.append(overlap(nb.labels, truth))
            adjacency_scores.append(overlap(adjacency.labels, truth))
            signals.append(nb.signal_eigenvalue)
            radii.append(nb.bulk_radius)
        points.append(
            SweepPoint(
                assortativity=assortativity,
                mean_degree=mean_degree,
                threshold=threshold,
                nonbacktracking_overlap=float(np.mean(nb_scores)),
                adjacency_overlap=float(np.mean(adjacency_scores)),
                signal_eigenvalue=float(np.mean(signals)),
                bulk_radius=float(np.mean(radii)),
                samples=samples,
            )
        )
    return points
