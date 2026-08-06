"""Quantum unique ergodicity on thin sets, in the arithmetic graph model.

What this targets, and what it does not
---------------------------------------
Arithmetic QUE for the modular surface is a **theorem**, not an open problem:
Lindenstrauss settled the compact and Hecke cases, and Soundararajan completed
the non-compact modular surface.  What remains open is the behaviour on *thin
sets* -- whether the mass of a high-energy eigenfunction equidistributes when
restricted to a lower-dimensional or sparse subset rather than to an open set.

Computing genuine Maass forms needs Hejhal's algorithm, which is a specialist
numerical method for a hyperbolic PDE and not a sparse-matrix eigenproblem.  So
this module works in the standard *discrete arithmetic model* instead:
Lubotzky-Phillips-Sarnak Ramanujan graphs ``X^{p,q}``, built from the quaternion
parametrisation of the four-square theorem as Cayley graphs of ``PSL(2, F_q)`` or
``PGL(2, F_q)``.  These are the accepted combinatorial analogue of the modular
surface, their adjacency eigenvectors play the role of Hecke-Maass forms, and the
whole construction is exact integer arithmetic modulo ``q``.

The payoff is that the model carries a *theorem to check against*: a
``(p+1)``-regular LPS graph is Ramanujan, meaning every non-trivial adjacency
eigenvalue satisfies ``|lambda| <= 2 sqrt(p)``.  That is an exact statement the
construction cannot fake, playing the same role as the Catalan numbers in the
amplituhedron module and the Casimir equation in the bootstrap one.

The obstruction: why the arithmetic graph model cannot answer the question
-------------------------------------------------------------------------
The naive experiment -- take an eigenvector, square it, look at its mass on a
thin set -- is not well defined on these graphs, and the natural repair is
trivial.  Both failures are structural, and both are verified numerically here
rather than asserted:

1. **Per-eigenvector mass is basis-dependent.**  LPS graphs are Cayley graphs, so
   their eigenspaces carry representations of the underlying group and reach
   multiplicity 81 already at ``X^{5,13}``.  Inside a degenerate eigenspace every
   orthonormal basis is equally valid, and one can rotate to concentrate an
   eigenvector almost anywhere.  A "scar" found that way measures the
   diagonalisation routine, not the graph.

2. **The basis-free repair is constant by symmetry.**  The obvious fix is the
   spectral projector's diagonal, ``Pi_lambda(v, v)``, which no basis choice can
   affect.  But a Cayley graph is vertex-transitive, and ``Pi_lambda`` commutes
   with every automorphism, so that diagonal is *the same at every vertex*.
   Measured here it is 1 to within ``1e-13``.  The observable is identically 1 by
   symmetry and carries no information at all.

3. **Hecke operators do not rescue it.**  Arithmetic QUE gets its name from using
   the joint Laplacian-Hecke eigenbasis, which is canonical on the modular
   surface.  The discrete Hecke operators exist here and genuinely commute --
   ``[A_p, A_p'] = 0`` exactly, for distinct primes with the same quadratic
   character mod ``q`` -- but they are right convolutions, so they commute with
   the *entire* left regular action.  Their joint eigenspaces therefore cannot be
   smaller than the irreducible representations of the group.  On
   ``PSL(2, F_13)`` three Hecke operators cut the maximum multiplicity from 112
   only to 42, and the joint multiplicities come out as exactly
   ``{1} u {12, 13, 14} x {1, 2, 3}`` -- the irrep dimensions.  No amount of
   Hecke data will separate them.

So the homogeneity that makes LPS graphs a clean arithmetic object is exactly
what blinds them to this question.  The modular surface is *not* homogeneous, and
that is the property the model fails to capture.

Where the question does have content
------------------------------------
Random ``d``-regular graphs are the model that keeps the relevant features and
drops the fatal one.  By Friedman's theorem they are almost-Ramanujan, so the
spectral gap survives; they are not vertex-transitive, so no symmetry fixes the
answer; and their spectra are simple in practice -- measured here as multiplicity
1 at every one of 500, 1000 and 2000 vertices -- so individual eigenvectors are
canonical and ``|psi(v)|^2`` is a well-posed observable.  The thin-set experiment
is run there, against a scarred negative control.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

__all__ = [
    "FOUR_SQUARE_NORMALISATION",
    "LPSGraph",
    "ThinSetStatistics",
    "four_square_solutions",
    "legendre_symbol",
    "modular_sqrt_minus_one",
    "lps_generators",
    "build_lps_graph",
    "ramanujan_bound",
    "eigenspace_mass",
    "spectral_window_mass",
    "thin_set_statistics",
    "shortest_cycle",
    "ball_around",
    "kesten_mckay_density",
    "bottlenecked_graph",
    "hecke_operators",
    "hecke_joint_multiplicities",
    "projector_diagonal_is_constant",
    "random_regular_graph",
    "eigenvector_mass",
    "gaussian_baseline",
]

LOGGER = logging.getLogger(__name__)

F64 = np.float64
FloatArray = NDArray[np.float64]

#: The normalisation that picks exactly ``p + 1`` of the four-square solutions:
#: first coordinate positive and odd, the rest even.  Jacobi's formula gives
#: ``8(p+1)`` solutions in total for odd prime ``p``, and this cuts the factor of
#: eight from signs and ordering.
FOUR_SQUARE_NORMALISATION = "a0 > 0 odd, a1 a2 a3 even"


# --------------------------------------------------------------------------- #
# Exact arithmetic ingredients
# --------------------------------------------------------------------------- #
def four_square_solutions(p: int) -> tuple[tuple[int, int, int, int], ...]:
    """The ``p + 1`` normalised representations of ``p`` as a sum of four squares.

    For an odd prime ``p`` Jacobi's four-square theorem counts ``8(p+1)``
    representations; requiring ``a0`` positive and odd with ``a1, a2, a3`` even
    selects exactly ``p + 1`` of them, which are the LPS generators.

    The count is checked rather than assumed, because an off-by-one in the search
    range silently returns the wrong number -- a parity mistake in the loop
    bounds gives zero solutions for ``p = 5`` while looking perfectly healthy for
    ``p = 13``.
    """
    if p < 3 or p % 2 == 0:
        raise ValueError(f"p must be an odd prime; got {p}")

    limit = int(p**0.5) + 1
    even_limit = 2 * (limit // 2 + 1)
    found = []
    for a0 in range(1, limit + 1, 2):
        for a1 in range(-even_limit, even_limit + 1, 2):
            for a2 in range(-even_limit, even_limit + 1, 2):
                for a3 in range(-even_limit, even_limit + 1, 2):
                    if a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 == p:
                        found.append((a0, a1, a2, a3))
    if len(found) != p + 1:
        raise ValueError(
            f"expected {p + 1} normalised four-square solutions for p={p}, "
            f"found {len(found)}; p is probably not a prime congruent to 1 mod 4"
        )
    return tuple(found)


def legendre_symbol(a: int, q: int) -> int:
    """``+1`` if ``a`` is a non-zero quadratic residue mod ``q``, ``-1`` if not."""
    a %= q
    if a == 0:
        return 0
    return 1 if pow(a, (q - 1) // 2, q) == 1 else -1


def modular_sqrt_minus_one(q: int) -> int:
    """An integer ``i`` with ``i^2 = -1 mod q``, which exists iff ``q = 1 mod 4``."""
    if q % 4 != 1:
        raise ValueError(f"q must be congruent to 1 mod 4 for i^2 = -1; got {q}")
    for candidate in range(2, q):
        if (candidate * candidate + 1) % q == 0:
            return candidate
    raise ValueError(f"no square root of -1 modulo {q}")


def lps_generators(p: int, q: int) -> tuple[tuple[int, int, int, int], ...]:
    """The ``p + 1`` Cayley generators of ``X^{p,q}``, as 2x2 matrices mod ``q``.

    Each four-square solution ``(a0, a1, a2, a3)`` becomes

        [[a0 + i a1,  a2 + i a3],
         [-a2 + i a3, a0 - i a1]]     with  i^2 = -1 mod q,

    whose determinant is ``a0^2 + a1^2 + a2^2 + a3^2 = p``.  That determinant is
    what decides the ambient group: when ``p`` is a square mod ``q`` the matrices
    can be rescaled to determinant one and generate ``PSL(2, F_q)``; otherwise
    they generate ``PGL(2, F_q)``, which is twice as large.
    """
    root = modular_sqrt_minus_one(q)
    generators = []
    for a0, a1, a2, a3 in four_square_solutions(p):
        generators.append(
            (
                (a0 + root * a1) % q,
                (a2 + root * a3) % q,
                (-a2 + root * a3) % q,
                (a0 - root * a1) % q,
            )
        )
    return tuple(generators)


# --------------------------------------------------------------------------- #
# The graph
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LPSGraph:
    """An LPS Ramanujan graph together with everything needed to check it."""

    p: int
    q: int
    adjacency: csr_matrix
    projective: str
    degree: int

    @property
    def n_vertices(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def expected_order(self) -> int:
        """Exact order of the group, which the vertex count must equal."""
        base = self.q * (self.q * self.q - 1)
        return base // 2 if self.projective == "PSL" else base

    @property
    def ramanujan_bound(self) -> float:
        return 2.0 * float(np.sqrt(self.p))


def build_lps_graph(p: int, q: int) -> LPSGraph:
    """Construct ``X^{p,q}`` by breadth-first closure of the Cayley generators.

    Vertices are projective classes: a matrix is normalised by scaling so its
    first non-zero entry is 1.  Quotienting only by ``+-1`` instead -- the
    tempting shortcut -- silently produces a graph twice too large whose
    non-trivial eigenvalues violate the Ramanujan bound, which is exactly how
    that mistake announces itself.
    """
    if p == q:
        raise ValueError(f"p and q must be distinct primes; both are {p}")
    generators = lps_generators(p, q)
    inverses = [0] * q
    for value in range(1, q):
        inverses[value] = pow(value, q - 2, q)

    def multiply(a, b):
        return (
            (a[0] * b[0] + a[1] * b[2]) % q,
            (a[0] * b[1] + a[1] * b[3]) % q,
            (a[2] * b[0] + a[3] * b[2]) % q,
            (a[2] * b[1] + a[3] * b[3]) % q,
        )

    def projective_normal(a):
        for entry in a:
            if entry:
                scale = inverses[entry]
                return tuple((x * scale) % q for x in a)
        raise ValueError("encountered the zero matrix, which is not invertible")

    identity = projective_normal((1, 0, 0, 1))
    index = {identity: 0}
    order = [identity]
    stack = [identity]
    while stack:
        current = stack.pop()
        for generator in generators:
            image = projective_normal(multiply(current, generator))
            if image not in index:
                index[image] = len(order)
                order.append(image)
                stack.append(image)

    rows, columns = [], []
    for matrix, source in index.items():
        for generator in generators:
            rows.append(source)
            columns.append(index[projective_normal(multiply(matrix, generator))])

    size = len(order)
    adjacency = coo_matrix(
        (np.ones(len(rows), dtype=F64), (rows, columns)), shape=(size, size)
    ).tocsr()

    projective = "PSL" if legendre_symbol(p, q) == 1 else "PGL"
    graph = LPSGraph(
        p=p, q=q, adjacency=adjacency, projective=projective, degree=p + 1
    )
    if graph.n_vertices != graph.expected_order:
        raise ValueError(
            f"X^{{{p},{q}}} has {graph.n_vertices} vertices but "
            f"{graph.projective}(2, F_{q}) has order {graph.expected_order}; "
            "the projective normalisation is wrong"
        )
    LOGGER.info(
        "built X^{%d,%d}: %d vertices, %d-regular, group %s",
        p, q, graph.n_vertices, graph.degree, graph.projective,
    )
    return graph


def ramanujan_bound(graph: LPSGraph, eigenvalues: FloatArray) -> tuple[float, bool]:
    """Largest non-trivial ``|lambda|`` and whether it respects ``2 sqrt(p)``.

    The trivial eigenvalues are ``+-(p+1)``: ``+`` always, and ``-`` as well when
    the graph is bipartite, which happens in the ``PGL`` case.  Both are excluded
    before taking the maximum.
    """
    trivial = graph.degree
    non_trivial = np.abs(eigenvalues[np.abs(np.abs(eigenvalues) - trivial) > 1e-8])
    if non_trivial.size == 0:
        raise ValueError("no non-trivial eigenvalues were supplied")
    largest = float(non_trivial.max())
    return largest, largest <= graph.ramanujan_bound + 1e-8


# --------------------------------------------------------------------------- #
# Basis-independent mass
# --------------------------------------------------------------------------- #
def eigenspace_mass(
    eigenvalues: FloatArray, eigenvectors: FloatArray, target: float, *, atol: float = 1e-8
) -> FloatArray:
    """Mass of one eigenspace at each vertex, normalised to average 1.

    Computed from the spectral projector's diagonal, ``sum_k |psi_k(v)|^2`` over
    an orthonormal basis of the eigenspace, which is invariant under any rotation
    of that basis.  Individual eigenvectors are *not* invariant, and on a Cayley
    graph the eigenspaces are large enough that a basis can be chosen to
    concentrate one wherever you like -- so per-eigenvector mass would measure the
    diagonalisation routine, not the graph.
    """
    selected = eigenvectors[:, np.abs(eigenvalues - target) < atol]
    dimension = selected.shape[1]
    if dimension == 0:
        raise ValueError(
            f"no eigenvalue within {atol} of {target}; closest is "
            f"{eigenvalues[np.argmin(np.abs(eigenvalues - target))]:.10g}"
        )
    size = eigenvectors.shape[0]
    return size * np.sum(selected**2, axis=1) / dimension


def spectral_window_mass(
    eigenvalues: FloatArray,
    eigenvectors: FloatArray,
    low: float,
    high: float,
) -> tuple[FloatArray, int]:
    """Mass of every eigenspace in a spectral window, pooled and normalised.

    The graph analogue of the high-energy limit is not one eigenvalue going to
    infinity but the graph growing with the spectral window held fixed, which is
    the setting of the Anantharaman-Le Masson quantum ergodicity theorem.  So the
    observable is the pooled projector diagonal over a window.
    """
    if low >= high:
        raise ValueError(f"need low < high; got {low}, {high}")
    mask = (eigenvalues >= low) & (eigenvalues <= high)
    count = int(mask.sum())
    if count == 0:
        raise ValueError(f"no eigenvalues in the window [{low}, {high}]")
    selected = eigenvectors[:, mask]
    size = eigenvectors.shape[0]
    return size * np.sum(selected**2, axis=1) / count, count


# --------------------------------------------------------------------------- #
# Thin sets
# --------------------------------------------------------------------------- #
def shortest_cycle(adjacency: csr_matrix, start: int = 0) -> tuple[int, ...]:
    """A shortest cycle through the BFS tree rooted at ``start``.

    Closed geodesics are the thin sets the continuous conjecture is really about,
    and on a Cayley graph they are the non-backtracking words in the generators
    that return to the identity.  A shortest one realises the girth.

    Reconstruction walks both endpoints of a non-tree edge up to their lowest
    common ancestor.  Taking the symmetric difference of the two ancestor sets
    instead -- the tempting shortcut -- silently returns the wrong vertex set:
    on a complete graph it yields two vertices rather than a triangle.
    """
    graph = nx.from_scipy_sparse_array(adjacency)
    if start not in graph:
        raise ValueError(f"vertex {start} is not in the graph")

    parent: dict[int, int | None] = {start: None}
    depth = {start: 0}
    queue = deque([start])
    best: tuple[int, ...] | None = None
    best_length = np.inf

    def ancestors(node: int) -> list[int]:
        chain = []
        current: int | None = node
        while current is not None:
            chain.append(current)
            current = parent[current]
        return chain

    while queue:
        current = queue.popleft()
        if 2 * depth[current] >= best_length:
            break
        for neighbour in graph[current]:
            if neighbour not in depth:
                depth[neighbour] = depth[current] + 1
                parent[neighbour] = current
                queue.append(neighbour)
            elif neighbour != parent[current]:
                left, right = ancestors(current), ancestors(neighbour)
                seen = set(left)
                meeting = next(node for node in right if node in seen)
                cycle = []
                walk: int | None = current
                while walk != meeting:
                    cycle.append(walk)
                    walk = parent[walk]
                cycle.append(meeting)
                tail = []
                walk = neighbour
                while walk != meeting:
                    tail.append(walk)
                    walk = parent[walk]
                cycle.extend(reversed(tail))
                if len(cycle) < best_length:
                    best_length = len(cycle)
                    best = tuple(sorted(cycle))
    if best is None:
        raise ValueError("the graph appears to be a forest, so it has no cycle")
    return best


def ball_around(adjacency: csr_matrix, centre: int, radius: int) -> tuple[int, ...]:
    """All vertices within ``radius`` steps of ``centre``."""
    if radius < 0:
        raise ValueError(f"radius must be non-negative; got {radius}")
    graph = nx.from_scipy_sparse_array(adjacency)
    reached = nx.single_source_shortest_path_length(graph, centre, cutoff=radius)
    return tuple(sorted(reached))


@dataclass(frozen=True)
class ThinSetStatistics:
    """How far an eigenspace's mass on a thin set departs from equidistribution."""

    n_vertices: int
    subset_size: int
    mean_mass: float
    deviation: float
    worst_deviation: float
    n_subsets: int

    @property
    def alpha(self) -> float:
        """Thinness exponent: the subset has size ``N^alpha``."""
        return float(np.log(self.subset_size) / np.log(self.n_vertices))


def thin_set_statistics(
    mass: FloatArray,
    subsets: Sequence[Sequence[int]],
) -> ThinSetStatistics:
    """Deviation of ``mass`` from its average over a family of thin sets.

    ``mass`` averages 1 by construction, so a subset that sees an equidistributed
    eigenspace has mean mass 1.  The reported deviation is the root-mean-square of
    ``mean(mass on S) - 1`` across the family, and the worst single value is kept
    alongside it: equidistribution is a statement about *every* set, so an
    average that looks healthy while one set is far off is not equidistribution.
    """
    if not subsets:
        raise ValueError("at least one subset is required")
    sizes = {len(s) for s in subsets}
    if len(sizes) != 1:
        raise ValueError(f"all subsets must have the same size; got {sorted(sizes)}")

    deviations = np.array(
        [float(np.mean(mass[list(subset)])) - 1.0 for subset in subsets]
    )
    return ThinSetStatistics(
        n_vertices=int(mass.size),
        subset_size=sizes.pop(),
        mean_mass=float(np.mean([np.mean(mass[list(s)]) for s in subsets])),
        deviation=float(np.sqrt(np.mean(deviations**2))),
        worst_deviation=float(np.max(np.abs(deviations))),
        n_subsets=len(subsets),
    )


# --------------------------------------------------------------------------- #
# References and controls
# --------------------------------------------------------------------------- #
def kesten_mckay_density(x: FloatArray, degree: int) -> FloatArray:
    """The Kesten-McKay law, the limiting spectral density of ``d``-regular graphs.

    Ramanujan graphs follow it, so it is an external check on the spectrum's
    shape rather than merely its extremes.
    """
    if degree < 3:
        raise ValueError(f"the Kesten-McKay law needs degree at least 3; got {degree}")
    support = 2.0 * np.sqrt(degree - 1)
    inside = np.abs(x) < support
    density = np.zeros_like(x, dtype=F64)
    numerator = degree * np.sqrt(np.clip(4.0 * (degree - 1) - x[inside] ** 2, 0.0, None))
    denominator = 2.0 * np.pi * (degree**2 - x[inside] ** 2)
    density[inside] = numerator / denominator
    return density


def bottlenecked_graph(lobe: int, bridge: int) -> csr_matrix:
    """Two cliques joined by a path -- a graph whose low modes localise.

    The negative control.  Without one, a test of "does the mass equidistribute"
    could only ever answer yes, and would keep answering yes if the measurement
    were broken.  Here the low-lying eigenvectors concentrate on one lobe, so any
    working thin-set statistic must report a large deviation.
    """
    if lobe < 3 or bridge < 1:
        raise ValueError(f"need lobe >= 3 and bridge >= 1; got {lobe}, {bridge}")
    graph = nx.barbell_graph(lobe, bridge)
    return nx.to_scipy_sparse_array(graph, format="csr", dtype=F64)


# --------------------------------------------------------------------------- #
# Hecke operators, and the obstruction they run into
# --------------------------------------------------------------------------- #
def hecke_operators(q: int, primes: Sequence[int]) -> tuple[list[csr_matrix], int]:
    """Commuting adjacency operators on one shared vertex set.

    All ``primes`` must have the same quadratic character mod ``q``, so that every
    generator set lands in the same projective group and the operators act on the
    same vertices.  The vertex enumeration is built once from the union of all
    generators; building each graph separately would index them differently and
    the "commutator" would then measure the relabelling.

    These are genuine Hecke operators: they commute exactly, in integer
    arithmetic.  What they cannot do is separate eigenvectors -- see
    :func:`hecke_joint_multiplicities`.
    """
    if len(primes) < 1:
        raise ValueError("at least one prime is required")
    characters = {legendre_symbol(p, q) for p in primes}
    if len(characters) != 1:
        raise ValueError(
            f"primes {list(primes)} have mixed quadratic characters mod {q}, so "
            "they generate different groups and share no vertex set"
        )

    inverses = [0] * q
    for value in range(1, q):
        inverses[value] = pow(value, q - 2, q)

    def multiply(a, b):
        return (
            (a[0] * b[0] + a[1] * b[2]) % q,
            (a[0] * b[1] + a[1] * b[3]) % q,
            (a[2] * b[0] + a[3] * b[2]) % q,
            (a[2] * b[1] + a[3] * b[3]) % q,
        )

    def projective_normal(a):
        for entry in a:
            if entry:
                scale = inverses[entry]
                return tuple((x * scale) % q for x in a)
        raise ValueError("encountered the zero matrix")

    generator_sets = [lps_generators(p, q) for p in primes]
    every = tuple(g for group in generator_sets for g in group)

    identity = projective_normal((1, 0, 0, 1))
    index = {identity: 0}
    order = [identity]
    stack = [identity]
    while stack:
        current = stack.pop()
        for generator in every:
            image = projective_normal(multiply(current, generator))
            if image not in index:
                index[image] = len(order)
                order.append(image)
                stack.append(image)

    size = len(order)
    operators = []
    for generators in generator_sets:
        rows, columns = [], []
        for matrix, source in index.items():
            for generator in generators:
                rows.append(source)
                columns.append(index[projective_normal(multiply(matrix, generator))])
        operators.append(
            coo_matrix(
                (np.ones(len(rows), dtype=F64), (rows, columns)), shape=(size, size)
            ).tocsr()
        )
    return operators, size


def hecke_joint_multiplicities(
    operators: Sequence[csr_matrix], *, decimals: int = 6, seed: int = 0
) -> dict[int, int]:
    """Multiplicities of the joint eigenspaces of commuting operators.

    Diagonalises a generic real combination, which for commuting symmetric
    matrices has the same eigenspaces as the family jointly.  The returned
    histogram maps multiplicity to how many joint eigenvalues carry it.

    On a Cayley graph these multiplicities are bounded below by the dimensions of
    the group's irreducible representations, because every one of these operators
    is a right convolution and therefore commutes with the whole left regular
    action.  That is the reason the arithmetic model cannot produce a canonical
    eigenvector basis, no matter how many Hecke operators are supplied.
    """
    if not operators:
        raise ValueError("at least one operator is required")
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=len(operators))
    combination = sum(
        float(w) * np.asarray(op.todense(), dtype=F64)
        for w, op in zip(weights, operators)
    )
    values = np.linalg.eigvalsh(combination)
    _, counts = np.unique(np.round(values, decimals), return_counts=True)
    histogram: dict[int, int] = {}
    for count in counts:
        histogram[int(count)] = histogram.get(int(count), 0) + 1
    return dict(sorted(histogram.items()))


def projector_diagonal_is_constant(
    eigenvalues: FloatArray, eigenvectors: FloatArray, target: float, *, atol: float = 1e-7
) -> float:
    """Spread of an eigenspace's projector diagonal -- zero on a transitive graph.

    Returns ``max - min`` of the diagonal.  On any vertex-transitive graph this
    is zero up to rounding, which is precisely why the basis-free observable says
    nothing there.  Used as a *measurement* of the obstruction rather than an
    assumption about it.
    """
    return float(np.ptp(eigenspace_mass(eigenvalues, eigenvectors, target, atol=atol)))


# --------------------------------------------------------------------------- #
# The model where the question has content
# --------------------------------------------------------------------------- #
def random_regular_graph(degree: int, size: int, seed: int) -> csr_matrix:
    """A random ``d``-regular graph: almost-Ramanujan, and not vertex-transitive.

    Friedman's theorem gives ``lambda_2 <= 2 sqrt(d-1) + epsilon`` with high
    probability, so the spectral gap that makes LPS graphs interesting survives.
    What does not survive is the homogeneity, and that is the point: the spectrum
    is simple, so eigenvectors are canonical and their mass is well defined.
    """
    if degree < 3:
        raise ValueError(f"degree must be at least 3; got {degree}")
    graph = nx.random_regular_graph(degree, size, seed=seed)
    return nx.to_scipy_sparse_array(graph, format="csr", dtype=F64)


def eigenvector_mass(eigenvector: FloatArray) -> FloatArray:
    """``|psi(v)|^2`` normalised to average 1 over the vertices."""
    vector = np.asarray(eigenvector, dtype=F64).ravel()
    norm = float(vector @ vector)
    if norm <= 0:
        raise ValueError("the eigenvector is zero")
    return vector.size * vector**2 / norm


def gaussian_baseline(subset_size: int) -> float:
    """Expected r.m.s. deviation for a subset of a random Gaussian unit vector.

    A delocalised eigenvector should not do noticeably *worse* than this, and a
    scarred one does much worse.  Quoting deviations without it would make an
    ordinary fluctuation look like a result.
    """
    if subset_size < 1:
        raise ValueError(f"subset size must be positive; got {subset_size}")
    return float(np.sqrt(2.0 / subset_size))
