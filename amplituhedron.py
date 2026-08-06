"""Exact tilings of the ``k = 1`` amplituhedron, and the spectrum of their flip graph.

Why this is the honest target
-----------------------------
The amplituhedron ``A(n, k, m)`` is the image of the nonnegative Grassmannian
``Gr_{>=0}(k, n)`` under ``C |-> C Z``, for a matrix ``Z`` whose maximal minors
are all positive.  For general ``k`` its tiles are curved semialgebraic sets --
images of positroid cells -- so there is no separating hyperplane between two of
them and no polytope machinery to bring to bear.

At ``k = 1`` the object is different in kind.  A point of ``Gr_{>=0}(1, n)`` is a
positive row vector ``c``, so ``Y = c Z`` is a positive combination of the rows
of ``Z``; positivity of the minors of ``Z`` says exactly that those rows sit in
convex position like points on the moment curve.  Hence

    A(n, 1, m)  =  the cyclic polytope  C(n, m)  in  P^m,

a genuine polytope with integer vertices, and

    tilings of A(n, 1, m)  =  triangulations of C(n, m).

Everything below is therefore exact integer arithmetic with no floating point
anywhere: vertices on the moment curve at ``t = 1 .. n`` are integers, the
Separating Axis Theorem generalises to ``R^m`` over the integers, and volumes are
integer determinants.  Where a classical theorem gives an independent answer
(Gale's evenness condition for the facets, Catalan numbers for ``m = 2``,
Eulerian numbers for the hypersimplex) it is used as a check on the machinery
rather than as an ingredient of it.

The spectral question
---------------------
Triangulations of ``C(n, m)`` are connected by bistellar flips, and the flip
graph is the natural finite object attached to the tiling problem.  Its Laplacian
spectrum is well posed, unlike a Yangian action on a finite complex, and its
degeneracies are forced by the symmetry group of the point configuration.  For
``m = 2`` the flip graph is the 1-skeleton of the associahedron, which pins the
whole pipeline against known combinatorics.
"""

from __future__ import annotations

import itertools
import logging
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import comb, factorial
from typing import Iterable, Sequence

import networkx as nx

__all__ = [
    "IntPoint",
    "IntMatrix",
    "Simplex",
    "Tiling",
    "TilingReport",
    "integer_determinant",
    "generalised_cross",
    "rationals_to_integers",
    "moment_curve_point",
    "cyclic_polytope",
    "hypersimplex",
    "gale_facets",
    "normalised_volume",
    "polytope_normalised_volume",
    "configuration_volume",
    "catalan",
    "tiling_summary",
    "eulerian_number",
    "separating_axis",
    "interiors_disjoint",
    "full_dimensional_simplices",
    "enumerate_tilings",
    "is_tiling",
    "radon_partition",
    "bistellar_neighbours",
    "flip_graph",
    "configuration_automorphisms",
]

LOGGER = logging.getLogger(__name__)

IntPoint = tuple[int, ...]
IntMatrix = tuple[IntPoint, ...]
Simplex = tuple[int, ...]
Tiling = frozenset[Simplex]


# --------------------------------------------------------------------------- #
# Exact integer linear algebra
# --------------------------------------------------------------------------- #
def integer_determinant(matrix: Sequence[Sequence[int]]) -> int:
    """Determinant of a square integer matrix, exactly.

    Bareiss fraction-free elimination: every intermediate entry is itself a
    determinant of a minor, so the whole computation stays in ``Z`` and the
    result is exact.  A floating-point determinant would be worse than merely
    imprecise here -- it can round a singular matrix into an invertible one, and
    the entire tiling test rests on telling those two cases apart.
    """
    rows = [list(row) for row in matrix]
    size = len(rows)
    if size == 0:
        return 1
    if any(len(row) != size for row in rows):
        raise ValueError("integer_determinant requires a square matrix")

    sign = 1
    previous = 1
    for k in range(size - 1):
        if rows[k][k] == 0:
            for swap in range(k + 1, size):
                if rows[swap][k] != 0:
                    rows[k], rows[swap] = rows[swap], rows[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, size):
            for j in range(k + 1, size):
                numerator = rows[i][j] * rows[k][k] - rows[i][k] * rows[k][j]
                rows[i][j] = numerator // previous
        previous = rows[k][k]
    return sign * rows[size - 1][size - 1]


def generalised_cross(vectors: Sequence[Sequence[int]]) -> IntPoint:
    """A vector in ``Z^d`` orthogonal to ``d - 1`` given vectors.

    Signed maximal minors, i.e. the Laplace expansion of ``det([v; vectors])``
    along its first row.  That identity is what makes the result orthogonal to
    every input: substituting one of the inputs for ``v`` repeats a row.  The
    result is the zero vector exactly when the inputs are dependent, which is
    the case a caller must treat as "this axis carries no information".
    """
    rows = [list(v) for v in vectors]
    if not rows:
        raise ValueError("generalised_cross needs at least one vector")
    dimension = len(rows[0])
    if any(len(row) != dimension for row in rows):
        raise ValueError("generalised_cross requires vectors of equal length")
    if len(rows) != dimension - 1:
        raise ValueError(
            f"generalised_cross in dimension {dimension} needs {dimension - 1} "
            f"vectors; got {len(rows)}"
        )

    result = []
    for j in range(dimension):
        minor = [row[:j] + row[j + 1 :] for row in rows]
        cofactor = integer_determinant(minor)
        result.append(cofactor if j % 2 == 0 else -cofactor)
    return tuple(result)


def rationals_to_integers(points: Iterable[Sequence[Fraction | int]]) -> IntMatrix:
    """Clear denominators from rational coordinates by a single common scaling.

    Scaling every point by one positive integer is a homothety: it preserves
    incidence, convexity, and all volume *ratios*, so every tiling question is
    unchanged while the arithmetic becomes integer.
    """
    rows = [[Fraction(c) for c in point] for point in points]
    if not rows:
        raise ValueError("rationals_to_integers needs at least one point")
    multiplier = 1
    for row in rows:
        for value in row:
            multiplier = multiplier * value.denominator // _gcd(multiplier, value.denominator)
    return tuple(tuple(int(value * multiplier) for value in row) for row in rows)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


# --------------------------------------------------------------------------- #
# The polytopes
# --------------------------------------------------------------------------- #
def moment_curve_point(t: int, m: int) -> IntPoint:
    """``(t, t^2, ..., t^m)`` -- a point on the moment curve, in integers."""
    if m < 1:
        raise ValueError(f"dimension m must be at least 1; got {m}")
    return tuple(t**power for power in range(1, m + 1))


def cyclic_polytope(n: int, m: int, parameters: Sequence[int] | None = None) -> IntMatrix:
    """The cyclic polytope ``C(n, m)``, which *is* the amplituhedron ``A(n, 1, m)``.

    Parameters
    ----------
    n:
        Number of vertices.  Must exceed ``m``, or the hull is not full
        dimensional.
    m:
        Ambient dimension.  ``m = 4`` is the physical case; ``m = 2`` is the
        standard toy model and the one with known combinatorics.
    parameters:
        Strictly increasing integer moment-curve parameters.  Defaults to
        ``1 .. n``.  The combinatorial type does not depend on the choice, but
        the coordinates -- and hence the exact volumes -- do.
    """
    if n <= m:
        raise ValueError(f"cyclic_polytope needs n > m; got n={n}, m={m}")
    values = list(range(1, n + 1)) if parameters is None else [int(t) for t in parameters]
    if len(values) != n:
        raise ValueError(f"expected {n} parameters; got {len(values)}")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("moment-curve parameters must be strictly increasing")
    return tuple(moment_curve_point(t, m) for t in values)


def hypersimplex(k: int, n: int) -> IntMatrix:
    """The hypersimplex ``Delta(k, n)``, projected to full dimension in ``Z^(n-1)``.

    ``Delta(k, n)`` is the convex hull of the 0/1 vectors in ``R^n`` with
    coordinate sum ``k``.  It lives in the hyperplane ``sum x_i = k``, so it is
    ``(n-1)``-dimensional; dropping the last coordinate is an affine bijection of
    that hyperplane onto ``R^(n-1)`` with unimodular linear part, which leaves
    every normalised volume unchanged.

    It appears here because of T-duality (Lukowski-Parisi-Williams): positroidal
    subdivisions of ``Delta(k+1, n)`` correspond to tilings of the ``m = 2``
    amplituhedron ``A(n, k, 2)``.  That is the one route by which a genuinely
    ``k > 1`` amplituhedron question becomes a polytope question.
    """
    if not 0 < k < n:
        raise ValueError(f"hypersimplex needs 0 < k < n; got k={k}, n={n}")
    vertices = []
    for support in itertools.combinations(range(n), k):
        point = [0] * n
        for index in support:
            point[index] = 1
        vertices.append(tuple(point[:-1]))
    return tuple(vertices)


def gale_facets(n: int, m: int) -> frozenset[tuple[int, ...]]:
    """Facets of ``C(n, m)`` by Gale's evenness condition.

    An ``m``-subset ``S`` of the vertices spans a facet exactly when every two
    vertices outside ``S`` are separated, along the moment curve, by an even
    number of elements of ``S``.  This is classical and purely combinatorial, so
    it gives the facets without any hull computation -- and therefore an
    independent check on anything that computes them geometrically.
    """
    if n <= m:
        raise ValueError(f"gale_facets needs n > m; got n={n}, m={m}")
    facets = []
    for subset in itertools.combinations(range(n), m):
        chosen = set(subset)
        outside = [v for v in range(n) if v not in chosen]
        if all(
            sum(1 for s in subset if a < s < b) % 2 == 0
            for a, b in itertools.combinations(outside, 2)
        ):
            facets.append(subset)
    return frozenset(facets)


# --------------------------------------------------------------------------- #
# Exact volumes
# --------------------------------------------------------------------------- #
def normalised_volume(vertices: IntMatrix, simplex: Simplex) -> int:
    """``m!`` times the Euclidean volume of a simplex -- always an integer.

    Working in normalised units keeps every volume an exact integer, so "these
    pieces exactly fill that polytope" becomes an integer identity with no
    tolerance anywhere in it.
    """
    dimension = len(vertices[0])
    if len(simplex) != dimension + 1:
        raise ValueError(
            f"a simplex in dimension {dimension} has {dimension + 1} vertices; "
            f"got {len(simplex)}"
        )
    base = vertices[simplex[0]]
    edges = [
        [vertices[index][axis] - base[axis] for axis in range(dimension)]
        for index in simplex[1:]
    ]
    return abs(integer_determinant(edges))


def polytope_normalised_volume(vertices: IntMatrix, facets: Iterable[Simplex]) -> int:
    """Normalised volume of a *simplicial* polytope, by pulling at vertex 0.

    Coning every facet that misses vertex 0 back to vertex 0 triangulates the
    polytope; facets containing vertex 0 lie in the cone's boundary and
    contribute nothing.  Cyclic polytopes are simplicial, so their facets are
    already simplices and no further subdivision is needed.
    """
    total = 0
    for facet in facets:
        if 0 in facet:
            continue
        total += normalised_volume(vertices, (0,) + tuple(facet))
    return total


def eulerian_number(n: int, k: int) -> int:
    """``A(n, k)``: permutations of ``n`` letters with ``k`` descents.

    The normalised volume of the hypersimplex ``Delta(k+1, n+1)`` equals
    ``A(n, k)`` (Laplace; see Stanley).  Used purely as an external check.
    """
    if n < 0 or k < 0:
        raise ValueError(f"eulerian_number needs non-negative arguments; got {n}, {k}")
    return sum(
        (-1) ** j * comb(n + 1, j) * (k + 1 - j) ** n for j in range(k + 1)
    )


# --------------------------------------------------------------------------- #
# The Separating Axis Theorem in R^m, over the integers
# --------------------------------------------------------------------------- #
def _projection_range(axis: IntPoint, vertices: IntMatrix, indices: Simplex) -> tuple[int, int]:
    values = [sum(a * vertices[i][d] for d, a in enumerate(axis)) for i in indices]
    return min(values), max(values)


@lru_cache(maxsize=None)
def _face_spans(vertices: IntMatrix, simplex: Simplex) -> tuple[tuple[tuple[IntPoint, ...], ...], ...]:
    """Direction vectors spanning each face of a simplex, indexed by dimension.

    Entry ``d`` lists, for every ``d``-dimensional face, the ``d`` vectors that
    span its affine hull.  Cached per simplex, since each simplex is paired with
    very many others.
    """
    dimension = len(vertices[0])
    by_dimension = []
    for face_dimension in range(dimension):
        spans = []
        for subset in itertools.combinations(simplex, face_dimension + 1):
            base = vertices[subset[0]]
            spans.append(
                tuple(
                    tuple(vertices[v][d] - base[d] for d in range(dimension))
                    for v in subset[1:]
                )
            )
        by_dimension.append(tuple(spans))
    return tuple(by_dimension)


def _candidate_axes(
    vertices: IntMatrix, left: Simplex, right: Simplex
) -> Iterable[IntPoint]:
    """Every axis that could possibly separate two simplices, cheapest first.

    Completeness argument.  Two convex polytopes have disjoint interiors exactly
    when the origin is not interior to the Minkowski difference ``P + (-Q)``, and
    every facet of that difference is a sum ``F + (-G)`` of a face ``F`` of ``P``
    and a face ``G`` of ``Q`` with ``dim F + dim G = m - 1``.  Its normal is
    orthogonal to the affine hulls of both.  So enumerating *face pairs* of
    complementary dimension is exactly exhaustive -- and it is much tighter than
    enumerating ``(m-1)``-subsets of the combined edge directions, which is a
    superset containing many subsets that span no face pair at all and many
    duplicates of ones that do.  A face of dimension 0 contributes no direction,
    so those pairs collapse onto the other simplex's facet normals; in ``R^4``
    that leaves 210 axes against 910 for edge subsets.

    In ``R^3`` this reproduces the familiar tetrahedron axes exactly: the
    ``(2,0)`` and ``(0,2)`` pairs are the 4 + 4 face normals, and the ``(1,1)``
    pairs are the 36 edge-pair crosses.  In ``R^2`` only ``(1,0)`` and ``(0,1)``
    occur, so edge normals alone are complete, as they should be.

    The pure facet normals -- face pairs against a single vertex -- resolve the
    great majority of pairs, so they are yielded first.
    """
    dimension = len(vertices[0])
    left_faces = _face_spans(vertices, left)
    right_faces = _face_spans(vertices, right)

    # (m-1, 0) and (0, m-1): the two simplices' own facet normals.
    for span in left_faces[dimension - 1]:
        yield generalised_cross(span)
    for span in right_faces[dimension - 1]:
        yield generalised_cross(span)

    # Mixed pairs. Dimension 0 on either side contributes no direction, so those
    # cases are exactly the facet normals already yielded.
    for left_dimension in range(1, dimension - 1):
        right_dimension = dimension - 1 - left_dimension
        for left_span in left_faces[left_dimension]:
            for right_span in right_faces[right_dimension]:
                yield generalised_cross(list(left_span) + list(right_span))


def separating_axis(
    vertices: IntMatrix, left: Simplex, right: Simplex
) -> IntPoint | None:
    """An exact integer axis separating two simplices' interiors, or ``None``.

    Separation is weak: touching along a shared face counts as separated, which
    is the condition a tiling actually requires.
    """
    for axis in _candidate_axes(vertices, left, right):
        if not any(axis):
            continue  # dependent directions carry no information
        low_left, high_left = _projection_range(axis, vertices, left)
        low_right, high_right = _projection_range(axis, vertices, right)
        if high_left <= low_right or high_right <= low_left:
            return axis
    return None


def interiors_disjoint(vertices: IntMatrix, left: Simplex, right: Simplex) -> bool:
    """Whether two simplices meet in at most a shared face."""
    return separating_axis(vertices, left, right) is not None


# --------------------------------------------------------------------------- #
# Enumerating every tiling
# --------------------------------------------------------------------------- #
def full_dimensional_simplices(vertices: IntMatrix) -> tuple[Simplex, ...]:
    """Every vertex subset spanning a full-dimensional simplex, in fixed order."""
    dimension = len(vertices[0])
    return tuple(
        subset
        for subset in itertools.combinations(range(len(vertices)), dimension + 1)
        if normalised_volume(vertices, subset) > 0
    )


def configuration_volume(vertices: IntMatrix) -> int:
    """Normalised volume of the convex hull of a point configuration, exactly.

    No facet enumeration and no assumption that the polytope is simplicial, so
    this works where :func:`polytope_normalised_volume` does not -- the
    hypersimplex in particular.

    The identity it exploits: the hull's volume is the *maximum* total volume of
    any family of full-dimensional simplices with pairwise disjoint interiors.
    No such family can exceed the hull, since its pieces are disjoint and
    contained in it; and any triangulation attains the hull exactly.  So the
    maximum is the volume, and a branch-and-bound over the same compatibility
    structure the tiling search already needs computes it.

    The bound is seeded greedily, which usually finds a genuine triangulation on
    the first descent and makes the remaining search almost pure pruning.
    """
    raw = full_dimensional_simplices(vertices)
    if not raw:
        raise ValueError("configuration_volume needs a full-dimensional configuration")

    # Visit the largest pieces first.  The branch-and-bound cutoff compares the
    # volume still reachable against the best found so far, so a descending order
    # both raises the incumbent immediately and shrinks the tail sum fastest --
    # in index order the bound barely bites until the search is nearly done.
    order = sorted(range(len(raw)), key=lambda i: -normalised_volume(vertices, raw[i]))
    candidates = [raw[i] for i in order]
    count = len(candidates)
    volumes = [normalised_volume(vertices, s) for s in candidates]

    compatible = [0] * count
    for i in range(count):
        for j in range(i + 1, count):
            if interiors_disjoint(vertices, candidates[i], candidates[j]):
                compatible[i] |= 1 << j
                compatible[j] |= 1 << i

    best = 0
    allowed = (1 << count) - 1
    for i in range(count):  # greedy seed, already largest-first
        if allowed >> i & 1:
            best += volumes[i]
            allowed &= compatible[i]

    def search(position: int, volume: int, allowed: int) -> None:
        nonlocal best
        if volume > best:
            best = volume
        remaining = allowed >> position << position
        reachable = volume
        cursor = remaining
        while cursor:
            low = cursor & -cursor
            reachable += volumes[low.bit_length() - 1]
            cursor ^= low
        if reachable <= best:
            return
        for i in range(position, count):
            if allowed >> i & 1:
                search(i + 1, volume + volumes[i], allowed & compatible[i])

    search(0, 0, (1 << count) - 1)
    return best


@dataclass(frozen=True)
class TilingReport:
    """Everything the enumeration establishes about one point configuration."""

    vertices: IntMatrix
    dimension: int
    total_volume: int
    candidates: tuple[Simplex, ...]
    tilings: tuple[Tiling, ...]

    @property
    def n_tilings(self) -> int:
        return len(self.tilings)

    @property
    def sizes(self) -> dict[int, int]:
        """How many tilings use each number of simplices."""
        histogram: dict[int, int] = {}
        for tiling in self.tilings:
            histogram[len(tiling)] = histogram.get(len(tiling), 0) + 1
        return dict(sorted(histogram.items()))


def is_tiling(vertices: IntMatrix, simplices: Iterable[Simplex], total_volume: int) -> bool:
    """Exact test that a family of simplices tiles a polytope.

    Two conditions, both exact: the interiors are pairwise disjoint (integer
    SAT), and the normalised volumes sum to the polytope's.  Neither alone
    suffices -- disjointness permits gaps, and the volume identity permits
    overlaps that cancel against gaps -- but together they are decisive, since
    disjoint pieces inside the polytope can only reach its total volume by
    leaving nothing uncovered.
    """
    family = list(simplices)
    if sum(normalised_volume(vertices, s) for s in family) != total_volume:
        return False
    return all(
        interiors_disjoint(vertices, a, b)
        for a, b in itertools.combinations(family, 2)
    )


def enumerate_tilings(
    vertices: IntMatrix,
    total_volume: int | None = None,
    *,
    limit: int | None = None,
) -> TilingReport:
    """Every tiling of the polytope, by exhaustive search over exact geometry.

    The search decides candidate simplices one at a time in a fixed order,
    branching on include/exclude.  That is complete and duplicate-free without
    assuming anything about the configuration -- in particular without assuming
    the flip graph is connected, which is what the flip graph is then used to
    *test*.

    Pruning is exact: at every node the volume still reachable is the sum over
    undecided candidates that remain interior-disjoint from everything chosen,
    and if the chosen volume plus that bound falls short of the target the
    branch cannot possibly complete.

    Parameters
    ----------
    vertices:
        Integer point configuration in convex position.
    total_volume:
        Normalised volume of the hull.  Computed by :func:`configuration_volume`
        when omitted; pass it explicitly when an independent route is available
        (Gale's condition for cyclic polytopes) so the two can be cross-checked.
    limit:
        Stop after this many tilings.  ``None`` enumerates all of them.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be positive when given; got {limit}")
    if total_volume is None:
        total_volume = configuration_volume(vertices)

    candidates = full_dimensional_simplices(vertices)
    count = len(candidates)
    volumes = [normalised_volume(vertices, s) for s in candidates]

    compatible = [0] * count
    for i in range(count):
        for j in range(i + 1, count):
            if interiors_disjoint(vertices, candidates[i], candidates[j]):
                compatible[i] |= 1 << j
                compatible[j] |= 1 << i
    LOGGER.debug("enumerate_tilings: %d candidate simplices", count)

    all_bits = (1 << count) - 1
    found: list[Tiling] = []

    def search(position: int, chosen: list[int], volume: int, allowed: int) -> None:
        if volume == total_volume:
            found.append(frozenset(candidates[i] for i in chosen))
            return
        if volume > total_volume or position >= count:
            return
        if limit is not None and len(found) >= limit:
            return

        remaining = allowed >> position << position
        reachable = volume
        cursor = remaining
        while cursor:
            low = cursor & -cursor
            reachable += volumes[low.bit_length() - 1]
            cursor ^= low
        if reachable < total_volume:
            return

        if allowed >> position & 1:
            chosen.append(position)
            search(position + 1, chosen, volume + volumes[position], allowed & compatible[position])
            chosen.pop()
        search(position + 1, chosen, volume, allowed)

    search(0, [], 0, all_bits)
    return TilingReport(
        vertices=vertices,
        dimension=len(vertices[0]),
        total_volume=total_volume,
        candidates=candidates,
        tilings=tuple(found),
    )


# --------------------------------------------------------------------------- #
# Bistellar flips and the flip graph
# --------------------------------------------------------------------------- #
def radon_partition(
    vertices: IntMatrix, support: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The unique Radon circuit of ``m + 2`` points in ``R^m``.

    Any ``m + 2`` points in ``R^m`` carry an affine dependence ``sum lambda_i p_i
    = 0`` with ``sum lambda_i = 0``, unique up to scale when the points are
    otherwise generic.  Its sign pattern splits them into the two parts whose
    convex hulls intersect -- and those two parts index precisely the two ways to
    triangulate the circuit, which is what a bistellar flip exchanges.
    """
    dimension = len(vertices[0])
    if len(support) != dimension + 2:
        raise ValueError(
            f"a circuit in dimension {dimension} has {dimension + 2} points; "
            f"got {len(support)}"
        )
    rows = [[vertices[v][axis] for v in support] for axis in range(dimension)]
    rows.append([1] * len(support))
    coefficients = generalised_cross(rows)
    positive = tuple(support[i] for i, c in enumerate(coefficients) if c > 0)
    negative = tuple(support[i] for i, c in enumerate(coefficients) if c < 0)
    return positive, negative


def bistellar_neighbours(vertices: IntMatrix, left: Tiling, right: Tiling) -> bool:
    """Whether two tilings differ by exactly one bistellar flip.

    The test is structural rather than a size heuristic: the simplices that
    differ must be supported on exactly ``m + 2`` vertices, and the two differing
    families must be precisely the two triangulations of that circuit, namely
    ``support`` minus each element of the positive part and minus each element of
    the negative part.
    """
    dimension = len(vertices[0])
    only_left = left - right
    only_right = right - left
    if not only_left or not only_right:
        return False

    support = sorted({v for simplex in only_left | only_right for v in simplex})
    if len(support) != dimension + 2:
        return False
    if len(only_left) + len(only_right) != dimension + 2:
        return False

    positive, negative = radon_partition(vertices, support)
    if len(positive) + len(negative) != dimension + 2:
        return False  # a degenerate circuit is not a flip

    whole = set(support)
    side_a = frozenset(tuple(sorted(whole - {v})) for v in positive)
    side_b = frozenset(tuple(sorted(whole - {v})) for v in negative)
    return {only_left, only_right} == {side_a, side_b}


def flip_graph(vertices: IntMatrix, tilings: Sequence[Tiling]) -> nx.Graph:
    """The graph on tilings whose edges are single bistellar flips.

    Nodes are integer indices into ``tilings``, carrying the tiling itself as a
    node attribute, so the result drops straight into the Laplacian pipeline.
    """
    graph = nx.Graph()
    for index, tiling in enumerate(tilings):
        graph.add_node(index, tiling=tiling, size=len(tiling))
    for i, j in itertools.combinations(range(len(tilings)), 2):
        if bistellar_neighbours(vertices, tilings[i], tilings[j]):
            graph.add_edge(i, j)
    return graph


def configuration_automorphisms(vertices: IntMatrix) -> tuple[tuple[int, ...], ...]:
    """Vertex permutations preserving the oriented matroid up to global sign.

    These are the symmetries that must act on the set of tilings, and therefore
    the only ones that can force degeneracies in the flip graph's spectrum.  A
    permutation qualifies when it either preserves the sign of every full
    simplex's determinant or reverses all of them; both cases carry
    triangulations to triangulations.

    The search is restricted to the dihedral group generated by the cyclic shift
    and the order reversal.  For a configuration on the moment curve those are
    the natural candidates, and checking them exactly is cheap; a full symmetric
    group sweep would cost ``n!`` determinant passes for symmetries that a
    strictly convex curve cannot have.
    """
    n = len(vertices)
    dimension = len(vertices[0])
    simplices = list(itertools.combinations(range(n), dimension + 1))

    def orientation(permutation: Sequence[int], simplex: Simplex) -> int:
        image = tuple(sorted(permutation[v] for v in simplex))
        base = vertices[image[0]]
        edges = [
            [vertices[v][d] - base[d] for d in range(dimension)] for v in image[1:]
        ]
        value = integer_determinant(edges)
        # Sorting the image may permute rows relative to the source ordering;
        # compare against the source's own sorted determinant, which is what the
        # simplex list uses throughout.
        return (value > 0) - (value < 0)

    reference = {s: orientation(tuple(range(n)), s) for s in simplices}
    found = []
    for shift in range(n):
        for reverse in (False, True):
            permutation = tuple(
                (shift + (n - 1 - v if reverse else v)) % n for v in range(n)
            )
            signs = {orientation(permutation, s) for s in simplices if reference[s] != 0}
            mismatch = {
                orientation(permutation, s) * reference[s]
                for s in simplices
                if reference[s] != 0
            }
            if signs and mismatch <= {1}:
                found.append(permutation)
            elif signs and mismatch <= {-1}:
                found.append(permutation)
    return tuple(sorted(set(found)))


def catalan(n: int) -> int:
    """``C_n`` -- the number of triangulations of an ``(n + 2)``-gon."""
    if n < 0:
        raise ValueError(f"catalan needs a non-negative index; got {n}")
    return comb(2 * n, n) // (n + 1)


def tiling_summary(vertices: IntMatrix, report: TilingReport) -> dict[str, object]:
    """Structured facts about one configuration, for logs and dataframes."""
    graph = flip_graph(vertices, report.tilings)
    degrees = [d for _, d in graph.degree()]
    return {
        "n_vertices": len(vertices),
        "dimension": report.dimension,
        "total_normalised_volume": report.total_volume,
        "n_candidate_simplices": len(report.candidates),
        "n_tilings": report.n_tilings,
        "tiling_sizes": report.sizes,
        "flip_graph_edges": graph.number_of_edges(),
        "flip_graph_connected": nx.is_connected(graph) if graph else False,
        "flip_degree_min": min(degrees) if degrees else 0,
        "flip_degree_max": max(degrees) if degrees else 0,
    }
