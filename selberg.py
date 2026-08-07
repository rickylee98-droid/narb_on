"""The Selberg trace formula on finite graphs, in exact integer arithmetic.

What the object is
------------------
For a hyperbolic surface the Selberg trace formula couples the spectrum of the
Laplacian to the lengths of closed geodesics.  A finite graph has an exact
analogue, and unlike the surface case every term in it is a finite integer that
can be computed two independent ways -- which is the reason to do this
numerically at all.

The geodesics are *non-backtracking tailless closed walks*.  Backtracking is the
discrete analogue of a path that doubles back and can be contracted, so it must
be excluded before "geodesic" means anything.  They are counted by the Hashimoto
edge operator ``B`` on the ``2|E|`` directed edges,

    B[(u,v), (v,w)] = 1  iff  w != u ,

whose powers count exactly those walks: ``N_m = tr(B^m)``.  The Ihara zeta
function is

    zeta(u) = 1 / det(I - u B) ,

and Bass's theorem re-expresses that ``2|E| x 2|E|`` determinant through the
``|V| x |V|`` adjacency matrix.  For a ``(q+1)``-regular graph with
``r = |E| - |V| + 1``,

    det(I - u B) = (1 - u^2)^{r-1} det(I - A u + q u^2 I) .

Everything here is refereed rather than asserted
------------------------------------------------
The formulas above are standard, but this module treats none of them as known.
Each is computed along two independent routes and the routes are compared as
exact integers or exact polynomials:

* ``N_m`` from ``tr(B^m)`` against ``N_m`` from brute-force enumeration of
  closed non-backtracking tailless walks.  No floating point on either side.
* ``det(I - u B)`` expanded as an integer polynomial against Bass's right-hand
  side, coefficient by coefficient.
* The Euler product over prime geodesics against the zeta power series.
* The spectral side of the trace formula against the geodesic side.

A formula this module cannot verify is one this module does not use.  The
literature statement is the hypothesis; the integer identity is the evidence.

The Riemann hypothesis for graphs
---------------------------------
Substituting ``u = q^{-s}`` turns the pole structure of ``zeta`` into a statement
about a critical line.  A ``(q+1)``-regular graph satisfies the RH analogue
exactly when it is *Ramanujan*: every adjacency eigenvalue other than the trivial
``+-(q+1)`` obeys ``|lambda| <= 2 sqrt(q)``.  That is the same condition that
governs the spectral gap, and it controls the error term in the graph prime
number theorem

    pi(m) ~ q^m / m ,

for the number of primitive closed geodesics of length ``m``.  Measuring that
error exponent, on graphs that are and are not Ramanujan, is the quantitative
question this module is built for.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Iterator, Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DirectedEdge",
    "directed_edges",
    "hashimoto_operator",
    "integer_matrix_power_trace",
    "closed_geodesic_count",
    "enumerate_closed_geodesics",
    "IntPolynomial",
    "poly_multiply",
    "poly_power",
    "characteristic_polynomial",
    "ihara_zeta_reciprocal",
    "bass_reciprocal",
    "is_regular",
    "graph_degree",
    "prime_geodesic_count",
    "prime_geodesic_counts",
    "geodesic_counts_exact",
    "geodesic_counts_from_spectrum",
    "moebius",
    "zeta_series",
    "euler_product_series",
    "adjacency_spectrum",
    "RamanujanReport",
    "ramanujan_report",
    "TraceFormulaCheck",
    "trace_formula_check",
    "RiemannHypothesisTest",
    "riemann_hypothesis_test",
    "PrimeGeodesicFit",
    "prime_geodesic_fit",
    "spectral_radii",
    "dominant_nontrivial_radius",
    "hurwitz_units",
    "binary_tetrahedral_cayley",
    "quaternion_multiply",
]

LOGGER = logging.getLogger(__name__)

DirectedEdge = tuple[int, int]
IntPolynomial = list[int]


# --------------------------------------------------------------------------- #
# The non-backtracking (Hashimoto) operator
# --------------------------------------------------------------------------- #
def directed_edges(graph: nx.Graph) -> tuple[DirectedEdge, ...]:
    """Both orientations of every edge, in a fixed reproducible order."""
    edges: list[DirectedEdge] = []
    for u, v in sorted(tuple(sorted(e)) for e in graph.edges()):
        if u == v:
            raise ValueError(
                f"self-loop at {u}: the non-backtracking operator is not defined "
                "for graphs with loops"
            )
        edges.append((u, v))
        edges.append((v, u))
    return tuple(edges)


def hashimoto_operator(graph: nx.Graph) -> tuple[NDArray[np.object_], tuple[DirectedEdge, ...]]:
    """The ``2|E| x 2|E|`` operator whose powers count non-backtracking walks.

    ``B[(u,v), (v,w)] = 1`` unless ``w == u``.  Stored with ``dtype=object`` so
    that ``B**m`` stays in exact Python integers; the counts grow like ``q^m``
    and overflow 64-bit arithmetic well before the interesting range.
    """
    edges = directed_edges(graph)
    index = {edge: i for i, edge in enumerate(edges)}
    size = len(edges)
    operator = np.zeros((size, size), dtype=object)
    for i, (u, v) in enumerate(edges):
        for w in graph.neighbors(v):
            if w == u:
                continue
            operator[i, index[(v, w)]] = 1
    return operator, edges


def integer_matrix_power_trace(matrix: NDArray[np.object_], power: int) -> int:
    """``tr(M^power)`` in exact integers, by repeated squaring."""
    if power < 1:
        raise ValueError(f"power must be positive; got {power}")
    result = None
    base = matrix
    exponent = power
    while exponent:
        if exponent & 1:
            result = base if result is None else result @ base
        exponent >>= 1
        if exponent:
            base = base @ base
    assert result is not None
    return int(np.trace(result))


def closed_geodesic_count(graph: nx.Graph, length: int) -> int:
    """``N_m``: closed non-backtracking tailless walks of length ``m``, via ``tr(B^m)``."""
    operator, _ = hashimoto_operator(graph)
    return integer_matrix_power_trace(operator, length)


def enumerate_closed_geodesics(
    graph: nx.Graph, length: int
) -> Iterator[tuple[DirectedEdge, ...]]:
    """Brute-force enumeration of the same walks, as sequences of directed edges.

    Exists to referee :func:`closed_geodesic_count`.  A closed non-backtracking
    tailless walk is a cyclic sequence of directed edges in which consecutive
    edges do not reverse -- *including* the wrap-around from the last edge to the
    first, which is what "tailless" means and what a naive path enumeration
    forgets.
    """
    if length < 1:
        raise ValueError(f"length must be positive; got {length}")
    edges = directed_edges(graph)

    def extend(walk: list[DirectedEdge]) -> Iterator[tuple[DirectedEdge, ...]]:
        if len(walk) == length:
            head, tail = walk[0], walk[-1]
            if tail[1] == head[0] and head[1] != tail[0]:
                yield tuple(walk)
            return
        last = walk[-1]
        for neighbour in graph.neighbors(last[1]):
            if neighbour == last[0]:
                continue
            walk.append((last[1], neighbour))
            yield from extend(walk)
            walk.pop()

    for start in edges:
        yield from extend([start])


# --------------------------------------------------------------------------- #
# Exact integer polynomials
# --------------------------------------------------------------------------- #
def poly_multiply(left: Sequence[int], right: Sequence[int]) -> IntPolynomial:
    """Product of two integer polynomials, coefficients in ascending degree."""
    if not left or not right:
        return [0]
    out = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        if a == 0:
            continue
        for j, b in enumerate(right):
            out[i + j] += a * b
    return out


def poly_power(base: Sequence[int], exponent: int) -> IntPolynomial:
    """Integer polynomial raised to a non-negative power."""
    if exponent < 0:
        raise ValueError(f"exponent must be non-negative; got {exponent}")
    result: IntPolynomial = [1]
    current = list(base)
    while exponent:
        if exponent & 1:
            result = poly_multiply(result, current)
        exponent >>= 1
        if exponent:
            current = poly_multiply(current, current)
    return result


def characteristic_polynomial(matrix: NDArray[np.object_]) -> IntPolynomial:
    """``det(I - u M)`` as an exact integer polynomial in ``u``.

    Uses the Faddeev--LeVerrier recursion on traces, which stays in the integers
    for an integer matrix because the divisions are exact.  Expanding the
    determinant directly would be factorial; eigenvalue routines would be
    floating point and could not referee anything.
    """
    size = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square; got shape {matrix.shape}")
    # Newton's identities: e_k from power sums p_k = tr(M^k).
    powers: list[int] = []
    current = np.eye(size, dtype=object)
    for _ in range(size):
        current = current @ matrix
        powers.append(int(np.trace(current)))
    elementary = [Fraction(1)]
    for k in range(1, size + 1):
        total = Fraction(0)
        for i in range(1, k + 1):
            total += Fraction((-1) ** (i - 1)) * elementary[k - i] * powers[i - 1]
        elementary.append(total / k)
    coefficients = []
    for k, value in enumerate(elementary):
        if value.denominator != 1:
            raise ValueError(
                f"characteristic polynomial coefficient {k} is not an integer "
                f"({value}); the matrix was not integral"
            )
        coefficients.append(int(value) * (-1) ** k)
    return coefficients


def ihara_zeta_reciprocal(graph: nx.Graph) -> IntPolynomial:
    """``1/zeta(u) = det(I - u B)``, computed from the edge operator directly."""
    operator, _ = hashimoto_operator(graph)
    return characteristic_polynomial(operator)


def graph_degree(graph: nx.Graph) -> int:
    """Common degree of a regular graph."""
    degrees = {d for _, d in graph.degree()}
    if len(degrees) != 1:
        raise ValueError(f"graph is not regular; degrees present: {sorted(degrees)}")
    return degrees.pop()


def is_regular(graph: nx.Graph) -> bool:
    return len({d for _, d in graph.degree()}) == 1


def bass_reciprocal(graph: nx.Graph) -> IntPolynomial:
    """``(1-u^2)^{r-1} det(I - A u + q u^2 I)`` for a ``(q+1)``-regular graph.

    The right-hand side of Bass's theorem, over the ``|V| x |V|`` adjacency
    matrix instead of the ``2|E| x 2|E|`` edge operator.  Compared against
    :func:`ihara_zeta_reciprocal` coefficient by coefficient, this is a genuine
    check: the two are computed from different matrices of different sizes by
    different recursions.
    """
    degree = graph_degree(graph)
    q = degree - 1
    nodes = sorted(graph)
    size = len(nodes)
    rank = graph.number_of_edges() - size + 1
    index = {node: i for i, node in enumerate(nodes)}

    # det(I - Au + q u^2 I) via Faddeev-LeVerrier on the matrix pencil, done by
    # expanding in u with exact integer polynomial entries is expensive; instead
    # use det(I - Au + qu^2 I) = prod_j (1 - lambda_j u + q u^2) and get the
    # elementary symmetric functions of the lambda_j from char poly of A.
    adjacency = np.zeros((size, size), dtype=object)
    for u_node, v_node in graph.edges():
        i, j = index[u_node], index[v_node]
        adjacency[i, j] = 1
        adjacency[j, i] = 1
    # char poly of A as det(x I - A): coefficients of x^k
    det_i_minus_ua = characteristic_polynomial(adjacency)  # det(I - uA)

    # det(I - Au + qu^2 I) = det((1 + qu^2) I - u A)
    #                      = (1 + qu^2)^n det(I - (u/(1+qu^2)) A)
    # Writing det(I - vA) = sum_k c_k v^k with c_k from det_i_minus_ua, the
    # substitution v = u/(1+qu^2) gives
    #   sum_k c_k u^k (1+qu^2)^{n-k},
    # which is an exact integer polynomial with no division.
    total: IntPolynomial = [0] * (2 * size + 1)
    for k, c_k in enumerate(det_i_minus_ua):
        if c_k == 0:
            continue
        term = poly_multiply([c_k], poly_power([1, 0, q], size - k))
        for power, value in enumerate(term):
            total[power + k] += value
    while len(total) > 1 and total[-1] == 0:
        total.pop()

    return poly_multiply(total, poly_power([1, 0, -1], rank - 1))


# --------------------------------------------------------------------------- #
# Prime geodesics
# --------------------------------------------------------------------------- #
def moebius(n: int) -> int:
    """Moebius function, by trial division."""
    if n < 1:
        raise ValueError(f"argument must be positive; got {n}")
    if n == 1:
        return 1
    result = 1
    remaining = n
    factor = 2
    while factor * factor <= remaining:
        if remaining % factor == 0:
            remaining //= factor
            if remaining % factor == 0:
                return 0
            result = -result
        factor += 1
    if remaining > 1:
        result = -result
    return result


def prime_geodesic_count(counts: dict[int, int], length: int) -> int:
    """Number of *primitive* closed geodesics of length ``m``, up to rotation.

    Every closed geodesic of length ``m`` is a power of a primitive one whose
    length divides ``m``, and a primitive geodesic of length ``d`` contributes
    ``d`` rotations.  Hence ``N_m = sum_{d | m} d * pi(d)``, inverted by Moebius:

        pi(m) = (1/m) sum_{d | m} mu(m/d) N_d .
    """
    total = 0
    for divisor in range(1, length + 1):
        if length % divisor:
            continue
        if divisor not in counts:
            raise KeyError(f"need N_{divisor} to compute pi({length})")
        total += moebius(length // divisor) * counts[divisor]
    if total % length:
        raise ValueError(
            f"Moebius inversion gave {total} for length {length}, which is not "
            f"divisible by {length}; the geodesic counts are inconsistent"
        )
    return total // length


def prime_geodesic_counts(
    graph: nx.Graph, max_length: int, *, spectral: bool | None = None
) -> dict[int, int]:
    """``pi(m)`` for every ``m`` up to ``max_length``.

    ``spectral`` selects how the underlying ``N_m`` are obtained: the exact
    ``tr(B^m)`` route, or :func:`geodesic_counts_from_spectrum`.  Left as ``None``
    it picks the exact route while the edge operator is small enough to power in
    Python integers and the spectral one beyond that, since the exact route costs
    a ``2|E| x 2|E|`` matrix power and becomes hopeless in the low hundreds.
    """
    if spectral is None:
        spectral = 2 * graph.number_of_edges() > 400
    if spectral:
        # Exact where the adjacency matrix is small enough to multiply in Python
        # integers; the floating-point spectral route is the last resort, and
        # carries a precision floor that the exact recurrence does not.
        counts = (
            geodesic_counts_exact(graph, max_length)
            if graph.number_of_nodes() <= 400
            else geodesic_counts_from_spectrum(graph, max_length)
        )
    else:
        operator, _ = hashimoto_operator(graph)
        counts = {}
        current = np.eye(operator.shape[0], dtype=object)
        for length in range(1, max_length + 1):
            current = current @ operator
            counts[length] = int(np.trace(current))
    return {m: prime_geodesic_count(counts, m) for m in range(1, max_length + 1)}


def zeta_series(reciprocal: Sequence[int], terms: int) -> list[Fraction]:
    """Power series of ``1 / reciprocal`` to the given number of terms."""
    if not reciprocal or reciprocal[0] == 0:
        raise ValueError("reciprocal must have non-zero constant term")
    out = [Fraction(0)] * terms
    out[0] = Fraction(1, reciprocal[0])
    for n in range(1, terms):
        total = Fraction(0)
        for k in range(1, min(n, len(reciprocal) - 1) + 1):
            total += Fraction(reciprocal[k]) * out[n - k]
        out[n] = -total / reciprocal[0]
    return out


def euler_product_series(primes: dict[int, int], terms: int) -> list[Fraction]:
    """``prod_m (1 - u^m)^{-pi(m)}`` as a power series.

    The Euler product over prime geodesics.  Comparing it with
    :func:`zeta_series` is the check that ``pi`` really counts what the zeta
    function's factorisation says it counts.
    """
    series = [Fraction(0)] * terms
    series[0] = Fraction(1)
    for length, multiplicity in sorted(primes.items()):
        if multiplicity == 0 or length >= terms:
            continue
        for _ in range(multiplicity):
            # multiply by (1 - u^length)^{-1} = 1 + u^l + u^{2l} + ...
            updated = list(series)
            for power in range(length, terms):
                updated[power] += updated[power - length]
            series = updated
    return series


# --------------------------------------------------------------------------- #
# Spectrum, Ramanujan property, and the trace formula
# --------------------------------------------------------------------------- #
def adjacency_spectrum(graph: nx.Graph) -> NDArray[np.float64]:
    """Adjacency eigenvalues, ascending."""
    nodes = sorted(graph)
    matrix = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.float64)
    return np.linalg.eigvalsh(matrix)


@dataclass(frozen=True)
class RamanujanReport:
    """Whether a regular graph meets the Ramanujan bound, and by how much."""

    degree: int
    order: int
    eigenvalues: tuple[float, ...]
    trivial: tuple[float, ...]
    bound: float
    worst: float

    @property
    def is_ramanujan(self) -> bool:
        return self.worst <= self.bound + 1e-9

    @property
    def spectral_gap(self) -> float:
        return self.degree - self.worst


def ramanujan_report(graph: nx.Graph, *, atol: float = 1e-8) -> RamanujanReport:
    """Test ``|lambda| <= 2 sqrt(q)`` for every non-trivial eigenvalue.

    The trivial eigenvalues are ``+(q+1)`` once per connected component and
    ``-(q+1)`` once per bipartite component; excluding them by value rather than
    by counting components would silently drop a genuine eigenvalue that happens
    to sit at the bound.
    """
    degree = graph_degree(graph)
    q = degree - 1
    values = adjacency_spectrum(graph)
    components = list(nx.connected_components(graph))
    trivial: list[float] = []
    remaining = list(values)
    for component in components:
        subgraph = graph.subgraph(component)
        for target in ([degree, -degree] if nx.is_bipartite(subgraph) else [degree]):
            match = min(remaining, key=lambda v: abs(v - target))
            if abs(match - target) > 1e-6:
                raise ValueError(
                    f"expected a trivial eigenvalue near {target} for a component "
                    f"of size {len(component)}, closest was {match}"
                )
            remaining.remove(match)
            trivial.append(float(match))
    worst = max((abs(v) for v in remaining), default=0.0)
    return RamanujanReport(
        degree=degree,
        order=graph.number_of_nodes(),
        eigenvalues=tuple(float(v) for v in values),
        trivial=tuple(sorted(trivial)),
        bound=2.0 * math.sqrt(q),
        worst=float(worst),
    )


@dataclass(frozen=True)
class TraceFormulaCheck:
    """Spectral side against geodesic side, at each length."""

    lengths: tuple[int, ...]
    geodesic: tuple[int, ...]
    spectral: tuple[int, ...]

    @property
    def agrees(self) -> bool:
        return self.geodesic == self.spectral

    @property
    def first_disagreement(self) -> int | None:
        for length, left, right in zip(self.lengths, self.geodesic, self.spectral):
            if left != right:
                return length
        return None


def trace_formula_check(graph: nx.Graph, max_length: int) -> TraceFormulaCheck:
    """Compare ``tr(B^m)`` with the sum over the adjacency spectrum.

    For a ``(q+1)``-regular graph each adjacency eigenvalue ``lambda`` splits
    into the two roots of ``x^2 - lambda x + q = 0``, and the geodesic counts are
    their power sums, plus a contribution from the ``(1-u^2)^{r-1}`` factor that
    appears only at even lengths:

        N_m = sum_j (alpha_j^m + beta_j^m) + (r - 1)(1 + (-1)^m) .

    Both sides are produced as exact integers -- the spectral side via the
    integer coefficients of Bass's polynomial rather than via floating-point
    eigenvalues -- so agreement is an identity check and not a numerical one.
    """
    reciprocal = bass_reciprocal(graph)
    series = _log_derivative_coefficients(reciprocal, max_length)
    operator, _ = hashimoto_operator(graph)
    geodesic: list[int] = []
    current = np.eye(operator.shape[0], dtype=object)
    for _ in range(max_length):
        current = current @ operator
        geodesic.append(int(np.trace(current)))
    return TraceFormulaCheck(
        lengths=tuple(range(1, max_length + 1)),
        geodesic=tuple(geodesic),
        spectral=tuple(series),
    )


def _log_derivative_coefficients(
    reciprocal: Sequence[int], terms: int
) -> list[int]:
    """Coefficients of ``-u d/du log(reciprocal(u))``, which are the ``N_m``.

    Since ``1/zeta = det(I - uB) = prod (1 - alpha u)``, taking
    ``-u (d/du) log`` gives ``sum_m tr(B^m) u^m`` directly.  Working from the
    integer coefficients keeps the identity exact.
    """
    if not reciprocal or reciprocal[0] != 1:
        raise ValueError(
            f"expected a polynomial with constant term 1; got {list(reciprocal)[:3]}"
        )
    coefficients = [Fraction(c) for c in reciprocal]
    out: list[Fraction] = []
    for m in range(1, terms + 1):
        # Newton's identity for the power sums of the reciprocal roots.
        total = Fraction(-m) * (coefficients[m] if m < len(coefficients) else Fraction(0))
        for k in range(1, m):
            if k >= len(coefficients):
                break
            total -= coefficients[k] * out[m - k - 1]
        out.append(total)
    result: list[int] = []
    for m, value in enumerate(out, start=1):
        if value.denominator != 1:
            raise ValueError(f"geodesic count at length {m} is not an integer: {value}")
        result.append(int(value))
    return result


# --------------------------------------------------------------------------- #
# The graph prime number theorem
# --------------------------------------------------------------------------- #
def spectral_radii(graph: nx.Graph) -> tuple[float, ...]:
    """``|alpha|`` for every root of ``x^2 - lambda x + q``, over the spectrum.

    Each adjacency eigenvalue ``lambda`` of a ``(q+1)``-regular graph produces
    two poles of the zeta function, at the reciprocals of the roots of
    ``x^2 - lambda x + q = 0``.  When ``|lambda| <= 2 sqrt q`` the roots are a
    complex conjugate pair of modulus exactly ``sqrt q``; when ``|lambda|``
    exceeds the Ramanujan bound they are real and the larger one is strictly
    bigger.  That dichotomy is the whole content of the RH analogue.
    """
    degree = graph_degree(graph)
    q = degree - 1
    radii: list[float] = []
    for value in adjacency_spectrum(graph):
        discriminant = value * value - 4.0 * q
        if discriminant >= 0.0:
            root = math.sqrt(discriminant)
            radii.extend((abs(value + root) / 2.0, abs(value - root) / 2.0))
        else:
            radii.extend((math.sqrt(q), math.sqrt(q)))
    return tuple(sorted(radii))


def dominant_nontrivial_radius(graph: nx.Graph, *, atol: float = 1e-8) -> float:
    """Largest ``|alpha|`` after removing the poles forced by the trivial eigenvalues.

    The trivial eigenvalues ``+-(q+1)`` give ``|alpha| = q``, which is the pole
    responsible for the main term ``q^m/m`` and must be excluded before asking
    how big the *error* is.
    """
    degree = graph_degree(graph)
    q = degree - 1
    radii = [r for r in spectral_radii(graph) if abs(r - q) > 1e-6 and r > atol]
    if not radii:
        raise ValueError("no non-trivial poles; the graph is too small to fit an error")
    return max(radii)


def geodesic_counts_exact(graph: nx.Graph, max_length: int) -> dict[int, int]:
    """``N_m`` in exact integers, without eigenvalues and without powering ``B``.

    Each adjacency eigenvalue ``lambda`` contributes ``s_m = alpha^m + beta^m``
    for the roots of ``x^2 - lambda x + q``, and those obey the three-term
    recurrence

        s_m = lambda s_{m-1} - q s_{m-2},   s_0 = 2,  s_1 = lambda .

    Summing over the spectrum turns it into a recurrence in the *matrix*,

        T_m(A) = A T_{m-1}(A) - q T_{m-2}(A),   T_0 = 2I,  T_1 = A ,

    so that ``sum_j s_m^{(j)} = tr(T_m(A))``, and

        N_m = tr(T_m(A)) + (r - 1)(1 + (-1)^m) .

    This is the route that should have been used first.  It is exact, so it has
    no precision floor at all -- unlike :func:`geodesic_counts_from_spectrum`,
    whose guard caps the usable length at $11$ for a degree-14 graph and thereby
    made the statistic unreliable.  And it costs ``max_length`` multiplications of
    the ``|V| x |V|`` integer adjacency matrix rather than a ``2|E| x 2|E|``
    matrix power, so it reaches graphs the direct route cannot: for the
    ``120``-vertex LPS graph, ``2|E| = 1680`` is hopeless while ``|V| = 120`` is
    routine.

    Entries grow like ``degree^m``, so the arbitrary-precision integers get large
    but stay exact; the cost is cubic in ``|V|`` and linear in ``max_length``.
    """
    if max_length < 1:
        raise ValueError(f"max_length must be positive; got {max_length}")
    degree = graph_degree(graph)
    q = degree - 1
    nodes = sorted(graph)
    size = len(nodes)
    rank = graph.number_of_edges() - size + 1
    index = {node: i for i, node in enumerate(nodes)}
    adjacency = np.zeros((size, size), dtype=object)
    for u_node, v_node in graph.edges():
        i, j = index[u_node], index[v_node]
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    previous = 2 * np.eye(size, dtype=object)  # T_0
    current = adjacency.copy()  # T_1
    counts: dict[int, int] = {}
    for m in range(1, max_length + 1):
        if m == 1:
            term = current
        else:
            term = adjacency @ current - q * previous
            previous, current = current, term
        counts[m] = int(np.trace(term)) + (rank - 1) * (1 + (-1) ** m)
    return counts


def geodesic_counts_from_spectrum(
    graph: nx.Graph, max_length: int, *, relative_margin: float = 1e-3
) -> dict[int, int]:
    """``N_m`` from the adjacency spectrum, for graphs too large to power ``B``.

    Each eigenvalue ``lambda`` contributes the power sums of the roots of
    ``x^2 - lambda x + q``, and the ``(1-u^2)^{r-1}`` factor of Bass's formula
    contributes at even lengths only:

        N_m = sum_j (alpha_j^m + beta_j^m) + (r - 1)(1 + (-1)^m) .

    The exact route through ``tr(B^m)`` costs a ``2|E| x 2|E|`` matrix power in
    Python integers, which is hopeless past a few hundred edges; this costs one
    eigendecomposition.  It is refereed against the exact route on every graph
    small enough for both.

    **Precision guard.**  The quantity this feeds is a cancellation: the main
    term is ``q^m`` and the error being measured is ``q^{m/2}``, so eigenvalues
    known to relative accuracy ``eps`` inject an absolute error of roughly
    ``eps * m * q^{m-1} * |V|`` into ``N_m``.  Once that approaches ``q^{m/2}``
    the measured error is the eigensolver's, and this function refuses rather
    than returning it.
    """
    degree = graph_degree(graph)
    q = degree - 1
    size = graph.number_of_nodes()
    rank = graph.number_of_edges() - size + 1
    values = adjacency_spectrum(graph)
    eps = 16.0 * float(np.finfo(np.float64).eps) * degree

    counts: dict[int, int] = {}
    for m in range(1, max_length + 1):
        signal = q ** (m / 2.0)
        noise = eps * m * q ** (m - 1.0) * size
        if noise > relative_margin * signal:
            raise ValueError(
                f"at length {m} the eigenvalue round-off contributes about "
                f"{noise:.3e} to N_m while the error term being measured is only "
                f"{signal:.3e}; reduce max_length below {m} for this graph"
            )
        total = 0.0
        for value in values:
            discriminant = value * value - 4.0 * q
            if discriminant >= 0.0:
                root = math.sqrt(discriminant)
                total += ((value + root) / 2.0) ** m + ((value - root) / 2.0) ** m
            else:
                # complex conjugate pair of modulus sqrt(q): the power sum is
                # 2 q^{m/2} cos(m theta), evaluated without complex arithmetic
                theta = math.atan2(math.sqrt(-discriminant), value)
                total += 2.0 * q ** (m / 2.0) * math.cos(m * theta)
        total += (rank - 1) * (1 + (-1) ** m)
        counts[m] = int(round(total))
    return counts


@dataclass(frozen=True)
class RiemannHypothesisTest:
    """The RH analogue as a boundedness statement, with no fitting.

    The error in the graph prime number theorem is a sum of terms ``alpha_j^m``
    over the non-trivial poles.  For a Ramanujan graph every such ``alpha`` has
    modulus exactly ``sqrt q``, so the *normalised* error

        R(m) = |pi(m) - main(m)| * m / q^{m/2}

    is bounded in ``m``.  If some ``|alpha| > sqrt q`` -- that is, if the graph is
    not Ramanujan -- then ``R(m)`` grows geometrically, at rate
    ``|alpha_max| / sqrt q``.

    Boundedness is the right test and a fitted exponent is not.  The individual
    ``alpha_j^m`` oscillate in phase, so the error passes near zero at some
    lengths, and regressing ``log|error|`` on ``m`` through those dips returns a
    slope that reflects the sampling rather than the growth.  Attempting it here
    gave fit residuals of ``0.8`` to ``1.0`` in the log -- factor-of-three scatter
    -- and exponents that missed the predicted value at every graph tested,
    including ones where the prediction is exact by construction.
    """

    degree: int
    lengths: tuple[int, ...]
    normalised: tuple[float, ...]
    is_ramanujan: bool
    bipartite: bool
    dominant_radius: float

    @property
    def growth(self) -> float:
        """Geometric growth of ``R(m)`` over the measured range.

        Taken as the ratio of the largest value in the upper half of the range to
        the largest in the lower half, converted to a per-step factor.  Maxima
        rather than individual values, because the oscillation means any single
        length can land in a dip.
        """
        half = len(self.normalised) // 2
        low = max(self.normalised[:half])
        high = max(self.normalised[half:])
        span = self.lengths[-1] - self.lengths[half - 1]
        if low <= 0.0 or span <= 0:
            return float("nan")
        return float((high / low) ** (1.0 / span))

    @property
    def predicted_growth(self) -> float:
        """``|alpha_max| / sqrt q``: unity exactly when the graph is Ramanujan."""
        return self.dominant_radius / math.sqrt(self.degree - 1)

    @property
    def bounded(self) -> bool:
        return self.growth < 1.05

    @property
    def agrees_with_spectrum(self) -> bool:
        """Whether the measured growth reproduces the spectral prediction."""
        return abs(self.growth - self.predicted_growth) <= 0.08


def riemann_hypothesis_test(
    graph: nx.Graph,
    *,
    max_length: int = 18,
    min_length: int = 4,
    spectral: bool | None = None,
) -> RiemannHypothesisTest:
    """Normalised prime-geodesic error against the Ramanujan bound.

    Computes ``R(m) = |pi(m) - main(m)| m / q^{m/2}`` from exact integer geodesic
    counts, and compares its growth with ``|alpha_max|/sqrt q`` read off the
    spectrum.  The two are independent: one comes from counting closed walks, the
    other from diagonalising the adjacency matrix.
    """
    degree = graph_degree(graph)
    q = degree - 1
    if q < 2:
        raise ValueError(f"need degree at least 3; got {degree}")
    if not nx.is_connected(graph):
        # Each component contributes its own main term, so a disconnected graph
        # counts c q^m/m and the missing (c-1) q^m/m masquerades as an error that
        # grows like sqrt(q) per step -- large, clean, and entirely an artefact.
        raise ValueError(
            f"graph has {nx.number_connected_components(graph)} components; the "
            "prime geodesic theorem's main term is per-component, so the test is "
            "meaningless here"
        )
    bipartite = nx.is_bipartite(graph)
    counts = prime_geodesic_counts(graph, max_length, spectral=spectral)
    lengths = tuple(
        m for m in range(min_length, max_length + 1) if not bipartite or m % 2 == 0
    )
    if len(lengths) < 4:
        raise ValueError(f"need at least 4 usable lengths; got {len(lengths)}")
    # The subtraction must happen in exact integers.  pi(m) is of size q^m/m --
    # about 10^44 at m = 40, q = 13 -- while the error being measured is only
    # q^{m/2}, about 10^22.  Coercing pi(m) to a float to subtract a float main
    # term destroys everything below 10^28, which silently annihilated the tail
    # of every sequence and left a few spurious spikes where it did not.
    # Multiplying through by m clears the denominator and keeps it integral:
    #
    #     R(m) = |pi(m) m - w q^m| / q^{m/2} .
    weight = 2 if bipartite else 1
    normalised = tuple(
        float(abs(counts[m] * m - weight * q**m)) / q ** (m / 2.0) for m in lengths
    )
    return RiemannHypothesisTest(
        degree=degree,
        lengths=lengths,
        normalised=normalised,
        is_ramanujan=ramanujan_report(graph).is_ramanujan,
        bipartite=bipartite,
        dominant_radius=dominant_nontrivial_radius(graph),
    )


@dataclass(frozen=True)
class PrimeGeodesicFit:
    """Measured error exponent in ``pi(m) ~ q^m / m``, against its prediction.

    Retained for the record.  The fitted exponent is *not* reliable, because the
    error oscillates; see :class:`RiemannHypothesisTest`, which is the instrument
    the conclusions use.  ``residual`` is the tell: values near $1$ mean the
    log-linear model does not describe the sequence.
    """

    degree: int
    lengths: tuple[int, ...]
    counts: tuple[int, ...]
    main_terms: tuple[float, ...]
    exponent: float
    residual: float
    predicted_exponent: float
    ramanujan_exponent: float
    is_ramanujan: bool
    bipartite: bool

    @property
    def matches_prediction(self) -> bool:
        """Whether the measured exponent reproduces ``log|alpha_max|``.

        This has no free parameter: the prediction comes entirely from the
        spectrum, and the measurement entirely from the integer geodesic counts.
        """
        return abs(self.exponent - self.predicted_exponent) <= 0.06

    @property
    def saturates_ramanujan_bound(self) -> bool:
        """Whether the error grows at exactly the rate the RH analogue permits."""
        return abs(self.exponent - self.ramanujan_exponent) <= 0.06


def prime_geodesic_fit(
    graph: nx.Graph, *, max_length: int = 14, min_length: int = 4
) -> PrimeGeodesicFit:
    """Measure how fast ``pi(m) - q^m/m`` grows, against the Ramanujan prediction.

    The graph prime number theorem gives ``pi(m) ~ q^m / m``.  The size of the
    correction is governed by the largest non-trivial ``|alpha_j|``, where the
    ``alpha_j`` are the roots of ``x^2 - lambda_j x + q``.  For a Ramanujan graph
    ``|lambda| <= 2 sqrt q`` forces ``|alpha| = sqrt q``, so the error is
    ``O(q^{m/2}/m)`` and the fitted exponent should be ``log(sqrt q) = log(q)/2``.
    A non-Ramanujan graph has some ``|alpha| > sqrt q`` and must exceed it.

    This is the sharpest quantitative content of the RH analogue that a finite
    computation can reach, and it is a prediction rather than a fit: the
    exponent is not free.
    """
    if min_length < 2 or max_length <= min_length:
        raise ValueError(
            f"need 2 <= min_length < max_length; got {min_length}, {max_length}"
        )
    degree = graph_degree(graph)
    q = degree - 1
    if q < 2:
        raise ValueError(f"need degree at least 3 for a growing count; got {degree}")
    bipartite = nx.is_bipartite(graph)
    counts = prime_geodesic_counts(graph, max_length)
    # A bipartite graph has no closed walk of odd length at all, so pi vanishes
    # identically there and the main term is carried entirely by even lengths,
    # with twice the weight.  Fitting across both parities would measure the
    # parity, not the error.
    lengths = tuple(
        m
        for m in range(min_length, max_length + 1)
        if not bipartite or m % 2 == 0
    )
    if len(lengths) < 3:
        raise ValueError(
            f"only {len(lengths)} usable lengths in [{min_length}, {max_length}]"
        )
    weight = 2.0 if bipartite else 1.0
    main = tuple(weight * q**m / m for m in lengths)
    errors = np.array(
        [abs(counts[m] - expected) for m, expected in zip(lengths, main)],
        dtype=np.float64,
    )
    usable = errors > 0
    if usable.sum() < 3:
        raise ValueError(
            f"only {int(usable.sum())} usable lengths; the counts match the main "
            "term too closely to fit an error exponent"
        )
    index = np.array(lengths, dtype=np.float64)[usable]
    slope, intercept = np.polyfit(index, np.log(errors[usable]), 1)
    residual = float(np.std(np.log(errors[usable]) - (slope * index + intercept)))
    report = ramanujan_report(graph)
    return PrimeGeodesicFit(
        degree=degree,
        lengths=lengths,
        counts=tuple(counts[m] for m in lengths),
        main_terms=main,
        exponent=float(slope),
        residual=residual,
        predicted_exponent=math.log(dominant_nontrivial_radius(graph)),
        ramanujan_exponent=0.5 * math.log(q),
        is_ramanujan=report.is_ramanujan,
        bipartite=bipartite,
    )


# --------------------------------------------------------------------------- #
# The 24 Hurwitz units
# --------------------------------------------------------------------------- #
Quaternion = tuple[Fraction, Fraction, Fraction, Fraction]


def quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    """Hamilton product, in exact rationals."""
    a1, b1, c1, d1 = left
    a2, b2, c2, d2 = right
    return (
        a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
        a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
        a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
        a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
    )


def hurwitz_units() -> tuple[Quaternion, ...]:
    """The 24 units of the Hurwitz order: the vertices of the 24-cell.

    Eight of the form ``+-1, +-i, +-j, +-k`` and sixteen of the form
    ``(+-1 +-i +-j +-k)/2``.  As a group under multiplication this is the binary
    tetrahedral group of order 24, the double cover of the rotation group of the
    tetrahedron -- which is where this connects back to the packing problem the
    project started from.
    """
    half = Fraction(1, 2)
    units: list[Quaternion] = []
    zero, one = Fraction(0), Fraction(1)
    for position in range(4):
        for sign in (one, -one):
            coordinates = [zero, zero, zero, zero]
            coordinates[position] = sign
            units.append(tuple(coordinates))  # type: ignore[arg-type]
    for signs in range(16):
        coordinates = tuple(
            half if not (signs >> bit) & 1 else -half for bit in range(4)
        )
        units.append(coordinates)  # type: ignore[arg-type]
    if len(units) != 24:
        raise ValueError(f"expected 24 units, built {len(units)}")
    return tuple(units)


def binary_tetrahedral_cayley(generators: Sequence[Quaternion] | None = None) -> nx.Graph:
    """Cayley graph of the binary tetrahedral group on a symmetric generating set.

    The default is ``{+-i, omega, omega^{-1}}`` with
    ``omega = (-1 + i + j + k)/2``, giving a ``4``-regular connected graph on the
    ``24`` units.

    The obvious choice ``{+-i, +-j, +-k}`` is *wrong* and was used here first: those
    six units generate the quaternion group of order $8$, not the binary
    tetrahedral group of order $24$, so the Cayley graph falls into three
    components of eight.  Nothing about the resulting graph looks broken -- it is
    ``6``-regular on ``24`` vertices and passes the Ramanujan test -- but the prime
    geodesic theorem counts ``3 q^m/m`` rather than ``q^m/m``, and the missing
    factor shows up as a spurious error term growing like ``sqrt q`` per step.
    An element of order $3$ is needed, and no product of ``i``, ``j``, ``k`` has one.

    The generated subgroup is verified rather than assumed.
    """
    units = hurwitz_units()
    index = {unit: i for i, unit in enumerate(units)}
    if generators is None:
        half = Fraction(1, 2)
        zero, one = Fraction(0), Fraction(1)
        omega = (-half, half, half, half)
        omega_inverse = (-half, -half, -half, -half)
        generators = [
            (zero, one, zero, zero),
            (zero, -one, zero, zero),
            omega,
            omega_inverse,
        ]
    for generator in generators:
        if generator not in index:
            raise ValueError(f"generator {generator} is not a Hurwitz unit")
    generator_set = set(generators)
    for generator in generators:
        conjugate = (generator[0], -generator[1], -generator[2], -generator[3])
        if conjugate not in generator_set:
            raise ValueError(
                f"generating set is not closed under inverse ({generator} lacks "
                f"{conjugate}), so the Cayley graph would not be undirected"
            )

    identity = (Fraction(1), Fraction(0), Fraction(0), Fraction(0))
    reached = {identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            product = quaternion_multiply(current, generator)
            if product not in reached:
                reached.add(product)
                frontier.append(product)
    if len(reached) != len(units):
        raise ValueError(
            f"generators span a subgroup of order {len(reached)}, not the full "
            f"{len(units)}; the Cayley graph would be disconnected into "
            f"{len(units) // len(reached)} components"
        )

    graph = nx.Graph()
    graph.add_nodes_from(range(len(units)))
    for unit in units:
        for generator in generators:
            product = quaternion_multiply(unit, generator)
            graph.add_edge(index[unit], index[product])
    return graph
