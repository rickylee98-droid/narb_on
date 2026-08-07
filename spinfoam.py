"""Exact recoupling algebra and the semiclassical limit of spin networks.

What this targets, and what it does not
---------------------------------------
The advertised open problem in this area is usually stated as "prove the spin
foam vertex amplitude converges to general relativity".  Two corrections are
needed before writing any code.

First, the large-spin asymptotics of the Barrett-Crane ``10j`` symbol under
uniform scaling is **not open**.  Baez, Christensen and Egan showed that the
symbol is dominated by *degenerate* configurations, so the oscillatory term
carrying the Regge action is subdominant and the amplitude does **not** approach
the classical geometry in the hoped-for way.  Chasing that phase numerically
would be chasing a signal that is known to be buried.

Second, the model that does have correct semiclassical asymptotics is EPRL/FK,
whose vertex amplitude was shown by Barrett, Dowdall, Fairbairn, Gomes and
Hellmann to reproduce the Regge action.  That is a different and much heavier
computation, requiring ``SL(2,C)`` booster functions.

What *is* exactly computable, sharply refereed, and genuinely about convergence
to classical geometry is the semiclassical limit of the ``6j`` symbol.  The
Ponzano-Regge formula states that for six spins whose edge lengths
``l_i = j_i + 1/2`` bound a Euclidean tetrahedron of volume ``V``,

    {6j}  ~  cos(S_R + pi/4) / sqrt(12 pi V),      S_R = sum_i l_i theta_i,

where ``theta_i`` are the *exterior* dihedral angles and ``S_R`` is the Regge
action.  The classical action of discrete gravity appears explicitly, the
formula is checkable to any precision, and the rate at which the exact symbol
approaches it is a real and measurable quantity rather than a lookup.  When the
spins do not bound a Euclidean tetrahedron the formula is replaced by
exponential decay, which is an equally sharp prediction in the opposite regime.

Exactness
---------
Every recoupling coefficient here is computed in exact arithmetic.  The Racah
formula gives

    {6j} = S * sqrt(R),      S, R rational,

so a ``6j`` symbol is represented as the pair ``(S, R)`` and never as a float
until a float is asked for.  Spins are handled as doubled integers internally,
so half-integer spins never produce a half-integer factorial.

The implementation is refereed against statements it cannot fake: the
Biedenharn-Elliott pentagon identity, the two orthogonality relations, the Regge
symmetries, and an independent computation by ``sympy``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Surd",
    "triangle_holds",
    "triangle_coefficient_squared",
    "wigner_6j_exact",
    "wigner_6j",
    "wigner_3j_exact",
    "wigner_3j",
    "TetrahedronGeometry",
    "SIXJ_EDGE_VERTICES",
    "edge_lengths",
    "cayley_menger_volume_squared",
    "tetrahedron_geometry",
    "regge_action",
    "ponzano_regge",
    "PonzanoReggeComparison",
    "compare_to_ponzano_regge",
    "AliasingDiagnostic",
    "aliasing_diagnostic",
    "ConvergenceMeasurement",
    "convergence_exponent",
]

LOGGER = logging.getLogger(__name__)

F64 = np.float64


#: Bits of numerator and denominator retained when converting a huge exact
#: fraction to a float.  Well beyond the 53 bits a float carries, so the
#: truncation is invisible, while keeping both integers inside float range.
_RATIO_BITS = 900


def _fraction_to_float(value: Fraction) -> float:
    """Convert a possibly enormous exact fraction to a float.

    ``float(Fraction)`` raises ``OverflowError`` once numerator and denominator
    exceed float range individually, which happens long before the *ratio* does:
    a ``6j`` symbol at ``j = 60`` has parts with thousands of digits and a value
    near ``1e-3``.  Shifting both by the same amount preserves the ratio to
    within ``2^-900``.
    """
    numerator, denominator = value.numerator, value.denominator
    if numerator == 0:
        return 0.0
    shift = max(numerator.bit_length(), denominator.bit_length()) - _RATIO_BITS
    if shift > 0:
        numerator >>= shift
        denominator >>= shift
        if denominator == 0:
            raise ValueError("fraction underflowed during conversion")
    return numerator / denominator


# --------------------------------------------------------------------------- #
# Exact values of the form S * sqrt(R)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Surd:
    """The exact real number ``rational * sqrt(radicand)``.

    Recoupling coefficients are of this shape: the Racah sum is rational and the
    four triangle coefficients contribute a single square root between them.
    Keeping the two parts separate means a ``6j`` symbol is an exact object, and
    a float appears only where one is explicitly requested.
    """

    rational: Fraction
    radicand: Fraction

    def __post_init__(self) -> None:
        if self.radicand < 0:
            raise ValueError(f"radicand must be non-negative; got {self.radicand}")

    def __float__(self) -> float:
        """Convert to a float without overflowing on the way.

        At large spins the rational part and the radicand are separately far
        outside float range -- the rational overflows while the radicand
        underflows -- even though the value itself is of order ``1e-3``.
        Converting each and multiplying therefore fails.  Squaring first
        collapses them into a single well-scaled rational, and the sign is
        restored afterwards.
        """
        if self.is_zero:
            return 0.0
        magnitude = math.sqrt(_fraction_to_float(self.square()))
        return magnitude if self.rational > 0 else -magnitude

    @property
    def is_zero(self) -> bool:
        return self.rational == 0 or self.radicand == 0

    def __mul__(self, other: "Surd | Fraction | int") -> "Surd":
        if isinstance(other, Surd):
            return Surd(self.rational * other.rational, self.radicand * other.radicand)
        return Surd(self.rational * Fraction(other), self.radicand)

    __rmul__ = __mul__

    def square(self) -> Fraction:
        """The exact square, which is rational."""
        return self.rational * self.rational * self.radicand


# --------------------------------------------------------------------------- #
# Spins as doubled integers
# --------------------------------------------------------------------------- #
def _double(j) -> int:
    """Convert a spin to ``2j``, rejecting anything that is not a half-integer."""
    value = Fraction(j) * 2
    if value.denominator != 1:
        raise ValueError(f"spin must be an integer or half-integer; got {j}")
    if value < 0:
        raise ValueError(f"spin must be non-negative; got {j}")
    return int(value)


def triangle_holds(j1, j2, j3) -> bool:
    """Whether ``(j1, j2, j3)`` satisfies the triangle and integrality conditions.

    Both are required: the lengths must close a triangle, and the sum must be an
    integer, since a half-integer perimeter makes the recoupling coefficient
    vanish identically rather than merely being small.
    """
    a, b, c = _double(j1), _double(j2), _double(j3)
    if (a + b + c) % 2 != 0:
        return False
    return abs(a - b) <= c <= a + b


def _phase(n: int) -> int:
    """``(-1)**n`` as an exact integer, including for negative ``n``.

    Python's ``(-1) ** -1`` evaluates to the *float* ``-1.0``, which silently
    contaminates otherwise exact rational arithmetic; the ``3j`` sign exponent
    ``(2j1 - 2j2 - 2m3)/2`` is negative for perfectly ordinary arguments.
    """
    return -1 if n % 2 else 1


@lru_cache(maxsize=None)
def _factorial(n: int) -> int:
    if n < 0:
        raise ValueError(f"factorial of a negative integer: {n}")
    return math.factorial(n)


def triangle_coefficient_squared(j1, j2, j3) -> Fraction:
    """``Delta(j1,j2,j3)^2``, which is rational.

    ``Delta^2 = (a+b-c)! (a-b+c)! (-a+b+c)! / (a+b+c+1)!``.  Squaring is what
    keeps the arithmetic rational; the square roots are collected once, at the
    end, into a single :class:`Surd`.
    """
    if not triangle_holds(j1, j2, j3):
        return Fraction(0)
    a, b, c = _double(j1), _double(j2), _double(j3)
    return Fraction(
        _factorial((a + b - c) // 2)
        * _factorial((a - b + c) // 2)
        * _factorial((-a + b + c) // 2),
        _factorial((a + b + c) // 2 + 1),
    )


# --------------------------------------------------------------------------- #
# The 6j symbol
# --------------------------------------------------------------------------- #
def wigner_6j_exact(j1, j2, j3, j4, j5, j6) -> Surd:
    """The Wigner ``6j`` symbol as an exact ``rational * sqrt(rational)``.

    Racah's single-sum formula,

        {6j} = Delta(j1 j2 j3) Delta(j1 j5 j6) Delta(j4 j2 j6) Delta(j4 j5 j3)
               * sum_t (-1)^t (t+1)! / [ prod of six factorials ],

    with the four triangle conditions checked first.  A ``6j`` symbol vanishes
    identically unless all four triangles close, so that check is part of the
    definition rather than an optimisation.
    """
    triads = ((j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3))
    if not all(triangle_holds(*t) for t in triads):
        return Surd(Fraction(0), Fraction(0))

    a1, a2, a3, a4, a5, a6 = (_double(x) for x in (j1, j2, j3, j4, j5, j6))

    # The four triads and the three "square" sums, in doubled units then halved.
    triad_sums = [
        (a1 + a2 + a3) // 2,
        (a1 + a5 + a6) // 2,
        (a4 + a2 + a6) // 2,
        (a4 + a5 + a3) // 2,
    ]
    square_sums = [
        (a1 + a2 + a4 + a5) // 2,
        (a2 + a3 + a5 + a6) // 2,
        (a3 + a1 + a6 + a4) // 2,
    ]

    lower = max(triad_sums)
    upper = min(square_sums)
    total = Fraction(0)
    for t in range(lower, upper + 1):
        denominator = 1
        for s in triad_sums:
            denominator *= _factorial(t - s)
        for s in square_sums:
            denominator *= _factorial(s - t)
        total += Fraction(_phase(t) * _factorial(t + 1), denominator)

    radicand = Fraction(1)
    for triad in triads:
        radicand *= triangle_coefficient_squared(*triad)
    return Surd(total, radicand)


def wigner_6j(j1, j2, j3, j4, j5, j6) -> float:
    """Floating-point value of the ``6j`` symbol."""
    return float(wigner_6j_exact(j1, j2, j3, j4, j5, j6))


# --------------------------------------------------------------------------- #
# The 3j symbol
# --------------------------------------------------------------------------- #
def wigner_3j_exact(j1, j2, j3, m1, m2, m3) -> Surd:
    """The Wigner ``3j`` symbol as an exact ``rational * sqrt(rational)``.

    Included mainly so that the orthogonality relations, which mix ``3j`` and
    ``6j`` data, can be checked exactly.
    """
    if not triangle_holds(j1, j2, j3):
        return Surd(Fraction(0), Fraction(0))
    if Fraction(m1) + Fraction(m2) + Fraction(m3) != 0:
        return Surd(Fraction(0), Fraction(0))

    a1, a2, a3 = _double(j1), _double(j2), _double(j3)
    b1, b2, b3 = (int(Fraction(m) * 2) for m in (m1, m2, m3))
    for a, b in ((a1, b1), (a2, b2), (a3, b3)):
        if (a - b) % 2 != 0 or abs(b) > a:
            return Surd(Fraction(0), Fraction(0))

    prefactor_square = triangle_coefficient_squared(j1, j2, j3) * Fraction(
        _factorial((a1 + b1) // 2)
        * _factorial((a1 - b1) // 2)
        * _factorial((a2 + b2) // 2)
        * _factorial((a2 - b2) // 2)
        * _factorial((a3 + b3) // 2)
        * _factorial((a3 - b3) // 2)
    )

    lower = max(0, (a2 - a3 - b1) // 2, (a1 - a3 + b2) // 2)
    upper = min((a1 + a2 - a3) // 2, (a1 - b1) // 2, (a2 + b2) // 2)
    total = Fraction(0)
    for t in range(lower, upper + 1):
        denominator = (
            _factorial(t)
            * _factorial((a1 + a2 - a3) // 2 - t)
            * _factorial((a1 - b1) // 2 - t)
            * _factorial((a2 + b2) // 2 - t)
            * _factorial((a3 - a2 + b1) // 2 + t)
            * _factorial((a3 - a1 - b2) // 2 + t)
        )
        total += Fraction(_phase(t), denominator)

    sign = _phase((a1 - a2 - b3) // 2)
    return Surd(sign * total, prefactor_square)


def wigner_3j(j1, j2, j3, m1, m2, m3) -> float:
    """Floating-point value of the ``3j`` symbol."""
    return float(wigner_3j_exact(j1, j2, j3, m1, m2, m3))


# --------------------------------------------------------------------------- #
# The tetrahedron a 6j symbol describes
# --------------------------------------------------------------------------- #
#: Which pair of tetrahedron vertices each spin of a ``6j`` symbol labels.
#:
#: This is forced, not conventional, and getting it wrong is the easiest
#: mistake in the whole subject.  The four triads of the symbol --
#: ``(j1,j2,j3)``, ``(j1,j5,j6)``, ``(j4,j2,j6)``, ``(j4,j5,j3)`` -- must be the
#: four *faces*.  Each spin lies in exactly two triads, and identifying triad
#: ``i`` with the face opposite vertex ``i`` forces the spin in triads ``i`` and
#: ``k`` onto the edge joining the two remaining vertices.  That gives the map
#: below, under which every triad is a genuine triangle.
#:
#: The tempting alternative, ``j1, j2, j3`` on the edges meeting at one vertex,
#: is a vertex star rather than a face.  It is invariant under many symmetric
#: spin choices -- the regular tetrahedron, and anything palindromic -- so it
#: reproduces the correct asymptotics on exactly the test cases one reaches for
#: first, and fails only on a shape like ``(2,2,2,3,3,3)``.
SIXJ_EDGE_VERTICES: tuple[tuple[int, int], ...] = (
    (2, 3),  # j1
    (1, 3),  # j2
    (1, 2),  # j3
    (0, 1),  # j4
    (0, 2),  # j5
    (0, 3),  # j6
)
@dataclass(frozen=True)
class TetrahedronGeometry:
    """Euclidean tetrahedron reconstructed from six spins.

    In the Ponzano-Regge correspondence the six spins of a ``6j`` symbol are edge
    lengths ``l_i = j_i + 1/2`` of a tetrahedron, with ``(j1,j4)``, ``(j2,j5)``
    and ``(j3,j6)`` opposite pairs.  The shift by ``1/2`` is not cosmetic: it is
    what makes the asymptotics come out right, and using ``l_i = j_i`` visibly
    degrades the agreement.
    """

    lengths: tuple[float, ...]
    volume_squared: float
    dihedral_angles: tuple[float, ...] | None

    @property
    def is_euclidean(self) -> bool:
        """Whether the six lengths actually bound a Euclidean tetrahedron."""
        return self.volume_squared > 0.0

    @property
    def volume(self) -> float:
        if not self.is_euclidean:
            raise ValueError(
                "the spins do not bound a Euclidean tetrahedron, so there is no "
                "real volume; the asymptotics is exponential decay instead"
            )
        return math.sqrt(self.volume_squared)


def edge_lengths(j1, j2, j3, j4, j5, j6) -> tuple[float, ...]:
    """Ponzano-Regge edge lengths ``l_i = j_i + 1/2``."""
    return tuple(float(Fraction(j)) + 0.5 for j in (j1, j2, j3, j4, j5, j6))


def cayley_menger_volume_squared(lengths: Sequence[float]) -> float:
    """Squared volume of a tetrahedron from its six edge lengths.

    Uses the Cayley-Menger determinant.  The sign of the result is the whole
    point: a negative value means the lengths do not bound a Euclidean
    tetrahedron, which is precisely the classically forbidden regime where the
    ``6j`` symbol decays exponentially instead of oscillating.

    ``lengths`` is indexed by the spins ``(j1, ..., j6)`` and mapped to vertex
    pairs by :data:`SIXJ_EDGE_VERTICES`.
    """
    if len(lengths) != 6:
        raise ValueError(f"a tetrahedron has six edges; got {len(lengths)}")
    side = {
        frozenset(pair): float(lengths[i])
        for i, pair in enumerate(SIXJ_EDGE_VERTICES)
    }
    l12 = side[frozenset((0, 1))]
    l13 = side[frozenset((0, 2))]
    l14 = side[frozenset((0, 3))]
    l23 = side[frozenset((1, 2))]
    l24 = side[frozenset((1, 3))]
    l34 = side[frozenset((2, 3))]
    matrix = np.array(
        [
            [0.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, l12**2, l13**2, l14**2],
            [1.0, l12**2, 0.0, l23**2, l24**2],
            [1.0, l13**2, l23**2, 0.0, l34**2],
            [1.0, l14**2, l24**2, l34**2, 0.0],
        ],
        dtype=F64,
    )
    return float(np.linalg.det(matrix) / 288.0)


def tetrahedron_geometry(j1, j2, j3, j4, j5, j6) -> TetrahedronGeometry:
    """Reconstruct the tetrahedron and its exterior dihedral angles.

    The dihedral angle at an edge is obtained from the two adjacent faces via the
    vertex coordinates, which are recovered by placing the tetrahedron explicitly;
    this is more robust than a trigonometric identity in the near-degenerate
    cases that matter most here.
    """
    lengths = edge_lengths(j1, j2, j3, j4, j5, j6)
    volume_squared = cayley_menger_volume_squared(lengths)
    if volume_squared <= 0.0:
        return TetrahedronGeometry(lengths, volume_squared, None)

    side = {
        frozenset(pair): float(lengths[i])
        for i, pair in enumerate(SIXJ_EDGE_VERTICES)
    }
    l12 = side[frozenset((0, 1))]
    l13 = side[frozenset((0, 2))]
    l14 = side[frozenset((0, 3))]
    l23 = side[frozenset((1, 2))]
    l24 = side[frozenset((1, 3))]
    l34 = side[frozenset((2, 3))]

    # Place vertices 1..4 explicitly.
    p1 = np.zeros(3)
    p2 = np.array([l12, 0.0, 0.0])
    x3 = (l12**2 + l13**2 - l23**2) / (2 * l12)
    y3sq = l13**2 - x3**2
    if y3sq <= 0:
        return TetrahedronGeometry(lengths, volume_squared, None)
    p3 = np.array([x3, math.sqrt(y3sq), 0.0])
    x4 = (l12**2 + l14**2 - l24**2) / (2 * l12)
    y4 = (l13**2 + l14**2 - l34**2 - 2 * x3 * x4) / (2 * p3[1])
    z4sq = l14**2 - x4**2 - y4**2
    if z4sq <= 0:
        return TetrahedronGeometry(lengths, volume_squared, None)
    p4 = np.array([x4, y4, math.sqrt(z4sq)])
    points = (p1, p2, p3, p4)

    def dihedral(a: int, b: int) -> float:
        other = [i for i in range(4) if i not in (a, b)]
        c, d = other
        axis = points[b] - points[a]
        axis = axis / np.linalg.norm(axis)
        u = points[c] - points[a]
        v = points[d] - points[a]
        u = u - np.dot(u, axis) * axis
        v = v - np.dot(v, axis) * axis
        cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        interior = math.acos(max(-1.0, min(1.0, cosine)))
        return math.pi - interior  # exterior angle, as the Regge action requires

    angles = tuple(dihedral(a, b) for a, b in SIXJ_EDGE_VERTICES)
    return TetrahedronGeometry(lengths, volume_squared, angles)


def regge_action(geometry: TetrahedronGeometry) -> float:
    """``S_R = sum_i l_i theta_i`` with exterior dihedral angles.

    This is the Regge action of a single tetrahedron, the discrete counterpart of
    the Einstein-Hilbert action. Its appearance in the phase of the ``6j``
    symbol is the precise sense in which the recoupling algebra knows about
    classical geometry.
    """
    if geometry.dihedral_angles is None:
        raise ValueError("no dihedral angles: the tetrahedron is not Euclidean")
    return float(
        sum(l * t for l, t in zip(geometry.lengths, geometry.dihedral_angles))
    )


def ponzano_regge(j1, j2, j3, j4, j5, j6) -> float:
    """The Ponzano-Regge asymptotic prediction for the ``6j`` symbol.

    In the classically allowed regime,

        {6j} ~ cos(S_R + pi/4) / sqrt(12 pi V).

    In the forbidden regime the symbol decays exponentially and this function
    raises rather than returning a meaningless number; use
    :func:`compare_to_ponzano_regge`, which handles both regimes.
    """
    geometry = tetrahedron_geometry(j1, j2, j3, j4, j5, j6)
    if not geometry.is_euclidean:
        raise ValueError(
            "the spins are classically forbidden (negative Cayley-Menger "
            "determinant); the oscillatory formula does not apply"
        )
    action = regge_action(geometry)
    return math.cos(action + math.pi / 4) / math.sqrt(
        12.0 * math.pi * geometry.volume
    )


@dataclass(frozen=True)
class PonzanoReggeComparison:
    """Exact symbol against its semiclassical prediction, in either regime."""

    spins: tuple[Fraction, ...]
    exact: float
    predicted: float | None
    volume_squared: float
    action: float | None

    @property
    def classically_allowed(self) -> bool:
        return self.volume_squared > 0.0

    @property
    def absolute_error(self) -> float | None:
        if self.predicted is None:
            return None
        return abs(self.exact - self.predicted)


def compare_to_ponzano_regge(j1, j2, j3, j4, j5, j6) -> PonzanoReggeComparison:
    """Compare the exact ``6j`` symbol with the Ponzano-Regge formula."""
    exact = wigner_6j(j1, j2, j3, j4, j5, j6)
    geometry = tetrahedron_geometry(j1, j2, j3, j4, j5, j6)
    spins = tuple(Fraction(j) for j in (j1, j2, j3, j4, j5, j6))
    if not geometry.is_euclidean:
        return PonzanoReggeComparison(
            spins, exact, None, geometry.volume_squared, None
        )
    action = regge_action(geometry)
    predicted = math.cos(action + math.pi / 4) / math.sqrt(
        12.0 * math.pi * geometry.volume
    )
    return PonzanoReggeComparison(
        spins, exact, predicted, geometry.volume_squared, action
    )


# --------------------------------------------------------------------------- #
# Measuring the rate of approach to classical geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AliasingDiagnostic:
    """How the Regge phase advances per unit of the scaling parameter.

    Spins are quantised, so a scaling study can only sample at integer multiples
    of a shape.  If the Regge phase advances by nearly a whole number of cycles
    per step, consecutive samples are phase-locked and the measured oscillation
    is a slow *beat* rather than the true one.  Any envelope estimated over a
    window shorter than that beat is meaningless.
    """

    cycles_per_step: float
    distance_to_integer: float
    beat_period: float

    @property
    def is_aliased(self) -> bool:
        """Whether the sampling cannot resolve the oscillation."""
        return self.distance_to_integer < 0.02


def aliasing_diagnostic(shape: Sequence[int], probe: int = 100) -> AliasingDiagnostic:
    """Detect aliasing between the Regge phase and integer spin sampling.

    Measured rather than predicted: the phase rate is obtained by differencing
    the Regge action at two large scalings of the shape, which avoids having to
    model how the ``+1/2`` shift perturbs the geometry.
    """
    if len(shape) != 6:
        raise ValueError(f"a shape has six spins; got {len(shape)}")
    low = compare_to_ponzano_regge(*[probe * s for s in shape])
    high = compare_to_ponzano_regge(*[2 * probe * s for s in shape])
    if low.action is None or high.action is None:
        raise ValueError("the shape is classically forbidden, so it has no phase")
    rate = (high.action - low.action) / probe
    cycles = rate / (2.0 * math.pi)
    distance = abs(cycles - round(cycles))
    beat = float("inf") if distance <= 0.0 else 1.0 / distance
    return AliasingDiagnostic(cycles, distance, beat)


@dataclass(frozen=True)
class ConvergenceMeasurement:
    """Rate at which the exact symbol approaches Ponzano-Regge."""

    shape: tuple[int, ...]
    exponent: float
    n_points: int
    diagnostic: AliasingDiagnostic

    @property
    def is_trustworthy(self) -> bool:
        return not self.diagnostic.is_aliased


def convergence_exponent(
    shape: Sequence[int],
    *,
    max_scale: int = 200,
    window: int = 10,
    fit_from: int = 40,
) -> ConvergenceMeasurement:
    """Fit ``|exact - Ponzano-Regge| * sqrt(12 pi V) ~ lambda^alpha``.

    Normalising by the envelope is essential: the raw difference is dominated by
    the oscillation, and a relative error diverges wherever the cosine passes
    through zero, so neither is a usable measure of convergence.

    The result carries an :class:`AliasingDiagnostic`.  When that reports
    aliasing the exponent is not meaningful, and ``is_trustworthy`` says so
    rather than leaving a spurious number to be read as physics.
    """
    scales, errors = [], []
    for scale in range(1, max_scale + 1):
        spins = tuple(scale * s for s in shape)
        geometry = tetrahedron_geometry(*spins)
        if not geometry.is_euclidean:
            raise ValueError(f"shape {tuple(shape)} is classically forbidden")
        comparison = compare_to_ponzano_regge(*spins)
        envelope = math.sqrt(12.0 * math.pi * geometry.volume)
        scales.append(float(scale))
        errors.append(abs(comparison.exact - comparison.predicted) * envelope)

    x = np.array(scales, dtype=F64)
    y = np.array(errors, dtype=F64)
    upper = np.array(
        [y[max(0, i - window) : i + window + 1].max() for i in range(y.size)]
    )
    mask = x >= fit_from
    if mask.sum() < 10:
        raise ValueError("not enough points above fit_from to fit an exponent")
    slope = float(np.polyfit(np.log(x[mask]), np.log(upper[mask]), 1)[0])
    return ConvergenceMeasurement(
        shape=tuple(int(s) for s in shape),
        exponent=slope,
        n_points=int(mask.sum()),
        diagnostic=aliasing_diagnostic(shape),
    )
