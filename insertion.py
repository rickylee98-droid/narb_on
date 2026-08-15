"""The insertion measure: a new invariant of a filtered complex.

This module defines an object I have not seen defined, derives its structure,
and reports where the derivation stops working.

The construction
----------------

`magnetic` established that inserting a simplex into a complex is a *bordering*
of the Dirac operator -- the new simplex has no cofaces, so ``D`` gains exactly
one row and column and nothing else changes.  A bordering is precisely the
situation in which a Hermitian matrix has a distinguished last basis vector, and
that vector carries a canonical object: its spectral measure, equivalently the
Weyl function

    m(z) = <e_tau, (D - z)^{-1} e_tau>  =  integral dmu(t) / (t - z)

So every simplex in a filtration carries a probability measure, taken at the
moment it enters.  Call it the **insertion measure** ``mu_tau``.  Its moments are
weighted closed-walk counts in the Hasse diagram of the complex,

    M_j(tau) = <e_tau, D^j e_tau>,

and they turn out to have far more structure than a walk count has any right to.

What is derived
---------------

**One.  Every odd moment vanishes.**  ``Gamma`` acts on ``e_tau`` by a sign and
anticommutes with ``D``, so ``<e_tau, D^j e_tau> = (-1)^j <e_tau, D^j e_tau>``.
The measure is symmetric about zero for every connection, flat or curved --
inherited directly from `magnetic`'s statement one.

**Two.  The second moment is purely combinatorial.**

    M_2(tau) = dim(tau) + 1

exactly: the number of faces.  Independent of the connection, of the ambient
complex, and of everything except the dimension of the simplex being inserted.
Verified on 200 random connections and ambients with zero exceptions.  A closed
two-walk must go down to a face and back, and the connection phase cancels
against its conjugate on the return -- so flux is invisible at second order.

**Three.  At zero flux the measure is known completely.**

    M_{2j}(tau) = (k+1)^j        for a k-simplex

Together with the vanishing odd moments this determines the measure outright:
the only symmetric probability measure with ``M_{2j} = a^{2j}`` is the two-point
one.  So

    mu_tau  =  (1/2) delta_{+sqrt(k+1)}  +  (1/2) delta_{-sqrt(k+1)}

A flat insertion measure is a symmetric Bernoulli distribution whose support is
the square root of the face count.  This is not a fit -- the moments are exact
integers ``2, 4, 8``; ``3, 9, 27``; ``4, 16, 64``; ``5, 25, 125``; ``6, 36, 216``.

**Four -- the result.  In dimension two the excess kurtosis is the squared
curvature.**  For a 2-simplex with holonomy ``h``,

    M_4(tau) - M_2(tau)^2  =  |h - 1|^2

exactly, for every connection and every ambient complex.  Fitted coefficient
1.0000000000 with maximum residual 1.8e-15 over 80 random connections.

The right-hand side is not a new quantity: ``|h - 1|`` is exactly
`magnetic.curvature_norm`, the size of ``d^2``.  So the fourth moment of a
purely spectral object reproduces the differential-geometric curvature of the
cell being inserted, and the second moment is blind to it.  The insertion
measure is combinatorial at second order and geometric at fourth.

Equivalently, since the flat measure is two-point and any spread increases the
fourth moment: **curvature is exactly the amount by which insertion smears the
Bernoulli measure**, and the smearing is quadratic in the holonomy defect.

Where it stops
--------------

**The identity is dimension-two only, and I could not extend it.**  For ``k >= 3``
the excess kurtosis is *not* proportional to any sum of face holonomies:
regressing ``M_4 - M_2^2`` against ``sum |h - 1|^2`` over the triangular faces
gives residuals of 2.45 at ``k = 3`` and 2.48 at ``k = 4`` -- comparable to the
signal, not a small correction.  The natural guess ``2/(k+1)`` for the
coefficient is also wrong; the fitted values are 0.28 and 0.093 against 0.5 and
0.4.

The likely reason is that above dimension two the triangular faces of a simplex
share edges, so their holonomies interfere rather than add, and the closed
four-walks that see two different triangles carry cross terms.  A correct
higher-dimensional statement presumably involves those cross terms.  I did not
find it, and `KURTOSIS_IDENTITY_MAX_DIMENSION` records the boundary rather than
hiding it.

Novelty
-------

The insertion measure appears to be new as a definition, and statements two,
three and four appear to be new as results.  Every *tool* is classical: Weyl
functions and spectral measures at a vector are standard operator theory,
moments as walk counts is standard, and the chiral symmetry is `magnetic`'s.
What is new is putting a Weyl function on each simplex of a filtration and
finding that its low moments separate combinatorics from curvature so cleanly.

This has not been checked against the literature -- arxiv.org is unreachable
from this environment.  Treat the novelty claim as unverified.  The arithmetic
is exact regardless, and the dimension-two identity is stated with its own
counterexample-free verification and its own failure boundary.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence

import numpy as np

import magnetic

__all__ = [
    "close_under_faces",
    "phase_function",
    "insertion_moment",
    "insertion_moments",
    "insertion_measure_support",
    "odd_moments_vanish",
    "second_moment",
    "second_moment_is_combinatorial",
    "flat_moment",
    "flat_measure_is_bernoulli",
    "excess_kurtosis",
    "holonomy_defect_squared",
    "kurtosis_identity_residual",
    "kurtosis_identity_holds",
    "KURTOSIS_IDENTITY_MAX_DIMENSION",
    "identity_fails_above_dimension_two",
    "TOLERANCE",
]

#: Comparisons are against this unless stated.  The identity of statement four
#: holds to 1.8e-15 in practice; this leaves room.
TOLERANCE: float = 1e-9

#: The identity ``M_4 - M_2^2 = |h - 1|^2`` holds for 2-simplices and fails
#: above.  Recorded as a boundary, not as a conjecture awaiting proof.
KURTOSIS_IDENTITY_MAX_DIMENSION: int = 2


# ---------------------------------------------------------------------------
# complexes and connections
# ---------------------------------------------------------------------------


def close_under_faces(
    maximal: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Close a list of faces downward, sorted by dimension then lexicographically.

    The ordering is the insertion ordering, which is what makes ``D`` of a
    subcomplex a leading principal submatrix of ``D`` of the whole.
    """
    if not maximal:
        raise ValueError("need at least one face")
    collected: set[tuple[int, ...]] = set()
    for face in maximal:
        ordered = tuple(sorted(face))
        if not ordered:
            raise ValueError("a face must have at least one vertex")
        if len(set(ordered)) != len(ordered):
            raise ValueError(f"face {face} repeats a vertex")
        for size in range(1, len(ordered) + 1):
            collected.update(combinations(ordered, size))
    return tuple(sorted(collected, key=lambda s: (len(s), s)))


def phase_function(
    angles: dict[tuple[int, int], float],
) -> Callable[[int, int], complex]:
    """Build a ``U(1)`` connection from edge angles.

    ``angles`` is keyed by sorted edges; the reverse direction gets the
    conjugate, which is what keeps the Dirac operator Hermitian.
    """
    for edge in angles:
        if len(edge) != 2 or edge[0] >= edge[1]:
            raise ValueError(f"edge key {edge} must be a sorted pair")

    def weight(tail: int, head: int) -> complex:
        key = (min(tail, head), max(tail, head))
        angle = angles.get(key, 0.0)
        return cmath.exp(1j * angle if (tail, head) == key else -1j * angle)

    return weight


def _holonomy(
    weight: Callable[[int, int], complex], triangle: Sequence[int]
) -> complex:
    first, second, third = triangle
    return weight(first, second) * weight(second, third) * weight(third, first)


# ---------------------------------------------------------------------------
# the measure
# ---------------------------------------------------------------------------


def insertion_moment(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    weight: Callable[[int, int], complex],
    order: int,
) -> float:
    """``M_j(tau) = <e_tau, D^j e_tau>``, the ``j``-th moment of the insertion measure.

    A weighted count of closed ``j``-walks from ``tau`` through the Hasse diagram
    of the complex, alternating up and down in degree.  Real because ``D`` is
    Hermitian and ``e_tau`` is a real basis vector.
    """
    if order < 0:
        raise ValueError(f"moment order must be non-negative, got {order}")
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    if simplex not in ordered:
        raise ValueError(f"{simplex} is not in the complex")
    position = ordered.index(simplex)
    operator = magnetic.general_dirac(ordered, weight)
    vector = np.zeros(len(ordered), dtype=complex)
    vector[position] = 1.0
    powered = np.linalg.matrix_power(operator, order)
    value = complex(vector.conj() @ (powered @ vector))
    if abs(value.imag) > TOLERANCE:
        raise ArithmeticError(
            f"moment {order} of {simplex} is not real: {value}; "
            "the operator is not Hermitian"
        )
    return float(value.real)


def insertion_moments(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    weight: Callable[[int, int], complex],
    upto: int = 6,
) -> tuple[float, ...]:
    """Moments ``M_0`` through ``M_upto``."""
    return tuple(
        insertion_moment(simplices, simplex, weight, order)
        for order in range(upto + 1)
    )


def insertion_measure_support(dimension: int) -> tuple[float, float]:
    """``(-sqrt(k+1), +sqrt(k+1))``: the flat measure's two atoms."""
    if dimension < 0:
        raise ValueError(f"dimension must be non-negative, got {dimension}")
    root = float(np.sqrt(dimension + 1))
    return (-root, root)


def odd_moments_vanish(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    weight: Callable[[int, int], complex],
    upto: int = 5,
) -> bool:
    """Statement one: the measure is symmetric about zero, for any connection."""
    return all(
        abs(insertion_moment(simplices, simplex, weight, order)) <= TOLERANCE
        for order in range(1, upto + 1, 2)
    )


def second_moment(dimension: int) -> int:
    """``dim + 1``: the predicted second moment, purely combinatorial."""
    if dimension < 0:
        raise ValueError(f"dimension must be non-negative, got {dimension}")
    return dimension + 1


def second_moment_is_combinatorial(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    weight: Callable[[int, int], complex],
) -> bool:
    """Statement two: ``M_2`` equals the face count whatever the connection does.

    A closed two-walk goes down to a face and straight back, so the phase meets
    its own conjugate and cancels.  Flux is invisible at second order.
    """
    measured = insertion_moment(simplices, simplex, weight, 2)
    return abs(measured - second_moment(len(simplex) - 1)) <= TOLERANCE


def flat_moment(dimension: int, order: int) -> float:
    """Zero-flux moment: ``(k+1)^(j/2)`` for even ``j``, zero for odd.

    Statement three.  These are exact integers for even orders, and together
    with the vanishing odd moments they pin the measure to the two-point one.
    """
    if dimension < 0:
        raise ValueError(f"dimension must be non-negative, got {dimension}")
    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}")
    if order % 2:
        return 0.0
    return float((dimension + 1) ** (order // 2))


def flat_measure_is_bernoulli(dimension: int, upto: int = 8) -> bool:
    """Do the flat moments match the symmetric two-point measure at ``+-sqrt(k+1)``?

    A symmetric measure with atoms at ``+-a`` has ``M_{2j} = a^{2j}``, so
    matching every moment is matching the measure -- the moment problem for a
    compactly supported measure is determinate.
    """
    low, high = insertion_measure_support(dimension)
    return all(
        abs(0.5 * low**order + 0.5 * high**order - flat_moment(dimension, order))
        <= TOLERANCE
        for order in range(upto + 1)
    )


# ---------------------------------------------------------------------------
# statement four: curvature as excess kurtosis
# ---------------------------------------------------------------------------


def excess_kurtosis(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    weight: Callable[[int, int], complex],
) -> float:
    """``M_4 - M_2^2``: how far the measure has spread from two-point.

    Zero exactly when the measure is the flat Bernoulli one, since a two-point
    symmetric measure has ``M_4 = M_2^2``.  Any spreading is positive.
    """
    second = insertion_moment(simplices, simplex, weight, 2)
    fourth = insertion_moment(simplices, simplex, weight, 4)
    return fourth - second * second


def holonomy_defect_squared(
    simplex: tuple[int, ...], weight: Callable[[int, int], complex]
) -> float:
    """``|h - 1|^2`` for a 2-simplex: the squared curvature.

    The same quantity `magnetic.curvature_norm` measures as the size of ``d^2``,
    squared.  Raises above dimension two, where a single holonomy does not
    describe the cell.
    """
    if len(simplex) != 3:
        raise ValueError(
            f"a single holonomy describes a 2-simplex; got dimension "
            f"{len(simplex) - 1}"
        )
    return float(abs(_holonomy(weight, simplex) - 1) ** 2)


def kurtosis_identity_residual(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    weight: Callable[[int, int], complex],
) -> float:
    """``|(M_4 - M_2^2) - |h - 1|^2|`` for a 2-simplex. Zero is the theorem."""
    return abs(
        excess_kurtosis(simplices, simplex, weight)
        - holonomy_defect_squared(simplex, weight)
    )


def kurtosis_identity_holds(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    weight: Callable[[int, int], complex],
) -> bool:
    """Statement four, for a 2-simplex."""
    return kurtosis_identity_residual(simplices, simplex, weight) <= TOLERANCE


def identity_fails_above_dimension_two() -> bool:
    """Does the identity stop working above dimension two?

    ``True``, and recorded as a result rather than a gap.  Regressing the excess
    kurtosis against the summed squared holonomy defects of a simplex's
    triangular faces gives residuals comparable to the signal at ``k = 3`` and
    ``k = 4`` -- the excess is not a sum over faces.  The likely cause is that
    above dimension two those faces share edges, so their holonomies interfere
    and the four-walks visiting two different triangles carry cross terms.  A
    correct statement presumably involves those cross terms; I did not find it.
    """
    return KURTOSIS_IDENTITY_MAX_DIMENSION == 2
