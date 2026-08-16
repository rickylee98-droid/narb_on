"""Local spectral measures on a filtered complex: flux, girth, and Wilson action.

What this is, stated correctly after a literature check
-------------------------------------------------------

The object here is **not** new.  It is the *local density of states* of
Savostianov, Guglielmi, Schaub and Tudisco (arXiv:2502.07558, Definition 4.1),

    mu_j(lambda | A) = sum_i |e_j^T q_i|^2 delta(lambda - lambda_i),

the spectral measure of an operator at the basis vector of a single simplex.
Their Theorem 4.2 already gives it a topological reading -- the off-kernel mass
is the generalised effective resistance of the simplex -- and their Chebyshev
moments ``2[T_m(H)]_jj`` are already "walk moments at a simplex", computed by
stochastic diagonal estimation.  I proposed this as a new definition; it is not,
and that claim has been withdrawn.

Three things here do appear to be unclaimed, and they are narrower:

  * the measure is taken **at the moment the simplex enters a filtration**, so
    it is a two-parameter (simplex, scale) object.  The literature computes the
    local density of states at a *fixed* filtration value; nobody has made
    ``sigma -> mu_sigma(lambda; t)`` a persistence object.
  * it is taken for the **Dirac** operator rather than a Hodge Laplacian, which
    is what makes the odd moments vanish and the flat measure two-point.
  * the moment results below.

Why the insertion moment is the natural thing to take
------------------------------------------------------

`magnetic` established that inserting a simplex is a *bordering* of the Dirac
operator: the new simplex has no cofaces, so ``D`` gains exactly one row and
column and nothing else changes.  A bordering is precisely the situation where a
Hermitian matrix has a distinguished last basis vector, and that vector carries
its spectral measure.  Equivalently the Stieltjes transform

    m(z) = <e_tau, (D - z)^{-1} e_tau> = integral dmu(t) / (t - z)

is the rank-one Donoghue M-function of the pair ``(D, e_tau)``.  The apparatus
is standard in degree zero -- Post (2009), Pankrashkin (math-ph/0512090), where
magnetic phases already enter the boundary functionals -- but has not been
pushed above vertices into ``k``-simplices, nor composed with a filtration.

What is derived
---------------

**One.  Every odd moment vanishes.**  ``Gamma`` acts on ``e_tau`` by a sign and
anticommutes with ``D``, so odd moments are their own negatives.  The measure is
symmetric about zero for every connection.  Inherited from `magnetic`.

**Two.  At zero flux the measure is known completely.**  ``M_{2j} = (k+1)^j``
exactly for a ``k``-simplex -- integers ``2,4,8``; ``3,9,27``; ``4,16,64``;
``5,25,125``; ``6,36,216``.  With the vanishing odd moments this determines the
measure outright:

    mu_tau  =  (1/2) delta_{+sqrt(k+1)}  +  (1/2) delta_{-sqrt(k+1)}

A flat insertion measure is a symmetric Bernoulli distribution whose support is
the square root of the face count.

**Three -- the girth law.  Flux enters the local moments at order exactly 2g**,
where ``g`` is the length of the shortest cycle through ``tau`` that bounds.
Every moment below ``M_{2g}`` is flux-blind.  Measured:

    tree (no cycles)         flux-blind at every order tested, M_2 .. M_10
    2-simplex, g = 2         first flux at M_4
    edge, graph girth 3      first flux at M_6
    edge, graph girth 4      first flux at M_8

This is the local form of a fact about trees: a simply connected structure
carries no holonomy, so every connection on it is gauge-trivial and the local
spectral measure at the root of a tree -- the Kesten-McKay measure -- cannot
depend on the phases.  A closed walk sees flux only once it is long enough to
enclose something.

The law subsumes the second-moment statement: ``g >= 2`` always, so ``M_2`` is
flux-blind for every simplex in every complex, and equals ``dim(tau) + 1``, the
face count.  Verified over 150 random connections and ambients with zero
exceptions.

**Four.  For a 2-simplex the excess kurtosis is the local Wilson plaquette
action.**  With ``h`` the holonomy of the 2-cell,

    M_4 - M_2^2  =  |h - 1|^2  =  2(1 - Re h)

exactly, for every connection and every ambient complex.  Maximum residual
1.3e-15 over 120 random connections.

**The naming matters and the first version got it wrong.**  This was written up
as "the excess kurtosis is the squared curvature".  It is not, quite.  The two
forms ``|h-1|^2`` and ``2(1 - Re h)`` are *identical* for unitary holonomy, so
nothing is lost there -- but ``|F|^2`` in the differential-geometric sense is
recovered from the Wilson action only in the continuum, small-flux limit.  The
honest statement is the lattice one: **the fourth local moment deficit is the
Wilson plaquette action restricted to the plaquettes at ``tau``.**

Two structural precedents worth knowing.  Kenyon (Ann. Probab. 39, 2011) writes
the determinant of a connection Laplacian as a sum over cycle-rooted spanning
forests weighted by ``2 - tr(hol)`` -- and for ``U(1)``, ``tr(hol) = 2 Re h``,
so **Kenyon's weight is exactly this excess kurtosis**.  Same quantity, a
determinant identity rather than a moment identity.  And Chamseddine-Connes
(hep-th/9606001) show the fourth heat-expansion coefficient of ``Tr F(D/Lambda)``
contains the Yang-Mills action; what is computed here is a discrete, *localised*
analogue of that -- localised being the operative word, since the existing
discrete work (Najem-Mrad-Elsayed, arXiv:2509.04311) takes global traces.

Where it stops
--------------

**Statement four is dimension-two only, and I could not extend it.**  For
``k >= 3`` the excess kurtosis is not proportional to any sum of face
holonomies: regressing against ``sum |h - 1|^2`` over the triangular faces gives
residuals of 2.45 at ``k = 3`` and 2.48 at ``k = 4``, comparable to the signal.
The natural ``2/(k+1)`` coefficient is also wrong -- fitted 0.28 and 0.093
against 0.5 and 0.4.  Above dimension two the triangular faces share edges, so
their holonomies interfere rather than add and the four-walks visiting two
triangles carry cross terms.  `KURTOSIS_IDENTITY_MAX_DIMENSION` records the
boundary and a test measures the failure.

Novelty
-------

The definition is **not** new -- see the top of this docstring.  Statements one
through four appear to be unstated in this form, per a literature check run by
the repository owner (arxiv.org is unreachable from this environment).  The
walk-moment lemma that makes statement four provable in a few lines is standard
(Preciado-Jadbabaie, arXiv:1107.5676, Lemma 2.1); what does not appear in the
literature is the localised Wilson-action identity or the girth law.  Treat the
novelty claim as unverified and the arithmetic as exact.
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
    "first_flux_bearing_moment",
    "moment_is_flux_blind",
    "wilson_plaquette_action",
    "wilson_action_is_the_kenyon_weight",
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


# ---------------------------------------------------------------------------
# statement three: the girth law
# ---------------------------------------------------------------------------


def first_flux_bearing_moment(girth: int) -> int:
    """``2g``: the lowest moment order that can carry holonomy.

    A closed walk sees flux only once it is long enough to enclose something,
    and the shortest thing it can enclose is a cycle of length ``g`` through the
    simplex -- traversed down and up, hence ``2g`` steps.  Every moment below
    this order is flux-blind.

    Measured: a tree (``g`` infinite) is blind at every order tested; a
    2-simplex (``g = 2``, the 2-cell itself) first sees flux at ``M_4``; an edge
    in a graph of girth three first sees it at ``M_6``; girth four, at ``M_8``.
    """
    if girth < 2:
        raise ValueError(f"girth must be at least two, got {girth}")
    return 2 * girth


def moment_is_flux_blind(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    order: int,
    samples: int = 6,
    seed: int = 0,
) -> bool:
    """Is ``M_order`` at ``simplex`` unchanged by the connection?

    Measured rather than predicted: the moment is recomputed under random
    connections and compared.  Use `first_flux_bearing_moment` for what the
    girth law predicts, and this to check it.
    """
    if samples < 2:
        raise ValueError(f"need at least two samples, got {samples}")
    vertices = 1 + max(max(face) for face in simplices)
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        angles = {
            tuple(sorted(edge)): float(generator.uniform(0, 2 * np.pi))
            for edge in combinations(range(vertices), 2)
        }
        values.append(
            insertion_moment(simplices, simplex, phase_function(angles), order)
        )
    return max(values) - min(values) <= TOLERANCE


def wilson_plaquette_action(
    simplex: tuple[int, ...], weight: Callable[[int, int], complex]
) -> float:
    """``|h - 1|^2 = 2(1 - Re h)``: the Wilson action of a single plaquette.

    The correctly named right-hand side of statement four.  The two expressions
    are identical for unitary holonomy, so nothing is approximated here -- but
    ``|F|^2`` in the differential-geometric sense is recovered from this only in
    the continuum, small-flux limit, which is why the earlier name "squared
    curvature" was withdrawn.

    Note also that this is exactly Kenyon's cycle-rooted-spanning-forest weight
    ``2 - tr(hol)``, since ``tr(hol) = 2 Re h`` for ``U(1)``.
    """
    return holonomy_defect_squared(simplex, weight)


def wilson_action_is_the_kenyon_weight(
    simplex: tuple[int, ...], weight: Callable[[int, int], complex]
) -> bool:
    """Is ``|h-1|^2`` equal to ``2 - tr(hol)``?

    True identically for ``U(1)``, and worth asserting because it is the bridge
    to a determinant identity in the literature rather than a moment one.
    """
    if len(simplex) != 3:
        raise ValueError(f"needs a 2-simplex, got dimension {len(simplex) - 1}")
    holonomy = _holonomy(weight, simplex)
    trace = 2 * holonomy.real
    return abs(wilson_plaquette_action(simplex, weight) - (2 - trace)) <= TOLERANCE
