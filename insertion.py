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

**Four -- the fourth-moment law, in every dimension.**  For any simplex ``tau``
of any dimension, any connection, any ambient complex:

    M_4(tau)  =  M_2(tau)^2  +  S(tau)  +  sum_g |1 - omega_g|^2

Three terms with disjoint meanings, and the separation is the result:

  * ``M_2^2`` -- the two-point Bernoulli baseline of statement two, the value
    when the measure has not spread at all;
  * ``S(tau)`` -- the **sibling count**: pairs ``(sigma, f)`` where ``f`` is a
    facet of ``tau`` shared with another simplex of the same dimension.  Purely
    combinatorial and flux-blind, because the walk ``tau -> f -> sigma -> f ->
    tau`` sends each phase against its own conjugate;
  * the **Wilson sum** over ``g`` running through the *codimension-two* faces of
    ``tau``.  Each such ``g`` lies in exactly two facets ``f_1, f_2``, so
    ``tau -> f_1 -> g -> f_2 -> tau`` is a canonical Hasse plaquette, and
    ``omega_g`` is its holonomy, normalised so that ``omega_g = 1`` at zero flux
    -- the normalising sign being exactly the ``d^2 = 0`` cancellation.

Each ``omega_g`` is gauge invariant, being a closed-loop holonomy (checked to
1e-16 under vertex gauge transformations).  Verified to 1e-15 on lone simplices
of dimension two through four, on fans of triangles, on a tetrahedron shell, and
on pairs of tetrahedra.

**So the fourth moment separates combinatorics from curvature exactly.**  The
first two terms are blind to the connection; the third is the only place it
enters, and it is the Wilson action of the plaquettes at ``tau``.

**Retraction.**  An earlier version of this module reported that the identity was
dimension-two only, with the excess kurtosis failing to be a sum over face
holonomies for ``k >= 3``.  That was two mistakes, not an obstruction.  The
regression was indexed over the *triangular* faces of the simplex when the
plaquettes are indexed by its *codimension-two* faces -- for a tetrahedron, its
six edges, not its four triangles -- and the sibling term was missing besides.
`KURTOSIS_IDENTITY_MAX_DIMENSION` and `identity_fails_above_dimension_two` are
kept, the latter now returning ``False``, so the retraction stays visible in the
API instead of vanishing from it.

**Naming.**  The right-hand side is a **Wilson plaquette action**, not a squared
curvature.  ``|1 - omega|^2`` and ``2(1 - Re omega)`` are identical for unitary
holonomy so nothing is approximated between them, but ``|F|^2`` in the
differential-geometric sense is recovered from the Wilson action only in the
continuum, small-flux limit.

Two structural precedents.  Kenyon (Ann. Probab. 39, 2011) weights cycle-rooted
spanning forests by ``2 - tr(hol)``, which for ``U(1)`` is exactly the summand
here -- the same quantity in a determinant identity rather than a moment one.
And Chamseddine-Connes (hep-th/9606001) show the fourth heat-expansion
coefficient of ``Tr F(D/Lambda)`` contains the Yang-Mills action; this is a
discrete and *localised* analogue, localised being the operative word, since the
existing discrete work (Najem-Mrad-Elsayed, arXiv:2509.04311) takes global
traces.

**Five -- the filtration invariant.**  Sum the fourth-moment law over an entire
filtration.  Individually the terms move with the insertion order: a simplex
entering early sees fewer siblings and a smaller subcomplex.  The totals do not.

    sum_tau M_4(tau)  =  B(K)  +  P(K)  +  W(K)

with ``B`` the sum of squared facet counts, ``P`` the number of facet-sharing
pairs -- each counted once, when the second of the pair arrives -- and ``W`` the
**total Wilson action of the complex**.  All three are independent of which
linear extension of the face poset is used.  Verified across six random orders
per complex, with spread ``0`` to ``2.8e-14`` and residual ``0`` to ``7e-15``.

Rearranged, that is a recovery statement:

    W(K)  =  sum_tau M_4(tau)  -  B(K)  -  P(K)

**A global gauge-theoretic quantity, obtained from strictly local spectral data**
-- each moment computed on a subcomplex, at the moment one simplex entered, with
no global operator ever formed.  Chamseddine-Connes get the Yang-Mills action
from the fourth heat-expansion coefficient of a *global* trace
``Tr F(D/Lambda)``, and the existing discrete work in that direction
(Najem-Mrad-Elsayed, arXiv:2509.04311) also takes global traces.  This assembles
the same order of the same expansion from local, filtration-adapted pieces.

**A dimension-zero correction the sum surfaced.**  Statement two originally read
``M_2 = dim + 1`` for every simplex.  It is wrong for a vertex: a 0-simplex's
only facet is the empty face, which is not a simplex, so a closed two-walk has
nowhere to go and ``M_2 = 0``.  Every earlier check used ``dimension >= 1``, so
the case was untested until vertices became unavoidable in a filtration.  The
same oversight made `sibling_count` treat every other vertex as a sibling, since
``combinations(tau, 0)`` yields the empty tuple and the empty set is contained in
everything.  Both are fixed; the corrected statement is ``M_2 = number of facets
present``.

Novelty
-------

The definition is **not** new -- see the top of this docstring.  Statements one
through four appear to be unstated in this form, per a literature check run by
the repository owner (arxiv.org is unreachable from this environment).  The
walk-moment lemma underlying statement four is standard (Preciado-Jadbabaie,
arXiv:1107.5676, Lemma 2.1); what does not appear in the literature is the
three-term fourth-moment law, the identification of its plaquette index set with
the codimension-two faces, or the girth law.  Treat the novelty claim as
unverified and the arithmetic as exact.
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
    "hasse_squares",
    "wilson_sum",
    "sibling_count",
    "fourth_moment_law",
    "fourth_moment_law_residual",
    "fourth_moment_law_holds",
    "linear_extension",
    "FiltrationTotals",
    "filtration_totals",
    "total_wilson_action",
    "filtration_invariance_residual",
    "recovered_wilson_action",
    "wilson_action_is_order_independent",
    "TOLERANCE",
]

#: Comparisons are against this unless stated.  The identity of statement four
#: holds to 1.8e-15 in practice; this leaves room.
TOLERANCE: float = 1e-9

#: Retained only because an earlier version of this module reported a
#: dimension-two ceiling on the fourth-moment identity.  **There is no ceiling.**
#: `fourth_moment_law` holds in every dimension; the apparent failure came from
#: summing over triangular faces instead of codimension-two faces, and from
#: omitting the sibling term.  The constant is kept so the retraction stays
#: visible rather than being silently deleted.
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
    """The number of facets a ``k``-simplex has *present in the complex*.

    ``k + 1`` for ``k >= 1``, and **zero for a vertex** -- a 0-simplex's only
    facet is the empty face, which is not a simplex, so there is nowhere for a
    closed two-walk to go.  An earlier version returned ``dim + 1`` uniformly and
    was wrong at dimension zero; the case was untested because every check used
    ``dimension >= 1``.  It surfaced when the moments were summed over a whole
    filtration, where vertices are unavoidable.
    """
    if dimension < 0:
        raise ValueError(f"dimension must be non-negative, got {dimension}")
    return 0 if dimension == 0 else dimension + 1


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
    """**Retracted.**  Returns ``False``: the identity does not fail.

    An earlier version reported that the fourth-moment identity was
    dimension-two only, having regressed the excess kurtosis against the summed
    holonomy defects of a simplex's *triangular* faces and found residuals
    comparable to the signal.  That regression used the wrong index set.  The
    Hasse plaquettes at a simplex are indexed by its **codimension-two** faces --
    for a tetrahedron, its six edges, not its four triangles -- and a sibling
    term was missing besides.  With both corrected, `fourth_moment_law` is exact
    in every dimension tested.

    The function is kept, returning the corrected answer, so that the retraction
    is visible in the API rather than disappearing from it.
    """
    return False


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


# ---------------------------------------------------------------------------
# the fourth-moment law, in every dimension
# ---------------------------------------------------------------------------
#
# This supersedes the dimension-two-only statement.  The earlier ceiling was not
# an obstruction; it was two mistakes.  The sum was indexed over the *triangular
# faces* of the simplex, where it belongs on the *codimension-two* faces -- for a
# tetrahedron those are its six edges, not its four triangles -- and the sibling
# term below was missing entirely.


def hasse_squares(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    weight: Callable[[int, int], complex],
) -> dict[tuple[int, ...], complex]:
    """Holonomy of each Hasse plaquette at ``tau``, indexed by codim-2 face.

    A codimension-two face ``g`` of ``tau`` lies in exactly two codimension-one
    faces ``f_1, f_2``, so ``tau -> f_1 -> g -> f_2 -> tau`` is a canonical
    four-cycle in the Hasse diagram.  Its holonomy is the product of the four
    Dirac entries, normalised by the sign that makes it ``1`` at zero flux --
    the sign being exactly the ``d^2 = 0`` cancellation.

    Each value is gauge invariant: it is the holonomy of a closed loop, so a
    vertex gauge transformation leaves it fixed (checked to 1e-16 in the tests).
    """
    if len(simplex) < 3:
        raise ValueError(
            f"a codimension-two face needs dimension at least two, got "
            f"{len(simplex) - 1}"
        )
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    if simplex not in ordered:
        raise ValueError(f"{simplex} is not in the complex")
    operator = magnetic.general_dirac(ordered, weight)
    index = {face: position for position, face in enumerate(ordered)}
    result: dict[tuple[int, ...], complex] = {}
    for lower in combinations(simplex, len(simplex) - 2):
        pair = [
            face
            for face in combinations(simplex, len(simplex) - 1)
            if set(lower) <= set(face)
        ]
        if len(pair) != 2:
            raise ArithmeticError(
                f"codimension-two face {lower} of {simplex} lies in "
                f"{len(pair)} facets, expected two"
            )
        first, second = pair
        product = (
            operator[index[simplex], index[first]]
            * operator[index[first], index[lower]]
            * operator[index[lower], index[second]]
            * operator[index[second], index[simplex]]
        )
        result[lower] = -product
    return result


def wilson_sum(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    weight: Callable[[int, int], complex],
) -> float:
    """``sum_g |1 - omega_g|^2`` over the Hasse plaquettes at ``tau``.

    The local Wilson action.  Zero exactly when every plaquette at ``tau`` is
    flat, and gauge invariant term by term.
    """
    return float(
        sum(abs(1 - value) ** 2 for value in hasse_squares(simplices, simplex, weight).values())
    )


def sibling_count(
    simplices: Sequence[tuple[int, ...]], simplex: tuple[int, ...]
) -> int:
    """Pairs ``(sigma, f)`` with ``f`` a facet of ``tau`` shared by a same-dimension
    ``sigma != tau``.

    Purely combinatorial and flux-blind: the walk ``tau -> f -> sigma -> f ->
    tau`` contributes ``|<tau,f>|^2 |<sigma,f>|^2 = 1`` whatever the connection
    does, because each phase meets its own conjugate.
    """
    if simplex not in set(simplices):
        raise ValueError(f"{simplex} is not in the complex")
    size = len(simplex)
    if size == 1:
        # A vertex has no facet in the complex.  Without this guard
        # ``combinations(tau, 0)`` yields the empty tuple, which is a subset of
        # every simplex, and every other vertex is miscounted as a sibling.
        return 0
    total = 0
    for face in combinations(simplex, size - 1):
        total += sum(
            1
            for other in simplices
            if len(other) == size and other != simplex and set(face) <= set(other)
        )
    return total


def fourth_moment_law(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    weight: Callable[[int, int], complex],
) -> float:
    """``M_2^2 + S(tau) + sum_g |1 - omega_g|^2``: the predicted fourth moment.

    Three terms with disjoint meanings, and that separation is the result:

      * ``M_2^2`` -- the two-point Bernoulli baseline, the value when the
        insertion measure has not spread at all;
      * ``S(tau)`` -- the sibling count, combinatorial and flux-blind;
      * the Wilson sum -- the geometry, and the only term the connection reaches.

    So the fourth moment separates combinatorics from curvature exactly, in
    every dimension.  Verified to 1e-15 on lone simplices of dimension two
    through four, on fans of triangles, on a tetrahedron shell and on pairs of
    tetrahedra.
    """
    second = insertion_moment(simplices, simplex, weight, 2)
    return (
        second * second
        + sibling_count(simplices, simplex)
        + wilson_sum(simplices, simplex, weight)
    )


def fourth_moment_law_residual(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    weight: Callable[[int, int], complex],
) -> float:
    """``|M_4 - law|``. Zero is the theorem, in any dimension."""
    return abs(
        insertion_moment(simplices, simplex, weight, 4)
        - fourth_moment_law(simplices, simplex, weight)
    )


def fourth_moment_law_holds(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    weight: Callable[[int, int], complex],
) -> bool:
    """Does the three-term law reproduce the measured fourth moment?"""
    return fourth_moment_law_residual(simplices, simplex, weight) <= TOLERANCE


# ---------------------------------------------------------------------------
# the filtration invariant: a localised, order-free spectral action
# ---------------------------------------------------------------------------
#
# Summing the fourth-moment law over a whole filtration.  Each individual term
# depends on the insertion order -- a simplex entering early sees fewer siblings
# and a smaller subcomplex -- but the totals do not, and the geometric part of
# the total is exactly the Wilson action of the whole complex.


def linear_extension(
    simplices: Sequence[tuple[int, ...]], seed: int = 0
) -> tuple[tuple[int, ...], ...]:
    """A random valid insertion order: every face before its cofaces.

    Any linear extension of the face poset is a filtration order, and the
    theorem below says the totals do not depend on which one is chosen.
    """
    generator = np.random.default_rng(seed)
    remaining = list(simplices)
    placed: set[tuple[int, ...]] = set()
    order: list[tuple[int, ...]] = []
    while remaining:
        ready = [
            candidate
            for candidate in remaining
            if all(
                face in placed
                for size in range(1, len(candidate))
                for face in combinations(candidate, size)
            )
        ]
        if not ready:
            raise ValueError("complex is not closed under faces")
        chosen = ready[int(generator.integers(len(ready)))]
        order.append(chosen)
        placed.add(chosen)
        remaining.remove(chosen)
    return tuple(order)


@dataclass(frozen=True)
class FiltrationTotals:
    """The three order-independent totals of a filtration.

    Attributes
    ----------
    fourth_moments:
        ``sum_tau M_4(tau)``, each taken at the moment ``tau`` enters.
    baseline:
        ``sum_tau (facet count)^2`` -- combinatorial.
    siblings:
        ``sum_tau S(tau)``, counting each facet-sharing pair once, at the
        moment the second of the pair arrives -- combinatorial.
    wilson:
        ``sum_tau W(tau)`` -- the total Wilson action of the complex.
    """

    fourth_moments: float
    baseline: int
    siblings: int
    wilson: float


def filtration_totals(
    simplices: Sequence[tuple[int, ...]],
    weight: Callable[[int, int], complex],
    seed: int = 0,
) -> FiltrationTotals:
    """Accumulate the fourth-moment law along a random filtration order."""
    order = linear_extension(simplices, seed)
    fourth = 0.0
    baseline_total = 0
    sibling_total = 0
    wilson_total = 0.0
    for position, simplex in enumerate(order):
        prefix = order[: position + 1]
        fourth += insertion_moment(prefix, simplex, weight, 4)
        baseline_total += second_moment(len(simplex) - 1) ** 2
        sibling_total += sibling_count(prefix, simplex)
        if len(simplex) >= 3:
            wilson_total += wilson_sum(prefix, simplex, weight)
    return FiltrationTotals(
        fourth_moments=fourth,
        baseline=baseline_total,
        siblings=sibling_total,
        wilson=wilson_total,
    )


def total_wilson_action(
    simplices: Sequence[tuple[int, ...]],
    weight: Callable[[int, int], complex],
) -> float:
    """The Wilson action of the whole complex, computed directly.

    Summed over every Hasse plaquette of every simplex, with no filtration
    involved.  The theorem is that `filtration_totals` recovers this from local
    insertion moments alone.
    """
    return float(
        sum(
            wilson_sum(simplices, simplex, weight)
            for simplex in simplices
            if len(simplex) >= 3
        )
    )


def filtration_invariance_residual(
    simplices: Sequence[tuple[int, ...]],
    weight: Callable[[int, int], complex],
    seed: int = 0,
) -> float:
    """``|sum M_4 - (baseline + siblings + wilson)|``. Zero is the theorem."""
    totals = filtration_totals(simplices, weight, seed)
    return abs(
        totals.fourth_moments
        - (totals.baseline + totals.siblings + totals.wilson)
    )


def recovered_wilson_action(
    simplices: Sequence[tuple[int, ...]],
    weight: Callable[[int, int], complex],
    seed: int = 0,
) -> float:
    """The Wilson action read off a filtration's local spectral data.

        W(K)  =  sum_tau M_4(tau)  -  baseline(K)  -  siblings(K)

    A global gauge-theoretic quantity obtained from purely local moments, each
    computed on a subcomplex at the moment one simplex entered.  Neither the
    moments nor the sibling counts are order-independent individually; their
    totals are.

    This is what the module has been building toward.  Chamseddine-Connes
    obtain the Yang-Mills action from the fourth heat-expansion coefficient of a
    *global* trace ``Tr F(D/Lambda)``, and the existing discrete work in that
    direction (Najem-Mrad-Elsayed, arXiv:2509.04311) also takes global traces.
    Here the same order of the same expansion is assembled from strictly local,
    filtration-adapted data.
    """
    totals = filtration_totals(simplices, weight, seed)
    return totals.fourth_moments - totals.baseline - totals.siblings


def wilson_action_is_order_independent(
    simplices: Sequence[tuple[int, ...]],
    weight: Callable[[int, int], complex],
    orders: int = 5,
) -> bool:
    """Does every filtration order recover the same Wilson action?

    The claim that makes the recovery meaningful.  Individually the local terms
    move with the order; the total does not.
    """
    if orders < 2:
        raise ValueError(f"need at least two orders to compare, got {orders}")
    direct = total_wilson_action(simplices, weight)
    return all(
        abs(recovered_wilson_action(simplices, weight, seed) - direct) <= TOLERANCE
        for seed in range(orders)
    )
