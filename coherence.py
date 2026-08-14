"""An explicit mode architecture whose entire triad list is the Obukhov graph.

The coherence problem
---------------------
:mod:`embedding` settles the coefficient question for *one* gate: the Euler
nonlinearity does supply the ``|k_low|`` amplification the Obukhov model asks
for, at efficiency ``1/2``, attained when the low mode sits at ``45 degrees`` to
the high pair.  What it does not address is the reason Step 2 of Tao's program is
hard: a shell model prescribes a very sparse *interaction graph* -- shell ``j``
talks to ``j-1`` and ``j+1`` and to nothing else -- while the true nonlinearity
couples every triple of modes that closes.  An embedding must therefore suppress

* **local triads** ``(j, j, j)``, which drive the turbulent cascade that makes the
  Katz--Pavlovic and exponential-shell Obukhov models globally regular, and
* **long-range triads** ``(i, j, j)`` with ``i <= j-2``, which are pure error.

This module builds mode sets for which those are not suppressed but *absent*, and
verifies it by exhaustive enumeration in exact integer arithmetic.

The reduction
-------------
Reality forces ``S_j = -S_j``, so ``S_j + S_j = S_j - S_j =: D_j`` and the three
requirements collapse to statements about one difference set:

    local triads absent        <=>   D_j n S_j = {}
    the gate exists            <=>   D_j n S_{j-1} != {}
    long-range triads absent   <=>   D_j n S_i = {}   for i <= j-2

The middle line also supplies the model's *other* interaction for free.  A triad
``k_{j-1} + k_j + k'_j = 0`` is simultaneously the amplification of the high pair
by the low mode and the drain of the low mode by the high pair -- Palasek's two
Obukhov terms are the two ends of one triad, which is why they conserve energy
against each other.  There is nothing separate to arrange.

The architecture
----------------
Two directions per shell suffice.  Take ``d`` and ``e`` integer, orthogonal, of
equal length, and set

    p = A (d + e) ,    q = p - d ,    (d, e) <- (p, A(e - d))

The recursion is closed: ``(d+e)`` and ``(e-d)`` are again orthogonal and of equal
length, because ``(d+e).(e-d) = |e|^2 - |d|^2 = 0``.  Then ``S_j = {+-p, +-q}``,
and by construction

* ``p - q = d`` is a mode of shell ``j-1`` -- the gate;
* ``angle(p, d) = 45 degrees`` exactly, since ``p.d = A|d|^2`` and
  ``|p| = A|d| sqrt(2)`` -- the optimal efficiency of :mod:`embedding`.  The
  *other* high mode ``q = p - d`` is not at ``45`` exactly; it misses by
  ``O(N_{j-1}/N_j)``, which is ``0.004`` degrees at a separation ratio of
  ``10^4`` (:func:`gate_angle_deviation`).  Since the model needs
  super-exponential separation anyway, that gap closes faster than any
  requirement on it;
* every other element of ``D_j`` has length of order ``N_j``, so it cannot meet
  any lower shell.

For the four-shell instance built by :func:`architecture` from ``d = (3,4,0)``,
``e = (0,0,5)``, exhaustive search over all fourteen modes finds **six** closing
triads, and all six are nearest-neighbour gates: no local triad, no long-range
triad (:func:`triad_census`).  The interaction graph is exactly the Obukhov graph.

How much room is left
---------------------
Two modes per shell is a space-filling field, ``alpha = 1``, the wrong end of the
intermittency range.  Widening a shell to many modes is what ``alpha > 1``
requires, and it reintroduces local triads once the shell is wide enough in
angle.  The threshold is sharp and is **not** the obvious one:

For ``|a| = |b| = N``, the sum ``a + b`` returns to the shell exactly when the
angle between them is ``120 degrees``.  Confining a shell to a spherical cap of
half-angle ``phi`` allows pairwise angles in ``[0, 2 phi]``, which suggests
``phi < 60``.  That is wrong: reality adds the antipodal cap, and pairs drawn
from opposite caps span ``[180 - 2 phi, 180]``.  That interval reaches ``120``
as soon as ``phi >= 30``.  So

    **a shell inside a cap of half-angle less than 30 degrees carries no local
    triads, and 30 degrees is sharp** (:data:`SUM_FREE_CAP_LIMIT_DEGREES`).

The reasoning that gives ``60`` is this module's first answer, and it is wrong for
a reason worth keeping: it forgets that a real velocity field cannot populate a
cap without populating its antipode.

The budget that follows is tight in an interesting way.  Concentrating a field
onto a volume fraction ``mu`` needs of order ``1/mu`` Fourier modes, and Palasek's
intermittency dictionary sets ``mu_k = N_k^{-2(alpha-1)}``, so a shell needs

    M_k  ~  N_k^{2(alpha - 1)}

modes.  A dyadic shell holds of order ``N_k^3`` lattice points and a ``30``-degree
cap keeps a fixed fraction of them, so the construction has room exactly when
``2(alpha - 1) < 3``, that is ``alpha < 5/2`` -- and saturates at ``alpha = 5/2``
(:func:`budget_fits`).  That is the top of the three-dimensional intermittency
range, which the model gets from the uncertainty principle instead.  Two unrelated
routes to the same endpoint.

What this settles, and what it does not
---------------------------------------
Settled: the interaction graph.  There is an explicit, exactly verified family of
integer mode sets whose complete list of closing triads is the Obukhov graph and
nothing else, with every gate at the optimal angle up to ``O(N_{j-1}/N_j)``, and
a sharp rule for how wide a shell may be before local triads return.

Not settled: everything analytic.  Which triads exist is not the same as what the
amplitudes do.  Time dependence, the Leray projection acting on products of
non-monochromatic fields, the errors from a solution not being a finite set of
exact modes, and whether the resulting finite system reproduces the Obukhov
coefficients with the right signs throughout the evolution -- none of that is
here.  This is the combinatorial half of the coherence problem, and the analytic
half is the one that is hard.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, Sequence

__all__ = [
    "Vector",
    "Architecture",
    "orthogonal_equal_length",
    "architecture",
    "closing_triads",
    "classify_triad",
    "triad_census",
    "gate_angles",
    "gate_angle_deviation",
    "difference_set",
    "SUM_FREE_CAP_LIMIT_DEGREES",
    "cap_admits_local_triad",
    "cap_is_sum_free",
    "mode_budget_exponent",
    "cap_capacity_exponent",
    "budget_fits",
    "saturating_alpha",
]

LOGGER = logging.getLogger(__name__)

Vector = tuple[int, int, int]

#: Half-angle, in degrees, below which a spherical cap and its antipode carry no
#: local triads.  Sharp: at exactly ``30`` the antipodal pairs reach the
#: ``120``-degree opening that returns a sum to its own shell.
SUM_FREE_CAP_LIMIT_DEGREES = 30

#: The angle a sum must open to return to its own shell, for two modes of equal
#: length: ``|a + b| = |a|`` forces ``cos = -1/2``.
LOCAL_TRIAD_ANGLE_DEGREES = 120


def _dot(a: Sequence[int], b: Sequence[int]) -> int:
    return sum(x * y for x, y in zip(a, b))


def _add(a: Sequence[int], b: Sequence[int]) -> Vector:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def _negate(a: Sequence[int]) -> Vector:
    return tuple(-x for x in a)  # type: ignore[return-value]


def orthogonal_equal_length(d: Sequence[int], e: Sequence[int]) -> tuple[Vector, Vector]:
    """The next orthogonal equal-length pair, ``(d + e, e - d)``.

    Closed form, which is what makes the architecture recursive: no search for a
    perpendicular partner of the same length is needed at any stage, because
    ``(d+e).(e-d) = |e|^2 - |d|^2 = 0`` and ``|d+e|^2 = |e-d|^2 = |d|^2 + |e|^2``
    whenever ``d`` and ``e`` are themselves orthogonal and of equal length.
    """
    if _dot(d, e) != 0:
        raise ValueError("the seed pair must be orthogonal")
    if _dot(d, d) != _dot(e, e):
        raise ValueError("the seed pair must have equal length")
    return _add(d, e), tuple(y - x for x, y in zip(d, e))  # type: ignore[return-value]


@dataclass(frozen=True)
class Architecture:
    """Mode sets ``S_0, ..., S_n``, each closed under negation."""

    shells: tuple[tuple[Vector, ...], ...]

    def __post_init__(self) -> None:
        for index, shell in enumerate(self.shells):
            for mode in shell:
                if _negate(mode) not in shell:
                    raise ValueError(
                        f"shell {index} is not closed under negation, as reality requires"
                    )

    def modes(self) -> Iterator[tuple[int, Vector]]:
        for index, shell in enumerate(self.shells):
            for mode in shell:
                yield index, mode

    def radius(self, index: int) -> float:
        return math.sqrt(_dot(self.shells[index][0], self.shells[index][0]))


def architecture(
    seed: Sequence[int], partner: Sequence[int], factors: Sequence[int]
) -> Architecture:
    """Build the shells from a seed pair and one growth factor per step.

    ``factors[j]`` sets ``N_{j+1} / N_j = factors[j] * sqrt(2)``, so a
    super-exponentially growing sequence of factors gives super-exponentially
    separated shells -- the regime in which Palasek's model blows up.
    """
    d: Vector = tuple(int(x) for x in seed)  # type: ignore[assignment]
    e: Vector = tuple(int(x) for x in partner)  # type: ignore[assignment]
    if _dot(d, e) != 0 or _dot(d, d) != _dot(e, e):
        raise ValueError("the seed pair must be orthogonal and of equal length")
    shells: list[tuple[Vector, ...]] = [(d, _negate(d))]
    for factor in factors:
        if factor < 1:
            raise ValueError("growth factors must be at least one")
        combined, difference = orthogonal_equal_length(d, e)
        p: Vector = tuple(factor * x for x in combined)  # type: ignore[assignment]
        q: Vector = tuple(x - y for x, y in zip(p, d))  # type: ignore[assignment]
        shells.append((p, _negate(p), q, _negate(q)))
        d, e = p, tuple(factor * x for x in difference)  # type: ignore[assignment]
    return Architecture(tuple(shells))


def difference_set(shell: Sequence[Vector]) -> set[Vector]:
    """``D = S - S``, which because ``S = -S`` is also ``S + S``."""
    return {_add(a, b) for a in shell for b in shell}


def closing_triads(arch: Architecture) -> list[tuple[tuple[int, Vector], ...]]:
    """Every unordered triple of modes summing to zero, found by exhaustion.

    No structure is assumed and nothing is filtered: this is the complete
    interaction list of the architecture under the Euler nonlinearity, so it is
    what the claim has to be made against.
    """
    modes = list(arch.modes())
    found = set()
    for first, second, third in itertools.combinations(modes, 3):
        if _add(_add(first[1], second[1]), third[1]) == (0, 0, 0):
            found.add(tuple(sorted((first, second, third))))
    return sorted(found)


def classify_triad(triad: Sequence[tuple[int, Vector]]) -> str:
    """``"gate"``, ``"local"`` or ``"long-range"``.

    A gate is the wanted ``(j-1, j, j)``; ``local`` is ``(j, j, j)``, the
    turbulent cascade; ``long-range`` is anything else, and is pure error.
    """
    indices = sorted(index for index, _ in triad)
    if indices[0] == indices[1] == indices[2]:
        return "local"
    if indices[1] == indices[2] and indices[0] == indices[1] - 1:
        return "gate"
    return "long-range"


def triad_census(arch: Architecture) -> dict[str, int]:
    """How many triads of each kind the architecture actually contains."""
    census = {"gate": 0, "local": 0, "long-range": 0}
    for triad in closing_triads(arch):
        census[classify_triad(triad)] += 1
    return census


def gate_angles(arch: Architecture) -> list[float]:
    """Angle in degrees between the low mode and a high mode, for every gate.

    Two per gate, one for each high mode.  One is exactly ``45``, the maximiser
    of :func:`embedding.gate_efficiency_limit`; the other misses by
    ``O(N_{j-1}/N_j)``, since the two high modes differ by the low one and so
    cannot both be at the same angle to it at finite separation.
    """
    angles = []
    for triad in closing_triads(arch):
        if classify_triad(triad) != "gate":
            continue
        low = min(triad, key=lambda entry: entry[0])[1]
        high = max(triad, key=lambda entry: entry[0])[1]
        cosine = abs(_dot(low, high)) / math.sqrt(_dot(low, low) * _dot(high, high))
        angles.append(math.degrees(math.acos(min(1.0, cosine))))
    return angles


# --------------------------------------------------------------------------- #
# How wide a shell may be
# --------------------------------------------------------------------------- #
def cap_admits_local_triad(half_angle_degrees: float) -> bool:
    """Whether a cap of this half-angle, with its antipode, can close a local triad.

    Two modes of equal length sum back into their own shell exactly at a
    ``120``-degree opening.  A cap alone spans ``[0, 2 phi]``; reality adds the
    antipodal cap, contributing ``[180 - 2 phi, 180]``.  The second interval is
    what matters, and it reaches ``120`` at ``phi = 30`` -- not at ``60``, which
    is what the first interval alone would say.
    """
    if not 0 < half_angle_degrees <= 90:
        raise ValueError("a cap half-angle lies in (0, 90] degrees")
    within = 0 <= LOCAL_TRIAD_ANGLE_DEGREES <= 2 * half_angle_degrees
    across = 180 - 2 * half_angle_degrees <= LOCAL_TRIAD_ANGLE_DEGREES <= 180
    return within or across


def cap_is_sum_free(half_angle_degrees: float) -> bool:
    """Whether a shell confined to this cap carries no local triads."""
    return not cap_admits_local_triad(half_angle_degrees)


def mode_budget_exponent(alpha: Fraction | int) -> Fraction:
    """``2(alpha - 1)``: the exponent of ``N_k`` in the modes a shell needs.

    Concentrating a field onto a volume fraction ``mu`` costs of order ``1/mu``
    Fourier modes, and the intermittency dictionary sets
    ``mu_k = N_k^{-2(alpha-1)}``.
    """
    return 2 * (Fraction(alpha) - 1)


def cap_capacity_exponent() -> int:
    """``3``: a dyadic shell holds of order ``N_k^3`` lattice points.

    A cap of fixed half-angle keeps a fixed fraction of them, so the constraint
    is on the exponent alone.
    """
    return 3


def budget_fits(alpha: Fraction | int) -> bool:
    """Whether a shell can hold the modes intermittency ``alpha`` demands."""
    return mode_budget_exponent(alpha) < cap_capacity_exponent()


def saturating_alpha() -> Fraction:
    """The ``alpha`` at which the mode budget exactly fills a shell.

    ``2(alpha - 1) = 3`` gives ``alpha = 5/2`` -- the top of the
    three-dimensional intermittency range, which the model derives instead from
    the uncertainty principle.  Two unrelated routes to one endpoint.
    """
    return Fraction(5, 2)


def gate_angle_deviation(arch: Architecture) -> float:
    """Largest departure from ``45`` degrees over all gates, in degrees.

    Zero would be impossible at finite separation: the two high modes of a gate
    differ by the low mode, so they cannot both meet it at the same angle.  What
    is true is that the deviation is ``O(N_{j-1}/N_j)`` and therefore vanishes in
    the separated limit the model requires -- from ``4.4`` degrees at a ratio of
    ``10`` to ``0.004`` at ``10^4``.
    """
    angles = gate_angles(arch)
    if not angles:
        raise ValueError("this architecture has no gates")
    return max(abs(angle - 45.0) for angle in angles)
