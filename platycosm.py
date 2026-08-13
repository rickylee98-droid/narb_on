"""Linearisation instability on the compact flat 3-manifolds, not only the torus.

The question
------------
Fischer, Marsden and Moncrief: a spacetime with a compact Cauchy surface and a
Killing field is *linearisation unstable*.  A solution of the linearised Einstein
constraints need not be tangent to any curve of real solutions; it must in
addition satisfy one integral condition per element of the kernel of the adjoint
constraint operator -- per Killing initial datum, or KID.

Moncrief worked this out in detail for flat spacetime with **toroidal** spatial
sections, and showed quantum mechanically that physical states must be invariant
under the symmetries the conditions generate.  Group averaging reproduces it.
The three-torus is, however, only one of the six orientable compact flat
three-manifolds -- the *platycosms* -- and the others carry strictly less
symmetry.  This module computes the structure on all of them.

Why it is not merely bookkeeping
--------------------------------
The usual statement of the phenomenon reads "symmetry causes instability", and
the quantum version reads "states must be symmetry-invariant".  Both suggest the
effect should weaken as symmetry is removed.  The platycosms let that be tested,
because the holonomy progressively kills the translational Killing fields:

* the torus ``G1`` has all three translations,
* ``G2``--``G5`` retain exactly one, the rotation axis,
* the Hantzsche--Wendt manifold ``G6`` has **none at all** -- its holonomy is
  ``Z_2 x Z_2`` acting with no non-zero fixed vector, and its first Betti number
  is zero.

Yet the constant lapse is a KID on *every* flat compact slice, since
``Hess N - (Delta N) delta`` vanishes for constant ``N``.  So ``G6`` has a
one-dimensional KID space with no spatial symmetry whatsoever, and the
instability persists there with nothing for a symmetry group to average over.

The computation
---------------
Every platycosm is ``R^3 / Gamma`` for a Bieberbach group ``Gamma``, whose
elements are pairs ``(A, a)`` with ``A`` in a finite point group and ``a`` a
translation.  Fields on the quotient are exactly the ``Gamma``-equivariant
fields upstairs, so the whole analysis reduces to the ``Gamma``-invariant
subspace of the torus computation already in :mod:`adm` -- which is what makes
``G1`` a referee rather than a separate case.

Two things are computed for each platycosm:

* the KID dimension, ``1 + dim Fix(point group)``, the lapse contributing the
  one;
* the inertia of the second-order obstruction restricted to the invariant
  linearised solutions, which decides rigidity.

The second is the point.  On the torus :mod:`adm` finds inertia ``(4,4,0)``:
negative definite modulo gauge, so no non-gauge linearised solution integrates.
Restricting a negative semi-definite form to a subspace keeps it negative
semi-definite, so rigidity descends to every platycosm -- including ``G6``, where
a *single* condition does the work that four do on the torus.  The three
translational KIDs of the torus are, for rigidity, redundant.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, Sequence

import adm

__all__ = [
    "PLATYCOSMS",
    "Platycosm",
    "point_group",
    "fixed_subspace_dimension",
    "kid_dimension",
    "is_bieberbach",
    "invariant_modes",
    "mode_orbit",
    "PlatycosmReport",
    "analyse",
]

LOGGER = logging.getLogger(__name__)

IntMatrix = tuple[tuple[int, ...], ...]

_IDENTITY: IntMatrix = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
#: Half-turn about the z-axis.
_HALF_Z: IntMatrix = ((-1, 0, 0), (0, -1, 0), (0, 0, 1))
#: Half-turn about the x-axis.
_HALF_X: IntMatrix = ((1, 0, 0), (0, -1, 0), (0, 0, -1))
#: Half-turn about the y-axis.
_HALF_Y: IntMatrix = ((-1, 0, 0), (0, 1, 0), (0, 0, -1))
#: Quarter-turn about the z-axis.
_QUARTER_Z: IntMatrix = ((0, -1, 0), (1, 0, 0), (0, 0, 1))


@dataclass(frozen=True)
class Platycosm:
    """A compact flat orientable 3-manifold, as generators over a cubic lattice.

    ``generators`` are the point-group parts of the non-translational Bieberbach
    generators; ``shifts`` are their translation parts as fractions of the
    lattice.  The translation subgroup is the standard cubic lattice throughout,
    which is why the two platycosms requiring a hexagonal lattice are excluded
    here -- see :data:`PLATYCOSMS`.
    """

    name: str
    label: str
    generators: tuple[IntMatrix, ...]
    shifts: tuple[tuple[Fraction, Fraction, Fraction], ...]
    betti_one: int

    def __post_init__(self) -> None:
        if len(self.generators) != len(self.shifts):
            raise ValueError(
                f"{self.label}: {len(self.generators)} generators against "
                f"{len(self.shifts)} shifts"
            )


#: The orientable platycosms realisable on a cubic lattice.  G3 (tricosm) and G5
#: (hexacosm) need a hexagonal lattice and are deliberately omitted: their point
#: groups have order 3 and 6 and do not act on Z^3 by integer matrices, so the
#: exact-integer mode analysis used here does not apply to them unchanged.  Their
#: KID dimension is nonetheless 2, by the same fixed-subspace argument, since a
#: rotation of any order about an axis fixes exactly that axis.
PLATYCOSMS: tuple[Platycosm, ...] = (
    Platycosm(
        name="3-torus",
        label="G1",
        generators=(),
        shifts=(),
        betti_one=3,
    ),
    Platycosm(
        name="half-turn space (dicosm)",
        label="G2",
        generators=(_HALF_Z,),
        shifts=((Fraction(0), Fraction(0), Fraction(1, 2)),),
        betti_one=1,
    ),
    Platycosm(
        name="quarter-turn space (tetracosm)",
        label="G4",
        generators=(_QUARTER_Z,),
        shifts=((Fraction(0), Fraction(0), Fraction(1, 4)),),
        betti_one=1,
    ),
    Platycosm(
        name="Hantzsche-Wendt (didicosm)",
        label="G6",
        # Each generator must be a *screw* about its own invariant axis: the
        # translation part needs a non-zero component along the axis the
        # rotation fixes, or the element has honest fixed points and the
        # quotient is an orbifold rather than a manifold.  A presentation with
        # zero x-component on the half-turn about x was tried first and
        # is_bieberbach rejected it, correctly.
        generators=(_HALF_X, _HALF_Y),
        shifts=(
            (Fraction(1, 2), Fraction(1, 2), Fraction(0)),
            (Fraction(0), Fraction(1, 2), Fraction(1, 2)),
        ),
        betti_one=0,
    ),
)


# --------------------------------------------------------------------------- #
# Point groups
# --------------------------------------------------------------------------- #
def _multiply(left: IntMatrix, right: IntMatrix) -> IntMatrix:
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def point_group(space: Platycosm) -> tuple[IntMatrix, ...]:
    """The finite group generated by the rotational parts, closed exactly.

    Closure is computed rather than assumed, so a mis-stated generator shows up
    as a group of the wrong order instead of silently truncating the orbit.
    """
    elements = {_IDENTITY}
    frontier = [_IDENTITY]
    while frontier:
        current = frontier.pop()
        for generator in space.generators:
            product = _multiply(current, generator)
            if product not in elements:
                elements.add(product)
                frontier.append(product)
        if len(elements) > 48:
            raise ValueError(
                f"{space.label}: point group exceeded 48 elements, so the "
                "generators do not generate a crystallographic point group"
            )
    return tuple(sorted(elements))


def fixed_subspace_dimension(space: Platycosm) -> int:
    """``dim { v : Av = v for every A in the point group }``, exactly.

    Computed as ``3 - rank(stack of (A - I))`` over the rationals; these are the
    parallel vector fields on the quotient, hence its Killing fields.
    """
    rows: list[list[int]] = []
    for matrix in point_group(space):
        for i in range(3):
            rows.append(
                [matrix[i][j] - (1 if i == j else 0) for j in range(3)]
            )
    if not rows:
        return 3
    return 3 - adm.integer_rank(rows)


def kid_dimension(space: Platycosm) -> int:
    """Killing initial data on the flat, time-symmetric slice: lapse plus Killings.

    The constant lapse is a KID on every compact flat slice, because
    ``Hess N - (Delta N) delta`` vanishes identically for constant ``N``.  The
    remaining KIDs are the parallel vector fields, i.e. the point-group-invariant
    vectors.  So the count is ``1 + dim Fix``, and it is $1$ exactly when the
    holonomy fixes no direction.
    """
    return 1 + fixed_subspace_dimension(space)


def is_bieberbach(space: Platycosm, *, span: int = 2) -> bool:
    """Check the defining property: no non-identity element has a fixed point.

    A Bieberbach group acts freely, which is what makes the quotient a manifold
    rather than an orbifold.  For ``(A, a)`` with ``A != I`` a fixed point exists
    modulo the lattice exactly when ``a`` lies in the image of ``A - I`` over the
    relevant lattice translates, so this searches those translates directly.
    """
    group = point_group(space)
    shift_of = {_IDENTITY: (Fraction(0), Fraction(0), Fraction(0))}
    frontier = [(_IDENTITY, shift_of[_IDENTITY])]
    while frontier:
        matrix, offset = frontier.pop()
        for generator, shift in zip(space.generators, space.shifts):
            product = _multiply(matrix, generator)
            moved = tuple(
                sum(matrix[i][j] * shift[j] for j in range(3)) + offset[i]
                for i in range(3)
            )
            if product not in shift_of:
                shift_of[product] = moved
                frontier.append((product, moved))
    for matrix, offset in shift_of.items():
        if matrix == _IDENTITY:
            continue
        difference = [
            [matrix[i][j] - (1 if i == j else 0) for j in range(3)]
            for i in range(3)
        ]
        for lattice in itertools.product(range(-span, span + 1), repeat=3):
            target = [offset[i] + lattice[i] for i in range(3)]
            if _solvable(difference, target):
                return False
    return True


def _solvable(matrix: Sequence[Sequence[int]], target: Sequence[Fraction]) -> bool:
    """Whether ``matrix v = target`` has a rational solution."""
    augmented = [
        [Fraction(matrix[i][j]) for j in range(3)] + [Fraction(target[i])]
        for i in range(3)
    ]
    plain = [row[:3] for row in augmented]
    return adm.integer_rank(
        [[int(x * 12) for x in row] for row in plain]
    ) == adm.integer_rank([[int(x * 12) for x in row] for row in augmented])


# --------------------------------------------------------------------------- #
# Invariant modes
# --------------------------------------------------------------------------- #
def mode_orbit(space: Platycosm, wave: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Orbit of a Fourier mode under the point group acting on wave vectors."""
    group = point_group(space)
    orbit = set()
    for matrix in group:
        orbit.add(
            tuple(sum(matrix[j][i] * wave[j] for j in range(3)) for i in range(3))
        )
    return tuple(sorted(orbit))


def invariant_modes(space: Platycosm, limit: int) -> Iterator[tuple[int, ...]]:
    """One representative per point-group orbit of modes in ``[-limit, limit]^3``.

    Fields on the quotient are the equivariant fields upstairs, so the mode
    content is organised by orbits rather than by individual wave vectors.
    """
    seen: set[tuple[int, ...]] = set()
    span = range(-limit, limit + 1)
    for wave in itertools.product(span, span, span):
        if wave in seen:
            continue
        orbit = mode_orbit(space, wave)
        seen.update(orbit)
        yield min(orbit)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlatycosmReport:
    """Symmetry and obstruction data for one compact flat 3-manifold."""

    label: str
    name: str
    point_group_order: int
    killing_fields: int
    kid_dimension: int
    betti_one: int
    free_action: bool
    torus_inertia: tuple[int, int, int]

    @property
    def has_no_spatial_symmetry(self) -> bool:
        return self.killing_fields == 0

    @property
    def rigid(self) -> bool:
        """Whether every non-gauge *inhomogeneous* linearised solution is obstructed.

        The obstruction form on the covering torus is negative semi-definite at
        each ``k != 0`` with radical exactly the gauge span, and restricting such
        a form to the invariant subspace keeps it negative semi-definite.  A
        single condition therefore already forces any solution satisfying it
        mode by mode to be pure gauge -- so this rigidity holds as soon as there
        is at least one KID, which there always is.

        **The restriction to one mode at a time is essential.**  The stability
        condition is a single integral over the whole slice, and the homogeneous
        mode is not negative: its inertia is ``(5, 0, 1)``, the positive
        direction being isotropic expansion.  Summed over all modes the
        condition is not rigidity but a balance, and :mod:`graviton` shows it is
        the Friedmann constraint with the wave energy as its source.  Read
        globally, a linearised solution with gravitational waves is obstructed
        only if it lacks the matching expansion.
        """
        negative, _, positive = self.torus_inertia
        return positive == 0 and negative > 0 and self.kid_dimension >= 1


def analyse(space: Platycosm, *, probe: Sequence[int] = (1, 1, 1)) -> PlatycosmReport:
    """Compute the symmetry and obstruction data for one platycosm."""
    group = point_group(space)
    return PlatycosmReport(
        label=space.label,
        name=space.name,
        point_group_order=len(group),
        killing_fields=fixed_subspace_dimension(space),
        kid_dimension=kid_dimension(space),
        betti_one=space.betti_one,
        free_action=is_bieberbach(space),
        torus_inertia=adm.physical_signature(probe),
    )
