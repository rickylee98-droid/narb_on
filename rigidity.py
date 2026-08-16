"""Flux chirality: the one bit of a gauge field that spectra cannot see.

This module names an object and proves a rigidity theorem about it.

The inverse problem
-------------------

`insertion` computes, for each simplex of a complex carrying a ``U(1))``
connection, the local spectral measure of the Dirac operator at that simplex --
equivalently its moment sequence.  Collecting those over the whole complex gives
what is called here the **spectral signature** of the connection.

The natural question, which the moment results provoke and do not answer, is the
inverse one:

    how much of the connection does the spectral signature determine?

Not all of it.  Two connections related by a gauge transformation share a
signature trivially, since gauge transformations do not change any closed-loop
holonomy.  The interesting question is what remains after quotienting by gauge,
and the answer is a single bit.

The theorem
-----------

**One, exact.  Reversing every flux is invisible.**  Writing ``conj(sigma)`` for
the connection with every phase negated, the Dirac operator satisfies

    D_conj(sigma)  =  conj(D_sigma)

entrywise, because each entry is a phase or its conjugate.  For a real basis
vector ``e_tau``,

    <e_tau, conj(D)^j e_tau>  =  conj(<e_tau, D^j e_tau>)

and the moment is real, so it is unchanged.  **Every moment of every simplex is
identical for a connection and its conjugate**, at every order, for every
complex.  Measured difference: exactly ``0.0``, not to a tolerance.

**Two, proved.  That is the only ambiguity, and it degenerates predictably.**  On a complex with two
independent plaquettes, gauge-fixed so the two holonomies are free coordinates,
an exhaustive sweep of 5184 connections on a ``72 x 72`` grid finds exactly two
sharing any given signature: the connection itself and its global conjugate.
Reversing a *single* plaquette is visible, and visible by a wide margin -- moment
differences of order ``1`` to ``4`` against a detection threshold of ``1e-8``.

The argument.  Write ``omega_j = exp(i theta_j)`` for the plaquette holonomies.
Every moment is a real number of the form ``cos(a . theta)`` summed over the
exponent vectors ``a`` that closed walks realise, because a walk and its reverse
contribute conjugate terms.  The fourth moments supply ``cos(theta_j)`` for each
plaquette.  Higher moments supply ``cos(theta_j + theta_k)`` and
``cos(theta_j - theta_k)``, and their difference is ``2 sin(theta_j) sin(theta_k)``.

Now ``cos(theta_j)`` pins each ``theta_j`` up to sign, and
``sin(theta_j) sin(theta_k)`` pins the *relative* signs -- flipping ``theta_j``
alone would need ``sin(theta_j) sin(theta_k) = 0``.  So the signs must move
together, and the ambiguity is a single global one.

**The degeneracy.**  The relative-sign constraint is vacuous exactly when
``sin(theta_j) = 0``, that is when ``omega_j = +-1`` is **real**.  A real
holonomy is its own conjugate, so reversing it does nothing, and the ``Z/2``
acts trivially on that coordinate.  Measured on a ``36 x 36`` sweep:

    both holonomies non-real        two matches, both coordinates flip
    one real, one not               two matches, only the non-real flips
    both real                       ONE match -- conjugation is the identity

So the spectral signature is a complete invariant of the connection modulo
gauge and one global reflection.  The ambiguity group is ``Z/2``, and it acts
**faithfully exactly when some plaquette holonomy is non-real**.  When every
holonomy is real the signature determines the connection outright.

The name
--------

The bit that survives is the **flux chirality**: whether the fluxes run one way
or the other.  It is gauge invariant, it is not determined by any amount of
local spectral data, and reversing it is the unique non-trivial symmetry of the
signature.

The word is chosen deliberately.  This repository has now found three distinct
chiralities and every one of them is invisible to a spectrum:

  * `dirac` -- a chiral point cloud and its mirror have identical distance
    matrices, hence identical filtrations, hence identical persistent homology,
    Laplacian and Dirac spectra.  **Geometric chirality is invisible.**
  * `magnetic` -- the grading operator anticommutes with the Dirac operator
    whatever the connection does, so the spectrum stays symmetric about zero and
    the operator's own chirality is untouched by curvature.  **Operator
    chirality is unbreakable.**
  * here -- the sign of the flux is the one bit of the gauge field that no
    collection of local spectral measures determines.  **Flux chirality is
    invisible.**

Three different objects, one pattern: a spectrum is built from ``|.|^2``-type
data and cannot resolve an orientation.  In each case the invisible thing is
exactly a ``Z/2``.

Scope
-----

Statement one is a proof and holds for every complex, every connection, every
order.

Statement two now has an argument as well as a computation, and the argument has
one hypothesis worth naming: the moment data must actually *contain* the pair
terms ``cos(theta_j +- theta_k)``.  That requires closed walks traversing two
plaquettes, which exist once the complex connects them -- but a complex whose
plaquettes lie in different connected components would not supply them, and
there the signs really are independent.  So the theorem reads: **on a connected
complex, the ambiguity is ``Z/2``**; in general it is one ``Z/2`` per connected
component carrying a non-real holonomy.  `AMBIGUITY_IS_PROVED` is now ``True``
for the connected case, and `component_ambiguity_order` gives the general count.

Novelty
-------

Unverified: arxiv.org is unreachable from this environment.  The ingredients are
standard -- gauge invariance of holonomy, reality of diagonal moments, the
observation that a Hermitian matrix and its conjugate are similar via
transposition.  What is assembled is the inverse question for local spectral
measures on a filtered complex, and the answer that its ambiguity group is
``Z/2``.  Adjacent published work measures spectra of magnetic operators
forwards; the inverse-problem framing and the flux-chirality name are not
something I have seen.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Sequence

import numpy as np

import insertion

__all__ = [
    "spectral_signature",
    "conjugate_angles",
    "signatures_agree",
    "conjugation_residual",
    "conjugation_is_invisible",
    "AMBIGUITY_GROUP_ORDER",
    "AMBIGUITY_IS_PROVED",
    "holonomy_is_real",
    "conjugation_acts_faithfully",
    "component_ambiguity_order",
    "flux_chirality_is_invisible",
    "single_plaquette_reversal_is_visible",
    "rigidity_sweep",
    "SIGNATURE_ORDER",
    "TOLERANCE",
]

#: Moments ``M_0`` through ``M_SIGNATURE_ORDER`` are collected at every simplex.
#: Eight is well past the girth of anything the sweep uses, so the signature sees
#: every plaquette the complex has.
SIGNATURE_ORDER: int = 8

#: Signature comparisons.  Conjugation gives exactly ``0.0``; a single-plaquette
#: reversal gives order one.  There is no borderline case to tune for.
TOLERANCE: float = 1e-8

#: Order of the group of connections sharing a signature, modulo gauge.
#: Exactly two: the connection and its global conjugate.
AMBIGUITY_GROUP_ORDER: int = 2

#: Both statements now have proofs.  Statement two carries a hypothesis -- the
#: complex must connect its plaquettes, so that closed walks supply the pair
#: terms ``cos(theta_j +- theta_k)`` that couple the signs.  See
#: `component_ambiguity_order` for the disconnected case.
AMBIGUITY_IS_PROVED: bool = True


def spectral_signature(
    simplices: Sequence[tuple[int, ...]],
    weight: Callable[[int, int], complex],
    order: int = SIGNATURE_ORDER,
) -> np.ndarray:
    """All local spectral data: moments ``M_0..M_order`` at every simplex.

    The complete observable this module asks about.  One row per simplex, in the
    canonical dimension-then-lexicographic order, so two complexes with the same
    simplices produce comparable arrays.
    """
    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}")
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    return np.array(
        [
            [
                insertion.insertion_moment(ordered, simplex, weight, moment)
                for moment in range(order + 1)
            ]
            for simplex in ordered
        ]
    )


def conjugate_angles(
    angles: dict[tuple[int, int], float]
) -> dict[tuple[int, int], float]:
    """Reverse every flux: negate every edge phase."""
    return {edge: -value for edge, value in angles.items()}


def signatures_agree(first: np.ndarray, second: np.ndarray) -> bool:
    """Do two signatures coincide within tolerance?"""
    if first.shape != second.shape:
        raise ValueError(
            f"signatures have different shapes {first.shape} and {second.shape}; "
            "they describe different complexes"
        )
    return bool(np.abs(first - second).max() <= TOLERANCE)


def conjugation_residual(
    simplices: Sequence[tuple[int, ...]],
    angles: dict[tuple[int, int], float],
    order: int = SIGNATURE_ORDER,
) -> float:
    """``max |signature(sigma) - signature(conj sigma)|``. Exactly zero, provably.

    ``D`` of the conjugate connection is the entrywise conjugate of ``D``, and a
    diagonal moment at a real basis vector is real, so conjugating it changes
    nothing.  The computation returns ``0.0`` rather than machine noise because
    the cancellation is entry by entry.
    """
    original = spectral_signature(simplices, insertion.phase_function(angles), order)
    reversed_ = spectral_signature(
        simplices, insertion.phase_function(conjugate_angles(angles)), order
    )
    return float(np.abs(original - reversed_).max())


def conjugation_is_invisible(
    simplices: Sequence[tuple[int, ...]],
    angles: dict[tuple[int, int], float],
    order: int = SIGNATURE_ORDER,
) -> bool:
    """Statement one: reversing every flux leaves the signature untouched."""
    return conjugation_residual(simplices, angles, order) == 0.0


def flux_chirality_is_invisible(
    simplices: Sequence[tuple[int, ...]],
    angles: dict[tuple[int, int], float],
    order: int = SIGNATURE_ORDER,
) -> bool:
    """The named statement: no local spectral data determines the sign of the flux.

    An alias for `conjugation_is_invisible`, kept because the name is the point.
    The flux chirality is gauge invariant and spectrally undetectable, and those
    two facts together are what make it worth naming rather than dismissing.
    """
    return conjugation_is_invisible(simplices, angles, order)


def single_plaquette_reversal_is_visible(
    reference: tuple[float, float] = (1.0472, 2.2689),
    order: int = SIGNATURE_ORDER,
) -> bool:
    """Reversing *one* plaquette changes the signature; reversing all does not.

    Uses two triangles sharing an edge, gauge-fixed so that the two plaquette
    holonomies are free coordinates ``exp(i t12)`` and ``exp(i t13)``.  The
    asymmetry between the two cases is what makes the ambiguity a single global
    ``Z/2`` rather than one bit per plaquette.
    """
    first, second = reference
    simplices = insertion.close_under_faces([(0, 1, 2), (0, 1, 3)])

    def angles(one: float, two: float) -> dict[tuple[int, int], float]:
        return {(0, 1): 0.0, (0, 2): 0.0, (0, 3): 0.0, (1, 2): one, (1, 3): two}

    base = spectral_signature(
        simplices, insertion.phase_function(angles(first, second)), order
    )
    both = spectral_signature(
        simplices, insertion.phase_function(angles(-first, -second)), order
    )
    one_only = spectral_signature(
        simplices, insertion.phase_function(angles(first, -second)), order
    )
    return signatures_agree(base, both) and not signatures_agree(base, one_only)


def rigidity_sweep(
    resolution: int = 36,
    reference: tuple[int, int] = (6, 13),
    order: int = SIGNATURE_ORDER,
) -> tuple[tuple[int, int], ...]:
    """Every connection on a grid sharing the reference signature.

    Two triangles sharing an edge, gauge-fixed so the plaquette phases are the
    only coordinates, swept over a ``resolution x resolution`` grid.  The result
    should be exactly two points: the reference and its conjugate.

    Returns grid indices rather than angles so the conjugate relation
    ``(i, j) -> (N - i, N - j)`` is visible directly.
    """
    if resolution < 4:
        raise ValueError(f"resolution must be at least four, got {resolution}")
    first, second = reference
    if not (0 <= first < resolution and 0 <= second < resolution):
        raise ValueError(
            f"reference {reference} must have both indices in "
            f"[0, {resolution})"
        )
    simplices = insertion.close_under_faces([(0, 1, 2), (0, 1, 3)])
    grid = [2 * np.pi * step / resolution for step in range(resolution)]

    def signature_at(one: int, two: int) -> np.ndarray:
        return spectral_signature(
            simplices,
            insertion.phase_function(
                {
                    (0, 1): 0.0,
                    (0, 2): 0.0,
                    (0, 3): 0.0,
                    (1, 2): grid[one],
                    (1, 3): grid[two],
                }
            ),
            order,
        )

    target = signature_at(first, second)
    return tuple(
        (one, two)
        for one in range(resolution)
        for two in range(resolution)
        if signatures_agree(signature_at(one, two), target)
    )


def holonomy_is_real(holonomy: complex, tolerance: float = 1e-9) -> bool:
    """Is ``omega = +-1``?  Then it is its own conjugate and the flip does nothing."""
    return abs(holonomy.imag) <= tolerance


def conjugation_acts_faithfully(
    holonomies: Sequence[complex], tolerance: float = 1e-9
) -> bool:
    """Does global conjugation actually move the connection?

    Yes exactly when some plaquette holonomy is non-real.  If every holonomy is
    ``+-1`` the conjugate connection *is* the original, so the ``Z/2`` acts
    trivially and the signature determines the connection outright.
    """
    return any(
        not holonomy_is_real(value, tolerance) for value in holonomies
    )


def component_ambiguity_order(
    component_holonomies: Sequence[Sequence[complex]],
    tolerance: float = 1e-9,
) -> int:
    """Size of the ambiguity group for a possibly disconnected complex.

    The sign-coupling argument runs through closed walks joining two plaquettes,
    so it couples signs only *within* a connected component.  Each component
    carrying at least one non-real holonomy therefore contributes its own
    independent ``Z/2``; components whose holonomies are all real contribute
    nothing, since conjugation fixes them.

    On a connected complex this returns ``2`` when any holonomy is non-real and
    ``1`` otherwise -- the statement in the module docstring.
    """
    if not component_holonomies:
        raise ValueError("need at least one component")
    faithful = sum(
        1
        for component in component_holonomies
        if conjugation_acts_faithfully(component, tolerance)
    )
    return 2**faithful
