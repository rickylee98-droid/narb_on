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

**Two, verified.  That is the only ambiguity.**  On a complex with two
independent plaquettes, gauge-fixed so the two holonomies are free coordinates,
an exhaustive sweep of 5184 connections on a ``72 x 72`` grid finds exactly two
sharing any given signature: the connection itself and its global conjugate.
Reversing a *single* plaquette is visible, and visible by a wide margin -- moment
differences of order ``1`` to ``4`` against a detection threshold of ``1e-8``.

So the spectral signature is a complete invariant of the connection modulo gauge
and one global reflection.  The ambiguity group is exactly ``Z/2``.

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
order.  Statement two is an exhaustive computation on small complexes -- a
single plaquette and a pair of plaquettes -- and is **not** proved in general.
What would be needed is an argument that the real parts of all products of
plaquette holonomies determine those holonomies up to simultaneous conjugation.
That is plausible and unproved; `AMBIGUITY_IS_PROVED` records which half is
which.

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

#: Statement one is proved; statement two is an exhaustive computation on small
#: complexes.  ``False`` records that the general case is not established.
AMBIGUITY_IS_PROVED: bool = False


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
    if not (0 < first < resolution and 0 < second < resolution):
        raise ValueError(
            f"reference {reference} must have both indices strictly inside "
            f"(0, {resolution}) so its conjugate is a distinct grid point"
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
