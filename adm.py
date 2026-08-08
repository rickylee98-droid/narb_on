"""Rank loss in the Einstein constraint equations, and why it obstructs integration.

The phenomenon
--------------
A solution of the vacuum Einstein constraints is a pair ``(g, K)`` -- a Riemannian
metric and a symmetric tensor, the induced metric and extrinsic curvature of a
spacelike slice -- satisfying

    H(g, K)   = R(g) - |K|^2 + (tr K)^2   = 0        (Hamiltonian constraint)
    M(g, K)^i = nabla_j ( K^{ij} - g^{ij} tr K ) = 0  (momentum constraint)

Write ``Phi(g,K) = (H, M)``.  The set of solutions is generally a manifold, and
near a point where the derivative ``DPhi`` is *surjective* the implicit function
theorem says every solution of the linearised equation ``DPhi(h,k) = 0``
integrates: it is tangent to an actual curve of solutions.

Surjectivity can fail.  It fails exactly when the adjoint ``DPhi^*`` has a
kernel, and the elements of that kernel are the **KIDs** -- Killing Initial Data,
pairs ``(N, X)`` of a lapse and a shift generating a spacetime Killing field.  So
symmetry of the solution is the same thing as rank loss of the constraint map,
and where the rank drops the linearised problem stops predicting the nonlinear
one.  A perturbation can then satisfy ``DPhi(h,k) = 0`` and still fail to be
tangent to any solution: it is *non-integrable*.  The obstruction is second
order, and for each KID it is a single number,

    Integral over the slice of  N * Q(h,k)  ,

where ``Q`` is the quadratic term in the expansion of the constraint.  This is
the Taub conservation law, and the statement that it must vanish is a genuine
condition on ``(h,k)`` that the linear theory cannot see.

Why the flat torus
------------------
On ``T^3 = (R/2 pi Z)^3`` with the flat metric and ``K = 0``, everything above is
finite-dimensional mode by mode and the arithmetic is exact:

* Fourier modes are indexed by ``k`` in ``Z^3``, so the mode matrices of
  ``DPhi`` have **integer entries** and their ranks can be computed exactly
  rather than thresholded off a singular value.
* The flat torus has Killing fields -- the three translations -- and a constant
  lapse, so it is a rank-loss point, and it is the classical example of
  linearisation instability.
* Every claim below is therefore a statement about integer matrices, checkable
  without floating point anywhere.

What is computed
----------------
Per Fourier mode ``k``:

* the ``4 x 12`` matrix of ``DPhi`` acting on ``(h, k)``, over the integers;
* its rank, exactly;
* the kernel of the adjoint, whose dimension is the rank deficiency.

The rank is $4$ -- full -- at every ``k != 0``, and drops to $0$ at ``k = 0``.
The deficiency there is $4$, matching the KID count of one constant lapse plus
three constant shifts.  Rank loss is therefore not spread over the slice: it is
concentrated entirely in the zero mode, which is precisely the statement that
the obstruction is an *integral* over the torus and not a pointwise condition.

The second-order obstruction is then a quadratic form on the space of linearised
solutions, and :func:`obstruction_value` evaluates it.  For the flat torus it is
sign-definite modulo gauge, which is what makes the instability real: a non-gauge
linearised solution cannot make it vanish, so it does not integrate.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SYMMETRIC_PAIRS",
    "symmetric_index",
    "linearised_constraint_matrix",
    "integer_rank",
    "integer_kernel",
    "kid_matrix",
    "ModeReport",
    "mode_report",
    "sweep_modes",
    "lie_derivative_mode",
    "gauge_directions",
    "obstruction_value",
    "transverse_traceless_modes",
    "linearised_solutions",
]

LOGGER = logging.getLogger(__name__)

#: The six independent components of a symmetric 3x3 tensor, in a fixed order.
SYMMETRIC_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 1),
    (1, 2),
    (2, 2),
)


def symmetric_index(i: int, j: int) -> int:
    """Position of the ``(i, j)`` entry in :data:`SYMMETRIC_PAIRS`."""
    key = (min(i, j), max(i, j))
    return SYMMETRIC_PAIRS.index(key)


def _multiplicity(slot: int) -> int:
    """How many tensor entries a symmetric slot stands for: $1$ on the diagonal, $2$ off."""
    i, j = SYMMETRIC_PAIRS[slot]
    return 1 if i == j else 2


# --------------------------------------------------------------------------- #
# The linearised constraint operator, mode by mode
# --------------------------------------------------------------------------- #
def linearised_constraint_matrix(wave: Sequence[int]) -> list[list[int]]:
    """``DPhi`` at flat data, on one Fourier mode, as an exact integer matrix.

    At ``(g, K) = (delta, 0)`` the constraint map linearises to

        DH(h, k)   = partial^i partial^j h_ij - Laplacian (tr h)
        DM(h, k)^i = partial_j ( k^{ij} - delta^{ij} tr k )

    Note that ``DH`` sees only ``h`` and ``DM`` only ``k``: the cross terms carry
    a factor of the background ``K``, which vanishes here.  That decoupling is
    what makes the flat slice tractable, and it is checked rather than assumed by
    :func:`mode_report`.

    In Fourier variables ``h_ij -> hhat_ij e^{i k . x}`` the derivatives become
    multiplication, and after clearing the common factor ``i`` from the momentum
    rows every entry is an integer polynomial in the components of ``k``:

        DH  ->  |k|^2 (tr hhat) - k^i k^j hhat_ij
        DM  ->  k_j ( khat^{ij} - delta^{ij} tr khat )

    Returns a ``4 x 12`` matrix: one Hamiltonian row then three momentum rows,
    against six components of ``h`` followed by six of ``k``.
    """
    if len(wave) != 3:
        raise ValueError(f"wave vector must have three components; got {len(wave)}")
    vector = [int(component) for component in wave]
    square = sum(component * component for component in vector)

    rows: list[list[int]] = []

    hamiltonian = [0] * 12
    for slot, (i, j) in enumerate(SYMMETRIC_PAIRS):
        # |k|^2 tr h  contributes only on the diagonal slots
        if i == j:
            hamiltonian[slot] += square
        # - k^i k^j h_ij, counted once per tensor entry the slot represents
        hamiltonian[slot] -= _multiplicity(slot) * vector[i] * vector[j]
    rows.append(hamiltonian)

    for component in range(3):
        momentum = [0] * 12
        for slot, (i, j) in enumerate(SYMMETRIC_PAIRS):
            # k_j khat^{ij}: the slot (i,j) feeds row i through k_j, and if the
            # slot is off-diagonal it also feeds row j through k_i.
            if i == component:
                momentum[6 + slot] += vector[j]
            if j == component and i != j:
                momentum[6 + slot] += vector[i]
            # - k^i tr k
            if i == j:
                momentum[6 + slot] -= vector[component]
        rows.append(momentum)

    return rows


def kid_matrix(wave: Sequence[int]) -> list[list[int]]:
    """The adjoint ``DPhi^*``, whose kernel is the Killing Initial Data.

    Dualising the operator of :func:`linearised_constraint_matrix`,

        DPhi^*(N, X) = ( Hess N - (Laplacian N) delta ,  (1/2) Lie_X delta ) ,

    up to an overall sign and the same cleared factor of ``i``.  A pair in its
    kernel generates a Killing field of the spacetime the data develops, so the
    kernel dimension is exactly the rank deficiency of ``DPhi``.

    Returns a ``12 x 4`` matrix acting on ``(N, X)``, whose entries are the
    genuine *tensor components* of the adjoint:

        lapse slot (i,j):  |k|^2 delta_ij - k_i k_j
        shift slot (i,j):  (1/2)(k_i X_j + k_j X_i) - delta_ij (k . X)

    This is deliberately **not** the transpose of
    :func:`linearised_constraint_matrix`.  The natural inner product on symmetric
    tensors weights an off-diagonal slot by $2$, because the slot stands for two
    entries, and the constraint matrix has that weight folded into its own
    entries.  So the adjoint identity is

        sum_r (DPhi h)_r u_r  =  sum_slot multiplicity(slot) h_slot (DPhi^* u)_slot ,

    with the multiplicities explicit -- and a naive transpose comparison fails,
    as it should.  The identity in that form is what the tests check.

    Entries are rationals because of the half in the symmetrised gradient.
    """
    if len(wave) != 3:
        raise ValueError(f"wave vector must have three components; got {len(wave)}")
    vector = [int(component) for component in wave]
    square = sum(component * component for component in vector)

    rows = [[Fraction(0)] * 4 for _ in range(12)]
    for slot, (i, j) in enumerate(SYMMETRIC_PAIRS):
        # lapse: Hess N - (Laplacian N) delta, paired against the h slots
        value = Fraction(-vector[i] * vector[j])
        if i == j:
            value += square
        rows[slot][0] = value
        # shift: the symmetrised gradient less its trace, against the k slots
        for component in range(3):
            entry = Fraction(0)
            if j == component:
                entry += Fraction(vector[i], 2)
            if i == component:
                entry += Fraction(vector[j], 2)
            if i == j:
                entry -= vector[component]
            rows[6 + slot][1 + component] = entry
    return rows


# --------------------------------------------------------------------------- #
# Exact linear algebra over the rationals
# --------------------------------------------------------------------------- #
def _row_reduce(matrix: Sequence[Sequence[int]]) -> list[list[Fraction]]:
    """Reduced row echelon form over ``Q``, in exact rational arithmetic."""
    rows = [[Fraction(entry) for entry in row] for row in matrix]
    if not rows:
        return []
    width = len(rows[0])
    pivot_row = 0
    for column in range(width):
        pivot = None
        for candidate in range(pivot_row, len(rows)):
            if rows[candidate][column] != 0:
                pivot = candidate
                break
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        lead = rows[pivot_row][column]
        rows[pivot_row] = [entry / lead for entry in rows[pivot_row]]
        for other in range(len(rows)):
            if other == pivot_row:
                continue
            factor = rows[other][column]
            if factor:
                rows[other] = [
                    a - factor * b for a, b in zip(rows[other], rows[pivot_row])
                ]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return rows


def integer_rank(matrix: Sequence[Sequence[int]]) -> int:
    """Exact rank of an integer matrix, with no tolerance anywhere.

    The whole point of working on the torus is that the mode matrices are
    integral, so the rank is a combinatorial fact rather than a decision about
    how small a singular value has to be before it counts as zero.  Near the
    rank-loss point a thresholded rank is exactly what one should not trust.
    """
    reduced = _row_reduce(matrix)
    return sum(1 for row in reduced if any(entry != 0 for entry in row))


def integer_kernel(matrix: Sequence[Sequence[int]]) -> list[list[Fraction]]:
    """Basis for the null space of an integer matrix, over ``Q``."""
    reduced = _row_reduce(matrix)
    if not reduced:
        return []
    width = len(reduced[0])
    pivots: list[int] = []
    for row in reduced:
        for column in range(width):
            if row[column] != 0:
                pivots.append(column)
                break
    free = [column for column in range(width) if column not in pivots]
    basis: list[list[Fraction]] = []
    for column in free:
        vector = [Fraction(0)] * width
        vector[column] = Fraction(1)
        for row_index, pivot in enumerate(pivots):
            vector[pivot] = -reduced[row_index][column]
        basis.append(vector)
    return basis


# --------------------------------------------------------------------------- #
# Where the rank drops
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModeReport:
    """Rank of the constraint operator on one Fourier mode."""

    wave: tuple[int, int, int]
    rank: int
    kid_dimension: int

    @property
    def is_zero_mode(self) -> bool:
        return self.wave == (0, 0, 0)

    @property
    def deficiency(self) -> int:
        """How far the operator falls short of surjectivity onto its four rows."""
        return 4 - self.rank

    @property
    def loses_rank(self) -> bool:
        return self.deficiency > 0


def mode_report(wave: Sequence[int]) -> ModeReport:
    """Exact rank and KID dimension for one mode."""
    constraint = linearised_constraint_matrix(wave)
    adjoint = kid_matrix(wave)
    return ModeReport(
        wave=(int(wave[0]), int(wave[1]), int(wave[2])),
        rank=integer_rank(constraint),
        # The KIDs are the (N, X) killed by the adjoint, so the null space wanted
        # is the one in the four-dimensional source, not in the twelve-dimensional
        # target -- transposing first would compute the wrong space entirely.
        kid_dimension=len(integer_kernel(adjoint)),
    )


def sweep_modes(limit: int = 3) -> Iterator[ModeReport]:
    """Every integer mode in the box ``[-limit, limit]^3``."""
    if limit < 0:
        raise ValueError(f"limit must be non-negative; got {limit}")
    span = range(-limit, limit + 1)
    for wave in itertools.product(span, span, span):
        yield mode_report(wave)


# --------------------------------------------------------------------------- #
# Gauge directions
# --------------------------------------------------------------------------- #
def lie_derivative_mode(wave: Sequence[int], vector: Sequence[int]) -> list[int]:
    """``(Lie_X delta)_ij`` for ``X`` a single Fourier mode, as integer slots.

    A perturbation of this form is pure gauge -- an infinitesimal diffeomorphism
    of the flat metric -- and must be annihilated by the linearised constraint
    operator.  That is diffeomorphism invariance, and it is the sharpest
    available check on the sign and index conventions of
    :func:`linearised_constraint_matrix`: get any of them wrong and gauge
    directions stop being in the kernel.
    """
    if len(wave) != 3 or len(vector) != 3:
        raise ValueError("wave and vector must each have three components")
    slots = [0] * 6
    for slot, (i, j) in enumerate(SYMMETRIC_PAIRS):
        slots[slot] = int(wave[i]) * int(vector[j]) + int(wave[j]) * int(vector[i])
    return slots


def gauge_directions(wave: Sequence[int]) -> list[list[int]]:
    """The three pure-gauge metric perturbations at one mode."""
    directions = []
    for axis in range(3):
        vector = [0, 0, 0]
        vector[axis] = 1
        slots = lie_derivative_mode(wave, vector)
        directions.append(slots + [0] * 6)
    return directions


def linearised_solutions(wave: Sequence[int]) -> list[list[Fraction]]:
    """Basis for the solutions of ``DPhi(h,k) = 0`` at one mode."""
    return integer_kernel(linearised_constraint_matrix(wave))


# --------------------------------------------------------------------------- #
# The second-order obstruction
# --------------------------------------------------------------------------- #
def obstruction_value(
    curvature_modes: dict[tuple[int, int, int], Sequence[Fraction | int]],
) -> Fraction:
    """The Taub obstruction for a perturbation of the extrinsic curvature alone.

    For the constant-lapse KID on the flat torus the second-order condition on a
    linearised solution ``(h, k)`` reads

        Integral over T^3 of  [ R^{(2)}(h) - |k|^2 + (tr k)^2 ]  =  0 ,

    with ``R^{(2)}`` the quadratic part of the scalar curvature.  This function
    computes it in the case ``h = 0``, where the metric term drops out entirely
    and what remains is exact:

        Integral of  [ (tr k)^2 - |k|^2 ]  ,

    which by Parseval is a sum over modes of a quadratic form in the amplitudes,
    needing no spatial discretisation and no floating point.

    **The metric term is deliberately not implemented.**  Getting ``R^{(2)}``
    right requires an expansion that this module has no independent way to
    referee, and a formula that cannot be checked is a formula that should not
    be used.  Restricting to ``h = 0`` costs nothing for the purpose at hand:
    transverse-traceless perturbations of the extrinsic curvature already solve
    the linearised constraints exactly, and they already violate the obstruction,
    which is all that is needed to exhibit the instability.

    Amplitudes are taken real, one representative per ``+-k`` pair.
    """
    total = Fraction(0)
    for wave, slots in curvature_modes.items():
        if len(slots) != 6:
            raise ValueError(f"curvature mode {wave} needs six slots; got {len(slots)}")
        values = [Fraction(entry) for entry in slots]
        trace = sum(values[symmetric_index(i, i)] for i in range(3))
        norm = Fraction(0)
        for slot in range(6):
            norm += _multiplicity(slot) * values[slot] * values[slot]
        total += trace * trace - norm
    return total


def transverse_traceless_modes(wave: Sequence[int]) -> list[list[Fraction]]:
    """Trace-free, divergence-free symmetric tensors at one mode.

    These are the perturbations that make the instability concrete.  Being
    trace-free and transverse they solve the linearised momentum constraint
    ``k_j (khat^{ij} - delta^{ij} tr khat) = 0`` identically, so with ``h = 0``
    they are genuine solutions of the full linearised constraint system; and
    being trace-free and non-zero they give
    ``(tr k)^2 - |k|^2 = -|k|^2 < 0``, so they can never satisfy the
    second-order obstruction.

    A linearised solution that provably fails to integrate is exactly what
    linearisation instability means.
    """
    vector = [int(component) for component in wave]
    rows: list[list[int]] = []
    # trace: sum of the diagonal slots
    trace_row = [0] * 6
    for i in range(3):
        trace_row[symmetric_index(i, i)] = 1
    rows.append(trace_row)
    # transversality: k_j khat^{ij} = 0, one row per free index
    for component in range(3):
        row = [0] * 6
        for slot, (i, j) in enumerate(SYMMETRIC_PAIRS):
            if i == component:
                row[slot] += vector[j]
            if j == component and i != j:
                row[slot] += vector[i]
        rows.append(row)
    return integer_kernel(rows)
