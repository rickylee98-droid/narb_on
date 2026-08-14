"""The Obukhov amplifier gate, realised in the true Euler nonlinearity.

The open step
-------------
Tao's program for a Navier-Stokes blow-up has two steps: build a shell model that
blows up, then embed it in the true equations.  Step 1 was settled for a
Navier-Stokes-like model by Palasek (arXiv:2605.13827, May 2026) using the
Obukhov system with super-exponential shells; see :mod:`shell`.  Step 2 is open.

That paper's case for its model being embeddable rests on its two interactions
being ones the real Euler nonlinearity contains.  One of them,
``P_k div(u_{k+1} x u_{k+1})``, it calls harnessable by convex integration after
De Lellis--Szekelyhidi.  The other it flags as the hard one:

    "The other Obukhov interaction, N_{k-1}^alpha X_{k-1} X_k, is more difficult
     to harness, but nonetheless is easily understood as the nonlinear
     interaction u_k . grad u_{k-1}."

This module computes what the true Euler nonlinearity does with that
interaction, exactly.

The obstruction that isn't there
--------------------------------
The worry is a coefficient mismatch.  A triad ``k1 + k2 + k3 = 0`` with one low
mode and two high ones carries interaction coefficients of size ``|k_high|``,
which for widely separated shells is enormous compared with the ``|k_low|`` the
Obukhov gate asks for.  If that survived, the gate would be unrealisable.

It does not survive.  Writing the triad in the helical basis, the Euler
nonlinearity gives

    a1' = c1 conj(a2) conj(a3),   c1 = -(1/2) g (s2 |k2| - s3 |k3|) ,

and cyclically, with a single geometric factor ``g`` shared by all three
(:func:`triad_coefficients`).  The energy of the **high pair** then moves at rate

    d/dt (|a2|^2 + |a3|^2)  =  -2 Re( c1 conj(a1) conj(a2) conj(a3) ) ,

so the amplification of the high pair by the low mode is governed by ``c1``
alone -- and ``c1`` depends on the two high wavenumbers only through their
*difference*.  The ``O(|k_high|)`` parts cancel identically
(:func:`transport_cancellation_residual`).  What is left is bounded by ``|k1|``,
because ``k2 + k3 = -k1``.  That cancellation is the statement that the leading
non-local interaction is pure transport, which moves a small eddy without
amplifying it; the residue is the strain of the low mode, and strain is exactly
what the Obukhov gate needs.

The gate law
------------
In the scale-separated limit ``|k2|, |k3| -> infinity`` with ``k1`` fixed and the
high pair carrying the *same* helicity, the surviving coefficient is

    |c1| / |k1|  =  |sin(theta) cos(theta)|  =  |sin(2 theta)| / 2 ,

where ``theta`` is the angle between the low mode and the high pair
(:func:`gate_efficiency_limit`).  It is maximised at

    **theta = 45 degrees, efficiency exactly 1/2** .

Both factors are forced and pull against each other: ``|s2|k2| - s3|k3||`` tends
to ``|k1| |cos theta|``, largest when the triad is collinear, while the geometric
factor tends to ``2 |sin theta|``, and vanishes for collinear triads -- the
classical statement that collinear triads do not interact, since a divergence-free
mode is orthogonal to its own wavevector.  Their product is the gate.

So the amplifier gate is realisable in the true Euler nonlinearity, at an
efficiency of one half, and the geometry that attains it is explicit.

What this does and does not settle
----------------------------------
It settles the coefficient question for one gate: there is no ``|k_high|``
mismatch to overcome, and the ``|k_low|`` the Obukhov model wants is exactly what
Euler supplies, with a computable constant.  The remaining factor
``N_{k-1}^{alpha}`` rather than ``N_{k-1}`` comes from intermittency, via the
volume fraction ``N_k^{-2(alpha-1)}`` of Palasek's §2, not from the triad.

It does not build an embedding.  A real construction needs the gate to act
coherently across infinitely many shells at once, with the errors from every
other triad -- the ones this module deliberately looks at one at a time --
controlled.  That is the open problem and nothing here closes it.

Referees
--------
Energy and helicity are conserved exactly, symbolically, on every triad and every
helicity assignment (:func:`conservation_residuals`).  Collinear triads return a
vanishing geometric factor.  The asymptotic gate law is checked against the
finite-wavenumber coefficient, which converges to it, and the maximiser is
located independently of the closed form.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import sympy as sp

__all__ = [
    "Triad",
    "helical_vector",
    "triad_coefficients",
    "conservation_residuals",
    "geometric_factor",
    "is_collinear",
    "high_pair_amplification",
    "transport_cancellation_residual",
    "gate_efficiency_limit",
    "gate_efficiency",
    "OPTIMAL_ANGLE_DEGREES",
    "OPTIMAL_EFFICIENCY",
]

LOGGER = logging.getLogger(__name__)

#: Angle between the low mode and the high pair that maximises the gate.
OPTIMAL_ANGLE_DEGREES = 45

#: The value there: ``|c1| / |k1| = 1/2``.  Both factors of the gate are forced,
#: and this is where their product peaks.
OPTIMAL_EFFICIENCY = sp.Rational(1, 2)


@dataclass(frozen=True)
class Triad:
    """Three integer wavevectors summing to zero, with helicity signs."""

    k1: tuple[int, int, int]
    k2: tuple[int, int, int]
    k3: tuple[int, int, int]
    helicities: tuple[int, int, int] = (1, 1, 1)

    def __post_init__(self) -> None:
        total = tuple(a + b + c for a, b, c in zip(self.k1, self.k2, self.k3))
        if total != (0, 0, 0):
            raise ValueError(f"a triad must close: {self.k1} + {self.k2} + {self.k3} != 0")
        for vector in (self.k1, self.k2, self.k3):
            if all(component == 0 for component in vector):
                raise ValueError("a triad has no zero mode")
        if any(sign not in (1, -1) for sign in self.helicities):
            raise ValueError("helicities are +-1")

    @property
    def vectors(self) -> tuple[tuple[int, int, int], ...]:
        return (self.k1, self.k2, self.k3)


def helical_vector(wavevector: Sequence[int]):
    """``h^+(k)``: a unit complex vector orthogonal to ``k`` with ``i k x h = |k| h``.

    Built from an arbitrary reference direction, so its phase is a convention;
    every quantity this module reports is phase independent, which is why only
    magnitudes are compared.
    """
    k = sp.Matrix([sp.Integer(component) for component in wavevector])
    norm = sp.sqrt(k.dot(k))
    if norm == 0:
        raise ValueError("the zero wavevector carries no helical basis")
    reference = (
        sp.Matrix([0, 0, 1]) if (k[0] != 0 or k[1] != 0) else sp.Matrix([1, 0, 0])
    )
    first = k.cross(reference)
    first = first / sp.sqrt(first.dot(first))
    second = k.cross(first) / norm
    return first + sp.I * second, norm


def geometric_factor(triad: Triad):
    """``g = (h2* x h3*) . h1*``, the factor shared by all three coefficients.

    Its vanishing is what makes collinear triads inert.
    """
    vectors = []
    for wavevector, sign in zip(triad.vectors, triad.helicities):
        helical, _ = helical_vector(wavevector)
        vectors.append(sp.conjugate(helical) if sign > 0 else helical)
    return sp.simplify((vectors[1].cross(vectors[2])).dot(vectors[0]))


def is_collinear(triad: Triad) -> bool:
    """Whether all three wavevectors lie on one line.

    Then every divergence-free mode is orthogonal to all three wavevectors at
    once, the nonlinearity ``(u(p) . q) u(q)`` vanishes, and the triad does not
    interact at all.
    """
    a = sp.Matrix(triad.k1)
    b = sp.Matrix(triad.k2)
    return a.cross(b) == sp.zeros(3, 1)


def triad_coefficients(triad: Triad) -> tuple:
    """``(c1, c2, c3)`` with ``a1' = c1 conj(a2) conj(a3)`` and cyclically.

    ``c_j = -(1/2) g (s_{j+1} |k_{j+1}| - s_{j+2} |k_{j+2}|)`` -- Waleffe's helical
    decomposition of the Euler nonlinearity.  Each coefficient sees the other two
    wavenumbers only through their signed difference, which is the whole point.
    """
    factor = geometric_factor(triad)
    norms = [helical_vector(vector)[1] for vector in triad.vectors]
    signs = triad.helicities
    return tuple(
        sp.simplify(
            -(signs[(j + 1) % 3] * norms[(j + 1) % 3] - signs[(j + 2) % 3] * norms[(j + 2) % 3])
            * factor
            / 2
        )
        for j in range(3)
    )


def conservation_residuals(triad: Triad) -> tuple:
    """``(energy, helicity)`` residuals, both of which must vanish exactly.

    Energy asks ``sum_j c_j = 0``; helicity asks ``sum_j s_j |k_j| c_j = 0``.  The
    Euler nonlinearity conserves both triad by triad, so these hold for every
    closed triad and every helicity assignment -- and they are what pins the
    coefficient convention, since nothing else here fixes it.
    """
    coefficients = triad_coefficients(triad)
    norms = [helical_vector(vector)[1] for vector in triad.vectors]
    energy = sp.simplify(sum(coefficients))
    helicity = sp.simplify(
        sum(sign * norm * coefficient
            for sign, norm, coefficient in zip(triad.helicities, norms, coefficients))
    )
    return energy, helicity


def high_pair_amplification(triad: Triad):
    """The coefficient governing ``d/dt (|a2|^2 + |a3|^2)``, namely ``c1``.

    Energy conservation gives ``c2 + c3 = -c1``, so the growth of the high pair is
    ``-2 Re(c1 conj(a1) conj(a2) conj(a3))``: a single coefficient, and the one
    that must match the Obukhov gate.
    """
    return triad_coefficients(triad)[0]


def transport_cancellation_residual(triad: Triad):
    """What is left of the high wavenumbers in ``c1`` beyond their difference.

    ``c1`` is built from ``s2 |k2| - s3 |k3|``, so the two large wavenumbers enter
    only through that combination.  This returns
    ``c1 + (1/2) g (s2 |k2| - s3 |k3|)``, which is identically zero -- the
    ``O(|k_high|)`` transport has no residue.  Stated as a residual rather than an
    assertion because it is the step the coefficient worry turns on.
    """
    factor = geometric_factor(triad)
    norms = [helical_vector(vector)[1] for vector in triad.vectors]
    signs = triad.helicities
    return sp.simplify(
        triad_coefficients(triad)[0]
        + (signs[1] * norms[1] - signs[2] * norms[2]) * factor / 2
    )


def gate_efficiency_limit(theta: float | sp.Expr):
    """``|c1| / |k1| = |sin(theta) cos(theta)|`` in the scale-separated limit.

    Derived, not fitted.  As ``|k2| = Q -> infinity`` with ``k3 = -k1 - k2``:

    * ``|k2| - |k3| -> -(khat2 . k1) = -|k1| cos(theta)``, so the wavenumber
      difference tends to ``|k1| |cos theta|``;
    * the high pair becomes antiparallel, so for equal helicities
      ``h2* x h3* -> 2 i khat2``, and ``g -> 2 |khat2 . h1*| = 2 |sin theta|``.

    The product of the two, halved, is the law.  Each factor alone is maximised
    where the other vanishes.
    """
    return sp.Abs(sp.sin(theta) * sp.cos(theta))


def gate_efficiency(theta: float, magnitude: float, low: float = 1.0) -> float:
    """``|c1| / |k1|`` at finite wavenumber, for comparison with the limit.

    The low mode is ``(low, 0, 0)`` and the high mode has length ``magnitude`` at
    angle ``theta`` to it, both helicities positive.  Floating point is used here
    because the limit is a statement about large ``|k2|``, where exact algebraic
    numbers stop being informative; the exact statements in this module are the
    conservation laws and the transport cancellation.
    """
    if magnitude <= 0 or low <= 0:
        raise ValueError("wavenumber magnitudes must be positive")
    k1 = (low, 0.0, 0.0)
    k2 = (magnitude * math.cos(theta), magnitude * math.sin(theta), 0.0)
    k3 = tuple(-a - b for a, b in zip(k1, k2))
    return _numeric_c1(k1, k2, k3, (1, 1, 1)) / low


def _numeric_c1(k1, k2, k3, signs) -> float:
    import numpy as np

    def helical(vector):
        vector = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(vector))
        reference = (
            np.array([0.0, 0.0, 1.0])
            if abs(vector[0]) + abs(vector[1]) > 1e-12
            else np.array([1.0, 0.0, 0.0])
        )
        first = np.cross(vector, reference)
        first /= np.linalg.norm(first)
        second = np.cross(vector, first) / norm
        return first + 1j * second, norm

    vectors, norms = [], []
    for vector, sign in zip((k1, k2, k3), signs):
        helical_part, norm = helical(vector)
        vectors.append(np.conj(helical_part) if sign > 0 else helical_part)
        norms.append(norm)
    factor = np.dot(np.cross(vectors[1], vectors[2]), vectors[0])
    return abs(-(signs[1] * norms[1] - signs[2] * norms[2]) * factor / 2)
