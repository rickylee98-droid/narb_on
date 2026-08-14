"""The blow-up window of the Obukhov shell model, and where it sits.

Where the problem actually stands
---------------------------------
Global regularity for 3D Navier-Stokes is open, and the standard route toward a
counterexample is Tao's two-step program: build a shell model that blows up, then
embed it in the true equations.  Step 1 has a short and very recent history:

* Katz--Pavlovic model -- inviscid blow-up, but the three-dimensional viscous
  parameters are globally well-posed (Barbato--Morandin--Romito, Cheskidov).
* Obukhov model with exponential shells ``N_k = lambda^k`` -- globally regular,
  inviscid and viscous alike.
* Tao's model -- blows up with viscosity, but its interactions have no clear
  counterpart in the Euler nonlinearity, which stalls Step 2.
* **Obukhov model with super-exponential shells** ``N_k = N_0^{b^k}`` -- Palasek,
  arXiv:2605.13827 (May 2026), Theorems 1.3 and 1.8: finite-time blow-up, viscous
  with smooth forcing for ``alpha > 2``, and inviscid unforced for ``alpha >= 1``.

That last model is three months old and is the current best candidate for Step 2.
This module is about its parameter geometry.  The model is

    X_k' = -nu N_k^2 X_k + N_{k-1}^alpha X_{k-1} X_k - N_k^alpha X_{k+1}^2 + f_k ,

with ``X_{-1} = 0``; ``alpha`` is an intermittency parameter, physically in
``[1, 5/2]`` in three dimensions.

What is derived here
--------------------
**1. The cascade exponent, for arbitrary shell separation.**  The model has an
exact power-law stationary state ``X_k = c N_k^{-gamma}`` with

    gamma  =  alpha / (2b + 1) ,

verified as an identity between exponents rather than numerically
(:func:`fixed_point_exponent_residual`).  At ``b = 1``, exponential shells, this
is ``alpha/3``, and at ``alpha = 1`` it is the Kolmogorov ``1/3`` -- so the
formula is anchored to a classical value it was not fitted to.

The consequence is quantitative and is the point: ``gamma`` **decreases in** ``b``.
Wider shell separation flattens the cascade state that regularises the model, and
in the limit ``b -> infinity`` it flattens to nothing.  This is the mechanism by
which super-exponential separation buys blow-up, expressed as one exponent.

**2. The trapping region lies strictly above the cascade.**  Palasek's barriers
are ``A_k = N_k^beta`` in the rescaled variable, and the cascade state is
``N_k^{2 alpha b/(2b+1)}`` there.  His viscous constraint ``beta > 2b`` puts the
region above the cascade exactly when

    2b + 1  >=  alpha ,

which is automatic for ``b > 1`` whenever ``alpha <= 3`` -- and so throughout the
three-dimensional window (:func:`clears_cascade`).  In the inviscid case it holds
unconditionally.  The blow-up mechanism is therefore *escape above the
regularising cascade*, not a competition with it.

**3. A tension in the parameters worth naming.**  Palasek's viscous theorem needs
``b in (1, alpha/2)``, which is non-empty exactly when ``alpha > 2``
(:func:`viscous_window`) -- recovering his sharpness claim from the constraints
alone, without any dynamics.  But the physically relevant intermittency range in
three dimensions is ``alpha in [1, 5/2]``, so the candidate blow-up window is
``alpha in (2, 5/2]`` and there

    b  <  5/4 .

The separation is super-exponential but only barely.  Section 4 of the paper
argues that wide separation is what makes an embedding into the true nonlinearity
tractable, since it suppresses cross-scale errors; the physically relevant window
is where that advantage is thinnest.  This module states the trade-off exactly
(:func:`separation_budget`); it does not resolve it.

What this is not
----------------
Not a contribution to the Navier-Stokes problem itself.  The open step is the
embedding, which is a PDE construction and is untouched here.  What is offered is
exact arithmetic on the parameter geometry of the model that is currently the
best candidate for it, one cascade exponent that appears not to have been written
down for ``b > 1``, and referees for the structural identities the model rests on.

Referees
--------
The cascade exponent reduces to Kolmogorov's ``1/3``.  The nonlinearity's
contribution to the energy telescopes to exactly zero, in rational arithmetic, on
truncations (:func:`energy_identity_residual`).  The invariance of frequency
truncation -- Palasek's Remark 1.10, which is why his blow-up is unstable in
every ``C^s`` -- is checked as an exact property of the vector field
(:func:`truncation_residual`).  And the viscous parameter window is computed to be
non-empty precisely for ``alpha > 2``, which the paper obtains independently from
energy criticality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

__all__ = [
    "ObukhovModel",
    "log_frequency",
    "kolmogorov_exponent",
    "fixed_point_exponent_residual",
    "rescaled_cascade_exponent",
    "energy_identity_residual",
    "truncation_residual",
    "viscous_window",
    "inviscid_window",
    "beta_window",
    "separation_budget",
    "clears_cascade",
    "INTERMITTENCY_RANGE",
    "THREE_D_BLOWUP_WINDOW",
]

LOGGER = logging.getLogger(__name__)

#: Physical range of the intermittency parameter in three dimensions, from the
#: model's derivation: ``alpha = 1`` is space-filling, ``alpha = 5/2`` is the most
#: intermittent case considered.
INTERMITTENCY_RANGE = (Fraction(1), Fraction(5, 2))

#: Where the two overlap: viscous blow-up needs ``alpha > 2``, so the candidate
#: window for a three-dimensional scenario is ``(2, 5/2]``.
THREE_D_BLOWUP_WINDOW = (Fraction(2), Fraction(5, 2))


@dataclass(frozen=True)
class ObukhovModel:
    """Parameters of the Obukhov shell model with shells ``N_k = N_0^{b^k}``.

    ``b = 1`` is the classical exponential model, which is globally regular;
    ``b > 1`` is Palasek's super-exponential variant.  ``alpha`` is the
    intermittency parameter and ``viscous`` selects which of the two theorems'
    constraints apply.
    """

    alpha: Fraction
    b: Fraction
    viscous: bool = True

    def __post_init__(self) -> None:
        if self.alpha < 1:
            raise ValueError("the intermittency parameter satisfies alpha >= 1")
        if self.b < 1:
            raise ValueError("shells must grow at least exponentially, so b >= 1")


def log_frequency(b: Fraction | int, k: int) -> Fraction:
    """``b^k``: the exponent of ``N_0`` in ``N_k``, kept exact.

    Every scaling relation in the model is linear in these exponents, so working
    with them rather than with ``N_k`` keeps the arithmetic rational even though
    ``N_k`` itself is a double exponential and generally irrational.
    """
    if k < 0:
        raise ValueError("shell index must be non-negative")
    return Fraction(b) ** k


def kolmogorov_exponent(alpha: Fraction | int, b: Fraction | int) -> Fraction:
    """``gamma = alpha / (2b + 1)`` in the stationary state ``X_k = c N_k^{-gamma}``.

    Derived from the fixed-point equation, not fitted.  At ``b = 1`` it is
    ``alpha/3``, and at ``alpha = 1`` that is Kolmogorov's ``1/3``.
    """
    return Fraction(alpha) / (2 * Fraction(b) + 1)


def rescaled_cascade_exponent(alpha: Fraction | int, b: Fraction | int) -> Fraction:
    """The same state in Palasek's rescaled variable ``x_k = N_k^alpha X_k``.

    ``x_k = N_k^{2 alpha b / (2b + 1)}``, which is what the trapping barriers
    ``A_k = N_k^beta`` have to be compared against.
    """
    return 2 * Fraction(alpha) * Fraction(b) / (2 * Fraction(b) + 1)


def fixed_point_exponent_residual(
    alpha: Fraction | int, b: Fraction | int, k: int
) -> Fraction:
    """Residual of the stationary condition, as an exact identity in exponents.

    The rescaled fixed point ``x_{k-1} x_k = delta_k x_{k+1}^2`` with
    ``delta_k = (N_k / N_{k+1})^{2 alpha}`` and ``x_k = N_k^{q}`` becomes, after
    taking ``log`` base ``N_0``,

        q (b^{k-1} + b^k)  =  2 alpha (b^k - b^{k+1}) + 2 q b^{k+1} ,

    a linear equation whose solution is :func:`rescaled_cascade_exponent`.  This
    returns the two sides' difference, which must vanish for every ``k`` -- an
    exact rational zero, not a numerical one.
    """
    if k < 1:
        raise ValueError("the fixed-point relation needs a left neighbour, so k >= 1")
    q = rescaled_cascade_exponent(alpha, b)
    left = q * (log_frequency(b, k - 1) + log_frequency(b, k))
    right = 2 * Fraction(alpha) * (
        log_frequency(b, k) - log_frequency(b, k + 1)
    ) + 2 * q * log_frequency(b, k + 1)
    return left - right


def energy_identity_residual(
    amplitudes: Sequence[Fraction | int], weights: Sequence[Fraction | int]
) -> Fraction:
    """``sum_k X_k * (nonlinear part of X_k')``, which must vanish identically.

    With ``weights[k]`` standing for ``N_k^alpha``, the nonlinearity contributes

        sum_k  N_{k-1}^alpha X_{k-1} X_k^2  -  sum_k N_k^alpha X_k X_{k+1}^2 ,

    and reindexing the second sum by ``j = k + 1`` makes the two identical.  The
    cancellation is exact and term-by-term, which is why the model conserves
    energy for any choice of shells; it is the one structural property that must
    survive whatever ``N_k`` is chosen.
    """
    values = [Fraction(entry) for entry in amplitudes]
    scales = [Fraction(entry) for entry in weights]
    if len(scales) != len(values):
        raise ValueError("need one frequency weight per amplitude")
    total = Fraction(0)
    for k in range(len(values)):
        if k >= 1:
            total += scales[k - 1] * values[k - 1] * values[k] * values[k]
        if k + 1 < len(values):
            total -= scales[k] * values[k] * values[k + 1] * values[k + 1]
    return total


def truncation_residual(
    amplitudes: Sequence[Fraction | int], weights: Sequence[Fraction | int], cutoff: int
) -> list[Fraction]:
    """Velocity of the shells above ``cutoff`` when they start at zero.

    Palasek's Remark 1.10: if the data vanishes above some shell, the solution
    stays supported below it forever, so the blow-up is unstable in every ``C^s``
    -- truncating the data, which changes it by an arbitrarily small amount in
    any fixed norm, destroys the singularity.  Returned here as the exact vector
    field on the truncated subspace, which must be identically zero above the
    cutoff.
    """
    values = [Fraction(entry) for entry in amplitudes]
    scales = [Fraction(entry) for entry in weights]
    if len(scales) != len(values):
        raise ValueError("need one frequency weight per amplitude")
    if not 0 <= cutoff < len(values):
        raise ValueError("cutoff must index a shell")
    truncated = [value if index <= cutoff else Fraction(0) for index, value in enumerate(values)]
    out: list[Fraction] = []
    for k in range(cutoff + 1, len(truncated)):
        drift = Fraction(0)
        if k >= 1:
            drift += scales[k - 1] * truncated[k - 1] * truncated[k]
        if k + 1 < len(truncated):
            drift -= scales[k] * truncated[k + 1] * truncated[k + 1]
        out.append(drift)
    return out


# --------------------------------------------------------------------------- #
# The parameter geometry of the blow-up theorems
# --------------------------------------------------------------------------- #
def viscous_window(alpha: Fraction | int) -> tuple[Fraction, Fraction] | None:
    """Admissible shell separation ``b in (1, alpha/2)`` for viscous blow-up.

    Empty exactly when ``alpha <= 2``, which recovers the sharpness of the
    viscous theorem's hypothesis from the parameter constraints alone -- the
    paper obtains it independently, from energy criticality.  Returned as an open
    interval, or ``None`` when there is none.
    """
    upper = Fraction(alpha) / 2
    return (Fraction(1), upper) if upper > 1 else None


def inviscid_window(alpha: Fraction | int) -> tuple[Fraction, None]:
    """Admissible separation for inviscid blow-up: ``b > 1``, unbounded above."""
    if Fraction(alpha) < 1:
        raise ValueError("the inviscid theorem needs alpha >= 1")
    return (Fraction(1), None)


def beta_window(
    alpha: Fraction | int, b: Fraction | int, s: Fraction | int, *, viscous: bool = True
) -> tuple[Fraction, Fraction] | None:
    """The barrier exponent window, ``max(2b, alpha - s) < beta < alpha``.

    Inviscid drops the ``2b`` requirement to ``0``.  ``s > 0`` is the regularity
    index in which the solution becomes unbounded.
    """
    if Fraction(s) <= 0:
        raise ValueError("the blow-up norm index satisfies s > 0")
    floor = max(2 * Fraction(b), Fraction(alpha) - Fraction(s)) if viscous else max(
        Fraction(0), Fraction(alpha) - Fraction(s)
    )
    ceiling = Fraction(alpha)
    return (floor, ceiling) if floor < ceiling else None


def separation_budget(alpha: Fraction | int) -> Fraction | None:
    """How much super-exponential separation the viscous theorem allows.

    The supremum of ``b``, namely ``alpha/2``.  Over the three-dimensional
    candidate window ``alpha in (2, 5/2]`` this never exceeds ``5/4``: the shells
    ``N_k = N_0^{b^k}`` are super-exponential, but only barely.  The paper's case
    for embedding into the true nonlinearity rests on wide separation suppressing
    cross-scale errors, and this is the range where that margin is thinnest.
    """
    window = viscous_window(alpha)
    return window[1] if window else None


def clears_cascade(alpha: Fraction | int, b: Fraction | int) -> bool:
    """Whether the viscous trapping region sits above the cascade state.

    The barriers satisfy ``beta > 2b`` and the cascade state sits at
    ``2 alpha b / (2b + 1)``, so the region clears it as soon as

        2b  >=  2 alpha b / (2b + 1)   <=>   2b + 1 >= alpha .

    For ``b > 1`` this holds whenever ``alpha <= 3``, hence throughout the
    three-dimensional window.  The blow-up is an escape above the regularising
    cascade rather than a competition with it.
    """
    return 2 * Fraction(b) + 1 >= Fraction(alpha)
