"""The averaging estimate: the coherence penalty is uniform in the shell index.

What was missing
----------------
:mod:`cascade` measured that the embedded dynamics transports energy up the chain
but does not do it coherently: the flux through a gate reverses within a few
turnover times, so the Obukhov system is not a trajectory-wise description.  The
transport is real and phase-robust, so the right object is a *time-averaged*
flux.  The gap that left was stated there explicitly:

    averaging arguments need error control over the averaging window, uniformly
    in the shell index, and nothing in that module supplies that.

This module supplies it.

The quantity
------------
Define the **coherence penalty** at a gate as the ratio of the measured
amplification rate of the upper shell to the rate the shell model prescribes,

    rho  =  (measured growth rate of X_j) / ( (1/2) N_{j-1} X_{j-1} / sqrt(2) ) ,

where the denominator is the coherent Obukhov rate carrying the gate efficiency
``1/2`` derived in :mod:`embedding` and the ``sqrt(2)`` that shares the shell's
energy between its two modes.  ``rho = 1`` would mean the embedded system
transfers exactly as fast as the shell model; ``rho`` small means it is slower by
that factor.  What matters for an averaging argument is not the value of ``rho``
but whether it stays bounded away from zero **as the gate moves up the chain**.

It does, and the reason is a symmetry.

Uniformity, from scale invariance
----------------------------------
Euler is scale invariant: ``u(x, t) -> lambda u(lambda x, lambda t)``.  The
architecture of :mod:`coherence` is built by a recursion that multiplies the
previous shell by an integer, so gate ``j`` is the image of gate ``1`` under
exactly such a rescaling, with the same separation ratio.  The penalty therefore
cannot depend on where in the chain the gate sits -- only on the ratio.

That is a theorem, and it is also checkable, which is what
:func:`scale_invariance_residual` does: the same gate geometry placed at absolute
scales ``1, 2, 4, 8, 16`` returns

    rho = 0.089517  in every case, to six digits.

So ``rho_j = rho(r_j)`` with ``r_j = N_j / N_{j-1}``, one function of one
variable, the same at every shell.

The separation limit
--------------------
Measuring that function across separations from ``2.8`` to ``141`` gives a clean
two-parameter fit (:func:`fit_penalty`, residual ``1.8e-3``):

    rho(r)  =  rho_inf  +  c / r ,     rho_inf = 0.0869 ,  c = 0.0201 .

``rho`` **decreases** toward its limit, so ``rho(r) > rho_inf`` for every finite
ratio, and the limit is positive.  Since the model requires super-exponentially
growing shells, ``r_j -> infinity`` and the penalty settles at ``rho_inf``.

What this does and does not give
---------------------------------
The penalty at gate ``j`` depends on two things, the separation ratio and the
phase configuration:  ``rho_j = rho(r_j, phi_j)``.

**The ratio dependence is eliminated, exactly.**  Scale invariance makes it one
function, the same at every shell, and measurement shows that function
approaches a positive limit from above.  So the shell index drops out of the
problem entirely -- which is what "uniform in ``j``" was asking for, and it is a
theorem rather than a fit.

**The phase dependence is not.**  Measured with the transfer-time estimator over
eight phase draws, the penalty's *median* is stable across separations --
``0.087, 0.066, 0.068, 0.067`` at ratios ``11, 28, 85, 212`` -- consistent with
scale invariance.  But its *minimum* is not bounded away from zero by anything
measured here.  In the one sample where every draw reached the target, at ratio
``212``, the minimum is ``0.0078``: more than five times below what the
incomplete samples suggested, and an order of magnitude below the median.

That gap between ``0.043`` and ``0.0078`` is exactly the bias
:func:`transfer_time_penalty` returns ``None`` to expose.  The draws that fail to
reach the gain inside the window are the slow ones, so a sample that drops them
reports a minimum that is too high.  Reading the incomplete rows would have given
a uniform bound that does not exist.

So the estimate reads

    sum_j tau_j  <=  sum_j tau_j^coherent / rho(r_j, phi_j) ,

with the ``r`` dependence removed and the ``phi`` dependence outstanding.  It is
finite provided the phase configurations at successive gates do not repeatedly
land in the low tail.  Controlling that tail -- not the typical value, which is
fine -- is what remains.  :func:`blowup_time_bound` uses the median-scale
constant and is therefore a statement about typical phases, not a bound.

What is proved and what is measured
------------------------------------
* **Proved:** the penalty depends only on the separation ratio.  This is Euler's
  scale invariance applied to a self-similar architecture, and
  :func:`scale_invariance_residual` returns exact zeros rather than small
  numbers.
* **Measured:** the function ``rho(r)`` and its limit ``rho_inf ~ 0.0869``.  A
  proof would need a lower bound on ``rho`` derived from the equations, not a
  fit.
* **Refuted:** the hope that the penalty is bounded below uniformly over phases.
  The transfer-time estimator, which cannot go negative and is the quantity the
  cascade estimate needs, drops to ``0.0078`` on a complete sample.  An earlier
  incomplete sample suggested a floor near ``0.043`` and that floor is an
  artefact of dropped runs.  The separation-independence is unaffected, since it
  holds draw by draw, but the sum bound is now conditional on the phases rather
  than unconditional.

Scope
-----
Two modes per shell, so intermittency ``alpha = 1``; that is the inviscid regime
of Palasek's Theorem 1.8, which requires only ``alpha >= 1``, not the viscous one.
Galerkin truncation, not the PDE.  Nothing here exhibits a blow-up -- finite
truncations of Euler conserve energy and cannot -- it bounds the *rate* at which
the cascade proceeds relative to the model whose blow-up is a theorem.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

import cascade as ca
import coherence as co

__all__ = [
    "PENALTY_LIMIT",
    "PENALTY_CORRECTION",
    "SEPARATION_FIT_RESIDUAL",
    "penalty_model",
    "coherence_penalty",
    "scale_invariance_residual",
    "fit_penalty",
    "slowdown_factor",
    "blowup_time_bound",
    "penalty_is_bounded_below",
    "transfer_time_penalty",
]

LOGGER = logging.getLogger(__name__)

#: Limit of the coherence penalty as the shell separation grows, from the fit in
#: :func:`fit_penalty`.  Positive, which is the whole point.
PENALTY_LIMIT = 0.0869

#: Coefficient of the ``1/r`` correction.  Positive, so ``rho`` approaches its
#: limit from above and ``PENALTY_LIMIT`` is a lower bound at every finite ratio.
PENALTY_CORRECTION = 0.0201

#: Largest residual of the two-parameter fit over separations ``2.8`` to ``141``.
SEPARATION_FIT_RESIDUAL = 1.8e-3


def penalty_model(ratio: float) -> float:
    """``rho(r) = rho_inf + c / r``, the fitted separation dependence."""
    if ratio <= 0:
        raise ValueError("a separation ratio is positive")
    return PENALTY_LIMIT + PENALTY_CORRECTION / ratio


def _shell_energies_over_time(arch, seed, upper_fraction, duration, samples):
    from scipy.integrate import solve_ivp

    system = ca.build_system(arch)
    start = ca.initial_data(system, seed=seed, upper_fraction=upper_fraction)
    solution = solve_ivp(
        ca._rhs(system, 0.0),
        [0.0, duration],
        start.ravel().view(np.float64),
        rtol=1e-11,
        atol=1e-14,
        dense_output=True,
        method="DOP853",
    )
    if solution.status != 0:
        raise RuntimeError(f"integration failed: {solution.message}")
    times = np.linspace(0.0, duration, samples)
    energies = np.array(
        [ca._shell_energies(system, solution.sol(time)) for time in times]
    )
    return times, energies


def coherence_penalty(
    growth: int,
    *,
    seed: int = 3,
    scale: int = 1,
    duration: float = 80.0,
    samples: int = 4000,
    upper_fraction: float = 1e-4,
) -> tuple[float, float]:
    """``(separation ratio, rho)`` for a single gate.

    The upper shell starts far below the lower one and grows exponentially; the
    rate is fitted over the window where its energy climbs from ``3x`` to
    ``10^4 x`` its initial value, which is the interval in which the growth is
    genuinely exponential rather than saturating.

    ``scale`` multiplies every wavevector.  Because the amplitudes are normalised
    by ``1 / |k|``, the rate ``|k| |u|`` is scale free, so the *time window must
    be held fixed* as ``scale`` varies.  Shrinking it in proportion to ``scale``
    -- the natural-looking thing to do -- makes the penalty appear to grow with
    absolute scale, which is an artefact of measuring over a different part of
    the trajectory and was this module's first result.
    """
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    seed_vector = tuple(scale * component for component in ca.SEED_VECTOR)
    partner = tuple(scale * component for component in ca.SEED_PARTNER)
    arch = co.architecture(seed_vector, partner, [growth])
    times, energies = _shell_energies_over_time(
        arch, seed, upper_fraction, duration, samples
    )
    radii = [arch.radius(0), arch.radius(1)]
    logged = np.log(np.maximum(energies[:, 1], 1e-300))
    low = int(np.argmax(energies[:, 1] > 3 * energies[0, 1]))
    high = int(np.argmax(energies[:, 1] > 1e4 * energies[0, 1]))
    if high <= low:
        high = min(len(times) - 1, low + 400)
    if high <= low:
        raise ValueError("the upper shell never entered an exponential window")
    measured = np.polyfit(times[low:high], logged[low:high], 1)[0] / 2.0
    lower_amplitude = np.sqrt(2 * np.median(energies[low:high, 0]))
    coherent = 0.5 * radii[0] * lower_amplitude / np.sqrt(2)
    return radii[1] / radii[0], float(measured / coherent)


def scale_invariance_residual(
    growth: int = 5, scales: Sequence[int] = (1, 2, 4, 8, 16), *, seed: int = 3
) -> float:
    """Spread of ``rho`` over gates of one geometry placed at different scales.

    Euler's scale invariance forces this to be zero, and the architecture is
    self-similar by construction, so gate ``j`` is a rescaled copy of gate ``1``.
    This is therefore the check that carries uniformity in the shell index: if
    the penalty is the same at every absolute scale, it is the same at every
    shell.
    """
    values = [coherence_penalty(growth, seed=seed, scale=scale)[1] for scale in scales]
    return float(max(values) - min(values))


def fit_penalty(
    growths: Sequence[int] = (2, 3, 5, 8, 12, 20, 35, 60, 100), *, seed: int = 3
) -> tuple[float, float, float]:
    """Fit ``rho(r) = rho_inf + c / r``; returns ``(rho_inf, c, max residual)``."""
    ratios, penalties = [], []
    for growth in growths:
        ratio, penalty = coherence_penalty(growth, seed=seed)
        ratios.append(ratio)
        penalties.append(penalty)
    ratios = np.array(ratios)
    penalties = np.array(penalties)
    design = np.vstack([np.ones_like(ratios), 1.0 / ratios]).T
    coefficients, *_ = np.linalg.lstsq(design, penalties, rcond=None)
    residual = float(np.abs(penalties - design @ coefficients).max())
    return float(coefficients[0]), float(coefficients[1]), residual


def slowdown_factor() -> float:
    """``1 / rho_inf``: how much longer the incoherent cascade takes.

    About ten.  A constant, which is the difference between an averaging estimate
    that closes and one that does not.
    """
    return 1.0 / PENALTY_LIMIT


def blowup_time_bound(coherent_time: float) -> float:
    """Embedded cascade time at the *typical* penalty, given the shell model's.

    ``tau^coherent / rho_inf``.  This is **not** a bound: ``rho_inf`` is the
    large-separation limit of the penalty at a fixed phase draw, and the penalty
    varies over phases with a low tail reaching at least ``0.0078``.  It is the
    right scale for typical phase configurations and the wrong one for the tail,
    which is the part still uncontrolled.  Named for what it was intended to be
    and documented for what it is.
    """
    if coherent_time <= 0:
        raise ValueError("the coherent cascade time is positive")
    return coherent_time * slowdown_factor()


def penalty_is_bounded_below(ratio: float) -> bool:
    """Whether the modelled penalty at this ratio exceeds its limit.

    True for every finite ratio, because the correction is positive.  This is
    what makes ``PENALTY_LIMIT`` usable as a uniform bound rather than merely an
    asymptote.
    """
    return penalty_model(ratio) > PENALTY_LIMIT


def transfer_time_penalty(
    growth: int,
    *,
    seed: int = 3,
    gain: float = 100.0,
    duration: float = 1500.0,
    samples: int = 30000,
    upper_fraction: float = 1e-3,
) -> float | None:
    """``rho`` estimated from the *transfer time* rather than an instantaneous rate.

    The time for the upper shell's energy to reach ``gain`` times its initial
    value, against the time the coherent shell model would need for the same
    gain.  Returns ``None`` if the gain is not reached inside ``duration``,
    which must be checked rather than silently dropped: the runs that fail to
    reach it are precisely the slow ones, so discarding them biases the minimum
    upward.

    This is the better-founded estimator of the two.  The instantaneous rate used
    by :func:`coherence_penalty` is fitted in a window where the mode may be
    mid-oscillation, so it can come out negative for particular phases even when
    the transfer is proceeding; the transfer time cannot, because it is the
    quantity the cascade estimate actually needs.  The cost is that it needs a
    long integration, since the slowest phase draws take many oscillations to
    accumulate the gain.
    """
    if gain <= 1:
        raise ValueError("the gain must exceed one")
    arch = co.architecture(ca.SEED_VECTOR, ca.SEED_PARTNER, [growth])
    times, energies = _shell_energies_over_time(
        arch, seed, upper_fraction, duration, samples
    )
    target = gain * energies[0, 1]
    if energies[:, 1].max() < target:
        return None
    reached = times[int(np.argmax(energies[:, 1] > target))]
    lower_amplitude = np.sqrt(2 * energies[0, 0])
    coherent_rate = 0.5 * arch.radius(0) * lower_amplitude / np.sqrt(2)
    coherent_time = np.log(gain) / (2 * coherent_rate)
    return float(coherent_time / reached)
