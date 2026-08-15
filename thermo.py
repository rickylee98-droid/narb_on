"""What logic actually costs, and where the singularity really is.

The brief this answers proposes a "Goedel-Landauer-Prigogine trilemma": that a
physical computer cannot simultaneously be logically consistent, computationally
efficient and thermodynamically stable, and that heat output should diverge
exactly when the logic becomes complete.  No such conjecture exists in the
literature -- it is three names in a trenchcoat -- but underneath it there are
two real and separable effects, and conflating them is where the framing goes
wrong.  This module computes both exactly and reports which claims survive.

Verdict, up front
-----------------

    "erasing perfectly costs infinite heat"     FALSE.  Bounded by ln 2.
    "there is a dissipation singularity"        TRUE, twice, and neither is
                                                where the brief puts it.
    "it is a phase transition"                  TRUE of one, FALSE of the other.
    "the trilemma is the TUR"                   Partly.  The TUR bounds
                                                precision against dissipation
                                                but says nothing about logic.

What is computed
----------------

**One: maintenance is logarithmic, not singular.**  A bit held at error rate
``eps`` against a bath that randomises it at rate ``gamma`` is modelled as an
exactly solvable two-channel Markov jump process -- one channel the bath, one a
demon driven by an affinity ``A``.  Everything is closed form:

    eps  = (g_B + g_D e^{-A/2}) / (2 g_B + g_D e^{-A/2} + g_D e^{A/2})
    J    = p_0 k^D_{01} - p_1 k^D_{10}
    Sdot = |J| A

and in the reliable limit ``A -> infinity`` the flux saturates at the noise rate,
``|J| -> gamma``, while ``A -> 2 ln(1/eps) + const``, giving

    Sdot  ->  2 gamma ln(1/eps)              (`MAINTENANCE_COEFFICIENT` = 2)

So the cost of consistency diverges, but *logarithmically*, with a coefficient
that is exactly twice the noise rate.  That is a smooth trade-off with no
singular derivative structure -- nothing that deserves the word "transition".

**Two: erasure is bounded, so the headline claim is false.**  Erasing a bit to
residual error ``eps`` costs at most ``ln 2`` and *less* when ``eps > 0``:

    W(eps) = ln 2 - H(eps),      H(eps) = -eps ln eps - (1-eps) ln(1-eps)

which increases monotonically to ``ln 2`` as ``eps -> 0`` and never exceeds it.
Perfect erasure is finitely priced.  The divergence in part one is the cost of
*holding* a bit against noise, which is a different quantity entirely, and the
brief's error is to attribute the first to the second.

**Three: the real phase transition is the fault-tolerance threshold.**  Under
majority-vote concatenation the error rate obeys

    p' = 3 p^2 - 2 p^3

with fixed points ``0``, ``1/2``, ``1``.  The unstable one at ``p = 1/2`` is a
genuine critical point: below it the error falls doubly exponentially and the
gate count -- hence, by Landauer, the dissipation -- to reach any target ``delta``
is ``polylog(1/delta)`` with exponent ``log 3 / log 2``; above it no amount of
dissipation buys reliability at all.  Finite versus infinite, at a sharp
parameter value.  That is the dissipation singularity, and it sits at a *noise*
value, not at "logical completeness".

**Four: the TUR is the trilemma's honest core, and it is weaker than claimed.**
The thermodynamic uncertainty relation

    Var(J)/<J>^2 * Sigma  >=  2

is checked here against the exact scaled cumulant generating function of the
tilted generator, whose largest eigenvalue is available in closed form for a
two-state chain.  It is a genuine precision-versus-dissipation trade-off, which
is the defensible content of the "trilemma".  But it constrains the *current*,
not the logic: nothing in it refers to consistency, completeness, or
self-reference, and no Goedel statement enters at any point.

Novelty
-------

Nothing in this module is new mathematics.  Landauer (1961), Bennett (1982),
the finite-time Landauer corrections, the Barato-Seifert thermodynamic
uncertainty relation (2015) and its Gingrich-Horowitz-Perunov-England proof
(2016), and the Aharonov-Ben-Or / Knill-Laflamme-Zurek threshold theorem are all
classical.  The contribution is a computed answer to a badly posed conjecture:
which of its four claims survive, with exact numbers attached, and where the
singularity actually lives.  The one thing I have not seen assembled is the
side-by-side statement that the "trilemma" is two unrelated phenomena -- a
smooth logarithmic maintenance cost and a sharp threshold transition -- and that
the Goedel half contributes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import mpmath as mp

__all__ = [
    "DemonBit",
    "MAINTENANCE_COEFFICIENT",
    "maintenance_rate_law",
    "maintenance_law_residual",
    "binary_entropy",
    "erasure_work",
    "LANDAUER_CEILING",
    "erasure_cost_is_bounded",
    "TUR_BOUND",
    "majority_vote",
    "FAULT_TOLERANCE_THRESHOLD",
    "CRITICAL_MULTIPLIER",
    "critical_derivative",
    "critical_slowing_levels",
    "concatenation_levels",
    "gate_overhead",
    "OVERHEAD_EXPONENT",
    "dissipation_to_target",
    "dissipation_is_finite",
    "threshold_is_sharp",
    "PRECISION",
]

#: Working precision in decimal digits.  The cumulant derivatives are taken
#: numerically, so the arithmetic needs headroom the double type does not have.
PRECISION: int = 40

mp.mp.dps = PRECISION


# ---------------------------------------------------------------------------
# one: maintaining a bit against noise
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemonBit:
    """A two-state bit driven by a bath and a demon, solved exactly.

    Two channels connect the states.  The bath is unbiased and randomises at
    rate ``bath_rate``; the demon is driven by ``affinity`` (in units of
    ``k_B T``) and pushes the bit back toward zero.  The competition has a
    non-equilibrium steady state carrying a cycle current, and every quantity
    below is closed form.

    Attributes
    ----------
    bath_rate:
        Symmetric flip rate induced by the environment.  The noise.
    demon_rate:
        Bare attempt rate of the correcting channel.
    affinity:
        Thermodynamic force driving the demon, in units of ``k_B T``.  Zero
        means no correction and an error rate of one half.
    """

    bath_rate: float
    demon_rate: float
    affinity: float

    def __post_init__(self) -> None:
        if self.bath_rate <= 0:
            raise ValueError(f"bath rate must be positive, got {self.bath_rate}")
        if self.demon_rate <= 0:
            raise ValueError(f"demon rate must be positive, got {self.demon_rate}")
        if self.affinity < 0:
            raise ValueError(
                f"affinity must be non-negative, got {self.affinity}; "
                "a negative affinity is the same model with the states relabelled"
            )

    # -- rates ------------------------------------------------------------

    def _rates(self) -> tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
        """``(bath 0->1, bath 1->0, demon 0->1, demon 1->0)``."""
        bath = mp.mpf(self.bath_rate)
        demon = mp.mpf(self.demon_rate)
        affinity = mp.mpf(self.affinity)
        return (
            bath,
            bath,
            demon * mp.e ** (-affinity / 2),
            demon * mp.e ** (affinity / 2),
        )

    def steady_state(self) -> tuple[mp.mpf, mp.mpf]:
        """Stationary occupations ``(p_0, p_1)``."""
        bath_up, bath_down, demon_up, demon_down = self._rates()
        up = bath_up + demon_up
        down = bath_down + demon_down
        total = up + down
        return down / total, up / total

    def error_rate(self) -> mp.mpf:
        """Stationary probability of the wrong state."""
        return self.steady_state()[1]

    def cycle_flux(self) -> mp.mpf:
        """Net current through the demon channel.

        Negative in the reliable regime: the demon runs ``1 -> 0``, undoing what
        the bath does.  Its magnitude saturates at `bath_rate`, because the
        demon can only correct errors as fast as the bath makes them.
        """
        occupancy_zero, occupancy_one = self.steady_state()
        _, _, demon_up, demon_down = self._rates()
        return occupancy_zero * demon_up - occupancy_one * demon_down

    def entropy_production_rate(self) -> mp.mpf:
        """``|J| A``: dissipation per unit time, in units of ``k_B``."""
        return abs(self.cycle_flux()) * mp.mpf(self.affinity)

    # -- fluctuations -----------------------------------------------------

    def _scaled_cumulant(self, tilt: mp.mpf) -> mp.mpf:
        """Largest eigenvalue of the generator tilted by the demon current.

        For a two-state chain the tilted generator is ``2 x 2``, so the
        eigenvalue is the closed-form root of a quadratic and no iterative
        solver is involved.
        """
        bath_up, bath_down, demon_up, demon_down = self._rates()
        diagonal_zero = -(bath_up + demon_up)
        diagonal_one = -(bath_down + demon_down)
        off_up = bath_up + demon_up * mp.e**tilt
        off_down = bath_down + demon_down * mp.e ** (-tilt)
        trace = diagonal_zero + diagonal_one
        determinant = diagonal_zero * diagonal_one - off_up * off_down
        return (trace + mp.sqrt(trace**2 - 4 * determinant)) / 2

    def current_mean(self) -> mp.mpf:
        """First cumulant of the demon current, from the generating function."""
        return mp.diff(self._scaled_cumulant, mp.mpf(0), 1)

    def current_variance(self) -> mp.mpf:
        """Second cumulant: ``Var(J) / t`` in the long-time limit."""
        return mp.diff(self._scaled_cumulant, mp.mpf(0), 2)

    def tur_ratio(self) -> mp.mpf:
        """``Var(J)/<J>^2 * Sigma``, which the TUR bounds below by 2.

        The thermodynamic uncertainty relation is the defensible core of the
        "trilemma": precision in a current costs dissipation.  Note what it does
        *not* mention -- consistency, completeness, or self-reference.
        """
        mean = self.current_mean()
        if mean == 0:
            raise ZeroDivisionError(
                "current vanishes at zero affinity; the TUR ratio is undefined "
                "at equilibrium"
            )
        return self.current_variance() / mean**2 * self.entropy_production_rate()


#: In the reliable limit the demon flux saturates at the noise rate and the
#: affinity grows as ``2 ln(1/eps)``, so dissipation approaches
#: ``2 * gamma * ln(1/eps)``.  Logarithmic -- not singular.
MAINTENANCE_COEFFICIENT: int = 2


def maintenance_rate_law(bath_rate: float, error: float) -> float:
    """Asymptotic dissipation rate for holding a bit at error ``error``.

    ``2 gamma ln(1/eps)``.  The divergence as ``eps -> 0`` is real, and it is
    the slowest kind: halving the error adds a fixed increment, never a spike.
    """
    if bath_rate <= 0:
        raise ValueError(f"bath rate must be positive, got {bath_rate}")
    if not 0 < error < Fraction(1, 2):
        raise ValueError(f"error must lie in (0, 1/2), got {error}")
    return MAINTENANCE_COEFFICIENT * bath_rate * float(mp.log(1 / mp.mpf(error)))


def maintenance_law_residual(bit: DemonBit) -> float:
    """Relative gap between the exact dissipation rate and the asymptotic law.

    Goes to zero as the affinity grows, which is the check that the coefficient
    ``2`` is right rather than fitted.
    """
    exact = bit.entropy_production_rate()
    predicted = maintenance_rate_law(bit.bath_rate, float(bit.error_rate()))
    if predicted == 0:
        raise ZeroDivisionError("asymptotic law vanishes; residual is undefined")
    return float(abs(exact - predicted) / abs(predicted))


# ---------------------------------------------------------------------------
# two: erasure is bounded, so the headline claim fails
# ---------------------------------------------------------------------------


def binary_entropy(error: float) -> float:
    """``H(eps)`` in nats."""
    if not 0 <= error <= 1:
        raise ValueError(f"error must lie in [0, 1], got {error}")
    if error in (0, 1):
        return 0.0
    value = mp.mpf(error)
    return float(-value * mp.log(value) - (1 - value) * mp.log(1 - value))


#: ``ln 2``: the Landauer cost of erasing one bit perfectly, in units of
#: ``k_B T``.  A ceiling, not an asymptote to infinity.
LANDAUER_CEILING: float = float(mp.log(2))


def erasure_work(error: float) -> float:
    """Minimum work to erase a bit down to residual error ``error``.

    ``ln 2 - H(eps)``.  Monotonically increasing in reliability and capped at
    ``ln 2``, so the brief's claim that perfect logic dissipates infinite heat
    is false for erasure: perfect erasure is finitely priced.
    """
    if not 0 <= error <= Fraction(1, 2):
        raise ValueError(f"error must lie in [0, 1/2], got {error}")
    return LANDAUER_CEILING - binary_entropy(error)


def erasure_cost_is_bounded(samples: int = 64) -> bool:
    """Is the erasure cost bounded by ``ln 2`` all the way to zero error?

    The refutation, computed rather than asserted.
    """
    if samples < 2:
        raise ValueError(f"need at least two samples, got {samples}")
    previous = -mp.inf
    for index in range(samples):
        error = mp.mpf(1) / 2 ** (index + 1)
        work = erasure_work(float(error))
        if work > LANDAUER_CEILING:
            return False
        if work < previous:
            return False
        previous = work
    return True


#: The thermodynamic uncertainty relation's lower bound on
#: ``Var(J)/<J>^2 * Sigma``.
TUR_BOUND: int = 2


# ---------------------------------------------------------------------------
# three: the real phase transition
# ---------------------------------------------------------------------------


def majority_vote(error: Fraction) -> Fraction:
    """One level of three-bit majority-vote concatenation.

    ``p' = 3 p^2 - 2 p^3``: the encoded block fails when two or three of its
    three copies do.  Exact in rationals.
    """
    if not 0 <= error <= 1:
        raise ValueError(f"error must lie in [0, 1], got {error}")
    return 3 * error**2 - 2 * error**3


#: The unstable fixed point of `majority_vote`, and a genuine critical point.
#: Below it concatenation drives the error to zero doubly exponentially; above
#: it, to one.  This is where the dissipation singularity actually lives.
FAULT_TOLERANCE_THRESHOLD: Fraction = Fraction(1, 2)


#: ``d/dp (3 p^2 - 2 p^3) = 6 p (1 - p)``, evaluated at the threshold.  The
#: unstable eigenvalue of the critical fixed point: distance from threshold
#: grows by exactly this factor per level of concatenation.
CRITICAL_MULTIPLIER: Fraction = Fraction(3, 2)


def critical_derivative(error: Fraction) -> Fraction:
    """``6 p (1 - p)``: the multiplier of the majority-vote map."""
    if not 0 <= error <= 1:
        raise ValueError(f"error must lie in [0, 1], got {error}")
    return 6 * error * (1 - error)


def critical_slowing_levels(distance: float) -> float:
    """Levels needed to escape the neighbourhood of the threshold.

    Linearising at the critical point gives geometric growth of the distance
    with ratio `CRITICAL_MULTIPLIER`, so escaping from a starting distance
    ``delta`` takes ``ln(1/delta) / ln(3/2)`` levels.  This is critical slowing
    down: approaching the threshold, the overhead diverges logarithmically in
    the distance *before* the doubly exponential convergence sets in.
    """
    if not 0 < distance < FAULT_TOLERANCE_THRESHOLD:
        raise ValueError(f"distance must lie in (0, 1/2), got {distance}")
    return float(mp.log(1 / mp.mpf(distance)) / mp.log(mp.mpf(3) / 2))


def concatenation_levels(error: Fraction, target: Fraction, limit: int = 256) -> int:
    """Levels of concatenation needed to push ``error`` below ``target``.

    Raises if the error is at or above threshold, because then no finite number
    of levels suffices -- which is the content of the transition.

    The iteration runs in extended-precision floating point rather than exact
    rationals.  That is not laziness: each level squares the denominator, so an
    exact orbit near the threshold -- where critical slowing down demands many
    levels -- grows past any practical size.  `majority_vote` stays exact and is
    the referee; this is the orbit, and its precision is checked against the
    exact map in the tests.
    """
    if not 0 < error < 1:
        raise ValueError(f"error must lie in (0, 1), got {error}")
    if not 0 < target < 1:
        raise ValueError(f"target must lie in (0, 1), got {target}")
    if error >= FAULT_TOLERANCE_THRESHOLD:
        raise ValueError(
            f"error {error} is at or above the threshold "
            f"{FAULT_TOLERANCE_THRESHOLD}; no finite concatenation reaches "
            f"{target}"
        )
    current = mp.mpf(error.numerator) / error.denominator
    goal = mp.mpf(target.numerator) / target.denominator
    for level in range(limit + 1):
        if current < goal:
            return level
        current = 3 * current**2 - 2 * current**3
    raise ValueError(
        f"did not reach {target} from {error} within {limit} levels"
    )


#: ``log 3 / log 2``.  Three physical gates per logical gate per level, while the
#: error squares, so the overhead is polylogarithmic with this exponent.
OVERHEAD_EXPONENT: float = float(mp.log(3) / mp.log(2))


def gate_overhead(error: Fraction, target: Fraction, limit: int = 256) -> int:
    """Physical gates per logical gate to reach ``target``: ``3^levels``."""
    return 3 ** concatenation_levels(error, target, limit)


def dissipation_to_target(error: Fraction, target: Fraction, limit: int = 256) -> float:
    """Landauer cost per logical gate of reaching ``target``, in ``k_B T``.

    Each majority vote discards two of its three inputs, so a level costs two
    bit erasures per gate, and the cost is ``2 ln 2`` times the gate overhead.
    Finite below threshold and unreachable above it -- the singularity.
    """
    return 2 * LANDAUER_CEILING * gate_overhead(error, target, limit)


def dissipation_is_finite(error: Fraction, target: Fraction) -> bool:
    """Can any finite dissipation reach ``target`` from ``error``?"""
    try:
        dissipation_to_target(error, target)
    except ValueError:
        return False
    return True


def threshold_is_sharp(gap: Fraction = Fraction(1, 1000)) -> bool:
    """Finite just below the threshold, infinite just above. A phase transition.

    The contrast with part one is the whole point: maintenance cost diverges
    smoothly and logarithmically, while this one is finite on one side of a
    sharp parameter value and unattainable on the other.
    """
    if not 0 < gap < FAULT_TOLERANCE_THRESHOLD:
        raise ValueError(f"gap must lie in (0, 1/2), got {gap}")
    target = Fraction(1, 10**6)
    below = dissipation_is_finite(FAULT_TOLERANCE_THRESHOLD - gap, target)
    above = dissipation_is_finite(FAULT_TOLERANCE_THRESHOLD + gap, target)
    at = dissipation_is_finite(FAULT_TOLERANCE_THRESHOLD, target)
    return below and not above and not at
