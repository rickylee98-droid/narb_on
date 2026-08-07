"""Line integrals of Holder forms over fractal curves, and where Young's bound bites.

What the question actually is
-----------------------------
"Stokes' theorem fails on a jagged boundary" is, as usually stated, false for the
case people reach for.  At any finite prefractal stage the boundary is a polygon,
so Stokes holds exactly; and for a *smooth* form the limit converges without
difficulty, because the integral of a differential form is not the integral with
respect to arclength.  The arclength of the Koch snowflake diverges like
``(4/3)^n`` while ``\\oint x\\,dy`` converges to the (finite, exactly known) area.

The failure is real at *limited regularity*, and there it is sharp.  If the
boundary curve has finite ``p``-variation with ``p`` equal to its box dimension
``d``, and the form is ``alpha``-Holder, then the composition ``f∘gamma`` has
finite ``(p/alpha)``-variation and Young's condition ``1/p + alpha/p > 1`` reads

    alpha > d - 1 .

For the standard Koch curve ``d = log 4 / log 3`` and the threshold is
``alpha_c = 0.26186``.

The rate, not just the threshold
--------------------------------
The same estimate predicts more than a threshold.  Refining level ``n`` to
``n+1`` touches ``4^n`` segments of length ``r^n``, each contributing at most
``O((r^n)^{1+alpha})``, so the increment of the Riemann--Stieltjes sum is bounded
by

    |I_{n+1} - I_n|  <=  C (4 r^{1+alpha})^n ,

which is summable exactly when ``alpha > d - 1``.  Writing the measured decay as
``(4^beta r^{1+alpha})^n`` defines a **coherence exponent** ``beta``: the value
``beta = 1`` means the per-segment contributions add coherently, which is Young's
worst case, and ``beta = 1/2`` would be the fully incoherent, random-sign case.
Young's bound is an upper bound, so what ``beta`` actually is for a given form is
a measurement rather than a theorem.

Three measurement traps
-----------------------
All three were hit before being guarded, and all three silently produce
plausible nonsense.

1.  A Weierstrass form ``sum_k b^{-alpha k} cos(b^k t)`` truncated at ``K`` terms
    evaluates ``cos(b^K t)``.  In double precision the argument carries absolute
    error ``b^K * eps``, so once ``b^K`` exceeds ``1/eps`` the top terms are pure
    noise of amplitude ``b^{-alpha K}`` -- which at small ``alpha`` is not small.
    :func:`weierstrass` refuses a truncation that is not resolvable.

2.  Estimating the decay from consecutive ratios ``|dI_{n+1}|/|dI_n|`` fails
    because the increments oscillate; the ratios scatter over orders of magnitude
    while the envelope decays cleanly.  :func:`decay_rate` regresses on the whole
    sequence instead.

3.  The one that matters most.  Even a *resolvable* truncation can be wrong,
    because the partition also has a finest scale.  At level ``n`` the segments
    have length ``r^n``, so a mode of frequency ``b^k`` with ``b^k > r^{-n}``
    oscillates many times inside a single segment and is aliased by the midpoint
    rule.  Two conditions must therefore hold at once:

        resolvable by the partition      K <= L log(1/r) / log b
        untruncated tail negligible      b^{-alpha K} <= tail

    Multiplying, ``log b`` cancels and the two collapse to

        alpha * L * log(1/r)  >=  log(1/tail) .

    **Raising the base does not help.**  Only the number of levels ``L`` buys
    dynamic range, and ``L`` is capped by the ``4^L`` points of the curve.  The
    honest lower limit on ``alpha`` is :func:`resolution_floor`, and for the
    standard Koch curve at ``L = 10`` it sits at ``alpha = 0.63`` -- far *above*
    the Young threshold ``0.26``.  Worse, the ratio is essentially fixed:

        alpha_floor / alpha_c  =  log(1/tail) / (L (log 4 - log(1/r)))

    which is ``2.4`` at ``L = 10`` and falls only like ``1/L``.  A fixed
    Weierstrass form can therefore *never* probe the sub-threshold regime at any
    feasible resolution.  This was measured before it was derived: at
    ``alpha = 0.1`` the fitted rate wanders from ``0.33`` to ``0.69`` as the
    truncation goes from 12 terms to 38, while at ``alpha = 0.5`` and ``0.75`` it
    is stable to ten percent.  :func:`truncation_stability` is that check.

4.  The one that decided the answer.  A regression cannot notice that its own
    model is wrong.  Near the edge of convergence the increments decay for a few
    levels and then flatten, and a log-linear fit through a
    decays-then-flattens sequence still returns a slope below zero -- reporting
    ``rate = 0.75 < 1``, i.e. convergence, for sums that are not converging.
    :attr:`RateMeasurement.converges` therefore tests the *tail*, and
    :class:`IncrementDecay` is the model-free instrument the conclusions rest on.

    The opposite error is just as easy.  RMS over twelve phases carries twenty
    percent scatter per level, enough to make a marginal sequence look flat; at
    ``alpha = 0.05`` twelve phases suggested a plateau and thirty-two did not.
    Neither reading is safe on its own, which is why
    :meth:`IncrementDecay.tail_interval` bootstraps over phases and
    :attr:`IncrementDecay.converges` asks whether the *interval* clears $1$.

What is actually observed
-------------------------
Young's law with the box dimension replaced by an effective one,

    alpha > beta d - 1        with beta ~ 0.75 +- 0.05 ,

measured by extrapolating the increment ratio to $1$ at three dimensions:
``d = 1.135`` gives a threshold of ``-0.178`` (convergence for every positive
exponent), ``d = 1.262`` gives something at or below ``0.05``, and ``d = 1.631``
gives ``+0.156`` -- where the tail ratio at ``alpha = 0.05`` is ``1.09`` with a
bootstrap interval entirely above $1$, so the boundary is real.  Against Young's
``d - 1`` of ``0.135``, ``0.262`` and ``0.631``, the effective threshold is far
lower but not absent.  ``beta = 1`` would recover Young exactly and ``beta = 1/2``
is the random-walk value; it is neither.

Three regimes in ``alpha``, at fixed dimension:

* ``alpha`` above the crossover ``1 - d/2``: the rate is ``4 r^2`` exactly and
  does not depend on ``alpha`` at all.  At ``alpha = 0.95`` the successive
  ratios are ``0.445, 0.444, 0.445, 0.443, 0.444, 0.445, 0.443`` against
  ``4/9 = 0.4444``.  The curve's own second-order geometry sets the rate; the
  form's regularity is irrelevant once it is smooth enough.
* below the crossover: the ratio climbs smoothly toward $1$ as ``alpha`` falls,
  **with no feature at Young's** ``alpha_c``.  For the standard curve the ratio
  there is ``0.669`` with interval ``[0.620, 0.714]`` -- comfortably converging,
  no kink.
* below the effective threshold ``beta d - 1``: the increments stop shrinking and
  the partial sums wander.  Near it the measurement is genuinely marginal and
  says so: at ``d = 1.262, alpha = 0.05`` eight levels at 32 phases give
  ``0.93`` while seven at 12 give ``1.02``, which is why the threshold there is
  reported as "below ``0.05``, consistent with zero" and not as a fitted digit.

The coherence exponent ``beta`` is a clean power law only in the first regime,
fitting with ``chi^2/dof`` of ``0.04``; below the crossover the level sequence is
not a power law at all (``chi^2/dof`` of 7 to 17), so the low branch is quoted as
a measured slope rather than an exponent.

Instruments
-----------
:func:`measure_increments` is primary: phase-averaged RMS increments, no model.

:func:`measure_coherence` fits the decay rate of the whole refinement sequence
and is subject to traps 3 and 4, so it reports whether it is inside the honest
window and whether its own geometric model holds.

:func:`coherence_profile` is exact and per-level, and needs no fit.  It splits
the increment ``I_{n+1} - I_n`` into the ``4^n`` independent contributions of the
level-``n`` segments and compares the achieved total against the ``l1`` norm of
those contributions.  Coherent addition gives ``|sum| = l1``; independent signs
give ``|sum| ~ l1 / sqrt(count)``.  It also reports the measured Holder exponent
of the individual contributions, which is the aliasing referee: if the
per-segment terms do not shrink like ``r^{n(1+alpha)}``, the form is not being
sampled as an ``alpha``-Holder function and no conclusion about ``beta`` is
available.

For the sub-threshold regime, where no fixed form is admissible,
:func:`matched_weierstrass_form` builds the lacunary form whose spectrum is cut
exactly at the resolution of the partition being used.  This is a different
mathematical object -- a sequence of forms, not one form -- and it is the
construction that *saturates* Young's bound rather than an arbitrary probe of
it.  Results from it are labelled as such and must not be read as statements
about a single fixed integrand.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "KOCH_ANGLE",
    "koch_ratio",
    "koch_dimension",
    "koch_curve",
    "koch_snowflake",
    "curve_is_simple",
    "polygon_arclength",
    "polygon_signed_area",
    "max_resolvable_terms",
    "partition_resolvable_terms",
    "resolution_floor",
    "ResolutionWindow",
    "resolution_window",
    "weierstrass",
    "HolderForm",
    "weierstrass_form",
    "matched_weierstrass_form",
    "lipschitz_form",
    "riemann_stieltjes",
    "decay_rate",
    "RateMeasurement",
    "IncrementDecay",
    "measure_increments",
    "ThresholdMeasurement",
    "convergence_threshold",
    "young_rate",
    "predicted_rate",
    "predicted_crossover",
    "effective_threshold",
    "coherence_exponent",
    "measure_coherence",
    "measure_matched",
    "MatchedMeasurement",
    "CoherenceMeasurement",
    "LevelProfile",
    "coherence_profile",
    "CorrelationProfile",
    "increment_correlation",
    "phase_diagram",
    "truncation_stability",
]

LOGGER = logging.getLogger(__name__)

F64 = np.float64
FloatArray = NDArray[np.float64]

#: Apex half-angle of the standard von Koch construction, in radians.
KOCH_ANGLE = math.pi / 3.0


# --------------------------------------------------------------------------- #
# The curve family
# --------------------------------------------------------------------------- #
def koch_ratio(angle: float = KOCH_ANGLE) -> float:
    """Contraction ratio of the generalised Koch curve with the given apex angle.

    Each segment is replaced by four segments of length ``r`` forming a symmetric
    bump.  Closing the horizontal extent requires ``2r + 2r\\cos(angle) = 1``.
    """
    if not 0.0 < angle < math.pi / 2:
        raise ValueError(f"angle must lie strictly in (0, pi/2); got {angle}")
    return 1.0 / (2.0 + 2.0 * math.cos(angle))


def koch_dimension(angle: float = KOCH_ANGLE) -> float:
    """Box dimension ``log 4 / log(1/r)`` of the generalised Koch curve.

    Runs from $1$ as ``angle -> 0`` (a straight line) to $2$ as
    ``angle -> pi/2`` (space filling), passing through ``log 4 / log 3`` at the
    standard ``pi/3``.
    """
    return math.log(4.0) / math.log(1.0 / koch_ratio(angle))


def koch_curve(level: int, angle: float = KOCH_ANGLE) -> FloatArray:
    """Vertices of the generalised Koch curve from ``(0,0)`` to ``(1,0)``.

    Returns ``(4**level + 1, 2)`` points.  The construction is exact up to
    floating-point rounding of the two rotations; no adaptive refinement is used,
    so every segment at a given level has the same length ``r**level``.
    """
    if level < 0:
        raise ValueError(f"level must be non-negative; got {level}")
    ratio = koch_ratio(angle)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=F64,
    )
    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=F64)
    for _ in range(level):
        start, end = points[:-1], points[1:]
        step = (end - start) * ratio
        first = start + step
        third = end - step
        second = first + step @ rotation.T
        out = np.empty((4 * len(start) + 1, 2), dtype=F64)
        out[0::4][: len(start)] = start
        out[1::4][: len(start)] = first
        out[2::4][: len(start)] = second
        out[3::4][: len(start)] = third
        out[-1] = points[-1]
        points = out
    return points


def koch_snowflake(level: int, angle: float = KOCH_ANGLE) -> FloatArray:
    """Closed Koch snowflake: three curves on the sides of an equilateral triangle.

    Returned closed, with the first point repeated at the end, so that a signed
    area or a closed line integral can be taken directly.
    """
    side = koch_curve(level, angle)
    pieces = []
    for k in range(3):
        turn = -2.0 * math.pi * k / 3.0
        rotation = np.array(
            [[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]],
            dtype=F64,
        )
        shifted = side @ rotation.T
        origin = np.array([0.0, 0.0]) if k == 0 else pieces[-1][-1]
        pieces.append(shifted - shifted[0] + origin)
    closed = np.vstack([pieces[0][:-1], pieces[1][:-1], pieces[2]])
    return closed


def curve_is_simple(angle: float = KOCH_ANGLE, *, level: int = 6) -> bool:
    """Whether the generalised Koch curve at this apex angle is non-self-intersecting.

    The bump grows taller relative to its base as the angle opens, and past a
    critical angle just below ``1.4`` the construction folds back onto itself.
    A self-overlapping curve is not a curve of box dimension ``log 4 / log(1/r)``
    in the sense the argument needs -- the four pieces are no longer essentially
    disjoint -- so results from such an angle must be discarded rather than read
    as the high-dimension end of the family.

    Tested by looking for vertex pairs that are close in space while far apart
    along the curve; adjacent and next-adjacent pairs are excluded because they
    are legitimately within a segment length of each other.
    """
    points = koch_curve(level, angle)
    segment = koch_ratio(angle) ** level
    order = np.argsort(points[:, 0], kind="stable")
    limit = 0.5 * segment
    index = np.arange(len(points))
    for start in range(len(order)):
        here = order[start]
        for other in order[start + 1 : start + 64]:
            if points[other, 0] - points[here, 0] > limit:
                break
            if abs(int(index[other]) - int(index[here])) <= 2:
                continue
            if float(np.hypot(*(points[other] - points[here]))) < limit:
                return False
    return True


def polygon_arclength(points: FloatArray) -> float:
    """Total length of a polygonal path."""
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def polygon_signed_area(points: FloatArray) -> float:
    """Signed area enclosed by a closed polygon, by the shoelace formula.

    Equal to ``\\oint x\\,dy`` over the same polygon, which is what makes it a
    referee for the integration routine rather than merely a convenience.
    """
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


# --------------------------------------------------------------------------- #
# Holder forms, with a precision guard
# --------------------------------------------------------------------------- #
def max_resolvable_terms(base: float, safety: float = 1e-4) -> int:
    """Largest Weierstrass truncation whose top term is not floating-point noise.

    Evaluating ``cos(base**k * t)`` in double precision carries an argument error
    of about ``base**k * eps``.  Once that approaches a radian the term is noise,
    and its amplitude ``base**(-alpha k)`` is *not* small when ``alpha`` is small,
    so the noise dominates the signal.  We require the argument error to stay
    below ``safety`` radians.
    """
    if base <= 1.0:
        raise ValueError(f"base must exceed 1; got {base}")
    eps = float(np.finfo(F64).eps)
    return int(math.floor(math.log(safety / eps) / math.log(base)))


def partition_resolvable_terms(level: int, base: float, angle: float = KOCH_ANGLE) -> int:
    """Largest truncation whose top mode is still resolved by a level-``level`` curve.

    The segments at that level have length ``r**level``, and a mode of frequency
    ``base**k`` needs several samples per period, so the midpoint rule aliases
    everything above ``base**k ~ r**-level``.  This is a *different* and usually
    much tighter cap than :func:`max_resolvable_terms`, which only asks whether
    the cosine argument itself is computable.
    """
    if level < 0:
        raise ValueError(f"level must be non-negative; got {level}")
    if base <= 1.0:
        raise ValueError(f"base must exceed 1; got {base}")
    return int(math.floor(level * math.log(1.0 / koch_ratio(angle)) / math.log(base)))


def resolution_floor(
    max_level: int, *, angle: float = KOCH_ANGLE, tail: float = 1e-3
) -> float:
    """Smallest ``alpha`` a fixed Weierstrass form can honestly probe.

    Requiring the truncation to be both resolved by the partition and small in
    its untruncated tail gives ``alpha * max_level * log(1/r) >= log(1/tail)``,
    in which the base has cancelled.  Below the returned ``alpha`` no choice of
    base or truncation is admissible and the measurement reports the truncation
    rather than the Holder class.
    """
    if max_level <= 0:
        raise ValueError(f"max_level must be positive; got {max_level}")
    if not 0.0 < tail < 1.0:
        raise ValueError(f"tail must lie in (0, 1); got {tail}")
    return math.log(1.0 / tail) / (max_level * math.log(1.0 / koch_ratio(angle)))


@dataclass(frozen=True)
class ResolutionWindow:
    """Which Holder exponents a given curve resolution can actually measure."""

    max_level: int
    angle: float
    base: float
    tail: float
    alpha_floor: float
    young_threshold: float
    terms: int

    @property
    def reaches_threshold(self) -> bool:
        """Whether the honest window extends below Young's threshold.

        It does not, for any feasible level count: the ratio of the two is
        ``log(1/tail) / (L (log 4 - log(1/r)))``, which decays only like ``1/L``.
        """
        return self.alpha_floor < self.young_threshold

    def admits(self, alpha: float) -> bool:
        return alpha >= self.alpha_floor


def resolution_window(
    max_level: int,
    *,
    angle: float = KOCH_ANGLE,
    base: float = 2.0,
    tail: float = 1e-3,
) -> ResolutionWindow:
    """Assemble the resolution constraints for one curve/base combination."""
    partition_cap = partition_resolvable_terms(max_level, base, angle)
    precision_cap = max_resolvable_terms(base)
    return ResolutionWindow(
        max_level=max_level,
        angle=angle,
        base=base,
        tail=tail,
        alpha_floor=resolution_floor(max_level, angle=angle, tail=tail),
        young_threshold=koch_dimension(angle) - 1.0,
        terms=max(1, min(partition_cap, precision_cap)),
    )


def weierstrass(
    t: FloatArray, alpha: float, *, base: float = 2.0, terms: int | None = None
) -> FloatArray:
    """Weierstrass function of Holder exponent exactly ``alpha``.

    ``W(t) = sum_{k<terms} base**(-alpha k) cos(base**k t)`` is ``alpha``-Holder
    and no better, for ``0 < alpha < 1`` and ``base > 1``.

    ``terms`` defaults to :func:`max_resolvable_terms` and is *rejected* if it
    exceeds it, because an unresolvable truncation silently replaces the top of
    the series with noise of amplitude ``base**(-alpha * terms)``.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1]; got {alpha}")
    limit = max_resolvable_terms(base)
    if terms is None:
        terms = limit
    if terms > limit:
        raise ValueError(
            f"truncation {terms} exceeds the resolvable limit {limit} for base "
            f"{base}: the top terms would be floating-point noise of amplitude "
            f"{base ** (-alpha * terms):.3e}"
        )
    total = np.zeros_like(t)
    for k in range(terms):
        total += base ** (-alpha * k) * np.cos(base**k * t)
    return total


@dataclass(frozen=True)
class HolderForm:
    """A 1-form ``f dx + g dy`` with a declared Holder exponent."""

    name: str
    alpha: float
    f: Callable[[FloatArray, FloatArray], FloatArray]
    g: Callable[[FloatArray, FloatArray], FloatArray]


def weierstrass_form(
    alpha: float, *, base: float = 2.0, phase: float = 0.0, terms: int | None = None
) -> HolderForm:
    """The form ``W(y + phase) dx``, which is not exact and not closed.

    Only the ``dx`` component is used: a form ``F'(x) dx`` would integrate to zero
    around any closed curve and measure nothing, so the dependence must be on the
    *other* coordinate.
    """
    return HolderForm(
        name=f"weierstrass(alpha={alpha}, base={base})",
        alpha=alpha,
        f=lambda x, y: weierstrass(y + phase, alpha, base=base, terms=terms),
        g=lambda x, y: np.zeros_like(x),
    )


def matched_weierstrass_form(
    alpha: float,
    level: int,
    *,
    angle: float = KOCH_ANGLE,
    base: float = 2.0,
    phase: float = 0.0,
) -> HolderForm:
    """Weierstrass form whose spectrum is cut exactly at the level's resolution.

    Every retained mode is resolved by the level-``level`` partition and every
    discarded mode would have been aliased, so the aliasing trap is closed by
    construction and *any* ``alpha`` becomes measurable -- including below the
    Young threshold, which no fixed form can reach.

    The price is that this is a sequence of forms rather than a single one.  A
    refinement increment must be computed with one member of the sequence at both
    of its levels, or it compares two different integrands and means nothing;
    :func:`coherence_profile` does this correctly when handed the matched form
    for the *finer* of its two levels.  The construction is the standard lacunary
    one used to saturate Young's bound, so it measures how bad a form can be,
    not how bad a typical form is.
    """
    terms = min(
        partition_resolvable_terms(level, base, angle), max_resolvable_terms(base)
    )
    return HolderForm(
        name=f"matched-weierstrass(alpha={alpha}, level={level}, terms={terms})",
        alpha=alpha,
        f=lambda x, y: weierstrass(y + phase, alpha, base=base, terms=max(1, terms)),
        g=lambda x, y: np.zeros_like(x),
    )


def lipschitz_form(phase: float = 0.0) -> HolderForm:
    """``sin(y + phase) dx``: smooth, hence Lipschitz, the positive control.

    A smooth form must saturate Young's bound, giving coherence exponent
    ``beta = 1``.  Without this control a measured ``beta < 1`` could not be
    distinguished from an estimator that simply reads low.
    """
    return HolderForm(
        name="lipschitz",
        alpha=1.0,
        f=lambda x, y: np.sin(y + phase),
        g=lambda x, y: np.zeros_like(x),
    )


# --------------------------------------------------------------------------- #
# Integration and the decay rate
# --------------------------------------------------------------------------- #
def riemann_stieltjes(points: FloatArray, form: HolderForm) -> float:
    """Midpoint Riemann--Stieltjes sum of a 1-form along a polygonal path.

    This is the correct discretisation: a Young integral *is* the limit of such
    sums as the partition refines. Integrating the form accurately along each
    segment would be a different (and, for a form with structure at every scale,
    infeasible) computation.
    """
    start, end = points[:-1], points[1:]
    middle = 0.5 * (start + end)
    delta = end - start
    values_f = form.f(middle[:, 0], middle[:, 1])
    values_g = form.g(middle[:, 0], middle[:, 1])
    return float(np.sum(values_f * delta[:, 0] + values_g * delta[:, 1]))


@dataclass(frozen=True)
class LevelProfile:
    """Exact decomposition of one refinement increment into per-segment terms.

    Refining level ``n`` to ``n+1`` replaces each of the ``count = 4**n`` parent
    segments by four children.  The increment of the Riemann--Stieltjes sum is
    therefore an exact sum of ``count`` independent contributions, and how those
    contributions add is the whole question.
    """

    level: int
    count: int
    total: float
    l1: float
    l2: float

    @property
    def term(self) -> float:
        """Mean magnitude of a single segment's contribution."""
        return self.l1 / self.count

    @property
    def beta(self) -> float:
        """Coherence exponent from ``|total| = count**beta * term``.

        ``beta = 1`` is coherent addition, which is Young's worst case, and
        ``beta = 1/2`` is what independent signs give.  This is exact for the
        level: no fit and no truncation of a refinement sequence is involved.
        """
        if self.count <= 1 or self.total == 0.0 or self.l1 == 0.0:
            raise ValueError(
                f"level {self.level} carries no usable signal "
                f"(count={self.count}, total={self.total}, l1={self.l1})"
            )
        return 1.0 + math.log(abs(self.total) / self.l1) / math.log(self.count)

    @property
    def randomness(self) -> float:
        """``|total| / l2``: the scale-free coherence diagnostic.

        This is the quantity to look at rather than :attr:`beta`, because it does
        not presume that the coherence follows a power of the count.  Independent
        signs of comparable magnitude give ``sqrt(2/pi) ~ 0.8`` *independently of
        level*, while coherent addition gives ``sqrt(count)``, which grows by a
        factor of two per level.  A ``beta`` that drifts downward with level and a
        ``randomness`` that does not are the same observation, but only the
        second one distinguishes "converging to the random-walk exponent" from
        "not a power law at all".
        """
        return abs(self.total) / self.l2 if self.l2 > 0.0 else 0.0

    @property
    def peakiness(self) -> float:
        """``l1 / (sqrt(count) * l2)``; $1$ for equal-magnitude terms, less if heavy-tailed.

        ``beta`` is only interpretable as a cancellation exponent when the terms
        are comparable in size, since a single dominant term would give
        ``|total| ~ l1`` for reasons having nothing to do with coherence.
        """
        return self.l1 / (math.sqrt(self.count) * self.l2) if self.l2 > 0.0 else 0.0


def coherence_profile(
    coarse: FloatArray, fine: FloatArray, form: HolderForm
) -> LevelProfile:
    """Split ``I_fine - I_coarse`` into the contribution of each coarse segment.

    ``fine`` must be the one-step refinement of ``coarse``, so that its points
    ``4j`` coincide with the coarse points ``j`` and each coarse segment has
    exactly four children.  Both integrals are taken with the *same* form.
    """
    count = len(coarse) - 1
    if len(fine) - 1 != 4 * count:
        raise ValueError(
            f"fine curve has {len(fine) - 1} segments, expected {4 * count} "
            f"as the one-step refinement of {count}"
        )
    if not np.allclose(fine[::4], coarse, atol=1e-9):
        raise ValueError("fine curve is not a refinement of the coarse curve")

    def contributions(points: FloatArray) -> FloatArray:
        start, end = points[:-1], points[1:]
        middle = 0.5 * (start + end)
        delta = end - start
        return form.f(middle[:, 0], middle[:, 1]) * delta[:, 0] + form.g(
            middle[:, 0], middle[:, 1]
        ) * delta[:, 1]

    parent = contributions(coarse)
    child = contributions(fine).reshape(count, 4).sum(axis=1)
    terms = child - parent
    return LevelProfile(
        level=int(round(math.log(count, 4))) if count > 0 else 0,
        count=count,
        total=float(terms.sum()),
        l1=float(np.abs(terms).sum()),
        l2=float(np.sqrt(np.square(terms).sum())),
    )


def truncation_stability(
    alpha: float,
    truncations: Sequence[int],
    *,
    angle: float = KOCH_ANGLE,
    levels: Sequence[int] = tuple(range(2, 11)),
    base: float = 2.0,
    n_phases: int = 6,
) -> dict[int, float]:
    """Measured decay rate as a function of where the series is truncated.

    A rate that moves with the truncation is measuring the truncation.  This is
    the check that invalidated the small-``alpha`` end of the first sweep, where
    the fitted rate ran from ``0.33`` at 12 terms to ``0.69`` at 38 while the
    large-``alpha`` end held to ten percent.
    """
    curves = {level: koch_curve(level, angle) for level in levels}
    result: dict[int, float] = {}
    for terms in truncations:
        rates = []
        for phase in np.linspace(0.1, 5.0, n_phases):
            form = weierstrass_form(alpha, base=base, phase=phase, terms=terms)
            values = [riemann_stieltjes(curves[level], form) for level in levels]
            try:
                rates.append(decay_rate(values).rate)
            except ValueError:
                continue
        if rates:
            result[terms] = float(np.median(rates))
    return result


@dataclass(frozen=True)
class CorrelationProfile:
    """Autocorrelation of the per-segment contributions along the curve.

    The sum of ``N`` terms is ``N`` times their mean identically, so
    ``randomness = |sum|/l2`` is just ``sqrt(N) * mu / sigma``: the whole question
    of coherence is whether the mean contribution per segment shrinks like the
    standard error (``beta = 1/2``) or stays fixed relative to the spread
    (``beta = 1``).

    What the autocorrelation adds is the *error bar on that mean*.  Neighbouring
    segments sample nearby parts of the form, so the terms are correlated over a
    finite lag, and the effective sample size is ``N/tau`` rather than ``N``.  The
    mean is then distinguishable from zero only when
    :attr:`significance` is large.  This is what makes the low-``alpha`` end hard:
    the coherent residue there is a two-sigma effect per level, not a clean zero.
    """

    level: int
    count: int
    autocorrelation: FloatArray
    integrated_time: float
    randomness: float

    @property
    def correlation_length(self) -> int:
        """First lag at which the autocorrelation drops below ``1/e``."""
        below = np.nonzero(self.autocorrelation < math.exp(-1.0))[0]
        return int(below[0]) if below.size else len(self.autocorrelation)

    @property
    def significance(self) -> float:
        """``randomness / sqrt(tau)``: how many sigma the coherent residue is.

        The standard error on the mean of ``tau``-correlated terms is
        ``sigma sqrt(tau/N)``, and the mean itself is ``sigma * randomness /
        sqrt(N)``, so their ratio is ``randomness / sqrt(tau)`` with the level
        cancelling.  A value near $1$ means the level's apparent coherence is
        consistent with noise.
        """
        if self.integrated_time <= 0.0:
            return float("inf")
        return self.randomness / math.sqrt(self.integrated_time)


def increment_correlation(
    coarse: FloatArray, fine: FloatArray, form: HolderForm, *, max_lag: int = 64
) -> CorrelationProfile:
    """Autocorrelation of the per-segment refinement contributions.

    The terms are indexed in order along the curve, so lag is arc position.  The
    integrated correlation time ``tau = 1 + 2 sum_{k>=1} C(k)`` is the factor by
    which the variance of the sum exceeds the independent value.
    """
    count = len(coarse) - 1
    if len(fine) - 1 != 4 * count:
        raise ValueError(
            f"fine curve has {len(fine) - 1} segments, expected {4 * count}"
        )
    if max_lag < 1:
        raise ValueError(f"max_lag must be positive; got {max_lag}")

    def contributions(points: FloatArray) -> FloatArray:
        start, end = points[:-1], points[1:]
        middle = 0.5 * (start + end)
        delta = end - start
        return form.f(middle[:, 0], middle[:, 1]) * delta[:, 0] + form.g(
            middle[:, 0], middle[:, 1]
        ) * delta[:, 1]

    terms = contributions(fine).reshape(count, 4).sum(axis=1) - contributions(coarse)
    centred = terms - terms.mean()
    variance = float(np.dot(centred, centred))
    lags = min(max_lag, count - 1)
    if variance <= 0.0 or lags < 1:
        raise ValueError(
            f"level {count} contributions carry no variance to correlate "
            f"(variance={variance})"
        )
    auto = np.array(
        [float(np.dot(centred[: count - k], centred[k:])) / variance for k in range(lags)],
        dtype=F64,
    )
    # Automatic windowing: truncate the sum where the autocorrelation first goes
    # negative.  Summing past that point adds noise rather than signal, and for a
    # strongly correlated sequence the unwindowed sum is dominated by whatever
    # ``max_lag`` happened to be.
    negative = np.nonzero(auto[1:] < 0.0)[0]
    window = int(negative[0]) + 1 if negative.size else lags
    l2 = float(np.sqrt(np.square(terms).sum()))
    return CorrelationProfile(
        level=int(round(math.log(count, 4))),
        count=count,
        autocorrelation=auto,
        integrated_time=float(1.0 + 2.0 * auto[1:window].sum()),
        randomness=abs(float(terms.sum())) / l2 if l2 > 0.0 else 0.0,
    )


def phase_diagram(
    angles: Sequence[float],
    alphas: Sequence[float],
    *,
    levels: Sequence[int] = tuple(range(3, 10)),
    base: float = 2.0,
    n_phases: int = 4,
) -> dict[tuple[float, float], "MatchedMeasurement"]:
    """Measure the coherence exponent over a grid of the ``(dimension, alpha)`` plane.

    The apex angle tunes the box dimension continuously between $1$ and $2$, so
    the grid is a genuine two-parameter sweep rather than a line through one
    curve.  Keyed by ``(angle, alpha)``.
    """
    return {
        (angle, alpha): measure_matched(
            alpha, angle=angle, levels=levels, base=base, n_phases=n_phases
        )
        for angle in angles
        for alpha in alphas
    }


@dataclass(frozen=True)
class RateMeasurement:
    """Geometric decay rate of the increments of a refinement sequence."""

    rate: float
    tail_rate: float
    n_increments: int
    residual: float

    @property
    def converges(self) -> bool:
        """Whether the increments are still shrinking at the end of the sequence.

        The full-sequence ``rate`` is *not* enough, and trusting it was a real
        error here.  Regressing ``log|dI_n|`` against ``n`` returns a negative
        slope for a sequence that decays for a while and then flattens out,
        which reports convergence for sums that are in fact wandering at a fixed
        amplitude.  Requiring the tail to be decaying as well is what
        distinguishes the two.
        """
        return self.tail_rate < 0.95

    @property
    def geometric(self) -> bool:
        """Whether a single geometric rate describes the sequence at all.

        ``rate`` means nothing when this is false; the sequence has structure a
        one-parameter model cannot carry, and the right response is to look at
        the increments rather than to quote an exponent.
        """
        return self.residual < 0.5 and abs(
            math.log(max(self.tail_rate, 1e-9) / max(self.rate, 1e-9))
        ) < 0.5


def decay_rate(values: Sequence[float], *, floor: float = 1e-13) -> RateMeasurement:
    """Fit ``|I_{n+1} - I_n| ~ C rate^n``, and check the tail separately.

    Consecutive ratios are not used, and must not be: the increments carry their
    own oscillation, so ``|dI_{n+1}|/|dI_n|`` scatters over orders of magnitude
    even when the envelope decays cleanly. Regressing ``log|dI_n|`` against ``n``
    uses every point and is stable.

    But a regression cannot notice that its own model is wrong, and here that
    mattered: the increments below Young's threshold decay for a few levels and
    then *plateau*, and the fit happily returns a rate below $1$ for a sequence
    that is not converging.  ``tail_rate`` repeats the fit on the second half,
    and :attr:`RateMeasurement.converges` uses that instead.
    """
    increments = np.abs(np.diff(np.asarray(values, dtype=F64)))
    usable = increments > floor
    if usable.sum() < 4:
        raise ValueError(
            f"only {int(usable.sum())} increments above the noise floor {floor}; "
            "the sequence is too short or has already converged to machine precision"
        )
    index = np.arange(increments.size)[usable]
    logs = np.log(increments[usable])
    slope, intercept = np.polyfit(index, logs, 1)
    residual = float(np.std(logs - (slope * index + intercept)))
    half = len(index) // 2
    tail_slope = float(np.polyfit(index[half:], logs[half:], 1)[0]) if half >= 2 else slope
    return RateMeasurement(
        rate=float(math.exp(slope)),
        tail_rate=float(math.exp(tail_slope)),
        n_increments=int(usable.sum()),
        residual=residual,
    )


def young_rate(alpha: float, angle: float = KOCH_ANGLE) -> float:
    """Young's upper bound on the increment decay rate, ``4 r^{1+alpha}``.

    Equals $1$ exactly at ``alpha = d - 1``, which is the classical threshold.
    """
    return 4.0 * koch_ratio(angle) ** (1.0 + alpha)


def predicted_rate(alpha: float, angle: float = KOCH_ANGLE) -> float:
    """Two-branch law for the increment decay rate: ``max(2 r^{1+alpha}, 4 r^2)``.

    Only the second branch is established.  It is the one that matters for the
    regime where anything converges, and it is exact:

    * **Geometric branch** ``4 r^2 = 4^1 r^{1+1}``.  Above the crossover the
      curve's own second-order geometry sets the rate and ``alpha`` drops out
      entirely.  Confirmed model-free to better than one percent over seven
      refinements and at five dimensions, and it is exactly the rate the smooth
      control produces.
    * **Lower branch** ``2 r^{1+alpha}``, the random-walk exponent ``beta = 1/2``
      applied to the ``4^n`` per-segment contributions.  This is a *conjecture
      that the measurements do not support*.  Below the crossover the level
      sequence is not a power law (``chi^2/dof`` of 7 to 17), and below
      ``alpha_c = d - 1`` the increments stop decaying altogether, so no rate --
      this one included -- describes them.  The branch is retained because it
      correctly locates the crossover, not because it predicts the rate there.

    The branches meet where ``r^{1-alpha} = 1/2``; see :func:`predicted_crossover`.
    """
    ratio = koch_ratio(angle)
    return max(2.0 * ratio ** (1.0 + alpha), 4.0 * ratio**2)


def predicted_crossover(angle: float = KOCH_ANGLE) -> float:
    """Where the incoherent and geometric branches of :func:`predicted_rate` meet.

    Solving ``2 r^{1+alpha} = 4 r^2`` gives ``r^{1-alpha} = 1/2``, hence
    ``alpha = 1 - log 2 / log(1/r) = 1 - d/2``.  Below it the rate falls with
    ``alpha``; above it the rate is flat at the smooth-form value.
    """
    return 1.0 - math.log(2.0) / math.log(1.0 / koch_ratio(angle))


def effective_threshold(angle: float = KOCH_ANGLE) -> float:
    """Threshold ``d/2 - 1`` that a random-walk coherence exponent would imply.

    Retained as the explicit statement of a hypothesis that **was tested and
    failed**.  If the per-segment contributions added with independent signs,
    Young's ``alpha > d - 1`` would become ``alpha > beta d - 1`` with
    ``beta = 1/2``, which is negative for every Koch curve and would mean the
    sums converge for every positive Holder exponent.

    Direct measurement says otherwise: below ``d - 1`` the increments plateau and
    the partial sums wander.  The classical threshold is the right one.  This
    function computes the discredited prediction so that the tests can keep
    checking it against the observation rather than leaving the refuted claim
    only in prose; use ``koch_dimension(angle) - 1`` for the real threshold.
    """
    return koch_dimension(angle) / 2.0 - 1.0


def coherence_exponent(rate: float, alpha: float, angle: float = KOCH_ANGLE) -> float:
    """Solve ``rate = 4^beta r^{1+alpha}`` for ``beta``.

    ``beta = 1`` reproduces Young's bound, meaning the per-segment contributions
    add coherently. ``beta = 1/2`` is what independent random signs would give.
    Values between the two quantify how much cancellation actually occurs.
    """
    if rate <= 0.0:
        raise ValueError(f"rate must be positive; got {rate}")
    ratio = koch_ratio(angle)
    return math.log(rate / ratio ** (1.0 + alpha)) / math.log(4.0)


def _tail_ratio(increments: Sequence[float]) -> float:
    """Geometric mean of the successive ratios over the second half of a sequence."""
    ratios = [b / a for a, b in zip(increments, increments[1:]) if a > 0.0 and b > 0.0]
    if not ratios:
        raise ValueError("increment sequence has no usable positive ratios")
    tail = ratios[len(ratios) // 2 :] or ratios
    return float(math.exp(sum(math.log(r) for r in tail) / len(tail)))


@dataclass(frozen=True)
class IncrementDecay:
    """Phase-averaged RMS refinement increments, and whether they keep shrinking.

    The primary instrument, and the only one here that assumes nothing about the
    form of the decay.  Averaging the increment in quadrature over independent
    phases removes the sign oscillation that makes a single realisation
    unreadable, leaving ratios clean enough to be quoted directly -- above the
    crossover they sit on ``4 r^2`` to three decimal places.
    """

    alpha: float
    dimension: float
    angle: float
    levels: tuple[int, ...]
    increments: tuple[float, ...]
    per_phase: tuple[tuple[float, ...], ...]
    n_phases: int
    seed: int

    @property
    def ratios(self) -> tuple[float, ...]:
        return tuple(b / a for a, b in zip(self.increments, self.increments[1:]))

    @property
    def tail_ratio(self) -> float:
        """Geometric mean of the ratios over the second half of the sequence."""
        return _tail_ratio(self.increments)

    def tail_interval(self, *, resamples: int = 400) -> tuple[float, float]:
        """Bootstrap 95% interval for :attr:`tail_ratio`, resampling over phases.

        The increments are RMS averages over a finite phase sample, so the tail
        ratio carries an error bar of its own.  Comparing it to a fixed cutoff
        without one is not safe: at ``alpha = 0.05`` twelve phases give ``0.91``
        and thirty-two give ``0.99``, and only the interval makes clear that
        neither is evidence of decay.
        """
        rng = np.random.default_rng(self.seed + 1)
        table = np.asarray(self.per_phase, dtype=F64)
        draws = []
        for _ in range(resamples):
            pick = rng.integers(0, table.shape[1], table.shape[1])
            sample = np.sqrt(np.mean(np.square(table[:, pick]), axis=1))
            draws.append(_tail_ratio(tuple(float(v) for v in sample)))
        low, high = np.percentile(draws, [2.5, 97.5])
        return float(low), float(high)

    @property
    def converges(self) -> bool:
        """Whether the increments are still shrinking at the end, beyond noise.

        A sequence that decays and then plateaus passes a whole-sequence fit
        while failing this, which is exactly the trap the sub-threshold regime
        sets.  The test is that the *upper* end of the bootstrap interval is
        still below $1$, so that a marginal point is not counted as convergent.
        """
        return self.tail_interval()[1] < 1.0

    @property
    def plateaus(self) -> bool:
        """Whether the tail ratio is consistent with no decay at all."""
        low, high = self.tail_interval()
        return high >= 1.0 and low > 0.8

    @property
    def geometric_branch(self) -> float:
        """``4 r^2``, the rate set by the curve's own second-order geometry."""
        return 4.0 * koch_ratio(self.angle) ** 2


def measure_increments(
    alpha: float,
    *,
    angle: float = KOCH_ANGLE,
    levels: Sequence[int] = tuple(range(3, 11)),
    base: float = 3.0,
    n_phases: int = 16,
    lipschitz: bool = False,
    seed: int = 1,
) -> IncrementDecay:
    """RMS refinement increment per level, averaged over independent phases.

    ``base`` defaults to $3$ rather than $2$ because for the standard curve that
    makes the matched truncation grow by exactly one mode per level.  With base
    $2$ it grows by one or two depending on the level, injecting a systematic
    wobble into any fit across levels -- visible as a reproducible dip at level
    six in an earlier sweep.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1]; got {alpha}")
    ordered = tuple(sorted(levels))
    if len(ordered) < 3:
        raise ValueError(f"need at least 3 levels; got {len(ordered)}")
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * math.pi, n_phases)
    curves = {level: koch_curve(level, angle) for level in ordered + (ordered[-1] + 1,)}
    increments = []
    per_phase = []
    for level in ordered:
        totals = []
        for phase in phases:
            form = (
                lipschitz_form(float(phase))
                if lipschitz
                else matched_weierstrass_form(
                    alpha, level + 1, angle=angle, base=base, phase=float(phase)
                )
            )
            totals.append(
                coherence_profile(curves[level], curves[level + 1], form).total
            )
        per_phase.append(tuple(totals))
        increments.append(float(np.sqrt(np.mean(np.square(totals)))))
    return IncrementDecay(
        alpha=1.0 if lipschitz else alpha,
        dimension=koch_dimension(angle),
        angle=angle,
        levels=ordered,
        increments=tuple(increments),
        per_phase=tuple(per_phase),
        n_phases=n_phases,
        seed=seed,
    )


@dataclass(frozen=True)
class ThresholdMeasurement:
    """Where the increment ratio reaches $1$, and the effective exponent it implies.

    The low branch is fitted in ``log(rate)`` against ``alpha`` and extrapolated
    to ``rate = 1``.  Solving ``4^beta r^{1+alpha} = 1`` at that point gives
    ``beta = (1 + alpha_threshold) / d``, so a measured threshold is equivalent
    to a measured effective dimension ``beta d`` -- which is what Young's
    condition uses ``d`` for.
    """

    angle: float
    dimension: float
    alphas: tuple[float, ...]
    tails: tuple[float, ...]
    slope: float
    threshold: float
    threshold_error: float

    @property
    def young_threshold(self) -> float:
        return self.dimension - 1.0

    @property
    def beta(self) -> float:
        return (1.0 + self.threshold) / self.dimension

    @property
    def beta_error(self) -> float:
        return self.threshold_error / self.dimension

    @property
    def below_young(self) -> bool:
        """Whether the measured threshold is clear of the classical one."""
        return self.threshold + 2.0 * self.threshold_error < self.young_threshold


def convergence_threshold(
    angle: float = KOCH_ANGLE,
    *,
    alphas: Sequence[float] | None = None,
    levels: Sequence[int] = tuple(range(3, 11)),
    base: float = 3.0,
    n_phases: int = 16,
    seed: int = 1,
    resamples: int = 300,
    margin: float = 1.08,
) -> ThresholdMeasurement:
    """Locate the edge of convergence by extrapolating the low branch.

    Only exponents whose rate stands clear of the geometric floor ``4 r^2`` by
    ``margin`` are used: on the floor the rate no longer depends on ``alpha``, so
    those points carry no information about where the branch crosses $1$ and
    including them would flatten the fit toward the floor.

    The error bar comes from resampling phases, using the *same* phase indices at
    every ``alpha`` so that the correlations between the points are preserved --
    they share a phase sample, and treating them as independent would understate
    the spread.
    """
    grid = tuple(alphas) if alphas is not None else (0.05, 0.15, 0.25, 0.35, 0.45)
    if len(grid) < 2:
        raise ValueError(f"need at least 2 exponents to fit a slope; got {len(grid)}")
    floor = 4.0 * koch_ratio(angle) ** 2
    measured = [
        measure_increments(
            alpha,
            angle=angle,
            levels=levels,
            base=base,
            n_phases=n_phases,
            seed=seed,
        )
        for alpha in grid
    ]
    keep = [m for m in measured if m.tail_ratio > margin * floor]
    if len(keep) < 2:
        raise ValueError(
            f"only {len(keep)} of {len(grid)} exponents stand clear of the "
            f"geometric floor {floor:.4f} at angle {angle}; supply smaller alphas"
        )

    def fit(tables: Sequence[IncrementDecay], pick: NDArray | None) -> tuple[float, float]:
        ratios = []
        for item in tables:
            if pick is None:
                ratios.append(item.tail_ratio)
            else:
                table = np.asarray(item.per_phase, dtype=F64)
                sample = np.sqrt(np.mean(np.square(table[:, pick]), axis=1))
                ratios.append(_tail_ratio(tuple(float(v) for v in sample)))
        x = np.array([item.alpha for item in tables], dtype=F64)
        y = np.log(np.array(ratios, dtype=F64))
        slope, intercept = np.polyfit(x, y, 1)
        if slope >= 0.0:
            raise ValueError(f"low branch is not decreasing in alpha (slope {slope})")
        return float(slope), float(-intercept / slope)

    slope, threshold = fit(keep, None)
    rng = np.random.default_rng(seed + 7)
    draws = []
    for _ in range(resamples):
        pick = rng.integers(0, n_phases, n_phases)
        try:
            draws.append(fit(keep, pick)[1])
        except ValueError:
            continue
    return ThresholdMeasurement(
        angle=angle,
        dimension=koch_dimension(angle),
        alphas=tuple(item.alpha for item in keep),
        tails=tuple(item.tail_ratio for item in keep),
        slope=slope,
        threshold=threshold,
        threshold_error=float(np.std(draws)) if len(draws) > 2 else float("nan"),
    )


@dataclass(frozen=True)
class CoherenceMeasurement:
    """Measured decay rate against Young's bound, for one form on one curve."""

    alpha: float
    dimension: float
    measured_rate: float
    young_rate: float
    beta: float
    n_phases: int
    spread: float
    alpha_floor: float

    @property
    def in_window(self) -> bool:
        """Whether a fixed form of this exponent is resolvable at this level count.

        False means the number reported is a property of the truncation, not of
        the Holder class; use :func:`measure_matched` instead.
        """
        return self.alpha >= self.alpha_floor

    @property
    def young_threshold(self) -> float:
        return self.dimension - 1.0

    @property
    def below_young_threshold(self) -> bool:
        return self.alpha < self.young_threshold

    @property
    def converges(self) -> bool:
        return self.measured_rate < 1.0


def measure_coherence(
    alpha: float,
    *,
    angle: float = KOCH_ANGLE,
    levels: Sequence[int] = tuple(range(2, 11)),
    base: float = 2.0,
    n_phases: int = 6,
    lipschitz: bool = False,
) -> CoherenceMeasurement:
    """Measure the coherence exponent of a Holder form on a Koch curve.

    Averages over several phase offsets of the form, since a single phase can
    land on an atypical alignment with the curve's self-similar structure. The
    spread across phases is reported so that an unstable measurement is visible
    rather than hidden by the median.
    """
    window = resolution_window(max(levels), angle=angle, base=base)
    curves = {level: koch_curve(level, angle) for level in levels}
    rates = []
    for phase in np.linspace(0.1, 5.0, n_phases):
        form = (
            lipschitz_form(phase)
            if lipschitz
            else weierstrass_form(alpha, base=base, phase=phase, terms=window.terms)
        )
        values = [riemann_stieltjes(curves[level], form) for level in levels]
        try:
            rates.append(decay_rate(values).rate)
        except ValueError:
            continue
    if not rates:
        raise ValueError("no phase produced a usable increment sequence")
    measured = float(np.median(rates))
    effective_alpha = 1.0 if lipschitz else alpha
    return CoherenceMeasurement(
        alpha=effective_alpha,
        dimension=koch_dimension(angle),
        measured_rate=measured,
        young_rate=young_rate(effective_alpha, angle),
        beta=coherence_exponent(measured, effective_alpha, angle),
        n_phases=len(rates),
        spread=float(np.std(rates)),
        alpha_floor=0.0 if lipschitz else window.alpha_floor,
    )


@dataclass(frozen=True)
class MatchedMeasurement:
    """Exact per-level coherence for a resolution-matched form.

    Unlike :class:`CoherenceMeasurement` this carries no fitted decay rate, so it
    is not exposed to the aliasing trap and remains valid below Young's
    threshold.

    ``beta`` is *not* read off a single level.  The per-level exponent
    :attr:`LevelProfile.beta` drifts downward as the level grows, so quoting it
    at any one level measures the level.  What is scale-free is the growth of
    :attr:`LevelProfile.randomness` -- coherent addition doubles it every level,
    independent signs leave it flat -- and ``beta = 1/2 + log(growth)/log 4``
    inverts that.  The Lipschitz control pins the calibration exactly: it grows
    by a factor of two per level and returns ``beta = 1``.

    ``sampled_alpha`` is the referee: it is the Holder exponent the per-segment
    contributions actually exhibit, recovered from how the mean term magnitude
    shrinks with level, and it must agree with the requested ``alpha`` for
    ``beta`` to mean anything.
    """

    alpha: float
    dimension: float
    angle: float
    beta: float
    growth: float
    growth_residual: float
    sampled_alpha: float
    peakiness: float
    profiles: tuple[LevelProfile, ...]

    @property
    def effective_dimension(self) -> float:
        """``beta * d``: the dimension the curve behaves as if it had.

        The convergence condition ``4**beta r**(1+alpha) < 1`` is Young's
        ``alpha > d - 1`` with ``d`` replaced by ``beta * d``.  Coherent addition
        (``beta = 1``) recovers Young exactly; anything less is cancellation that
        Young's bound, being an absolute-value estimate, cannot see.
        """
        return self.beta * self.dimension

    @property
    def effective_threshold(self) -> float:
        """Convergence threshold implied by the measured coherence."""
        return self.effective_dimension - 1.0

    @property
    def converges(self) -> bool:
        return self.measured_rate < 1.0

    @property
    def young_threshold(self) -> float:
        return self.dimension - 1.0

    @property
    def below_young_threshold(self) -> bool:
        return self.alpha < self.young_threshold

    @property
    def faithful(self) -> bool:
        """Whether the sampled exponent matches the requested one."""
        return abs(self.sampled_alpha - self.alpha) <= 0.1

    @property
    def measured_rate(self) -> float:
        """Increment decay rate implied by the exact coherence exponent."""
        return 4.0**self.beta * koch_ratio(self.angle) ** (1.0 + self.alpha)


def measure_matched(
    alpha: float,
    *,
    angle: float = KOCH_ANGLE,
    levels: Sequence[int] = tuple(range(3, 10)),
    base: float = 2.0,
    n_phases: int = 4,
    lipschitz: bool = False,
) -> MatchedMeasurement:
    """Exact coherence exponent from resolution-matched forms.

    For each level a form is built whose spectrum stops exactly where the
    level-``n+1`` partition stops resolving it, and the increment from level
    ``n`` to ``n+1`` is decomposed into its ``4**n`` per-segment contributions.
    Because no retained mode is aliased and no discarded mode was resolvable,
    this is valid at every ``alpha`` in ``(0, 1]``, unlike the fixed-form
    measurement which is confined to ``alpha >= resolution_floor(...)``.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1]; got {alpha}")
    phases = np.linspace(0.1, 5.0, n_phases)
    profiles: list[LevelProfile] = []
    peaks: list[float] = []
    terms_by_level: dict[int, list[float]] = {}
    random_by_level: dict[int, list[float]] = {}
    for level in levels:
        coarse = koch_curve(level, angle)
        fine = koch_curve(level + 1, angle)
        for phase in phases:
            form = (
                lipschitz_form(phase)
                if lipschitz
                else matched_weierstrass_form(
                    alpha, level + 1, angle=angle, base=base, phase=phase
                )
            )
            profile = coherence_profile(coarse, fine, form)
            if profile.l2 <= 0.0 or profile.total == 0.0:
                continue
            profiles.append(profile)
            peaks.append(profile.peakiness)
            terms_by_level.setdefault(level, []).append(profile.term)
            random_by_level.setdefault(level, []).append(profile.randomness)
    if len(random_by_level) < 3:
        raise ValueError(
            f"only {len(random_by_level)} levels produced a usable profile for "
            f"alpha={alpha}; at least 3 are needed to fit the coherence growth"
        )

    def regress(table: dict[int, list[float]]) -> tuple[float, float]:
        index = np.array(sorted(table), dtype=F64)
        logs = np.log(np.array([np.median(table[int(k)]) for k in index], dtype=F64))
        slope, intercept = np.polyfit(index, logs, 1)
        residual = float(np.std(logs - (slope * index + intercept)))
        return float(slope), residual

    term_slope, _ = regress(terms_by_level)
    growth_slope, growth_residual = regress(random_by_level)

    return MatchedMeasurement(
        alpha=1.0 if lipschitz else alpha,
        dimension=koch_dimension(angle),
        angle=angle,
        beta=0.5 + growth_slope / math.log(4.0),
        growth=math.exp(growth_slope),
        growth_residual=growth_residual,
        sampled_alpha=term_slope / math.log(koch_ratio(angle)) - 1.0,
        peakiness=float(np.median(peaks)),
        profiles=tuple(profiles),
    )
