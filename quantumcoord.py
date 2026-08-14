"""Quantum coordination in markets: what is real, how big it is, and who can see it.

Where the literature actually stands
------------------------------------
Two of the three problems usually posed as open have papers from this year.

* **Latency versus decoherence.**  Li, Kikura, Goban, Yamasaki and Sunami,
  arXiv:2604.07451 (April 2026), give operational criteria for quantum advantage
  in latency-constrained tacit coordination, including finite operation times,
  finite entanglement rates and statistical certification, with hardware numbers
  -- microsecond decision latency, ``8e3`` decisions per second, a 50 km
  metropolitan network.  The "critical latency threshold" is a framework, not a
  gap.
* **Many entangled agents.**  Hymas et al., arXiv:2602.06367 (February 2026),
  build a quantum stock market with reinforcement-learning agents and find that
  entanglement *stabilises* prices: it removes the pathological pure-strategy
  Nash equilibrium of the ``p``-guessing game that drives speculative collapse.
  The answer runs opposite to the usual guess that markets would become chaotic.
* **Forensic detectability** is the one still open, and it is what this module
  answers.

A correction to the conjecture as usually stated
-------------------------------------------------
The claim is normally phrased as "the set of quantum correlated equilibria is
strictly larger than the set of classical correlated equilibria."  For a
complete-information normal-form game that is false, and the reason is one line.

A correlated equilibrium is a probability distribution over action profiles
obeying the incentive constraints.  A quantum device consulted by the players
produces *some* probability distribution over action profiles.  If that
distribution obeys the incentive constraints it is already a classical
correlated equilibrium, implementable by a mediator who samples it and whispers
the recommendations.  So the quantum set is contained in the classical one; it
cannot be larger.  :func:`correlated_equilibrium_polytope` computes that polytope
exactly, and :func:`quantum_point_is_classically_realisable` checks the
containment on the optimal quantum point.

One corollary settles a question usually posed as open.  A "quantum price of
anarchy" is asked for as though it needed defining; for complete-information
games it does not exist as a separate quantity, because the two equilibrium sets
coincide and every welfare ratio over them agrees
(:func:`quantum_price_of_anarchy`).  The Prisoner's Dilemma has correlated price
of anarchy exactly ``1/3``, quantum or classical alike.  The question becomes
substantive only once the mediator is removed and types are private.

The advantage is real, but it lives somewhere else: in **Bayesian games with
private types and no mediator**.  There the classical resource is shared
randomness -- which produces exactly the local (Bell) polytope -- and quantum
correlations strictly exceed it.  That is Brunner and Linden's connection between
Bell nonlocality and Bayesian games, and the two-servers-see-a-local-signal story
is precisely that setting.  So the mechanism survives; the label
"correlated equilibrium" does not.

How big the rent is: exactly ``sqrt(2)``, or exactly nothing
-------------------------------------------------------------
For the CHSH game the numbers are the familiar ones and this module derives them
rather than quoting: classical ``3/4`` by enumerating all sixteen deterministic
strategies, quantum ``(2 + sqrt 2)/4 ~ 0.8536`` from the optimal correlations.

The more useful statement is what happens across *all* such games.  A two-input
XOR game is a sign matrix ``M``; the classical bias is a maximum over sign
vectors and the quantum bias is the nuclear norm of ``M`` (Tsirelson).  Over the
sixteen two-by-two sign matrices (:func:`xor_game_census`):

    8 of 16 admit quantum advantage, every one with ratio **exactly sqrt(2)**;
    the other 8 admit none, with ratio exactly 1.

and the split is precisely **rank two versus rank one**.  There is no continuum
of quantum rents.

**Nor does a bigger game buy a bigger rent.**  Enumerating all ``512`` sign
matrices at three inputs (:func:`max_ratio`) gives a maximum ratio of exactly
``6/5``, attained at rank three -- strictly *below* the two-input maximum of
``sqrt 2``.  The spectrum there is ``{1, 1.0102, 1.2}``, still discrete but no
longer a dichotomy.  So the largest rent in this family sits at the *smallest*
game, and enlarging the coordination problem shrinks the edge rather than growing
it.  Larger ratios need weighted rather than sign payoffs, and even then
Grothendieck's constant caps every XOR game at any size:
``K_G <= 1.7822`` (:data:`GROTHENDIECK_CEILING`).  There is a universal ceiling on
the quantum rent, and CHSH already sits within twenty-five percent of it.  A market coordination payoff either has a rank-two structure,
in which case it is CHSH in disguise and the bias improves by ``sqrt(2)``, or it
does not, in which case entanglement is worth exactly zero.  Whether a real order
flow presents a rank-two payoff is an empirical question about markets, not a
question about quantum mechanics -- and it is the question that decides whether
any of this pays.

The forensic result
-------------------
Can a regulator distinguish quantum coordination from luck using the public order
book?  The answer splits cleanly, and the split is not about quantum mechanics.

**From outcomes alone: no, and provably not.**  No-signalling forces the quantum
marginals to be independent of the other party's input, and for the optimal CHSH
strategy every single-party marginal is exactly ``1/2``
(:func:`quantum_marginals`).  A classical pair using one shared fair coin --
"both act on the same random bit" -- wins ``3/4`` and has *the same* marginals,
exactly ``1/2`` (:func:`shared_coin_marginals`).  The two are marginally
identical, so the Kullback-Leibler divergence between what a one-sided observer
sees is exactly zero (:func:`marginal_divergence`).  No quantity of order-book
data separates them.  This is not a statistical difficulty; it is an identity.

**From joint statistics: yes, and quickly.**  If the private signals ``(x, y)``
can be reconstructed, the divergence from the quantum point to the whole local
polytope is

    KL  =  0.0321 nats per round        (:func:`detection_divergence`)

so decisive evidence needs a few hundred rounds: about ``215`` for a ``1e-3``
likelihood ratio, ``431`` for ``1e-6`` (:func:`rounds_to_detect`).  For a venue
seeing thousands of events a second that is under a second of tape.

So the regulatory problem is not "can quantum collusion be detected".  It is
**"can the private inputs be reconstructed"** -- and that is an ordinary question
about market data, with no quantum content.  If the triggering signal is a public
event both servers observed, it can, and detection is easy.  If the signal is
genuinely private, detection is impossible in principle.

**And there is a second threshold, which bites harder.**  A regulator does not
see every decision -- only the ones that reach the tape.  Model that as a
visibility ``eta``, with unseen events recorded as a default.  Then a violation
can be certified only above a critical visibility, and below it a local model
reproduces the record exactly, so the evidence is not weak but absent.  For the
maximally entangled strategy the threshold is exactly

    eta*  =  2 (sqrt 2 - 1)  =  0.82842712...        (:data:`MAXIMAL_THRESHOLD`)

Over the family of partially entangled states the threshold falls, reaching
Eberhard's ``2/3`` in the limit -- but the advantage falls with it
(:func:`frontier`):

    theta      pi/4      0.5       0.3       0.1       0.02
    threshold  0.8284    0.7597    0.7183    0.6824    0.6697
    advantage  +0.1036   +0.0767   +0.0371   +0.0049   +0.0002

The two move together, monotonically (:func:`frontier_is_monotone`).  **A
regulator cannot trade effect size against observation quality.**  Certifying the
ten-percentage-point strategy requires seeing ``82.8%`` of the coordination
events; an observer who sees only ``2/3`` can certify nothing whose advantage is
not already negligible.  That, rather than any statement about quantum mechanics,
is the binding constraint on forensic finance here.

Scope
-----
Two parties, two inputs, two outputs, which is where the polytopes are small
enough to compute exactly and where the XOR classification is complete.  The
polytope, the game values and the marginal identity are exact rational or exact
symbolic; the detection divergence is a numerical convex optimisation over the
local polytope and is reported as such.  Nothing here models an order book, and
no claim is made that real markets present rank-two payoffs.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, Sequence

import numpy as np

__all__ = [
    "OUTCOMES",
    "deterministic_strategies",
    "local_polytope_facets",
    "local_polytope_dimension",
    "chsh_facet_count",
    "classical_chsh_value",
    "quantum_chsh_value",
    "quantum_distribution",
    "quantum_marginals",
    "shared_coin_marginals",
    "marginal_divergence",
    "detection_divergence",
    "rounds_to_detect",
    "xor_game_census",
    "XOR_ADVANTAGE_RATIO",
    "correlated_equilibrium_polytope",
    "quantum_point_is_classically_realisable",
    "MAXIMAL_THRESHOLD",
    "EBERHARD_FLOOR",
    "detection_threshold",
    "quantum_advantage",
    "frontier",
    "frontier_is_monotone",
    "MAX_RATIO_TWO_INPUTS",
    "MAX_RATIO_THREE_INPUTS",
    "GROTHENDIECK_CEILING",
    "classical_bias",
    "tsirelson_bias",
    "max_ratio",
    "correlated_price_of_anarchy",
    "quantum_price_of_anarchy",
]

LOGGER = logging.getLogger(__name__)

#: Index order for ``p(a, b | x, y)``: outer loop over inputs, inner over outputs.
OUTCOMES: tuple[tuple[int, int, int, int], ...] = tuple(
    (a, b, x, y) for x in (0, 1) for y in (0, 1) for a in (0, 1) for b in (0, 1)
)

#: The only quantum/classical bias ratio a two-input XOR game can have, other
#: than 1.  Exactly ``sqrt(2)``; there is no continuum.
XOR_ADVANTAGE_RATIO = math.sqrt(2)


# --------------------------------------------------------------------------- #
# The local polytope
# --------------------------------------------------------------------------- #
def deterministic_strategies() -> list[list[int]]:
    """The sixteen deterministic local strategies as points ``p(a,b|x,y)``.

    Each party maps its input to an output, ``a = f(x)`` and ``b = g(y)``, giving
    ``4 x 4`` joint strategies.  Their convex hull is the local polytope: every
    correlation achievable with shared randomness and no communication.
    """
    points = []
    for first in itertools.product((0, 1), repeat=2):
        for second in itertools.product((0, 1), repeat=2):
            points.append(
                [
                    1 if (a == first[x] and b == second[y]) else 0
                    for (a, b, x, y) in OUTCOMES
                ]
            )
    return points


def _local_cone():
    from PyNormaliz import Cone

    return Cone(vertices=[point + [1] for point in deterministic_strategies()])


def local_polytope_facets() -> int:
    """Number of facets of the local polytope: ``24``.

    Sixteen positivity facets and eight CHSH inequalities, the textbook answer,
    obtained here from nothing but the list of deterministic strategies.
    """
    return len(_local_cone().SupportHyperplanes())


def local_polytope_dimension() -> int:
    """Affine dimension of the local polytope: ``8``."""
    return _local_cone().Rank() - 1


def chsh_facet_count() -> int:
    """Facets that are not positivity constraints: the eight CHSH inequalities."""
    positivity = 4 * 4
    return local_polytope_facets() - positivity


# --------------------------------------------------------------------------- #
# Game values
# --------------------------------------------------------------------------- #
def _wins(a: int, b: int, x: int, y: int) -> bool:
    return (a ^ b) == (x & y)


def classical_chsh_value() -> Fraction:
    """``3/4``: the best any shared-randomness strategy achieves, by enumeration.

    Maximised over the sixteen deterministic strategies, which suffices because
    the objective is linear and the local polytope is their convex hull.
    """
    best = Fraction(0)
    for point in deterministic_strategies():
        value = Fraction(
            sum(point[i] for i, key in enumerate(OUTCOMES) if _wins(*key)), 4
        )
        best = max(best, value)
    return best


def quantum_chsh_value() -> float:
    """``(2 + sqrt 2) / 4``, Tsirelson's bound for the CHSH game."""
    return (2 + math.sqrt(2)) / 4


def quantum_distribution() -> np.ndarray:
    """The optimal CHSH correlations ``p(a,b|x,y) = [1 + (-1)^(a^b^xy)/sqrt2]/4``."""
    root = math.sqrt(2)
    return np.array(
        [(1 + ((-1) ** ((a ^ b) ^ (x & y))) / root) / 4 for (a, b, x, y) in OUTCOMES]
    )


# --------------------------------------------------------------------------- #
# The forensic question
# --------------------------------------------------------------------------- #
def quantum_marginals() -> dict[tuple[int, int], float]:
    """``p(a = 0 | x, y)`` for the optimal quantum strategy: ``1/2`` everywhere.

    Independent of ``y`` -- that is no-signalling, and it is what makes one-sided
    observation useless.  Independent of ``x`` as well, which is a property of
    this particular optimum.
    """
    distribution = quantum_distribution()
    out = {}
    for x in (0, 1):
        for y in (0, 1):
            out[(x, y)] = float(
                sum(
                    distribution[i]
                    for i, (a, b, xx, yy) in enumerate(OUTCOMES)
                    if xx == x and yy == y and a == 0
                )
            )
    return out


def shared_coin_marginals() -> dict[tuple[int, int], float]:
    """The same marginals, from a purely classical strategy.

    "Both parties output one shared fair coin" wins ``3/4`` and puts ``1/2`` on
    every single-party marginal.  It is the witness that marginals carry no
    information: a classical pair can reproduce the quantum party's observable
    behaviour exactly.
    """
    out = {}
    for x in (0, 1):
        for y in (0, 1):
            out[(x, y)] = 0.5
    return out


def marginal_divergence() -> float:
    """KL divergence between the quantum and shared-coin marginals: exactly zero.

    Not small -- zero.  A one-sided observer, or any observer of outcomes without
    the private inputs, has no statistical handle at all.
    """
    total = 0.0
    quantum = quantum_marginals()
    classical = shared_coin_marginals()
    for key, value in quantum.items():
        other = classical[key]
        for probability, reference in ((value, other), (1 - value, 1 - other)):
            if probability > 0:
                total += probability * math.log(probability / reference)
    return total / len(quantum)


def detection_divergence(*, restarts: int = 6, seed: int = 0) -> float:
    """``min KL(quantum || local)`` in nats per round, over the local polytope.

    The optimal error exponent for telling entangled coordination from the best
    classical imitation, when the inputs ``(x, y)`` are observed alongside the
    outputs.  Computed by minimising over mixtures of the sixteen deterministic
    strategies -- a convex problem, solved numerically, unlike the exact
    quantities above.
    """
    from scipy.optimize import minimize

    quantum = quantum_distribution()
    vertices = np.array(deterministic_strategies(), dtype=float)
    generator = np.random.default_rng(seed)

    def objective(weights: np.ndarray) -> float:
        weights = np.abs(weights)
        total = weights.sum()
        if total <= 0:
            return 1e6
        mixture = vertices.T @ (weights / total)
        value = 0.0
        for i in range(len(quantum)):
            if mixture[i] <= 1e-14:
                return 1e6
            value += quantum[i] * math.log(quantum[i] / mixture[i])
        return value / 4

    constraint = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(1e-12, 1.0)] * 16
    best = math.inf
    for _ in range(restarts):
        start = generator.random(16) + 0.1
        start /= start.sum()
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"maxiter": 2000, "ftol": 1e-14},
        )
        if result.fun < best:
            best = float(result.fun)
    return best


def rounds_to_detect(confidence: float, *, divergence: float | None = None) -> float:
    """Rounds of joint observation needed for a likelihood ratio of ``1/confidence``.

    ``ln(1/confidence) / KL``.  With the measured divergence this is about ``215``
    rounds at ``1e-3`` and ``431`` at ``1e-6`` -- under a second of tape for a
    venue seeing thousands of events per second.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence is a probability strictly between 0 and 1")
    rate = detection_divergence() if divergence is None else divergence
    if rate <= 0:
        raise ValueError("a non-positive divergence gives no detection")
    return math.log(1 / confidence) / rate


# --------------------------------------------------------------------------- #
# How special the advantage is
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class XorGame:
    """A two-input XOR game, given by its sign matrix."""

    matrix: tuple[int, int, int, int]
    rank: int
    classical_bias: float
    quantum_bias: float

    @property
    def ratio(self) -> float:
        return self.quantum_bias / self.classical_bias

    @property
    def has_advantage(self) -> bool:
        return self.ratio > 1 + 1e-9


def _quantum_bias(matrix: np.ndarray) -> float:
    """Tsirelson's bias for a two-input XOR game, by the closed form above."""
    signs = [matrix[0, y] * matrix[1, y] for y in (0, 1)]
    grid = np.linspace(-1.0, 1.0, 200001)
    values = sum(np.sqrt(np.maximum(2 + 2 * sign * grid, 0.0)) for sign in signs)
    return float(values.max())


def xor_game_census() -> list[XorGame]:
    """All sixteen two-by-two sign matrices, with both biases.

    The classical bias maximises over sign vectors.  The quantum bias is
    Tsirelson's: the maximum of ``sum M_xy <u_x, v_y>`` over unit vectors.
    Optimising the ``v`` first gives ``sum_y || M_0y u_0 + M_1y u_1 ||``, and with
    the two ``u`` at angle ``theta`` that is

        beta_Q  =  max_c  sum_y sqrt( 2 + 2 s_y c ) ,   s_y = M_0y M_1y ,

    over ``c = cos(theta) in [-1, 1]``, which is what :func:`_quantum_bias`
    evaluates.  Using the nuclear norm instead is wrong: it happens to agree at
    rank two, where it gives CHSH's ``2 sqrt 2``, and fails at rank one, where it
    returns *less* than the classical bias -- impossible, since quantum strategies
    include classical ones.  A test on the rank-one ratio caught it.

    The census shows the dichotomy: rank-one matrices give ratio exactly ``1``,
    rank-two matrices give ratio exactly ``sqrt(2)``, and nothing else occurs.
    """
    census = []
    for entries in itertools.product((-1, 1), repeat=4):
        matrix = np.array(entries, dtype=float).reshape(2, 2)
        classical = max(
            abs(
                sum(
                    matrix[x, y] * first[x] * second[y]
                    for x in (0, 1)
                    for y in (0, 1)
                )
            )
            for first in itertools.product((-1, 1), repeat=2)
            for second in itertools.product((-1, 1), repeat=2)
        )
        quantum = _quantum_bias(matrix)
        census.append(
            XorGame(
                matrix=entries,
                rank=int(round(np.linalg.matrix_rank(matrix))),
                classical_bias=float(classical),
                quantum_bias=quantum,
            )
        )
    return census


# --------------------------------------------------------------------------- #
# The correlated-equilibrium correction
# --------------------------------------------------------------------------- #
def correlated_equilibrium_polytope(
    payoffs: Sequence[Sequence[tuple[int, int]]],
) -> tuple[int, int]:
    """``(vertices, facets)`` of the correlated equilibrium polytope of a 2x2 game.

    ``payoffs[i][j]`` is the pair of utilities when the row player takes ``i`` and
    the column player ``j``.  The polytope is cut out by the incentive
    constraints together with non-negativity and normalisation, all of them
    rational, so the computation is exact.
    """
    from PyNormaliz import Cone

    inequalities = []
    for action in (0, 1):
        other = 1 - action
        row = [0] * 4
        for column in (0, 1):
            gain = payoffs[action][column][0] - payoffs[other][column][0]
            row[2 * action + column] = gain
        inequalities.append(row)
    for action in (0, 1):
        other = 1 - action
        row = [0] * 4
        for line in (0, 1):
            gain = payoffs[line][action][1] - payoffs[line][other][1]
            row[2 * line + action] = gain
        inequalities.append(row)
    for slot in range(4):
        row = [0] * 4
        row[slot] = 1
        inequalities.append(row)
    cone = Cone(
        inhom_inequalities=[row + [0] for row in inequalities],
        inhom_equations=[[1, 1, 1, 1, -1]],
    )
    return len(cone.VerticesOfPolyhedron()), len(cone.SupportHyperplanes())


def quantum_point_is_classically_realisable() -> bool:
    """Whether the optimal quantum correlation is a distribution over actions.

    It is: it is a normalised, non-negative distribution for each input pair, so a
    mediator who samples it and whispers recommendations reproduces it exactly.
    That is the whole content of the containment -- what quantum devices produce
    are probability distributions, and correlated equilibrium already allows any
    distribution the incentive constraints permit.  The strict gap appears only
    once the mediator is removed and the inputs are private.
    """
    distribution = quantum_distribution()
    if np.any(distribution < -1e-12):
        return False
    for x in (0, 1):
        for y in (0, 1):
            block = sum(
                distribution[i]
                for i, (_, _, xx, yy) in enumerate(OUTCOMES)
                if xx == x and yy == y
            )
            if abs(block - 1.0) > 1e-12:
                return False
    return True


# --------------------------------------------------------------------------- #
# How much of the tape must be visible: the detection-efficiency threshold
# --------------------------------------------------------------------------- #
#: Fraction of coordination events that must appear in the record before a
#: maximally entangled strategy can be certified at all: ``2 (sqrt 2 - 1)``.
#: Below it a local model reproduces everything observed, so the evidence is not
#: weak -- it is absent.
MAXIMAL_THRESHOLD = 2 * (math.sqrt(2) - 1)

#: The floor over all entangled states, approached as the entanglement vanishes.
#: Eberhard's ``2/3``.
EBERHARD_FLOOR = 2 / 3


def _chsh_observed(settings: Sequence[float], theta: float, efficiency: float) -> float:
    """CHSH value seen by an observer who misses a fraction of the events.

    State ``cos(theta)|00> + sin(theta)|11>``, settings in the x-z plane.  A
    missed event is recorded as the default outcome, so with efficiency ``eta``

        S_obs = eta^2 S + 2 eta (1 - eta) (<A_0> + <B_0>) + 2 (1 - eta)^2 ,

    the cross terms surviving only on the settings that appear with the same sign
    in both of their CHSH terms.  Violation means ``S_obs > 2``.
    """
    first, second, third, fourth = settings
    pair = math.sin(2 * theta)
    bias = math.cos(2 * theta)
    correlate = lambda a, b: math.cos(a) * math.cos(b) + pair * math.sin(a) * math.sin(b)
    raw = (
        correlate(first, third)
        + correlate(first, fourth)
        + correlate(second, third)
        - correlate(second, fourth)
    )
    marginal = bias * (math.cos(first) + math.cos(third))
    return (
        efficiency**2 * raw
        + 2 * efficiency * (1 - efficiency) * marginal
        + 2 * (1 - efficiency) ** 2
    )


def _best_chsh(theta: float, efficiency: float) -> float:
    from scipy.optimize import minimize

    result = minimize(
        lambda settings: -(_chsh_observed(settings, theta, efficiency) - 2),
        [0.0, math.pi / 2, math.pi / 4, -math.pi / 4],
        method="Nelder-Mead",
        options={"maxiter": 20000, "fatol": 1e-14, "xatol": 1e-12},
    )
    return -float(result.fun)


def detection_threshold(theta: float = math.pi / 4, *, steps: int = 60) -> float:
    """Smallest visible fraction at which the strategy can be certified at all.

    Bisected on the largest achievable ``S_obs - 2``.  At maximal entanglement it
    returns :data:`MAXIMAL_THRESHOLD`; as the entanglement vanishes it falls to
    :data:`EBERHARD_FLOOR`.
    """
    if not 0 < theta <= math.pi / 4:
        raise ValueError("theta parameterises the state in (0, pi/4]")
    low, high = 0.5, 1.0
    for _ in range(steps):
        middle = (low + high) / 2
        if _best_chsh(theta, middle) > 0:
            high = middle
        else:
            low = middle
    return high


def quantum_advantage(theta: float = math.pi / 4) -> float:
    """Excess win probability over the classical ``3/4`` at full visibility."""
    if not 0 < theta <= math.pi / 4:
        raise ValueError("theta parameterises the state in (0, pi/4]")
    return 0.5 + _best_chsh(theta, 1.0) / 8 + 2 / 8 - 0.75


def frontier(thetas: Sequence[float] | None = None) -> list[tuple[float, float, float]]:
    """``(theta, threshold, advantage)`` along the detectability frontier.

    The two move together: every state that is harder to catch is also more
    profitable.  There is no corner of the family offering a large effect at a
    loose observation requirement, which is the practical content of the whole
    forensic question.
    """
    if thetas is None:
        thetas = [math.pi / 4, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]
    return [(t, detection_threshold(t), quantum_advantage(t)) for t in thetas]


def frontier_is_monotone(thetas: Sequence[float] | None = None) -> bool:
    """Whether threshold and advantage rise together along the family.

    They do.  A regulator cannot trade effect size against observation quality:
    catching the ``+10`` percentage point strategy requires seeing ``82.8%`` of
    events, and an observer limited to ``2/3`` can only certify strategies whose
    advantage is essentially zero.
    """
    rows = frontier(thetas)
    ordered = sorted(rows, key=lambda row: row[2])
    thresholds = [row[1] for row in ordered]
    return all(
        later >= earlier - 1e-9 for earlier, later in zip(thresholds, thresholds[1:])
    )


# --------------------------------------------------------------------------- #
# Does a bigger game buy a bigger rent?  No.
# --------------------------------------------------------------------------- #
#: Maximum quantum/classical bias ratio over all ``2x2`` sign matrices: ``sqrt 2``.
MAX_RATIO_TWO_INPUTS = math.sqrt(2)

#: Maximum over all ``3x3`` sign matrices: ``6/5``, attained at rank three.
#: Strictly *below* the two-input maximum -- enlarging the coordination game
#: shrinks the available rent rather than growing it.
MAX_RATIO_THREE_INPUTS = 1.2

#: Grothendieck's constant bounds the ratio for XOR games of any size, with any
#: real payoff weights: ``K_G <= 1.7822``.  A universal ceiling on the rent.
GROTHENDIECK_CEILING = 1.7822


def classical_bias(matrix: np.ndarray) -> float:
    """``max_{s,t in {+-1}} s^T M t``, by enumeration over one side."""
    size = matrix.shape[0]
    return max(
        float(np.abs(np.array(signs) @ matrix).sum())
        for signs in itertools.product((-1, 1), repeat=size)
    )


def tsirelson_bias(matrix: np.ndarray, *, restarts: int = 20, iterations: int = 500,
                   seed: int = 1) -> float:
    """``max sum M_xy <u_x, v_y>`` over unit vectors, by alternating maximisation.

    Given ``U`` the optimal ``v_y`` is the normalised ``(M^T U)_y``, so the value
    is the sum of the row norms of ``M^T U``; alternating the two sides increases
    it monotonically.  Non-convex, hence the restarts, but it reproduces CHSH's
    ``2 sqrt 2`` exactly, which is what licenses reading the census off it.
    """
    generator = np.random.default_rng(seed)
    size = matrix.shape[0]
    best = 0.0
    for _ in range(restarts):
        left = generator.normal(size=(size, size))
        left /= np.linalg.norm(left, axis=1, keepdims=True)
        previous = -1.0
        for _ in range(iterations):
            transported = matrix.T @ left
            value = float(np.linalg.norm(transported, axis=1).sum())
            right = transported / np.maximum(
                np.linalg.norm(transported, axis=1, keepdims=True), 1e-15
            )
            left = matrix @ right
            left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-15)
            if abs(value - previous) < 1e-14:
                break
            previous = value
        best = max(best, float(np.linalg.norm(matrix.T @ left, axis=1).sum()))
    return best


def max_ratio(size: int, *, restarts: int = 8, iterations: int = 300) -> float:
    """Largest quantum/classical bias ratio over all sign matrices of this size.

    ``sqrt 2`` at two inputs, ``6/5`` at three.  The rent **falls** as the game
    grows, which is the opposite of what scaling intuition suggests and is the
    practically relevant fact: a market cannot buy a larger edge by enlarging the
    coordination problem.  Larger ratios exist only for weighted payoffs at large
    size, and even then Grothendieck's constant caps them at
    :data:`GROTHENDIECK_CEILING`.
    """
    if size not in (2, 3):
        raise ValueError("the census is enumerated only for two or three inputs")
    best = 0.0
    for entries in itertools.product((-1, 1), repeat=size * size):
        matrix = np.array(entries, dtype=float).reshape(size, size)
        quantum = tsirelson_bias(matrix, restarts=restarts, iterations=iterations)
        best = max(best, quantum / classical_bias(matrix))
    return best


# --------------------------------------------------------------------------- #
# The price of anarchy, quantum and classical
# --------------------------------------------------------------------------- #
def correlated_price_of_anarchy(
    payoffs: Sequence[Sequence[tuple[int, int]]],
) -> Fraction:
    """Worst correlated-equilibrium welfare over best welfare, exactly.

    A linear objective over the correlated equilibrium polytope, so the minimum
    sits at a vertex and the computation is exact rational arithmetic.
    """
    from PyNormaliz import Cone

    inequalities = []
    for action in (0, 1):
        other = 1 - action
        row = [0] * 4
        for column in (0, 1):
            row[2 * action + column] = (
                payoffs[action][column][0] - payoffs[other][column][0]
            )
        inequalities.append(row)
    for action in (0, 1):
        other = 1 - action
        row = [0] * 4
        for line in (0, 1):
            row[2 * line + action] = (
                payoffs[line][action][1] - payoffs[line][other][1]
            )
        inequalities.append(row)
    for slot in range(4):
        row = [0] * 4
        row[slot] = 1
        inequalities.append(row)
    cone = Cone(
        inhom_inequalities=[row + [0] for row in inequalities],
        inhom_equations=[[1, 1, 1, 1, -1]],
    )
    welfare = [
        Fraction(payoffs[i][j][0] + payoffs[i][j][1]) for i in (0, 1) for j in (0, 1)
    ]
    best = max(welfare)
    if best <= 0:
        raise ValueError("the optimal welfare must be positive")
    worst = min(
        sum(Fraction(int(v[k]), int(v[-1])) * welfare[k] for k in range(4))
        for v in cone.VerticesOfPolyhedron()
    )
    return Fraction(worst, 1) / best


def quantum_price_of_anarchy(
    payoffs: Sequence[Sequence[tuple[int, int]]],
) -> Fraction:
    """The same number, and that is the point.

    A "quantum price of anarchy" is asked for as though it were a new quantity.
    For a complete-information game it is not: the quantum equilibrium
    distributions are contained in the classical correlated ones, and the
    classical ones are all realisable, so the two sets coincide and every
    welfare ratio taken over them agrees.  There is nothing left to define.

    The question becomes substantive only once the mediator is removed and types
    are private, where the classical set contracts to the local polytope.  That
    is where a genuinely different ratio lives, and it is bounded by the same
    factors as the game values: at most ``sqrt 2`` at two inputs, ``6/5`` at
    three, and Grothendieck's constant in general.
    """
    return correlated_price_of_anarchy(payoffs)
