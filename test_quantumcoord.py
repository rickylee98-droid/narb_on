"""Referees for the quantum coordination module.

Four checks the module does not control:

* The local polytope is built from nothing but the sixteen deterministic
  strategies, and must come out with the textbook 24 facets in dimension 8 --
  sixteen positivity constraints and the eight CHSH inequalities.
* The correlated equilibrium polytope of the Prisoner's Dilemma must have exactly
  one vertex, since defect-defect is its unique correlated equilibrium. That is a
  known answer the construction could easily miss, and the first version did,
  by omitting non-negativity.
* The quantum and shared-coin marginals must agree exactly, not approximately.
  The forensic conclusion rests on an identity, so a small number would mean a
  bug rather than a weak signal.
* The XOR census must split cleanly by matrix rank, with every advantaged game at
  the same ratio. A continuum of ratios would falsify the dichotomy.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

import quantumcoord as qc


class TestLocalPolytope:
    def test_there_are_sixteen_deterministic_strategies(self) -> None:
        strategies = qc.deterministic_strategies()
        assert len(strategies) == 16
        assert all(len(point) == 16 for point in strategies)

    def test_each_strategy_is_a_normalised_distribution(self) -> None:
        for point in qc.deterministic_strategies():
            for x in (0, 1):
                for y in (0, 1):
                    block = sum(
                        point[i]
                        for i, (_, _, xx, yy) in enumerate(qc.OUTCOMES)
                        if xx == x and yy == y
                    )
                    assert block == 1

    def test_the_textbook_facet_count(self) -> None:
        """24 facets, dimension 8: positivity plus the eight CHSH inequalities."""
        assert qc.local_polytope_facets() == 24
        assert qc.local_polytope_dimension() == 8
        assert qc.chsh_facet_count() == 8


class TestGameValues:
    def test_the_classical_value_is_three_quarters(self) -> None:
        assert qc.classical_chsh_value() == Fraction(3, 4)

    def test_the_quantum_value_is_tsirelson(self) -> None:
        assert qc.quantum_chsh_value() == pytest.approx((2 + math.sqrt(2)) / 4)
        assert qc.quantum_chsh_value() > float(qc.classical_chsh_value())

    def test_the_quantum_distribution_realises_that_value(self) -> None:
        """Computed from the distribution, not quoted."""
        distribution = qc.quantum_distribution()
        won = sum(
            distribution[i]
            for i, (a, b, x, y) in enumerate(qc.OUTCOMES)
            if (a ^ b) == (x & y)
        ) / 4
        assert won == pytest.approx(qc.quantum_chsh_value())

    def test_the_quantum_distribution_is_a_valid_distribution(self) -> None:
        assert qc.quantum_point_is_classically_realisable()


class TestTheForensicResult:
    def test_the_quantum_marginals_are_uniform(self) -> None:
        """Independent of both inputs: no-signalling, exactly."""
        for value in qc.quantum_marginals().values():
            assert value == pytest.approx(0.5, abs=1e-12)

    def test_a_classical_pair_matches_them_exactly(self) -> None:
        assert qc.quantum_marginals().keys() == qc.shared_coin_marginals().keys()
        for key, value in qc.quantum_marginals().items():
            assert abs(value - qc.shared_coin_marginals()[key]) < 1e-12

    def test_the_marginal_divergence_is_zero(self) -> None:
        """Not small -- zero. The conclusion is an identity, not a weak signal."""
        assert qc.marginal_divergence() == 0.0

    def test_the_joint_divergence_is_positive_and_reproducible(self) -> None:
        """Convex, so different starts must land on the same minimum."""
        values = [qc.detection_divergence(seed=seed) for seed in (0, 1, 2)]
        assert all(value > 0 for value in values)
        assert max(values) - min(values) < 1e-6

    def test_detection_needs_hundreds_of_rounds_not_millions(self) -> None:
        assert 100 < qc.rounds_to_detect(1e-3) < 1000
        assert qc.rounds_to_detect(1e-6) > qc.rounds_to_detect(1e-3)

    def test_the_two_channels_disagree_completely(self) -> None:
        """The dichotomy: zero information one side, ample information jointly."""
        assert qc.marginal_divergence() == 0.0
        assert qc.detection_divergence() > 1e-3

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 2.0])
    def test_a_bad_confidence_is_refused(self, bad) -> None:
        with pytest.raises(ValueError, match="probability"):
            qc.rounds_to_detect(bad)

    def test_a_non_positive_divergence_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no detection"):
            qc.rounds_to_detect(0.5, divergence=0.0)


class TestTheDichotomy:
    def test_half_the_xor_games_have_advantage(self) -> None:
        census = qc.xor_game_census()
        assert len(census) == 16
        assert sum(1 for game in census if game.has_advantage) == 8

    def test_every_advantage_is_exactly_root_two(self) -> None:
        """No continuum of quantum rents: one value, or none."""
        ratios = {
            round(game.ratio, 10) for game in qc.xor_game_census() if game.has_advantage
        }
        assert ratios == {round(qc.XOR_ADVANTAGE_RATIO, 10)}

    def test_the_split_is_by_rank(self) -> None:
        for game in qc.xor_game_census():
            assert game.has_advantage == (game.rank == 2)

    def test_the_unadvantaged_games_have_ratio_one(self) -> None:
        for game in qc.xor_game_census():
            if not game.has_advantage:
                assert game.ratio == pytest.approx(1.0)


class TestTheCorrelatedEquilibriumCorrection:
    def test_the_prisoners_dilemma_has_one_correlated_equilibrium(self) -> None:
        """Defect-defect, uniquely. The check that caught a missing constraint.

        The first version of the polytope omitted non-negativity and returned a
        single vertex for a coordination game too, which is wrong; this game is
        the one whose right answer really is one.
        """
        vertices, _ = qc.correlated_equilibrium_polytope(
            [[(3, 3), (0, 5)], [(5, 0), (1, 1)]]
        )
        assert vertices == 1

    def test_a_coordination_game_has_several(self) -> None:
        """The control: a game whose correlated equilibrium set is genuinely big."""
        vertices, facets = qc.correlated_equilibrium_polytope(
            [[(2, 2), (0, 0)], [(0, 0), (1, 1)]]
        )
        assert vertices > 1
        assert facets >= 4

    def test_the_quantum_point_is_a_distribution(self) -> None:
        """The one-line containment: quantum devices emit probability distributions.

        Which is why the advantage cannot live in the correlated equilibrium set
        of a complete-information game, and must come from removing the mediator.
        """
        assert qc.quantum_point_is_classically_realisable()


class TestDetectionThreshold:
    """The second threshold: how much of the tape must be visible."""

    def test_the_maximal_case_is_exactly_two_root_two_minus_one(self) -> None:
        """``2(sqrt 2 - 1)``, bisected to eight digits against the closed form."""
        assert qc.detection_threshold() == pytest.approx(qc.MAXIMAL_THRESHOLD, abs=1e-8)
        assert qc.MAXIMAL_THRESHOLD == pytest.approx(2 / (1 + math.sqrt(2)))

    def test_the_maximal_advantage_is_tsirelson(self) -> None:
        assert qc.quantum_advantage() == pytest.approx(
            qc.quantum_chsh_value() - 0.75, abs=1e-6
        )

    @pytest.mark.parametrize("theta", [0.5, 0.3, 0.1])
    def test_weaker_entanglement_lowers_the_threshold(self, theta) -> None:
        assert qc.detection_threshold(theta) < qc.MAXIMAL_THRESHOLD

    def test_the_threshold_approaches_the_eberhard_floor(self) -> None:
        """``2/3`` in the vanishing-entanglement limit, and never below it."""
        near = qc.detection_threshold(0.02)
        assert near > qc.EBERHARD_FLOOR
        assert near - qc.EBERHARD_FLOOR < 0.01

    def test_the_frontier_is_monotone(self) -> None:
        """The practical content: effect size and observation quality move together.

        There is no state in the family that is both very profitable and cheap to
        catch, so a regulator cannot trade one against the other.
        """
        assert qc.frontier_is_monotone()

    def test_the_cheap_to_catch_end_is_worthless(self) -> None:
        """At the Eberhard end the advantage is negligible."""
        _, threshold, advantage = qc.frontier([0.02])[0]
        assert threshold < 0.68
        assert advantage < 1e-3

    def test_the_profitable_end_needs_most_of_the_tape(self) -> None:
        _, threshold, advantage = qc.frontier([math.pi / 4])[0]
        assert threshold > 0.82
        assert advantage > 0.10

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.0, 2.0])
    def test_an_out_of_range_state_is_refused(self, bad) -> None:
        with pytest.raises(ValueError, match="pi/4"):
            qc.detection_threshold(bad)
        with pytest.raises(ValueError, match="pi/4"):
            qc.quantum_advantage(bad)


class TestTheRentDoesNotScale:
    """Enlarging the coordination game shrinks the edge."""

    def test_the_optimiser_reproduces_chsh(self) -> None:
        """The licence for reading anything off the census: exact ``2 sqrt 2``."""
        import numpy as np

        chsh = np.array([[1.0, 1.0], [1.0, -1.0]])
        assert qc.classical_bias(chsh) == pytest.approx(2.0)
        assert qc.tsirelson_bias(chsh) == pytest.approx(2 * math.sqrt(2), abs=1e-9)

    def test_two_inputs_max_at_root_two(self) -> None:
        assert qc.max_ratio(2) == pytest.approx(qc.MAX_RATIO_TWO_INPUTS, abs=1e-7)

    def test_three_inputs_max_at_six_fifths(self) -> None:
        assert qc.max_ratio(3) == pytest.approx(qc.MAX_RATIO_THREE_INPUTS, abs=1e-5)

    def test_the_rent_falls_with_game_size(self) -> None:
        """The finding, and it runs against scaling intuition."""
        assert qc.max_ratio(3) < qc.max_ratio(2)
        assert qc.MAX_RATIO_THREE_INPUTS < qc.MAX_RATIO_TWO_INPUTS

    def test_everything_sits_under_the_grothendieck_ceiling(self) -> None:
        """A universal cap on the rent for XOR games of any size."""
        assert qc.MAX_RATIO_TWO_INPUTS < qc.GROTHENDIECK_CEILING
        assert qc.MAX_RATIO_THREE_INPUTS < qc.GROTHENDIECK_CEILING
        assert qc.MAX_RATIO_TWO_INPUTS / qc.GROTHENDIECK_CEILING > 0.75

    def test_the_quantum_bias_never_falls_below_the_classical_one(self) -> None:
        """Quantum strategies include classical ones -- the check that caught the
        nuclear-norm bug, applied to the general-size optimiser."""
        import itertools

        import numpy as np

        for entries in itertools.product((-1, 1), repeat=4):
            matrix = np.array(entries, dtype=float).reshape(2, 2)
            assert qc.tsirelson_bias(matrix) >= qc.classical_bias(matrix) - 1e-9

    def test_an_unenumerated_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="two or three"):
            qc.max_ratio(4)


class TestPriceOfAnarchy:
    """The corollary that dissolves the 'quantum price of anarchy' question."""

    def test_the_prisoners_dilemma_is_one_third(self) -> None:
        """Welfare 2 at the unique correlated equilibrium against 6 at the optimum."""
        value = qc.correlated_price_of_anarchy([[(3, 3), (0, 5)], [(5, 0), (1, 1)]])
        assert value == Fraction(1, 3)

    @pytest.mark.parametrize(
        "game",
        [
            [[(3, 3), (0, 5)], [(5, 0), (1, 1)]],
            [[(2, 2), (0, 0)], [(0, 0), (1, 1)]],
            [[(4, 1), (0, 0)], [(0, 0), (1, 4)]],
        ],
    )
    def test_the_quantum_ratio_is_the_classical_one(self, game) -> None:
        """Not approximately -- identically, because the sets coincide.

        There is no separate quantity to define for a complete-information game.
        """
        assert qc.quantum_price_of_anarchy(game) == qc.correlated_price_of_anarchy(game)

    def test_the_ratio_is_a_proper_fraction(self) -> None:
        for game in ([[(3, 3), (0, 5)], [(5, 0), (1, 1)]],
                     [[(2, 2), (0, 0)], [(0, 0), (1, 1)]]):
            value = qc.correlated_price_of_anarchy(game)
            assert isinstance(value, Fraction)
            assert 0 < value <= 1

    def test_a_degenerate_game_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            qc.correlated_price_of_anarchy([[(0, 0), (0, 0)], [(0, 0), (0, 0)]])
