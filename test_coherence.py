"""Referees for the mode architecture and its interaction graph.

Four checks the module does not control:

* The triad list is found by exhaustive search over every unordered triple of
  modes, with nothing assumed and nothing filtered. If a local or long-range
  triad existed, the search would find it.
* The recursion's closure -- that ``(d+e)`` and ``(e-d)`` are again orthogonal and
  of equal length -- is verified at every shell, not just asserted once.
* The cap threshold is computed from the geometry of ``120``-degree openings, and
  the wrong answer (``60``, from forgetting the antipodal cap) is kept as a test
  so the correction stays visible.
* The mode budget saturates at ``alpha = 5/2``, which is the top of the
  three-dimensional intermittency range by an unrelated argument.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

import coherence as co

SEED = (3, 4, 0)
PARTNER = (0, 0, 5)


def build(factor: int = 70, shells: int = 3) -> co.Architecture:
    return co.architecture(SEED, PARTNER, [factor] * shells)


class TestArchitecture:
    def test_the_seed_pair_is_orthogonal_and_equal_length(self) -> None:
        assert sum(a * b for a, b in zip(SEED, PARTNER)) == 0
        assert sum(a * a for a in SEED) == sum(b * b for b in PARTNER)

    def test_the_recursion_stays_closed(self) -> None:
        """``(d+e) . (e-d) = |e|^2 - |d|^2 = 0`` at every step, checked exactly."""
        d, e = SEED, PARTNER
        for _ in range(6):
            combined, difference = co.orthogonal_equal_length(d, e)
            assert sum(a * b for a, b in zip(combined, difference)) == 0
            assert sum(a * a for a in combined) == sum(b * b for b in difference)
            d, e = combined, difference

    def test_a_bad_seed_is_refused(self) -> None:
        with pytest.raises(ValueError, match="orthogonal"):
            co.orthogonal_equal_length((1, 0, 0), (1, 0, 0))
        with pytest.raises(ValueError, match="equal length"):
            co.orthogonal_equal_length((1, 0, 0), (0, 2, 0))
        with pytest.raises(ValueError, match="orthogonal and of equal length"):
            co.architecture((1, 0, 0), (0, 2, 0), [2])

    def test_every_shell_is_closed_under_negation(self) -> None:
        """Reality is not optional: a velocity field is real."""
        arch = build()
        for shell in arch.shells:
            for mode in shell:
                assert tuple(-x for x in mode) in shell

    def test_a_shell_not_closed_under_negation_is_refused(self) -> None:
        with pytest.raises(ValueError, match="closed under negation"):
            co.Architecture((((1, 0, 0),),))

    def test_shells_separate(self) -> None:
        arch = build()
        radii = [arch.radius(j) for j in range(len(arch.shells))]
        assert all(later > 10 * earlier for earlier, later in zip(radii, radii[1:]))

    def test_a_degenerate_growth_factor_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            co.architecture(SEED, PARTNER, [0])


class TestInteractionGraph:
    """The result: the complete triad list is the Obukhov graph."""

    @pytest.mark.parametrize("factor", [7, 70, 700])
    @pytest.mark.parametrize("shells", [2, 3, 4])
    def test_only_nearest_neighbour_gates_close(self, factor, shells) -> None:
        census = co.triad_census(co.architecture(SEED, PARTNER, [factor] * shells))
        assert census["local"] == 0
        assert census["long-range"] == 0
        assert census["gate"] == 2 * shells

    def test_the_search_is_exhaustive(self) -> None:
        """Every triple is examined, so an unwanted triad could not hide."""
        arch = build(shells=3)
        modes = list(arch.modes())
        assert len(modes) == 2 + 4 * 3
        triads = co.closing_triads(arch)
        assert all(
            sum(a + b + c for a, b, c in zip(*(mode for _, mode in triad))) == 0
            for triad in triads
        )

    def test_the_classification_covers_the_cases(self) -> None:
        assert co.classify_triad([(2, (0, 0, 0))] * 3) == "local"
        assert co.classify_triad([(1, (0, 0, 0)), (2, (0, 0, 0)), (2, (0, 0, 0))]) == "gate"
        assert (
            co.classify_triad([(0, (0, 0, 0)), (2, (0, 0, 0)), (2, (0, 0, 0))])
            == "long-range"
        )

    def test_the_difference_set_meets_only_the_neighbour(self) -> None:
        """The reduction, checked directly rather than through the triad search."""
        arch = build(shells=3)
        for j in range(1, len(arch.shells)):
            differences = co.difference_set(arch.shells[j])
            assert differences & set(arch.shells[j - 1]), "the gate must exist"
            assert not differences & set(arch.shells[j]), "no local triads"
            for i in range(j - 1):
                assert not differences & set(arch.shells[i]), "no long-range triads"


class TestGateGeometry:
    def test_one_high_mode_sits_at_exactly_forty_five_degrees(self) -> None:
        arch = build()
        assert any(abs(angle - 45.0) < 1e-9 for angle in co.gate_angles(arch))

    @pytest.mark.parametrize(
        "factor,bound", [(7, 5.0), (70, 0.5), (700, 0.05), (7000, 0.005)]
    )
    def test_the_deviation_falls_with_separation(self, factor, bound) -> None:
        """``O(N_{j-1}/N_j)``: a decade of separation buys a decade of angle."""
        assert co.gate_angle_deviation(co.architecture(SEED, PARTNER, [factor] * 2)) < bound

    def test_the_deviation_cannot_be_zero_at_finite_separation(self) -> None:
        """The two high modes differ by the low one, so they cannot share an angle.

        Recorded because claiming every gate is at exactly 45 degrees was this
        module's first phrasing, and it is false for a structural reason.
        """
        assert co.gate_angle_deviation(build(factor=7)) > 0

    def test_an_architecture_with_no_gates_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no gates"):
            co.gate_angle_deviation(co.architecture(SEED, PARTNER, []))


class TestCapThreshold:
    @pytest.mark.parametrize("half_angle", [1, 10, 20, 25, 29, 29.9])
    def test_narrow_caps_carry_no_local_triads(self, half_angle) -> None:
        assert co.cap_is_sum_free(half_angle)

    @pytest.mark.parametrize("half_angle", [30, 31, 45, 60, 75, 90])
    def test_wide_caps_do(self, half_angle) -> None:
        assert not co.cap_is_sum_free(half_angle)

    def test_the_threshold_is_thirty_not_sixty(self) -> None:
        """The correction, kept as a test.

        A cap alone spans pairwise angles ``[0, 2 phi]``, which reaches ``120``
        only at ``phi = 60``. Reality adds the antipodal cap, spanning
        ``[180 - 2 phi, 180]``, which reaches ``120`` already at ``phi = 30``.
        The first reasoning gives ``60`` and is wrong.
        """
        assert co.SUM_FREE_CAP_LIMIT_DEGREES == 30
        assert co.cap_is_sum_free(29.9) and not co.cap_is_sum_free(30.1)
        within_only = 2 * 45 >= co.LOCAL_TRIAD_ANGLE_DEGREES
        assert not within_only, "a 45-degree cap alone would look safe"
        assert not co.cap_is_sum_free(45), "but with its antipode it is not"

    def test_the_local_triad_angle(self) -> None:
        """``|a + b| = |a| = |b|`` forces ``cos = -1/2``, so ``120`` degrees."""
        assert co.LOCAL_TRIAD_ANGLE_DEGREES == 120
        assert math.isclose(math.cos(math.radians(120)), -0.5, abs_tol=1e-12)

    def test_a_degenerate_cap_is_refused(self) -> None:
        for bad in (0, -5, 120):
            with pytest.raises(ValueError, match="half-angle"):
                co.cap_admits_local_triad(bad)


class TestModeBudget:
    @pytest.mark.parametrize(
        "alpha,fits", [(1, True), (Fraction(3, 2), True), (2, True), (Fraction(5, 2), False), (3, False)]
    )
    def test_the_budget_fits_below_five_halves(self, alpha, fits) -> None:
        assert co.budget_fits(alpha) is fits

    def test_it_saturates_exactly_at_the_top_of_the_range(self) -> None:
        """``2(alpha-1) = 3`` gives ``5/2``, the intermittency limit by another route."""
        alpha = co.saturating_alpha()
        assert alpha == Fraction(5, 2)
        assert co.mode_budget_exponent(alpha) == co.cap_capacity_exponent()

    def test_the_space_filling_case_needs_no_modes(self) -> None:
        assert co.mode_budget_exponent(1) == 0

    def test_the_budget_grows_with_intermittency(self) -> None:
        exponents = [co.mode_budget_exponent(a) for a in (1, Fraction(3, 2), 2, Fraction(5, 2))]
        assert all(later > earlier for earlier, later in zip(exponents, exponents[1:]))
