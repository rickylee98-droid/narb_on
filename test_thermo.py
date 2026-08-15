"""Referees for `thermo`.

The load-bearing checks:

  * the mean current read off the cumulant generating function must equal the
    steady-state cycle flux, computed by a disjoint path;
  * the asymptotic coefficient ``2`` in ``Sdot -> 2 gamma ln(1/eps)`` is checked
    by confirming that neighbouring integers *fail*, so it is derived rather
    than fitted;
  * the floating-point orbit used near the threshold is checked against the
    exact rational map;
  * the TUR is checked to hold, and separately checked not to be vacuous.
"""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp
import pytest

import thermo


# ---------------------------------------------------------------------------
# the demon bit
# ---------------------------------------------------------------------------


class TestDemonBitConstruction:
    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_non_positive_bath_rate(self, bad: float):
        with pytest.raises(ValueError, match="bath rate"):
            thermo.DemonBit(bath_rate=bad, demon_rate=1.0, affinity=1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_non_positive_demon_rate(self, bad: float):
        with pytest.raises(ValueError, match="demon rate"):
            thermo.DemonBit(bath_rate=1.0, demon_rate=bad, affinity=1.0)

    def test_rejects_negative_affinity(self):
        with pytest.raises(ValueError, match="affinity"):
            thermo.DemonBit(bath_rate=1.0, demon_rate=1.0, affinity=-1.0)

    def test_zero_affinity_is_allowed(self):
        bit = thermo.DemonBit(bath_rate=1.0, demon_rate=1.0, affinity=0.0)
        assert bit.error_rate() == pytest.approx(0.5, abs=1e-12)


class TestSteadyState:
    @pytest.mark.parametrize("affinity", [0.0, 0.5, 1.0, 4.0, 10.0, 30.0])
    def test_normalisation(self, affinity: float):
        occupancy = thermo.DemonBit(1.0, 1.0, affinity).steady_state()
        assert float(sum(occupancy)) == pytest.approx(1.0, abs=1e-25)

    @pytest.mark.parametrize("affinity", [0.0, 1.0, 5.0, 20.0])
    def test_occupations_are_probabilities(self, affinity: float):
        for value in thermo.DemonBit(1.0, 1.0, affinity).steady_state():
            assert 0 <= float(value) <= 1

    def test_error_decreases_with_affinity(self):
        errors = [
            float(thermo.DemonBit(1.0, 1.0, affinity).error_rate())
            for affinity in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
        ]
        assert errors == sorted(errors, reverse=True)

    def test_no_correction_means_maximal_error(self):
        assert float(thermo.DemonBit(1.0, 1.0, 0.0).error_rate()) == pytest.approx(
            0.5, abs=1e-25
        )

    @pytest.mark.parametrize("affinity", [20.0, 30.0, 40.0])
    def test_error_vanishes_at_strong_drive(self, affinity: float):
        assert float(thermo.DemonBit(1.0, 1.0, affinity).error_rate()) < 1e-4


class TestFlux:
    @pytest.mark.parametrize("affinity", [1.0, 4.0, 10.0, 25.0])
    def test_flux_is_bounded_by_the_noise_rate(self, affinity: float):
        """The demon cannot correct faster than the bath corrupts."""
        bit = thermo.DemonBit(bath_rate=1.0, demon_rate=1.0, affinity=affinity)
        assert float(abs(bit.cycle_flux())) <= 1.0 + 1e-20

    @pytest.mark.parametrize("bath_rate", [0.5, 1.0, 3.0])
    def test_flux_saturates_at_the_noise_rate(self, bath_rate: float):
        """The gap closes as the drive grows, rather than at any fixed tolerance.

        Saturation is asymptotic in the affinity and the residual carries a
        factor of ``bath_rate / demon_rate``, so a single absolute tolerance
        passes at one noise rate and fails at another -- which is exactly how
        the first version of this test failed, at ``bath_rate = 3``.  Checking
        that the gap *shrinks* is the claim that was actually meant.
        """
        gaps = []
        for affinity in (20.0, 30.0, 40.0, 50.0):
            bit = thermo.DemonBit(bath_rate, demon_rate=1.0, affinity=affinity)
            gaps.append(abs(float(abs(bit.cycle_flux())) - bath_rate) / bath_rate)
        assert gaps == sorted(gaps, reverse=True)
        assert gaps[-1] < 1e-9

    def test_no_flux_at_equilibrium(self):
        assert float(thermo.DemonBit(1.0, 1.0, 0.0).cycle_flux()) == pytest.approx(
            0.0, abs=1e-25
        )

    def test_no_dissipation_at_equilibrium(self):
        bit = thermo.DemonBit(1.0, 1.0, 0.0)
        assert float(bit.entropy_production_rate()) == pytest.approx(0.0, abs=1e-25)

    @pytest.mark.parametrize("affinity", [0.5, 1.0, 2.0, 4.0, 8.0])
    def test_dissipation_is_non_negative(self, affinity: float):
        assert float(thermo.DemonBit(1.0, 1.0, affinity).entropy_production_rate()) >= 0


class TestGeneratingFunctionAgreesWithSteadyState:
    """The strongest referee here: two disjoint routes to the same current."""

    @pytest.mark.parametrize("affinity", [0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    @pytest.mark.parametrize("demon_rate", [0.5, 1.0, 2.0])
    def test_mean_current_matches_cycle_flux(
        self, affinity: float, demon_rate: float
    ):
        bit = thermo.DemonBit(1.0, demon_rate, affinity)
        assert float(abs(bit.current_mean())) == pytest.approx(
            float(abs(bit.cycle_flux())), rel=1e-12
        )

    @pytest.mark.parametrize("affinity", [1.0, 4.0, 10.0])
    def test_variance_is_positive(self, affinity: float):
        assert float(thermo.DemonBit(1.0, 1.0, affinity).current_variance()) > 0

    def test_cumulant_vanishes_at_zero_tilt(self):
        """The untilted generator has a zero eigenvalue: probability is conserved."""
        bit = thermo.DemonBit(1.0, 1.0, 3.0)
        assert float(bit._scaled_cumulant(mp.mpf(0))) == pytest.approx(0.0, abs=1e-25)


class TestThermodynamicUncertaintyRelation:
    @pytest.mark.parametrize("affinity", [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    @pytest.mark.parametrize("demon_rate", [0.5, 1.0, 4.0])
    def test_bound_holds(self, affinity: float, demon_rate: float):
        ratio = thermo.DemonBit(1.0, demon_rate, affinity).tur_ratio()
        assert float(ratio) >= thermo.TUR_BOUND - 1e-12

    def test_bound_is_nearly_saturated_near_equilibrium(self):
        """Not a vacuous inequality: the ratio approaches 2 as the drive weakens."""
        ratio = thermo.DemonBit(1.0, 1.0, 0.05).tur_ratio()
        assert float(ratio) == pytest.approx(thermo.TUR_BOUND, abs=1e-2)

    def test_ratio_grows_far_from_equilibrium(self):
        ratios = [
            float(thermo.DemonBit(1.0, 1.0, affinity).tur_ratio())
            for affinity in (0.5, 2.0, 8.0, 16.0)
        ]
        assert ratios == sorted(ratios)

    def test_undefined_at_equilibrium(self):
        with pytest.raises(ZeroDivisionError, match="equilibrium"):
            thermo.DemonBit(1.0, 1.0, 0.0).tur_ratio()

    def test_bound_value(self):
        assert thermo.TUR_BOUND == 2


class TestMaintenanceLaw:
    @pytest.mark.parametrize("affinity", [8.0, 16.0, 24.0, 32.0])
    def test_residual_shrinks(self, affinity: float):
        bit = thermo.DemonBit(1.0, 1.0, affinity)
        assert thermo.maintenance_law_residual(bit) < 0.05

    def test_residual_is_monotone_in_affinity(self):
        residuals = [
            thermo.maintenance_law_residual(thermo.DemonBit(1.0, 1.0, affinity))
            for affinity in (4.0, 8.0, 16.0, 24.0, 32.0)
        ]
        assert residuals == sorted(residuals, reverse=True)

    @pytest.mark.parametrize("wrong", [1, 3, 4])
    def test_neighbouring_coefficients_fail(self, wrong: int):
        """The ``2`` is derived, not fitted: nothing near it also works.

        Without this the residual test would pass for any coefficient that
        happened to be close, and the asymptotic claim would be decoration.
        """
        bit = thermo.DemonBit(bath_rate=1.0, demon_rate=1.0, affinity=32.0)
        exact = float(bit.entropy_production_rate())
        error = float(bit.error_rate())
        predicted = wrong * 1.0 * float(mp.log(1 / mp.mpf(error)))
        assert abs(exact - predicted) / predicted > 0.1

    def test_coefficient_value(self):
        assert thermo.MAINTENANCE_COEFFICIENT == 2

    @pytest.mark.parametrize("bath_rate", [0.5, 1.0, 2.0, 5.0])
    def test_law_is_linear_in_the_noise_rate(self, bath_rate: float):
        assert thermo.maintenance_rate_law(bath_rate, 1e-6) == pytest.approx(
            bath_rate * thermo.maintenance_rate_law(1.0, 1e-6), rel=1e-12
        )

    def test_law_is_logarithmic_not_singular(self):
        """Each decade of reliability costs the same fixed increment."""
        increments = [
            thermo.maintenance_rate_law(1.0, 10.0 ** -(power + 1))
            - thermo.maintenance_rate_law(1.0, 10.0**-power)
            for power in (1, 2, 3, 4, 5)
        ]
        for increment in increments:
            assert increment == pytest.approx(increments[0], rel=1e-12)

    @pytest.mark.parametrize("bad", [0.0, 0.5, 0.9, -0.1])
    def test_rejects_out_of_range_error(self, bad: float):
        with pytest.raises(ValueError):
            thermo.maintenance_rate_law(1.0, bad)

    def test_rejects_bad_bath_rate(self):
        with pytest.raises(ValueError):
            thermo.maintenance_rate_law(0.0, 1e-3)


# ---------------------------------------------------------------------------
# erasure: the claim that fails
# ---------------------------------------------------------------------------


class TestErasureIsBounded:
    def test_landauer_ceiling(self):
        assert thermo.LANDAUER_CEILING == pytest.approx(float(mp.log(2)), abs=1e-15)

    def test_perfect_erasure_costs_exactly_ln_two(self):
        assert thermo.erasure_work(0.0) == pytest.approx(
            thermo.LANDAUER_CEILING, abs=1e-15
        )

    def test_maximal_error_is_free(self):
        assert thermo.erasure_work(0.5) == pytest.approx(0.0, abs=1e-15)

    @pytest.mark.parametrize("error", [0.5, 0.25, 0.1, 1e-3, 1e-9, 1e-30, 0.0])
    def test_never_exceeds_the_ceiling(self, error: float):
        assert thermo.erasure_work(error) <= thermo.LANDAUER_CEILING + 1e-15

    def test_monotone_in_reliability(self):
        works = [
            thermo.erasure_work(error)
            for error in (0.5, 0.25, 0.1, 1e-2, 1e-4, 1e-8, 0.0)
        ]
        assert works == sorted(works)

    def test_the_refutation(self):
        """The brief's headline claim, computed and false.

        "As a system tries to become perfectly logical it must dissipate
        infinite heat" is not true of erasure: the cost rises monotonically to
        ``ln 2`` and stops.  The divergence the brief is reaching for belongs to
        *maintenance*, a different quantity.
        """
        assert thermo.erasure_cost_is_bounded() is True

    def test_binary_entropy_endpoints(self):
        assert thermo.binary_entropy(0.0) == 0.0
        assert thermo.binary_entropy(1.0) == 0.0

    def test_binary_entropy_maximum(self):
        assert thermo.binary_entropy(0.5) == pytest.approx(float(mp.log(2)), abs=1e-15)

    def test_binary_entropy_symmetry(self):
        for error in (0.1, 0.25, 0.4):
            assert thermo.binary_entropy(error) == pytest.approx(
                thermo.binary_entropy(1 - error), abs=1e-15
            )

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_binary_entropy_rejects_out_of_range(self, bad: float):
        with pytest.raises(ValueError):
            thermo.binary_entropy(bad)

    @pytest.mark.parametrize("bad", [-0.1, 0.6, 1.0])
    def test_erasure_rejects_out_of_range(self, bad: float):
        with pytest.raises(ValueError):
            thermo.erasure_work(bad)

    def test_bounded_check_rejects_tiny_sample(self):
        with pytest.raises(ValueError):
            thermo.erasure_cost_is_bounded(samples=1)


# ---------------------------------------------------------------------------
# the threshold: the transition that is real
# ---------------------------------------------------------------------------


class TestMajorityVote:
    @pytest.mark.parametrize(
        "point", [Fraction(0), Fraction(1, 2), Fraction(1)], ids=["0", "half", "1"]
    )
    def test_fixed_points(self, point: Fraction):
        assert thermo.majority_vote(point) == point

    def test_fixed_points_are_exactly_these_three(self):
        """``3p^2 - 2p^3 = p`` has no other root in ``[0, 1]``."""
        import sympy as sp

        variable = sp.Symbol("p")
        roots = sp.solve(sp.Eq(3 * variable**2 - 2 * variable**3, variable), variable)
        assert sorted(roots) == [0, sp.Rational(1, 2), 1]

    @pytest.mark.parametrize(
        "error", [Fraction(1, 10), Fraction(1, 4), Fraction(49, 100)]
    )
    def test_improves_below_threshold(self, error: Fraction):
        assert thermo.majority_vote(error) < error

    @pytest.mark.parametrize("error", [Fraction(51, 100), Fraction(9, 10)])
    def test_worsens_above_threshold(self, error: Fraction):
        assert thermo.majority_vote(error) > error

    def test_is_exact_in_rationals(self):
        assert thermo.majority_vote(Fraction(1, 3)) == Fraction(7, 27)

    @pytest.mark.parametrize("bad", [Fraction(-1, 2), Fraction(3, 2)])
    def test_rejects_out_of_range(self, bad: Fraction):
        with pytest.raises(ValueError):
            thermo.majority_vote(bad)


class TestCriticalPoint:
    def test_threshold_value(self):
        assert thermo.FAULT_TOLERANCE_THRESHOLD == Fraction(1, 2)

    def test_multiplier_is_three_halves(self):
        """``6 p (1 - p)`` at ``p = 1/2``, exactly."""
        assert thermo.critical_derivative(Fraction(1, 2)) == Fraction(3, 2)
        assert thermo.CRITICAL_MULTIPLIER == Fraction(3, 2)

    def test_multiplier_exceeds_one_so_the_point_is_unstable(self):
        assert thermo.CRITICAL_MULTIPLIER > 1

    def test_stable_fixed_points_have_zero_multiplier(self):
        assert thermo.critical_derivative(Fraction(0)) == 0
        assert thermo.critical_derivative(Fraction(1)) == 0

    def test_critical_slowing_down_slope(self):
        """Levels per decade must match ``ln 10 / ln(3/2)``.

        The absolute level count carries an offset from the doubly exponential
        phase that follows escape, so the slope is the honest check, not the
        intercept.
        """
        target = Fraction(1, 10**6)
        counts = []
        for power in (1, 2, 3, 4):
            distance = Fraction(1, 10**power)
            counts.append(
                thermo.concatenation_levels(
                    thermo.FAULT_TOLERANCE_THRESHOLD - distance, target
                )
            )
        measured = (counts[-1] - counts[0]) / 3
        predicted = float(mp.log(10) / mp.log(mp.mpf(3) / 2))
        assert measured == pytest.approx(predicted, rel=0.1)

    @pytest.mark.parametrize("bad", [0.0, 0.5, 0.9, -0.1])
    def test_slowing_rejects_out_of_range(self, bad: float):
        with pytest.raises(ValueError):
            thermo.critical_slowing_levels(bad)


class TestThresholdIsTheSingularity:
    @pytest.mark.parametrize(
        "error, finite",
        [
            (Fraction(1, 10), True),
            (Fraction(1, 4), True),
            (Fraction(49, 100), True),
            (Fraction(499, 1000), True),
            (Fraction(1, 2), False),
            (Fraction(51, 100), False),
            (Fraction(9, 10), False),
        ],
    )
    def test_finite_below_unreachable_above(self, error: Fraction, finite: bool):
        assert thermo.dissipation_is_finite(error, Fraction(1, 10**6)) is finite

    def test_the_transition_is_sharp(self):
        """Finite on one side of ``p = 1/2``, unattainable on the other."""
        assert thermo.threshold_is_sharp() is True

    @pytest.mark.parametrize("gap", [Fraction(1, 100), Fraction(1, 10**4)])
    def test_sharpness_survives_a_narrower_window(self, gap: Fraction):
        assert thermo.threshold_is_sharp(gap) is True

    def test_gate_overhead_is_three_to_the_levels(self):
        target = Fraction(1, 10**6)
        for error in (Fraction(1, 10), Fraction(1, 4), Fraction(4, 10)):
            levels = thermo.concatenation_levels(error, target)
            assert thermo.gate_overhead(error, target) == 3**levels

    def test_dissipation_is_two_erasures_per_gate(self):
        target = Fraction(1, 10**6)
        error = Fraction(1, 10)
        assert thermo.dissipation_to_target(error, target) == pytest.approx(
            2 * thermo.LANDAUER_CEILING * thermo.gate_overhead(error, target),
            rel=1e-15,
        )

    def test_overhead_exponent(self):
        assert thermo.OVERHEAD_EXPONENT == pytest.approx(
            float(mp.log(3) / mp.log(2)), abs=1e-15
        )
        assert 1.58 < thermo.OVERHEAD_EXPONENT < 1.59

    def test_levels_raise_above_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            thermo.concatenation_levels(Fraction(6, 10), Fraction(1, 10**6))

    def test_levels_raise_at_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            thermo.concatenation_levels(Fraction(1, 2), Fraction(1, 10**6))

    def test_levels_respect_the_iteration_limit(self):
        with pytest.raises(ValueError, match="within"):
            thermo.concatenation_levels(
                Fraction(499, 1000), Fraction(1, 10**6), limit=2
            )

    @pytest.mark.parametrize("bad", [Fraction(0), Fraction(1)])
    def test_levels_reject_degenerate_inputs(self, bad: Fraction):
        with pytest.raises(ValueError):
            thermo.concatenation_levels(bad, Fraction(1, 10**6))
        with pytest.raises(ValueError):
            thermo.concatenation_levels(Fraction(1, 10), bad)

    def test_sharpness_rejects_bad_gap(self):
        with pytest.raises(ValueError):
            thermo.threshold_is_sharp(Fraction(3, 4))


class TestFloatOrbitMatchesExactMap:
    """The precision promise made in `concatenation_levels`' docstring."""

    @pytest.mark.parametrize(
        "start", [Fraction(1, 10), Fraction(1, 4), Fraction(4, 10), Fraction(49, 100)]
    )
    def test_orbits_agree_for_the_first_levels(self, start: Fraction):
        exact = start
        approximate = mp.mpf(start.numerator) / start.denominator
        for _ in range(6):
            exact = thermo.majority_vote(exact)
            approximate = 3 * approximate**2 - 2 * approximate**3
            assert float(approximate) == pytest.approx(float(exact), rel=1e-20)


class TestWhatThisDoesNotClaim:
    """The boundary, asserted so it cannot drift."""

    def test_no_godel_statement_appears(self):
        """No incompleteness result enters this module at any point.

        The brief's framing makes logical incompleteness the source of the
        thermodynamic cost.  Nothing computed here needs it: the maintenance
        law, the erasure bound, the TUR and the threshold are all statements
        about noise, current and gate count.  The Goedel half contributes
        nothing, and pretending otherwise would be the overclaim.
        """
        assert not hasattr(thermo, "incompleteness")
        assert not hasattr(thermo, "halting")
        assert not hasattr(thermo, "self_reference")

    def test_the_two_singularities_are_unrelated(self):
        """One is smooth and logarithmic, the other sharp. They are not one thing.

        Conflating them is the brief's central error.  The maintenance cost
        diverges as ``2 gamma ln(1/eps)`` with every decade costing the same
        increment; the threshold is finite on one side of a parameter value and
        unattainable on the other.
        """
        first = thermo.maintenance_rate_law(1.0, 1e-9) - thermo.maintenance_rate_law(
            1.0, 1e-8
        )
        second = thermo.maintenance_rate_law(1.0, 1e-15) - thermo.maintenance_rate_law(
            1.0, 1e-14
        )
        assert first == pytest.approx(second, rel=1e-12)
        assert thermo.threshold_is_sharp() is True

    def test_the_tur_says_nothing_about_logic(self):
        """It bounds a current against dissipation. That is all it does."""
        bit = thermo.DemonBit(1.0, 1.0, 2.0)
        assert float(bit.tur_ratio()) >= thermo.TUR_BOUND

    def test_no_quantum_error_correction_is_modelled(self):
        """The threshold here is the classical majority-vote one."""
        assert not hasattr(thermo, "surface_code")
        assert not hasattr(thermo, "stabiliser")
