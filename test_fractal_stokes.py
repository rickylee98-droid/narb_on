"""Tests for line integrals of Holder forms over Koch curves.

The geometry is pinned against quantities known in closed form -- the arclength
``(4r)^n``, the snowflake area ``8/5`` of its triangle -- so that the integrator
is refereed by something other than itself.  The coherence measurement is pinned
against the smooth control, which must return the coherent exponent ``beta = 1``
exactly and Young's rate ``4 r^2``; a measurement that cannot recover the case
where the answer is known has no standing in the cases where it is not.

The resolution arithmetic is tested as arithmetic, because it is the part that
decides which results are admissible at all.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import fractal_stokes as fs

SQRT3 = math.sqrt(3.0)
ANGLES = [0.5, 0.8, fs.KOCH_ANGLE, 1.2, 1.4]


# --------------------------------------------------------------------------- #
# The curve family
# --------------------------------------------------------------------------- #
class TestKochGeometry:
    def test_standard_angle_gives_the_classical_ratio(self) -> None:
        assert fs.koch_ratio() == pytest.approx(1.0 / 3.0)
        assert fs.koch_dimension() == pytest.approx(math.log(4.0) / math.log(3.0))

    def test_dimension_runs_from_one_to_two(self) -> None:
        assert fs.koch_dimension(1e-6) == pytest.approx(1.0, abs=1e-6)
        assert fs.koch_dimension(math.pi / 2 - 1e-9) == pytest.approx(2.0, abs=1e-6)

    def test_dimension_is_increasing_in_the_apex_angle(self) -> None:
        values = [fs.koch_dimension(angle) for angle in ANGLES]
        assert values == sorted(values)

    @pytest.mark.parametrize("angle", [0.0, math.pi / 2, -0.1, 2.0])
    def test_rejects_angles_outside_the_open_quadrant(self, angle: float) -> None:
        with pytest.raises(ValueError, match="angle"):
            fs.koch_ratio(angle)

    def test_rejects_a_negative_level(self) -> None:
        with pytest.raises(ValueError, match="level"):
            fs.koch_curve(-1)

    @pytest.mark.parametrize("level", range(6))
    def test_vertex_count_and_endpoints(self, level: int) -> None:
        curve = fs.koch_curve(level)
        assert curve.shape == (4**level + 1, 2)
        assert curve[0] == pytest.approx([0.0, 0.0])
        assert curve[-1] == pytest.approx([1.0, 0.0])

    @pytest.mark.parametrize("angle", ANGLES)
    @pytest.mark.parametrize("level", range(6))
    def test_arclength_is_exactly_four_r_to_the_level(
        self, level: int, angle: float
    ) -> None:
        curve = fs.koch_curve(level, angle)
        expected = (4.0 * fs.koch_ratio(angle)) ** level
        assert fs.polygon_arclength(curve) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("angle", ANGLES)
    @pytest.mark.parametrize("level", range(5))
    def test_every_segment_at_a_level_has_the_same_length(
        self, level: int, angle: float
    ) -> None:
        curve = fs.koch_curve(level, angle)
        lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        assert lengths == pytest.approx(fs.koch_ratio(angle) ** level, rel=1e-12)

    @pytest.mark.parametrize("level", range(5))
    def test_a_finer_curve_refines_a_coarser_one(self, level: int) -> None:
        coarse = fs.koch_curve(level)
        fine = fs.koch_curve(level + 1)
        assert fine[::4] == pytest.approx(coarse, abs=1e-12)


class TestSnowflake:
    @pytest.mark.parametrize("level", range(6))
    def test_snowflake_is_closed(self, level: int) -> None:
        flake = fs.koch_snowflake(level)
        assert flake[0] == pytest.approx(flake[-1], abs=1e-12)

    @pytest.mark.parametrize("level", range(8))
    def test_area_matches_the_exact_finite_level_formula(self, level: int) -> None:
        """``A_n = T (1 + (3/5)(1 - (4/9)^n))`` with ``T`` the triangle's area.

        Stronger than checking the ``8/5 T`` limit: it pins every level, so a
        construction that converged to the right value by the wrong route would
        still fail.
        """
        triangle = SQRT3 / 4.0
        expected = triangle * (1.0 + 0.6 * (1.0 - (4.0 / 9.0) ** level))
        area = abs(fs.polygon_signed_area(fs.koch_snowflake(level)))
        assert area == pytest.approx(expected, rel=1e-11)

    def test_area_converges_to_eight_fifths_of_the_triangle(self) -> None:
        triangle = SQRT3 / 4.0
        area = abs(fs.polygon_signed_area(fs.koch_snowflake(7)))
        assert area == pytest.approx(1.6 * triangle, rel=2e-3)

    def test_area_is_monotone_while_arclength_diverges(self) -> None:
        areas = [abs(fs.polygon_signed_area(fs.koch_snowflake(n))) for n in range(6)]
        lengths = [fs.polygon_arclength(fs.koch_snowflake(n)) for n in range(6)]
        assert areas == sorted(areas)
        assert lengths == sorted(lengths)
        assert lengths[-1] > 5.0 * lengths[0] / 3.0

    @pytest.mark.parametrize("level", [3, 5, 6])
    def test_integrating_x_dy_reproduces_the_shoelace_area(self, level: int) -> None:
        """The referee: the integrator and the closed-form area must agree.

        The midpoint rule is exact for a linear integrand on a segment, so this
        is an identity rather than an approximation, and it fails immediately if
        the orientation, the midpoint, or the increment is wrong.
        """
        flake = fs.koch_snowflake(level)
        form = fs.HolderForm(
            name="x dy", alpha=1.0, f=lambda x, y: np.zeros_like(x), g=lambda x, y: x
        )
        assert fs.riemann_stieltjes(flake, form) == pytest.approx(
            fs.polygon_signed_area(flake), rel=1e-12
        )

    def test_an_exact_form_integrates_to_zero_around_a_closed_curve(self) -> None:
        flake = fs.koch_snowflake(5)
        form = fs.HolderForm(
            name="d(xy)",
            alpha=1.0,
            f=lambda x, y: y,
            g=lambda x, y: x,
        )
        assert fs.riemann_stieltjes(flake, form) == pytest.approx(0.0, abs=1e-10)


# --------------------------------------------------------------------------- #
# Resolution arithmetic
# --------------------------------------------------------------------------- #
class TestResolution:
    def test_max_resolvable_terms_keeps_the_argument_error_small(self) -> None:
        for base in (2.0, 3.0, 1.5):
            terms = fs.max_resolvable_terms(base)
            eps = float(np.finfo(np.float64).eps)
            assert base**terms * eps <= 1e-4
            assert base ** (terms + 1) * eps > 1e-4

    def test_rejects_a_base_at_or_below_one(self) -> None:
        with pytest.raises(ValueError, match="base"):
            fs.max_resolvable_terms(1.0)

    def test_weierstrass_refuses_an_unresolvable_truncation(self) -> None:
        limit = fs.max_resolvable_terms(2.0)
        with pytest.raises(ValueError, match="resolvable"):
            fs.weierstrass(np.zeros(3), 0.5, base=2.0, terms=limit + 1)

    @pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.5])
    def test_weierstrass_rejects_an_exponent_outside_the_unit_interval(
        self, alpha: float
    ) -> None:
        with pytest.raises(ValueError, match="alpha"):
            fs.weierstrass(np.zeros(3), alpha)

    def test_partition_resolvable_terms_grows_linearly_with_level(self) -> None:
        counts = [fs.partition_resolvable_terms(n, 2.0) for n in range(1, 12)]
        steps = np.diff(counts)
        assert set(steps) <= {1, 2}
        for level, count in zip(range(1, 12), counts):
            assert count == int(level * math.log(3.0) / math.log(2.0))

    def test_a_larger_base_resolves_fewer_terms(self) -> None:
        assert fs.partition_resolvable_terms(8, 3.0) < fs.partition_resolvable_terms(
            8, 2.0
        )

    def test_resolution_floor_matches_its_closed_form(self) -> None:
        floor = fs.resolution_floor(10, tail=1e-3)
        assert floor == pytest.approx(math.log(1e3) / (10.0 * math.log(3.0)))
        assert floor == pytest.approx(0.6288, rel=1e-3)

    def test_resolution_floor_falls_like_one_over_the_level_count(self) -> None:
        assert fs.resolution_floor(20) == pytest.approx(
            0.5 * fs.resolution_floor(10), rel=1e-12
        )

    def test_the_floor_never_reaches_the_young_threshold(self) -> None:
        """The negative result: a fixed form cannot probe below Young's threshold.

        The ratio of the two is ``log(1/tail) / (L (log 4 - log(1/r)))``, which
        falls only like ``1/L``, so no feasible level count -- and the level
        count is capped by the ``4^L`` points of the curve -- brings the window
        down to the threshold.
        """
        for level in (8, 10, 12, 14):
            window = fs.resolution_window(level)
            assert not window.reaches_threshold
            assert window.alpha_floor > window.young_threshold

    def test_the_base_cancels_out_of_the_floor(self) -> None:
        """Raising the base buys nothing: it caps the truncation as fast as it
        deepens each term."""
        windows = [fs.resolution_window(10, base=b) for b in (1.5, 2.0, 4.0, 8.0)]
        floors = {round(window.alpha_floor, 12) for window in windows}
        assert len(floors) == 1

    def test_window_admits_exactly_what_it_should(self) -> None:
        window = fs.resolution_window(10)
        assert not window.admits(0.1)
        assert window.admits(0.8)

    @pytest.mark.parametrize("bad", [0, -3])
    def test_resolution_floor_rejects_a_non_positive_level(self, bad: int) -> None:
        with pytest.raises(ValueError, match="max_level"):
            fs.resolution_floor(bad)

    @pytest.mark.parametrize("bad", [0.0, 1.0, 2.0])
    def test_resolution_floor_rejects_an_impossible_tail(self, bad: float) -> None:
        with pytest.raises(ValueError, match="tail"):
            fs.resolution_floor(10, tail=bad)


class TestMatchedForm:
    @pytest.mark.parametrize("level", range(2, 9))
    def test_matched_form_uses_exactly_the_resolvable_truncation(
        self, level: int
    ) -> None:
        form = fs.matched_weierstrass_form(0.5, level)
        expected = min(
            fs.partition_resolvable_terms(level, 2.0), fs.max_resolvable_terms(2.0)
        )
        assert f"terms={expected}" in form.name

    def test_matched_form_deepens_with_the_level(self) -> None:
        coarse = fs.matched_weierstrass_form(0.5, 3)
        fine = fs.matched_weierstrass_form(0.5, 8)
        sample = np.linspace(0.0, 1.0, 64)
        assert not np.allclose(
            coarse.f(sample, sample), fine.f(sample, sample), atol=1e-6
        )


# --------------------------------------------------------------------------- #
# Rate estimation
# --------------------------------------------------------------------------- #
class TestDecayRate:
    def test_recovers_a_planted_geometric_rate(self) -> None:
        rate = 0.37
        values = np.cumsum([0.0] + [rate**n for n in range(12)])
        measured = fs.decay_rate(values)
        assert measured.rate == pytest.approx(rate, rel=1e-9)
        assert measured.tail_rate == pytest.approx(rate, rel=1e-9)
        assert measured.geometric
        assert measured.converges

    def test_survives_oscillating_signs(self) -> None:
        """Consecutive ratios cannot do this; regression on the whole sequence can."""
        rate = 0.5
        increments = [(-1.0) ** n * rate**n for n in range(14)]
        values = np.cumsum([0.0] + increments)
        assert fs.decay_rate(values).rate == pytest.approx(rate, rel=1e-9)

    def test_a_plateau_is_not_convergence(self) -> None:
        """The failure that inverted an earlier conclusion.

        A sequence that decays for a while and then flattens still yields a
        negative log-linear slope, so the whole-sequence rate comes out below $1$
        and reads as convergence.  Only the tail distinguishes the two.
        """
        increments = [0.5**n for n in range(6)] + [0.5**5] * 6
        values = np.cumsum([0.0] + increments)
        measured = fs.decay_rate(values)
        assert measured.rate < 1.0
        assert measured.tail_rate == pytest.approx(1.0, abs=1e-9)
        assert not measured.converges
        assert not measured.geometric

    def test_rejects_a_sequence_that_is_too_short(self) -> None:
        with pytest.raises(ValueError, match="noise floor"):
            fs.decay_rate([0.0, 1.0, 1.5])


class TestIncrementDecay:
    """The model-free instrument, which is the one the conclusions rest on."""

    def test_the_smooth_control_sits_on_four_r_squared(self) -> None:
        for angle in (0.8, fs.KOCH_ANGLE, 1.4):
            measured = fs.measure_increments(
                1.0, angle=angle, levels=tuple(range(3, 9)), n_phases=2, lipschitz=True
            )
            floor = 4.0 * fs.koch_ratio(angle) ** 2
            for ratio in measured.ratios:
                assert ratio == pytest.approx(floor, rel=1e-3)
            assert measured.converges
            assert not measured.plateaus

    def test_a_smooth_holder_form_reaches_the_same_floor(self) -> None:
        """Above the crossover the rate stops depending on the exponent entirely."""
        measured = fs.measure_increments(
            0.95, levels=tuple(range(3, 10)), n_phases=8
        )
        assert measured.tail_ratio == pytest.approx(4.0 / 9.0, rel=0.02)
        assert measured.converges

    def test_the_rate_rises_monotonically_as_the_form_roughens(self) -> None:
        tails = [
            fs.measure_increments(a, levels=tuple(range(3, 9)), n_phases=8).tail_ratio
            for a in (0.9, 0.6, 0.35, 0.15)
        ]
        assert tails == sorted(tails)

    def test_nothing_happens_at_youngs_threshold(self) -> None:
        """The classical threshold leaves no signature in the measured rate.

        At ``alpha_c`` the increments are still decaying at roughly two thirds
        per level, far from the flattening that a genuine edge of convergence
        produces, and the neighbouring exponents show no kink around it.
        """
        alpha_c = fs.koch_dimension() - 1.0
        measured = fs.measure_increments(
            alpha_c, levels=tuple(range(3, 10)), n_phases=12
        )
        assert measured.tail_ratio < 0.8
        assert measured.converges
        assert not measured.plateaus

    def test_the_bootstrap_interval_brackets_the_point_estimate(self) -> None:
        measured = fs.measure_increments(0.5, levels=tuple(range(3, 9)), n_phases=8)
        low, high = measured.tail_interval(resamples=200)
        assert low <= measured.tail_ratio <= high
        assert low > 0.0

    def test_a_flat_sequence_is_reported_as_a_plateau(self) -> None:
        flat = fs.IncrementDecay(
            alpha=0.05,
            dimension=fs.koch_dimension(),
            angle=fs.KOCH_ANGLE,
            levels=(3, 4, 5, 6),
            increments=(1e-3, 1e-3, 1e-3, 1e-3),
            per_phase=((1e-3,), (1e-3,), (1e-3,), (1e-3,)),
            n_phases=1,
            seed=0,
        )
        assert flat.tail_ratio == pytest.approx(1.0)
        assert not flat.converges
        assert flat.plateaus

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.2])
    def test_rejects_an_exponent_outside_the_unit_interval(self, bad: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            fs.measure_increments(bad)

    def test_rejects_too_few_levels(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            fs.measure_increments(0.5, levels=(3, 4))


class TestPredictedLaw:
    def test_young_rate_is_one_exactly_at_the_threshold(self) -> None:
        for angle in ANGLES:
            threshold = fs.koch_dimension(angle) - 1.0
            assert fs.young_rate(threshold, angle) == pytest.approx(1.0, rel=1e-12)

    def test_coherence_exponent_inverts_the_rate(self) -> None:
        for alpha in (0.2, 0.5, 0.9):
            rate = fs.young_rate(alpha)
            assert fs.coherence_exponent(rate, alpha) == pytest.approx(1.0, abs=1e-12)

    def test_the_geometric_branch_equals_the_smooth_rate(self) -> None:
        for angle in ANGLES:
            assert fs.predicted_rate(1.0, angle) == pytest.approx(
                fs.young_rate(1.0, angle), rel=1e-12
            )

    def test_the_branches_meet_at_the_predicted_crossover(self) -> None:
        for angle in ANGLES:
            alpha = fs.predicted_crossover(angle)
            ratio = fs.koch_ratio(angle)
            assert 2.0 * ratio ** (1.0 + alpha) == pytest.approx(
                4.0 * ratio**2, rel=1e-12
            )

    def test_crossover_is_one_minus_half_the_dimension(self) -> None:
        for angle in ANGLES:
            assert fs.predicted_crossover(angle) == pytest.approx(
                1.0 - fs.koch_dimension(angle) / 2.0, rel=1e-12
            )

    def test_predicted_rate_never_exceeds_young(self) -> None:
        for angle in ANGLES:
            for alpha in np.linspace(0.02, 1.0, 25):
                assert fs.predicted_rate(alpha, angle) <= fs.young_rate(alpha, angle)

    def test_the_random_walk_threshold_is_the_refuted_prediction(self) -> None:
        """Kept so the discredited hypothesis stays checkable, not just described.

        If the per-segment contributions added with independent signs the
        threshold would be ``d/2 - 1``, negative at every dimension in this
        family, meaning convergence for every positive exponent.  Direct
        measurement puts the edge near ``alpha = 0.01`` instead: below Young's
        ``d - 1`` by a wide margin, but not absent.
        """
        for angle in ANGLES + [1.5, 1.55]:
            assert fs.effective_threshold(angle) < 0.0
            assert fs.predicted_rate(1e-6, angle) < 1.0
            assert fs.effective_threshold(angle) < fs.koch_dimension(angle) - 1.0

    def test_the_predicted_crossover_sits_below_the_observed_one(self) -> None:
        """``1 - d/2 = 0.369`` for the standard curve; the measured break is near ``0.52``.

        The crossover formula inherits the random-walk assumption for the lower
        branch, so it is displaced by the same amount that assumption is wrong.
        The geometric branch it crosses *into* is exact; only the location of the
        meeting point is off.
        """
        assert fs.predicted_crossover() == pytest.approx(0.369, abs=1e-3)
        assert fs.predicted_crossover() < 0.52


# --------------------------------------------------------------------------- #
# The exact per-level decomposition
# --------------------------------------------------------------------------- #
class TestCoherenceProfile:
    @pytest.mark.parametrize("level", [3, 5, 7])
    def test_the_profile_total_is_the_refinement_increment(self, level: int) -> None:
        """The identity the whole decomposition rests on.

        If the per-segment terms do not sum to ``I_fine - I_coarse``, the
        grouping of children to parents is wrong and every coherence number
        derived from it is meaningless.
        """
        coarse = fs.koch_curve(level)
        fine = fs.koch_curve(level + 1)
        form = fs.weierstrass_form(0.7, phase=1.3, terms=12)
        profile = fs.coherence_profile(coarse, fine, form)
        expected = fs.riemann_stieltjes(fine, form) - fs.riemann_stieltjes(coarse, form)
        assert profile.total == pytest.approx(expected, abs=1e-12)

    def test_counts_and_norms_are_consistent(self) -> None:
        coarse, fine = fs.koch_curve(4), fs.koch_curve(5)
        profile = fs.coherence_profile(coarse, fine, fs.lipschitz_form(0.4))
        assert profile.count == 4**4
        assert profile.level == 4
        assert abs(profile.total) <= profile.l1 + 1e-15
        assert profile.l2 <= profile.l1 + 1e-15
        assert profile.l1 <= math.sqrt(profile.count) * profile.l2 + 1e-15

    def test_rejects_a_curve_that_is_not_a_refinement(self) -> None:
        with pytest.raises(ValueError, match="refinement"):
            fs.coherence_profile(
                fs.koch_curve(3), fs.koch_curve(5), fs.lipschitz_form()
            )

    def test_rejects_a_shifted_fine_curve(self) -> None:
        fine = fs.koch_curve(4) + 0.5
        with pytest.raises(ValueError, match="not a refinement"):
            fs.coherence_profile(fs.koch_curve(3), fine, fs.lipschitz_form())

    def test_a_smooth_form_adds_perfectly_coherently(self) -> None:
        """Every one of the per-segment contributions carries the same sign.

        For a smooth form the increment on a segment is a definite-sign
        second-order term, so ``|total| = l1`` exactly -- the strongest possible
        statement of coherence, and one no fitted rate could ever produce.
        """
        for level in (3, 5, 7):
            profile = fs.coherence_profile(
                fs.koch_curve(level), fs.koch_curve(level + 1), fs.lipschitz_form(0.1)
            )
            assert abs(profile.total) == pytest.approx(profile.l1, rel=1e-12)
            assert profile.beta == pytest.approx(1.0, abs=1e-12)

    def test_randomness_doubles_per_level_for_a_smooth_form(self) -> None:
        values = [
            fs.coherence_profile(
                fs.koch_curve(n), fs.koch_curve(n + 1), fs.lipschitz_form(0.1)
            ).randomness
            for n in range(3, 8)
        ]
        ratios = [b / a for a, b in zip(values, values[1:])]
        assert ratios == pytest.approx([2.0] * len(ratios), rel=1e-3)

    def test_beta_is_undefined_when_there_is_no_signal(self) -> None:
        zero = fs.HolderForm(
            "zero", 1.0, lambda x, y: np.zeros_like(x), lambda x, y: np.zeros_like(x)
        )
        profile = fs.coherence_profile(fs.koch_curve(3), fs.koch_curve(4), zero)
        assert profile.randomness == 0.0
        assert profile.peakiness == 0.0
        with pytest.raises(ValueError, match="no usable signal"):
            _ = profile.beta


# --------------------------------------------------------------------------- #
# The measurement, against the case where the answer is known
# --------------------------------------------------------------------------- #
class TestSmoothControl:
    @pytest.mark.parametrize("angle", ANGLES)
    def test_the_smooth_control_is_exactly_coherent(self, angle: float) -> None:
        """``beta = 1`` and Young's rate ``4 r^2``, at every dimension.

        This is the calibration on which every Holder measurement depends: an
        estimator that cannot return the known answer here has no standing
        elsewhere.
        """
        measured = fs.measure_matched(
            1.0, angle=angle, levels=tuple(range(3, 8)), lipschitz=True, n_phases=2
        )
        assert measured.growth == pytest.approx(2.0, rel=1e-3)
        assert measured.beta == pytest.approx(1.0, abs=1e-3)
        assert measured.sampled_alpha == pytest.approx(1.0, abs=1e-2)
        assert measured.measured_rate == pytest.approx(
            fs.young_rate(1.0, angle), rel=1e-3
        )

    def test_the_fitted_estimator_also_recovers_the_control(self) -> None:
        measured = fs.measure_coherence(
            1.0, levels=tuple(range(2, 9)), lipschitz=True, n_phases=3
        )
        assert measured.beta == pytest.approx(1.0, abs=0.02)
        assert measured.converges


class TestMatchedMeasurement:
    @pytest.mark.parametrize("alpha", [0.15, 0.4, 0.75])
    def test_the_sampled_exponent_matches_the_requested_one(self, alpha: float) -> None:
        """The aliasing referee.

        The matched form is cut exactly at the partition's resolution, so the
        per-segment contributions must shrink like ``r^{n(1+alpha)}``.  If they
        do not, the form is not being sampled as an ``alpha``-Holder function and
        the coherence number means nothing.
        """
        measured = fs.measure_matched(alpha, levels=tuple(range(3, 9)), n_phases=3)
        assert measured.sampled_alpha == pytest.approx(alpha, abs=0.1)
        assert measured.faithful

    def test_coherence_is_bounded_by_the_two_extremes(self) -> None:
        for alpha in (0.1, 0.5, 0.9):
            measured = fs.measure_matched(
                alpha, levels=tuple(range(3, 9)), n_phases=3
            )
            assert 0.35 <= measured.beta <= 1.05

    def test_coherence_increases_with_smoothness(self) -> None:
        rough = fs.measure_matched(0.2, levels=tuple(range(3, 9)), n_phases=3)
        smooth = fs.measure_matched(0.9, levels=tuple(range(3, 9)), n_phases=3)
        assert rough.beta < smooth.beta

    def test_the_measured_rate_stays_below_young(self) -> None:
        """The central claim, stated as the inequality it actually is.

        Failing Young's condition does not prove divergence -- Young's theorem
        merely goes silent -- so the claim is that the sums converge inside the
        region the classical sufficient condition leaves open, because of sign
        cancellation an absolute-value estimate discards.
        """
        for alpha in (0.05, 0.2, 0.5):
            measured = fs.measure_matched(
                alpha, levels=tuple(range(3, 9)), n_phases=3
            )
            assert measured.measured_rate < fs.young_rate(alpha)

    def test_convergence_is_clear_well_below_youngs_threshold(self) -> None:
        """Refereed by the model-free instrument, not by the coherence fit.

        The exponent route reports a rate below $1$ even for a plateau, so the
        claim is checked against the tail of the increments instead.  At
        ``alpha = 0.15``, little more than half of Young's ``0.262``, the
        increments are still shrinking by a fifth per level with the whole
        bootstrap interval clear of $1$.
        """
        alpha = 0.15
        assert alpha < fs.koch_dimension() - 1.0
        direct = fs.measure_increments(alpha, levels=tuple(range(3, 10)), n_phases=12)
        assert direct.tail_interval(resamples=200)[1] < 1.0
        assert direct.converges

    def test_the_smallest_exponents_are_not_resolved(self) -> None:
        """Honest negative: at ``alpha = 0.05`` the answer depends on the sampling.

        Seven levels at twelve phases give a tail ratio above $1$; eight levels
        at thirty-two give ``0.93``.  The accessible level range does not settle
        whether the sums converge there, and reporting either number alone would
        overstate what was measured.  This is the reason the empirical threshold
        is quoted as "below ``0.05``, consistent with zero" rather than as the
        ``0.011`` that extrapolating the fit produces.
        """
        direct = fs.measure_increments(0.05, levels=tuple(range(3, 10)), n_phases=12)
        low, high = direct.tail_interval(resamples=200)
        assert low < 1.05 and high > 0.9, "expected a tail ratio consistent with 1"

    @pytest.mark.parametrize("bad", [0.0, -0.2, 1.4])
    def test_rejects_an_exponent_outside_the_unit_interval(self, bad: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            fs.measure_matched(bad)

    def test_rejects_too_few_levels_to_fit_a_growth(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            fs.measure_matched(0.5, levels=(3, 4))


class TestPhaseDiagram:
    def test_the_grid_is_keyed_by_angle_and_alpha(self) -> None:
        angles = (0.8, fs.KOCH_ANGLE)
        alphas = (0.2, 0.8)
        grid = fs.phase_diagram(
            angles, alphas, levels=tuple(range(3, 8)), n_phases=2
        )
        assert set(grid) == {(a, b) for a in angles for b in alphas}
        for (angle, alpha), measured in grid.items():
            assert measured.alpha == alpha
            assert measured.dimension == pytest.approx(fs.koch_dimension(angle))

    def test_the_geometric_branch_is_flat_in_alpha(self) -> None:
        """Above the crossover the rate stops depending on the Holder exponent.

        It is then set by the curve's own second-order geometry rather than by
        the form, which is what makes ``4 r^2`` the right value there.  Measured
        model-free, and taken above the *observed* crossover near ``0.55``
        rather than the ``1 - d/2`` one, which inherits the refuted random-walk
        assumption for the branch below.
        """
        angle = fs.KOCH_ANGLE
        floor = 4.0 * fs.koch_ratio(angle) ** 2
        tails = [
            fs.measure_increments(
                alpha, angle=angle, levels=tuple(range(3, 10)), n_phases=8
            ).tail_ratio
            for alpha in (0.65, 0.8, 0.95)
        ]
        assert max(tails) / min(tails) < 1.08
        for tail in tails:
            assert tail == pytest.approx(floor, rel=0.06)


# --------------------------------------------------------------------------- #
# The trap that invalidated the first sweep
# --------------------------------------------------------------------------- #
class TestTruncationStability:
    def test_a_large_exponent_is_stable_under_truncation(self) -> None:
        rates = fs.truncation_stability(
            0.75, (12, 20, 28), levels=tuple(range(2, 9)), n_phases=3
        )
        assert len(rates) == 3
        assert max(rates.values()) / min(rates.values()) < 1.25

    def test_a_small_exponent_is_not(self) -> None:
        """The measurement that invalidated the first sweep.

        At ``alpha = 0.1`` the fitted rate moves by more than half as the
        truncation deepens, because the series has no converged tail at any
        truncation the partition can resolve.  The number being reported is a
        property of where the sum was cut off.
        """
        rates = fs.truncation_stability(
            0.1, (12, 20, 28, 36), levels=tuple(range(2, 9)), n_phases=3
        )
        assert max(rates.values()) / min(rates.values()) > 1.4

    def test_the_fixed_form_measurement_flags_its_own_window(self) -> None:
        levels = tuple(range(2, 11))
        assert not fs.measure_coherence(0.1, levels=levels).in_window
        assert fs.measure_coherence(0.8, levels=levels).in_window
