"""Referees for the averaging estimate.

Three checks the module does not control:

* Scale invariance is a symmetry of Euler, not a fitted fact, so placing one gate
  geometry at different absolute scales must return the *same* penalty. It does,
  to solver precision. This is what carries uniformity in the shell index, since
  the architecture makes every gate a rescaled copy of the first.
* The separation dependence must approach its limit from above, so that the limit
  is a bound at every finite ratio rather than only an asymptote.
* The measured penalty at a given ratio must agree with the fitted model, which
  is a two-parameter fit to nine measurements and so could easily fail.
"""

from __future__ import annotations

import numpy as np
import pytest

import averaging as av


class TestScaleInvariance:
    def test_the_penalty_does_not_depend_on_absolute_scale(self) -> None:
        """Euler's scale invariance, checked rather than assumed.

        The architecture is self-similar, so gate ``j`` is a rescaled gate ``1``.
        If the penalty is scale free it is shell free, which is the uniformity
        the averaging estimate needs.
        """
        assert av.scale_invariance_residual() < 1e-8

    @pytest.mark.parametrize("scale", [1, 2, 4])
    def test_each_scale_returns_the_same_number(self, scale) -> None:
        _, reference = av.coherence_penalty(5, scale=1)
        _, moved = av.coherence_penalty(5, scale=scale)
        assert abs(moved - reference) < 1e-8

    def test_the_separation_ratio_is_scale_free_too(self) -> None:
        first, _ = av.coherence_penalty(5, scale=1)
        second, _ = av.coherence_penalty(5, scale=4)
        assert abs(first - second) < 1e-12

    def test_the_time_window_must_be_held_fixed(self) -> None:
        """The mistake that made the penalty look scale dependent.

        Amplitudes are normalised by ``1/|k|``, so the rate ``|k||u|`` is scale
        free and the window must not be rescaled. Shrinking it in proportion to
        the scale measures a different part of the trajectory and produces a
        spurious drift, which was this module's first result.
        """
        _, fixed = av.coherence_penalty(5, scale=4, duration=80.0)
        _, shrunk = av.coherence_penalty(5, scale=4, duration=20.0)
        assert abs(fixed - shrunk) > 1e-6

    def test_a_bad_scale_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            av.coherence_penalty(5, scale=0)


class TestSeparationDependence:
    @pytest.mark.parametrize("growth", [3, 8, 20])
    def test_the_measurement_matches_the_model(self, growth) -> None:
        ratio, measured = av.coherence_penalty(growth)
        assert abs(measured - av.penalty_model(ratio)) < 5 * av.SEPARATION_FIT_RESIDUAL

    def test_the_fit_reproduces_the_recorded_constants(self) -> None:
        limit, correction, residual = av.fit_penalty()
        assert abs(limit - av.PENALTY_LIMIT) < 2e-3
        assert abs(correction - av.PENALTY_CORRECTION) < 5e-3
        assert residual <= av.SEPARATION_FIT_RESIDUAL * 1.5

    def test_the_penalty_decreases_toward_its_limit(self) -> None:
        """Approach from above, so the limit bounds every finite ratio."""
        penalties = [av.coherence_penalty(g)[1] for g in (3, 8, 20, 60)]
        assert penalties[0] > penalties[-1]
        assert all(p > av.PENALTY_LIMIT * 0.95 for p in penalties)

    @pytest.mark.parametrize("ratio", [2.0, 7.07, 50.0, 1e4])
    def test_the_limit_is_a_lower_bound_at_every_finite_ratio(self, ratio) -> None:
        assert av.penalty_is_bounded_below(ratio)
        assert av.penalty_model(ratio) > av.PENALTY_LIMIT

    def test_the_limit_is_positive(self) -> None:
        """The whole estimate turns on this one sign."""
        assert av.PENALTY_LIMIT > 0
        assert av.PENALTY_CORRECTION > 0

    def test_a_degenerate_ratio_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            av.penalty_model(0.0)


class TestTheEstimate:
    def test_the_slowdown_is_a_constant_of_order_ten(self) -> None:
        assert 5.0 < av.slowdown_factor() < 20.0

    def test_finite_coherent_time_gives_finite_embedded_time(self) -> None:
        """The estimate: a constant factor, not a divergence."""
        for coherent in (0.1, 1.0, 37.5):
            bound = av.blowup_time_bound(coherent)
            assert np.isfinite(bound)
            assert bound == pytest.approx(coherent / av.PENALTY_LIMIT)

    def test_the_bound_is_monotone(self) -> None:
        assert av.blowup_time_bound(1.0) < av.blowup_time_bound(2.0)

    def test_a_non_positive_coherent_time_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            av.blowup_time_bound(0.0)

    def test_the_sum_bound_is_what_closes_it(self) -> None:
        """``sum tau_j <= (1/rho_inf) sum tau_j^coherent`` term by term."""
        coherent_terms = [2.0**-j for j in range(30)]
        embedded_bound = sum(av.blowup_time_bound(t) for t in coherent_terms)
        assert embedded_bound == pytest.approx(
            sum(coherent_terms) / av.PENALTY_LIMIT
        )
        assert np.isfinite(embedded_bound)
