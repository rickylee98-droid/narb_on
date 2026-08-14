"""Referees for the Obukhov shell model's parameter geometry.

Four checks the module does not control:

* The cascade exponent must collapse to Kolmogorov's ``1/3`` at ``alpha = b = 1``.
  It was derived from the fixed-point equation with no reference to that value.
* The nonlinearity's contribution to the energy must telescope to exactly zero,
  in rational arithmetic, for arbitrary amplitudes and arbitrary shells. This is
  the one structural property the model must have whatever ``N_k`` is.
* Frequency truncation must be invariant -- Palasek's Remark 1.10, and the reason
  his blow-up is unstable in every ``C^s``.
* The viscous parameter window must be non-empty precisely for ``alpha > 2``.
  The paper obtains that threshold from energy criticality, by a route with
  nothing in common with counting an interval.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

import shell


class TestCascadeExponent:
    def test_it_is_kolmogorov_at_the_classical_parameters(self) -> None:
        """``alpha = b = 1`` gives ``1/3``: the external anchor."""
        assert shell.kolmogorov_exponent(1, 1) == Fraction(1, 3)

    def test_exponential_shells_give_alpha_over_three(self) -> None:
        for alpha in [Fraction(1), Fraction(3, 2), Fraction(5, 2), Fraction(4)]:
            assert shell.kolmogorov_exponent(alpha, 1) == alpha / 3

    @pytest.mark.parametrize(
        "alpha,b",
        [
            (Fraction(1), Fraction(1)),
            (Fraction(5, 2), Fraction(6, 5)),
            (Fraction(1), Fraction(3, 2)),
            (Fraction(3), Fraction(7, 5)),
            (Fraction(9, 4), Fraction(11, 10)),
        ],
    )
    @pytest.mark.parametrize("k", [1, 2, 3, 5, 8])
    def test_the_fixed_point_is_exact(self, alpha, b, k) -> None:
        assert shell.fixed_point_exponent_residual(alpha, b, k) == 0

    def test_a_wrong_exponent_does_not_satisfy_it(self) -> None:
        """Negative control: perturbing the exponent breaks the identity.

        Without this the residual could be vanishing for a trivial reason.
        """
        alpha, b, k = Fraction(5, 2), Fraction(6, 5), 3
        correct = shell.rescaled_cascade_exponent(alpha, b)
        wrong = correct + Fraction(1, 10)
        left = wrong * (shell.log_frequency(b, k - 1) + shell.log_frequency(b, k))
        right = 2 * alpha * (
            shell.log_frequency(b, k) - shell.log_frequency(b, k + 1)
        ) + 2 * wrong * shell.log_frequency(b, k + 1)
        assert left - right != 0

    def test_the_cascade_flattens_as_shells_separate(self) -> None:
        """``gamma`` strictly decreasing in ``b``: the mechanism, as one exponent.

        Wider separation flattens the state that regularises the model, which is
        why super-exponential shells admit blow-up where exponential ones do not.
        """
        alpha = Fraction(5, 2)
        values = [
            shell.kolmogorov_exponent(alpha, b)
            for b in [Fraction(1), Fraction(5, 4), Fraction(2), Fraction(10), Fraction(1000)]
        ]
        assert all(a > c for a, c in zip(values, values[1:]))
        assert values[-1] < Fraction(1, 500)

    def test_the_two_variables_agree(self) -> None:
        """``X_k = N_k^{-gamma}`` and ``x_k = N_k^{alpha - gamma}`` are one state."""
        for alpha, b in [(Fraction(1), Fraction(1)), (Fraction(5, 2), Fraction(6, 5))]:
            rescaled = shell.rescaled_cascade_exponent(alpha, b)
            assert rescaled == alpha - shell.kolmogorov_exponent(alpha, b)

    def test_a_negative_shell_index_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            shell.log_frequency(Fraction(6, 5), -1)

    def test_the_fixed_point_needs_a_left_neighbour(self) -> None:
        with pytest.raises(ValueError, match="k >= 1"):
            shell.fixed_point_exponent_residual(Fraction(2), Fraction(6, 5), 0)


class TestStructuralIdentities:
    @pytest.mark.parametrize(
        "amplitudes",
        [
            [Fraction(1), Fraction(2), Fraction(3)],
            [Fraction(1, 2), Fraction(-3, 5), Fraction(7), Fraction(11, 3)],
            [Fraction(0), Fraction(5), Fraction(0), Fraction(-2), Fraction(9, 7)],
        ],
    )
    def test_the_nonlinearity_conserves_energy_exactly(self, amplitudes) -> None:
        weights = [Fraction(k + 1) ** 3 + Fraction(1, k + 2) for k in range(len(amplitudes))]
        assert shell.energy_identity_residual(amplitudes, weights) == 0

    def test_it_holds_for_arbitrary_shells(self) -> None:
        """No property of ``N_k`` is used, which is why the model is well posed
        for exponential and super-exponential shells alike."""
        amplitudes = [Fraction(3, 2), Fraction(-1), Fraction(5, 4), Fraction(2)]
        for weights in (
            [Fraction(1)] * 4,
            [Fraction(2) ** k for k in range(4)],
            [Fraction(10) ** (2**k) for k in range(4)],
        ):
            assert shell.energy_identity_residual(amplitudes, weights) == 0

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="one frequency weight"):
            shell.energy_identity_residual([Fraction(1)], [Fraction(1), Fraction(2)])

    @pytest.mark.parametrize("cutoff", [0, 1, 2, 3])
    def test_frequency_truncation_is_invariant(self, cutoff) -> None:
        """Remark 1.10: truncated data stays truncated, so the blow-up is unstable."""
        amplitudes = [Fraction(2), Fraction(3, 2), Fraction(5), Fraction(1, 3), Fraction(7)]
        weights = [Fraction(4) ** k for k in range(5)]
        assert all(value == 0 for value in shell.truncation_residual(amplitudes, weights, cutoff))

    def test_untruncated_data_does_move(self) -> None:
        """The control: without the truncation the high shells are not at rest."""
        amplitudes = [Fraction(2), Fraction(3, 2), Fraction(5)]
        weights = [Fraction(4) ** k for k in range(3)]
        drift = weights[1] * amplitudes[1] * amplitudes[2]
        assert drift != 0

    def test_a_cutoff_outside_the_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match="index a shell"):
            shell.truncation_residual([Fraction(1)] * 3, [Fraction(1)] * 3, 5)


class TestParameterGeometry:
    @pytest.mark.parametrize(
        "alpha,expected",
        [
            (Fraction(1), False),
            (Fraction(2), False),
            (Fraction(9, 4), True),
            (Fraction(5, 2), True),
            (Fraction(4), True),
        ],
    )
    def test_the_viscous_window_is_non_empty_exactly_above_two(self, alpha, expected) -> None:
        """The threshold the paper derives from energy criticality, recovered by
        counting an interval instead."""
        assert (shell.viscous_window(alpha) is not None) is expected

    def test_the_threshold_is_exactly_two(self) -> None:
        assert shell.viscous_window(Fraction(2)) is None
        assert shell.viscous_window(Fraction(2) + Fraction(1, 10**6)) is not None

    def test_the_three_dimensional_window_is_the_overlap(self) -> None:
        low, high = shell.THREE_D_BLOWUP_WINDOW
        assert low == 2 and high == shell.INTERMITTENCY_RANGE[1]
        assert shell.viscous_window(high) is not None
        assert shell.viscous_window(low) is None

    def test_the_separation_budget_is_thin_in_three_dimensions(self) -> None:
        """``b < 5/4`` across the whole candidate window: barely super-exponential."""
        assert shell.separation_budget(Fraction(5, 2)) == Fraction(5, 4)
        for alpha in [Fraction(21, 10), Fraction(9, 4), Fraction(5, 2)]:
            budget = shell.separation_budget(alpha)
            assert budget is not None and 1 < budget <= Fraction(5, 4)

    def test_the_budget_grows_without_bound_outside_three_dimensions(self) -> None:
        """The contrast: the constraint only bites because ``alpha <= 5/2``."""
        assert shell.separation_budget(Fraction(100)) == 50

    @pytest.mark.parametrize("alpha", [Fraction(21, 10), Fraction(9, 4), Fraction(5, 2)])
    def test_the_trapping_region_clears_the_cascade(self, alpha) -> None:
        """Automatic for every admissible ``b`` in the three-dimensional window."""
        window = shell.viscous_window(alpha)
        assert window is not None
        for b in [Fraction(1), (1 + window[1]) / 2, window[1]]:
            assert shell.clears_cascade(alpha, b)

    def test_clearing_the_cascade_can_fail_far_outside_the_window(self) -> None:
        """The condition ``2b + 1 >= alpha`` is not vacuous."""
        assert not shell.clears_cascade(Fraction(100), Fraction(2))
        assert shell.clears_cascade(Fraction(100), Fraction(50))

    @pytest.mark.parametrize("alpha", [Fraction(9, 4), Fraction(5, 2), Fraction(3)])
    def test_the_barrier_window_is_non_empty_on_the_admissible_set(self, alpha) -> None:
        window = shell.viscous_window(alpha)
        assert window is not None
        b = (1 + window[1]) / 2
        assert shell.beta_window(alpha, b, Fraction(1, 100), viscous=True) is not None

    def test_the_inviscid_barrier_window_is_wider(self) -> None:
        alpha, b, s = Fraction(1), Fraction(3), Fraction(1, 10)
        assert shell.beta_window(alpha, b, s, viscous=True) is None
        assert shell.beta_window(alpha, b, s, viscous=False) is not None

    def test_a_non_positive_blowup_index_is_refused(self) -> None:
        with pytest.raises(ValueError, match="s > 0"):
            shell.beta_window(Fraction(5, 2), Fraction(6, 5), Fraction(0))

    def test_the_model_refuses_unphysical_parameters(self) -> None:
        with pytest.raises(ValueError, match="alpha >= 1"):
            shell.ObukhovModel(Fraction(1, 2), Fraction(6, 5))
        with pytest.raises(ValueError, match="b >= 1"):
            shell.ObukhovModel(Fraction(5, 2), Fraction(1, 2))

    def test_the_exponential_model_is_the_b_equals_one_case(self) -> None:
        """The globally regular classical model sits on the boundary of the window."""
        model = shell.ObukhovModel(Fraction(5, 2), Fraction(1))
        window = shell.viscous_window(model.alpha)
        assert window is not None and model.b == window[0]
