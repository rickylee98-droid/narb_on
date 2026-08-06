"""Tests for the two-dimensional conformal bootstrap module.

As elsewhere in this project, the machinery is pinned against results it has no
way to know about: the quadratic Casimir equation for the blocks, and the exact
2d Ising dimensions ``(1/8, 1)`` for the bound.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

import bootstrap as bs


@pytest.fixture(scope="module")
def basis() -> bs.DerivativeBasis:
    return bs.DerivativeBasis(max_order=9)


#: Smaller grids for the test suite.  Verified to give the same bound at the
#: Ising point as the defaults (1.0028 either way), at a fraction of the cost.
FAST = dict(max_spin=12, n_samples=40, audit_samples=150, audit_delta_max=40.0)


# --------------------------------------------------------------------------- #
# Blocks, refereed by the Casimir equation
# --------------------------------------------------------------------------- #
class TestConformalBlocks:
    @pytest.mark.parametrize(
        "delta, spin", [(1, 0), (2, 0), (0.25, 0), (2, 2), (3, 2), (4.5, 4), (7, 6)]
    )
    @pytest.mark.parametrize("z, zbar", [(0.3, 0.4), (0.6, 0.25)])
    def test_blocks_satisfy_the_casimir_equation(
        self, delta: float, spin: int, z: float, zbar: float
    ) -> None:
        """The definitive check: a block *is* a Casimir eigenfunction.

        This tests the closed form against the representation theory it is meant
        to encode, not against another implementation of the same formula.
        """
        previous = mp.mp.dps
        mp.mp.dps = 30
        try:
            residual = bs.casimir_residual(delta, spin, mp.mpf(z), mp.mpf(zbar))
            assert abs(residual) < mp.mpf("1e-20")
        finally:
            mp.mp.dps = previous

    def test_casimir_rejects_unimplemented_dimensions(self) -> None:
        with pytest.raises(ValueError, match="only d = 2"):
            bs.casimir_residual(1.0, 0, 0.3, 0.4, dimension=3)

    @pytest.mark.parametrize("beta", [0.25, 1.0, 2.0, 4.5, 8.0, 17.0])
    def test_series_matches_direct_differentiation(self, beta: float) -> None:
        """The termwise series against mpmath's own derivative of the closed form."""
        previous = mp.mp.dps
        mp.mp.dps = 40
        try:
            series = bs.sl2_block_derivatives(mp.mpf(beta), 4, terms=400)
            for m in range(5):
                reference = mp.diff(lambda x: bs._sl2(mp.mpf(beta), x), mp.mpf(1) / 2, m)
                assert abs(series[m] - reference) / abs(reference) < mp.mpf("1e-20")
        finally:
            mp.mp.dps = previous

    @pytest.mark.parametrize("beta", [0.25, 1.0, 4.5, 8.0, 30.0])
    def test_fast_and_exact_paths_agree(self, beta: float) -> None:
        """The float64 path carries the whole bootstrap, so it is checked, not assumed."""
        previous = mp.mp.dps
        mp.mp.dps = 40
        try:
            fast = bs.sl2_block_derivatives_fast(beta, 9)
            exact = bs.sl2_block_derivatives(mp.mpf(beta), 9, terms=400)
            for m in range(10):
                assert abs(fast[m] - float(exact[m])) / abs(float(exact[m])) < 1e-13
        finally:
            mp.mp.dps = previous

    def test_beta_zero_is_the_constant_block(self) -> None:
        """A conserved current sits at the unitarity bound, where beta = 0."""
        assert bs.sl2_block_derivatives(mp.mpf(0), 3) == [1, 0, 0, 0]
        np.testing.assert_array_equal(
            bs.sl2_block_derivatives_fast(0.0, 3), [1.0, 0.0, 0.0, 0.0]
        )

    def test_truncation_tail_is_negligible(self) -> None:
        previous = mp.mp.dps
        mp.mp.dps = 40
        try:
            for beta in (0.25, 8.0, 30.0):
                tail = bs._series_tail(mp.mpf(beta), 120, mp.mpf(1) / 2)
                assert tail < mp.mpf("1e-30")
        finally:
            mp.mp.dps = previous

    def test_rejects_nonsense_parameters(self) -> None:
        with pytest.raises(ValueError, match="order"):
            bs.sl2_block_derivatives(mp.mpf(1), -1)
        with pytest.raises(ValueError, match="terms"):
            bs.sl2_block_derivatives_fast(1.0, 2, terms=0)


# --------------------------------------------------------------------------- #
# The crossing function
# --------------------------------------------------------------------------- #
class TestCrossingFunction:
    def test_basis_uses_only_odd_total_order(self, basis) -> None:
        """Even derivatives vanish identically by antisymmetry, so they carry nothing."""
        assert all((m + n) % 2 == 1 for m, n in basis.components)
        assert all(m < n for m, n in basis.components)
        assert basis.size == 15

    def test_identity_contribution_is_nonzero(self, basis) -> None:
        vector = bs.identity_crossing_derivatives(mp.mpf(1) / 8, basis)
        assert vector.shape == (basis.size,)
        assert np.all(np.isfinite(vector))
        assert np.linalg.norm(vector) > 0

    @pytest.mark.parametrize("spin", [0, 2, 4])
    def test_crossing_vectors_are_finite(self, basis, spin: int) -> None:
        vector = bs.crossing_derivatives(spin + 1.5, spin, mp.mpf(1) / 8, basis)
        assert vector.shape == (basis.size,)
        assert np.all(np.isfinite(vector))

    def test_odd_spin_is_rejected(self, basis) -> None:
        """Only even spins appear in the OPE of two identical scalars."""
        with pytest.raises(ValueError, match="even spins"):
            bs.crossing_derivatives(3.0, 1, mp.mpf(1) / 8, basis)

    def test_negative_spin_is_rejected(self, basis) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            bs.crossing_derivatives(3.0, -2, mp.mpf(1) / 8, basis)

    def test_crossing_vector_matches_a_direct_two_variable_derivative(self, basis) -> None:
        """Independent check of the factorised formula, by brute-force differentiation.

        The module never differentiates in two variables -- it exploits the fact
        that everything factorises.  This recomputes one component the slow,
        obvious way and demands agreement.
        """
        previous = mp.mp.dps
        mp.mp.dps = 30
        try:
            delta, spin, delta_phi = mp.mpf(3), 2, mp.mpf(1) / 4

            def block(z, zbar):
                low, high = delta - spin, delta + spin
                return bs._sl2(low, z) * bs._sl2(high, zbar) + bs._sl2(high, z) * bs._sl2(
                    low, zbar
                )

            def crossing(z, zbar):
                u, v = z * zbar, (1 - z) * (1 - zbar)
                return v**delta_phi * block(z, zbar) - u**delta_phi * block(1 - z, 1 - zbar)

            half = mp.mpf(1) / 2
            computed = bs.crossing_derivatives(delta, spin, delta_phi, basis)
            for index, (m, n) in enumerate(basis.components[:4]):
                reference = mp.diff(crossing, (half, half), (m, n)) / (
                    mp.factorial(m) * mp.factorial(n)
                )
                assert abs(computed[index] - float(reference)) <= 1e-9 * max(
                    1.0, abs(float(reference))
                )
        finally:
            mp.mp.dps = previous


class TestUnitarity:
    @pytest.mark.parametrize(
        "spin, dimension, expected", [(0, 2, 0), (2, 2, 2), (4, 2, 4), (0, 3, 0.5), (2, 3, 3)]
    )
    def test_bounds(self, spin: int, dimension: int, expected: float) -> None:
        assert float(bs.unitarity_bound(spin, dimension)) == pytest.approx(expected)

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError, match="spin"):
            bs.unitarity_bound(-1)
        with pytest.raises(ValueError, match="dimension"):
            bs.unitarity_bound(0, dimension=1)

    def test_samples_start_at_the_assumed_gap_and_the_unitarity_bounds(self) -> None:
        samples = bs.spectrum_samples(mp.mpf(2), max_spin=4, n_samples=5)
        scalars = [float(d) for d, s in samples if s == 0]
        spin_two = [float(d) for d, s in samples if s == 2]
        assert min(scalars) == pytest.approx(2.0)
        assert min(spin_two) == pytest.approx(2.0)
        assert all(d >= 2.0 for d in scalars)

    def test_rejects_a_degenerate_sample_count(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            bs.spectrum_samples(mp.mpf(1), n_samples=1)


# --------------------------------------------------------------------------- #
# The functional, and the reason it must be audited
# --------------------------------------------------------------------------- #
class TestFunctional:
    def test_a_large_gap_is_excluded(self, basis) -> None:
        result = bs.find_functional(mp.mpf(1) / 8, 2.5, basis, **FAST)
        assert result.found
        assert result.excludes
        assert result.coefficients is not None

    def test_a_small_gap_is_not_excluded(self, basis) -> None:
        """Below the bound no functional exists -- which proves nothing on its own."""
        assert not bs.find_functional(mp.mpf(1) / 8, 0.2, basis, **FAST).found

    def test_reported_functionals_are_non_negative_off_the_training_grid(
        self, basis
    ) -> None:
        """The check that a single-shot linear program silently fails.

        Without the audit loop the solution dips to about -1e-3 between its own
        sample points, which would make the "exclusion" meaningless.  A reported
        exclusion must survive re-testing on a grid it never trained on.
        """
        for delta_phi, gap in ((0.125, 1.5), (0.25, 2.0), (0.0625, 1.1)):
            result = bs.find_functional(mp.mpf(delta_phi), gap, basis, **FAST)
            assert result.found
            assert result.margin is not None and result.margin > 0.0
            assert bs.functional_margin(result) > 0.0

    def test_margin_requires_a_functional(self, basis) -> None:
        result = bs.find_functional(mp.mpf(1) / 8, 0.2, basis, **FAST)
        with pytest.raises(ValueError, match="no functional"):
            bs.functional_margin(result)


class TestIsingCalibration:
    """The whole point: reproduce exactly known physics.

    The 2d Ising model has ``Delta_sigma = 1/8`` and ``Delta_epsilon = 1``
    exactly, and the single-correlator bound is known to pass essentially
    through that point.
    """

    def test_bound_at_the_ising_point_is_close_to_one(self, basis) -> None:
        bound = bs.scalar_gap_bound(
            bs.ISING_2D_SIGMA, basis, low=0.05, high=4.5, tolerance=2e-3, **FAST
        )
        assert bound == pytest.approx(float(bs.ISING_2D_EPSILON), abs=0.02)

    def test_the_ising_model_itself_is_not_excluded(self, basis) -> None:
        """A gap just below Ising's must survive; the real theory sits there."""
        assert not bs.find_functional(bs.ISING_2D_SIGMA, 0.98, basis, **FAST).found

    def test_a_gap_well_above_ising_is_excluded(self, basis) -> None:
        assert bs.find_functional(bs.ISING_2D_SIGMA, 1.30, basis, **FAST).found

    def test_the_bound_increases_with_delta_phi(self, basis) -> None:
        """Monotonic below the kink -- a basic sanity property of the curve."""
        low = bs.scalar_gap_bound(0.08, basis, low=0.05, high=4.5, tolerance=4e-3, **FAST)
        high = bs.scalar_gap_bound(0.115, basis, low=0.05, high=4.5, tolerance=4e-3, **FAST)
        assert low < high < 1.0

    def test_more_derivatives_never_weaken_the_bound(self) -> None:
        """A larger basis contains the smaller one, so the bound cannot get worse."""
        coarse = bs.scalar_gap_bound(
            0.125, bs.DerivativeBasis(max_order=5), low=0.05, high=4.5, tolerance=4e-3
        )
        fine = bs.scalar_gap_bound(
            0.125, bs.DerivativeBasis(max_order=9), low=0.05, high=4.5, tolerance=4e-3
        )
        assert fine <= coarse + 1e-2

    def test_rejects_an_inverted_bracket(self, basis) -> None:
        with pytest.raises(ValueError, match="low < high"):
            bs.scalar_gap_bound(0.125, basis, low=3.0, high=1.0)


class TestExclusionIsMonotone:
    """The bisection in :func:`scalar_gap_bound` needs a monotone predicate.

    Raising the assumed gap removes constraints, so exclusion can only get
    easier.  A numerical failure that breaks that -- and one did: the audit loop
    running out of rounds at one gap while succeeding on both sides of it --
    silently corrupts every bound the bisection returns.
    """

    def test_exclusion_never_turns_off_as_the_gap_grows(self, basis) -> None:
        gaps = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.1, 2.2, 4.5]
        found = [bs.find_functional(0.0625, gap, basis, **FAST).found for gap in gaps]
        first_true = next((i for i, f in enumerate(found) if f), len(found))
        assert all(found[i] for i in range(first_true, len(found))), (
            f"exclusion is not monotone in the gap: {list(zip(gaps, found))}"
        )

    def test_converged_results_report_a_positive_margin(self, basis) -> None:
        result = bs.find_functional(0.125, 2.0, basis, **FAST)
        assert result.lp_feasible
        assert result.converged
        assert result.margin is not None and result.margin > 0.0

    def test_the_margin_changes_sign_at_the_bound(self, basis) -> None:
        """The predicate is the sign of an optimum, not a tolerance comparison.

        Feasibility-with-a-tolerance was the original formulation and it broke:
        the audit loop stalled near -1e-8 at some gaps while succeeding on both
        sides, so the bisection returned nonsense.  Maximising the worst value
        gives a quantity that varies continuously with the gap.
        """
        below = bs.find_functional(0.0625, 0.30, basis, **FAST)
        above = bs.find_functional(0.0625, 0.40, basis, **FAST)
        assert below.margin < 0.0 < above.margin
        assert not below.found and above.found

    def test_unexcluded_gaps_yield_no_functional(self, basis) -> None:
        result = bs.find_functional(0.125, 0.2, basis, **FAST)
        assert not result.found
        assert result.coefficients is None
        assert result.margin is not None and result.margin < 0.0

    def test_the_bound_curve_is_increasing_across_the_ising_point(self, basis) -> None:
        bounds = [
            bs.scalar_gap_bound(d, basis, low=0.05, high=4.5, tolerance=4e-3, **FAST)
            for d in (0.0625, 0.10, 0.125, 0.15)
        ]
        assert bounds == sorted(bounds), bounds
