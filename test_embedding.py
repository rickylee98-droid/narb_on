"""Referees for the Obukhov amplifier gate in the Euler nonlinearity.

Four checks the module does not control:

* Energy and helicity are conserved by every closed triad and every helicity
  assignment, exactly and symbolically. Nothing else here fixes the coefficient
  convention, so these two identities are what pin it.
* Collinear triads must be inert: a divergence-free mode is orthogonal to its own
  wavevector, so if all three wavevectors lie on a line the nonlinearity vanishes.
* The transport cancellation -- the step the whole coefficient worry turns on --
  is checked as a residual that must be identically zero, not asserted.
* The asymptotic gate law is compared against the finite-wavenumber coefficient,
  which converges to it, and the maximiser is located by search rather than read
  off the closed form.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

import embedding as em

TRIADS = [
    ((1, 0, 0), (0, 1, 0), (-1, -1, 0)),
    ((2, 1, 0), (-1, 3, 1), (-1, -4, -1)),
    ((1, 1, 1), (2, -1, 0), (-3, 0, -1)),
    ((3, 0, 1), (0, 2, -2), (-3, -2, 1)),
    ((1, 0, 0), (5, 2, 1), (-6, -2, -1)),
]
HELICITIES = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1), (-1, -1, -1)]


class TestConservation:
    @pytest.mark.parametrize("vectors", TRIADS)
    @pytest.mark.parametrize("signs", HELICITIES)
    def test_energy_and_helicity_are_exactly_conserved(self, vectors, signs) -> None:
        energy, helicity = em.conservation_residuals(em.Triad(*vectors, helicities=signs))
        assert energy == 0
        assert helicity == 0

    def test_an_open_triad_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must close"):
            em.Triad((1, 0, 0), (0, 1, 0), (0, 0, 1))

    def test_a_zero_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no zero mode"):
            em.Triad((0, 0, 0), (1, 0, 0), (-1, 0, 0))

    def test_a_bad_helicity_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"helicities are"):
            em.Triad((1, 0, 0), (0, 1, 0), (-1, -1, 0), helicities=(1, 0, 1))

    def test_the_helical_basis_needs_a_nonzero_wavevector(self) -> None:
        with pytest.raises(ValueError, match="zero wavevector"):
            em.helical_vector((0, 0, 0))


class TestCollinearTriadsAreInert:
    @pytest.mark.parametrize(
        "vectors",
        [
            ((2, 0, 0), (1, 0, 0), (-3, 0, 0)),
            ((0, 3, 0), (0, -1, 0), (0, -2, 0)),
            ((1, 1, 1), (2, 2, 2), (-3, -3, -3)),
        ],
    )
    def test_the_geometric_factor_vanishes(self, vectors) -> None:
        triad = em.Triad(*vectors)
        assert em.is_collinear(triad)
        assert em.geometric_factor(triad) == 0
        assert all(coefficient == 0 for coefficient in em.triad_coefficients(triad))

    @pytest.mark.parametrize("vectors", TRIADS)
    def test_non_collinear_triads_do_interact(self, vectors) -> None:
        """The control: otherwise the vanishing above would say nothing."""
        triad = em.Triad(*vectors)
        assert not em.is_collinear(triad)
        assert em.geometric_factor(triad) != 0


class TestTransportCancellation:
    """The step the coefficient worry turns on."""

    @pytest.mark.parametrize("vectors", TRIADS)
    @pytest.mark.parametrize("signs", HELICITIES)
    def test_the_high_wavenumbers_enter_only_through_their_difference(
        self, vectors, signs
    ) -> None:
        triad = em.Triad(*vectors, helicities=signs)
        assert em.transport_cancellation_residual(triad) == 0

    @pytest.mark.parametrize("vectors", TRIADS)
    def test_the_high_pair_is_governed_by_the_first_coefficient(self, vectors) -> None:
        """``c2 + c3 = -c1``, so one coefficient controls the pair's energy."""
        triad = em.Triad(*vectors)
        c1, c2, c3 = em.triad_coefficients(triad)
        assert sp.simplify(c2 + c3 + c1) == 0
        assert sp.simplify(em.high_pair_amplification(triad) - c1) == 0

    def test_the_bound_by_the_low_wavenumber(self) -> None:
        """``| |k2| - |k3| | <= |k1|`` since ``k2 + k3 = -k1``: the triangle
        inequality is what caps the gate at the low wavenumber."""
        for vectors in TRIADS:
            triad = em.Triad(*vectors)
            norms = [em.helical_vector(v)[1] for v in triad.vectors]
            assert sp.simplify(sp.Abs(norms[1] - norms[2]) - norms[0]) <= 0


class TestGateLaw:
    @pytest.mark.parametrize("degrees", [5, 15, 25, 35, 45, 55, 65, 75, 85])
    def test_the_finite_wavenumber_gate_converges_to_the_law(self, degrees) -> None:
        theta = math.radians(degrees)
        predicted = float(em.gate_efficiency_limit(theta))
        errors = [
            abs(em.gate_efficiency(theta, magnitude) - predicted)
            for magnitude in (1e3, 1e4, 1e5)
        ]
        assert errors[0] > errors[1] > errors[2], "must converge, not merely agree"
        assert errors[-1] < 1e-4

    def test_the_maximum_is_a_half_at_forty_five_degrees(self) -> None:
        """Located by search, not read off the closed form."""
        best_angle, best_value = None, -1.0
        for step in range(1, 900):
            theta = math.radians(step / 10)
            value = em.gate_efficiency(theta, 1e6)
            if value > best_value:
                best_angle, best_value = theta, value
        assert abs(math.degrees(best_angle) - em.OPTIMAL_ANGLE_DEGREES) < 0.2
        assert abs(best_value - float(em.OPTIMAL_EFFICIENCY)) < 1e-5

    def test_the_closed_form_agrees_at_the_maximum(self) -> None:
        assert sp.nsimplify(em.gate_efficiency_limit(sp.pi / 4)) == em.OPTIMAL_EFFICIENCY

    def test_the_two_factors_pull_against_each_other(self) -> None:
        """The wavenumber difference wants collinearity, the geometry forbids it.

        ``|cos theta|`` is maximal at zero and ``|sin theta|`` vanishes there, so
        neither factor alone locates the optimum; only the product does.
        """
        assert float(em.gate_efficiency_limit(0)) == 0
        assert abs(float(em.gate_efficiency_limit(math.pi / 2))) < 1e-15
        assert float(em.gate_efficiency_limit(math.pi / 4)) == pytest.approx(0.5)

    def test_the_gate_is_symmetric_about_the_optimum(self) -> None:
        for offset in (5, 15, 25, 35):
            low = em.gate_efficiency(math.radians(45 - offset), 1e5)
            high = em.gate_efficiency(math.radians(45 + offset), 1e5)
            assert abs(low - high) < 1e-3

    def test_the_efficiency_is_scale_free(self) -> None:
        """Doubling the low wavenumber doubles ``|c1|``, so the ratio is fixed."""
        theta = math.radians(45)
        assert em.gate_efficiency(theta, 1e5, low=1.0) == pytest.approx(
            em.gate_efficiency(theta, 2e5, low=2.0), rel=1e-6
        )

    def test_degenerate_magnitudes_are_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            em.gate_efficiency(0.5, 0.0)
        with pytest.raises(ValueError, match="must be positive"):
            em.gate_efficiency(0.5, 1.0, low=-1.0)


class TestWhatThisDoesNotClaim:
    def test_the_gate_is_one_triad_at_a_time(self) -> None:
        """The module examines triads in isolation; an embedding cannot.

        Recorded as a test so the scope stays attached to the code: the
        coefficient question is settled for a single gate, the coherence of
        infinitely many gates is not addressed anywhere in this module.
        """
        assert not hasattr(em, "embed")
        assert not hasattr(em, "blowup")
