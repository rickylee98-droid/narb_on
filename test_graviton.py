"""Referees for the identification of the obstruction with the Friedmann constraint.

The module claims three things, and each is checked against something it does not
control:

* the frequency extracted from the constraint algebra is ``|k|``, checked against
  ``|k|^2`` computed from the wave vector alone, at both polarisations, with the
  factor relating momentum to velocity isolated so that removing it is seen to
  break the result;
* the energy is Isaacson's with its standard normalisation, checked as an exact
  integer ratio against a formula written out independently of :mod:`adm`;
* the balance is the Friedmann constraint with shear, checked against a raw sum
  of :func:`adm.obstruction_value` over every mode, which shares no code with
  :func:`graviton.balance`.

Two external referees appear as well. The dimensions of the invariant homogeneous
momenta are the dimensions of the moduli spaces of flat metrics on the
platycosms, classically ``6, 4, 2, 3``, and are reproduced by a computation that
never mentions moduli. And the pure-trace mode must come out with negative
kinetic coefficient -- the conformal factor problem -- rather than as another
oscillator, which is what distinguishes a genuine identification from a formula
that happens to fit the transverse-traceless sector.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

import adm
import graviton as g
import platycosm as pc

MODES = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (1, 1, 1),
    (2, 0, 0),
    (0, 0, 2),
    (2, 1, 0),
    (2, 2, 1),
    (2, -1, 3),
    (3, 1, -2),
    (4, -3, 1),
    (5, 2, -4),
]

IDENTITY = [Fraction(1), Fraction(0), Fraction(0), Fraction(1), Fraction(0), Fraction(1)]

#: Dimensions of the moduli spaces of flat metrics on the orientable platycosms
#: handled by :mod:`platycosm`; classical, and independent of anything here.
KNOWN_MODULI = {"G1": 6, "G2": 4, "G4": 2, "G6": 3}


def gauge_tensor(wave, vector):
    """``h_ij = k_i xi_j + k_j xi_i``, a linearised diffeomorphism."""
    tensor = [Fraction(0)] * 6
    for slot, (i, j) in enumerate(adm.SYMMETRIC_PAIRS):
        tensor[slot] = Fraction(wave[i] * vector[j] + wave[j] * vector[i])
    return tensor


class TestDispersion:
    """``omega^2 = |k|^2``, extracted rather than assumed."""

    @pytest.mark.parametrize("wave", MODES)
    def test_every_polarisation_is_massless(self, wave) -> None:
        checks = g.check_dispersion(wave)
        assert len(checks) == 2, "two transverse-traceless polarisations"
        for check in checks:
            assert check.massless
            assert check.omega_squared == sum(c * c for c in wave)

    @pytest.mark.parametrize("wave", MODES)
    def test_the_energy_is_positive_on_both_polarisations(self, wave) -> None:
        for check in g.check_dispersion(wave):
            assert check.energy_positive

    @pytest.mark.parametrize("wave", MODES)
    def test_the_frequency_does_not_depend_on_normalisation(self, wave) -> None:
        """The coefficients scale with the polarisation basis; the ratio must not.

        :func:`adm.transverse_traceless_modes` returns an unnormalised rational
        basis, so the individual kinetic and potential coefficients differ from
        one mode to the next. A frequency read off them is only meaningful if it
        is invariant under rescaling, which it is.
        """
        for tensor in adm.transverse_traceless_modes(wave):
            base = g.dispersion(wave, tensor)
            for factor in (Fraction(2), Fraction(-3), Fraction(1, 7)):
                scaled = [factor * entry for entry in tensor]
                assert g.dispersion(wave, scaled) == base

    @pytest.mark.parametrize("wave", MODES)
    def test_the_closed_form_of_both_coefficients(self, wave) -> None:
        """``kin = |A|^2 / 2`` and ``pot = |k|^2 |A|^2 / 8`` on the TT sector.

        These are the two halves of ``-Q_k`` written out by hand from the
        docstring of :func:`adm.metric_obstruction_term`, using transversality to
        kill the ``|A k|^2`` term and trace-freeness to kill the ``(tr A)^2``
        term. Their ratio is what forces ``omega^2 = |k|^2`` once ``hdot = 2 K``
        is applied, so agreeing here is the analytic content of the derivation.
        """
        square = sum(c * c for c in wave)
        for tensor in adm.transverse_traceless_modes(wave):
            norm = g._tensor_norm(tensor)
            assert g.kinetic_coefficient(wave, tensor) == norm / 2
            assert g.potential_coefficient(wave, tensor) == square * norm / 8

    @pytest.mark.parametrize("wave", MODES)
    def test_the_velocity_factor_is_load_bearing(self, wave) -> None:
        """Drop ``hdot = 2K`` and the frequency comes out wrong by a factor of two.

        A negative control on the one physical input the derivation takes from
        outside the constraint algebra. If the answer were insensitive to it, the
        agreement with ``|k|^2`` would be telling us nothing.
        """
        square = sum(c * c for c in wave)
        for tensor in adm.transverse_traceless_modes(wave):
            naive = g.potential_coefficient(wave, tensor) / g.kinetic_coefficient(
                wave, tensor
            )
            assert naive == Fraction(square, 4)
            assert naive != square

    def test_the_homogeneous_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="expansion, not gravitons"):
            g.check_dispersion((0, 0, 0))
        with pytest.raises(ValueError, match="expansion, not gravitons"):
            g.mode_energy((0, 0, 0), IDENTITY, IDENTITY)

    def test_a_malformed_wave_is_refused(self) -> None:
        with pytest.raises(ValueError, match="three components"):
            g.dispersion((1, 0), IDENTITY)

    def test_a_malformed_tensor_is_refused(self) -> None:
        with pytest.raises(ValueError, match="six slots"):
            g.kinetic_coefficient((1, 0, 0), [1, 2, 3])


class TestNonPropagatingSectors:
    """The oscillator is specific to the transverse-traceless sector."""

    @pytest.mark.parametrize("wave", MODES)
    def test_the_conformal_mode_has_the_wrong_sign_kinetic_term(self, wave) -> None:
        """``h = phi delta`` gives ``omega^2 = -(5/3)|k|^2``: not a graviton.

        The conformal factor enters the gravitational action with the opposite
        sign, which is why it is not a propagating degree of freedom. That it
        comes out here as an exact negative multiple of ``|k|^2``, universal
        across modes, is a control: a derivation that returned ``|k|^2`` for
        every tensor would be extracting nothing.
        """
        square = sum(c * c for c in wave)
        assert g.kinetic_coefficient(wave, IDENTITY) == -3
        assert g.potential_coefficient(wave, IDENTITY) == Fraction(5, 4) * square
        assert g.dispersion(wave, IDENTITY) == -Fraction(5, 3) * square

    @pytest.mark.parametrize("wave", MODES)
    def test_a_linearised_diffeomorphism_carries_no_potential_energy(
        self, wave
    ) -> None:
        """``h_ij = k_i xi_j + k_j xi_i`` gives exactly zero, for every ``xi``.

        Gauge directions lie in the kernel of the metric half of the form. This
        is second-order gauge invariance of ``<R^(2)>``, and it is not built into
        :func:`adm.metric_obstruction_term`, which was matched against four
        quadratic invariants without reference to gauge.
        """
        for vector in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3), (-2, 5, 1)]:
            assert g.potential_coefficient(wave, gauge_tensor(wave, vector)) == 0


class TestIsaacson:
    """The energy is Isaacson's, with its standard ``1/(32 pi)``."""

    @pytest.mark.parametrize("wave", MODES)
    def test_the_ratio_is_exactly_sixteen(self, wave) -> None:
        for tensor in adm.transverse_traceless_modes(wave):
            other = adm.transverse_traceless_modes(wave)[1]
            for amplitude, velocity in [
                (tensor, tensor),
                (tensor, [3 * e for e in tensor]),
                (tensor, [Fraction(0)] * 6),
                ([Fraction(0)] * 6, tensor),
                (tensor, other),
            ]:
                density = g.isaacson_density(wave, amplitude, velocity)
                energy = g.mode_energy(wave, amplitude, velocity)
                assert energy == g.ISAACSON_TO_OBSTRUCTION * density

    @pytest.mark.parametrize("wave", MODES)
    def test_the_density_is_positive_on_a_real_wave(self, wave) -> None:
        for tensor in adm.transverse_traceless_modes(wave):
            assert g.isaacson_density(wave, tensor, tensor) > 0

    def test_a_static_configuration_still_carries_gradient_energy(self) -> None:
        tensor = adm.transverse_traceless_modes((1, 0, 0))[0]
        assert g.isaacson_density((1, 0, 0), tensor, [Fraction(0)] * 6) > 0


class TestHomogeneousSector:
    """One positive direction, and it is the isotropic one."""

    def test_the_inertia_has_exactly_one_positive_direction(self) -> None:
        block = [row[6:] for row in adm.obstruction_form((0, 0, 0))[6:]]
        assert adm._congruence_signature(block) == (5, 0, 1)

    def test_the_metric_half_vanishes_identically(self) -> None:
        """A constant metric perturbation is a change of moduli, not of curvature."""
        for tensor in [IDENTITY, [Fraction(1), Fraction(2), Fraction(-1)] * 2]:
            assert adm.metric_obstruction_term((0, 0, 0), tensor) == 0

    def test_pure_expansion_is_positive_and_pure_shear_is_negative(self) -> None:
        expansion = g.split_homogeneous([-e for e in IDENTITY])
        assert expansion.hubble == 1
        assert expansion.shear_squared == 0
        assert expansion.value == 6
        shear = g.split_homogeneous(
            [Fraction(1), Fraction(0), Fraction(0), Fraction(-1), Fraction(0), Fraction(0)]
        )
        assert shear.hubble == 0
        assert shear.shear_squared == 2
        assert shear.value == -2

    @pytest.mark.parametrize(
        "momentum",
        [
            [1, 0, 0, 1, 0, 1],
            [1, 2, 3, 4, 5, 6],
            [-3, 1, 0, 2, -1, 7],
            [0, 0, 0, 0, 0, 0],
        ],
    )
    def test_the_split_reproduces_the_obstruction(self, momentum) -> None:
        """``6 H^2 - |sigma|^2`` against :func:`adm.obstruction_value` itself."""
        split = g.split_homogeneous(momentum)
        assert split.value == adm.obstruction_value({(0, 0, 0): momentum})

    def test_the_shear_is_trace_free(self) -> None:
        split = g.split_homogeneous([1, 2, 3, 4, 5, 6])
        rebuilt = list(split.momentum)
        for i in range(3):
            rebuilt[adm.symmetric_index(i, i)] += split.hubble
        assert sum(rebuilt[adm.symmetric_index(i, i)] for i in range(3)) == 0

    def test_the_homogeneous_weight_is_one_not_a_half(self) -> None:
        """The correction this module rests on.

        ``<cos^2(k.x)> = 1/2`` away from the origin, ``<1> = 1`` at it. Without
        the distinction the homogeneous contribution comes out half its true size
        and the Friedmann coefficients are wrong by a factor of two, while every
        per-mode sign and signature is unchanged -- which is why it survived until
        the two sectors were added together.
        """
        assert adm.obstruction_value({(0, 0, 0): IDENTITY}) == 6
        assert adm.obstruction_value({(1, 0, 0): IDENTITY}) == 3

    def test_per_mode_results_are_unaffected_by_the_weight(self) -> None:
        assert adm.physical_signature((0, 0, 0)) == (5, 6, 1)
        for wave in [(1, 0, 0), (1, 1, 1), (2, -1, 3)]:
            assert adm.physical_signature(wave) == (4, 4, 0)


class TestBalance:
    """The second-order condition, assembled and checked against a raw sum."""

    def build(self, hubble):
        waves = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        modes = {}
        for wave in waves:
            tensor = adm.transverse_traceless_modes(wave)[0]
            modes[wave] = (tensor, tensor)
        momentum = [
            -hubble, Fraction(0), Fraction(0), -hubble, Fraction(0), -hubble
        ]
        return momentum, modes

    def test_the_condition_closes_exactly(self) -> None:
        """Three unit waves and ``H = 1/2``: the obstruction is zero, not small."""
        momentum, modes = self.build(Fraction(1, 2))
        report = g.balance(momentum, modes)
        assert report.graviton_energy == Fraction(3, 2)
        assert report.homogeneous_value == Fraction(3, 2)
        assert report.obstruction == 0
        assert report.satisfied
        assert report.friedmann_residual == 0

    def test_it_agrees_with_a_raw_sum_over_modes(self) -> None:
        """An independent route: :func:`adm.obstruction_value` mode by mode.

        :func:`graviton.balance` splits the homogeneous momentum into expansion
        and shear and halves the velocities itself; this recomputes the same
        number without any of that, so agreement pins the bookkeeping.
        """
        for hubble in [Fraction(0), Fraction(1, 2), Fraction(3), Fraction(-2, 7)]:
            momentum, modes = self.build(hubble)
            total = adm.obstruction_value({(0, 0, 0): momentum})
            for wave, (amplitude, velocity) in modes.items():
                total += adm.obstruction_value(
                    {wave: [e / 2 for e in velocity]}, {wave: amplitude}
                )
            assert g.balance(momentum, modes).obstruction == total

    def test_waves_without_expansion_fail(self) -> None:
        """The linearisation instability, in its correct form.

        Not "waves are forbidden" but "waves without the matching expansion are".
        """
        momentum, modes = self.build(Fraction(0))
        report = g.balance(momentum, modes)
        assert report.graviton_energy > 0
        assert report.obstruction < 0
        assert not report.satisfied

    def test_expansion_without_waves_also_fails(self) -> None:
        """The condition is an equality, so the balance cuts both ways."""
        momentum, _ = self.build(Fraction(1))
        report = g.balance(momentum, {})
        assert report.graviton_energy == 0
        assert report.obstruction == 6
        assert not report.satisfied

    def test_the_required_expansion_is_always_available(self) -> None:
        """``H^2 >= 0`` for any wave content, so the condition is never a veto."""
        for wave in MODES:
            for tensor in adm.transverse_traceless_modes(wave):
                modes = {wave: (tensor, tensor)}
                needed = g.required_hubble_squared(modes)
                assert needed > 0
                assert 6 * needed == g.mode_energy(wave, tensor, tensor)

    def test_shear_raises_the_required_expansion(self) -> None:
        wave = (1, 1, 0)
        tensor = adm.transverse_traceless_modes(wave)[0]
        modes = {wave: (tensor, tensor)}
        bare = g.required_hubble_squared(modes)
        sheared = g.required_hubble_squared(modes, Fraction(2))
        assert sheared == bare + Fraction(1, 3)

    def test_the_homogeneous_mode_is_refused_in_the_wave_sum(self) -> None:
        with pytest.raises(ValueError, match="belongs in the first argument"):
            g.balance(IDENTITY, {(0, 0, 0): (IDENTITY, IDENTITY)})

    def test_friedmann_residual_matches_the_report(self) -> None:
        momentum, modes = self.build(Fraction(1, 2))
        assert g.friedmann_residual(momentum, modes) == 0
        momentum, modes = self.build(Fraction(1))
        assert g.friedmann_residual(momentum, modes) == Fraction(9, 2)


class TestFlatManifolds:
    """How much of the homogeneous sector each holonomy group leaves."""

    @pytest.mark.parametrize("label,moduli", sorted(KNOWN_MODULI.items()))
    def test_the_invariant_momenta_are_the_flat_moduli(self, label, moduli) -> None:
        """External referee: these are the dimensions of the moduli spaces.

        The space of flat metrics on a compact flat manifold, up to scale and
        diffeomorphism, is the holonomy-invariant part of the symmetric tensors,
        classically ``6, 4, 2, 3`` for the four platycosms handled here. The
        computation below builds the action on symmetric tensors from the point
        group and never mentions moduli.
        """
        space = next(s for s in pc.PLATYCOSMS if s.label == label)
        assert g.shear_budget(space)[0] == moduli

    @pytest.mark.parametrize("space", pc.PLATYCOSMS, ids=lambda s: s.label)
    def test_the_isotropic_direction_always_survives(self, space) -> None:
        """``delta`` is invariant under every orthogonal matrix.

        So the expansion that pays for the gravitons exists on every compact flat
        manifold, however much holonomy is present. The balance is a statement
        about compactness, not about symmetry.
        """
        total, free = g.shear_budget(space)
        assert total == free + 1

    def test_the_shear_budget_shrinks_with_holonomy(self) -> None:
        budgets = {s.label: g.shear_budget(s)[1] for s in pc.PLATYCOSMS}
        assert budgets == {"G1": 5, "G2": 3, "G4": 1, "G6": 2}
        assert budgets["G1"] == 5, "the torus keeps every trace-free direction"

    def test_hantzsche_wendt_has_one_condition_and_still_balances(self) -> None:
        """The sharp case.

        No Killing vector fields, so the constant lapse is the only Killing
        initial datum and the Friedmann balance is the only second-order
        condition there is -- yet the isotropic direction is still available to
        satisfy it.
        """
        space = next(s for s in pc.PLATYCOSMS if s.label == "G6")
        assert pc.fixed_subspace_dimension(space) == 0
        assert pc.kid_dimension(space) == 1
        total, free = g.shear_budget(space)
        assert (total, free) == (3, 2)

    @pytest.mark.parametrize("space", pc.PLATYCOSMS, ids=lambda s: s.label)
    def test_the_invariant_tensors_really_are_invariant(self, space) -> None:
        """A direct check of the symmetric-tensor action, on the identity.

        ``A delta A^T = delta`` for orthogonal ``A``, so the identity must be
        fixed by every element; if the slot bookkeeping in
        :func:`graviton._symmetric_action` were wrong this would fail.
        """
        for matrix in pc.point_group(space):
            action = g._symmetric_action(matrix)
            image = [
                sum(action[slot][other] * IDENTITY[other] for other in range(6))
                for slot in range(6)
            ]
            assert image == IDENTITY
