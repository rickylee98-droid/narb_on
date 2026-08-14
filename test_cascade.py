"""Referees for the transport measurements on the architecture.

Four checks the module does not control:

* Energy is conserved by the Galerkin system to ``10^-12``. Euler conserves it
  exactly and so does any truncation, so this measures the integrator, and it is
  what licenses reading anything else off a trajectory.
* Reality and the divergence-free condition hold at every mode of the initial
  data, so what is being integrated really is a velocity field.
* The return fraction must fall as the chain lengthens. A single gate is a triad
  and triads are integrable, so near-full recovery at two shells is the control
  that makes the drop at three and beyond mean something.
* The net export must be almost phase-independent. If it were not, the transport
  would be an artefact of one lucky draw.
"""

from __future__ import annotations

import numpy as np
import pytest

import cascade as ca
import coherence as co


def system(shells: int = 3, growth: int = 2) -> ca.GalerkinSystem:
    arch = co.architecture(ca.SEED_VECTOR, ca.SEED_PARTNER, [growth] * (shells - 1))
    return ca.build_system(arch)


class TestTheSystemIsAVelocityField:
    @pytest.mark.parametrize("shells", [2, 3, 4])
    def test_initial_data_is_real_and_divergence_free(self, shells) -> None:
        built = system(shells)
        state = ca.initial_data(built)
        index = {mode: position for position, mode in enumerate(built.modes)}
        for position, mode in enumerate(built.modes):
            partner = index[tuple(-component for component in mode)]
            assert np.allclose(state[position], np.conj(state[partner])), "reality"
            assert abs(built.wavevectors[position] @ state[position]) < 1e-12, "transverse"

    @pytest.mark.parametrize("shells", [2, 3, 4])
    def test_every_quadratic_term_lands_on_a_mode(self, shells) -> None:
        """The index set is the architecture's triad list and nothing else."""
        built = system(shells)
        assert len(built.targets) == len(built.left) == len(built.right)
        for target, left, right in zip(built.targets, built.left, built.right):
            total = tuple(
                a + b for a, b in zip(built.modes[left], built.modes[right])
            )
            assert total == built.modes[target]

    def test_the_shell_labels_cover_every_mode(self) -> None:
        built = system(4)
        assert built.shell_count == 4
        assert len(built.shell) == len(built.modes)
        assert sorted(set(built.shell)) == [0, 1, 2, 3]

    def test_a_single_shell_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two shells"):
            ca.evolve(1, 10.0)


class TestConservation:
    @pytest.mark.parametrize("shells", [2, 3])
    def test_energy_is_conserved(self, shells) -> None:
        assert ca.energy_drift(shells, 100.0) < 1e-9

    def test_conservation_survives_a_long_run(self) -> None:
        """Hundreds of turnover times, which is the window the results use."""
        assert ca.energy_drift(3, 400.0) < 1e-9

    def test_viscosity_removes_energy(self) -> None:
        """A control: with viscosity the total must fall, so the check is not blind."""
        _, energies = ca.evolve(3, 20.0, samples=3, viscosity=1e-3)
        totals = energies.sum(axis=1)
        assert totals[-1] < totals[0]


class TestTransport:
    def test_energy_moves_up_the_chain(self) -> None:
        _, energies = ca.evolve(4, 300.0, samples=40)
        assert energies[-1, 0] < energies[0, 0], "the bottom shell drains"
        for shell in (1, 2, 3):
            assert energies[:, shell].max() > 5 * energies[0, shell], "upper shells fill"

    def test_a_single_gate_is_nearly_reversible(self) -> None:
        """Two shells is one triad, and a triad is integrable: it gives it back."""
        assert ca.return_fraction(2, 600.0) > 0.5

    def test_the_chain_suppresses_the_return(self) -> None:
        fractions = [ca.return_fraction(n, 600.0) for n in (2, 3, 4)]
        assert fractions[0] > fractions[1] > fractions[2]
        assert fractions[-1] < 0.35

    def test_the_bottom_shell_must_export_for_the_fraction_to_mean_anything(self) -> None:
        """The guard: a ratio against zero export would be meaningless."""
        with pytest.raises(ValueError, match="never exported"):
            ca.return_fraction(2, 0.0)


class TestIncoherence:
    """The obstruction, measured."""

    def test_the_flux_reverses_within_a_few_turnover_times(self) -> None:
        turnover = ca.turnover_time(3)
        reversal = ca.first_flux_reversal(3, 100.0)
        assert reversal < 20 * turnover
        assert reversal > 0

    @pytest.mark.parametrize("seed", [0, 1, 2, 4])
    def test_it_reverses_for_every_seed(self, seed) -> None:
        """Not one unlucky draw: the transfer is oscillatory whatever the phases."""
        assert ca.first_flux_reversal(3, 100.0, seed=seed) < 100.0

    def test_the_net_export_barely_depends_on_the_phases(self) -> None:
        """The surprise, and the reason to call the transport robust.

        Rotating every independent amplitude by an arbitrary phase changes the
        exported energy by well under a percent, so the cascade is not a
        fine-tuned coherent transfer -- which is also why it cannot be matched to
        a coherent shell-model trajectory.
        """
        baseline = ca.net_export(3, 20.0)
        generator = np.random.default_rng(11)
        for _ in range(4):
            rotated = ca.net_export(
                3, 20.0, phases=generator.uniform(0, 2 * np.pi, size=5)
            )
            assert abs(rotated - baseline) / baseline < 0.05

    def test_the_export_is_a_real_fraction_of_the_bottom_shell(self) -> None:
        """Small per turnover, but not negligible over the window."""
        _, energies = ca.evolve(3, 20.0, samples=2)
        assert 0.02 < ca.net_export(3, 20.0) / energies[0, 0] < 0.5

    def test_the_turnover_time_sets_the_scale(self) -> None:
        turnover = ca.turnover_time(3)
        assert 0.1 < turnover < 1.0
