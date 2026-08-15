"""Referees for `magnetic`.

The load-bearing checks:

  * the anticommutator is *structurally* zero, so it is asserted at exactly
    zero rather than at a tolerance -- if curvature could reach it, it would
    show up as machine noise instead;
  * the holonomy defect, ``|d^2|`` and the off-block norm of ``D^2`` are three
    separately computed quantities and must be one number;
  * statements one and two must come apart exactly at curvature, which is the
    whole content of the module.
"""

from __future__ import annotations

import cmath

import numpy as np
import pytest

import magnetic


FLAT = magnetic.flat_connection()
CURVED = magnetic.Connection((0.3, 0.4, 0.5))
STRONGLY_CURVED = magnetic.Connection((1.1, 0.2, 2.0))
TRIVIAL = magnetic.Connection((0.0, 0.0, 0.0))

CONNECTIONS = [TRIVIAL, FLAT, CURVED, STRONGLY_CURVED]
NAMES = ["trivial", "flat", "curved", "strongly-curved"]
CURVED_ONLY = [CURVED, STRONGLY_CURVED]


class TestConnection:
    def test_rejects_wrong_phase_count(self):
        with pytest.raises(ValueError, match="three edge phases"):
            magnetic.Connection((0.1, 0.2))

    def test_rejects_non_numeric_phase(self):
        with pytest.raises(TypeError):
            magnetic.Connection((0.1, "x", 0.3))

    def test_weights_have_unit_modulus(self):
        for connection in CONNECTIONS:
            for edge in magnetic.EDGES:
                assert abs(connection.weight(edge)) == pytest.approx(1.0)

    def test_reverse_traversal_conjugates(self):
        for edge in magnetic.EDGES:
            forward = CURVED.weight(edge)
            backward = CURVED.weight((edge[1], edge[0]))
            assert forward * backward == pytest.approx(1.0)

    def test_rejects_non_edge(self):
        with pytest.raises(ValueError, match="not an edge"):
            CURVED.weight((0, 5))

    def test_flat_connection_is_flat(self):
        assert magnetic.is_flat(FLAT)
        assert magnetic.is_flat(TRIVIAL)

    @pytest.mark.parametrize("connection", CURVED_ONLY, ids=["curved", "strong"])
    def test_curved_connections_are_not_flat(
        self, connection: magnetic.Connection
    ):
        assert not magnetic.is_flat(connection)

    def test_holonomy_of_flat_is_one(self):
        assert magnetic.triangle_holonomy(FLAT) == pytest.approx(1.0)

    def test_holonomy_has_unit_modulus(self):
        for connection in CONNECTIONS:
            assert abs(magnetic.triangle_holonomy(connection)) == pytest.approx(1.0)


class TestStatementOneChiralitySurvives:
    """Curvature cannot reach the grading."""

    @pytest.mark.parametrize("connection", CONNECTIONS, ids=NAMES)
    def test_anticommutator_is_exactly_zero(
        self, connection: magnetic.Connection
    ):
        """Asserted at exactly zero, not at a tolerance.

        ``Gamma`` negates precisely the blocks ``D`` occupies, so the sum
        cancels entry by entry whatever the entries are.  If curvature could
        reach this it would appear as machine noise; it does not appear at all.
        """
        assert magnetic.chiral_anticommutator_norm(connection) == 0.0

    @pytest.mark.parametrize("connection", CONNECTIONS, ids=NAMES)
    def test_spectrum_is_symmetric(self, connection: magnetic.Connection):
        assert magnetic.spectrum_asymmetry(connection) < 1e-12

    @pytest.mark.parametrize("connection", CONNECTIONS, ids=NAMES)
    def test_chirality_survives(self, connection: magnetic.Connection):
        assert magnetic.chirality_survives(connection) is True

    @pytest.mark.parametrize("connection", CONNECTIONS, ids=NAMES)
    def test_dirac_is_hermitian(self, connection: magnetic.Connection):
        operator = magnetic.magnetic_dirac(connection)
        assert np.allclose(operator, operator.conj().T)

    @pytest.mark.parametrize("connection", CONNECTIONS, ids=NAMES)
    def test_spectrum_is_real(self, connection: magnetic.Connection):
        assert np.all(np.isreal(magnetic.spectrum(connection)))

    def test_grading_is_an_involution(self):
        grading = magnetic.grading_operator()
        assert np.allclose(grading @ grading, np.eye(7))


class TestStatementTwoSupersymmetryDies:
    """What curvature actually breaks."""

    @pytest.mark.parametrize("connection", [TRIVIAL, FLAT], ids=["trivial", "flat"])
    def test_flat_keeps_supersymmetry(self, connection: magnetic.Connection):
        assert magnetic.supersymmetry_survives(connection) is True
        assert magnetic.curvature_norm(connection) < 1e-12

    @pytest.mark.parametrize("connection", CURVED_ONLY, ids=["curved", "strong"])
    def test_curvature_destroys_supersymmetry(
        self, connection: magnetic.Connection
    ):
        assert magnetic.supersymmetry_survives(connection) is False
        assert magnetic.curvature_norm(connection) > 1e-3

    @pytest.mark.parametrize("connection", CONNECTIONS, ids=NAMES)
    def test_three_quantities_are_one_number(
        self, connection: magnetic.Connection
    ):
        """Holonomy defect, ``|d^2|`` and the ``D^2`` off-block agree.

        Computed by three separate routes -- a product of edge weights, a matrix
        product, and a masked norm of a different matrix product -- and they
        must coincide.  This is the identity behind statement two.
        """
        defect = abs(magnetic.triangle_holonomy(connection) - 1)
        assert magnetic.curvature_norm(connection) == pytest.approx(
            defect, abs=1e-12
        )
        assert magnetic.dirac_square_defect(connection) == pytest.approx(
            defect, abs=1e-12
        )

    def test_curvature_formula(self):
        """``d_1 d_0`` has the single entry ``sigma_uv sigma_vw - sigma_uw``."""
        predicted = (
            CURVED.weight((0, 1)) * CURVED.weight((1, 2)) - CURVED.weight((0, 2))
        )
        matrix = magnetic.curvature_matrix(CURVED)
        assert matrix.shape == (1, 3)
        assert matrix[0, 2] == pytest.approx(predicted)


class TestTheStatementsComeApart:
    """The point of the module, in one test."""

    def test_curvature_separates_chirality_from_supersymmetry(self):
        """Both hold when flat; exactly one survives when curved."""
        assert magnetic.chirality_survives(FLAT)
        assert magnetic.supersymmetry_survives(FLAT)
        assert magnetic.chirality_survives(CURVED)
        assert not magnetic.supersymmetry_survives(CURVED)

    def test_the_briefs_premise_is_false(self):
        """"Magnetic phases break the spectral symmetry" -- they do not.

        This is the claim the module exists to correct.  The spectrum stays
        ``+-`` paired under arbitrary curvature.
        """
        for connection in CURVED_ONLY:
            assert magnetic.spectrum_asymmetry(connection) < 1e-12
            assert not magnetic.is_flat(connection)

    def test_curvature_does_change_the_spectrum(self):
        """It is not that the connection does nothing -- it splits degeneracies.

        The flat spectrum has a triple ``+-sqrt(3)``; curvature splits it.  So
        the magnetic operator sees holonomy, just not by breaking chirality.
        """
        flat_values = np.round(magnetic.spectrum(FLAT), 6)
        curved_values = np.round(magnetic.spectrum(CURVED), 6)
        assert len(set(flat_values)) < len(set(curved_values))
        assert not np.allclose(flat_values, curved_values)


class TestStatementThreeTheIndex:
    """The one genuine chiral asymmetry, and it is topological."""

    @pytest.mark.parametrize(
        "betti, split, index",
        [
            ((1, 1), (1, 1), 0),
            ((1, 0, 0), (1, 0), 1),
            ((1, 0, 1), (2, 0), 2),
            ((2, 0), (2, 0), 2),
            ((1, 0), (1, 0), 1),
        ],
        ids=["circle", "disk", "sphere", "two-edges", "path"],
    )
    def test_index_is_the_euler_characteristic(
        self, betti: tuple[int, ...], split: tuple[int, int], index: int
    ):
        assert magnetic.kernel_parity_split(betti) == split
        assert magnetic.index_from_betti(betti) == index
        assert index == sum((-1) ** k * b for k, b in enumerate(betti))

    def test_the_asymmetry_exists_at_zero_flux(self):
        """It is topological, not magnetic, and homology already reports it."""
        assert magnetic.index_from_betti((1, 0, 1)) == 2
        assert magnetic.is_flat(TRIVIAL)

    def test_only_the_circle_is_balanced(self):
        assert magnetic.index_from_betti((1, 1)) == 0
        assert magnetic.index_from_betti((1, 0, 0)) != 0

    def test_rejects_bad_betti(self):
        with pytest.raises(ValueError, match="at least one"):
            magnetic.kernel_parity_split(())
        with pytest.raises(ValueError, match="non-negative"):
            magnetic.kernel_parity_split((1, -1))


class TestWeylStability:
    """All that is claimed about stability, and it is free."""

    @pytest.mark.parametrize("first", CONNECTIONS, ids=NAMES)
    @pytest.mark.parametrize("second", CONNECTIONS, ids=NAMES)
    def test_bound_holds(
        self, first: magnetic.Connection, second: magnetic.Connection
    ):
        assert magnetic.weyl_bound_holds(first, second) is True

    def test_small_perturbations_move_eigenvalues_little(self):
        nudged = magnetic.Connection((0.3001, 0.4, 0.5))
        shift = np.abs(magnetic.spectrum(CURVED) - magnetic.spectrum(nudged)).max()
        assert shift < 1e-3


class TestWhatThisDoesNotClaim:
    """The boundary, asserted so it cannot drift."""

    def test_no_filtration_stability_is_claimed(self):
        """Weyl is fixed-combinatorics. The open problem is not.

        Along a filtration the operators change dimension, eigenvalues are not
        functorial under interleaving, and no bottleneck- or Wasserstein-
        stability theorem exists for raw eigenvalue-valued descriptors.  Nothing
        here addresses that, and a `persistent_magnetic_stability` would be
        claiming the open problem was solved.
        """
        assert not hasattr(magnetic, "persistent_magnetic_stability")
        assert not hasattr(magnetic, "bottleneck_distance")
        assert magnetic.weyl_bound_holds(FLAT, CURVED)

    def test_no_diamagnetic_inequality_is_assumed(self):
        """It fails above degree zero for magnetic Hodge Laplacians.

        Recorded because the natural intuition -- that flux only raises the
        spectral gap -- is false in higher degree, so nothing here may lean on
        it.
        """
        assert not hasattr(magnetic, "diamagnetic_bound")

    def test_nothing_here_is_new_mathematics(self):
        """The ingredients are all known; only the conjunction appears unstated."""
        assert "None of the ingredients" in magnetic.__doc__
        assert "not as new mathematics" in magnetic.__doc__

    def test_cohomology_is_not_defined_under_curvature(self):
        """A non-flat connection gives a curved dg-module, not a cochain complex.

        So `index_from_betti` takes Betti numbers as *input* rather than
        computing them from a curved operator -- there is nothing to compute
        them from.  The index statement is about the flat case.
        """
        assert not magnetic.supersymmetry_survives(CURVED)
        assert magnetic.index_from_betti((1, 0, 1)) == 2
