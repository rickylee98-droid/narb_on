"""Referees for `rigidity`.

The load-bearing checks:

  * the conjugation residual must be **exactly** ``0.0``, not small -- the
    cancellation is entry by entry, so machine noise would mean the argument is
    wrong;
  * reversing a single plaquette must be visible, and by a wide margin, since
    that asymmetry is what makes the ambiguity one global bit rather than one
    bit per plaquette;
  * the sweep must return exactly two grid points, related by ``(i,j) ->
    (N-i, N-j)``.

`TestScope` records which half of the theorem is proved and which is computed.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

import insertion
import rigidity


TRIANGLE = insertion.close_under_faces([(0, 1, 2)])
TWO_TRIANGLES = insertion.close_under_faces([(0, 1, 2), (0, 1, 3)])
TETRAHEDRON = insertion.close_under_faces([(0, 1, 2, 3)])
SHELL = insertion.close_under_faces(
    [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
)

COMPLEXES = [TRIANGLE, TWO_TRIANGLES, TETRAHEDRON, SHELL]
NAMES = ["triangle", "two-triangles", "tetrahedron", "shell"]


def _angles(vertices: int, seed: int) -> dict[tuple[int, int], float]:
    generator = np.random.default_rng(seed)
    return {
        tuple(sorted(edge)): float(generator.uniform(0, 2 * np.pi))
        for edge in combinations(range(vertices), 2)
    }


class TestSignature:
    @pytest.mark.parametrize("complex_", COMPLEXES, ids=NAMES)
    def test_shape(self, complex_):
        signature = rigidity.spectral_signature(
            complex_, insertion.phase_function({}), order=6
        )
        assert signature.shape == (len(complex_), 7)

    def test_zeroth_column_is_all_ones(self):
        """Every insertion measure is a probability measure."""
        signature = rigidity.spectral_signature(TRIANGLE, insertion.phase_function({}))
        assert np.allclose(signature[:, 0], 1.0)

    def test_odd_columns_vanish(self):
        signature = rigidity.spectral_signature(
            TWO_TRIANGLES, insertion.phase_function(_angles(4, 1))
        )
        for column in (1, 3, 5, 7):
            assert np.abs(signature[:, column]).max() < 1e-10

    def test_rejects_negative_order(self):
        with pytest.raises(ValueError):
            rigidity.spectral_signature(TRIANGLE, insertion.phase_function({}), -1)

    def test_comparison_rejects_mismatched_shapes(self):
        first = rigidity.spectral_signature(TRIANGLE, insertion.phase_function({}))
        second = rigidity.spectral_signature(
            TETRAHEDRON, insertion.phase_function({})
        )
        with pytest.raises(ValueError, match="different complexes"):
            rigidity.signatures_agree(first, second)


class TestStatementOneConjugationIsInvisible:
    """Proved, and asserted at exactly zero."""

    @pytest.mark.parametrize("complex_, vertices", list(zip(COMPLEXES, [3, 4, 4, 4])), ids=NAMES)
    @pytest.mark.parametrize("seed", range(4))
    def test_residual_is_exactly_zero(self, complex_, vertices: int, seed: int):
        """Not ``< tolerance``. Exactly zero.

        Each entry of ``D`` for the conjugate connection is the conjugate of the
        corresponding entry, and a diagonal moment at a real basis vector is
        real, so the cancellation happens entry by entry.  Machine noise here
        would mean the argument is wrong.
        """
        assert rigidity.conjugation_residual(
            complex_, _angles(vertices, seed)
        ) == 0.0

    @pytest.mark.parametrize("complex_, vertices", list(zip(COMPLEXES, [3, 4, 4, 4])), ids=NAMES)
    def test_predicate(self, complex_, vertices: int):
        angles = _angles(vertices, 11)
        assert rigidity.conjugation_is_invisible(complex_, angles)
        assert rigidity.flux_chirality_is_invisible(complex_, angles)

    def test_conjugation_is_an_involution(self):
        angles = _angles(4, 3)
        twice = rigidity.conjugate_angles(rigidity.conjugate_angles(angles))
        assert twice == angles

    def test_conjugation_actually_changes_the_connection(self):
        """Otherwise the invisibility would be vacuous."""
        angles = _angles(3, 5)
        weight = insertion.phase_function(angles)
        conjugated = insertion.phase_function(rigidity.conjugate_angles(angles))
        holonomy = weight(0, 1) * weight(1, 2) * weight(2, 0)
        other = conjugated(0, 1) * conjugated(1, 2) * conjugated(2, 0)
        assert abs(holonomy - other) > 1e-3
        assert other == pytest.approx(holonomy.conjugate(), abs=1e-12)

    def test_flat_connection_is_its_own_conjugate(self):
        flat = {edge: 0.0 for edge in combinations(range(3), 2)}
        assert rigidity.conjugate_angles(flat) == {
            edge: -0.0 for edge in flat
        }
        assert rigidity.conjugation_residual(TRIANGLE, flat) == 0.0


class TestStatementTwoTheAmbiguityIsExactlyTwo:
    """Computed exhaustively on small complexes."""

    def test_single_plaquette_reversal_is_visible(self):
        assert rigidity.single_plaquette_reversal_is_visible() is True

    def test_visibility_margin_is_wide(self):
        """Order one against a 1e-8 threshold -- no borderline case to tune."""
        simplices = TWO_TRIANGLES

        def signature(one: float, two: float) -> np.ndarray:
            return rigidity.spectral_signature(
                simplices,
                insertion.phase_function(
                    {(0, 1): 0.0, (0, 2): 0.0, (0, 3): 0.0, (1, 2): one, (1, 3): two}
                ),
            )

        base = signature(1.0472, 2.2689)
        flipped = signature(1.0472, -2.2689)
        assert np.abs(base - flipped).max() > 1e-3

    def test_sweep_finds_exactly_the_conjugate_pair(self):
        matches = rigidity.rigidity_sweep(resolution=36, reference=(6, 13))
        assert matches == ((6, 13), (30, 23))
        assert len(matches) == rigidity.AMBIGUITY_GROUP_ORDER

    @pytest.mark.parametrize("reference", [(4, 9), (11, 5)])
    def test_sweep_is_conjugate_symmetric(self, reference):
        resolution = 24
        matches = rigidity.rigidity_sweep(resolution=resolution, reference=reference)
        assert len(matches) == 2
        first, second = matches
        assert first == reference
        assert second == (resolution - reference[0], resolution - reference[1])

    def test_ambiguity_group_order(self):
        assert rigidity.AMBIGUITY_GROUP_ORDER == 2

    def test_sweep_rejects_degenerate_input(self):
        with pytest.raises(ValueError, match="at least four"):
            rigidity.rigidity_sweep(resolution=2)
        with pytest.raises(ValueError, match="strictly inside"):
            rigidity.rigidity_sweep(resolution=12, reference=(0, 5))


class TestTheThreeChiralities:
    """The pattern this module names, checked across the three modules."""

    def test_geometric_chirality_is_invisible(self):
        """`dirac`: mirror point clouds give identical everything."""
        import dirac

        assert dirac.chirality_is_invisible_to_distances() is True

    def test_operator_chirality_is_unbreakable(self):
        """`magnetic`: the grading survives any connection."""
        import magnetic

        assert magnetic.chirality_survives(magnetic.Connection((1.1, 0.2, 2.0)))
        assert not magnetic.supersymmetry_survives(
            magnetic.Connection((1.1, 0.2, 2.0))
        )

    def test_flux_chirality_is_invisible(self):
        """Here: no local spectral data determines the sign of the flux."""
        assert rigidity.flux_chirality_is_invisible(TWO_TRIANGLES, _angles(4, 7))

    def test_all_three_are_a_single_bit(self):
        """Each invisible thing is a ``Z/2``, and that is the pattern."""
        assert rigidity.AMBIGUITY_GROUP_ORDER == 2


class TestScope:
    """Which half is proved and which is computed."""

    def test_the_general_case_is_not_proved(self):
        """Statement two is exhaustive on small complexes, not a theorem.

        What would be needed is an argument that the real parts of all products
        of plaquette holonomies determine those holonomies up to simultaneous
        conjugation.  Plausible and unproved; the flag says so.
        """
        assert rigidity.AMBIGUITY_IS_PROVED is False
        assert "not** proved in general" in rigidity.__doc__

    def test_statement_one_is_proved(self):
        """And its proof is why the residual is exactly zero rather than small."""
        assert rigidity.conjugation_residual(SHELL, _angles(4, 13)) == 0.0

    def test_novelty_is_marked_unverified(self):
        assert "Unverified" in rigidity.__doc__

    def test_no_reconstruction_is_offered(self):
        """Knowing the ambiguity is not the same as inverting the map.

        Nothing here recovers a connection from a signature; the theorem says
        only how many connections share one.
        """
        assert not hasattr(rigidity, "reconstruct_connection")
        assert not hasattr(rigidity, "invert_signature")
