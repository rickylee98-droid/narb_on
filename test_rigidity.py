"""Referees for `rigidity`.

The load-bearing checks:

  * the conjugation residual must be **exactly** ``0.0``, not small -- the
    cancellation is entry by entry, so machine noise would mean the argument is
    wrong;
  * reversing a single plaquette must be visible, and by a wide margin, since
    that asymmetry is what makes the ambiguity one global bit rather than one
    bit per plaquette;
  * the sweep must return exactly two grid points, related by ``(i,j) ->
    (N-i, N-j)`` -- *except* in the degenerate case where a holonomy is real and
    therefore its own conjugate, which `TestDegeneracy` pins down separately.

`TestScope` records that both statements are now proved and what hypothesis the
second one carries.
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

    @pytest.mark.parametrize("reference", [(12, 5), (5, 12), (-1, 5), (5, -1)])
    def test_sweep_rejects_out_of_range_reference(self, reference):
        with pytest.raises(ValueError, match=r"\[0, 12\)"):
            rigidity.rigidity_sweep(resolution=12, reference=reference)

    def test_index_zero_is_a_legal_reference(self):
        """It is a degenerate *case*, not a degenerate *input*.

        An earlier guard rejected index ``0`` on the stated grounds that its
        conjugate would not be a distinct grid point.  That justification was
        wrong -- index ``resolution/2`` is equally self-conjugate and was always
        accepted.  The correct handling is to admit both and let the degeneracy
        show up in the match count, which is what `TestDegeneracy` checks.
        """
        matches = rigidity.rigidity_sweep(resolution=12, reference=(0, 5))
        assert matches == ((0, 5), (0, 7))


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


class TestDegeneracy:
    """When the ``Z/2`` acts trivially, and why.

    The sign-coupling argument runs on ``sin(theta_j) sin(theta_k)``, so it says
    nothing when ``sin(theta_j) = 0`` -- that is, when the holonomy is real.  A
    real holonomy is its own conjugate, so the flip does nothing to it.  These
    tests check the prediction against the sweep, case by case.
    """

    @pytest.mark.parametrize(
        "holonomy, expected",
        [
            (1 + 0j, True),
            (-1 + 0j, True),
            (1j, False),
            (-1j, False),
            (complex(np.cos(0.3), np.sin(0.3)), False),
            (complex(-1.0, 1e-12), True),
        ],
    )
    def test_reality_predicate(self, holonomy, expected):
        assert rigidity.holonomy_is_real(holonomy) is expected

    def test_faithful_when_some_holonomy_is_non_real(self):
        assert rigidity.conjugation_acts_faithfully([1 + 0j, 1j]) is True
        assert rigidity.conjugation_acts_faithfully([1j]) is True

    def test_trivial_when_every_holonomy_is_real(self):
        assert rigidity.conjugation_acts_faithfully([1 + 0j, -1 + 0j]) is False
        assert rigidity.conjugation_acts_faithfully([]) is False

    @pytest.mark.parametrize(
        "reference, real_flags, expected",
        [
            ((6, 13), [False, False], 2),
            ((18, 13), [True, False], 2),
            ((6, 18), [False, True], 2),
            ((18, 18), [True, True], 1),
            ((0, 13), [True, False], 2),
            ((0, 0), [True, True], 1),
        ],
    )
    def test_sweep_matches_the_prediction(self, reference, real_flags, expected):
        """The measured match count, against what reality of the holonomies says.

        On a ``36``-point grid, indices ``0`` and ``18`` are the two real
        holonomies (``omega = +1`` and ``omega = -1``); everything else is
        non-real.
        """
        resolution = 36
        holonomies = [
            complex(
                np.cos(2 * np.pi * index / resolution),
                np.sin(2 * np.pi * index / resolution),
            )
            for index in reference
        ]
        assert [rigidity.holonomy_is_real(h) for h in holonomies] == real_flags

        predicted = rigidity.component_ambiguity_order([holonomies])
        assert predicted == expected

        matches = rigidity.rigidity_sweep(
            resolution=resolution, reference=reference
        )
        assert len(matches) == expected

    def test_both_real_gives_a_single_match(self):
        """The sharpest case: the signature determines the connection outright."""
        matches = rigidity.rigidity_sweep(resolution=36, reference=(18, 18))
        assert matches == ((18, 18),)

    def test_component_order_counts_faithful_components(self):
        assert rigidity.component_ambiguity_order([[1j], [1j]]) == 4
        assert rigidity.component_ambiguity_order([[1j], [1 + 0j]]) == 2
        assert rigidity.component_ambiguity_order([[1 + 0j, -1 + 0j]]) == 1

    def test_component_order_rejects_no_components(self):
        with pytest.raises(ValueError, match="at least one component"):
            rigidity.component_ambiguity_order([])

    def test_connected_case_reproduces_the_headline_constant(self):
        assert (
            rigidity.component_ambiguity_order([[1j, 1j]])
            == rigidity.AMBIGUITY_GROUP_ORDER
        )


class TestScope:
    """Which half is proved and which is computed."""

    def test_both_statements_are_proved(self):
        """Statement two now has an argument, not only a sweep.

        Moments are ``cos(a . theta)``; the fourth moments give ``cos(theta_j)``,
        which pins each angle up to sign, and higher moments give
        ``cos(theta_j +- theta_k)``, whose difference is
        ``2 sin(theta_j) sin(theta_k)``, which pins the relative signs.  The
        hypothesis is that those pair terms exist, which needs closed walks
        crossing two plaquettes -- so the theorem is stated for connected
        complexes and `component_ambiguity_order` handles the rest.
        """
        assert rigidity.AMBIGUITY_IS_PROVED is True
        assert "Two, proved" in rigidity.__doc__

    def test_the_hypothesis_is_stated(self):
        """A proof with an unnamed hypothesis is worse than an honest computation."""
        assert "connected" in rigidity.__doc__
        assert rigidity.component_ambiguity_order.__doc__ is not None
        assert "disconnected" in rigidity.component_ambiguity_order.__doc__

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
