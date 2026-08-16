"""Referees for `character`.

The load-bearing checks, in the order they carry weight:

  * **the two expansion paths must agree.**  One is an integer matrix power over
    ``Z[x^+-]``; the other is a Fourier transform of float moments.  They share
    no code and no arithmetic.  If they agree, the character claim is right; if
    they disagree, nothing downstream is worth reading.
  * **the support law must survive a walk.**  `first_order_carrying` reads the
    expansion; `shortest_walk_carrying` does breadth-first search in the
    ``Z^r``-cover.  Same number, different algorithm.
  * **the Fricke residual must be at machine precision**, since the identity is
    exact and any real error would show up far above ``1e-15``.
  * **the Burnside count must match brute force exactly.**  It is an integer
    prediction with no tolerance to hide in, and it is a test of the *rigidity
    theorem*, not of arithmetic: if the signature failed to separate orbits the
    measured count would come in strictly lower.

`TestNovelty` records which half of this module is Fricke's and which is not.
"""

from __future__ import annotations

import numpy as np
import pytest

import character
import insertion
import rigidity


TWO_TRIANGLES = character.TWO_TRIANGLES
CONNECTION = character.TWO_TRIANGLE_CONNECTION

#: A second complex, to keep the results from being facts about one example.
#: A triangle with a pendant edge: one plaquette, one dangling vertex.
PENDANT = insertion.close_under_faces([(0, 1, 2), (2, 3)])
PENDANT_CONNECTION: character.IntegerConnection = {(1, 2): (1,)}


class TestIntegralConnections:
    def test_angles_are_linear(self):
        angles = character.angles_from(CONNECTION, (0.5, 1.25))
        assert angles == {(1, 2): 0.5, (1, 3): 1.25}

    def test_difference_class_is_a_difference(self):
        connection = {(1, 2): (1, -1)}
        assert character.angles_from(connection, (0.5, 1.25))[(1, 2)] == pytest.approx(
            -0.75
        )

    def test_rejects_unsorted_edge(self):
        with pytest.raises(ValueError, match="sorted pair"):
            character.angles_from({(2, 1): (1, 0)}, (0.0, 0.0))

    def test_rejects_wrong_width(self):
        with pytest.raises(ValueError, match="expected rank"):
            character.angles_from({(1, 2): (1, 0, 0)}, (0.0, 0.0))

    @pytest.mark.parametrize(
        "vector, expected",
        [((1, 0), (1, 0)), ((-1, 0), (1, 0)), ((1, -1), (1, -1)), ((-1, 1), (1, -1))],
    )
    def test_canonical_class_folds_negation(self, vector, expected):
        assert character.canonical_class(vector) == expected

    def test_canonical_class_is_idempotent(self):
        for vector in [(1, 0), (-1, 2), (0, -3), (2, -2)]:
            once = character.canonical_class(vector)
            assert character.canonical_class(once) == once


class TestExactExpansion:
    """The integer path."""

    def test_zeroth_moment_is_one(self):
        assert character.character_expansion(
            TWO_TRIANGLES, (0,), CONNECTION, 0, 2
        ) == {(0, 0): 1}

    @pytest.mark.parametrize("simplex", [(0,), (2,), (0, 1), (0, 1, 2)])
    def test_second_moment_is_flux_blind(self, simplex):
        """The girth law's base case, now visible as a support statement."""
        expansion = character.character_expansion(
            TWO_TRIANGLES, simplex, CONNECTION, 2, 2
        )
        assert set(expansion) == {(0, 0)}

    @pytest.mark.parametrize("simplex", [(0,), (1,), (2,), (3,), (0, 1), (0, 1, 2)])
    @pytest.mark.parametrize("order", [0, 2, 4, 6, 8])
    def test_coefficients_are_integers(self, simplex, order):
        expansion = character.character_expansion(
            TWO_TRIANGLES, simplex, CONNECTION, order, 2
        )
        assert all(isinstance(value, int) for value in expansion.values())

    @pytest.mark.parametrize("simplex", [(0,), (2,), (3,), (0, 1), (1, 2), (0, 1, 3)])
    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_support_is_symmetric_under_negation(self, simplex, order):
        """``c_{-a} = c_a``: the character form of statement one of `rigidity`."""
        expansion = character.character_expansion(
            TWO_TRIANGLES, simplex, CONNECTION, order, 2
        )
        assert character.expansion_is_symmetric(expansion)

    @pytest.mark.parametrize("order", [1, 3, 5, 7])
    def test_odd_moments_vanish_identically(self, order):
        expansion = character.character_expansion(
            TWO_TRIANGLES, (0, 1), CONNECTION, order, 2
        )
        assert expansion == {}

    def test_constant_term_at_order_two_is_the_face_count(self):
        """``M_2 = dim + 1``, recovered as the only surviving character."""
        for simplex in TWO_TRIANGLES:
            expansion = character.character_expansion(
                TWO_TRIANGLES, simplex, CONNECTION, 2, 2
            )
            assert expansion[(0, 0)] == insertion.insertion_moment(
                TWO_TRIANGLES, simplex, insertion.phase_function({}), 2
            )

    def test_flat_evaluation_reproduces_the_float_moment(self):
        """Summing the coefficients is the moment at ``theta = 0``."""
        for order in (2, 4, 6, 8):
            expansion = character.character_expansion(
                TWO_TRIANGLES, (0,), CONNECTION, order, 2
            )
            assert sum(expansion.values()) == pytest.approx(
                insertion.insertion_moment(
                    TWO_TRIANGLES, (0,), insertion.phase_function({}), order
                )
            )

    def test_rejects_negative_order(self):
        with pytest.raises(ValueError, match="non-negative"):
            character.character_expansion(TWO_TRIANGLES, (0,), CONNECTION, -1, 2)

    def test_rejects_absent_simplex(self):
        with pytest.raises(ValueError, match="not in the complex"):
            character.character_expansion(TWO_TRIANGLES, (9,), CONNECTION, 2, 2)

    def test_rejects_zero_rank(self):
        with pytest.raises(ValueError, match="rank must be positive"):
            character.character_expansion(TWO_TRIANGLES, (0,), {}, 2, 0)


class TestTheTwoPathsAgree:
    """The referee. Integer matrix power against a Fourier transform of floats."""

    @pytest.mark.parametrize("simplex", [(0,), (2,), (3,), (0, 2), (0, 1, 2)])
    @pytest.mark.parametrize("order", [2, 4, 6])
    def test_expansions_agree(self, simplex, order):
        exact = character.character_expansion(
            TWO_TRIANGLES, simplex, CONNECTION, order, 2
        )
        transformed = character.character_expansion_by_transform(
            TWO_TRIANGLES, simplex, CONNECTION, order, 2, resolution=12
        )
        assert character.expansions_agree(exact, transformed)

    def test_agreement_at_the_order_that_matters(self):
        """Order eight, where the difference class first appears."""
        exact = character.character_expansion(
            TWO_TRIANGLES, (0,), CONNECTION, 8, 2
        )
        transformed = character.character_expansion_by_transform(
            TWO_TRIANGLES, (0,), CONNECTION, 8, 2, resolution=16
        )
        assert (1, -1) in exact
        assert character.expansions_agree(exact, transformed)

    @pytest.mark.parametrize("order", [4, 6, 8])
    def test_transform_coefficients_are_integral(self, order):
        transformed = character.character_expansion_by_transform(
            TWO_TRIANGLES, (2,), CONNECTION, order, 2, resolution=16
        )
        assert character.expansion_is_integral(transformed)

    def test_disagreement_is_detectable(self):
        """Otherwise the agreement test proves nothing."""
        exact = character.character_expansion(TWO_TRIANGLES, (2,), CONNECTION, 4, 2)
        wrong = {vector: float(value) + 0.5 for vector, value in exact.items()}
        assert not character.expansions_agree(exact, wrong)
        assert not character.expansions_agree(exact, {(0, 0): 8.0})

    def test_transform_coefficients_are_real(self):
        """The moments are real, so an imaginary part would mean the grid aliased."""
        transformed = character.character_expansion_by_transform(
            TWO_TRIANGLES, (0,), CONNECTION, 8, 2, resolution=16
        )
        assert all(isinstance(value, float) for value in transformed.values())

    def test_transform_rejects_a_coarse_grid(self):
        with pytest.raises(ValueError, match="at least four"):
            character.character_expansion_by_transform(
                TWO_TRIANGLES, (0,), CONNECTION, 4, 2, resolution=3
            )

    def test_rank_one_complex_agrees_too(self):
        """A different complex and a different rank, so this is not one example."""
        exact = character.character_expansion(
            PENDANT, (3,), PENDANT_CONNECTION, 6, 1
        )
        transformed = character.character_expansion_by_transform(
            PENDANT, (3,), PENDANT_CONNECTION, 6, 1, resolution=12
        )
        assert character.expansions_agree(exact, transformed)


class TestSupportLaw:
    """Expansion order equals shortest-walk length, class by class, simplex by simplex."""

    @pytest.mark.parametrize(
        "simplex, target, expected",
        [
            ((2,), (1, 0), 4),
            ((2,), (1, -1), 8),
            ((2,), (2, 0), 8),
            ((2,), (0, 1), 10),
            ((0,), (1, 0), 6),
            ((0,), (0, 1), 6),
            ((0,), (1, -1), 8),
            ((0,), (2, 0), 10),
        ],
    )
    def test_measured_orders(self, simplex, target, expected):
        assert (
            character.first_order_carrying(
                TWO_TRIANGLES, simplex, CONNECTION, target, 12, 2
            )
            == expected
        )

    @pytest.mark.parametrize(
        "simplex", [(0,), (2,), (3,), (1, 2), (0, 1, 2)]
    )
    @pytest.mark.parametrize("target", [(1, 0), (0, 1), (1, -1)])
    def test_the_two_routes_agree(self, simplex, target):
        assert character.support_law_holds(
            TWO_TRIANGLES, simplex, CONNECTION, target, 10, 2
        )

    @pytest.mark.parametrize("simplex", list(TWO_TRIANGLES))
    @pytest.mark.parametrize("target", [(1, 0), (0, 1), (1, -1), (2, 0)])
    def test_the_inequality_always_holds(self, simplex, target):
        """The part that is actually a theorem, checked at every simplex."""
        assert character.support_order_is_at_least_walk_length(
            TWO_TRIANGLES, simplex, CONNECTION, target, 10, 2
        )

    @pytest.mark.parametrize(
        "target, walk", [((1, 0), 6), ((0, 1), 6), ((1, -1), 10), ((2, 0), 10)]
    )
    def test_the_cancelled_walks_have_the_lengths_claimed(self, target, walk):
        """Pins the numbers the module docstring quotes for the shared edge."""
        assert (
            character.shortest_walk_carrying(
                TWO_TRIANGLES, (0, 1), CONNECTION, target, 14, 2
            )
            == walk
        )

    @pytest.mark.parametrize("target", [(1, 0), (0, 1), (1, -1)])
    def test_equality_fails_at_the_shared_edge(self, target):
        """The counterexample, kept visible rather than parametrised away.

        Walks carrying each class reach the shared edge, and every one of them
        cancels in the signed sum. Stating the support law as an equality would
        have been wrong, and this is why.
        """
        assert not character.support_law_holds(
            TWO_TRIANGLES, (0, 1), CONNECTION, target, 10, 2
        )
        assert (
            character.shortest_walk_carrying(
                TWO_TRIANGLES, (0, 1), CONNECTION, target, 12, 2
            )
            is not None
        )
        assert (
            character.first_order_carrying(
                TWO_TRIANGLES, (0, 1), CONNECTION, target, 12, 2
            )
            is None
        )

    def test_the_difference_class_needs_order_eight(self):
        """The fact that fixes `rigidity.SIGNATURE_ORDER`.

        Order six cannot see any relation between the two plaquettes, so a
        signature truncated there could not possibly prove the rigidity theorem.
        Eight is the first order that can, and it was picked by guessing.
        """
        for simplex in TWO_TRIANGLES:
            order = character.first_order_carrying(
                TWO_TRIANGLES, simplex, CONNECTION, (1, -1), 8, 2
            )
            assert order in (None, 8)
        assert rigidity.SIGNATURE_ORDER == 8

    def test_a_class_beyond_reach_returns_none(self):
        assert (
            character.first_order_carrying(
                TWO_TRIANGLES, (0,), CONNECTION, (3, 3), 6, 2
            )
            is None
        )
        assert (
            character.shortest_walk_carrying(
                TWO_TRIANGLES, (0,), CONNECTION, (3, 3), 6, 2
            )
            is None
        )

    def test_walk_search_rejects_a_mismatched_target(self):
        with pytest.raises(ValueError, match="expected 2"):
            character.shortest_walk_carrying(
                TWO_TRIANGLES, (0,), CONNECTION, (1,), 6, 2
            )

    def test_walk_search_rejects_negative_depth(self):
        with pytest.raises(ValueError, match="non-negative"):
            character.shortest_walk_carrying(
                TWO_TRIANGLES, (0,), CONNECTION, (1, 0), -1, 2
            )

    def test_flat_class_is_reached_at_order_two(self):
        assert (
            character.shortest_walk_carrying(
                TWO_TRIANGLES, (0,), CONNECTION, (0, 0), 6, 2
            )
            == 2
        )


class TestFluxBlindness:
    """Simplices the connection never reaches, and why they are not geometric."""

    def test_the_shared_edge_is_flux_blind(self):
        assert character.is_flux_blind(TWO_TRIANGLES, (0, 1), CONNECTION, 2, 10)

    @pytest.mark.parametrize("simplex", [(0,), (2,), (3,), (1, 2), (0, 1, 2)])
    def test_everything_else_is_not(self, simplex):
        assert not character.is_flux_blind(TWO_TRIANGLES, simplex, CONNECTION, 2, 10)

    @pytest.mark.parametrize("order, expected", [(2, 4), (4, 16), (6, 64), (8, 256)])
    def test_the_blind_moments_are_powers_of_the_degree(self, order, expected):
        """``4^{n/2}``: a two-point measure at ``+-2``, so ``e_tau`` is an
        eigenvector of the magnetic Hodge Laplacian for every connection."""
        expansion = character.character_expansion(
            TWO_TRIANGLES, (0, 1), CONNECTION, order, 2
        )
        assert expansion == {(0, 0): expected}

    @pytest.mark.parametrize("seed", range(3))
    def test_blindness_survives_a_random_connection(self, seed):
        """The expansion says it; the float path must agree."""
        generator = np.random.default_rng(seed)
        thetas = generator.uniform(0, 2 * np.pi, size=2)
        weight = insertion.phase_function(
            character.angles_from(CONNECTION, tuple(thetas))
        )
        for order, expected in ((2, 4.0), (4, 16.0), (6, 64.0), (8, 256.0)):
            assert insertion.insertion_moment(
                TWO_TRIANGLES, (0, 1), weight, order
            ) == pytest.approx(expected, abs=1e-9)


class TestTheBasepoint:
    """The scope correction: the operator knows the vertex ordering."""

    PERMUTATION = {0: 2, 1: 3, 2: 0, 3: 1}

    def test_the_permutation_is_an_isomorphism(self):
        """Two triangles glued along an edge, relabelled so the shared edge moves."""
        moved = character.relabel_complex(TWO_TRIANGLES, self.PERMUTATION)
        assert moved == insertion.close_under_faces([(0, 2, 3), (1, 2, 3)])

    def test_relabelling_negates_reversed_edges(self):
        """The bookkeeping the comparison stands or falls on."""
        moved = character.relabel_angles({(0, 2): 0.5}, self.PERMUTATION)
        assert moved == {(0, 2): -0.5}
        forward = character.relabel_angles({(0, 1): 0.5}, self.PERMUTATION)
        assert forward == {(2, 3): 0.5}

    def test_relabelling_is_an_involution_here(self):
        angles = {(0, 1): 0.4, (0, 2): 1.1, (1, 3): -0.3}
        twice = character.relabel_angles(
            character.relabel_angles(angles, self.PERMUTATION), self.PERMUTATION
        )
        assert twice == pytest.approx(angles)

    @pytest.mark.parametrize("seed", range(3))
    def test_flat_connections_are_relabelling_invariant(self, seed):
        """No curvature, no path dependence, no basepoint problem."""
        generator = np.random.default_rng(seed)
        angles = character.pure_gauge(4, generator.uniform(0, 2 * np.pi, size=4))
        assert not character.relabelling_changes_the_spectrum(
            TWO_TRIANGLES, angles, self.PERMUTATION
        )

    @pytest.mark.parametrize("seed", range(3))
    def test_curved_connections_are_not(self, seed):
        """The finding. The spectrum is an invariant of the *ordered* complex."""
        generator = np.random.default_rng(100 + seed)
        angles = {
            edge: float(generator.uniform(0, 2 * np.pi))
            for edge in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        }
        assert character.relabelling_changes_the_spectrum(
            TWO_TRIANGLES, angles, self.PERMUTATION
        )

    def test_zero_field_is_invariant(self):
        """The purely combinatorial case, which had better be fine."""
        assert not character.relabelling_changes_the_spectrum(
            TWO_TRIANGLES, {}, self.PERMUTATION
        )

    def test_moments_are_gauge_invariant(self):
        """The property that does hold, and that everything else rests on."""
        generator = np.random.default_rng(21)
        angles = {
            edge: float(generator.uniform(0, 2 * np.pi))
            for edge in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        }
        potentials = generator.uniform(0, 2 * np.pi, size=4)
        gauged = {
            edge: value + float(potentials[edge[1]] - potentials[edge[0]])
            for edge, value in angles.items()
        }
        for simplex in TWO_TRIANGLES:
            for order in (2, 4, 6, 8):
                assert insertion.insertion_moment(
                    TWO_TRIANGLES, simplex, insertion.phase_function(angles), order
                ) == pytest.approx(
                    insertion.insertion_moment(
                        TWO_TRIANGLES, simplex, insertion.phase_function(gauged), order
                    ),
                    abs=1e-9,
                )

    def test_pure_gauge_rejects_a_length_mismatch(self):
        with pytest.raises(ValueError, match="one potential per vertex"):
            character.pure_gauge(4, (0.1, 0.2))

    def test_the_character_data_is_intrinsic(self):
        """The point of working at this level.

        The moments differ between the two labellings; the recovered cosines do
        not, because they are functions of loop holonomies.
        """
        other = insertion.close_under_faces([(0, 2, 3), (1, 2, 3)])
        other_connection: character.IntegerConnection = {(0, 3): (1, 0), (1, 3): (0, 1)}
        thetas = (0.7, 1.3)
        here = character.recover_cosines(TWO_TRIANGLES, CONNECTION, thetas, 8)
        there = character.recover_cosines(other, other_connection, thetas, 10)
        for class_ in ((1, 0), (0, 1), (1, -1)):
            assert here[class_] == pytest.approx(there[class_], abs=1e-10)
        assert character.fricke_identity_holds(
            there[(1, 0)], there[(0, 1)], there[(1, -1)]
        )

    def test_the_convention_is_recorded(self):
        assert character.BASEPOINT_IS_THE_MINIMAL_VERTEX is True
        assert "minimal vertex" in character.__doc__


class TestRecovery:
    """Every cosine the moments reach, peeled off in order of appearance."""

    @pytest.mark.parametrize("thetas", [(0.7, 1.3), (2.0, 0.4), (1.0, 1.0), (0.1, 3.0)])
    def test_recovers_the_plaquette_cosines(self, thetas):
        recovered = character.recover_cosines(TWO_TRIANGLES, CONNECTION, thetas, 8)
        assert recovered[(1, 0)] == pytest.approx(np.cos(thetas[0]), abs=1e-9)
        assert recovered[(0, 1)] == pytest.approx(np.cos(thetas[1]), abs=1e-9)

    @pytest.mark.parametrize("thetas", [(0.7, 1.3), (2.0, 0.4), (0.1, 3.0)])
    def test_recovers_the_difference_cosine(self, thetas):
        """The one that needs order eight, and the one the cubic needs."""
        recovered = character.recover_cosines(TWO_TRIANGLES, CONNECTION, thetas, 8)
        assert recovered[(1, -1)] == pytest.approx(
            np.cos(thetas[0] - thetas[1]), abs=1e-9
        )

    def test_recovers_doubled_classes_too(self):
        thetas = (0.7, 1.3)
        recovered = character.recover_cosines(TWO_TRIANGLES, CONNECTION, thetas, 8)
        assert recovered[(2, 0)] == pytest.approx(np.cos(2 * thetas[0]), abs=1e-9)
        assert recovered[(0, 2)] == pytest.approx(np.cos(2 * thetas[1]), abs=1e-9)

    def test_order_six_cannot_reach_the_difference(self):
        """Which is the support law showing up as a limit on what is recoverable."""
        recovered = character.recover_cosines(TWO_TRIANGLES, CONNECTION, (0.7, 1.3), 6)
        assert (1, 0) in recovered
        assert (1, -1) not in recovered

    def test_rejects_mismatched_rank(self):
        with pytest.raises(ValueError, match="expected rank"):
            character.recover_cosines(TWO_TRIANGLES, CONNECTION, (0.7,), 8, rank=2)

    def test_rejects_negative_depth(self):
        with pytest.raises(ValueError, match="non-negative"):
            character.recover_cosines(TWO_TRIANGLES, CONNECTION, (0.7, 1.3), -2)

    def test_recovery_is_conjugation_blind(self):
        """It has to be -- `rigidity` says the moments cannot tell the difference."""
        forward = character.recover_cosines(TWO_TRIANGLES, CONNECTION, (0.7, 1.3), 8)
        backward = character.recover_cosines(TWO_TRIANGLES, CONNECTION, (-0.7, -1.3), 8)
        assert set(forward) == set(backward)
        for class_ in forward:
            assert forward[class_] == pytest.approx(backward[class_], abs=1e-12)


class TestFrickeCubic:
    """The identity, and the fact that it is satisfied by measured moments."""

    @pytest.mark.parametrize(
        "first, second",
        [(0.7, 1.3), (2.0, 0.4), (0.1, 3.0), (1.0, 1.0), (3.0, 3.0)],
    )
    def test_residual_is_at_machine_precision(self, first, second):
        coordinates = character.fricke_coordinates(thetas=(first, second))
        assert character.fricke_residual(*coordinates) < 1e-12
        assert character.fricke_identity_holds(*coordinates)

    @pytest.mark.parametrize("first", [0.0, 0.3, 1.1, 2.7, np.pi])
    @pytest.mark.parametrize("second", [0.0, 0.9, 2.2, np.pi])
    def test_identity_is_algebraic(self, first, second):
        """Directly from the cosines, no operator involved -- this is Fricke's."""
        assert character.fricke_identity_holds(
            float(np.cos(first)), float(np.cos(second)), float(np.cos(first - second))
        )

    def test_a_point_off_the_surface_is_rejected(self):
        """Otherwise the identity test is vacuous."""
        assert not character.fricke_identity_holds(0.5, 0.5, 0.5)
        assert character.fricke_residual(0.5, 0.5, 0.5) > 0.1

    def test_coordinates_match_the_truth(self):
        first, second = 0.7, 1.3
        one, two, three = character.fricke_coordinates(thetas=(first, second))
        assert one == pytest.approx(np.cos(first), abs=1e-9)
        assert two == pytest.approx(np.cos(second), abs=1e-9)
        assert three == pytest.approx(np.cos(first - second), abs=1e-9)

    def test_coordinates_report_an_unreachable_order(self):
        with pytest.raises(ValueError, match="do not reach the classes"):
            character.fricke_coordinates(thetas=(0.7, 1.3), upto=6)


class TestNodesAreTheDegeneracy:
    """The four nodes of the cubic are the four collapse points of `rigidity`."""

    def test_there_are_exactly_four(self):
        assert len(character.CAYLEY_CUBIC_NODES) == 4
        assert len(set(character.CAYLEY_CUBIC_NODES)) == 4
        assert character.node_count(2) == 4

    def test_nodes_lie_on_the_surface(self):
        for node in character.CAYLEY_CUBIC_NODES:
            assert character.fricke_identity_holds(*(float(x) for x in node))

    def test_nodes_have_vanishing_gradient(self):
        """Which is what makes them nodes rather than smooth points.

        ``grad(u^2+v^2+w^2-2uvw) = 2(u - vw, v - uw, w - uv)``.
        """
        for one, two, three in character.CAYLEY_CUBIC_NODES:
            assert one - two * three == 0
            assert two - one * three == 0
            assert three - one * two == 0

    def test_a_generic_point_is_smooth(self):
        one, two, three = character.fricke_coordinates(thetas=(0.7, 1.3))
        assert abs(one - two * three) > 0.1
        assert not character.is_cayley_node(one, two, three)

    def test_product_of_signs_is_positive(self):
        """The cubic reads ``3 - 2uvw = 1``, so ``uvw = 1``, so an even number of
        minus signs -- which is four sign patterns out of eight, not eight."""
        for one, two, three in character.CAYLEY_CUBIC_NODES:
            assert one * two * three == 1

    @pytest.mark.parametrize(
        "thetas", [(0.0, 0.0), (0.0, np.pi), (np.pi, 0.0), (np.pi, np.pi)]
    )
    def test_two_torsion_maps_to_a_node(self, thetas):
        """The exact correspondence: real holonomies <-> singular points.

        These are the four connections whose plaquette holonomies are both
        ``+-1``.  `rigidity` measured that conjugation acts trivially on them;
        here they land on the four points where the cubic surface is singular.
        """
        coordinates = character.fricke_coordinates(thetas=thetas)
        assert character.is_cayley_node(*coordinates)
        holonomies = [complex(np.cos(angle), np.sin(angle)) for angle in thetas]
        assert not rigidity.conjugation_acts_faithfully(holonomies)
        assert rigidity.component_ambiguity_order([holonomies]) == 1

    @pytest.mark.parametrize("thetas", [(0.7, 1.3), (np.pi, 1.3), (0.0, 2.0)])
    def test_everything_else_is_smooth_and_doubled(self, thetas):
        """The converse half: not a node, so the cover is genuinely two-to-one."""
        coordinates = character.fricke_coordinates(thetas=thetas)
        assert not character.is_cayley_node(*coordinates)
        holonomies = [complex(np.cos(angle), np.sin(angle)) for angle in thetas]
        assert rigidity.conjugation_acts_faithfully(holonomies)
        assert rigidity.component_ambiguity_order([holonomies]) == 2

    def test_the_correspondence_is_a_bijection(self):
        """Four two-torsion points, four nodes, one each -- not four onto one."""
        images = {
            (first, second): tuple(
                int(round(value))
                for value in character.fricke_coordinates(thetas=(first, second))
            )
            for first in (0.0, np.pi)
            for second in (0.0, np.pi)
        }
        assert len(set(images.values())) == 4
        assert set(images.values()) == set(character.CAYLEY_CUBIC_NODES)

    def test_node_count_rejects_zero_rank(self):
        with pytest.raises(ValueError, match="rank must be positive"):
            character.node_count(0)

    @pytest.mark.parametrize("rank, expected", [(1, 2), (2, 4), (3, 8), (5, 32)])
    def test_node_count_is_two_torsion(self, rank, expected):
        assert character.node_count(rank) == expected


class TestBurnsideCount:
    """An integer prediction with no tolerance to hide in."""

    @pytest.mark.parametrize(
        "resolution, expected",
        [(11, 61), (12, 74), (36, 650), (5, 13), (6, 20), (2, 4), (1, 1)],
    )
    def test_closed_form(self, resolution, expected):
        assert character.distinguishable_signature_count(resolution) == expected

    @pytest.mark.parametrize("resolution", [5, 6, 7, 8, 11, 12])
    def test_matches_brute_force(self, resolution):
        """The real content: the count is right only if the signature separates
        orbits, which is the rigidity theorem.  A failure would come in low."""
        assert character.measured_signature_count(
            resolution
        ) == character.distinguishable_signature_count(resolution)

    def test_odd_and_even_differ_by_the_two_torsion(self):
        """``N`` odd has one fixed point, ``N`` even has four."""
        assert character.distinguishable_signature_count(
            12
        ) - 144 // 2 == 4 // 2
        assert character.distinguishable_signature_count(11) - (121 - 1) // 2 == 1

    @pytest.mark.parametrize("rank", [1, 2, 3])
    def test_rank_dependence(self, rank):
        assert character.distinguishable_signature_count(4, rank) == (
            4**rank + 2**rank
        ) // 2

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError, match="resolution must be positive"):
            character.distinguishable_signature_count(0)
        with pytest.raises(ValueError, match="rank must be positive"):
            character.distinguishable_signature_count(4, 0)
        with pytest.raises(ValueError, match="resolution must be positive"):
            character.measured_signature_count(0)

    def test_the_fan_is_built_right(self):
        simplices, connection = character.plaquette_fan(3)
        assert simplices == insertion.close_under_faces(
            [(0, 1, 2), (0, 1, 3), (0, 1, 4)]
        )
        assert connection == {
            (1, 2): (1, 0, 0),
            (1, 3): (0, 1, 0),
            (1, 4): (0, 0, 1),
        }
        assert len(simplices) == 15

    def test_fan_rejects_zero_plaquettes(self):
        with pytest.raises(ValueError, match="at least one plaquette"):
            character.plaquette_fan(0)

    @pytest.mark.parametrize("resolution, expected", [(5, 63), (6, 112)])
    def test_rank_three_matches_too(self, resolution, expected):
        """A test that reaches past what `rigidity` proved.

        The ambiguity argument was made at rank two.  Counting says it is still
        exactly ``Z/2`` at rank three, on a fifteen-simplex complex, and brute
        force agrees.  An extra coincidence at rank three would show up as a
        count coming in low.
        """
        simplices, connection = character.plaquette_fan(3)
        assert character.distinguishable_signature_count(resolution, 3) == expected
        assert (
            character.measured_signature_count(
                resolution, simplices, connection, rank=3
            )
            == expected
        )

    def test_rank_one_matches_too(self):
        """The other end: a single triangle, one plaquette, ``T^1/+-``."""
        simplices, connection = character.plaquette_fan(1)
        for resolution in (7, 8):
            assert character.measured_signature_count(
                resolution, simplices, connection, rank=1
            ) == character.distinguishable_signature_count(resolution, 1)

    @pytest.mark.parametrize(
        "resolution, expected", [(7, 16), (8, 25), (5, 9), (6, 16)]
    )
    def test_truncated_count_matches_brute_force(self, resolution, expected):
        """Below the coupling order the signs are independent, and the count says so.

        The prediction is ``((N + g) / 2)^rank`` rather than ``(N^rank +
        g^rank)/2``, and the two differ: ``25`` against ``34`` at ``N = 8``.
        """
        assert (
            character.distinguishable_signature_count(resolution, 2, 6) == expected
        )
        assert character.measured_signature_count(resolution, order=6) == expected

    @pytest.mark.parametrize("resolution, expected", [(5, 27), (6, 64)])
    def test_truncated_count_at_rank_three(self, resolution, expected):
        simplices, connection = character.plaquette_fan(3)
        assert (
            character.distinguishable_signature_count(resolution, 3, 6) == expected
        )
        assert (
            character.measured_signature_count(
                resolution, simplices, connection, order=6, rank=3
            )
            == expected
        )

    def test_the_two_regimes_differ(self):
        """Otherwise the truncation law would be saying nothing."""
        assert character.distinguishable_signature_count(
            8, 2, 6
        ) < character.distinguishable_signature_count(8, 2, 8)

    def test_rank_one_has_no_regimes(self):
        """With a single class there is nothing to couple, so the two agree."""
        for resolution in (7, 8, 11):
            assert character.distinguishable_signature_count(
                resolution, 1, 6
            ) == character.distinguishable_signature_count(resolution, 1, 8)

    def test_measured_count_rejects_zero_rank(self):
        with pytest.raises(ValueError, match="rank must be positive"):
            character.measured_signature_count(4, rank=0)

    def test_the_count_is_below_the_grid(self):
        """Roughly half, which is the price of the invisible bit."""
        for resolution in (11, 12, 36):
            assert character.distinguishable_signature_count(resolution) < (
                resolution**2
            )
            assert character.distinguishable_signature_count(resolution) > (
                resolution**2
            ) // 2 - 1


class TestGeneralRank:
    """The theorem's hypothesis, discharged rather than assumed."""

    @pytest.mark.parametrize(
        "target, expected", [((1, 0), ((2,), 4)), ((0, 1), ((3,), 4)), ((1, -1), ((0,), 8))]
    )
    def test_which_simplex_resolves_which_class(self, target, expected):
        assert (
            character.resolving_simplex(TWO_TRIANGLES, CONNECTION, target, 2, 10)
            == expected
        )

    def test_an_unreachable_class_resolves_nowhere(self):
        assert (
            character.resolving_simplex(TWO_TRIANGLES, CONNECTION, (3, 3), 2, 6)
            is None
        )

    @pytest.mark.parametrize("count", [1, 2, 3])
    def test_the_hypothesis_holds_on_fans(self, count):
        """Which is why the rank-three count comes out at ``Z/2``."""
        simplices, connection = character.plaquette_fan(count)
        assert character.classes_are_resolved(simplices, connection, count, 10)

    def test_the_hypothesis_fails_under_truncation(self):
        """And it fails exactly where the ambiguity is known to be bigger.

        Truncated at six the difference class is resolved nowhere, the signs are
        independent, and the sweep returns four matches rather than two. The
        predicate and the measurement agree about where the theorem applies.
        """
        assert not character.classes_are_resolved(TWO_TRIANGLES, CONNECTION, 2, 6)
        assert character.truncated_ambiguity_order(6, 2) == 4
        assert character.classes_are_resolved(TWO_TRIANGLES, CONNECTION, 2, 8)
        assert character.truncated_ambiguity_order(8, 2) == 2

    def test_the_flux_blind_simplex_resolves_nothing(self):
        """The reason the hypothesis is a condition and not a formality."""
        for target in ((1, 0), (0, 1), (1, -1)):
            expansion_at_the_edge = [
                character.character_expansion(
                    TWO_TRIANGLES, (0, 1), CONNECTION, order, 2
                ).get(target, 0)
                for order in range(2, 13, 2)
            ]
            assert expansion_at_the_edge == [0] * 6

    def test_resolution_rejects_zero_rank(self):
        with pytest.raises(ValueError, match="rank must be positive"):
            character.classes_are_resolved(TWO_TRIANGLES, CONNECTION, 0)

    def test_the_open_part_is_marked_open(self):
        """A hypothesis that is checked per complex is not a hypothesis proved."""
        assert "Proving the hypothesis in general is open" in character.__doc__


class TestTruncation:
    """Why the signature order is load-bearing, and what happens below it."""

    def test_the_coupling_order_matches_the_signature_order(self):
        assert character.COUPLING_ORDER == rigidity.SIGNATURE_ORDER

    @pytest.mark.parametrize(
        "order, expected", [(0, False), (4, False), (6, False), (8, True), (10, True)]
    )
    def test_coupling_predicate(self, order, expected):
        assert character.moments_couple_the_signs(order) is expected

    def test_coupling_predicate_rejects_negative_order(self):
        with pytest.raises(ValueError, match="non-negative"):
            character.moments_couple_the_signs(-1)

    @pytest.mark.parametrize(
        "order, rank, expected",
        [(6, 2, 4), (8, 2, 2), (6, 3, 8), (8, 3, 2), (6, 1, 2), (8, 1, 2)],
    )
    def test_ambiguity_order(self, order, rank, expected):
        assert character.truncated_ambiguity_order(order, rank) == expected

    def test_ambiguity_order_rejects_zero_rank(self):
        with pytest.raises(ValueError, match="rank must be positive"):
            character.truncated_ambiguity_order(8, 0)

    @pytest.mark.parametrize("order, expected", [(4, 4), (6, 4), (8, 2)])
    def test_the_sweep_confirms_it(self, order, expected):
        """The sharpest form: a sweep truncated at six returns **four** matches.

        The two plaquette signs are independent until a walk crosses both, so
        below order eight the ambiguity really is ``(Z/2)^2``. The rigidity
        theorem is a statement about the untruncated signature, and this is the
        measurement that shows the truncation is not incidental.
        """
        matches = rigidity.rigidity_sweep(
            resolution=36, reference=(6, 13), order=order
        )
        assert len(matches) == expected
        assert len(matches) == character.truncated_ambiguity_order(order, 2)

    def test_the_four_matches_are_the_independent_flips(self):
        matches = rigidity.rigidity_sweep(resolution=36, reference=(6, 13), order=6)
        assert set(matches) == {(6, 13), (6, 23), (30, 13), (30, 23)}

    def test_the_two_matches_are_the_diagonal(self):
        matches = rigidity.rigidity_sweep(resolution=36, reference=(6, 13), order=8)
        assert set(matches) == {(6, 13), (30, 23)}


class TestNovelty:
    """Whose mathematics each piece is."""

    def test_the_cubic_is_frickes(self):
        assert "Fricke" in character.__doc__
        assert "1897" in character.__doc__
        assert "Cayley cubic" in character.__doc__

    def test_burnside_is_named(self):
        assert "Burnside" in character.__doc__

    def test_the_density_of_states_withdrawal_is_carried_forward(self):
        """`insertion` withdrew a novelty claim; this module must not re-make it."""
        assert "arXiv:2502.07558" in character.__doc__
        assert "withdrawal" in character.__doc__

    def test_the_new_part_is_stated_and_marked_unverified(self):
        assert "unverified against the literature" in character.__doc__
        assert "arxiv.org is unreachable" in character.__doc__

    def test_no_reconstruction_is_offered_here_either(self):
        """Recovering cosines is not recovering a connection: the cosines are
        exactly the orbit invariants, and the orbit is what the theorem says is
        the most that can be known."""
        assert not hasattr(character, "reconstruct_connection")
        assert not hasattr(rigidity, "reconstruct_connection")
