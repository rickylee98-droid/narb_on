"""Referees for `insertion`.

The load-bearing checks:

  * the flat moments must be exact integers ``(k+1)^j``, not approximations --
    that is what pins the measure to the two-point one;
  * ``M_2`` must equal the face count under random connections and random
    ambient complexes, since the claim is that flux is invisible at second
    order;
  * the kurtosis identity must reproduce `magnetic.curvature_norm` squared,
    computed by a completely different route -- a matrix moment against a
    product of edge weights.

`TestWhereItStops` records the dimension-two boundary as a result rather than
an omission.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

import insertion
import magnetic


FLAT = insertion.phase_function({})
TRIANGLE = insertion.close_under_faces([(0, 1, 2)])
TETRAHEDRON = insertion.close_under_faces([(0, 1, 2, 3)])


def _random_angles(vertices: int, seed: int) -> dict[tuple[int, int], float]:
    rng = np.random.default_rng(seed)
    return {
        tuple(sorted(edge)): float(rng.uniform(0, 2 * np.pi))
        for edge in combinations(range(vertices), 2)
    }


class TestConstruction:
    def test_closure_is_sorted_by_dimension(self):
        closed = insertion.close_under_faces([(0, 1, 2)])
        assert closed == ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))

    def test_closure_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one face"):
            insertion.close_under_faces([])

    def test_closure_rejects_empty_face(self):
        with pytest.raises(ValueError, match="at least one vertex"):
            insertion.close_under_faces([()])

    def test_closure_rejects_repeated_vertex(self):
        with pytest.raises(ValueError, match="repeats"):
            insertion.close_under_faces([(0, 0, 1)])

    def test_phase_function_conjugates_on_reversal(self):
        weight = insertion.phase_function({(0, 1): 0.7})
        assert weight(0, 1) * weight(1, 0) == pytest.approx(1.0)

    def test_phase_function_rejects_unsorted_key(self):
        with pytest.raises(ValueError, match="sorted pair"):
            insertion.phase_function({(1, 0): 0.5})

    def test_moment_rejects_missing_simplex(self):
        with pytest.raises(ValueError, match="not in the complex"):
            insertion.insertion_moment(TRIANGLE, (7, 8), FLAT, 2)

    def test_moment_rejects_negative_order(self):
        with pytest.raises(ValueError):
            insertion.insertion_moment(TRIANGLE, (0, 1, 2), FLAT, -1)


class TestStatementOneSymmetry:
    """The measure is symmetric about zero, for any connection."""

    def test_zeroth_moment_is_one(self):
        """It is a probability measure."""
        for simplex in [(0, 1), (0, 1, 2)]:
            assert insertion.insertion_moment(
                TRIANGLE, simplex, FLAT, 0
            ) == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_odd_moments_vanish_under_curvature(self, seed: int):
        weight = insertion.phase_function(_random_angles(3, seed))
        assert insertion.odd_moments_vanish(TRIANGLE, (0, 1, 2), weight)

    @pytest.mark.parametrize("seed", [5, 6])
    def test_odd_moments_vanish_in_higher_dimension(self, seed: int):
        weight = insertion.phase_function(_random_angles(4, seed))
        assert insertion.odd_moments_vanish(TETRAHEDRON, (0, 1, 2, 3), weight)

    def test_symmetry_is_inherited_from_the_grading(self):
        """The same chiral symmetry `magnetic` proves, seen on a measure."""
        weight = insertion.phase_function(_random_angles(3, 9))
        assert insertion.odd_moments_vanish(TRIANGLE, (0, 1, 2), weight)
        assert magnetic.chirality_survives(magnetic.Connection((0.3, 0.4, 0.5)))


class TestStatementTwoSecondMoment:
    """``M_2 = dim + 1``, and flux cannot reach it."""

    @pytest.mark.parametrize("dimension", range(6))
    def test_predicted_value(self, dimension: int):
        assert insertion.second_moment(dimension) == dimension + 1

    @pytest.mark.parametrize("dimension", [1, 2, 3, 4])
    def test_holds_at_zero_flux(self, dimension: int):
        simplex = tuple(range(dimension + 1))
        complex_ = insertion.close_under_faces([simplex])
        assert insertion.second_moment_is_combinatorial(complex_, simplex, FLAT)

    @pytest.mark.parametrize("seed", range(6))
    def test_holds_under_random_curvature(self, seed: int):
        weight = insertion.phase_function(_random_angles(4, seed))
        assert insertion.second_moment_is_combinatorial(
            TETRAHEDRON, (0, 1, 2, 3), weight
        )

    @pytest.mark.parametrize("seed", [11, 12, 13])
    def test_holds_in_a_larger_ambient(self, seed: int):
        """The ambient complex cannot change it either."""
        bigger = insertion.close_under_faces([(0, 1, 2), (0, 3), (1, 3), (2, 3)])
        weight = insertion.phase_function(_random_angles(4, seed))
        assert insertion.second_moment_is_combinatorial(bigger, (0, 1, 2), weight)

    def test_rejects_negative_dimension(self):
        with pytest.raises(ValueError):
            insertion.second_moment(-1)


class TestStatementThreeBernoulli:
    """At zero flux the measure is known completely."""

    @pytest.mark.parametrize("dimension", [1, 2, 3, 4, 5])
    def test_flat_moments_are_exact_powers(self, dimension: int):
        simplex = tuple(range(dimension + 1))
        complex_ = insertion.close_under_faces([simplex])
        for order in range(7):
            assert insertion.insertion_moment(
                complex_, simplex, FLAT, order
            ) == pytest.approx(insertion.flat_moment(dimension, order), abs=1e-9)

    @pytest.mark.parametrize("dimension", [1, 2, 3, 4, 5])
    def test_even_moments_are_integers(self, dimension: int):
        """``2, 4, 8``; ``3, 9, 27``; ``4, 16, 64`` -- not fitted, exact."""
        for order in (2, 4, 6):
            value = insertion.flat_moment(dimension, order)
            assert value == pytest.approx(round(value), abs=1e-12)
            assert value == (dimension + 1) ** (order // 2)

    @pytest.mark.parametrize("dimension", range(6))
    def test_measure_is_two_point(self, dimension: int):
        assert insertion.flat_measure_is_bernoulli(dimension)

    @pytest.mark.parametrize("dimension", range(6))
    def test_support_is_the_root_of_the_face_count(self, dimension: int):
        low, high = insertion.insertion_measure_support(dimension)
        assert high == pytest.approx(np.sqrt(dimension + 1))
        assert low == pytest.approx(-high)

    def test_flat_odd_moments_are_zero(self):
        for dimension in range(4):
            for order in (1, 3, 5):
                assert insertion.flat_moment(dimension, order) == 0.0

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            insertion.flat_moment(-1, 2)
        with pytest.raises(ValueError):
            insertion.flat_moment(2, -1)
        with pytest.raises(ValueError):
            insertion.insertion_measure_support(-1)


class TestStatementFourCurvatureIsKurtosis:
    """The result: excess kurtosis is the Wilson plaquette action, in dimension two."""

    def test_flat_triangle_has_no_excess(self):
        assert insertion.excess_kurtosis(TRIANGLE, (0, 1, 2), FLAT) == pytest.approx(
            0.0, abs=1e-9
        )

    @pytest.mark.parametrize("seed", range(8))
    def test_identity_holds_under_random_curvature(self, seed: int):
        weight = insertion.phase_function(_random_angles(3, seed))
        assert insertion.kurtosis_identity_holds(TRIANGLE, (0, 1, 2), weight)
        assert insertion.kurtosis_identity_residual(
            TRIANGLE, (0, 1, 2), weight
        ) < 1e-12

    @pytest.mark.parametrize("seed", [21, 22, 23])
    def test_identity_survives_a_larger_ambient(self, seed: int):
        bigger = insertion.close_under_faces([(0, 1, 2), (0, 3), (1, 3), (2, 3)])
        weight = insertion.phase_function(_random_angles(4, seed))
        assert insertion.kurtosis_identity_holds(bigger, (0, 1, 2), weight)

    def test_excess_is_non_negative(self):
        """A two-point measure minimises the fourth moment at fixed variance."""
        for seed in range(6):
            weight = insertion.phase_function(_random_angles(3, seed))
            assert insertion.excess_kurtosis(TRIANGLE, (0, 1, 2), weight) >= -1e-9

    def test_right_hand_side_is_the_magnetic_curvature(self):
        """Two disjoint routes to the same number.

        `magnetic.curvature_norm` computes ``|d^2|`` from a matrix product;
        `holonomy_defect_squared` computes ``|h-1|^2`` from a product of edge
        weights; and the left-hand side is a fourth matrix moment.  All three
        must agree.
        """
        angles = {(0, 1): 0.3, (0, 2): 0.9, (1, 2): 0.4}
        weight = insertion.phase_function(angles)
        connection = magnetic.Connection((0.3, 0.9, 0.4))
        assert insertion.holonomy_defect_squared(
            (0, 1, 2), weight
        ) == pytest.approx(magnetic.curvature_norm(connection) ** 2, abs=1e-9)
        assert insertion.excess_kurtosis(
            TRIANGLE, (0, 1, 2), weight
        ) == pytest.approx(magnetic.curvature_norm(connection) ** 2, abs=1e-9)

    def test_curvature_smears_the_bernoulli_measure(self):
        """The interpretation, made checkable.

        Flat gives a two-point measure with zero excess; curvature spreads it,
        and the amount of spreading is exactly the squared holonomy defect.
        """
        flat_excess = insertion.excess_kurtosis(TRIANGLE, (0, 1, 2), FLAT)
        curved = insertion.phase_function({(0, 1): 0.0, (0, 2): 0.0, (1, 2): 2.0})
        curved_excess = insertion.excess_kurtosis(TRIANGLE, (0, 1, 2), curved)
        assert flat_excess == pytest.approx(0.0, abs=1e-9)
        assert curved_excess > 1.0

    def test_defect_rejects_wrong_dimension(self):
        with pytest.raises(ValueError, match="2-simplex"):
            insertion.holonomy_defect_squared((0, 1, 2, 3), FLAT)
        with pytest.raises(ValueError, match="2-simplex"):
            insertion.holonomy_defect_squared((0, 1), FLAT)


class TestWhereItStops:
    """The dimension-two boundary, recorded as a result."""

    def test_the_ceiling_was_retracted(self):
        """There is no dimension-two ceiling; the earlier report was wrong.

        The constant is kept so the retraction stays visible, but the predicate
        now returns ``False``: `fourth_moment_law` is exact in every dimension
        once the plaquettes are indexed by codimension-two faces and the sibling
        term is included.
        """
        assert insertion.KURTOSIS_IDENTITY_MAX_DIMENSION == 2
        assert insertion.identity_fails_above_dimension_two() is False
        assert "Retraction" in insertion.__doc__

    def test_the_wrong_index_set_really_does_fail(self):
        """Why the earlier attempt failed, preserved as a measurement.

        Summing over *triangular* faces is wrong; the plaquettes live on
        *codimension-two* faces.  For a tetrahedron that is six edges against
        four triangles, so the two index sets do not even have the same size.
        """
        weight = insertion.phase_function(_random_angles(4, 41))
        excess = insertion.excess_kurtosis(TETRAHEDRON, (0, 1, 2, 3), weight)
        triangular = sum(
            insertion.holonomy_defect_squared(triangle, weight)
            for triangle in combinations(range(4), 3)
        )
        correct = insertion.wilson_sum(TETRAHEDRON, (0, 1, 2, 3), weight)
        assert abs(excess - triangular) > 1e-3
        assert excess == pytest.approx(correct, abs=1e-9)
        assert len(insertion.hasse_squares(TETRAHEDRON, (0, 1, 2, 3), weight)) == 6

    def test_the_definition_is_not_claimed_as_new(self):
        """The object is the local density of states, and the claim was withdrawn.

        Savostianov, Guglielmi, Schaub and Tudisco (arXiv:2502.07558, Def 4.1)
        define exactly this measure, with Chebyshev moments that are already
        walk moments at a simplex.  An earlier version of this module proposed
        it as a new definition.  Asserted here so the withdrawal cannot erode.
        """
        assert "is **not** new" in insertion.__doc__
        assert "2502.07558" in insertion.__doc__
        assert "that claim has been withdrawn" in insertion.__doc__

    def test_the_right_hand_side_is_named_wilson_not_curvature(self):
        """"Squared curvature" was the wrong name and is not used.

        ``|h-1|^2`` and ``2(1 - Re h)`` are identical for unitary holonomy, so
        no approximation is involved -- but ``|F|^2`` is recovered only in the
        continuum small-flux limit, so the lattice name is the honest one.
        """
        assert "Wilson plaquette action" in insertion.__doc__
        assert "**Naming.**" in insertion.__doc__
        assert "small-flux limit" in insertion.__doc__

    def test_novelty_claim_is_marked_unverified(self):
        """arxiv.org is unreachable here; the claim rests on nothing checked."""
        assert "unverified" in insertion.__doc__
        assert "arxiv.org is unreachable" in insertion.__doc__


class TestStatementThreeGirthLaw:
    """Flux enters the local moments at order exactly ``2g``."""

    TREE = insertion.close_under_faces([(0, 1), (1, 2), (2, 3), (3, 4)])
    TRIANGLE_GRAPH = insertion.close_under_faces([(0, 1), (0, 2), (1, 2)])
    SQUARE_GRAPH = insertion.close_under_faces([(0, 1), (1, 2), (2, 3), (0, 3)])
    CELL = insertion.close_under_faces([(0, 1, 2)])

    @pytest.mark.parametrize(
        "girth, order", [(2, 4), (3, 6), (4, 8), (5, 10)]
    )
    def test_prediction(self, girth: int, order: int):
        assert insertion.first_flux_bearing_moment(girth) == order

    def test_rejects_girth_below_two(self):
        with pytest.raises(ValueError, match="at least two"):
            insertion.first_flux_bearing_moment(1)

    @pytest.mark.parametrize("order", [2, 4, 6, 8, 10])
    def test_a_tree_is_flux_blind_at_every_order(self, order: int):
        """Simply connected, so every connection is gauge-trivial.

        This is the local form of the Kesten-McKay fact: the spectral measure at
        the root of a tree cannot depend on phases, because there is no cycle to
        carry holonomy.
        """
        assert insertion.moment_is_flux_blind(self.TREE, (1, 2), order, seed=3)

    def test_two_simplex_first_sees_flux_at_four(self):
        assert insertion.moment_is_flux_blind(self.CELL, (0, 1, 2), 2, seed=4)
        assert not insertion.moment_is_flux_blind(self.CELL, (0, 1, 2), 4, seed=4)
        assert insertion.first_flux_bearing_moment(2) == 4

    def test_edge_in_girth_three_first_sees_flux_at_six(self):
        for order in (2, 4):
            assert insertion.moment_is_flux_blind(
                self.TRIANGLE_GRAPH, (0, 1), order, seed=5
            )
        assert not insertion.moment_is_flux_blind(
            self.TRIANGLE_GRAPH, (0, 1), 6, seed=5
        )
        assert insertion.first_flux_bearing_moment(3) == 6

    def test_edge_in_girth_four_first_sees_flux_at_eight(self):
        for order in (2, 4, 6):
            assert insertion.moment_is_flux_blind(
                self.SQUARE_GRAPH, (0, 1), order, seed=6
            )
        assert not insertion.moment_is_flux_blind(
            self.SQUARE_GRAPH, (0, 1), 8, seed=6
        )
        assert insertion.first_flux_bearing_moment(4) == 8

    def test_law_subsumes_the_second_moment_statement(self):
        """``g >= 2`` always, so ``M_2`` is flux-blind everywhere."""
        for complex_, simplex in [
            (self.TREE, (1, 2)),
            (self.TRIANGLE_GRAPH, (0, 1)),
            (self.SQUARE_GRAPH, (0, 1)),
            (self.CELL, (0, 1, 2)),
        ]:
            assert insertion.moment_is_flux_blind(complex_, simplex, 2, seed=7)

    def test_blindness_check_rejects_tiny_sample(self):
        with pytest.raises(ValueError, match="at least two"):
            insertion.moment_is_flux_blind(self.CELL, (0, 1, 2), 4, samples=1)


#: Module level, not a class attribute: `phase_function` returns a *function*,
#: and a function stored on a class becomes a bound method on attribute access,
#: so ``self.WEIGHT(u, v)`` would pass ``self`` as a third argument.  That is how
#: the first version of these tests failed.
WILSON_WEIGHT = insertion.phase_function({(0, 1): 0.3, (0, 2): 0.9, (1, 2): 0.4})


class TestWilsonNaming:
    """The right-hand side, named correctly and cross-checked."""

    def test_wilson_action_equals_excess_kurtosis(self):
        assert insertion.wilson_plaquette_action(
            (0, 1, 2), WILSON_WEIGHT
        ) == pytest.approx(
            insertion.excess_kurtosis(TRIANGLE, (0, 1, 2), WILSON_WEIGHT), abs=1e-12
        )

    def test_wilson_action_is_the_kenyon_weight(self):
        """``|h-1|^2 = 2 - tr(hol)`` for ``U(1)``.

        Kenyon writes the determinant of a connection Laplacian as a sum over
        cycle-rooted spanning forests with exactly this weight -- so the same
        quantity appears in a determinant identity and in this moment identity.
        """
        assert insertion.wilson_action_is_the_kenyon_weight((0, 1, 2), WILSON_WEIGHT)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_kenyon_identity_under_random_phases(self, seed: int):
        weight = insertion.phase_function(_random_angles(3, seed))
        assert insertion.wilson_action_is_the_kenyon_weight((0, 1, 2), weight)

    def test_both_forms_agree_exactly(self):
        """``|h-1|^2`` and ``2(1 - Re h)`` are the same for unitary holonomy."""
        for seed in range(4):
            weight = insertion.phase_function(_random_angles(3, seed))
            holonomy = (
                weight(0, 1) * weight(1, 2) * weight(2, 0)
            )
            assert insertion.wilson_plaquette_action(
                (0, 1, 2), weight
            ) == pytest.approx(2 * (1 - holonomy.real), abs=1e-12)

    def test_rejects_wrong_dimension(self):
        with pytest.raises(ValueError, match="2-simplex"):
            insertion.wilson_action_is_the_kenyon_weight((0, 1), WILSON_WEIGHT)


class TestFourthMomentLaw:
    """The general law: baseline, siblings, Wilson action."""

    CASES = [
        ([(0, 1, 2)], (0, 1, 2), 0, 3),
        ([(0, 1, 2), (0, 1, 3)], (0, 1, 2), 1, 3),
        ([(0, 1, 2), (0, 1, 3), (0, 1, 4)], (0, 1, 2), 2, 3),
        ([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], (0, 1, 2), 3, 3),
        ([(0, 1, 2, 3)], (0, 1, 2, 3), 0, 6),
        ([(0, 1, 2, 3), (0, 1, 2, 4)], (0, 1, 2, 3), 1, 6),
        ([(0, 1, 2, 3, 4)], (0, 1, 2, 3, 4), 0, 10),
    ]

    @pytest.mark.parametrize("faces, simplex, expected_siblings, plaquettes", CASES)
    def test_law_is_exact(self, faces, simplex, expected_siblings, plaquettes):
        complex_ = insertion.close_under_faces(faces)
        vertices = 1 + max(max(face) for face in faces)
        assert insertion.sibling_count(complex_, simplex) == expected_siblings
        for seed in range(4):
            weight = insertion.phase_function(_random_angles(vertices, seed))
            assert len(insertion.hasse_squares(complex_, simplex, weight)) == plaquettes
            assert insertion.fourth_moment_law_holds(complex_, simplex, weight)
            assert insertion.fourth_moment_law_residual(
                complex_, simplex, weight
            ) < 1e-12

    def test_plaquettes_are_indexed_by_codimension_two_faces(self):
        """Six for a tetrahedron, not four -- edges, not triangles."""
        for dimension in (2, 3, 4, 5):
            simplex = tuple(range(dimension + 1))
            complex_ = insertion.close_under_faces([simplex])
            squares = insertion.hasse_squares(complex_, simplex, FLAT)
            assert len(squares) == len(list(combinations(simplex, dimension - 1)))

    def test_flat_plaquettes_are_all_one(self):
        """The normalising sign is exactly the ``d^2 = 0`` cancellation."""
        for dimension in (2, 3, 4):
            simplex = tuple(range(dimension + 1))
            complex_ = insertion.close_under_faces([simplex])
            for value in insertion.hasse_squares(complex_, simplex, FLAT).values():
                assert value == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("dimension", [2, 3])
    def test_plaquette_holonomies_are_gauge_invariant(self, dimension: int):
        """A vertex gauge transformation cannot move a closed-loop holonomy."""
        simplex = tuple(range(dimension + 1))
        complex_ = insertion.close_under_faces([simplex])
        rng = np.random.default_rng(dimension)
        angles = _random_angles(dimension + 1, 5)
        shift = {v: float(rng.uniform(0, 2 * np.pi)) for v in simplex}
        gauged = {
            edge: value + shift[edge[1]] - shift[edge[0]]
            for edge, value in angles.items()
        }
        before = insertion.hasse_squares(
            complex_, simplex, insertion.phase_function(angles)
        )
        after = insertion.hasse_squares(
            complex_, simplex, insertion.phase_function(gauged)
        )
        for face in before:
            assert before[face] == pytest.approx(after[face], abs=1e-12)

    def test_sibling_term_is_flux_blind(self):
        """It is a count, so no connection can change it."""
        complex_ = insertion.close_under_faces([(0, 1, 2), (0, 1, 3)])
        assert insertion.sibling_count(complex_, (0, 1, 2)) == 1
        assert insertion.sibling_count(complex_, (0, 1, 3)) == 1

    def test_wilson_sum_vanishes_at_zero_flux(self):
        for faces, simplex, _, _ in self.CASES:
            complex_ = insertion.close_under_faces(faces)
            assert insertion.wilson_sum(complex_, simplex, FLAT) == pytest.approx(
                0.0, abs=1e-12
            )

    def test_two_simplex_case_reduces_to_the_holonomy(self):
        """One plaquette carries the flux and the other two are trivial."""
        weight = insertion.phase_function(_random_angles(3, 2))
        assert insertion.wilson_sum(TRIANGLE, (0, 1, 2), weight) == pytest.approx(
            insertion.holonomy_defect_squared((0, 1, 2), weight), abs=1e-12
        )

    def test_rejects_low_dimension(self):
        with pytest.raises(ValueError, match="dimension at least two"):
            insertion.hasse_squares(TRIANGLE, (0, 1), FLAT)

    def test_rejects_missing_simplex(self):
        with pytest.raises(ValueError, match="not in the complex"):
            insertion.sibling_count(TRIANGLE, (7, 8, 9))
