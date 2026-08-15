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
    """The result: excess kurtosis equals squared curvature, in dimension two."""

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

    def test_boundary_is_two(self):
        assert insertion.KURTOSIS_IDENTITY_MAX_DIMENSION == 2
        assert insertion.identity_fails_above_dimension_two() is True

    def test_no_higher_dimensional_identity_is_offered(self):
        """The natural generalisation was tried and does not work.

        Regressing the excess kurtosis against the summed squared holonomy
        defects of a simplex's triangular faces gives residuals comparable to
        the signal at ``k = 3`` and ``k = 4``, and the natural ``2/(k+1)``
        coefficient is wrong by a factor of two or more.  Above dimension two
        the triangular faces share edges, so their holonomies interfere rather
        than add.  No function here claims otherwise.
        """
        assert not hasattr(insertion, "higher_kurtosis_identity")
        assert not hasattr(insertion, "curvature_sum_identity")

    def test_tetrahedron_excess_is_not_a_face_sum(self):
        """Measured, so the negative result is not merely asserted."""
        weight = insertion.phase_function(_random_angles(4, 41))
        excess = insertion.excess_kurtosis(TETRAHEDRON, (0, 1, 2, 3), weight)
        face_sum = sum(
            insertion.holonomy_defect_squared(triangle, weight)
            for triangle in combinations(range(4), 3)
        )
        assert excess > 0
        assert abs(excess - face_sum) > 1e-3

    def test_novelty_claim_is_marked_unverified(self):
        """arxiv.org is unreachable here; the claim rests on nothing checked."""
        assert "unverified" in insertion.__doc__
        assert "arxiv.org is unreachable" in insertion.__doc__
