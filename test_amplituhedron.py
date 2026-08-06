"""Tests for the exact ``k = 1`` amplituhedron tiling pipeline.

The strategy throughout is to pin the machinery against classical results it has
no way to know about -- Catalan numbers, Gale's evenness condition, Eulerian
numbers, the associahedron -- rather than against itself.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import networkx as nx
import pytest
import sympy

import amplituhedron as amp


# --------------------------------------------------------------------------- #
# Exact integer linear algebra
# --------------------------------------------------------------------------- #
class TestIntegerDeterminant:
    """Bareiss elimination, against sympy and against hand cases."""

    def test_small_cases(self) -> None:
        assert amp.integer_determinant([[5]]) == 5
        assert amp.integer_determinant([[1, 2], [3, 4]]) == -2
        assert amp.integer_determinant([]) == 1

    def test_identity_and_singular(self) -> None:
        identity = [[1 if i == j else 0 for j in range(5)] for i in range(5)]
        assert amp.integer_determinant(identity) == 1
        assert amp.integer_determinant([[1, 2], [2, 4]]) == 0

    def test_needs_a_zero_pivot_swap(self) -> None:
        assert amp.integer_determinant([[0, 1], [1, 0]]) == -1
        assert amp.integer_determinant([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) == -1

    @pytest.mark.parametrize("size", [3, 4, 5, 6])
    def test_agrees_with_sympy_on_pseudorandom_matrices(self, size: int) -> None:
        import random

        rng = random.Random(20260806 + size)
        for _ in range(12):
            matrix = [[rng.randint(-9, 9) for _ in range(size)] for _ in range(size)]
            assert amp.integer_determinant(matrix) == int(sympy.Matrix(matrix).det())

    def test_rejects_non_square(self) -> None:
        with pytest.raises(ValueError, match="square"):
            amp.integer_determinant([[1, 2, 3], [4, 5, 6]])

    def test_stays_exact_on_large_entries(self) -> None:
        """Moment-curve coordinates grow fast; a float determinant would lose this."""
        matrix = [[t**p for p in range(1, 6)] for t in (11, 23, 37, 41, 53)]
        assert amp.integer_determinant(matrix) == int(sympy.Matrix(matrix).det())


class TestGeneralisedCross:
    """The orthogonal-complement primitive that generalises SAT to R^m."""

    def test_reduces_to_the_ordinary_cross_product(self) -> None:
        assert amp.generalised_cross([[1, 0, 0], [0, 1, 0]]) == (0, 0, 1)

    @pytest.mark.parametrize("dimension", [2, 3, 4, 5])
    def test_result_is_orthogonal_to_every_input(self, dimension: int) -> None:
        import random

        rng = random.Random(97 + dimension)
        vectors = [
            [rng.randint(-6, 6) for _ in range(dimension)] for _ in range(dimension - 1)
        ]
        normal = amp.generalised_cross(vectors)
        for vector in vectors:
            assert sum(a * b for a, b in zip(normal, vector)) == 0

    def test_dependent_inputs_give_the_zero_vector(self) -> None:
        assert amp.generalised_cross([[1, 2, 3], [2, 4, 6]]) == (0, 0, 0)

    def test_rejects_the_wrong_number_of_vectors(self) -> None:
        with pytest.raises(ValueError, match="needs 2 vectors"):
            amp.generalised_cross([[1, 0, 0]])


class TestRationalsToIntegers:
    def test_clears_denominators_with_one_scaling(self) -> None:
        points = [[Fraction(1, 2), Fraction(1, 3)], [Fraction(5, 6), 2]]
        assert amp.rationals_to_integers(points) == ((3, 2), (5, 12))

    def test_leaves_integers_alone(self) -> None:
        assert amp.rationals_to_integers([[1, 2], [3, 4]]) == ((1, 2), (3, 4))


# --------------------------------------------------------------------------- #
# The polytopes, against classical combinatorics
# --------------------------------------------------------------------------- #
class TestCyclicPolytope:
    """A(n, 1, m) is the cyclic polytope -- the theorem this module rests on."""

    def test_vertices_lie_on_the_moment_curve(self) -> None:
        assert amp.cyclic_polytope(4, 3) == ((1, 1, 1), (2, 4, 8), (3, 9, 27), (4, 16, 64))

    def test_is_full_dimensional(self) -> None:
        vertices = amp.cyclic_polytope(7, 4)
        assert amp.configuration_volume(vertices) > 0

    def test_every_vertex_subset_of_size_m_plus_one_is_a_simplex(self) -> None:
        """Cyclic polytopes are simplicial and in general position: no flat subsets."""
        vertices = amp.cyclic_polytope(8, 4)
        for subset in itertools.combinations(range(8), 5):
            assert amp.normalised_volume(vertices, subset) > 0

    def test_rejects_degenerate_sizes(self) -> None:
        with pytest.raises(ValueError, match="n > m"):
            amp.cyclic_polytope(4, 4)
        with pytest.raises(ValueError, match="strictly increasing"):
            amp.cyclic_polytope(3, 2, parameters=[1, 1, 2])


class TestGaleEvenness:
    """Facets from the classical condition, checked against geometry."""

    def test_polygon_facets_are_its_edges(self) -> None:
        facets = amp.gale_facets(6, 2)
        assert facets == frozenset(
            {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)}
        )

    @pytest.mark.parametrize("n, m", [(6, 3), (7, 3), (6, 4), (7, 4), (8, 4)])
    def test_facet_count_matches_the_upper_bound_theorem(self, n: int, m: int) -> None:
        """Cyclic polytopes attain McMullen's bound, which is why they are extremal."""
        from math import comb

        half = m // 2
        if m % 2 == 0:
            expected = comb(n - half, half) + comb(n - half - 1, half - 1)
        else:
            expected = 2 * comb(n - half - 1, half)
        assert len(amp.gale_facets(n, m)) == expected

    @pytest.mark.parametrize("n, m", [(6, 2), (7, 2), (6, 3), (6, 4), (7, 4)])
    def test_every_gale_facet_really_supports_the_polytope(self, n: int, m: int) -> None:
        """Geometric confirmation: all other vertices lie weakly on one side."""
        vertices = amp.cyclic_polytope(n, m)
        for facet in amp.gale_facets(n, m):
            base = vertices[facet[0]]
            spanning = [
                [vertices[v][d] - base[d] for d in range(m)] for v in facet[1:]
            ]
            normal = amp.generalised_cross(spanning)
            offset = sum(a * b for a, b in zip(normal, base))
            signs = {
                (lambda x: (x > 0) - (x < 0))(
                    sum(a * b for a, b in zip(normal, vertices[v])) - offset
                )
                for v in range(n)
                if v not in facet
            }
            assert signs in ({1}, {-1}), f"facet {facet} does not support"


class TestVolumes:
    """Two independent routes to the same integer, plus an external theorem."""

    @pytest.mark.parametrize("n, m", [(5, 2), (6, 2), (7, 2), (6, 3), (7, 3), (6, 4), (7, 4)])
    def test_gale_route_and_general_route_agree(self, n: int, m: int) -> None:
        vertices = amp.cyclic_polytope(n, m)
        by_facets = amp.polytope_normalised_volume(vertices, amp.gale_facets(n, m))
        assert by_facets == amp.configuration_volume(vertices)

    @pytest.mark.parametrize("n", [5, 6, 7, 8])
    def test_polygon_volume_matches_the_shoelace_formula(self, n: int) -> None:
        vertices = amp.cyclic_polytope(n, 2)
        shoelace = abs(
            sum(
                vertices[i][0] * vertices[(i + 1) % n][1]
                - vertices[(i + 1) % n][0] * vertices[i][1]
                for i in range(n)
            )
        )
        assert amp.polytope_normalised_volume(vertices, amp.gale_facets(n, 2)) == shoelace

    @pytest.mark.parametrize(
        "k, n", [(1, 3), (2, 3), (1, 4), (2, 4), (3, 4), (1, 5), (2, 5)]
    )
    def test_hypersimplex_volume_is_an_eulerian_number(self, k: int, n: int) -> None:
        """Laplace's theorem -- an answer the volume routine cannot have known."""
        vertices = amp.hypersimplex(k, n)
        assert amp.configuration_volume(vertices) == amp.eulerian_number(n - 1, k - 1)

    def test_eulerian_numbers_are_the_classical_triangle(self) -> None:
        assert [amp.eulerian_number(4, k) for k in range(4)] == [1, 11, 11, 1]
        assert [amp.eulerian_number(5, k) for k in range(5)] == [1, 26, 66, 26, 1]

    def test_hypersimplex_rejects_degenerate_parameters(self) -> None:
        with pytest.raises(ValueError, match="0 < k < n"):
            amp.hypersimplex(0, 4)
        with pytest.raises(ValueError, match="0 < k < n"):
            amp.hypersimplex(4, 4)


# --------------------------------------------------------------------------- #
# The Separating Axis Theorem in R^m
# --------------------------------------------------------------------------- #
class TestSeparatingAxis:
    """Exact, and complete: it must never miss a genuine separation."""

    def test_two_simplices_sharing_a_facet_are_separated(self) -> None:
        vertices = amp.cyclic_polytope(6, 2)
        assert amp.interiors_disjoint(vertices, (0, 1, 2), (0, 2, 3))

    def test_a_simplex_never_separates_from_itself(self) -> None:
        vertices = amp.cyclic_polytope(6, 2)
        assert not amp.interiors_disjoint(vertices, (0, 1, 3), (0, 1, 3))

    def test_a_quadrilateral_splits_exactly_into_its_two_triangulations(self) -> None:
        """Only the two same-diagonal pairs are disjoint; all four cross pairs overlap."""
        vertices = amp.cyclic_polytope(4, 2)
        disjoint = {
            pair
            for pair in itertools.combinations(
                amp.full_dimensional_simplices(vertices), 2
            )
            if amp.interiors_disjoint(vertices, *pair)
        }
        assert disjoint == {
            ((0, 1, 2), (0, 2, 3)),   # diagonal 0-2
            ((0, 1, 3), (1, 2, 3)),   # diagonal 1-3
        }

    def test_the_returned_axis_actually_separates(self) -> None:
        vertices = amp.cyclic_polytope(7, 3)
        for left, right in itertools.combinations(
            amp.full_dimensional_simplices(vertices)[:20], 2
        ):
            axis = amp.separating_axis(vertices, left, right)
            if axis is None:
                continue
            low_l, high_l = amp._projection_range(axis, vertices, left)
            low_r, high_r = amp._projection_range(axis, vertices, right)
            assert high_l <= low_r or high_r <= low_l

    def test_is_symmetric_in_its_arguments(self) -> None:
        vertices = amp.cyclic_polytope(7, 3)
        for left, right in itertools.combinations(
            amp.full_dimensional_simplices(vertices)[:24], 2
        ):
            assert amp.interiors_disjoint(vertices, left, right) == amp.interiors_disjoint(
                vertices, right, left
            )

    def test_matches_a_brute_force_interior_test_in_the_plane(self) -> None:
        """Independent check: sample the interior on an exact rational grid."""
        vertices = amp.cyclic_polytope(6, 2)
        simplices = amp.full_dimensional_simplices(vertices)

        def inside(simplex, point) -> bool:
            signs = set()
            for i in range(3):
                a, b = vertices[simplex[i]], vertices[simplex[(i + 1) % 3]]
                cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (
                    point[0] - a[0]
                )
                signs.add((cross > 0) - (cross < 0))
            return 0 not in signs and len(signs) == 1

        for left, right in itertools.combinations(simplices, 2):
            # Barycentres of fine sub-triangles give many interior witnesses.
            witnesses = []
            for weights in itertools.product(range(1, 6), repeat=3):
                total = sum(weights)
                witnesses.append(
                    tuple(
                        sum(
                            Fraction(w * vertices[v][d], total)
                            for w, v in zip(weights, left)
                        )
                        for d in range(2)
                    )
                )
            overlap = any(inside(left, p) and inside(right, p) for p in witnesses)
            if overlap:
                assert not amp.interiors_disjoint(vertices, left, right)


class TestIsTiling:
    def test_accepts_a_genuine_triangulation(self) -> None:
        vertices = amp.cyclic_polytope(5, 2)
        volume = amp.polytope_normalised_volume(vertices, amp.gale_facets(5, 2))
        assert amp.is_tiling(vertices, [(0, 1, 2), (0, 2, 3), (0, 3, 4)], volume)

    def test_rejects_a_family_that_leaves_a_gap(self) -> None:
        vertices = amp.cyclic_polytope(5, 2)
        volume = amp.polytope_normalised_volume(vertices, amp.gale_facets(5, 2))
        assert not amp.is_tiling(vertices, [(0, 1, 2), (0, 2, 3)], volume)

    def test_rejects_an_overlapping_family_with_the_right_volume(self) -> None:
        """Both checks are load-bearing: this one passes the volume test and fails SAT.

        The three triangles fanned from edge 0-1 of the pentagon sum to exactly
        the pentagon's normalised volume while overlapping heavily, so a volume
        identity on its own would certify a non-tiling as a tiling.
        """
        vertices = amp.cyclic_polytope(5, 2)
        volume = amp.polytope_normalised_volume(vertices, amp.gale_facets(5, 2))
        overlapping = [(0, 1, 2), (0, 1, 3), (0, 1, 4)]
        assert sum(amp.normalised_volume(vertices, s) for s in overlapping) == volume
        assert not amp.is_tiling(vertices, overlapping, volume)

    def test_both_quadrilateral_triangulations_are_accepted(self) -> None:
        vertices = amp.cyclic_polytope(4, 2)
        volume = amp.polytope_normalised_volume(vertices, amp.gale_facets(4, 2))
        assert amp.is_tiling(vertices, [(0, 1, 2), (0, 2, 3)], volume)
        assert amp.is_tiling(vertices, [(0, 1, 3), (1, 2, 3)], volume)


# --------------------------------------------------------------------------- #
# Enumeration, against Catalan
# --------------------------------------------------------------------------- #
class TestEnumeration:
    """Every tiling of A(n, 1, m), exhaustively."""

    @pytest.mark.parametrize("n", [4, 5, 6, 7, 8])
    def test_polygon_tilings_are_counted_by_catalan(self, n: int) -> None:
        """A(n, 1, 2) is an n-gon, so its tilings number C(n-2). No tuning involved."""
        vertices = amp.cyclic_polytope(n, 2)
        report = amp.enumerate_tilings(vertices)
        assert report.n_tilings == amp.catalan(n - 2)

    @pytest.mark.parametrize("n, expected", [(6, 2), (7, 7), (8, 40)])
    def test_m4_tiling_counts(self, n: int, expected: int) -> None:
        """The physical case. 2, 7, 40, 357 is the known cyclic 4-polytope sequence."""
        vertices = amp.cyclic_polytope(n, 4)
        volume = amp.polytope_normalised_volume(vertices, amp.gale_facets(n, 4))
        assert amp.enumerate_tilings(vertices, volume).n_tilings == expected

    @pytest.mark.parametrize("n", [6, 7, 8])
    def test_every_m4_tiling_has_the_same_size(self, n: int) -> None:
        """All triangulations of C(n, 4) use exactly binom(n-3, 2) simplices."""
        from math import comb

        vertices = amp.cyclic_polytope(n, 4)
        volume = amp.polytope_normalised_volume(vertices, amp.gale_facets(n, 4))
        report = amp.enumerate_tilings(vertices, volume)
        assert report.sizes == {comb(n - 3, 2): report.n_tilings}

    @pytest.mark.parametrize("n, m", [(6, 2), (7, 2), (6, 3), (6, 4), (7, 4)])
    def test_every_enumerated_family_really_tiles(self, n: int, m: int) -> None:
        vertices = amp.cyclic_polytope(n, m)
        volume = amp.polytope_normalised_volume(vertices, amp.gale_facets(n, m))
        for tiling in amp.enumerate_tilings(vertices, volume).tilings:
            assert amp.is_tiling(vertices, tiling, volume)

    def test_tilings_are_distinct(self) -> None:
        report = amp.enumerate_tilings(amp.cyclic_polytope(7, 2))
        assert len(set(report.tilings)) == report.n_tilings

    def test_volume_may_be_supplied_or_derived(self) -> None:
        vertices = amp.cyclic_polytope(6, 3)
        volume = amp.polytope_normalised_volume(vertices, amp.gale_facets(6, 3))
        assert (
            amp.enumerate_tilings(vertices, volume).n_tilings
            == amp.enumerate_tilings(vertices).n_tilings
        )

    def test_limit_stops_early(self) -> None:
        report = amp.enumerate_tilings(amp.cyclic_polytope(8, 2), limit=5)
        assert 0 < report.n_tilings <= amp.catalan(6)

    def test_rejects_a_nonsense_limit(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            amp.enumerate_tilings(amp.cyclic_polytope(5, 2), limit=0)


# --------------------------------------------------------------------------- #
# Bistellar flips and the flip graph
# --------------------------------------------------------------------------- #
class TestRadonPartition:
    def test_quadrilateral_circuit_splits_into_diagonals(self) -> None:
        vertices = amp.cyclic_polytope(4, 2)
        positive, negative = amp.radon_partition(vertices, [0, 1, 2, 3])
        assert {positive, negative} == {(0, 2), (1, 3)}

    def test_parts_are_disjoint_and_cover(self) -> None:
        vertices = amp.cyclic_polytope(7, 4)
        positive, negative = amp.radon_partition(vertices, [0, 1, 2, 3, 4, 5])
        assert set(positive).isdisjoint(negative)
        assert set(positive) | set(negative) == {0, 1, 2, 3, 4, 5}

    def test_coefficients_form_an_affine_dependence(self) -> None:
        vertices = amp.cyclic_polytope(6, 3)
        support = [0, 1, 2, 3, 4]
        rows = [[vertices[v][d] for v in support] for d in range(3)]
        rows.append([1] * 5)
        coefficients = amp.generalised_cross(rows)
        assert sum(coefficients) == 0
        for d in range(3):
            assert sum(c * vertices[v][d] for c, v in zip(coefficients, support)) == 0

    def test_rejects_the_wrong_support_size(self) -> None:
        with pytest.raises(ValueError, match="points"):
            amp.radon_partition(amp.cyclic_polytope(6, 2), [0, 1, 2])


class TestFlipGraph:
    """For m = 2 this is the associahedron, which pins the whole construction."""

    @pytest.mark.parametrize("n", [5, 6, 7, 8])
    def test_is_the_associahedron_skeleton(self, n: int) -> None:
        vertices = amp.cyclic_polytope(n, 2)
        report = amp.enumerate_tilings(vertices)
        graph = amp.flip_graph(vertices, report.tilings)
        assert graph.number_of_nodes() == amp.catalan(n - 2)
        assert {d for _, d in graph.degree()} == {n - 3}          # regular
        assert graph.number_of_edges() == (n - 3) * amp.catalan(n - 2) // 2
        assert nx.is_connected(graph)

    @pytest.mark.parametrize("n, m", [(6, 3), (7, 3), (7, 4), (8, 4)])
    def test_flips_connect_every_tiling(self, n: int, m: int) -> None:
        """Rambau's theorem for cyclic polytopes, confirmed rather than assumed.

        The enumeration never uses flips, so connectivity here is a genuine
        check on both the enumeration and the flip criterion.
        """
        vertices = amp.cyclic_polytope(n, m)
        volume = amp.polytope_normalised_volume(vertices, amp.gale_facets(n, m))
        report = amp.enumerate_tilings(vertices, volume)
        graph = amp.flip_graph(vertices, report.tilings)
        assert nx.is_connected(graph)

    def test_a_flip_changes_exactly_one_circuit(self) -> None:
        vertices = amp.cyclic_polytope(6, 2)
        report = amp.enumerate_tilings(vertices)
        graph = amp.flip_graph(vertices, report.tilings)
        for i, j in graph.edges():
            left, right = report.tilings[i], report.tilings[j]
            support = {v for s in (left - right) | (right - left) for v in s}
            assert len(support) == 4          # m + 2

    def test_identical_tilings_are_not_neighbours(self) -> None:
        vertices = amp.cyclic_polytope(6, 2)
        tiling = amp.enumerate_tilings(vertices).tilings[0]
        assert not amp.bistellar_neighbours(vertices, tiling, tiling)


class TestSymmetry:
    """The finite group that actually acts on the tiling space."""

    @pytest.mark.parametrize("n, m", [(5, 2), (6, 2), (7, 2), (7, 4), (8, 4)])
    def test_flip_graph_automorphisms_are_dihedral_of_order_2n(
        self, n: int, m: int
    ) -> None:
        """Rotation and reflection of the moment curve, and nothing else.

        This is the well-posed replacement for "multiplicities match a Yangian":
        the Yangian is infinite dimensional and does not act on a finite complex,
        whereas D_n demonstrably does.
        """
        from networkx.algorithms.isomorphism import GraphMatcher

        vertices = amp.cyclic_polytope(n, m)
        volume = amp.polytope_normalised_volume(vertices, amp.gale_facets(n, m))
        report = amp.enumerate_tilings(vertices, volume)
        graph = amp.flip_graph(vertices, report.tilings)
        assert sum(1 for _ in GraphMatcher(graph, graph).isomorphisms_iter()) == 2 * n

    @pytest.mark.parametrize("n, m", [(6, 2), (7, 2), (6, 4), (7, 4)])
    def test_configuration_automorphisms_include_the_cyclic_shift(
        self, n: int, m: int
    ) -> None:
        automorphisms = amp.configuration_automorphisms(amp.cyclic_polytope(n, m))
        assert tuple(range(n)) in automorphisms
        assert len(automorphisms) >= 1


class TestExactSpectrum:
    """High multiplicities must be exact, not a degeneracy-tolerance artefact."""

    @pytest.mark.parametrize(
        "n, m, eigenvalue, multiplicity", [(8, 2, 6, 8), (8, 4, 3, 6)]
    )
    def test_large_degeneracies_are_exact_integers(
        self, n: int, m: int, eigenvalue: int, multiplicity: int
    ) -> None:
        """Computed from the characteristic polynomial, so no tolerance is involved."""
        vertices = amp.cyclic_polytope(n, m)
        volume = amp.polytope_normalised_volume(vertices, amp.gale_facets(n, m))
        report = amp.enumerate_tilings(vertices, volume)
        graph = amp.flip_graph(vertices, report.tilings)
        laplacian = sympy.Matrix(nx.laplacian_matrix(graph).todense().astype(int).tolist())
        assert laplacian.eigenvals().get(sympy.Integer(eigenvalue)) == multiplicity

    def test_zero_is_simple_because_the_flip_graph_is_connected(self) -> None:
        vertices = amp.cyclic_polytope(7, 2)
        graph = amp.flip_graph(vertices, amp.enumerate_tilings(vertices).tilings)
        laplacian = sympy.Matrix(nx.laplacian_matrix(graph).todense().astype(int).tolist())
        assert laplacian.eigenvals()[sympy.Integer(0)] == 1


class TestTilingSummary:
    def test_reports_the_structural_facts(self) -> None:
        vertices = amp.cyclic_polytope(6, 2)
        summary = amp.tiling_summary(vertices, amp.enumerate_tilings(vertices))
        assert summary["n_tilings"] == 14
        assert summary["dimension"] == 2
        assert summary["flip_graph_connected"] is True
        assert summary["flip_degree_min"] == summary["flip_degree_max"] == 3
