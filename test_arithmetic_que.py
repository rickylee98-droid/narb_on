"""Tests for the arithmetic QUE module.

The LPS construction is pinned against exact theorems it cannot fake: Jacobi's
four-square count, the orders of PSL(2, F_q) and PGL(2, F_q), and the Ramanujan
bound.  The obstruction results are *measured*, not asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

import arithmetic_que as aq


@pytest.fixture(scope="module")
def small_lps() -> aq.LPSGraph:
    return aq.build_lps_graph(5, 13)


@pytest.fixture(scope="module")
def small_spectrum(small_lps):
    matrix = np.asarray(small_lps.adjacency.todense(), dtype=float)
    return np.linalg.eigh(matrix)


# --------------------------------------------------------------------------- #
# Exact arithmetic ingredients
# --------------------------------------------------------------------------- #
class TestFourSquares:
    @pytest.mark.parametrize("p", [5, 13, 17, 29, 37, 41])
    def test_count_is_exactly_p_plus_one(self, p: int) -> None:
        """Jacobi's theorem, after the sign-and-order normalisation."""
        assert len(aq.four_square_solutions(p)) == p + 1

    @pytest.mark.parametrize("p", [5, 13, 17, 29])
    def test_every_solution_sums_to_p_with_the_right_parities(self, p: int) -> None:
        for a0, a1, a2, a3 in aq.four_square_solutions(p):
            assert a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 == p
            assert a0 > 0 and a0 % 2 == 1
            assert a1 % 2 == 0 and a2 % 2 == 0 and a3 % 2 == 0

    def test_p_equals_five_is_covered(self) -> None:
        """Regression: a parity slip in the loop bounds gave zero solutions here.

        ``range(-r, r+1, 2)`` with odd ``r`` enumerates odd values, so the even
        coordinates were never visited.  It looked healthy for p = 13 and p = 29,
        where ``r`` happens to be even, and silently failed for p = 5 and p = 17.
        """
        solutions = aq.four_square_solutions(5)
        assert len(solutions) == 6
        assert (1, -2, 0, 0) in solutions

    def test_rejects_even_input(self) -> None:
        with pytest.raises(ValueError, match="odd prime"):
            aq.four_square_solutions(4)


class TestModularArithmetic:
    @pytest.mark.parametrize(
        "a, q, expected", [(5, 13, -1), (17, 13, 1), (29, 13, 1), (13, 17, 1), (5, 17, -1)]
    )
    def test_legendre_symbol(self, a: int, q: int, expected: int) -> None:
        assert aq.legendre_symbol(a, q) == expected

    @pytest.mark.parametrize("q", [5, 13, 17, 29, 37])
    def test_sqrt_minus_one_squares_to_minus_one(self, q: int) -> None:
        root = aq.modular_sqrt_minus_one(q)
        assert (root * root + 1) % q == 0

    def test_sqrt_minus_one_needs_q_congruent_to_one(self) -> None:
        with pytest.raises(ValueError, match="1 mod 4"):
            aq.modular_sqrt_minus_one(7)

    @pytest.mark.parametrize("p, q", [(5, 13), (13, 17), (5, 29)])
    def test_generators_have_determinant_p(self, p: int, q: int) -> None:
        """The determinant is what decides PSL versus PGL, so it is checked."""
        for a, b, c, d in aq.lps_generators(p, q):
            assert (a * d - b * c) % q == p % q


# --------------------------------------------------------------------------- #
# The graph, against exact group orders and the Ramanujan bound
# --------------------------------------------------------------------------- #
class TestLPSGraph:
    @pytest.mark.parametrize(
        "p, q, projective", [(5, 13, "PGL"), (5, 17, "PGL"), (13, 17, "PSL")]
    )
    def test_vertex_count_is_the_group_order(self, p: int, q: int, projective: str) -> None:
        graph = aq.build_lps_graph(p, q)
        assert graph.projective == projective
        assert graph.n_vertices == graph.expected_order

    def test_projective_normalisation_regression(self) -> None:
        """Quotienting by only +-1 doubles the graph and breaks Ramanujan.

        That was the original bug: X^{5,13} came out with 4368 vertices instead
        of 2184, and its largest non-trivial eigenvalue was 5.677 against a bound
        of 4.472.  The vertex-count check is what catches it.
        """
        graph = aq.build_lps_graph(5, 13)
        assert graph.n_vertices == 2184
        assert graph.n_vertices != 4368

    @pytest.mark.parametrize("p, q", [(5, 13), (13, 17)])
    def test_is_regular_of_degree_p_plus_one(self, p: int, q: int) -> None:
        graph = aq.build_lps_graph(p, q)
        degrees = np.asarray(graph.adjacency.sum(axis=1)).ravel()
        assert set(degrees.astype(int)) == {p + 1}

    @pytest.mark.parametrize("p, q", [(5, 13), (5, 17), (13, 17)])
    def test_satisfies_the_ramanujan_bound(self, p: int, q: int) -> None:
        """The theorem the construction cannot fake."""
        graph = aq.build_lps_graph(p, q)
        eigenvalues = np.linalg.eigvalsh(np.asarray(graph.adjacency.todense()))
        largest, ok = aq.ramanujan_bound(graph, eigenvalues)
        assert ok, f"largest non-trivial |lambda| = {largest} > {graph.ramanujan_bound}"
        assert largest > 0.5 * graph.ramanujan_bound  # and it is genuinely tight

    def test_rejects_equal_primes(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            aq.build_lps_graph(13, 13)


class TestKestenMcKay:
    def test_density_integrates_to_one(self) -> None:
        degree = 6
        support = 2.0 * np.sqrt(degree - 1)
        grid = np.linspace(-support + 1e-9, support - 1e-9, 200001)
        density = aq.kesten_mckay_density(grid, degree)
        assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=1e-3)

    def test_vanishes_outside_the_support(self) -> None:
        degree = 6
        outside = np.array([-6.0, 5.0, 10.0])
        assert np.all(aq.kesten_mckay_density(outside, degree) == 0.0)

    def test_rejects_degree_below_three(self) -> None:
        with pytest.raises(ValueError, match="degree at least 3"):
            aq.kesten_mckay_density(np.array([0.0]), 2)


# --------------------------------------------------------------------------- #
# The obstruction, measured rather than asserted
# --------------------------------------------------------------------------- #
class TestTransitivityObstruction:
    """Why the basis-free observable carries no information on a Cayley graph."""

    def test_eigenspaces_are_massively_degenerate(self, small_spectrum) -> None:
        eigenvalues, _ = small_spectrum
        _, counts = np.unique(np.round(eigenvalues, 7), return_counts=True)
        assert counts.max() >= 50, "expected large Cayley-graph degeneracies"

    def test_projector_diagonal_is_constant(self, small_spectrum) -> None:
        """Vertex-transitivity forces it, so the observable is identically 1.

        This is the measurement that invalidates the naive experiment: the
        spectral projector commutes with every automorphism, and the graph is
        vertex-transitive, so its diagonal cannot vary.
        """
        eigenvalues, eigenvectors = small_spectrum
        values, counts = np.unique(np.round(eigenvalues, 7), return_counts=True)
        degenerate = [v for v, c in zip(values, counts) if c > 1][:5]
        assert degenerate
        for target in degenerate:
            spread = aq.projector_diagonal_is_constant(
                eigenvalues, eigenvectors, float(target)
            )
            assert spread < 1e-9, f"lambda={target}: spread {spread}"

    def test_mass_averages_to_one(self, small_spectrum) -> None:
        eigenvalues, eigenvectors = small_spectrum
        values, counts = np.unique(np.round(eigenvalues, 7), return_counts=True)
        target = float(next(v for v, c in zip(values, counts) if c > 1))
        mass = aq.eigenspace_mass(eigenvalues, eigenvectors, target, atol=1e-6)
        assert np.mean(mass) == pytest.approx(1.0, abs=1e-9)

    def test_rejects_an_absent_eigenvalue(self, small_spectrum) -> None:
        eigenvalues, eigenvectors = small_spectrum
        with pytest.raises(ValueError, match="no eigenvalue within"):
            aq.eigenspace_mass(eigenvalues, eigenvectors, 123.456)


@pytest.fixture(scope="module")
def operators():
    return aq.hecke_operators(13, [17, 29, 53])


class TestHeckeObstruction:
    """The Hecke operators are real, and they still cannot separate eigenvectors."""

    def test_they_commute_exactly(self, operators) -> None:
        """Integer arithmetic, so this is exact rather than approximate."""
        ops, _ = operators
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                commutator = ops[i] @ ops[j] - ops[j] @ ops[i]
                assert abs(commutator).max() == 0.0

    def test_they_share_one_vertex_set(self, operators) -> None:
        ops, size = operators
        assert size == 13 * (13 * 13 - 1) // 2
        assert all(op.shape == (size, size) for op in ops)

    def test_mixed_quadratic_characters_are_rejected(self) -> None:
        """p = 5 gives PGL mod 13 while p = 17 gives PSL, so they share nothing."""
        with pytest.raises(ValueError, match="mixed quadratic characters"):
            aq.hecke_operators(13, [5, 17])

    def test_joint_eigenspaces_stay_large(self, operators) -> None:
        """Three Hecke operators cut the maximum multiplicity only from 112 to 42.

        The surviving multiplicities are the irreducible representation
        dimensions of PSL(2, F_13) and their small multiples, which is the
        structural reason no canonical eigenvector basis exists: every one of
        these operators is a right convolution, so all of them commute with the
        entire left regular action.
        """
        ops, _ = operators
        histogram = aq.hecke_joint_multiplicities(ops)
        assert max(histogram) == 42
        assert set(histogram) <= {1, 12, 13, 14, 24, 26, 28, 36, 39, 42}
        assert set(histogram) - {1} <= {d * k for d in (12, 13, 14) for k in (1, 2, 3)}


# --------------------------------------------------------------------------- #
# Where the question has content
# --------------------------------------------------------------------------- #
class TestRandomRegularModel:
    @pytest.mark.parametrize("size", [400, 800])
    def test_spectrum_is_simple_so_eigenvectors_are_canonical(self, size: int) -> None:
        adjacency = aq.random_regular_graph(6, size, seed=size)
        eigenvalues = np.linalg.eigvalsh(np.asarray(adjacency.todense()))
        _, counts = np.unique(np.round(eigenvalues, 7), return_counts=True)
        assert counts.max() == 1

    def test_almost_ramanujan(self) -> None:
        """Friedman's theorem: lambda_2 <= 2 sqrt(d-1) + epsilon."""
        adjacency = aq.random_regular_graph(6, 1000, seed=1)
        eigenvalues = np.linalg.eigvalsh(np.asarray(adjacency.todense()))
        second = float(np.abs(eigenvalues[np.abs(eigenvalues - 6) > 1e-8]).max())
        assert second <= 2.0 * np.sqrt(5) + 0.25

    def test_mass_averages_to_one(self) -> None:
        adjacency = aq.random_regular_graph(6, 300, seed=2)
        _, vectors = np.linalg.eigh(np.asarray(adjacency.todense()))
        assert np.mean(aq.eigenvector_mass(vectors[:, 5])) == pytest.approx(1.0)

    def test_rejects_a_zero_vector(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            aq.eigenvector_mass(np.zeros(10))

    def test_thin_set_deviation_tracks_the_gaussian_baseline(self) -> None:
        """The positive result: no scarring, and no arithmetic enhancement either.

        Deviation on random thin sets sits within a few percent of what a random
        Gaussian vector would give, at every subset size tested.
        """
        rng = np.random.default_rng(4)
        size = 800
        adjacency = aq.random_regular_graph(6, size, seed=11)
        values, vectors = np.linalg.eigh(np.asarray(adjacency.todense()))
        bulk = np.flatnonzero((values >= -1.0) & (values <= 1.0))
        for subset_size in (20, 60):
            subsets = [rng.choice(size, subset_size, replace=False) for _ in range(40)]
            deviations = []
            for column in bulk[:: max(1, len(bulk) // 20)]:
                mass = aq.eigenvector_mass(vectors[:, column])
                deviations.extend(
                    abs(float(np.mean(mass[list(s)])) - 1.0) for s in subsets
                )
            observed = float(np.sqrt(np.mean(np.square(deviations))))
            ratio = observed / aq.gaussian_baseline(subset_size)
            assert 0.8 < ratio < 1.25, f"|S|={subset_size}: ratio {ratio}"

    def test_deviation_depends_on_subset_size_not_graph_size(self) -> None:
        """The sharp statement: fixing |S| makes the deviation N-independent."""
        rng = np.random.default_rng(6)
        observed = []
        for size in (400, 1600):
            adjacency = aq.random_regular_graph(6, size, seed=size)
            values, vectors = np.linalg.eigh(np.asarray(adjacency.todense()))
            bulk = np.flatnonzero((values >= -1.0) & (values <= 1.0))
            subsets = [rng.choice(size, 30, replace=False) for _ in range(40)]
            deviations = []
            for column in bulk[:: max(1, len(bulk) // 20)]:
                mass = aq.eigenvector_mass(vectors[:, column])
                deviations.extend(
                    abs(float(np.mean(mass[list(s)])) - 1.0) for s in subsets
                )
            observed.append(float(np.sqrt(np.mean(np.square(deviations)))))
        assert abs(observed[0] - observed[1]) < 0.06, observed


class TestNegativeControl:
    """Without this the thin-set statistic could only ever say "equidistributed"."""

    def test_a_bottlenecked_graph_is_detected_as_scarred(self) -> None:
        rng = np.random.default_rng(8)
        adjacency = aq.bottlenecked_graph(60, 30)
        size = adjacency.shape[0]
        values, vectors = np.linalg.eigh(np.asarray(adjacency.todense()))
        low = np.flatnonzero((values >= -0.05) & (values <= 0.4))
        assert low.size >= 1
        subsets = [rng.choice(size, 12, replace=False) for _ in range(40)]
        deviations = []
        for column in low:
            mass = aq.eigenvector_mass(vectors[:, column])
            deviations.extend(abs(float(np.mean(mass[list(s)])) - 1.0) for s in subsets)
        ratio = float(np.sqrt(np.mean(np.square(deviations)))) / aq.gaussian_baseline(12)
        assert ratio > 1.2, f"scarring not detected: ratio {ratio}"

    def test_its_mass_is_genuinely_concentrated(self) -> None:
        adjacency = aq.bottlenecked_graph(60, 30)
        values, vectors = np.linalg.eigh(np.asarray(adjacency.todense()))
        low = int(np.argmin(np.abs(values - 0.1)))
        assert aq.eigenvector_mass(vectors[:, low]).max() > 5.0

    def test_rejects_degenerate_parameters(self) -> None:
        with pytest.raises(ValueError, match="lobe"):
            aq.bottlenecked_graph(2, 1)


class TestThinSetStatistics:
    def test_reports_size_and_exponent(self) -> None:
        mass = np.ones(1000)
        stats = aq.thin_set_statistics(mass, [list(range(31)), list(range(31, 62))])
        assert stats.subset_size == 31
        assert stats.deviation == pytest.approx(0.0)
        assert stats.alpha == pytest.approx(np.log(31) / np.log(1000))

    def test_detects_a_deviation(self) -> None:
        mass = np.ones(100)
        mass[:10] = 3.0
        stats = aq.thin_set_statistics(mass, [list(range(10))])
        assert stats.deviation == pytest.approx(2.0)
        assert stats.worst_deviation == pytest.approx(2.0)

    def test_requires_uniform_subset_sizes(self) -> None:
        with pytest.raises(ValueError, match="same size"):
            aq.thin_set_statistics(np.ones(10), [[0, 1], [2, 3, 4]])

    def test_requires_at_least_one_subset(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            aq.thin_set_statistics(np.ones(10), [])


class TestGeometryHelpers:
    def test_shortest_cycle_of_a_ring_is_the_whole_ring(self) -> None:
        import networkx as nx

        adjacency = nx.to_scipy_sparse_array(nx.cycle_graph(9), format="csr", dtype=float)
        assert len(aq.shortest_cycle(adjacency)) == 9

    def test_shortest_cycle_of_a_complete_graph_is_a_triangle(self) -> None:
        import networkx as nx

        adjacency = nx.to_scipy_sparse_array(nx.complete_graph(6), format="csr", dtype=float)
        assert len(aq.shortest_cycle(adjacency)) == 3

    def test_a_tree_has_no_cycle(self) -> None:
        import networkx as nx

        adjacency = nx.to_scipy_sparse_array(nx.path_graph(6), format="csr", dtype=float)
        with pytest.raises(ValueError, match="forest"):
            aq.shortest_cycle(adjacency)

    def test_ball_grows_with_radius(self, small_lps) -> None:
        sizes = [len(aq.ball_around(small_lps.adjacency, 0, r)) for r in (0, 1, 2)]
        assert sizes == sorted(sizes)
        assert sizes[0] == 1
        assert sizes[1] == 1 + small_lps.degree

    def test_negative_radius_is_rejected(self, small_lps) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            aq.ball_around(small_lps.adjacency, 0, -1)


class TestSpectralWindow:
    def test_pools_the_window(self, small_spectrum) -> None:
        eigenvalues, eigenvectors = small_spectrum
        mass, count = aq.spectral_window_mass(eigenvalues, eigenvectors, -1.0, 1.0)
        assert count > 0
        assert np.mean(mass) == pytest.approx(1.0, abs=1e-9)

    def test_rejects_an_empty_window(self, small_spectrum) -> None:
        eigenvalues, eigenvectors = small_spectrum
        with pytest.raises(ValueError, match="no eigenvalues in the window"):
            aq.spectral_window_mass(eigenvalues, eigenvectors, 100.0, 200.0)

    def test_rejects_an_inverted_window(self, small_spectrum) -> None:
        eigenvalues, eigenvectors = small_spectrum
        with pytest.raises(ValueError, match="low < high"):
            aq.spectral_window_mass(eigenvalues, eigenvectors, 1.0, -1.0)
