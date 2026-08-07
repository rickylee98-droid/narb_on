"""Tests for the graph Selberg trace formula and the Ihara zeta function.

Nothing here is checked against a restatement of itself.  Every identity is
computed along two routes that share no code: geodesic counts from the edge
operator against brute-force walk enumeration, Bass's determinant over the
adjacency matrix against the determinant over the edge operator, the Euler
product over prime geodesics against the zeta power series, and the growth of
the prime geodesic error against the spectral radius read off the eigenvalues.

Small graphs with known combinatorics anchor the whole thing: the number of
prime geodesics of length equal to the girth must be twice the number of
shortest cycles, once per orientation.
"""

from __future__ import annotations

import math
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

import selberg as sb

CUBIC = [
    ("K4", nx.complete_graph(4)),
    ("K33", nx.complete_bipartite_graph(3, 3)),
    ("Petersen", nx.petersen_graph()),
    ("Heawood", nx.heawood_graph()),
]
REGULAR = CUBIC + [
    ("K5", nx.complete_graph(5)),
    ("Desargues", nx.desargues_graph()),
    ("CircLadder10", nx.circular_ladder_graph(10)),
]


# --------------------------------------------------------------------------- #
# The edge operator
# --------------------------------------------------------------------------- #
class TestHashimoto:
    @pytest.mark.parametrize("name,graph", REGULAR)
    def test_operator_has_the_right_shape_and_row_sums(self, name, graph) -> None:
        operator, edges = sb.hashimoto_operator(graph)
        assert operator.shape == (2 * graph.number_of_edges(),) * 2
        assert len(edges) == 2 * graph.number_of_edges()
        # from a directed edge into v, the continuations are the neighbours of v
        # other than where we came from
        for i, (u, v) in enumerate(edges):
            assert operator[i].sum() == graph.degree(v) - 1

    def test_rejects_a_self_loop(self) -> None:
        graph = nx.Graph([(0, 0), (0, 1)])
        with pytest.raises(ValueError, match="self-loop"):
            sb.hashimoto_operator(graph)

    def test_entries_are_exactly_the_non_backtracking_condition(self) -> None:
        graph = nx.petersen_graph()
        operator, edges = sb.hashimoto_operator(graph)
        index = {e: i for i, e in enumerate(edges)}
        for (a, b) in edges:
            for (c, d) in edges:
                expected = 1 if (b == c and d != a) else 0
                assert operator[index[(a, b)], index[(c, d)]] == expected

    def test_power_trace_matches_a_direct_power(self) -> None:
        operator, _ = sb.hashimoto_operator(nx.complete_graph(4))
        direct = np.eye(operator.shape[0], dtype=object)
        for power in range(1, 8):
            direct = direct @ operator
            assert sb.integer_matrix_power_trace(operator, power) == int(
                np.trace(direct)
            )

    def test_rejects_a_non_positive_power(self) -> None:
        operator, _ = sb.hashimoto_operator(nx.complete_graph(4))
        with pytest.raises(ValueError, match="power"):
            sb.integer_matrix_power_trace(operator, 0)


class TestGeodesicCountsAgainstEnumeration:
    """The referee that makes every later count trustworthy."""

    @pytest.mark.parametrize("name,graph", CUBIC)
    @pytest.mark.parametrize("length", range(3, 9))
    def test_trace_equals_brute_force(self, name, graph, length: int) -> None:
        counted = sb.closed_geodesic_count(graph, length)
        enumerated = sum(1 for _ in sb.enumerate_closed_geodesics(graph, length))
        assert counted == enumerated

    def test_enumeration_produces_genuinely_tailless_walks(self) -> None:
        """The wrap-around condition is what distinguishes tailless from closed."""
        for walk in sb.enumerate_closed_geodesics(nx.petersen_graph(), 6):
            for (a, b), (c, d) in zip(walk, walk[1:]):
                assert b == c and d != a
            assert walk[-1][1] == walk[0][0]
            assert walk[0][1] != walk[-1][0]

    def test_girth_forces_the_short_counts_to_vanish(self) -> None:
        for _, graph in REGULAR:
            girth = min(len(c) for c in nx.minimum_cycle_basis(graph))
            for length in range(1, girth):
                assert sb.closed_geodesic_count(graph, length) == 0

    def test_bipartite_graphs_have_no_odd_geodesics(self) -> None:
        for graph in (nx.heawood_graph(), nx.complete_bipartite_graph(3, 3)):
            for length in (3, 5, 7, 9):
                assert sb.closed_geodesic_count(graph, length) == 0


# --------------------------------------------------------------------------- #
# Exact polynomial algebra
# --------------------------------------------------------------------------- #
class TestPolynomials:
    def test_multiply_matches_numpy_convolution(self) -> None:
        left, right = [1, -3, 2], [4, 0, -1, 5]
        assert sb.poly_multiply(left, right) == list(
            np.convolve(left, right).astype(int)
        )

    def test_power_agrees_with_repeated_multiplication(self) -> None:
        base = [1, 0, -1]
        expected = [1]
        for exponent in range(6):
            assert sb.poly_power(base, exponent) == expected
            expected = sb.poly_multiply(expected, base)

    def test_rejects_a_negative_exponent(self) -> None:
        with pytest.raises(ValueError, match="exponent"):
            sb.poly_power([1, 1], -1)

    def test_characteristic_polynomial_of_a_known_matrix(self) -> None:
        """``det(I - uA)`` for the adjacency matrix of ``K3`` is ``1 - 3u^2 - 2u^3``."""
        adjacency = np.array(
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=object
        )
        assert sb.characteristic_polynomial(adjacency) == [1, 0, -3, -2]

    def test_characteristic_polynomial_matches_numpy_roots(self) -> None:
        graph = nx.petersen_graph()
        adjacency = nx.to_numpy_array(graph, nodelist=sorted(graph)).astype(int)
        exact = sb.characteristic_polynomial(adjacency.astype(object))
        # det(I - uA) has roots 1/lambda; compare via the value at a point
        point = Fraction(1, 7)
        expected = round(
            float(np.linalg.det(np.eye(10) - float(point) * adjacency)), 9
        )
        got = float(sum(c * point**k for k, c in enumerate(exact)))
        assert got == pytest.approx(expected, abs=1e-7)


# --------------------------------------------------------------------------- #
# Bass's theorem
# --------------------------------------------------------------------------- #
class TestBass:
    @pytest.mark.parametrize(
        "name,graph", REGULAR + [("2T", sb.binary_tetrahedral_cayley())]
    )
    def test_bass_reproduces_the_edge_determinant(self, name, graph) -> None:
        """Two determinants of different sizes, computed by different recursions."""
        left = sb.ihara_zeta_reciprocal(graph)
        right = sb.bass_reciprocal(graph)
        size = max(len(left), len(right))
        left = left + [0] * (size - len(left))
        right = right + [0] * (size - len(right))
        assert left == right

    def test_the_polynomial_has_the_expected_degree(self) -> None:
        for _, graph in REGULAR:
            reciprocal = sb.ihara_zeta_reciprocal(graph)
            assert len(reciprocal) - 1 == 2 * graph.number_of_edges()
            assert reciprocal[0] == 1

    def test_k4_zeta_matches_its_classical_value(self) -> None:
        """A closed-form anchor outside this module's own machinery.

        ``1/zeta_{K_4}(u) = (1-u^2)^2 (1-u)(1-2u)(1+u+2u^2)^3``.
        """
        expected = sb.poly_multiply(
            sb.poly_multiply(
                sb.poly_power([1, 0, -1], 2), sb.poly_multiply([1, -1], [1, -2])
            ),
            sb.poly_power([1, 1, 2], 3),
        )
        assert sb.ihara_zeta_reciprocal(nx.complete_graph(4)) == expected

    def test_rejects_an_irregular_graph(self) -> None:
        with pytest.raises(ValueError, match="not regular"):
            sb.bass_reciprocal(nx.path_graph(4))


# --------------------------------------------------------------------------- #
# Prime geodesics
# --------------------------------------------------------------------------- #
class TestMoebius:
    def test_known_values(self) -> None:
        expected = [1, -1, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0]
        assert [sb.moebius(n) for n in range(1, 13)] == expected

    def test_rejects_a_non_positive_argument(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            sb.moebius(0)

    def test_divisor_sum_is_the_indicator_of_one(self) -> None:
        for n in range(1, 40):
            total = sum(sb.moebius(d) for d in range(1, n + 1) if n % d == 0)
            assert total == (1 if n == 1 else 0)


class TestPrimeGeodesics:
    @pytest.mark.parametrize(
        "name,graph,girth,cycles",
        [
            ("K4", nx.complete_graph(4), 3, 4),
            ("Petersen", nx.petersen_graph(), 5, 12),
            ("Heawood", nx.heawood_graph(), 6, 28),
            ("K33", nx.complete_bipartite_graph(3, 3), 4, 9),
        ],
    )
    def test_shortest_geodesics_are_the_shortest_cycles(
        self, name, graph, girth: int, cycles: int
    ) -> None:
        """Independent combinatorial anchor: ``pi(girth) = 2 * (number of cycles)``.

        Each shortest cycle gives exactly two prime geodesics, one per
        orientation, and nothing shorter exists to interfere.
        """
        counts = sb.prime_geodesic_counts(graph, girth)
        assert counts[girth] == 2 * cycles
        for length in range(1, girth):
            assert counts[length] == 0

    def test_moebius_inversion_is_consistent(self) -> None:
        for _, graph in REGULAR:
            counts = sb.prime_geodesic_counts(graph, 12)
            assert all(value >= 0 for value in counts.values())

    def test_inversion_rejects_inconsistent_counts(self) -> None:
        with pytest.raises(ValueError, match="not divisible"):
            sb.prime_geodesic_count({1: 0, 2: 1}, 2)

    def test_inversion_reports_a_missing_divisor(self) -> None:
        with pytest.raises(KeyError, match="N_2"):
            sb.prime_geodesic_count({1: 0, 4: 8}, 4)

    @pytest.mark.parametrize("name,graph", REGULAR)
    def test_euler_product_reproduces_the_zeta_series(self, name, graph) -> None:
        """``zeta(u) = prod_P (1 - u^{len P})^{-1}`` as formal power series."""
        terms = 13
        series = sb.zeta_series(sb.ihara_zeta_reciprocal(graph), terms)
        product = sb.euler_product_series(
            sb.prime_geodesic_counts(graph, terms - 1), terms
        )
        assert series == product

    def test_zeta_series_rejects_a_zero_constant_term(self) -> None:
        with pytest.raises(ValueError, match="constant term"):
            sb.zeta_series([0, 1], 4)


# --------------------------------------------------------------------------- #
# The trace formula
# --------------------------------------------------------------------------- #
class TestTraceFormula:
    @pytest.mark.parametrize(
        "name,graph", REGULAR + [("2T", sb.binary_tetrahedral_cayley())]
    )
    def test_spectral_side_equals_geodesic_side(self, name, graph) -> None:
        check = sb.trace_formula_check(graph, 12)
        assert check.agrees
        assert check.first_disagreement is None

    def test_the_check_can_actually_fail(self) -> None:
        """A positive control on the comparison itself."""
        check = sb.TraceFormulaCheck(
            lengths=(1, 2), geodesic=(0, 4), spectral=(0, 5)
        )
        assert not check.agrees
        assert check.first_disagreement == 2


# --------------------------------------------------------------------------- #
# Ramanujan property and the RH analogue
# --------------------------------------------------------------------------- #
class TestRamanujan:
    @pytest.mark.parametrize(
        "name,graph,expected",
        [
            ("Petersen", nx.petersen_graph(), True),
            ("Heawood", nx.heawood_graph(), True),
            ("K5", nx.complete_graph(5), True),
            ("Desargues", nx.desargues_graph(), True),
            ("CircLadder20", nx.circular_ladder_graph(20), False),
            ("C20^2", nx.circulant_graph(20, [1, 2]), False),
        ],
    )
    def test_ramanujan_classification(self, name, graph, expected: bool) -> None:
        assert sb.ramanujan_report(graph).is_ramanujan is expected

    def test_petersen_spectrum_is_the_known_one(self) -> None:
        values = sorted(sb.adjacency_spectrum(nx.petersen_graph()))
        assert values[:4] == pytest.approx([-2, -2, -2, -2])
        assert values[-1] == pytest.approx(3)

    def test_bipartite_graphs_have_both_trivial_eigenvalues_removed(self) -> None:
        report = sb.ramanujan_report(nx.heawood_graph())
        assert report.trivial == pytest.approx((-3.0, 3.0))

    def test_spectral_radii_are_sqrt_q_exactly_when_ramanujan(self) -> None:
        """The dichotomy the RH analogue rests on."""
        for _, graph in [("Petersen", nx.petersen_graph()), ("Heawood", nx.heawood_graph())]:
            q = sb.graph_degree(graph) - 1
            radius = sb.dominant_nontrivial_radius(graph)
            assert radius == pytest.approx(math.sqrt(q), abs=1e-9)
        loose = nx.circular_ladder_graph(20)
        assert sb.dominant_nontrivial_radius(loose) > math.sqrt(2) + 0.05


class TestRiemannHypothesis:
    """The quantitative statement: error growth equals the spectral radius ratio."""

    @pytest.mark.parametrize(
        "name,graph",
        [
            ("Petersen", nx.petersen_graph()),
            ("Heawood", nx.heawood_graph()),
            ("K5", nx.complete_graph(5)),
            ("Pappus", nx.pappus_graph()),
            ("2T", sb.binary_tetrahedral_cayley()),
        ],
    )
    def test_ramanujan_graphs_have_bounded_normalised_error(self, name, graph) -> None:
        """``R(m) = |pi(m) - main| m / q^{m/2}`` does not grow when every pole is on
        the critical circle."""
        test = sb.riemann_hypothesis_test(graph, max_length=18)
        assert test.is_ramanujan
        assert test.predicted_growth == pytest.approx(1.0, abs=1e-9)
        assert test.growth == pytest.approx(1.0, abs=0.08)

    @pytest.mark.parametrize(
        "name,graph",
        [
            ("CircLadder20", nx.circular_ladder_graph(20)),
            ("CircLadder30", nx.circular_ladder_graph(30)),
            ("C20^2", nx.circulant_graph(20, [1, 2])),
            ("C24^2", nx.circulant_graph(24, [1, 2])),
        ],
    )
    def test_non_ramanujan_error_grows_at_the_predicted_rate(self, name, graph) -> None:
        """The prediction has no free parameter.

        The growth rate is read off the adjacency spectrum and the error is
        counted from closed walks; the two computations share nothing.
        """
        test = sb.riemann_hypothesis_test(graph, max_length=18)
        assert not test.is_ramanujan
        assert test.predicted_growth > 1.05
        assert test.growth == pytest.approx(test.predicted_growth, abs=0.08)
        assert test.agrees_with_spectrum

    def test_the_two_classes_are_actually_separated(self) -> None:
        good = sb.riemann_hypothesis_test(nx.petersen_graph(), max_length=18)
        bad = sb.riemann_hypothesis_test(
            nx.circular_ladder_graph(30), max_length=18
        )
        assert good.growth < 1.1 < bad.growth

    def test_rejects_a_disconnected_graph(self) -> None:
        """The bug this guard was written for.

        A disconnected graph counts one main term per component, and the missing
        copies look exactly like an error growing at ``sqrt q`` per step -- large,
        clean, and entirely an artefact.
        """
        graph = nx.disjoint_union(nx.petersen_graph(), nx.petersen_graph())
        with pytest.raises(ValueError, match="components"):
            sb.riemann_hypothesis_test(graph)

    def test_the_fitted_exponent_is_recorded_as_unreliable(self) -> None:
        """The discarded instrument, kept so its failure stays visible.

        The error oscillates, so a log-linear fit through it scatters badly; the
        residual is the tell.
        """
        fit = sb.prime_geodesic_fit(nx.petersen_graph(), max_length=16)
        assert fit.residual > 0.3
        assert not fit.matches_prediction


# --------------------------------------------------------------------------- #
# The 24 Hurwitz units
# --------------------------------------------------------------------------- #
class TestHurwitz:
    def test_there_are_twenty_four_units_all_of_norm_one(self) -> None:
        units = sb.hurwitz_units()
        assert len(units) == 24
        assert len(set(units)) == 24
        for unit in units:
            assert sum(c * c for c in unit) == 1

    def test_the_units_are_closed_under_multiplication(self) -> None:
        """They form the binary tetrahedral group, the double cover of the
        tetrahedron's rotation group -- which is where this meets the packing
        problem the project started from."""
        units = set(sb.hurwitz_units())
        for left in units:
            for right in units:
                assert sb.quaternion_multiply(left, right) in units

    def test_quaternion_multiplication_obeys_hamilton(self) -> None:
        zero, one = Fraction(0), Fraction(1)
        i = (zero, one, zero, zero)
        j = (zero, zero, one, zero)
        k = (zero, zero, zero, one)
        minus_one = (-one, zero, zero, zero)
        assert sb.quaternion_multiply(i, i) == minus_one
        assert sb.quaternion_multiply(i, j) == k
        assert sb.quaternion_multiply(j, k) == i
        assert sb.quaternion_multiply(k, i) == j
        assert sb.quaternion_multiply(j, i) == (zero, zero, zero, -one)

    def test_the_default_cayley_graph_is_connected_and_regular(self) -> None:
        graph = sb.binary_tetrahedral_cayley()
        assert graph.number_of_nodes() == 24
        assert nx.is_connected(graph)
        assert sb.graph_degree(graph) == 4

    def test_the_quaternion_generators_are_rejected_as_too_small(self) -> None:
        """The bug: ``{+-i, +-j, +-k}`` generates only the order-8 quaternion group.

        The resulting graph is a perfectly ordinary-looking 6-regular graph on 24
        vertices that passes the Ramanujan test, and is wrong.
        """
        zero, one = Fraction(0), Fraction(1)
        generators = [
            (zero, one, zero, zero),
            (zero, -one, zero, zero),
            (zero, zero, one, zero),
            (zero, zero, -one, zero),
            (zero, zero, zero, one),
            (zero, zero, zero, -one),
        ]
        with pytest.raises(ValueError, match="subgroup of order 8"):
            sb.binary_tetrahedral_cayley(generators)

    def test_rejects_a_generating_set_not_closed_under_inverse(self) -> None:
        zero, one = Fraction(0), Fraction(1)
        with pytest.raises(ValueError, match="closed under inverse"):
            sb.binary_tetrahedral_cayley([(zero, one, zero, zero)])

    def test_rejects_a_non_unit_generator(self) -> None:
        with pytest.raises(ValueError, match="not a Hurwitz unit"):
            sb.binary_tetrahedral_cayley([(Fraction(2), Fraction(0), Fraction(0), Fraction(0))])


# --------------------------------------------------------------------------- #
# The spectral fast path, and how far it reaches
# --------------------------------------------------------------------------- #
class TestSpectralRoute:
    @pytest.mark.parametrize(
        "name,graph",
        REGULAR + [("2T", sb.binary_tetrahedral_cayley())],
    )
    def test_spectral_counts_match_the_exact_ones(self, name, graph) -> None:
        """The fast path must reproduce ``tr(B^m)`` wherever both are affordable."""
        spectral = sb.geodesic_counts_from_spectrum(graph, 12)
        exact = sb.trace_formula_check(graph, 12).geodesic
        assert [spectral[m] for m in range(1, 13)] == list(exact)

    @pytest.mark.parametrize("name,graph", REGULAR)
    def test_the_auto_switch_changes_no_answer(self, name, graph) -> None:
        assert sb.prime_geodesic_counts(
            graph, 12, spectral=False
        ) == sb.prime_geodesic_counts(graph, 12, spectral=True)

    def test_the_precision_guard_refuses_an_unreachable_length(self) -> None:
        """The cancellation the guard protects.

        The main term is ``q^m`` and the error being measured is ``q^{m/2}``, so
        eigenvalues good to machine precision stop supporting the subtraction at
        a length that depends on the degree and the order.  Past it the measured
        error is the eigensolver's own.
        """
        graph = nx.random_regular_graph(14, 120, seed=0)
        with pytest.raises(ValueError, match="round-off"):
            sb.geodesic_counts_from_spectrum(graph, 40)

    def test_the_guard_permits_a_short_range_on_the_same_graph(self) -> None:
        graph = nx.random_regular_graph(14, 120, seed=0)
        counts = sb.geodesic_counts_from_spectrum(graph, 10)
        assert all(counts[m] >= 0 for m in counts)


class TestReachOfTheRHTest:
    """An honest bound on the method, found by a deliberate control."""

    def test_a_short_length_range_cannot_separate_the_classes(self) -> None:
        """The negative result.

        With ``max_length = 18`` the measured growth tracks the spectral
        prediction to ``0.08`` on every graph tested.  With ``max_length = 11``
        -- which is all the precision guard permits at degree 14 -- a random
        14-regular graph that *is* Ramanujan reads ``1.58`` against a prediction
        of ``1.00``.  The statistic needs a long baseline, and quoting it from a
        short one would turn noise into a claim.
        """
        graph = nx.random_regular_graph(14, 120, seed=1)
        assert sb.ramanujan_report(graph).is_ramanujan
        short = sb.riemann_hypothesis_test(
            graph, max_length=11, min_length=4, spectral=True
        )
        assert short.predicted_growth == pytest.approx(1.0, abs=1e-9)
        assert short.growth > 1.3, "expected the short range to mislead"
        assert not short.agrees_with_spectrum

    def test_the_long_range_verdict_is_the_one_to_trust(self) -> None:
        long_run = sb.riemann_hypothesis_test(
            nx.petersen_graph(), max_length=18, min_length=4
        )
        assert long_run.agrees_with_spectrum
        assert len(long_run.lengths) >= 15


class TestExactRecurrence:
    """The route that removed the precision floor entirely."""

    @pytest.mark.parametrize(
        "name,graph", REGULAR + [("2T", sb.binary_tetrahedral_cayley())]
    )
    def test_recurrence_matches_the_edge_operator(self, name, graph) -> None:
        exact = sb.geodesic_counts_exact(graph, 12)
        direct = sb.trace_formula_check(graph, 12).geodesic
        assert [exact[m] for m in range(1, 13)] == list(direct)

    def test_it_reaches_lengths_the_spectral_route_refuses(self) -> None:
        """Same graph, same length: one route refuses, the other is exact."""
        graph = nx.random_regular_graph(14, 120, seed=0)
        with pytest.raises(ValueError, match="round-off"):
            sb.geodesic_counts_from_spectrum(graph, 40)
        counts = sb.geodesic_counts_exact(graph, 40)
        assert counts[40] > 13**39

    def test_counts_are_python_integers_not_floats(self) -> None:
        counts = sb.geodesic_counts_exact(nx.petersen_graph(), 30)
        assert all(isinstance(v, int) for v in counts.values())

    def test_rejects_a_non_positive_length(self) -> None:
        with pytest.raises(ValueError, match="max_length"):
            sb.geodesic_counts_exact(nx.petersen_graph(), 0)


class TestFloatAnnihilation:
    """The bug that silently zeroed the tail of every long sequence.

    ``pi(m)`` reaches ``10^44`` at ``m = 40, q = 13`` while the error being
    measured is only ``10^22``.  Subtracting a float main term from that coerces
    the exact count to a float and destroys everything below ``10^28``, so the
    normalised error came out as an unbroken run of exact zeros, with occasional
    spurious spikes where the rounding happened to land elsewhere.  Multiplying
    through by ``m`` keeps the whole quantity integral.
    """

    def test_the_long_tail_is_not_identically_zero(self) -> None:
        for graph in (
            nx.petersen_graph(),
            nx.random_regular_graph(14, 120, seed=0),
        ):
            test = sb.riemann_hypothesis_test(
                graph, max_length=40, min_length=4, spectral=True
            )
            tail = test.normalised[-8:]
            assert sum(1 for value in tail if value == 0.0) <= 1

    def test_the_naive_float_subtraction_really_does_annihilate(self) -> None:
        """A positive control on the diagnosis itself."""
        q, m = 13, 40
        primes = sb.prime_geodesic_counts(
            nx.random_regular_graph(14, 120, seed=0), m, spectral=True
        )
        naive = abs(primes[m] - 1.0 * q**m / m) * m / q ** (m / 2.0)
        exact = float(abs(primes[m] * m - q**m)) / q ** (m / 2.0)
        assert exact > 1.0, "there is a real error term to detect"
        assert naive < 0.01 * exact, (
            f"the float route should have destroyed the signal, got {naive} "
            f"against {exact}"
        )

    def test_ramanujan_and_non_ramanujan_separate_at_length_forty(self) -> None:
        good = sb.riemann_hypothesis_test(
            nx.random_regular_graph(14, 120, seed=1), max_length=40, spectral=True
        )
        bad = sb.riemann_hypothesis_test(
            nx.circulant_graph(120, [1, 2, 3, 4, 5, 6, 7]),
            max_length=40,
            spectral=True,
        )
        assert good.is_ramanujan and not bad.is_ramanujan
        assert good.growth == pytest.approx(1.0, abs=0.05)
        assert bad.growth == pytest.approx(bad.predicted_growth, abs=0.05)
        assert bad.growth > 3.0
