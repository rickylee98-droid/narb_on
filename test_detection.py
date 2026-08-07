"""Tests for community detection as a failure of the Ramanujan bound.

The chain being tested runs from a proved statement to a working detector: the
root dichotomy of `selberg.py` says the non-backtracking spectrum of a graph
without structure lies on a circle, so anything outside it is structure, and the
assortativity at which a planted partition pushes an eigenvalue past the circle
is a threshold predicted with no fitted constant.

Each link is checked against something independent. The sparse operator is
compared entrywise with the exact integer one from `selberg.py`; the predicted
threshold is compared with where detection is actually observed to begin; and the
claimed advantage over the adjacency matrix is measured on the same graphs rather
than asserted.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

import detection as dt
import selberg as sb

SMALL = [
    ("K4", nx.complete_graph(4)),
    ("K33", nx.complete_bipartite_graph(3, 3)),
    ("Petersen", nx.petersen_graph()),
    ("Heawood", nx.heawood_graph()),
    ("Desargues", nx.desargues_graph()),
    ("2T", sb.binary_tetrahedral_cayley()),
]


# --------------------------------------------------------------------------- #
# The sparse operator, against the exact one
# --------------------------------------------------------------------------- #
class TestSparseHashimoto:
    @pytest.mark.parametrize("name,graph", SMALL)
    def test_entries_match_the_exact_operator(self, name, graph) -> None:
        """Entrywise, not merely up to a permutation.

        This only works because the sparse builder reuses
        `selberg.directed_edges` rather than repeating its convention. Built from
        ``graph.edges()`` in insertion order the two matrices are
        permutation-similar and agree on spectra while disagreeing entrywise,
        which would make this referee vacuous.
        """
        exact, exact_edges = sb.hashimoto_operator(graph)
        sparse, sparse_edges = dt.sparse_hashimoto(graph)
        assert tuple(exact_edges) == tuple(sparse_edges)
        assert np.array_equal(
            sparse.toarray().astype(int),
            np.array(exact.tolist(), dtype=int),
        )

    @pytest.mark.parametrize("name,graph", SMALL)
    def test_row_sums_are_the_branching_number(self, name, graph) -> None:
        sparse, edges = dt.sparse_hashimoto(graph)
        sums = np.asarray(sparse.sum(axis=1)).ravel()
        for i, (_, head) in enumerate(edges):
            assert sums[i] == graph.degree(head) - 1

    def test_rejects_a_self_loop(self) -> None:
        graph = nx.Graph([(0, 0), (0, 1)])
        with pytest.raises(ValueError, match="self-loop"):
            dt.sparse_hashimoto(graph)

    @pytest.mark.parametrize("name,graph", SMALL)
    def test_spectrum_matches_the_dense_eigenvalues(self, name, graph) -> None:
        exact, _ = sb.hashimoto_operator(graph)
        dense = np.array(exact.tolist(), dtype=float)
        expected = np.sort(np.abs(np.linalg.eigvals(dense)))[::-1][:3]
        got = np.sort(np.abs(dt.nonbacktracking_spectrum(graph, count=3)))[::-1]
        assert got == pytest.approx(expected, rel=1e-8)


# --------------------------------------------------------------------------- #
# The circle, and what sits on it
# --------------------------------------------------------------------------- #
class TestBulkRadius:
    @pytest.mark.parametrize(
        "name,graph",
        [("Petersen", nx.petersen_graph()), ("Heawood", nx.heawood_graph())],
    )
    def test_agrees_with_selberg_on_a_regular_ramanujan_graph(
        self, name, graph
    ) -> None:
        """For a Ramanujan graph the two notions of the circle must coincide.

        `selberg.dominant_nontrivial_radius` puts every non-trivial pole at
        ``sqrt q``; `bulk_radius` computes ``sqrt(c-1)`` from the mean degree.
        On a regular graph ``c = q + 1``, so they are the same number, arrived at
        from the spectrum and from the degree sequence respectively.
        """
        assert sb.ramanujan_report(graph).is_ramanujan
        assert dt.bulk_radius(graph) == pytest.approx(
            sb.dominant_nontrivial_radius(graph), rel=1e-9
        )

    def test_a_non_ramanujan_graph_has_an_eigenvalue_outside(self) -> None:
        """The link the whole module rests on: failing the bound means escaping."""
        graph = nx.circular_ladder_graph(30)
        assert not sb.ramanujan_report(graph).is_ramanujan
        assert sb.dominant_nontrivial_radius(graph) > dt.bulk_radius(graph)

    def test_rejects_a_graph_too_sparse_for_a_bulk(self) -> None:
        with pytest.raises(ValueError, match="mean degree"):
            dt.bulk_radius(nx.path_graph(2))

    def test_rejects_an_empty_graph(self) -> None:
        with pytest.raises(ValueError, match="no vertices"):
            dt.bulk_radius(nx.Graph())


class TestThresholdFormula:
    @pytest.mark.parametrize("degree", [3.0, 5.0, 10.0, 20.0])
    def test_threshold_is_where_the_signal_meets_the_bulk(self, degree) -> None:
        threshold = dt.kesten_stigum_threshold(degree)
        assert degree * threshold == pytest.approx(math.sqrt(degree - 1.0))

    def test_threshold_falls_as_the_graph_gets_denser(self) -> None:
        values = [dt.kesten_stigum_threshold(c) for c in (3.0, 5.0, 10.0, 40.0)]
        assert values == sorted(values, reverse=True)

    @pytest.mark.parametrize("bad", [1.0, 0.5, 0.0])
    def test_rejects_a_degree_that_cannot_support_a_bulk(self, bad) -> None:
        with pytest.raises(ValueError, match="mean degree"):
            dt.kesten_stigum_threshold(bad)


# --------------------------------------------------------------------------- #
# The generator
# --------------------------------------------------------------------------- #
class TestBlockModel:
    @pytest.mark.parametrize("assortativity", [0.0, 0.4, 0.8])
    def test_mean_degree_is_what_was_asked_for(self, assortativity) -> None:
        """The bug this pins moved the threshold being measured.

        Writing the block loop as ``for b in (a, 1)`` visits the ``(1,1)`` block
        twice, sampling group one's internal edges at double rate. The mean
        degree came out at ``6.22`` for a requested ``5``, and since the
        threshold is a function of the mean degree, the experiment was testing a
        prediction for a graph it was not generating.
        """
        graph, _ = dt.stochastic_block_model(2000, 5.0, assortativity, seed=0)
        degrees = [d for _, d in graph.degree()]
        assert float(np.mean(degrees)) == pytest.approx(5.0, abs=0.25)

    @pytest.mark.parametrize("assortativity", [0.0, 0.4, 0.8])
    def test_the_two_groups_have_equal_density(self, assortativity) -> None:
        """The same bug, seen as an asymmetry rather than a shift.

        Unequal group densities are structure that was never asked for, and a
        detector could find them while appearing to find the planted partition.
        """
        graph, labels = dt.stochastic_block_model(2000, 5.0, assortativity, seed=1)
        first = [graph.degree(v) for v in graph if labels[v] == 0]
        second = [graph.degree(v) for v in graph if labels[v] == 1]
        assert float(np.mean(first)) == pytest.approx(
            float(np.mean(second)), abs=0.35
        )

    def test_labels_line_up_with_the_returned_graph(self) -> None:
        graph, labels = dt.stochastic_block_model(500, 6.0, 0.5, seed=2)
        assert len(labels) == graph.number_of_nodes()
        assert set(graph.nodes()) == set(range(graph.number_of_nodes()))
        assert set(np.unique(labels)) <= {0, 1}

    def test_the_returned_graph_is_connected(self) -> None:
        graph, _ = dt.stochastic_block_model(1000, 5.0, 0.3, seed=3)
        assert nx.is_connected(graph)

    def test_zero_assortativity_plants_nothing(self) -> None:
        """A negative control on the generator itself."""
        graph, labels = dt.stochastic_block_model(2000, 5.0, 0.0, seed=4)
        crossing = sum(1 for u, v in graph.edges() if labels[u] != labels[v])
        assert crossing / graph.number_of_edges() == pytest.approx(0.5, abs=0.05)

    def test_high_assortativity_almost_separates_the_groups(self) -> None:
        graph, labels = dt.stochastic_block_model(2000, 5.0, 0.9, seed=5)
        crossing = sum(1 for u, v in graph.edges() if labels[u] != labels[v])
        assert crossing / graph.number_of_edges() < 0.1

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_rejects_an_impossible_assortativity(self, bad) -> None:
        with pytest.raises(ValueError, match="assortativity"):
            dt.stochastic_block_model(100, 5.0, bad)

    def test_rejects_an_odd_size(self) -> None:
        with pytest.raises(ValueError, match="even"):
            dt.stochastic_block_model(101, 5.0, 0.5)


class TestOverlap:
    def test_perfect_and_inverted_partitions_both_score_one(self) -> None:
        """A partition is unchanged by swapping the labels."""
        truth = np.array([0, 0, 1, 1])
        assert dt.overlap(truth, truth) == pytest.approx(1.0)
        assert dt.overlap(1 - truth, truth) == pytest.approx(1.0)

    def test_chance_scores_zero(self) -> None:
        truth = np.array([0, 0, 1, 1])
        assert dt.overlap(np.array([0, 1, 0, 1]), truth) == pytest.approx(0.0)

    def test_never_negative(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            truth = rng.integers(0, 2, 50)
            guess = rng.integers(0, 2, 50)
            assert dt.overlap(guess, truth) >= 0.0

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            dt.overlap(np.array([0, 1]), np.array([0, 1, 0]))


# --------------------------------------------------------------------------- #
# The prediction
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sweep() -> list[dt.SweepPoint]:
    return dt.threshold_sweep(
        [0.0, 0.2, 0.35, 0.5, 0.7],
        size=2000,
        mean_degree=5.0,
        samples=2,
        seed=0,
    )


class TestDetectionThreshold:
    def test_below_threshold_the_detector_is_at_chance(self, sweep) -> None:
        for point in sweep:
            if point.assortativity <= 0.35:
                assert point.nonbacktracking_overlap < 0.15

    def test_above_threshold_the_detector_works(self, sweep) -> None:
        strong = [p for p in sweep if p.assortativity >= 0.7]
        assert strong
        for point in strong:
            assert point.nonbacktracking_overlap > 0.4

    def test_the_signal_eigenvalue_is_pinned_below_and_tracks_above(
        self, sweep
    ) -> None:
        """The sharp form of the prediction: ``|lam_2| = max(bulk edge, c eps)``.

        Below the threshold the second eigenvalue does not move with the signal
        at all -- it is stuck at the bulk edge. Above it, it detaches and follows
        ``c * eps``. At ``c = 5`` the measured values at ``eps = 0.7, 0.8`` are
        ``3.500`` and ``4.000`` against ``3.50`` and ``4.00``.
        """
        below = [p.signal_eigenvalue for p in sweep if p.assortativity <= 0.35]
        assert max(below) / min(below) < 1.1, "pinned below the threshold"

        strong = [p for p in sweep if p.assortativity >= 0.7][0]
        assert strong.signal_eigenvalue == pytest.approx(
            strong.mean_degree * strong.assortativity, rel=0.15
        )

    def test_non_backtracking_beats_adjacency_near_the_threshold(
        self, sweep
    ) -> None:
        """The claimed advantage, measured on the same graphs.

        The adjacency spectrum of a sparse graph is spoiled by degree
        fluctuations, whose eigenvectors localise on high-degree vertices and
        carry no partition information. Far above the threshold both detectors
        work and the advantage disappears, so the comparison is only meaningful
        near the transition.
        """
        near = [p for p in sweep if p.assortativity == 0.5]
        assert near
        point = near[0]
        assert point.nonbacktracking_overlap > point.adjacency_overlap + 0.1

    def test_sweep_records_the_threshold_it_was_run_against(self, sweep) -> None:
        for point in sweep:
            assert point.threshold == pytest.approx(
                dt.kesten_stigum_threshold(5.0)
            )
            assert point.above_threshold == (
                point.assortativity > point.threshold
            )


class TestFiniteSize:
    """Why the asymptotic threshold is not the one the experiment obeys."""

    def test_the_measured_bulk_edge_exceeds_the_asymptotic_radius(self) -> None:
        edge = dt.empirical_bulk_edge(1000, 5.0, samples=2, seed=0)
        assert edge > math.sqrt(4.0)
        assert edge < 2.6

    def test_the_edge_is_stable_in_size_and_does_not_approach_the_limit(
        self,
    ) -> None:
        """It does *not* visibly converge, which is the point.

        Measured at ``c = 5`` over ``n = 500`` to ``8000`` the edge sits at
        ``2.28, 2.30, 2.28, 2.26, 2.26`` -- flat to about one percent and nowhere
        near the asymptotic ``2.00``. An earlier version of this test asserted a
        descent toward the limit and failed, because there is no descent to see
        at these sizes.

        The consequence is practical rather than cosmetic: the gap does not wash
        out by taking a bigger graph, so a detector that compares against
        ``sqrt(c-1)`` will keep reporting structure in structureless graphs at
        any size one can actually build.
        """
        edges = [
            dt.empirical_bulk_edge(size, 5.0, samples=2, seed=0)
            for size in (500, 2000)
        ]
        for edge in edges:
            assert edge > math.sqrt(4.0) + 0.15
            assert edge < 2.6
        assert max(edges) / min(edges) < 1.05

    def test_the_finite_size_threshold_exceeds_the_asymptotic_one(self) -> None:
        """And it is the one that predicts where detection actually begins.

        At ``c = 5``, ``n = 4000`` the asymptotic threshold is ``0.400`` and the
        finite-size one about ``0.452``; detection was observed to start between
        ``0.45`` and ``0.50``.
        """
        finite = dt.finite_size_threshold(1000, 5.0, samples=2, seed=0)
        assert finite > dt.kesten_stigum_threshold(5.0)
        assert finite == pytest.approx(0.46, abs=0.05)


class TestDetectionResult:
    def test_margin_and_escape_agree(self) -> None:
        result = dt.DetectionResult(
            labels=np.zeros(4, dtype=int), signal_eigenvalue=3.0, bulk_radius=2.0
        )
        assert result.escapes_bulk
        assert result.margin == pytest.approx(0.5)

        quiet = dt.DetectionResult(
            labels=np.zeros(4, dtype=int), signal_eigenvalue=1.9, bulk_radius=2.0
        )
        assert not quiet.escapes_bulk
        assert quiet.margin < 0.0

    def test_detector_refuses_a_graph_with_too_few_arcs(self) -> None:
        with pytest.raises(ValueError, match="too small"):
            dt.detect_communities(nx.path_graph(3))
