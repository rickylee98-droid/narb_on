"""Test suite for the tetrahedral point-cloud spectral analysis pipeline.

Run with::

    python -m pytest test_tetra_spectral.py -v
"""

from __future__ import annotations

import math
import subprocess
from fractions import Fraction
import sys

import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csr_matrix

import tetra_fastsat as tf
import tetra_geometry as tg
import tetra_lift as tl
import tetra_spectral as ts
import tetra_spectral_analysis as cli


# --------------------------------------------------------------------------- #
# Canonical tetrahedron
# --------------------------------------------------------------------------- #
class TestRegularTetrahedron:
    def test_all_six_edges_are_unit_length(self) -> None:
        v = tg.regular_tetrahedron(1.0)
        i, j = tg.TETRA_EDGES[:, 0], tg.TETRA_EDGES[:, 1]
        lengths = np.linalg.norm(v[i] - v[j], axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-15)

    def test_centroid_is_the_origin(self) -> None:
        np.testing.assert_allclose(tg.regular_tetrahedron().mean(axis=0), 0.0, atol=1e-15)

    def test_volume_matches_closed_form(self) -> None:
        v = tg.regular_tetrahedron(1.0)
        assert tg.tetrahedron_volume(v) == pytest.approx(1.0 / (6.0 * math.sqrt(2.0)))

    def test_circumradius_matches_closed_form(self) -> None:
        v = tg.regular_tetrahedron(1.0)
        radii = np.linalg.norm(v, axis=1)
        np.testing.assert_allclose(radii, tg.UNIT_TETRA_CIRCUMRADIUS, atol=1e-15)

    @pytest.mark.parametrize("edge", [0.5, 2.0, 7.25])
    def test_edge_scaling(self, edge: float) -> None:
        v = tg.regular_tetrahedron(edge)
        i, j = tg.TETRA_EDGES[:, 0], tg.TETRA_EDGES[:, 1]
        np.testing.assert_allclose(np.linalg.norm(v[i] - v[j], axis=1), edge, rtol=1e-14)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_invalid_edge(self, bad: float) -> None:
        with pytest.raises(ValueError):
            tg.regular_tetrahedron(bad)

    def test_dtype_is_float64(self) -> None:
        assert tg.regular_tetrahedron().dtype == np.float64


# --------------------------------------------------------------------------- #
# Separating axis theorem
# --------------------------------------------------------------------------- #
class TestOverlapDetection:
    def test_tetrahedron_overlaps_itself(self) -> None:
        v = tg.regular_tetrahedron()[None]
        assert tg.sat_overlap_depth(v, v)[0] > 0.0

    def test_distant_tetrahedra_are_separated(self) -> None:
        v = tg.regular_tetrahedron()[None]
        assert tg.sat_overlap_depth(v, v + np.array([10.0, 0.0, 0.0]))[0] <= 0.0

    def test_slight_offset_still_overlaps(self) -> None:
        v = tg.regular_tetrahedron()[None]
        assert tg.sat_overlap_depth(v, v + np.array([1e-3, 0.0, 0.0]))[0] > 0.0

    def test_face_sharing_dimer_is_contact_not_overlap(self) -> None:
        """The dimer motif must register as touching, never interpenetrating."""
        base = tg.regular_tetrahedron()
        quat, offset = tg._dimer_partner()
        rot = tg._quat_to_matrix(quat[None])[0]
        partner = (rot @ base.T).T + offset
        depth = tg.sat_overlap_depth(base[None], partner[None])[0]
        assert depth == pytest.approx(0.0, abs=1e-12)

    def test_dimer_partner_is_a_proper_rotation(self) -> None:
        quat, _ = tg._dimer_partner()
        rot = tg._quat_to_matrix(quat[None])[0]
        assert np.linalg.det(rot) == pytest.approx(1.0)
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)

    def test_dimer_partner_is_regular(self) -> None:
        base = tg.regular_tetrahedron()
        quat, offset = tg._dimer_partner()
        rot = tg._quat_to_matrix(quat[None])[0]
        partner = ((rot @ base.T).T + offset)[None]
        np.testing.assert_allclose(tg.edge_lengths(partner), 1.0, atol=1e-12)

    def test_batched_shapes_must_agree(self) -> None:
        v = tg.regular_tetrahedron()[None]
        with pytest.raises(ValueError):
            tg.sat_overlap_depth(v, np.zeros((2, 4, 3)))

    def test_empty_batch_returns_empty(self) -> None:
        assert tg.sat_overlap_depth(np.zeros((0, 4, 3)), np.zeros((0, 4, 3))).size == 0


# --------------------------------------------------------------------------- #
# Honeycomb backend
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def honeycomb() -> tg.TetraCloud:
    return tg.build_honeycomb(1000)


class TestHoneycomb:
    def test_meets_the_requested_size(self, honeycomb: tg.TetraCloud) -> None:
        assert honeycomb.n_tetrahedra >= 1000

    def test_every_tetrahedron_is_exactly_regular(self, honeycomb: tg.TetraCloud) -> None:
        honeycomb.assert_regular(edge=1.0, atol=1e-12)

    def test_no_tetrahedra_interpenetrate(self, honeycomb: tg.TetraCloud) -> None:
        assert tg.find_overlapping_pairs(honeycomb.tetrahedra).shape[0] == 0

    def test_is_reproducible(self) -> None:
        a = tg.build_honeycomb(200)
        b = tg.build_honeycomb(200)
        np.testing.assert_array_equal(a.tetrahedra, b.tetrahedra)

    def test_graph_is_connected_with_fcc_kissing_number(
        self, honeycomb: tg.TetraCloud
    ) -> None:
        merged = ts.merge_vertices(honeycomb.raw_points, atol=1e-5)
        bundle = ts.build_unit_distance_graph(merged.points)
        assert bundle.n_components == 1
        # No FCC site has more than 12 nearest neighbours.
        assert bundle.degrees.max() == 12

    def test_rejects_nonsense_size(self) -> None:
        with pytest.raises(ValueError):
            tg.build_honeycomb(0)

    def test_raises_when_target_is_unreachable(self) -> None:
        with pytest.raises(RuntimeError):
            tg.build_honeycomb(10**9, max_radius=4.0)


# --------------------------------------------------------------------------- #
# Packing backend
# --------------------------------------------------------------------------- #
class TestMotifs:
    def test_single_motif_is_one_tetrahedron(self) -> None:
        motif = tg.build_motif("single")
        assert motif.shape == (1, 4, 3)
        np.testing.assert_allclose(tg.edge_lengths(motif), 1.0, atol=1e-15)

    def test_dimer_motif_is_two_fused_tetrahedra(self) -> None:
        motif = tg.build_motif("dimer")
        assert motif.shape == (2, 4, 3)
        np.testing.assert_allclose(tg.edge_lengths(motif), 1.0, atol=1e-12)

    def test_dimer_motif_is_centred_on_its_centroid(self) -> None:
        motif = tg.build_motif("dimer")
        np.testing.assert_allclose(motif.reshape(-1, 3).mean(axis=0), 0.0, atol=1e-15)

    def test_dimer_halves_touch_without_overlapping(self) -> None:
        motif = tg.build_motif("dimer")
        assert tg.sat_overlap_depth(motif[0:1], motif[1:2])[0] == pytest.approx(0.0, abs=1e-12)

    def test_dimer_shares_exactly_three_vertices(self) -> None:
        """A face-fused pair shares a triangular face: 8 vertices become 5."""
        motif = tg.build_motif("dimer")
        merged = ts.merge_vertices(motif.reshape(-1, 3), atol=1e-9)
        assert merged.n_unique == 5

    def test_unknown_motif_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            tg.build_motif("octahedron")


class TestDensePacking:
    def test_search_yields_a_valid_certified_packing(self) -> None:
        cloud = tg.build_dense_packing(64, cycles=60, restarts=1, seed=3)
        cloud.assert_regular(edge=1.0, atol=1e-9)
        assert tg.find_overlapping_pairs(cloud.tetrahedra).shape[0] == 0

    def test_reports_a_physically_possible_density(self) -> None:
        cloud = tg.build_dense_packing(64, cycles=60, restarts=1, seed=3)
        phi = cloud.provenance["measured_packing_fraction"]
        assert isinstance(phi, float)
        assert 0.0 < phi < 1.0

    def test_restarts_never_worsen_the_result(self) -> None:
        one = tg.build_dense_packing(32, cycles=50, restarts=1, seed=11)
        many = tg.build_dense_packing(32, cycles=50, restarts=3, seed=11)
        assert (
            many.provenance["measured_packing_fraction"]
            >= one.provenance["measured_packing_fraction"]
        )

    def test_compression_is_monotone(self) -> None:
        """Volume must never increase: every lattice move is strictly densifying."""
        low = tg._asc_search(2, 40, seed=5)
        high = tg._asc_search(2, 200, seed=5)
        assert high.packing_fraction >= low.packing_fraction

    @pytest.mark.parametrize("motif, per_cell", [("single", 2), ("dimer", 4)])
    def test_cell_population_follows_the_motif(self, motif: str, per_cell: int) -> None:
        result = tg._asc_search(2, 30, seed=5, motif=motif)
        assert result.n_tetrahedra_per_cell == per_cell
        np.testing.assert_allclose(
            tg.edge_lengths(result.cell_tetrahedra), 1.0, atol=1e-12
        )

    def test_dimer_packing_is_certified_non_overlapping(self) -> None:
        cloud = tg.build_dense_packing(64, motif="dimer", cycles=80, restarts=1, seed=3)
        cloud.assert_regular(edge=1.0, atol=1e-9)
        assert tg.find_overlapping_pairs(cloud.tetrahedra).shape[0] == 0
        assert cloud.provenance["motif"] == "dimer"

    @pytest.mark.parametrize("kwargs", [{"cycles": 0}, {"restarts": 0}, {"n_particles": 0}])
    def test_rejects_invalid_parameters(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            tg.build_dense_packing(32, seed=1, **kwargs)

    def test_degenerate_cells_are_refused(self) -> None:
        """A cell too thin to certify must be rejected rather than under-tested."""
        flat = np.diag([1.0, 1.0, 1e-4])
        assert tg._periodic_setup(tg.regular_tetrahedron()[None], flat) is None


# --------------------------------------------------------------------------- #
# Vertex merging
# --------------------------------------------------------------------------- #
class TestMergeVertices:
    def test_exact_duplicates_collapse(self) -> None:
        pts = np.array([[0.0, 0, 0], [0, 0, 0], [1, 0, 0]])
        merged = ts.merge_vertices(pts, atol=1e-5)
        assert merged.n_unique == 2
        assert sorted(merged.cluster_sizes.tolist()) == [1, 2]

    def test_near_duplicates_within_tolerance_collapse(self) -> None:
        pts = np.array([[0.0, 0, 0], [1e-7, 0, 0], [1, 0, 0]])
        assert ts.merge_vertices(pts, atol=1e-5).n_unique == 2

    def test_points_outside_tolerance_stay_distinct(self) -> None:
        pts = np.array([[0.0, 0, 0], [1e-3, 0, 0]])
        assert ts.merge_vertices(pts, atol=1e-5).n_unique == 2

    def test_representative_is_the_cluster_centroid(self) -> None:
        pts = np.array([[0.0, 0, 0], [1e-6, 0, 0]])
        merged = ts.merge_vertices(pts, atol=1e-5)
        np.testing.assert_allclose(merged.points[0], [5e-7, 0, 0], atol=1e-15)

    def test_labels_index_back_into_the_merged_set(self) -> None:
        pts = np.array([[0.0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]])
        merged = ts.merge_vertices(pts, atol=1e-5)
        assert merged.labels.shape == (4,)
        assert merged.labels[0] == merged.labels[1]
        assert merged.labels[2] == merged.labels[3]
        assert merged.labels[0] != merged.labels[2]

    def test_honeycomb_merge_is_lossless_and_tight(self) -> None:
        cloud = tg.build_honeycomb(200)
        merged = ts.merge_vertices(cloud.raw_points, atol=1e-5)
        assert merged.n_raw == 4 * cloud.n_tetrahedra
        # Coincident vertices are bitwise identical here, so no chaining at all.
        assert merged.max_cluster_radius < 1e-12

    def test_chaining_is_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        # A chain of points each within atol of the next fuses transitively.
        pts = np.arange(50, dtype=float)[:, None] * 8e-6
        pts = np.pad(pts, ((0, 0), (0, 2)))
        with caplog.at_level("WARNING", logger="tetra_spectral"):
            merged = ts.merge_vertices(pts, atol=1e-5)
        assert merged.n_unique == 1
        assert "chained" in caplog.text

    @pytest.mark.parametrize(
        "bad_input, error",
        [
            (np.zeros((0, 3)), ValueError),
            (np.zeros((4, 2)), ValueError),
            (np.array([[np.nan, 0, 0]]), ValueError),
        ],
    )
    def test_rejects_bad_input(self, bad_input: np.ndarray, error: type) -> None:
        with pytest.raises(error):
            ts.merge_vertices(bad_input)

    def test_rejects_bad_tolerance(self) -> None:
        with pytest.raises(ValueError):
            ts.merge_vertices(np.zeros((2, 3)), atol=0.0)


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
class TestUnitDistanceGraph:
    def test_single_tetrahedron_gives_k4(self) -> None:
        bundle = ts.build_unit_distance_graph(tg.regular_tetrahedron())
        assert bundle.n_nodes == 4
        assert bundle.n_edges == 6
        assert bundle.n_components == 1
        np.testing.assert_array_equal(bundle.degrees, [3, 3, 3, 3])

    def test_two_separated_tetrahedra_give_two_components(self) -> None:
        a = tg.regular_tetrahedron()
        pts = np.vstack([a, a + np.array([50.0, 0, 0])])
        bundle = ts.build_unit_distance_graph(pts)
        assert bundle.n_components == 2
        assert bundle.n_edges == 12

    def test_non_unit_distances_produce_no_edges(self) -> None:
        pts = np.array([[0.0, 0, 0], [2.0, 0, 0], [4.0, 0, 0]])
        bundle = ts.build_unit_distance_graph(pts)
        assert bundle.n_edges == 0
        assert bundle.n_components == 3

    def test_adjacency_is_symmetric_with_zero_diagonal(self) -> None:
        bundle = ts.build_unit_distance_graph(tg.regular_tetrahedron())
        dense = bundle.adjacency.toarray()
        np.testing.assert_array_equal(dense, dense.T)
        np.testing.assert_array_equal(np.diag(dense), 0.0)

    def test_networkx_view_agrees_with_the_matrix(self) -> None:
        cloud = tg.build_honeycomb(200)
        merged = ts.merge_vertices(cloud.raw_points)
        bundle = ts.build_unit_distance_graph(merged.points)
        assert bundle.graph.number_of_nodes() == bundle.n_nodes
        assert bundle.graph.number_of_edges() == bundle.n_edges
        assert nx.number_connected_components(bundle.graph) == bundle.n_components

    def test_tolerance_must_be_below_the_edge_length(self) -> None:
        with pytest.raises(ValueError):
            ts.build_unit_distance_graph(tg.regular_tetrahedron(), atol=1.5)


# --------------------------------------------------------------------------- #
# Laplacian
# --------------------------------------------------------------------------- #
class TestLaplacian:
    def test_rows_sum_to_zero(self) -> None:
        bundle = ts.build_unit_distance_graph(tg.regular_tetrahedron())
        laplacian = ts.graph_laplacian(bundle.adjacency)
        np.testing.assert_allclose(laplacian.toarray().sum(axis=1), 0.0, atol=1e-14)

    def test_equals_degree_minus_adjacency(self) -> None:
        cloud = tg.build_honeycomb(200)
        merged = ts.merge_vertices(cloud.raw_points)
        bundle = ts.build_unit_distance_graph(merged.points)
        laplacian = ts.graph_laplacian(bundle.adjacency).toarray()
        adjacency = bundle.adjacency.toarray()
        np.testing.assert_allclose(laplacian, np.diag(adjacency.sum(axis=1)) - adjacency)

    def test_is_positive_semidefinite(self) -> None:
        bundle = ts.build_unit_distance_graph(tg.regular_tetrahedron())
        values = np.linalg.eigvalsh(ts.graph_laplacian(bundle.adjacency).toarray())
        assert values.min() > -1e-12

    def test_rejects_dense_input(self) -> None:
        with pytest.raises(TypeError):
            ts.graph_laplacian(np.eye(3))  # type: ignore[arg-type]

    def test_rejects_non_square_input(self) -> None:
        with pytest.raises(ValueError):
            ts.graph_laplacian(csr_matrix((3, 4)))


# --------------------------------------------------------------------------- #
# Eigensolver
# --------------------------------------------------------------------------- #
class TestSmallestEigenvalues:
    def test_k4_spectrum_is_exactly_zero_four_four_four(self) -> None:
        """K4's Laplacian spectrum is {0, 4, 4, 4}; anything else is a bug."""
        bundle = ts.build_unit_distance_graph(tg.regular_tetrahedron())
        spectrum = ts.smallest_eigenvalues(
            ts.graph_laplacian(bundle.adjacency), k=4, n_components=1
        )
        np.testing.assert_allclose(spectrum.eigenvalues, [0.0, 4.0, 4.0, 4.0], atol=1e-12)
        assert spectrum.zero_multiplicity == 1
        assert spectrum.algebraic_connectivity == pytest.approx(4.0)

    def test_kernel_dimension_equals_component_count(self) -> None:
        a = tg.regular_tetrahedron()
        pts = np.vstack([a, a + np.array([50.0, 0, 0]), a + np.array([0.0, 50.0, 0])])
        bundle = ts.build_unit_distance_graph(pts)
        spectrum = ts.smallest_eigenvalues(
            ts.graph_laplacian(bundle.adjacency), k=6, n_components=bundle.n_components
        )
        assert bundle.n_components == 3
        assert spectrum.zero_multiplicity == 3
        assert spectrum.algebraic_connectivity == 0.0

    def test_disconnected_graph_has_zero_fiedler_value(self) -> None:
        a = tg.regular_tetrahedron()
        bundle = ts.build_unit_distance_graph(np.vstack([a, a + 50.0]))
        spectrum = ts.smallest_eigenvalues(
            ts.graph_laplacian(bundle.adjacency), k=5, n_components=2
        )
        assert spectrum.algebraic_connectivity == 0.0
        assert spectrum.first_positive == pytest.approx(4.0)

    def test_eigenvalues_are_sorted_and_nonnegative(self) -> None:
        cloud = tg.build_honeycomb(200)
        merged = ts.merge_vertices(cloud.raw_points)
        bundle = ts.build_unit_distance_graph(merged.points)
        spectrum = ts.smallest_eigenvalues(ts.graph_laplacian(bundle.adjacency), k=30)
        assert np.all(np.diff(spectrum.eigenvalues) >= -1e-12)
        assert spectrum.eigenvalues.min() >= 0.0

    def test_sparse_and_dense_paths_agree(self) -> None:
        """ARPACK shift-invert must reproduce what dense LAPACK computes."""
        cloud = tg.build_honeycomb(1000)
        merged = ts.merge_vertices(cloud.raw_points)
        bundle = ts.build_unit_distance_graph(merged.points)
        laplacian = ts.graph_laplacian(bundle.adjacency)

        dense = ts.smallest_eigenvalues(laplacian, k=40, dense_threshold=10**9)
        sparse = ts.smallest_eigenvalues(laplacian, k=40, dense_threshold=0)
        assert dense.method == "dense"
        assert sparse.method == "shift-invert"
        np.testing.assert_allclose(dense.eigenvalues, sparse.eigenvalues, atol=1e-8)

    def test_matrix_dimension_is_recorded(self) -> None:
        bundle = ts.build_unit_distance_graph(tg.regular_tetrahedron())
        spectrum = ts.smallest_eigenvalues(ts.graph_laplacian(bundle.adjacency), k=2)
        assert spectrum.matrix_dimension == 4

    def test_k_is_clamped_to_matrix_dimension(self) -> None:
        bundle = ts.build_unit_distance_graph(tg.regular_tetrahedron())
        spectrum = ts.smallest_eigenvalues(ts.graph_laplacian(bundle.adjacency), k=99)
        assert spectrum.k_returned == 4

    def test_arpack_requires_k_below_dimension(self) -> None:
        bundle = ts.build_unit_distance_graph(tg.regular_tetrahedron())
        with pytest.raises(ValueError):
            ts.smallest_eigenvalues(
                ts.graph_laplacian(bundle.adjacency), k=4, dense_threshold=0
            )

    @pytest.mark.parametrize("bad_k", [0, -3])
    def test_rejects_invalid_k(self, bad_k: int) -> None:
        bundle = ts.build_unit_distance_graph(tg.regular_tetrahedron())
        with pytest.raises(ValueError):
            ts.smallest_eigenvalues(ts.graph_laplacian(bundle.adjacency), k=bad_k)

    def test_rejects_dense_input(self) -> None:
        with pytest.raises(TypeError):
            ts.smallest_eigenvalues(np.eye(4), k=2)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Degeneracy analysis
# --------------------------------------------------------------------------- #
class TestDegeneracyAnalysis:
    def test_groups_exactly_repeated_values(self) -> None:
        report = ts.analyse_degeneracies(np.array([0.0, 4.0, 4.0, 4.0]))
        assert report.n_levels == 2
        np.testing.assert_array_equal(report.multiplicities, [1, 3])
        assert report.multiplicity_histogram == {1: 1, 3: 1}

    def test_separates_values_beyond_tolerance(self) -> None:
        report = ts.analyse_degeneracies(np.array([1.0, 1.5, 2.0]), rtol=1e-8, atol=1e-9)
        assert report.n_levels == 3
        assert report.max_multiplicity == 1

    def test_merges_values_inside_tolerance(self) -> None:
        report = ts.analyse_degeneracies(np.array([1.0, 1.0 + 1e-12, 2.0]))
        assert report.n_levels == 2

    def test_raw_spacings_are_consecutive_differences(self) -> None:
        report = ts.analyse_degeneracies(np.array([0.0, 1.0, 3.0, 6.0]))
        np.testing.assert_allclose(report.raw_spacings, [1.0, 2.0, 3.0])

    def test_level_spacings_skip_degenerate_repeats(self) -> None:
        report = ts.analyse_degeneracies(np.array([0.0, 1.0, 1.0, 1.0, 4.0]))
        np.testing.assert_allclose(report.level_spacings, [1.0, 3.0])

    def test_degenerate_fraction(self) -> None:
        report = ts.analyse_degeneracies(np.array([0.0, 4.0, 4.0, 4.0]))
        assert report.degenerate_fraction == pytest.approx(0.75)

    def test_uniform_spacing_gives_unit_gap_ratio(self) -> None:
        report = ts.analyse_degeneracies(np.arange(10, dtype=float))
        assert report.gap_ratio_mean == pytest.approx(1.0)

    def test_gap_ratio_is_none_without_enough_levels(self) -> None:
        assert ts.analyse_degeneracies(np.array([0.0, 1.0])).gap_ratio_mean is None

    def test_gap_ratio_is_suppressed_on_too_few_samples(self) -> None:
        """A mean over two spacings carries no distributional meaning."""
        report = ts.analyse_degeneracies(np.array([0.0, 3.0, 5.0]))
        assert report.n_levels == 3
        assert report.gap_ratio_mean is None

    def test_gap_ratio_threshold_is_configurable(self) -> None:
        report = ts.analyse_degeneracies(
            np.array([0.0, 3.0, 5.0]), min_ratio_samples=1
        )
        assert report.gap_ratio_mean == pytest.approx(2.0 / 3.0)

    def test_rejects_empty_spectrum(self) -> None:
        with pytest.raises(ValueError):
            ts.analyse_degeneracies(np.zeros(0))

    def test_unsorted_input_is_sorted(self) -> None:
        report = ts.analyse_degeneracies(np.array([4.0, 0.0, 4.0, 4.0]))
        np.testing.assert_allclose(report.levels, [0.0, 4.0])


# --------------------------------------------------------------------------- #
# The physics claim: O_h degeneracy structure
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def report() -> ts.DegeneracyReport:
    """Degeneracy structure of the low spectrum of a 1000+ tetrahedron honeycomb."""
    cloud = tg.build_honeycomb(1000)
    merged = ts.merge_vertices(cloud.raw_points, atol=1e-5)
    bundle = ts.build_unit_distance_graph(merged.points)
    spectrum = ts.smallest_eigenvalues(
        ts.graph_laplacian(bundle.adjacency), k=100, n_components=bundle.n_components
    )
    return ts.analyse_degeneracies(spectrum.eigenvalues)


class TestOctahedralFingerprint:
    """The honeycomb cluster's multiplicities must obey O_h representation theory."""

    def test_multiplicities_never_exceed_the_largest_oh_irrep(
        self, report: ts.DegeneracyReport
    ) -> None:
        # O_h has irreps of dimension 1, 1, 2, 3, 3. Nothing may exceed 3
        # unless distinct irreps accidentally coincide.
        assert report.max_multiplicity <= 3

    def test_every_multiplicity_is_an_oh_irrep_dimension(
        self, report: ts.DegeneracyReport
    ) -> None:
        assert set(report.multiplicity_histogram) <= {1, 2, 3}

    def test_spectrum_is_substantially_degenerate(
        self, report: ts.DegeneracyReport
    ) -> None:
        assert report.degenerate_fraction > 0.5

    def test_triply_degenerate_levels_dominate(self, report: ts.DegeneracyReport) -> None:
        # O_h has two 3-dimensional irreps out of five, and they carry the most
        # states, so triplets should be the single most common multiplicity.
        histogram = report.multiplicity_histogram
        assert histogram.get(3, 0) == max(histogram.values())


# --------------------------------------------------------------------------- #
# Packings have no fingerprint: the contrast case
# --------------------------------------------------------------------------- #
class TestPackingHasNoSharedVertices:
    def test_dense_packing_shatters_into_disjoint_k4s(self) -> None:
        """Generic-position tetrahedra share no vertices, so the graph fragments."""
        cloud = tg.build_dense_packing(
            64, motif="single", cycles=60, restarts=1, seed=3
        )
        merged = ts.merge_vertices(cloud.raw_points, atol=1e-5)
        bundle = ts.build_unit_distance_graph(merged.points)

        assert bundle.n_components == cloud.n_tetrahedra
        np.testing.assert_array_equal(bundle.degrees, 3)

        spectrum = ts.smallest_eigenvalues(
            ts.graph_laplacian(bundle.adjacency),
            k=8,
            n_components=bundle.n_components,
            component_labels=bundle.component_labels,
        )
        # Algebraic connectivity of a disconnected graph is exactly zero.
        assert spectrum.algebraic_connectivity == 0.0


# --------------------------------------------------------------------------- #
# The Chen-Engel-Glotzer optimum
# --------------------------------------------------------------------------- #
class TestCEGPacking:
    """The densest known tetrahedron packing, phi = 4000/4671."""

    def test_unit_cell_reproduces_the_published_density(self) -> None:
        cell, lattice = tg.ceg_unit_cell()
        assert cell.shape == (4, 4, 3)
        volume = abs(float(np.linalg.det(lattice)))
        phi = 4 * tg.UNIT_TETRA_VOLUME / volume
        assert phi == pytest.approx(4000 / 4671, abs=1e-15)

    def test_unit_cell_matches_the_papers_lattice_volume(self) -> None:
        """Paper reports V = 2 det[a,b,c] = 42039/1000 at edge 3*sqrt(2)."""
        _, lattice = tg.ceg_unit_cell()
        rescaled = abs(float(np.linalg.det(lattice))) * (3 * math.sqrt(2)) ** 3
        assert rescaled == pytest.approx(42039 / 1000, rel=1e-12)

    def test_unit_cell_tetrahedra_are_regular_with_unit_edge(self) -> None:
        cell, _ = tg.ceg_unit_cell()
        np.testing.assert_allclose(tg.edge_lengths(cell), 1.0, atol=1e-12)

    def test_unit_cell_is_overlap_free_under_periodicity(self) -> None:
        cell, lattice = tg.ceg_unit_cell()
        assert tg._configuration_is_valid(cell, lattice)

    def test_tiled_packing_has_no_overlaps(self) -> None:
        cloud = tg.build_ceg_packing(200)
        assert tg.find_overlapping_pairs(cloud.tetrahedra).shape[0] == 0

    def test_dimers_share_faces_so_components_are_dipyramids(self) -> None:
        """Face contact inside a dimer is exact vertex sharing: 8 -> 5 vertices."""
        cloud = tg.build_ceg_packing(200)
        merged = ts.merge_vertices(cloud.raw_points, atol=1e-5)
        bundle = ts.build_unit_distance_graph(merged.points)

        assert bundle.n_components == cloud.n_tetrahedra // 2
        assert merged.n_unique == bundle.n_components * 5
        assert bundle.n_edges == bundle.n_components * 9
        assert int(merged.cluster_sizes.max()) == 2
        # A triangular dipyramid has two apexes (degree 3) and three equatorial
        # vertices (degree 4).
        assert sorted(np.bincount(bundle.degrees)[3:].tolist()) == sorted(
            [2 * bundle.n_components, 3 * bundle.n_components]
        )

    def test_spectrum_is_exactly_that_of_k5_minus_an_edge(self) -> None:
        """K5 minus an edge has Laplacian spectrum {0, 3, 5, 5, 5}."""
        cloud = tg.build_ceg_packing(200)
        merged = ts.merge_vertices(cloud.raw_points, atol=1e-5)
        bundle = ts.build_unit_distance_graph(merged.points)
        spectrum = ts.smallest_eigenvalues(
            ts.graph_laplacian(bundle.adjacency),
            k=merged.n_unique,
            n_components=bundle.n_components,
            component_labels=bundle.component_labels,
        )
        report = ts.analyse_degeneracies(spectrum.eigenvalues)

        np.testing.assert_allclose(report.levels, [0.0, 3.0, 5.0], atol=1e-9)
        n = bundle.n_components
        np.testing.assert_array_equal(report.multiplicities, [n, n, 3 * n])
        assert spectrum.algebraic_connectivity == 0.0
        assert spectrum.first_positive == pytest.approx(3.0)

    def test_kernel_dimension_equals_dimer_count(self) -> None:
        cloud = tg.build_ceg_packing(200)
        merged = ts.merge_vertices(cloud.raw_points, atol=1e-5)
        bundle = ts.build_unit_distance_graph(merged.points)
        spectrum = ts.smallest_eigenvalues(
            ts.graph_laplacian(bundle.adjacency),
            k=merged.n_unique,
            n_components=bundle.n_components,
            component_labels=bundle.component_labels,
        )
        assert spectrum.zero_multiplicity == bundle.n_components
        assert spectrum.method == "block-diagonal"

    def test_is_reproducible(self) -> None:
        np.testing.assert_array_equal(
            tg.build_ceg_packing(200).tetrahedra, tg.build_ceg_packing(200).tetrahedra
        )

    def test_rejects_nonsense_size(self) -> None:
        with pytest.raises(ValueError):
            tg.build_ceg_packing(0)


class TestCEGFamily:
    """The three-parameter double dimer family of eq. (6)."""

    @pytest.mark.parametrize("variant", sorted(tg.CEG_FAMILY_PRESETS))
    def test_presets_reproduce_their_published_density(self, variant: str) -> None:
        cloud = tg.build_ceg_packing(64, variant=variant)
        expected = tg.CEG_PRESET_FRACTIONS[variant]
        assert cloud.provenance["packing_fraction"] == pytest.approx(
            float(expected), abs=1e-14
        )
        assert cloud.provenance["packing_fraction_exact"] == str(expected)

    @pytest.mark.parametrize("variant", sorted(tg.CEG_FAMILY_PRESETS))
    def test_presets_are_certified_packings(self, variant: str) -> None:
        cloud = tg.build_ceg_packing(64, variant=variant)
        assert tg.find_overlapping_pairs(cloud.tetrahedra).shape[0] == 0
        np.testing.assert_allclose(tg.edge_lengths(cloud.tetrahedra), 1.0, atol=1e-12)

    @pytest.mark.parametrize("variant", sorted(tg.CEG_FAMILY_PRESETS))
    def test_presets_lie_in_the_restricted_space(self, variant: str) -> None:
        assert tg.ceg_in_restricted_space(*tg.CEG_FAMILY_PRESETS[variant])

    def test_family_reproduces_theorem_1_vectors(self) -> None:
        """eq. (6) at (3/160, 3/64, 0) must equal the printed optimum."""
        a, b, c, d = tg.ceg_family_vectors(Fraction(3, 160), Fraction(3, 64), Fraction(0))
        np.testing.assert_allclose(a, 3 / 320 * np.array([290, 107, -7]), atol=1e-15)
        np.testing.assert_allclose(b, 3 / 320 * np.array([-34, 277, 135]), atol=1e-15)
        np.testing.assert_allclose(c, 3 / 320 * np.array([94, -83, 247]), atol=1e-15)
        np.testing.assert_allclose(d, 1 / 320 * np.array([38, 5, -25]), atol=1e-15)

    def test_closed_form_density_matches_geometry(self) -> None:
        for u, v, w in [
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(3, 160), Fraction(3, 64), Fraction(0)),
            (Fraction(1, 200), Fraction(-1, 100), Fraction(1, 320)),
        ]:
            _, lattice = tg.ceg_unit_cell(u, v, w)
            geometric = 4 * tg.UNIT_TETRA_VOLUME / abs(float(np.linalg.det(lattice)))
            assert geometric == pytest.approx(float(tg.ceg_packing_fraction(u, v, w)), abs=1e-14)

    def test_density_is_independent_of_w(self) -> None:
        """w is a pure lattice shear, so it cannot change the density."""
        base = tg.ceg_packing_fraction(Fraction(3, 160), Fraction(3, 64), Fraction(0))
        for w in (Fraction(-1, 32), Fraction(1, 64), Fraction(1, 32)):
            assert tg.ceg_packing_fraction(Fraction(3, 160), Fraction(3, 64), w) == base

    def test_densest_connected_matches_the_papers_central_point(self) -> None:
        """Maximising |v| on the connected plane u=0 lands on C3+cen, 125/146."""
        u, v, w = tg.CEG_FAMILY_PRESETS["densest-connected"]
        assert u == 0
        assert v == Fraction(1, 20)
        assert tg.ceg_packing_fraction(u, v, w) == Fraction(125, 146)
        assert tg.ceg_in_restricted_space(u, v, w)

    def test_v_cannot_exceed_one_twentieth_on_the_connected_plane(self) -> None:
        """P'' constraints 2v - w <= 33/320 and v + w <= 3/64 cap v at 1/20."""
        just_over = Fraction(1, 20) + Fraction(1, 1000)
        assert not any(
            tg.ceg_in_restricted_space(Fraction(0), just_over, Fraction(k, 3200))
            for k in range(-400, 401)
        )

    def test_connectivity_costs_a_known_exact_amount(self) -> None:
        deficit = Fraction(4000, 4671) - Fraction(125, 146)
        assert deficit == Fraction(125, 681966)
        assert float(deficit) == pytest.approx(1.8329e-4, rel=1e-3)

    def test_optimal_is_the_densest_preset(self) -> None:
        best = max(tg.CEG_PRESET_FRACTIONS.values())
        assert best == Fraction(4000, 4671)

    def test_origin_is_the_kallus_elser_gravel_packing(self) -> None:
        assert tg.ceg_packing_fraction(Fraction(0), Fraction(0), Fraction(0)) == Fraction(100, 117)

    def test_restricted_space_rejects_far_exterior_points(self) -> None:
        assert not tg.ceg_in_restricted_space(Fraction(0), Fraction(1), Fraction(0))

    def test_rejects_unknown_variant(self) -> None:
        with pytest.raises(ValueError):
            tg.build_ceg_packing(32, variant="nonexistent")

    def test_rejects_partial_parameters(self) -> None:
        with pytest.raises(ValueError):
            tg.ceg_unit_cell(Fraction(0), Fraction(0))


def _plane_w(v: Fraction) -> Fraction:
    """A w keeping (0, v, w) inside P''; density is independent of it."""
    return min(Fraction(0), Fraction(3, 64) - abs(v))


class TestFamilyConnectivityContrast:
    """Density and graph connectivity are traded off within the family."""

    @staticmethod
    def _graph(variant: str, reps: int):
        cloud = tg.build_ceg_packing(4 * reps**3, variant=variant)
        assert cloud.provenance["replicas_per_axis"] == reps
        merged = ts.merge_vertices(cloud.raw_points, atol=1e-5)
        return cloud, merged, ts.build_unit_distance_graph(merged.points)

    def test_inter_dimer_contacts_exist_only_on_the_plane_u_equals_zero(self) -> None:
        """Rediscovers the paper's H_{a-b} condition (eq. 10) from the graph."""
        from scipy.spatial import cKDTree

        def inter_dimer_contacts(u, v, w) -> int:
            cloud = tg.build_ceg_packing(32, u=u, v=v, w=w)
            pts = cloud.raw_points
            dimer = np.repeat(np.arange(cloud.n_tetrahedra // 2), 8)
            pairs = cKDTree(pts).query_pairs(r=1.0 + 1e-12, output_type="ndarray")
            d = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
            unit = pairs[np.abs(d - 1.0) < 1e-12]
            return int((dimer[unit[:, 0]] != dimer[unit[:, 1]]).sum())

        # On the plane u = 0 the contacts are present for any v and any w.
        for v in (Fraction(0), Fraction(1, 64), Fraction(1, 20)):
            assert inter_dimer_contacts(Fraction(0), v, _plane_w(v)) > 0
        # Off it, they vanish -- even a thousandth of the way to the optimum.
        for u in (Fraction(3, 160000), Fraction(3, 1600), Fraction(3, 160)):
            assert inter_dimer_contacts(u, Fraction(0), Fraction(0)) == 0

    @pytest.mark.parametrize("variant", ["optimal", "torquato-jiao"])
    @pytest.mark.parametrize("reps", [2, 3, 4])
    def test_dimers_stay_isolated_at_every_size(self, variant: str, reps: int) -> None:
        cloud, merged, bundle = self._graph(variant, reps)
        assert bundle.n_components == cloud.n_tetrahedra // 2
        assert bundle.degrees.max() == 4
        assert np.bincount(bundle.component_labels).max() == 5

    @pytest.mark.parametrize("reps", [2, 3, 4, 5])
    def test_keg_components_scale_linearly_with_the_box(self, reps: int) -> None:
        """The less dense KEG packing links dimers; components grow with size."""
        cloud, _, bundle = self._graph("kallus-elser-gravel", reps)
        assert bundle.n_components == 12 * (reps - 1)
        # Dimer count is cubic in reps while the component count is linear, so
        # components must be extended rather than one-per-dimer.
        assert bundle.n_components < cloud.n_tetrahedra // 2
        assert bundle.degrees.max() > 4


class TestBlockDiagonalSolver:
    """Krylov methods cannot resolve a hugely degenerate kernel; blocks can."""

    @staticmethod
    def _disjoint_tetrahedra(count: int) -> ts.GraphBundle:
        base = tg.regular_tetrahedron()
        pts = np.vstack([base + np.array([50.0 * i, 0, 0]) for i in range(count)])
        return ts.build_unit_distance_graph(pts)

    def test_block_path_finds_the_whole_kernel(self) -> None:
        bundle = self._disjoint_tetrahedra(40)
        spectrum = ts.smallest_eigenvalues(
            ts.graph_laplacian(bundle.adjacency),
            k=40,
            n_components=bundle.n_components,
            component_labels=bundle.component_labels,
        )
        assert spectrum.method == "block-diagonal"
        assert spectrum.zero_multiplicity == 40
        np.testing.assert_allclose(spectrum.eigenvalues, 0.0, atol=1e-12)

    def test_block_path_agrees_with_dense_on_the_full_spectrum(self) -> None:
        bundle = self._disjoint_tetrahedra(12)
        laplacian = ts.graph_laplacian(bundle.adjacency)
        n = laplacian.shape[0]
        blocked = ts.smallest_eigenvalues(
            laplacian,
            k=n,
            n_components=bundle.n_components,
            component_labels=bundle.component_labels,
        )
        dense = np.sort(np.linalg.eigvalsh(laplacian.toarray()))
        np.testing.assert_allclose(blocked.eigenvalues, dense, atol=1e-10)

    def test_connected_graph_skips_the_block_path(self) -> None:
        cloud = tg.build_honeycomb(200)
        merged = ts.merge_vertices(cloud.raw_points)
        bundle = ts.build_unit_distance_graph(merged.points)
        spectrum = ts.smallest_eigenvalues(
            ts.graph_laplacian(bundle.adjacency),
            k=20,
            n_components=bundle.n_components,
            component_labels=bundle.component_labels,
        )
        assert bundle.n_components == 1
        assert spectrum.method != "block-diagonal"


# --------------------------------------------------------------------------- #
# Tabular output
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pieces() -> tuple[ts.Spectrum, ts.DegeneracyReport]:
    """Spectrum and degeneracy report for a single tetrahedron (K4)."""
    bundle = ts.build_unit_distance_graph(tg.regular_tetrahedron())
    spectrum = ts.smallest_eigenvalues(
        ts.graph_laplacian(bundle.adjacency), k=4, n_components=1
    )
    return spectrum, ts.analyse_degeneracies(spectrum.eigenvalues)


class TestFrames:
    def test_spectrum_frame_columns(self, pieces) -> None:
        frame = ts.spectrum_frame(pieces[0])
        assert list(frame.columns) == ["n", "lambda_n", "delta_lambda_n", "is_zero"]
        assert len(frame) == 4
        assert bool(frame["is_zero"].iloc[0])

    def test_spectrum_frame_respects_limit(self, pieces) -> None:
        assert len(ts.spectrum_frame(pieces[0], limit=2)) == 2

    def test_level_frame_reports_multiplicity(self, pieces) -> None:
        frame = ts.level_frame(pieces[1])
        assert frame["multiplicity"].tolist() == [1, 3]

    def test_multiplicity_frame_flags_oh_dimensions(self, pieces) -> None:
        frame = ts.multiplicity_frame(pieces[1])
        assert bool(frame["matches_O_h_irrep_dim"].all())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class TestCommandLine:
    def test_ceg_backend_runs_end_to_end(self, tmp_path) -> None:
        code = cli.main(
            [
                "--backend", "ceg",
                "--min-tetrahedra", "200",
                "--num-eigenvalues", "40",
                "--log-level", "ERROR",
                "--json-summary", str(tmp_path / "ceg.json"),
            ]
        )
        assert code == 0
        import json

        payload = json.loads((tmp_path / "ceg.json").read_text())
        assert payload["geometry"]["provenance"]["packing_fraction_exact"] == "4000/4671"
        assert payload["spectrum"]["algebraic_connectivity"] == 0.0

    def test_end_to_end_run_succeeds(self, tmp_path) -> None:
        code = cli.main(
            [
                "--min-tetrahedra", "200",
                "--num-eigenvalues", "20",
                "--top", "10",
                "--log-level", "ERROR",
                "--csv-prefix", str(tmp_path / "run"),
                "--json-summary", str(tmp_path / "summary.json"),
            ]
        )
        assert code == 0
        assert (tmp_path / "run_eigenvalues.csv").exists()
        assert (tmp_path / "run_levels.csv").exists()
        assert (tmp_path / "run_multiplicities.csv").exists()
        assert (tmp_path / "summary.json").exists()

    def test_json_summary_carries_the_first_twenty_eigenvalues(self, tmp_path) -> None:
        import json

        path = tmp_path / "summary.json"
        cli.main(
            [
                "--min-tetrahedra", "200",
                "--num-eigenvalues", "25",
                "--log-level", "ERROR",
                "--json-summary", str(path),
            ]
        )
        payload = json.loads(path.read_text())
        assert len(payload["spectrum"]["first_20_eigenvalues"]) == 20
        assert payload["spectrum"]["lambda_0"] == pytest.approx(0.0, abs=1e-9)
        assert payload["graph"]["n_components"] == 1

    @pytest.mark.parametrize(
        "argv",
        [
            ["--min-tetrahedra", "0"],
            ["--num-eigenvalues", "0"],
            ["--merge-atol", "0"],
            ["--edge-atol", "2.0"],
            ["--asc-restarts", "0"],
        ],
    )
    def test_invalid_arguments_are_rejected(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit):
            cli.main(argv)

    def test_module_runs_as_a_script(self) -> None:
        completed = subprocess.run(
            [sys.executable, "tetra_spectral_analysis.py",
             "--min-tetrahedra", "100", "--num-eigenvalues", "10",
             "--log-level", "ERROR"],
            capture_output=True, text=True, timeout=300,
        )
        assert completed.returncode == 0
        assert "LAPLACIAN SPECTRUM" in completed.stdout
        assert "lambda_0 == 0 detected" in completed.stdout


# --------------------------------------------------------------------------- #
# Regression: unwrapped particles defeated the periodic overlap test
# --------------------------------------------------------------------------- #
class TestPeriodicOverlapRegression:
    """A particle drifting out of the cell once hid overlaps entirely.

    The image-range search is derived from the lattice widths and is only valid
    for particles inside the cell.  Fractional coordinates were never wrapped,
    so a random walk could carry a particle several cells away; its true
    periodic neighbours then fell outside the enumerated range and were never
    tested.  Seed 1007 at N=3 reported phi = 0.982 -- an "impossible" density
    that an independent tiling check showed to be 581 overlapping pairs with a
    penetration depth of 0.457.
    """

    def test_particles_stay_inside_the_cell(self) -> None:
        result = tg._asc_search(3, 300, seed=1007, motif="single")
        inverse = np.linalg.inv(result.lattice)
        fractional = result.cell_tetrahedra.mean(axis=1) @ inverse
        assert np.all(fractional >= -1e-9)
        assert np.all(fractional <= 1.0 + 1e-9)

    @pytest.mark.parametrize("seed", [7, 1007])
    def test_search_results_survive_an_independent_tiling_check(self, seed: int) -> None:
        """Verify by assembling the cloud, not by reasoning about image ranges."""
        result = tg._asc_search(3, 300, seed=seed, motif="single")
        grid = np.arange(5.0)
        i, j, k = np.meshgrid(grid, grid, grid, indexing="ij")
        shifts = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) @ result.lattice
        tiled = (result.cell_tetrahedra[None] + shifts[:, None, None, :]).reshape(-1, 4, 3)
        assert tg.find_overlapping_pairs(tiled).shape[0] == 0

    def test_image_range_covers_the_fractional_offset(self) -> None:
        """Spans must exceed reach/width, since the box is centred on a rounded shift."""
        lattice = np.eye(3) * 0.8
        setup = tg._periodic_setup(tg.regular_tetrahedron()[None], lattice)
        assert setup is not None
        _, spans, reach = setup
        assert min(spans) >= math.ceil(reach / 0.8)

    def test_a_modest_lattice_shift_preserves_validity(self) -> None:
        """Translating by a lattice vector leaves the periodic packing identical."""
        result = tg._asc_search(3, 200, seed=11, motif="single")
        lattice = result.lattice
        shifted = result.cell_tetrahedra.copy()
        shifted[0] += lattice[0]
        assert tg._configuration_is_valid(result.cell_tetrahedra, lattice)
        assert tg._configuration_is_valid(shifted, lattice)

    @pytest.mark.parametrize("shift", [(1, 0, 0), (8, 0, -6), (-13, 7, 4)])
    def test_validity_is_invariant_under_any_lattice_translation(self, shift) -> None:
        """Moving a particle by whole lattice vectors is the same packing.

        Per-pair image boxes are centred on the shift each pair actually needs,
        so validity no longer depends on how far apart the cell's contents are.
        The earlier global-box design could not certify a spread-out cell at
        all, and an earlier one silently mis-certified it: a trimer of radius 10
        in a cell of size 0.7 reported phi = 0.99.
        """
        result = tg._asc_search(3, 200, seed=11, motif="single")
        lattice = result.lattice
        shifted = result.cell_tetrahedra.copy()
        shifted[0] += shift[0] * lattice[0] + shift[1] * lattice[1] + shift[2] * lattice[2]
        assert tg._configuration_is_valid(result.cell_tetrahedra, lattice)
        assert tg._configuration_is_valid(shifted, lattice)

    def test_compiled_and_array_paths_agree(self) -> None:
        """The numba kernel and the NumPy fallback must give identical verdicts."""
        rng = np.random.default_rng(5)
        checked = valid = 0
        for _ in range(120):
            n = int(rng.integers(1, 5))
            lattice = np.eye(3) * float(rng.uniform(0.7, 2.4)) + rng.normal(scale=0.2, size=(3, 3))
            if abs(np.linalg.det(lattice)) < 0.25:
                continue
            verts = tg._cell_vertices(
                tg.build_motif("single"), lattice, rng.random((n, 3)),
                tg._random_quaternions(rng, n),
            )
            setup = tg._periodic_setup(verts, lattice)
            if setup is None:
                continue
            inverse, spans, reach = setup
            array_path = tg._configuration_is_valid_numpy(
                verts, lattice, inverse, spans, reach, 1e-12
            )
            if tf.HAVE_NUMBA:
                compiled = tf.configuration_is_valid(
                    verts, lattice, inverse, spans, 1e-12, reach
                )
                assert compiled == array_path
            checked += 1
            valid += int(array_path)
        assert checked > 40
        assert 0 < valid < checked  # both outcomes exercised

    def test_builder_rejects_an_overlapping_search_result(self, monkeypatch) -> None:
        """build_dense_packing must refuse to return a non-packing."""
        good = tg._asc_search(2, 120, seed=5, motif="single")
        broken = tg.PackingResult(
            lattice=good.lattice * 0.35,      # far too small to hold the particles
            cell_tetrahedra=good.cell_tetrahedra,
            packing_fraction=9.9,
            accepted_moves=1,
            attempted_moves=1,
            n_particles=2,
            motif="single",
        )
        monkeypatch.setattr(tg, "_asc_search", lambda *a, **k: broken)
        with pytest.raises(AssertionError, match="overlapping"):
            tg.build_dense_packing(16, motif="single", cycles=10, restarts=1, seed=5)


# --------------------------------------------------------------------------- #
# Higher-dimensional lift obstructions
# --------------------------------------------------------------------------- #
class TestZModuleRank:
    """The rank test must be able to detect a real lift, not only deny one."""

    def test_cubic_lattice_has_rank_three(self) -> None:
        points = [[Fraction(x), Fraction(y), Fraction(z)]
                  for x in range(3) for y in range(3) for z in range(3)]
        result = tl.zmodule_rank(tl.rational_coefficients(points))
        assert result.rank == 3
        assert not result.admits_projection_lift

    def test_planar_set_has_rank_two(self) -> None:
        points = [[Fraction(x), Fraction(y), Fraction(0)] for x in range(4) for y in range(4)]
        assert tl.zmodule_rank(tl.rational_coefficients(points)).rank == 2

    def test_collinear_set_has_rank_one(self) -> None:
        points = [[Fraction(k), Fraction(0), Fraction(0)] for k in range(5)]
        assert tl.zmodule_rank(tl.rational_coefficients(points)).rank == 1

    def test_icosahedral_quasicrystal_has_rank_six(self) -> None:
        """The positive control: a genuine cut-and-project set must show rank 6.

        Icosahedron vertices are (0, +-1, +-phi) and cyclic permutations. Over
        the basis [1, phi] each coordinate is a pair of rationals, and the
        module they generate fills Q^6 -- exactly the rank-6 signature of an
        icosahedral quasicrystal, and greater than the ambient dimension 3.
        """
        one, phi = [Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]
        zero, neg_one, neg_phi = [Fraction(0), Fraction(0)], [Fraction(-1), Fraction(0)], [Fraction(0), Fraction(-1)]
        points = []
        for a in (one, neg_one):
            for b in (phi, neg_phi):
                points.append([zero, a, b])
                points.append([a, b, zero])
                points.append([b, zero, a])
        result = tl.zmodule_rank(points)
        assert result.basis_size == 2
        assert result.rank == 6
        assert result.admits_projection_lift

    def test_rejects_float_input(self) -> None:
        with pytest.raises(TypeError):
            tl.zmodule_rank([[[0.5], [0.0], [0.0]]])

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError):
            tl.zmodule_rank([])


class TestRootSystemObstruction:
    @staticmethod
    def _graph(variant: str):
        cloud = tg.build_ceg_packing(4 * 3**3, variant=variant)
        merged = ts.merge_vertices(cloud.raw_points, atol=1e-5)
        return merged, ts.build_unit_distance_graph(merged.points)

    def test_square_lattice_angles_are_root_legal(self) -> None:
        """Positive control: Z^2 unit-distance edges meet at 90 and 180 degrees."""
        pts = np.array([[x, y, 0.0] for x in range(4) for y in range(4)])
        bundle = ts.build_unit_distance_graph(pts)
        spectrum = tl.edge_angle_spectrum(pts, bundle.edges)
        assert spectrum.root_compatible
        assert spectrum.offending_angles.size == 0

    def test_tetrahedral_angle_is_not_root_legal(self) -> None:
        """arccos(-1/3) = 109.47 deg is the tetrahedron's own vertex angle."""
        tetra_angle = math.degrees(math.acos(-1 / 3))
        assert not any(abs(tetra_angle - a) <= 1e-6 for a in tl.ROOT_ANGLES_DEGREES)

    def test_ceg_edges_include_the_tetrahedral_angle(self) -> None:
        merged, bundle = self._graph("densest-connected")
        spectrum = tl.edge_angle_spectrum(merged.points, bundle.edges)
        tetra_angle = math.degrees(math.acos(-1 / 3))
        assert np.any(np.abs(spectrum.angles_degrees - tetra_angle) < 1e-4)

    def test_ceg_is_not_root_compatible(self) -> None:
        merged, bundle = self._graph("densest-connected")
        spectrum = tl.edge_angle_spectrum(merged.points, bundle.edges)
        assert not spectrum.root_compatible
        assert spectrum.offending_angles.size > 0

    @pytest.mark.parametrize(
        "variant", ["optimal", "densest-connected", "kallus-elser-gravel"]
    )
    def test_lift_is_excluded_for_every_variant(self, variant: str) -> None:
        u, v, w = tg.CEG_FAMILY_PRESETS[variant]
        exact = tg.ceg_exact_vertices(u, v, w, reps=2)
        merged, bundle = self._graph(variant)
        report = tl.root_system_report(
            tl.rational_coefficients(exact), merged.points, bundle.edges
        )
        assert report.zmodule.rank == 3
        assert not report.zmodule.admits_projection_lift
        assert report.admissible_root_systems == []
        assert report.verdict.startswith("LIFT EXCLUDED")

    def test_exact_vertices_agree_with_the_float_construction(self) -> None:
        """The rational frame must be the float frame up to the 3*sqrt(2) scale."""
        u, v, w = tg.CEG_FAMILY_PRESETS["optimal"]
        exact = np.array(
            [[float(c) for c in point] for point in tg.ceg_exact_vertices(u, v, w, reps=1)]
        ) / (3 * math.sqrt(2))
        cell, _ = tg.ceg_unit_cell(u, v, w)
        built = np.unique(np.round(cell.reshape(-1, 3), 9), axis=0)
        recovered = np.unique(np.round(exact, 9), axis=0)
        assert recovered.shape == built.shape
        np.testing.assert_allclose(recovered, built, atol=1e-9)


# --------------------------------------------------------------------------- #
# Three-fold screw-symmetric packings (the N = 3 phase)
# --------------------------------------------------------------------------- #
class TestP3Cell:
    @staticmethod
    def _quat() -> np.ndarray:
        return tg._random_quaternions(np.random.default_rng(0), 1)[0]

    def test_produces_three_regular_tetrahedra(self) -> None:
        cell, _ = tg.p3_cell(3.0, 3.0, 0.3, 0.2, self._quat(), screw=1)
        assert cell.shape == (3, 4, 3)
        np.testing.assert_allclose(tg.edge_lengths(cell), 1.0, atol=1e-12)

    def test_lattice_is_hexagonal(self) -> None:
        _, lattice = tg.p3_cell(2.5, 4.0, 0.1, 0.1, self._quat(), screw=1)
        a1, a2, a3 = lattice
        assert np.linalg.norm(a1) == pytest.approx(np.linalg.norm(a2))
        cosine = a1 @ a2 / (np.linalg.norm(a1) * np.linalg.norm(a2))
        assert math.degrees(math.acos(cosine)) == pytest.approx(120.0)
        assert a3 @ a1 == pytest.approx(0.0)
        assert a3 @ a2 == pytest.approx(0.0)

    @pytest.mark.parametrize("screw", tg.P3_SCREW_INDICES)
    def test_screw_maps_each_tetrahedron_to_the_next(self, screw: int) -> None:
        """S(T_k) must equal T_{k+1} up to a lattice translation."""
        a, c = 3.0, 3.5
        cell, lattice = tg.p3_cell(a, c, 0.3, 0.2, self._quat(), screw=screw)
        angle = 2 * math.pi / 3
        rot = np.array(
            [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
        )
        shift = np.array([0.0, 0.0, screw * c / 3.0])
        inverse = np.linalg.inv(lattice)
        for k in range(3):
            image = (rot @ cell[k].T).T + shift
            delta = (image.mean(axis=0) - cell[(k + 1) % 3].mean(axis=0)) @ inverse
            np.testing.assert_allclose(delta, np.round(delta), atol=1e-9)

    def test_volume_and_density_formulas_agree(self) -> None:
        a, c = 2.0, 3.0
        _, lattice = tg.p3_cell(a, c, 0.2, 0.4, self._quat(), screw=2)
        volume = abs(float(np.linalg.det(lattice)))
        assert volume == pytest.approx(0.5 * math.sqrt(3) * a * a * c)
        assert tg.p3_packing_fraction(a, c) == pytest.approx(
            3 * tg.UNIT_TETRA_VOLUME / volume
        )

    def test_offsets_are_fractional_so_growing_the_cell_separates_them(self) -> None:
        """Regression: with Cartesian offsets the tetrahedra stayed on the axis.

        Their mutual separation is set by distance from the rotation axis, which
        scales with ``a``.  A Cartesian offset left them overlapping no matter
        how large the cell grew, so no valid initial configuration ever existed.
        """
        quat = self._quat()
        small, _ = tg.p3_cell(0.35, 0.35, 0.3, 0.3, quat, screw=0)
        big_cell, big_lattice = tg.p3_cell(12.0, 12.0, 0.3, 0.3, quat, screw=0)
        assert not tg._configuration_is_valid(*tg.p3_cell(0.35, 0.35, 0.3, 0.3, quat, screw=0)[:2])
        assert tg._configuration_is_valid(big_cell, big_lattice)

    @pytest.mark.parametrize("kwargs", [{"a": 0.0}, {"a": -1.0}, {"c": 0.0}])
    def test_rejects_nonpositive_cell(self, kwargs: dict) -> None:
        params = {"a": 2.0, "c": 2.0, **kwargs}
        with pytest.raises(ValueError):
            tg.p3_cell(params["a"], params["c"], 0.1, 0.1, self._quat(), screw=1)

    def test_rejects_bad_screw_index(self) -> None:
        with pytest.raises(ValueError):
            tg.p3_cell(2.0, 2.0, 0.1, 0.1, self._quat(), screw=3)


class TestP3Search:
    @pytest.mark.parametrize("screw", tg.P3_SCREW_INDICES)
    def test_search_returns_a_valid_packing(self, screw: int) -> None:
        result = tg._p3_search(300, seed=7, screw=screw)
        cell, lattice = result.cell
        np.testing.assert_allclose(tg.edge_lengths(cell), 1.0, atol=1e-12)
        assert tg._configuration_is_valid(cell, lattice)
        assert 0.0 < result.packing_fraction < 1.0

    def test_search_preserves_the_screw_symmetry_throughout(self) -> None:
        result = tg._p3_search(300, seed=7, screw=1)
        cell, lattice = result.cell
        # Three tetrahedra related by a rigid motion have identical volumes.
        volumes = [tg.tetrahedron_volume(t) for t in cell]
        np.testing.assert_allclose(volumes, tg.UNIT_TETRA_VOLUME, atol=1e-12)

    def test_compression_is_monotone(self) -> None:
        low = tg._p3_search(80, seed=5, screw=1)
        high = tg._p3_search(400, seed=5, screw=1)
        assert high.packing_fraction >= low.packing_fraction

    def test_builder_survives_an_independent_tiling_check(self) -> None:
        cloud = tg.build_p3_packing(81, screw=1, cycles=200, restarts=1, seed=3)
        cloud.assert_regular(edge=1.0, atol=1e-9)
        assert tg.find_overlapping_pairs(cloud.tetrahedra).shape[0] == 0
        assert cloud.provenance["backend"] == "p3"
        assert cloud.provenance["space_group"] == "P3_1"

    def test_builder_reports_the_target_it_is_aiming_at(self) -> None:
        cloud = tg.build_p3_packing(81, screw=0, cycles=120, restarts=1, seed=3)
        assert cloud.provenance["target_packing_fraction"] == pytest.approx(2 / 3)
        assert cloud.provenance["measured_packing_fraction"] < 1.0

    @pytest.mark.parametrize("kwargs", [{"cycles": 0}, {"restarts": 0}])
    def test_rejects_invalid_effort(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            tg.build_p3_packing(27, screw=1, seed=1, **kwargs)


# --------------------------------------------------------------------------- #
# The N = 3 phase, recovered from Table I's description
# --------------------------------------------------------------------------- #
class TestC3AlignedTetrahedron:
    def test_is_regular_with_unit_edges(self) -> None:
        t = tg.c3_aligned_tetrahedron(0.37, +1)
        np.testing.assert_allclose(tg.edge_lengths(t[None]), 1.0, atol=1e-12)

    def test_centroid_is_the_origin(self) -> None:
        np.testing.assert_allclose(
            tg.c3_aligned_tetrahedron(1.1, -1).mean(axis=0), 0.0, atol=1e-12
        )

    @pytest.mark.parametrize("sign", [1, -1])
    def test_is_invariant_under_the_three_fold_rotation(self, sign: int) -> None:
        """This is the property the whole ansatz rests on."""
        t = tg.c3_aligned_tetrahedron(0.83, sign)
        image = (tg._ROT120 @ t.T).T
        distance = np.linalg.norm(image[:, None, :] - t[None, :, :], axis=2)
        np.testing.assert_allclose(distance.min(axis=1), 0.0, atol=1e-12)

    def test_height_and_base_radius_match_closed_forms(self) -> None:
        assert tg.C3_TETRA_HEIGHT == pytest.approx(math.sqrt(2 / 3))
        assert tg.C3_TETRA_BASE_RADIUS == pytest.approx(1 / math.sqrt(3))

    def test_rejects_bad_sign(self) -> None:
        with pytest.raises(ValueError):
            tg.c3_aligned_tetrahedron(0.0, 0)


class TestN3Phase:
    """Table I's N = 3 phase: 3 monomers, three-fold symmetric, phi = 2/3."""

    def test_cell_parameters_are_the_closed_forms(self) -> None:
        assert tg.N3_CELL_A == pytest.approx(math.sqrt(3) / 2)
        assert tg.N3_CELL_C == pytest.approx(math.sqrt(2 / 3))
        # c is exactly the tetrahedron's own height along its three-fold axis.
        assert tg.N3_CELL_C == pytest.approx(tg.C3_TETRA_HEIGHT)

    def test_density_is_exactly_two_thirds(self) -> None:
        _, lattice = tg.n3_unit_cell()
        volume = abs(float(np.linalg.det(lattice)))
        assert volume == pytest.approx(3 * math.sqrt(2) / 8, abs=1e-15)
        assert 3 * tg.UNIT_TETRA_VOLUME / volume == pytest.approx(2 / 3, abs=1e-15)

    def test_unit_cell_holds_three_regular_tetrahedra(self) -> None:
        cell, _ = tg.n3_unit_cell()
        assert cell.shape == (3, 4, 3)
        np.testing.assert_allclose(tg.edge_lengths(cell), 1.0, atol=1e-12)

    def test_unit_cell_is_overlap_free_under_periodicity(self) -> None:
        cell, lattice = tg.n3_unit_cell()
        assert tg._configuration_is_valid(cell, lattice, tolerance=1e-9)

    def test_tiled_packing_has_no_overlaps(self) -> None:
        cloud = tg.build_n3_packing(200)
        assert tg.find_overlapping_pairs(cloud.tetrahedra, tolerance=1e-9).shape[0] == 0

    def test_each_monomer_sits_on_its_own_three_fold_axis(self) -> None:
        """The rotation maps every tetrahedron to itself, not to another."""
        cell, lattice = tg.n3_unit_cell()
        inverse = np.linalg.inv(lattice)
        for k in range(3):
            image = (tg._ROT120 @ cell[k].T).T
            delta = (image.mean(axis=0) - cell[k].mean(axis=0)) @ inverse
            np.testing.assert_allclose(delta, np.round(delta), atol=1e-9)

    def test_monomers_occupy_the_three_distinct_wyckoff_axes(self) -> None:
        cell, lattice = tg.n3_unit_cell()
        fractional = (cell.mean(axis=1) @ np.linalg.inv(lattice))[:, :2] % 1.0
        expected = np.array([[float(x), float(y)] for x, y in tg.P3_AXIS_SITES])
        for site in expected:
            assert np.any(np.all(np.isclose(fractional, site, atol=1e-9), axis=1))

    def test_apex_directions_are_mixed(self) -> None:
        """All three pointing the same way collapses the density to ~0.43."""
        assert len(set(tg.N3_SIGNS)) == 2

    def test_is_reproducible(self) -> None:
        np.testing.assert_array_equal(
            tg.build_n3_packing(200).tetrahedra, tg.build_n3_packing(200).tetrahedra
        )

    def test_beats_every_searched_family(self) -> None:
        """2/3 exceeds the converged optimum of the single-orbit P3 family."""
        cloud = tg.build_n3_packing(200)
        assert cloud.provenance["packing_fraction"] > 0.60

    def test_graph_is_far_more_connected_than_the_dimer_packings(self) -> None:
        """Unlike CEG, the N = 3 phase shares vertices between distinct monomers."""
        cloud = tg.build_n3_packing(500)
        merged = ts.merge_vertices(cloud.raw_points, atol=1e-5)
        bundle = ts.build_unit_distance_graph(merged.points)

        assert merged.n_unique < merged.n_raw          # genuine vertex sharing
        assert bundle.n_components < cloud.n_tetrahedra // 8
        assert bundle.degrees.max() >= 8               # CEG never exceeds 4
        spectrum = ts.smallest_eigenvalues(
            ts.graph_laplacian(bundle.adjacency),
            k=merged.n_unique,
            n_components=bundle.n_components,
            component_labels=bundle.component_labels,
        )
        assert spectrum.zero_multiplicity == bundle.n_components
        assert 0.0 < spectrum.first_positive < 1.0     # CEG gives exactly 3

    def test_rejects_nonsense_size(self) -> None:
        with pytest.raises(ValueError):
            tg.build_n3_packing(0)


# --------------------------------------------------------------------------- #
# The N = 3 phase's symmetry fingerprint
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def n3_cluster():
    """C3-invariant cluster of the N = 3 phase, with its graph and group action."""
    from scipy.spatial import cKDTree

    cloud = tg.build_n3_cluster(radius=4.0, layers=5)
    merged = ts.merge_vertices(cloud.raw_points, atol=1e-5)
    bundle = ts.build_unit_distance_graph(merged.points)
    rotated = (tg._ROT120 @ merged.points.T).T
    distance, index = cKDTree(merged.points).query(rotated)
    return cloud, merged, bundle, distance.max(), index


class TestN3Fingerprint:
    """The N = 3 phase decomposes into 16 networks permuted by C3."""

    def test_cluster_is_exactly_c3_invariant(self, n3_cluster) -> None:
        _, _, _, max_distance, _ = n3_cluster
        assert max_distance < 1e-9

    def test_cell_based_cut_would_not_be_invariant(self) -> None:
        """Rotation moves the 1b monomer of cell (0,0) into cell (-1,-1)."""
        from scipy.spatial import cKDTree

        cell, lattice = tg.n3_unit_cell()
        grid = [(i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(3)]
        translations = np.array(grid, dtype=float) @ lattice
        tiled = (cell[None] + translations[:, None, None, :]).reshape(-1, 4, 3)
        centroids = tiled.mean(axis=1)
        # Keep whole cells inside a basal disc -- invariant as positions, but not
        # as a selection of tetrahedra.
        cell_pos = translations[:, :2]
        keep_cell = np.hypot(cell_pos[:, 0], cell_pos[:, 1]) <= 2.0 + 1e-9
        selected = tiled.reshape(len(grid), 3, 4, 3)[keep_cell].reshape(-1, 4, 3)
        points = np.unique(np.round(selected.reshape(-1, 3), 9), axis=0)
        rotated = (tg._ROT120 @ points.T).T
        assert cKDTree(points).query(rotated)[0].max() > 1e-6

    def test_graph_splits_into_sixteen_networks(self, n3_cluster) -> None:
        _, _, bundle, _, _ = n3_cluster
        assert bundle.n_components == 16

    @pytest.mark.parametrize("reps", [3, 4, 5, 6])
    def test_network_count_is_size_independent(self, reps: int) -> None:
        """16 at every size, each growing with the block: interpenetrating, not fragments."""
        cell, lattice = tg.n3_unit_cell()
        grid = np.arange(reps, dtype=float)
        i, j, k = np.meshgrid(grid, grid, grid, indexing="ij")
        translations = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) @ lattice
        tiled = (cell[None] + translations[:, None, None, :]).reshape(-1, 4, 3)
        merged = ts.merge_vertices(tiled.reshape(-1, 3), atol=1e-5)
        bundle = ts.build_unit_distance_graph(merged.points)
        assert bundle.n_components == 16

    def test_c3_orbits_are_four_fixed_and_four_triples(self, n3_cluster) -> None:
        _, _, bundle, _, index = n3_cluster
        image = {}
        for source, target in zip(bundle.component_labels, bundle.component_labels[index]):
            image.setdefault(int(source), set()).add(int(target))
        mapping = {k: v.pop() for k, v in image.items() if len(v) == 1}
        assert len(mapping) == 16

        fixed = [k for k, v in mapping.items() if v == k]
        assert len(fixed) == 4

        seen, orbits = set(), []
        for start in mapping:
            if start in seen:
                continue
            orbit, node = [start], start
            seen.add(start)
            while mapping[node] not in seen:
                node = mapping[node]
                orbit.append(node)
                seen.add(node)
            orbits.append(len(orbit))
        assert sorted(orbits) == [1, 1, 1, 1, 3, 3, 3, 3]

    @staticmethod
    def _component_multiplicities(bundle, label: int) -> dict[int, int]:
        laplacian = ts.graph_laplacian(bundle.adjacency)
        selected = np.nonzero(bundle.component_labels == label)[0]
        block = laplacian[selected][:, selected]
        values = np.linalg.eigvalsh(np.asarray(block.todense()))
        return ts.analyse_degeneracies(values).multiplicity_histogram

    def test_fixed_networks_show_only_c3_irrep_dimensions(self, n3_cluster) -> None:
        """C3 has real irreps of dimension 1 and 2. Nothing else may appear."""
        _, _, bundle, _, index = n3_cluster
        image = {}
        for source, target in zip(bundle.component_labels, bundle.component_labels[index]):
            image.setdefault(int(source), set()).add(int(target))
        mapping = {k: v.pop() for k, v in image.items() if len(v) == 1}
        fixed = [k for k, v in mapping.items() if v == k]

        for label in fixed:
            multiplicities = set(self._component_multiplicities(bundle, label))
            assert multiplicities <= {1, 2}, f"component {label}: {multiplicities}"

    def test_orbit_networks_have_no_internal_symmetry(self, n3_cluster) -> None:
        """A network C3 moves is not itself symmetric, so its spectrum is simple."""
        _, _, bundle, _, index = n3_cluster
        image = {}
        for source, target in zip(bundle.component_labels, bundle.component_labels[index]):
            image.setdefault(int(source), set()).add(int(target))
        mapping = {k: v.pop() for k, v in image.items() if len(v) == 1}
        moved = [k for k, v in mapping.items() if v != k]

        for label in moved[:4]:
            assert set(self._component_multiplicities(bundle, label)) == {1}

    def test_odd_layer_counts_avoid_the_pairing_artefact(self) -> None:
        """Even layer counts pair the networks up and double every multiplicity."""
        def multiplicities(layers: int) -> set[int]:
            cloud = tg.build_n3_cluster(radius=3.0, layers=layers)
            merged = ts.merge_vertices(cloud.raw_points, atol=1e-5)
            bundle = ts.build_unit_distance_graph(merged.points)
            spectrum = ts.smallest_eigenvalues(
                ts.graph_laplacian(bundle.adjacency), k=merged.n_unique,
                n_components=bundle.n_components,
                component_labels=bundle.component_labels,
            )
            report = ts.analyse_degeneracies(spectrum.eigenvalues)
            return {m for m, c in report.multiplicity_histogram.items() if c > 5}

        assert 1 in multiplicities(5)      # odd: singlets survive
        assert 1 not in multiplicities(4)  # even: everything doubles


# --------------------------------------------------------------------------- #
# Why 16: the quotient graph with voltages
# --------------------------------------------------------------------------- #
class TestSublatticeIndex:
    """Exact integer index of a subgroup of Z^3, with no symbolic dependency."""

    def test_standard_basis_has_index_one(self) -> None:
        index, _ = ts.sublattice_index(np.eye(3, dtype=np.int64))
        assert index == 1

    def test_doubled_lattice_has_index_eight(self) -> None:
        index, basis = ts.sublattice_index(2 * np.eye(3, dtype=np.int64))
        assert index == 8
        assert basis == ((2, 0, 0), (0, 2, 0), (0, 0, 2))

    def test_index_matches_the_determinant(self) -> None:
        generators = [(2, 0, 0), (1, 1, 0), (0, 0, 3)]
        index, _ = ts.sublattice_index(generators)
        assert index == abs(round(float(np.linalg.det(np.array(generators, float)))))
        assert index == 6

    def test_redundant_generators_do_not_change_the_index(self) -> None:
        base = [(2, 2, 0), (0, 2, 0), (0, 0, 2)]
        extra = base + [(2, 4, 0), (4, 0, 2), (-2, -2, 0)]
        assert ts.sublattice_index(extra)[0] == ts.sublattice_index(base)[0] == 8

    def test_a_non_spanning_set_has_no_finite_index(self) -> None:
        index, basis = ts.sublattice_index([(1, 0, 0), (0, 1, 0), (2, 3, 0)])
        assert index is None
        assert basis is None

    def test_no_generators_is_rank_zero(self) -> None:
        assert ts.sublattice_index([]) == (None, None)
        assert ts.sublattice_rank([]) == 0

    @pytest.mark.parametrize(
        "generators, expected",
        [
            ([], 0),
            ([(0, 0, 0)], 0),
            ([(1, 2, 3)], 1),
            ([(1, 0, 0), (2, 0, 0)], 1),
            ([(1, 0, 0), (0, 1, 0), (2, 3, 0)], 2),
            ([(1, 1, 1), (0, 1, 0), (0, 0, 5)], 3),
        ],
    )
    def test_rank_counts_independent_directions(self, generators, expected: int) -> None:
        assert ts.sublattice_rank(generators) == expected


class TestPeriodicGraphComponents:
    """The infinite graph's component count, from one unit cell and exact integers."""

    def test_n3_quotient_has_nine_orbits_in_two_pieces(self) -> None:
        cell, lattice = tg.n3_unit_cell()
        report = ts.periodic_graph_components(cell, lattice)
        assert report.n_orbits == 9          # 12 raw vertices, 3 pairs coincide
        assert len(report.components) == 2
        assert sorted(len(c.orbits) for c in report.components) == [4, 5]

    def test_each_n3_piece_closes_only_on_even_translations(self) -> None:
        """The whole explanation: cycle voltages generate exactly 2 Z^3."""
        cell, lattice = tg.n3_unit_cell()
        report = ts.periodic_graph_components(cell, lattice)
        for component in report.components:
            assert component.rank == 3
            assert component.index == 8
            basis = np.array(component.sublattice_basis, dtype=np.int64)
            assert np.all(basis % 2 == 0)                    # subgroup of 2 Z^3
            assert ts.sublattice_index(2 * np.eye(3, dtype=np.int64))[0] == 8

    def test_sixteen_is_two_pieces_times_eight_parity_classes(self) -> None:
        cell, lattice = tg.n3_unit_cell()
        report = ts.periodic_graph_components(cell, lattice)
        assert report.n_infinite_components == 16
        assert report.n_infinite_components == 2 * 2 ** 3

    @pytest.mark.parametrize("span", [1, 2, 3, 4])
    def test_the_answer_does_not_depend_on_the_search_span(self, span: int) -> None:
        cell, lattice = tg.n3_unit_cell()
        report = ts.periodic_graph_components(cell, lattice, span=span)
        assert report.n_infinite_components == 16

    @pytest.mark.parametrize("reps", [4, 5, 6])
    def test_agrees_with_direct_counts_on_finite_blocks(self, reps: int) -> None:
        cell, lattice = tg.n3_unit_cell()
        grid = np.arange(reps, dtype=float)
        i, j, k = np.meshgrid(grid, grid, grid, indexing="ij")
        translations = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) @ lattice
        tiled = (cell[None] + translations[:, None, None, :]).reshape(-1, 4, 3)
        merged = ts.merge_vertices(tiled.reshape(-1, 3), atol=1e-5)
        bundle = ts.build_unit_distance_graph(merged.points)
        report = ts.periodic_graph_components(cell, lattice)
        assert bundle.n_components == report.n_infinite_components

    def test_ceg_optimum_is_bounded_dimers_not_a_network(self) -> None:
        """Rank 0: every cycle closes at zero voltage, so nothing extends."""
        cell, lattice = tg.ceg_unit_cell(variant="optimal")
        report = ts.periodic_graph_components(cell, lattice)
        assert [c.rank for c in report.components] == [0, 0]
        assert report.n_infinite_components is None      # infinitely many dimers

    def test_the_connected_ceg_member_is_a_stack_of_sheets(self) -> None:
        """Rank 2, not 3: connectivity on the u = 0 plane is two-dimensional."""
        cell, lattice = tg.ceg_unit_cell(variant="densest-connected")
        report = ts.periodic_graph_components(cell, lattice)
        assert len(report.components) == 1
        assert report.components[0].rank == 2
        assert report.components[0].index is None
        assert report.n_infinite_components is None

    @pytest.mark.parametrize("reps", [3, 4, 5])
    def test_sheet_count_grows_with_the_block(self, reps: int) -> None:
        """One sheet per layer -- the direct counterpart of rank 2."""
        cell, lattice = tg.ceg_unit_cell(variant="densest-connected")
        grid = np.arange(reps, dtype=float)
        i, j, k = np.meshgrid(grid, grid, grid, indexing="ij")
        translations = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) @ lattice
        tiled = (cell[None] + translations[:, None, None, :]).reshape(-1, 4, 3)
        merged = ts.merge_vertices(tiled.reshape(-1, 3), atol=1e-5)
        bundle = ts.build_unit_distance_graph(merged.points)
        assert bundle.n_components == reps

    def test_rejects_malformed_input(self) -> None:
        cell, lattice = tg.n3_unit_cell()
        with pytest.raises(ValueError, match="shape"):
            ts.periodic_graph_components(cell.reshape(-1, 3), lattice)
        with pytest.raises(ValueError, match="shape"):
            ts.periodic_graph_components(cell, lattice[:2])
        with pytest.raises(ValueError, match="span"):
            ts.periodic_graph_components(cell, lattice, span=0)


# --------------------------------------------------------------------------- #
# The N = 2 phase: a monomer double lattice
# --------------------------------------------------------------------------- #
class TestDoubleLattice:
    """Table I's N = 2 phase, phi_2 = 9/(139 - 40 sqrt 10), "2 monomers, transitive"."""

    def test_target_density_matches_the_published_value(self) -> None:
        assert tg.N2_PACKING_FRACTION == pytest.approx(0.719486, abs=3e-6)

    def test_cell_holds_two_regular_tetrahedra(self) -> None:
        cell, _ = tg.double_lattice_cell(np.eye(3) * 3.0, np.array([1.4, 0.3, 0.2]))
        assert cell.shape == (2, 4, 3)
        np.testing.assert_allclose(tg.edge_lengths(cell), 1.0, atol=1e-12)

    def test_the_pair_is_related_by_a_point_inversion(self) -> None:
        """Inversion through d/2 exchanges them, which is what makes it transitive."""
        from scipy.spatial import cKDTree

        offset = np.array([1.4, 0.3, 0.2])
        cell, _ = tg.double_lattice_cell(np.eye(3) * 3.0, offset)
        centre = cell.mean(axis=(0, 1))
        image = 2 * centre - cell[0]
        assert cKDTree(cell[1]).query(image)[0].max() < 1e-9

    def test_the_pair_is_not_a_plain_lattice_packing(self) -> None:
        """Identical orientations would be an N=1 lattice packing, capped at 18/49."""
        from scipy.spatial import cKDTree

        cell, _ = tg.double_lattice_cell(np.eye(3) * 3.0, np.array([1.4, 0.3, 0.2]))
        centred = [t - t.mean(axis=0) for t in cell]
        assert cKDTree(centred[1]).query(centred[0])[0].max() > 0.1

    def test_search_returns_a_certified_packing(self) -> None:
        result = tg._double_lattice_search(400, seed=3)
        cell, lattice = result.cell
        np.testing.assert_allclose(tg.edge_lengths(cell), 1.0, atol=1e-12)
        assert tg._configuration_is_valid(cell, lattice)
        assert 0.0 < result.packing_fraction < tg.N2_PACKING_FRACTION + 1e-9

    def test_compression_is_monotone(self) -> None:
        low = tg._double_lattice_search(100, seed=5)
        high = tg._double_lattice_search(600, seed=5)
        assert high.packing_fraction >= low.packing_fraction

    def test_search_never_exceeds_the_published_optimum(self) -> None:
        """Beating phi_2 would mean the overlap test is wrong, not a discovery."""
        for seed in (11, 23, 37):
            result = tg._double_lattice_search(1500, seed=seed)
            assert result.packing_fraction < tg.N2_PACKING_FRACTION + 1e-9


# --------------------------------------------------------------------------- #
# The dodecagonal quasicrystal: a positive control for the lift test
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def quasilattice():
    """A dodecagonal cut-and-project set, in float and exact coefficient form."""
    return tl.dodecagonal_quasilattice(window_radius=1.6, index_range=5, layers=3)


class TestDodecagonalQuasicrystal:
    """Where a higher-dimensional lift is genuinely real, unlike the crystals."""

    def test_in_plane_module_has_rank_four(self, quasilattice) -> None:
        """Z[zeta_12] has rank 4: the minimal polynomial x^4 - x^2 + 1 is degree 4."""
        _, coefficients = quasilattice
        planar = [[c[0], c[1]] for c in coefficients]
        assert tl.zmodule_rank(planar).rank == 4

    def test_stacked_structure_has_rank_five(self, quasilattice) -> None:
        """Four in-plane plus one periodic axis: a projection from five dimensions."""
        _, coefficients = quasilattice
        result = tl.zmodule_rank(coefficients)
        assert result.rank == 5
        assert result.ambient_dimension == 3
        assert result.admits_projection_lift

    def test_is_twelve_fold_symmetric_in_the_core(self, quasilattice) -> None:
        from scipy.spatial import cKDTree

        points, _ = quasilattice
        planar = points[np.isclose(points[:, 2], 0.0)][:, :2]
        angle = math.pi / 6
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        distance, _ = cKDTree(planar).query(planar @ rotation.T)
        radius = np.linalg.norm(planar, axis=1)
        core = radius < radius.max() * 0.55
        assert distance[core].max() < 1e-9

    def test_is_a_discrete_point_set_not_a_dense_module(self, quasilattice) -> None:
        """The acceptance window is what makes it a quasicrystal rather than dense."""
        from scipy.spatial import cKDTree

        points, _ = quasilattice
        planar = points[np.isclose(points[:, 2], 0.0)][:, :2]
        distance, _ = cKDTree(planar).query(planar, k=2)
        assert distance[:, 1].min() > 0.2

    def test_the_crystals_do_not_admit_a_lift(self) -> None:
        """The same test that returns 5 here returns 3 for every packing in the repo."""
        for variant in ("optimal", "kallus-elser-gravel", "densest-connected"):
            u, v, w = tg.CEG_FAMILY_PRESETS[variant]
            exact = tg.ceg_exact_vertices(u, v, w, reps=2)
            result = tl.zmodule_rank(tl.rational_coefficients(exact))
            assert result.rank == 3
            assert not result.admits_projection_lift

    def test_rejects_bad_parameters(self) -> None:
        with pytest.raises(ValueError):
            tl.dodecagonal_quasilattice(window_radius=0.0)
        with pytest.raises(ValueError):
            tl.dodecagonal_quasilattice(index_range=0)


@pytest.fixture(scope="module")
def refined():
    """A coarse Monte Carlo result and its constrained-optimisation polish."""
    coarse = tg._double_lattice_search(4000, seed=17)
    return coarse, tg.refine_double_lattice(coarse.lattice, coarse.offset, rounds=4)


class TestDoubleLatticeRefinement:
    """Constrained optimisation finishes what Monte Carlo cannot."""

    def test_refinement_improves_on_the_search(self, refined) -> None:
        coarse, fine = refined
        assert fine.packing_fraction > coarse.packing_fraction

    def test_refined_result_is_still_a_certified_packing(self, refined) -> None:
        _, fine = refined
        cell, lattice = fine.cell
        np.testing.assert_allclose(tg.edge_lengths(cell), 1.0, atol=1e-12)
        assert tg._configuration_is_valid(cell, lattice, tolerance=1e-11)

    def test_refinement_never_exceeds_the_published_optimum(self, refined) -> None:
        """Beating phi_2 would mean the overlap test is broken, not a discovery."""
        _, fine = refined
        assert fine.packing_fraction < tg.N2_PACKING_FRACTION + 1e-9

    def test_a_positive_margin_costs_density(self) -> None:
        """Holding contacts apart by eps leaves the answer short by about eps."""
        coarse = tg._double_lattice_search(3000, seed=23)
        tight = tg.refine_double_lattice(coarse.lattice, coarse.offset, rounds=3, margin=0.0)
        loose = tg.refine_double_lattice(coarse.lattice, coarse.offset, rounds=3, margin=1e-3)
        assert tight.packing_fraction > loose.packing_fraction

    def test_contact_candidates_are_found(self) -> None:
        coarse = tg._double_lattice_search(2000, seed=5)
        candidates = tg._contact_candidates(coarse.lattice, coarse.offset, 0.25)
        assert len(candidates) > 0
        for i, j, shift in candidates:
            assert 0 <= i <= j <= 1
            assert shift.shape == (3,)


class TestQuadraticRecognition:
    """Turning a converged optimum into exact algebra."""

    @pytest.mark.parametrize(
        "value, rational, surd",
        [
            (16 * (139 - 40 * math.sqrt(10)) / 27, Fraction(2224, 27), Fraction(-640, 27)),
            (3 / 7 - 5 * math.sqrt(10) / 11, Fraction(3, 7), Fraction(-5, 11)),
            (math.sqrt(10), Fraction(0), Fraction(1)),
            (0.5, Fraction(1, 2), Fraction(0)),
        ],
    )
    def test_recognises_low_height_elements(self, value, rational, surd) -> None:
        form = tl.recognise_quadratic(
            value, 10, max_denominator=300, max_surd_numerator=3000
        )
        assert form is not None
        assert form.rational == rational
        assert form.surd == surd
        assert form.value() == pytest.approx(value, abs=1e-12)

    def test_returns_none_for_a_transcendental(self) -> None:
        """Returning None is the meaningful answer, not a failure to try."""
        assert tl.recognise_quadratic(
            math.pi, 10, max_denominator=60, max_surd_numerator=200, tolerance=1e-12
        ) is None

    def test_tolerance_below_the_data_precision_returns_none(self) -> None:
        """A value known to 1e-7 cannot be recognised at 1e-12."""
        approx = 16 * (139 - 40 * math.sqrt(10)) / 27 + 4e-8
        assert tl.recognise_quadratic(approx, 10, tolerance=1e-7,
                                      max_denominator=50, max_surd_numerator=2000) is not None
        assert tl.recognise_quadratic(approx, 10, tolerance=1e-12,
                                      max_denominator=50, max_surd_numerator=2000) is None

    def test_rejects_perfect_squares(self) -> None:
        with pytest.raises(ValueError):
            tl.recognise_quadratic(1.0, 9)

    def test_the_n2_determinant_is_recognised_from_the_optimum(self) -> None:
        """End to end: search, refine, then read off the exact determinant.

        In the integer-tetrahedron frame the N = 2 target determinant is
        16(139 - 40 sqrt10)/27, which lies in Q(sqrt 10) with no sqrt(2); the
        unit-edge frame instead gives (139 sqrt2 - 80 sqrt5)/54 and hides it.
        """
        coarse = tg._double_lattice_search(6000, seed=17)
        fine = tg.refine_double_lattice(coarse.lattice, coarse.offset, rounds=5)
        determinant = abs(float(np.linalg.det(fine.lattice * 2 * math.sqrt(2))))
        expected = 16 * (139 - 40 * math.sqrt(10)) / 27
        # The refinement gets within a few parts in 1e6 from a short search.
        assert determinant == pytest.approx(expected, rel=2e-4)


class TestGeneralRefinement:
    """refine_packing: what constrained optimisation can and cannot rescue."""

    @staticmethod
    def _p3_build(screw: int):
        from scipy.spatial.transform import Rotation

        def build(p):
            quat = np.roll(Rotation.from_rotvec(p[4:7]).as_quat(), 1)
            return tg.p3_cell(abs(p[0]), abs(p[1]), p[2], p[3], quat, screw)

        return build

    @staticmethod
    def _p3_params(result) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        rotvec = Rotation.from_quat(np.roll(result.quat, -1)).as_rotvec()
        return np.array([result.a, result.c, result.fx, result.fy, *rotvec])

    def test_rejects_an_infeasible_start(self) -> None:
        """Polishing cannot repair overlaps; reporting a density for one is a lie.

        This is a real bug that shipped: the refiner took its starting volume on
        trust, so a broken reconstruction was reported at the *search's* density
        while the configuration actually interpenetrated.
        """
        build = self._p3_build(1)
        params = np.array([0.3, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0])  # far too small a cell
        cell, lattice = build(params)
        assert not tg._configuration_is_valid(cell, lattice)
        with pytest.raises(ValueError, match="infeasible"):
            tg.refine_packing(build, params, rounds=1)

    def test_refinement_improves_the_p3_family(self) -> None:
        """The Monte Carlo plateau there is genuinely short of the optimum."""
        build = self._p3_build(2)
        coarse = tg._p3_search(2000, seed=503, screw=2)
        params = self._p3_params(coarse)
        if not tg._configuration_is_valid(*build(params)):
            pytest.skip("coarse search landed outside the feasible set")
        _, refined = tg.refine_packing(build, params, rounds=4, maxiter=300)
        assert refined >= coarse.packing_fraction

    def test_refined_result_is_certified(self) -> None:
        build = self._p3_build(2)
        coarse = tg._p3_search(2000, seed=503, screw=2)
        params = self._p3_params(coarse)
        if not tg._configuration_is_valid(*build(params)):
            pytest.skip("coarse search landed outside the feasible set")
        best, _ = tg.refine_packing(build, params, rounds=4, maxiter=300)
        cell, lattice = build(best)
        np.testing.assert_allclose(tg.edge_lengths(cell), 1.0, atol=1e-12)
        assert tg._configuration_is_valid(cell, lattice, tolerance=1e-11)

    def test_p3_family_stays_far_below_the_n3_phase(self) -> None:
        """Even refined, the single-orbit family cannot reach 2/3.

        This is what makes the N = 3 structural conclusion safe: the shortfall is
        not a search artefact.
        """
        build = self._p3_build(2)
        coarse = tg._p3_search(2000, seed=503, screw=2)
        params = self._p3_params(coarse)
        if not tg._configuration_is_valid(*build(params)):
            pytest.skip("coarse search landed outside the feasible set")
        _, refined = tg.refine_packing(build, params, rounds=4, maxiter=300)
        assert refined < 0.62 < float(tg.N3_PACKING_FRACTION)

    def test_search_results_carry_their_own_parameters(self) -> None:
        """Reconstructing orientations by Kabsch was error-prone and got it wrong."""
        result = tg._asc_search(3, 200, seed=5, motif="single")
        assert result.fractional is not None and result.quaternions is not None
        shape = tg.build_motif("single")
        rebuilt = tg._cell_vertices(
            shape, result.lattice, result.fractional, result.quaternions
        )
        np.testing.assert_allclose(rebuilt, result.cell_tetrahedra, atol=1e-12)

    def test_contact_candidates_generalise_beyond_two_bodies(self) -> None:
        result = tg._asc_search(4, 300, seed=7, motif="single")
        candidates = tg.contact_candidates(result.cell_tetrahedra, result.lattice, 0.25)
        assert len(candidates) > 0
        assert max(j for _i, j, _n in candidates) <= 3
