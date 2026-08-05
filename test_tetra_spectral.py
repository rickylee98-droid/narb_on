"""Test suite for the tetrahedral point-cloud spectral analysis pipeline.

Run with::

    python -m pytest test_tetra_spectral.py -v
"""

from __future__ import annotations

import math
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csr_matrix

import tetra_geometry as tg
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
        low = tg._asc_search(4, 40, seed=5)
        high = tg._asc_search(4, 200, seed=5)
        assert high.packing_fraction >= low.packing_fraction

    @pytest.mark.parametrize("kwargs", [{"cycles": 0}, {"restarts": 0}, {"n_particles": 0}])
    def test_rejects_invalid_parameters(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            tg.build_dense_packing(32, seed=1, **kwargs)

    def test_image_offsets_refuse_degenerate_cells(self) -> None:
        """A cell too thin to certify must return None rather than under-test."""
        flat = np.diag([1.0, 1.0, 1e-4])
        assert tg._image_offsets(flat) is None


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
        cloud = tg.build_dense_packing(64, cycles=60, restarts=1, seed=3)
        merged = ts.merge_vertices(cloud.raw_points, atol=1e-5)
        bundle = ts.build_unit_distance_graph(merged.points)

        assert bundle.n_components == cloud.n_tetrahedra
        np.testing.assert_array_equal(bundle.degrees, 3)

        spectrum = ts.smallest_eigenvalues(
            ts.graph_laplacian(bundle.adjacency),
            k=8,
            n_components=bundle.n_components,
        )
        # Algebraic connectivity of a disconnected graph is exactly zero.
        assert spectrum.algebraic_connectivity == 0.0
        assert spectrum.first_positive is None or spectrum.first_positive == pytest.approx(4.0)


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
