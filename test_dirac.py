"""Referees for `dirac`.

Three checks carry the module, and none of them was built to pass:

  * ``partial . partial = 0`` -- if the boundary signs were wrong, nothing below
    would mean anything;
  * ``dim ker D`` equals the sum of the Betti numbers, on complexes whose
    homology is known independently (circle 2, disk 1, two-sphere 2);
  * ``spec(D)`` reconstructed from ``spec(Delta)`` alone reproduces ``spec(D)``,
    which is the refutation itself.

`TestWhatThisDoesNotClaim` records the two things this module is careful not to
say.
"""

from __future__ import annotations

import pytest
import sympy as sp

import dirac


CIRCLE = dirac.complex_from_maximal_faces([(0, 1), (1, 2), (0, 2)])
DISK = dirac.complex_from_maximal_faces([(0, 1, 2)])
SPHERE = dirac.complex_from_maximal_faces(
    [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
)
TWO_EDGES = dirac.complex_from_maximal_faces([(0, 1), (2, 3)])
PATH = dirac.complex_from_maximal_faces([(0, 1), (1, 2), (2, 3)])

COMPLEXES = [CIRCLE, DISK, SPHERE, TWO_EDGES, PATH]
NAMES = ["circle", "disk", "sphere", "two-edges", "path"]


# ---------------------------------------------------------------------------
# the apparatus
# ---------------------------------------------------------------------------


class TestBoundaryOperator:
    @pytest.mark.parametrize("complex_", COMPLEXES, ids=NAMES)
    def test_boundary_squared_is_zero(self, complex_: dirac.SimplicialComplex):
        """``partial_{k} partial_{k+1} = 0``. Everything else rests on this."""
        for degree in range(complex_.top_dimension + 1):
            lower = dirac.boundary_matrix(complex_, degree)
            upper = dirac.boundary_matrix(complex_, degree + 1)
            if lower.cols and upper.rows == lower.cols and upper.cols:
                assert (lower * upper).is_zero_matrix

    def test_degree_zero_boundary_is_empty(self):
        assert dirac.boundary_matrix(CIRCLE, 0).rows == 0

    def test_edge_boundary_signs(self):
        """An edge maps to its head minus its tail."""
        matrix = dirac.boundary_matrix(PATH, 1)
        for column in range(matrix.cols):
            entries = sorted(matrix[:, column])
            assert entries[0] == -1
            assert entries[-1] == 1
            assert sum(matrix[:, column]) == 0

    def test_rejects_negative_degree(self):
        with pytest.raises(ValueError):
            dirac.boundary_matrix(CIRCLE, -1)


class TestComplexValidation:
    def test_rejects_unclosed_complex(self):
        with pytest.raises(ValueError, match="not closed"):
            dirac.SimplicialComplex(simplices=((0,), (1,), (0, 1, 2)))

    def test_rejects_unsorted_simplex(self):
        with pytest.raises(ValueError, match="not sorted"):
            dirac.SimplicialComplex(simplices=((0,), (1,), (1, 0)))

    def test_rejects_repeated_vertex(self):
        with pytest.raises(ValueError, match="repeats"):
            dirac.SimplicialComplex(simplices=((0,), (0, 0)))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            dirac.SimplicialComplex(simplices=())

    def test_rejects_empty_face(self):
        with pytest.raises(ValueError, match="at least one vertex"):
            dirac.complex_from_maximal_faces([()])

    def test_rejects_negative_degree(self):
        with pytest.raises(ValueError):
            CIRCLE.of_dimension(-1)

    def test_graph_complex_rejects_bad_edge(self):
        with pytest.raises(ValueError, match="two endpoints"):
            dirac.graph_complex(3, [(0, 1, 2)])

    def test_graph_complex_rejects_no_vertices(self):
        with pytest.raises(ValueError):
            dirac.graph_complex(0, [])

    def test_graph_complex_keeps_isolated_vertices(self):
        assert dirac.graph_complex(4, [(0, 1)]).dimensions == (4, 1)

    @pytest.mark.parametrize("complex_", COMPLEXES, ids=NAMES)
    def test_dimensions_sum_to_simplex_count(
        self, complex_: dirac.SimplicialComplex
    ):
        assert sum(complex_.dimensions) == len(complex_.simplices)


class TestDiracIsASquareRoot:
    @pytest.mark.parametrize("complex_", COMPLEXES, ids=NAMES)
    def test_dirac_squared_is_the_hodge_laplacian(
        self, complex_: dirac.SimplicialComplex
    ):
        assert dirac.dirac_squared_residual(complex_).is_zero_matrix

    @pytest.mark.parametrize("complex_", COMPLEXES, ids=NAMES)
    def test_dirac_is_symmetric(self, complex_: dirac.SimplicialComplex):
        operator = dirac.dirac_operator(complex_)
        assert (operator - operator.T).is_zero_matrix

    @pytest.mark.parametrize("complex_", COMPLEXES, ids=NAMES)
    def test_laplacian_is_positive_semidefinite(
        self, complex_: dirac.SimplicialComplex
    ):
        for degree in range(complex_.top_dimension + 1):
            for value in dirac.hodge_laplacian(complex_, degree).eigenvals():
                assert complex(sp.N(value)).real >= -1e-9


class TestChiralSymmetry:
    """The first refutation, in three steps."""

    @pytest.mark.parametrize("complex_", COMPLEXES, ids=NAMES)
    def test_grading_is_an_involution(self, complex_: dirac.SimplicialComplex):
        grading = dirac.chirality_operator(complex_)
        assert (grading * grading - sp.eye(grading.rows)).is_zero_matrix

    @pytest.mark.parametrize("complex_", COMPLEXES, ids=NAMES)
    def test_anticommutes_with_dirac(self, complex_: dirac.SimplicialComplex):
        """``Gamma D + D Gamma = 0``, exactly."""
        assert dirac.chiral_anticommutator_residual(complex_).is_zero_matrix

    @pytest.mark.parametrize("complex_", COMPLEXES, ids=NAMES)
    def test_spectrum_is_symmetric_about_zero(
        self, complex_: dirac.SimplicialComplex
    ):
        assert dirac.spectrum_is_symmetric(complex_)

    @pytest.mark.parametrize("complex_", COMPLEXES, ids=NAMES)
    def test_the_refutation(self, complex_: dirac.SimplicialComplex):
        """``spec(Delta)`` alone reproduces ``spec(D)``.

        The Dirac operator adds no spectral information over the Hodge
        Laplacian.  This is the claim the brief needs to be false and is not.
        """
        assert dirac.spectra_are_equivalent(complex_)

    @pytest.mark.parametrize("complex_", COMPLEXES, ids=NAMES)
    def test_spectrum_size_matches_the_complex(
        self, complex_: dirac.SimplicialComplex
    ):
        assert len(dirac.dirac_spectrum(complex_)) == sum(complex_.dimensions)


class TestHomologyReferee:
    """Betti numbers nobody fed in."""

    @pytest.mark.parametrize(
        "complex_, betti",
        [
            (CIRCLE, (1, 1)),
            (DISK, (1, 0, 0)),
            (SPHERE, (1, 0, 1)),
            (TWO_EDGES, (2, 0)),
            (PATH, (1, 0)),
        ],
        ids=NAMES,
    )
    def test_betti_numbers(
        self, complex_: dirac.SimplicialComplex, betti: tuple[int, ...]
    ):
        assert dirac.betti_numbers(complex_) == betti

    @pytest.mark.parametrize("complex_", COMPLEXES, ids=NAMES)
    def test_kernel_dimension_is_the_total_betti_number(
        self, complex_: dirac.SimplicialComplex
    ):
        """``dim ker D = sum_k beta_k``, the Hodge theorem in one line."""
        assert dirac.kernel_dimension(complex_) == sum(
            dirac.betti_numbers(complex_)
        )

    def test_circle_has_a_loop_and_sphere_a_void(self):
        assert dirac.betti_numbers(CIRCLE)[1] == 1
        assert dirac.betti_numbers(SPHERE)[2] == 1
        assert dirac.betti_numbers(DISK)[1] == 0


class TestWhatSurvives:
    """The one part of the brief that is true."""

    @pytest.mark.parametrize("complex_", COMPLEXES, ids=NAMES)
    def test_homology_discards_information(
        self, complex_: dirac.SimplicialComplex
    ):
        """Every non-zero eigenvalue is thrown away by persistent homology."""
        assert dirac.homology_discards(complex_) > 0

    @pytest.mark.parametrize("complex_", COMPLEXES, ids=NAMES)
    def test_discarded_plus_kernel_is_the_whole_spectrum(
        self, complex_: dirac.SimplicialComplex
    ):
        assert dirac.homology_discards(complex_) + dirac.kernel_dimension(
            complex_
        ) == len(dirac.dirac_spectrum(complex_))

    def test_the_gap_is_laplacian_versus_homology_not_dirac_versus_laplacian(self):
        """Both halves of the correct statement, side by side."""
        assert dirac.homology_discards(SPHERE) > 0
        assert dirac.spectra_are_equivalent(SPHERE)


# ---------------------------------------------------------------------------
# chirality
# ---------------------------------------------------------------------------


class TestChiralityIsInvisible:
    """The second refutation, which does not involve the Dirac operator at all."""

    def test_the_configuration_is_genuinely_chiral(self):
        mirrored = dirac.reflect(dirac.CHIRAL_POINTS)
        assert dirac.chirality_signature(
            dirac.CHIRAL_POINTS
        ) != dirac.chirality_signature(mirrored)

    def test_every_orientation_flips(self):
        mirrored = dirac.reflect(dirac.CHIRAL_POINTS)
        original = dirac.chirality_signature(dirac.CHIRAL_POINTS)
        reflected = dirac.chirality_signature(mirrored)
        assert all(left == -right for left, right in zip(original, reflected))

    def test_distance_matrices_are_bitwise_identical(self):
        """Not close. Identical. Reflection is an isometry."""
        mirrored = dirac.reflect(dirac.CHIRAL_POINTS)
        assert dirac.distance_matrix(dirac.CHIRAL_POINTS) == dirac.distance_matrix(
            mirrored
        )

    @pytest.mark.parametrize("radius", [0.9, 1.1, 1.5, 2.0])
    def test_rips_complexes_are_identical(self, radius: float):
        mirrored = dirac.reflect(dirac.CHIRAL_POINTS)
        assert dirac.rips_complex(dirac.CHIRAL_POINTS, radius) == dirac.rips_complex(
            mirrored, radius
        )

    @pytest.mark.parametrize("radius", [1.1, 1.5])
    def test_dirac_spectra_are_identical(self, radius: float):
        mirrored = dirac.reflect(dirac.CHIRAL_POINTS)
        assert dirac.dirac_spectrum(
            dirac.rips_complex(dirac.CHIRAL_POINTS, radius)
        ) == dirac.dirac_spectrum(dirac.rips_complex(mirrored, radius))

    def test_the_headline(self):
        assert dirac.chirality_is_invisible_to_distances() is True

    def test_signed_volume_is_not_a_distance_function(self):
        """What flips, and what the filtration never receives."""
        mirrored = dirac.reflect(dirac.CHIRAL_POINTS)
        for indices in [(0, 1, 2, 3), (1, 2, 3, 4)]:
            assert dirac.signed_volume(
                dirac.CHIRAL_POINTS, indices
            ) == -dirac.signed_volume(mirrored, indices)

    def test_reflection_is_an_involution(self):
        twice = dirac.reflect(dirac.reflect(dirac.CHIRAL_POINTS))
        assert twice == tuple(tuple(point) for point in dirac.CHIRAL_POINTS)

    def test_validation(self):
        with pytest.raises(ValueError):
            dirac.reflect([])
        with pytest.raises(ValueError, match="axis"):
            dirac.reflect(dirac.CHIRAL_POINTS, axis=7)
        with pytest.raises(ValueError):
            dirac.distance_matrix([])
        with pytest.raises(ValueError):
            dirac.rips_complex(dirac.CHIRAL_POINTS, -1.0)
        with pytest.raises(ValueError, match="four indices"):
            dirac.signed_volume(dirac.CHIRAL_POINTS, (0, 1, 2))
        with pytest.raises(ValueError, match="at least four"):
            dirac.chirality_signature(dirac.CHIRAL_POINTS[:3])


class TestInheritedBlindspot:
    """Every Laplacian cospectrality is a Dirac cospectrality."""

    def test_the_pair_is_not_isomorphic_as_complexes(self):
        first, second = (
            dirac.graph_complex(dirac.MINIMAL_COSPECTRAL_ORDER, edges)
            for edges in dirac.COSPECTRAL_PAIR
        )
        assert first != second

    def test_the_pair_shares_a_dirac_spectrum(self):
        assert dirac.cospectral_pair_is_a_blindspot() is True

    def test_minimal_order(self):
        assert dirac.MINIMAL_COSPECTRAL_ORDER == 6

    def test_both_graphs_have_the_same_size(self):
        """Same vertex and edge counts, so the blindspot is not a trivial one."""
        first, second = dirac.COSPECTRAL_PAIR
        assert len(first) == len(second)


class TestWhatThisDoesNotClaim:
    """The boundary, asserted so it cannot drift."""

    def test_no_persistence_module_is_built(self):
        """Single complexes, not filtrations.

        The refutations do not need persistence: the first is a statement about
        one operator on one complex, and the second says every complex in the
        filtration is identical, which is stronger than any statement about the
        persistence module built on top of it.  A `persistent_dirac` here would
        be scope the argument does not require.
        """
        assert not hasattr(dirac, "persistence_diagram")
        assert not hasattr(dirac, "persistent_dirac")

    def test_eigenvectors_are_not_claimed_to_be_equivalent(self):
        """Only the spectra are shown equivalent, and only the spectra were claimed.

        The Dirac eigenvectors genuinely mix degrees in a way the Laplacian's do
        not, and nothing here says otherwise.  The brief's conjecture is about
        the spectrum, so the refutation is about the spectrum.
        """
        assert not hasattr(dirac, "eigenvectors_are_equivalent")
        assert dirac.spectra_are_equivalent(SPHERE)

    def test_chirality_detection_is_not_offered(self):
        """No method here detects chirality, because the point is that none can.

        `signed_volume` sees it, and it is not a function of pairwise distances,
        so it is not available to a distance-based filtration.
        """
        assert not hasattr(dirac, "detect_chirality")
        assert dirac.chirality_is_invisible_to_distances() is True
