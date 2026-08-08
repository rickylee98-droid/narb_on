"""Tests for rank loss in the Einstein constraint equations on the flat torus.

The conventions of the linearised constraint operator are pinned by two things
it cannot satisfy by accident: pure-gauge perturbations must lie in its kernel,
which is diffeomorphism invariance, and its adjoint must satisfy the pairing
identity against it with the symmetric-tensor multiplicities in place. Get a sign
or an index wrong and both fail.

Everything is exact. Fourier modes on the torus are indexed by integer vectors,
so the mode matrices are integral and their ranks are combinatorial facts rather
than decisions about how small a singular value has to be. Near a rank-loss point
that distinction is the whole game.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import pytest

import adm

SMALL_MODES = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
    (2, 0, 0),
    (2, -1, 3),
    (3, 1, -2),
]
NONZERO_MODES = [mode for mode in SMALL_MODES if mode != (0, 0, 0)]


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #
class TestSymmetricSlots:
    def test_every_index_pair_maps_into_the_slots(self) -> None:
        for i in range(3):
            for j in range(3):
                slot = adm.symmetric_index(i, j)
                assert adm.SYMMETRIC_PAIRS[slot] == (min(i, j), max(i, j))

    def test_symmetry_of_the_index_map(self) -> None:
        for i in range(3):
            for j in range(3):
                assert adm.symmetric_index(i, j) == adm.symmetric_index(j, i)

    def test_multiplicities_sum_to_nine(self) -> None:
        """The six slots stand for the nine entries of a 3x3 tensor."""
        assert sum(adm._multiplicity(slot) for slot in range(6)) == 9


# --------------------------------------------------------------------------- #
# The two referees on the conventions
# --------------------------------------------------------------------------- #
class TestGaugeInvariance:
    """Pure gauge must be annihilated -- the sharpest check on the conventions."""

    @pytest.mark.parametrize("wave", SMALL_MODES)
    def test_lie_derivatives_lie_in_the_kernel(self, wave) -> None:
        matrix = adm.linearised_constraint_matrix(wave)
        for direction in adm.gauge_directions(wave):
            image = [
                sum(matrix[row][col] * direction[col] for col in range(12))
                for row in range(4)
            ]
            assert image == [0, 0, 0, 0]

    def test_an_arbitrary_shift_vector_is_also_gauge(self) -> None:
        rng = random.Random(0)
        for _ in range(40):
            wave = [rng.randint(-3, 3) for _ in range(3)]
            vector = [rng.randint(-3, 3) for _ in range(3)]
            slots = adm.lie_derivative_mode(wave, vector) + [0] * 6
            matrix = adm.linearised_constraint_matrix(wave)
            image = [
                sum(matrix[row][col] * slots[col] for col in range(12))
                for row in range(4)
            ]
            assert image == [0, 0, 0, 0]

    def test_the_check_is_not_vacuous(self) -> None:
        """A generic perturbation is *not* in the kernel, so the test can fail."""
        wave = (1, 2, 0)
        matrix = adm.linearised_constraint_matrix(wave)
        generic = [1, 0, 0, 0, 0, 0] + [0] * 6
        image = [
            sum(matrix[row][col] * generic[col] for col in range(12))
            for row in range(4)
        ]
        assert any(image)

    @pytest.mark.parametrize("bad", [[1, 2], [1, 2, 3, 4]])
    def test_rejects_a_wrong_length_wave(self, bad) -> None:
        with pytest.raises(ValueError, match="three components"):
            adm.linearised_constraint_matrix(bad)


class TestAdjoint:
    """The pairing identity, with the tensor multiplicities explicit.

    The adjoint is deliberately not the naive transpose: the inner product on
    symmetric tensors weights an off-diagonal slot by two, and the constraint
    matrix carries that weight in its own entries. A transpose comparison fails,
    and should.
    """

    @pytest.mark.parametrize("wave", SMALL_MODES)
    def test_pairing_identity(self, wave) -> None:
        rng = random.Random(hash(wave) & 0xFFFF)
        matrix = adm.linearised_constraint_matrix(wave)
        adjoint = adm.kid_matrix(wave)
        for _ in range(8):
            field = [Fraction(rng.randint(-4, 4)) for _ in range(12)]
            dual = [Fraction(rng.randint(-4, 4)) for _ in range(4)]
            left = sum(
                dual[row] * sum(matrix[row][col] * field[col] for col in range(12))
                for row in range(4)
            )
            right = sum(
                adm._multiplicity(slot % 6)
                * field[slot]
                * sum(adjoint[slot][col] * dual[col] for col in range(4))
                for slot in range(12)
            )
            assert left == right

    def test_the_naive_transpose_does_not_hold(self) -> None:
        """Recording why the identity needs the multiplicities."""
        wave = (1, 1, 0)
        matrix = adm.linearised_constraint_matrix(wave)
        adjoint = adm.kid_matrix(wave)
        mismatches = sum(
            1
            for row in range(4)
            for col in range(12)
            if matrix[row][col] != adjoint[col][row]
        )
        assert mismatches > 0

    @pytest.mark.parametrize("bad", [[0, 0], [0, 0, 0, 0]])
    def test_rejects_a_wrong_length_wave(self, bad) -> None:
        with pytest.raises(ValueError, match="three components"):
            adm.kid_matrix(bad)


# --------------------------------------------------------------------------- #
# Exact linear algebra
# --------------------------------------------------------------------------- #
class TestExactLinearAlgebra:
    def test_rank_of_a_known_matrix(self) -> None:
        assert adm.integer_rank([[1, 2], [2, 4]]) == 1
        assert adm.integer_rank([[1, 0], [0, 1]]) == 2
        assert adm.integer_rank([[0, 0], [0, 0]]) == 0

    def test_rank_plus_nullity(self) -> None:
        for wave in SMALL_MODES:
            matrix = adm.linearised_constraint_matrix(wave)
            rank = adm.integer_rank(matrix)
            kernel = adm.integer_kernel(matrix)
            assert rank + len(kernel) == 12

    def test_kernel_vectors_really_are_annihilated(self) -> None:
        for wave in SMALL_MODES:
            matrix = adm.linearised_constraint_matrix(wave)
            for vector in adm.integer_kernel(matrix):
                image = [
                    sum(matrix[row][col] * vector[col] for col in range(12))
                    for row in range(4)
                ]
                assert image == [0, 0, 0, 0]

    def test_kernel_basis_is_independent(self) -> None:
        for wave in NONZERO_MODES:
            basis = adm.integer_kernel(adm.linearised_constraint_matrix(wave))
            assert adm.integer_rank([[e for e in row] for row in basis]) == len(basis)


# --------------------------------------------------------------------------- #
# Where the rank drops
# --------------------------------------------------------------------------- #
class TestRankLoss:
    @pytest.mark.parametrize("wave", NONZERO_MODES)
    def test_every_non_zero_mode_has_full_rank(self, wave) -> None:
        report = adm.mode_report(wave)
        assert report.rank == 4
        assert report.deficiency == 0
        assert not report.loses_rank
        assert report.kid_dimension == 0

    def test_the_zero_mode_loses_all_of_it(self) -> None:
        report = adm.mode_report((0, 0, 0))
        assert report.is_zero_mode
        assert report.rank == 0
        assert report.deficiency == 4
        assert report.loses_rank

    def test_deficiency_equals_the_kid_dimension_at_every_mode(self) -> None:
        """The Fischer--Marsden correspondence, as an exact integer identity.

        Rank loss of the constraint map and existence of Killing initial data are
        the same phenomenon, and here they are the same number.
        """
        for report in adm.sweep_modes(3):
            assert report.deficiency == report.kid_dimension

    def test_the_kid_count_is_one_lapse_and_three_translations(self) -> None:
        assert adm.mode_report((0, 0, 0)).kid_dimension == 4

    def test_rank_loss_is_confined_to_a_single_mode(self) -> None:
        """Which is why the obstruction is an integral and not a pointwise condition."""
        losing = [r.wave for r in adm.sweep_modes(3) if r.loses_rank]
        assert losing == [(0, 0, 0)]

    def test_the_sweep_covers_the_whole_box(self) -> None:
        assert sum(1 for _ in adm.sweep_modes(2)) == 5**3

    def test_rejects_a_negative_limit(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            list(adm.sweep_modes(-1))


# --------------------------------------------------------------------------- #
# Transverse-traceless perturbations and the obstruction
# --------------------------------------------------------------------------- #
class TestTransverseTraceless:
    @pytest.mark.parametrize("wave", NONZERO_MODES)
    def test_there_are_exactly_two_polarisations(self, wave) -> None:
        assert len(adm.transverse_traceless_modes(wave)) == 2

    @pytest.mark.parametrize("wave", NONZERO_MODES)
    def test_they_are_traceless_and_transverse(self, wave) -> None:
        for tensor in adm.transverse_traceless_modes(wave):
            trace = sum(tensor[adm.symmetric_index(i, i)] for i in range(3))
            assert trace == 0
            for component in range(3):
                divergence = Fraction(0)
                for slot, (i, j) in enumerate(adm.SYMMETRIC_PAIRS):
                    if i == component:
                        divergence += wave[j] * tensor[slot]
                    if j == component and i != j:
                        divergence += wave[i] * tensor[slot]
                assert divergence == 0

    @pytest.mark.parametrize("wave", NONZERO_MODES)
    def test_they_solve_the_linearised_constraints(self, wave) -> None:
        """With ``h = 0`` a transverse-traceless ``k`` is an exact solution."""
        matrix = adm.linearised_constraint_matrix(wave)
        for tensor in adm.transverse_traceless_modes(wave):
            field = [Fraction(0)] * 6 + list(tensor)
            image = [
                sum(matrix[row][col] * field[col] for col in range(12))
                for row in range(4)
            ]
            assert image == [0, 0, 0, 0]


class TestObstruction:
    @pytest.mark.parametrize("wave", NONZERO_MODES)
    def test_the_obstruction_is_strictly_negative_on_every_polarisation(
        self, wave
    ) -> None:
        """Linearisation instability, exhibited.

        These perturbations solve the linearised constraints exactly, and the
        second-order condition they would have to satisfy in order to be tangent
        to a genuine solution is strictly negative -- never zero. So they do not
        integrate: the linear theory predicts a deformation that the nonlinear
        theory does not admit.
        """
        for tensor in adm.transverse_traceless_modes(wave):
            value = adm.obstruction_value({wave: tensor})
            assert value < 0

    def test_a_trace_free_perturbation_gives_minus_half_its_norm(self) -> None:
        """With ``tr k = 0`` the obstruction is ``-|k|^2/2``, the half from ``<cos^2>``.

        This test previously asserted ``-2``, the value before the torus average
        was normalised. The half makes no difference to a claim about the sign of
        this term alone, but it fixes the relative weight against the metric term
        of :func:`adm.metric_obstruction_term`, which is what makes the two
        addable.
        """
        tensor = [Fraction(0), Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0)]
        # one off-diagonal slot, multiplicity two, so |k|^2 = 2 and the value is -1
        assert adm.obstruction_value({(1, 0, 0): tensor}) == -1

    def test_a_pure_trace_perturbation_gives_a_positive_value(self) -> None:
        """The form is not negative-definite on all tensors, only on trace-free ones.

        ``(tr k)^2 - |k|^2`` is positive for ``k`` proportional to the identity,
        so the sign result is genuinely about the transverse-traceless sector
        rather than an artefact of the expression.
        """
        identity = [Fraction(1), Fraction(0), Fraction(0), Fraction(1), Fraction(0), Fraction(1)]
        assert adm.obstruction_value({(1, 0, 0): identity}) > 0

    def test_it_is_quadratic(self) -> None:
        tensor = adm.transverse_traceless_modes((1, 1, 0))[0]
        single = adm.obstruction_value({(1, 1, 0): tensor})
        doubled = adm.obstruction_value(
            {(1, 1, 0): [2 * entry for entry in tensor]}
        )
        assert doubled == 4 * single

    def test_modes_add(self) -> None:
        first = adm.transverse_traceless_modes((1, 0, 0))[0]
        second = adm.transverse_traceless_modes((0, 1, 0))[0]
        combined = adm.obstruction_value({(1, 0, 0): first, (0, 1, 0): second})
        separate = adm.obstruction_value({(1, 0, 0): first}) + adm.obstruction_value(
            {(0, 1, 0): second}
        )
        assert combined == separate

    def test_rejects_a_malformed_mode(self) -> None:
        with pytest.raises(ValueError, match="six slots"):
            adm.obstruction_value({(1, 0, 0): [1, 2, 3]})

    def test_the_empty_perturbation_is_unobstructed(self) -> None:
        assert adm.obstruction_value({}) == 0


# --------------------------------------------------------------------------- #
# The metric term, and the full obstruction
# --------------------------------------------------------------------------- #
class TestMetricObstruction:
    """``<R^{(2)}>`` derived symbolically and refereed at first order.

    The expansion was obtained by computing the scalar curvature of
    ``delta + eps A cos(k.x)`` to second order with a computer algebra system and
    matching against the four quadratic invariants; the match is exact. The same
    pipeline taken to *first* order reproduces ``DH`` from
    :func:`adm.linearised_constraint_matrix` as an identical polynomial, which is
    what gives the second-order coefficient its standing.
    """

    @pytest.mark.parametrize("wave", NONZERO_MODES)
    def test_negative_on_transverse_traceless_metric_perturbations(
        self, wave
    ) -> None:
        """With ``tr A = 0`` and ``A k = 0`` only the ``-|k|^2|A|^2/8`` term survives."""
        for tensor in adm.transverse_traceless_modes(wave):
            assert adm.metric_obstruction_term(wave, tensor) < 0

    def test_matches_the_closed_form_on_a_trace_free_transverse_case(self) -> None:
        wave = (1, 0, 0)
        tensor = adm.transverse_traceless_modes(wave)[0]
        norm = sum(
            adm._multiplicity(slot) * tensor[slot] ** 2 for slot in range(6)
        )
        assert adm.metric_obstruction_term(wave, tensor) == -Fraction(1, 8) * 1 * norm

    def test_vanishes_at_the_zero_mode(self) -> None:
        tensor = [Fraction(1), Fraction(2), Fraction(0), Fraction(1), Fraction(0), Fraction(3)]
        assert adm.metric_obstruction_term((0, 0, 0), tensor) == 0

    def test_is_quadratic(self) -> None:
        wave = (1, 2, 0)
        tensor = [Fraction(1), Fraction(-1), Fraction(2), Fraction(0), Fraction(1), Fraction(3)]
        single = adm.metric_obstruction_term(wave, tensor)
        doubled = adm.metric_obstruction_term(wave, [3 * e for e in tensor])
        assert doubled == 9 * single

    def test_rejects_a_malformed_mode(self) -> None:
        with pytest.raises(ValueError, match="six slots"):
            adm.metric_obstruction_term((1, 0, 0), [1, 2, 3])


class TestGaugeSpan:
    @pytest.mark.parametrize("wave", NONZERO_MODES)
    def test_the_lapse_direction_solves_the_constraints(self, wave) -> None:
        matrix = adm.linearised_constraint_matrix(wave)
        direction = adm.lapse_gauge_direction(wave)
        image = [
            sum(matrix[row][col] * direction[col] for col in range(12))
            for row in range(4)
        ]
        assert image == [0, 0, 0, 0]

    @pytest.mark.parametrize("wave", NONZERO_MODES)
    def test_there_are_four_gauge_directions_inside_an_eight_dimensional_kernel(
        self, wave
    ) -> None:
        span = adm.gauge_span(wave)
        kernel = adm.linearised_solutions(wave)
        assert len(span) == 4
        assert len(kernel) == 8
        assert adm.integer_rank([list(row) for row in span]) == 4

    @pytest.mark.parametrize("wave", NONZERO_MODES)
    def test_gauge_directions_lie_in_the_kernel(self, wave) -> None:
        matrix = adm.linearised_constraint_matrix(wave)
        for direction in adm.gauge_span(wave):
            image = [
                sum(matrix[row][col] * direction[col] for col in range(12))
                for row in range(4)
            ]
            assert image == [0, 0, 0, 0]


class TestObstructionForm:
    @pytest.mark.parametrize("wave", NONZERO_MODES[:5])
    def test_the_form_is_symmetric(self, wave) -> None:
        form = adm.obstruction_form(wave)
        for a in range(12):
            for b in range(12):
                assert form[a][b] == form[b][a]

    @pytest.mark.parametrize("wave", NONZERO_MODES[:5])
    def test_the_form_reproduces_the_value(self, wave) -> None:
        """Polarisation must invert: the matrix and the function cannot disagree."""
        rng = random.Random(hash(wave) & 0xFFFF)
        form = adm.obstruction_form(wave)
        for _ in range(5):
            vector = [Fraction(rng.randint(-3, 3)) for _ in range(12)]
            quadratic = sum(
                vector[a] * form[a][b] * vector[b] for a in range(12) for b in range(12)
            )
            direct = adm.obstruction_value(
                {wave: vector[6:]}, {wave: vector[:6]}
            )
            assert quadratic == direct

    @pytest.mark.parametrize("wave", NONZERO_MODES[:5])
    def test_gauge_lies_in_the_radical_on_the_solution_space(self, wave) -> None:
        """Pure gauge changes nothing physical, so it cannot be obstructed."""
        form = adm.obstruction_form(wave)
        kernel = adm.linearised_solutions(wave)
        for direction in adm.gauge_span(wave):
            for solution in kernel:
                paired = sum(
                    direction[a] * form[a][b] * solution[b]
                    for a in range(12)
                    for b in range(12)
                )
                assert paired == 0


class TestRigidity:
    """The complete statement, which the extrinsic-curvature-only case could not reach."""

    @pytest.mark.parametrize("wave", NONZERO_MODES)
    def test_the_signature_is_negative_semi_definite_with_gauge_radical(
        self, wave
    ) -> None:
        """Inertia ``(4, 4, 0)``: four negative directions, four null, none positive.

        The four null directions are exactly the gauge span. So the obstruction
        is strictly negative on every linearised solution that is not pure gauge,
        and no such solution integrates. This is the full rigidity of the flat
        torus, metric perturbations included -- the case the
        curvature-only computation left open.
        """
        negative, zero, positive = adm.physical_signature(wave)
        assert positive == 0
        assert zero == len(adm.gauge_span(wave)) == 4
        assert negative == 4
        assert negative + zero == len(adm.linearised_solutions(wave))

    @pytest.mark.parametrize("wave", NONZERO_MODES[:4])
    def test_a_non_gauge_solution_is_strictly_obstructed(self, wave) -> None:
        span = adm.gauge_span(wave)
        base_rank = adm.integer_rank([list(row) for row in span])
        for solution in adm.linearised_solutions(wave):
            extended = [list(row) for row in span] + [list(solution)]
            if adm.integer_rank(extended) > base_rank:
                value = adm.obstruction_value(
                    {wave: list(solution[6:])}, {wave: list(solution[:6])}
                )
                assert value < 0
                return
        pytest.fail("every kernel direction was gauge, which contradicts dim 8 > 4")

    @pytest.mark.parametrize("wave", NONZERO_MODES[:4])
    def test_pure_gauge_is_unobstructed(self, wave) -> None:
        for direction in adm.gauge_span(wave):
            value = adm.obstruction_value(
                {wave: list(direction[6:])}, {wave: list(direction[:6])}
            )
            assert value == 0


class TestSignatureHelper:
    def test_known_inertias(self) -> None:
        assert adm._congruence_signature([[Fraction(1)]]) == (0, 0, 1)
        assert adm._congruence_signature([[Fraction(-1)]]) == (1, 0, 0)
        assert adm._congruence_signature([[Fraction(0)]]) == (0, 1, 0)

    def test_a_hyperbolic_pair_has_one_of_each_sign(self) -> None:
        """No non-zero diagonal entry, yet the form is indefinite."""
        matrix = [[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]]
        assert adm._congruence_signature(matrix) == (1, 0, 1)

    def test_diagonal_matrices_are_read_off(self) -> None:
        matrix = [[Fraction(0)] * 3 for _ in range(3)]
        matrix[0][0] = Fraction(-2)
        matrix[1][1] = Fraction(5)
        assert adm._congruence_signature(matrix) == (1, 1, 1)
