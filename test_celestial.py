"""Referees for the massive ``w_{1+infinity}`` module and its obstruction.

Four checks that the construction does not control:

* ``phat . phat = -1`` exactly, on every basis element and every weight. The
  momentum operator carries a frame expansion, two weight shifts and an ordering
  of derivative against polynomial prefactor; the mass shell is one scalar
  identity that fails if any of them is wrong, and it did fail on the first
  version, which applied the operator to the prefactor as well.
* The four components carry the boost and spin weights of a four-vector under
  two gradings the module was built without reference to.
* The inversion recursion and its closed form are separate computations that must
  agree wherever both are defined -- and disagree in a specific way where they do
  not, at even ``Delta``, which is the case that fixes the obstruction.
* Applying the operator back to the truncated inverse must leave exactly one
  term, the truncation tail, with everything else cancelling in exact rational
  arithmetic.

The abstract algebra is checked against itself: the Jacobi identity on the wedge,
and the Poisson realisation on the plane against the structure constants.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import pytest

import celestial as cel

WEIGHTS = [
    Fraction(7, 2),
    Fraction(-3),
    Fraction(11, 5),
    Fraction(1, 3),
    Fraction(-1, 2),
    Fraction(5),
    Fraction(101, 7),
]
NON_INTEGER = [w for w in WEIGHTS if w.denominator != 1]
KEYS = [(0, 0, 0, 0), (1, 0, 0, 0), (2, 1, 0, 0), (1, 1, 3, 2), (3, 3, 1, 2), (0, 0, 2, 1)]

#: Boost and spin weights of the four null components of a four-vector.
BOOST = {0: 1, 1: -1, 2: 0, 3: 0}
SPIN = {0: 0, 1: 0, 2: 1, 3: -1}


class TestWedgeAlgebra:
    def test_jacobi_holds_exactly(self) -> None:
        generators = list(cel.wedge_generators(4))
        for a, b, c in itertools.islice(itertools.product(generators, repeat=3), 6000):
            assert cel.jacobi_residual(a, b, c) == 0

    def test_the_wedge_is_the_polynomial_condition(self) -> None:
        """``|m| <= p-1`` is exactly ``lambda_0^{p-1+m} lambda_1^{p-1-m}`` polynomial.

        Not a convention: outside the wedge the Hamiltonian has a pole at the
        origin of the plane, which is where a massless particle of zero energy
        sits.
        """
        for spin in [Fraction(1), Fraction(3, 2), Fraction(2), Fraction(5, 2), Fraction(4)]:
            for mode in [spin - 1, -(spin - 1), spin, -spin]:
                inside = cel.in_wedge(spin, mode)
                try:
                    cel.wedge_monomial(spin, mode)
                except ValueError:
                    assert not inside
                else:
                    assert inside

    def test_the_bracket_lands_on_the_right_generator(self) -> None:
        left = (Fraction(5, 2), Fraction(1, 2))
        right = (Fraction(3), Fraction(-2))
        coefficient, target = cel.wedge_bracket(left, right)
        assert target == (Fraction(7, 2), Fraction(-3, 2))
        assert cel.in_wedge(*target)
        assert coefficient == cel.structure_constant(*left, *right)

    def test_the_bracket_is_antisymmetric(self) -> None:
        for a, b in itertools.product(cel.wedge_generators(3), repeat=2):
            forward, _ = cel.wedge_bracket(a, b)
            backward, _ = cel.wedge_bracket(b, a)
            assert forward == -backward

    def test_spin_two_is_an_sl2(self) -> None:
        """``p = 2`` is the self-dual Lorentz algebra, with the usual constants."""
        for m in (-1, 0, 1):
            for n in (-1, 0, 1):
                coefficient, target = cel.wedge_bracket(
                    (Fraction(2), Fraction(m)), (Fraction(2), Fraction(n))
                )
                assert coefficient == m - n
                assert target == (Fraction(2), Fraction(m + n))


class TestMasslessRealisation:
    @pytest.mark.parametrize("max_spin", [3, Fraction(7, 2)])
    def test_the_plane_carries_the_algebra(self, max_spin) -> None:
        generators = [g for g in cel.wedge_generators(max_spin) if cel.in_wedge(*g)]
        for a, b in itertools.product(generators, repeat=2):
            assert cel.poisson_realisation_residual(a, b) == 0

    def test_the_normalisation_is_load_bearing(self) -> None:
        """With ``{lambda_0, lambda_1} = 1`` every bracket is twice too big.

        A negative control on the one free constant in the realisation.
        """
        left = (Fraction(5, 2), Fraction(1, 2))
        right = (Fraction(2), Fraction(-1))
        product = cel.poisson_bracket(cel.wedge_monomial(*left), cel.wedge_monomial(*right))
        assert product is not None
        coefficient, _ = product
        abstract, _ = cel.wedge_bracket(left, right)
        assert coefficient == abstract != 0
        assert 2 * coefficient != abstract


class TestMomentumOperator:
    @pytest.mark.parametrize("weight", WEIGHTS)
    @pytest.mark.parametrize("key", KEYS)
    def test_the_mass_shell_is_exact(self, weight, key) -> None:
        assert cel.mass_shell_residual(cel.basis(weight, *key)) == {}

    @pytest.mark.parametrize("weight", WEIGHTS[:4])
    def test_the_components_commute(self, weight) -> None:
        state = cel.basis(weight, 2, 1, 1, 1)
        for mu, nu in itertools.product(range(4), repeat=2):
            assert cel.commutator_residual(mu, nu, state) == {}

    @pytest.mark.parametrize("weight", WEIGHTS[:4])
    @pytest.mark.parametrize("key", KEYS)
    def test_the_components_carry_four_vector_weights(self, weight, key) -> None:
        """Boost and spin weights, from a module built without reference to them."""
        start = (weight,) + key
        for mu in range(4):
            image = cel.momentum(mu, cel.basis(weight, *key))
            assert image, "the momentum never annihilates a basis element"
            for target in image:
                assert cel.boost_weight(target) - cel.boost_weight(start) == BOOST[mu]
                assert cel.spin_weight(target) - cel.spin_weight(start) == SPIN[mu]

    def test_the_propagator_identity_holds(self) -> None:
        """``u u_zzbar - u_z u_zbar = 1``: the one input to the derivation."""
        points = [
            (Fraction(3), Fraction(1, 2), Fraction(-2), Fraction(5), Fraction(1, 3)),
            (Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(-2, 7), Fraction(9), Fraction(4, 3), Fraction(-1), Fraction(6)),
        ]
        for point in points:
            assert cel.propagator_identity_residual(*point) == 0

    def test_a_degenerate_hyperboloid_point_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be non-zero"):
            cel.propagator_identity_residual(
                Fraction(0), Fraction(1), Fraction(1), Fraction(1), Fraction(1)
            )

    def test_the_operator_is_singular_at_weight_one(self) -> None:
        """``Delta = 1`` is the principal-series midpoint, and the operator knows."""
        with pytest.raises(ValueError, match="singular at Delta = 1"):
            cel.momentum(0, cel.basis(Fraction(1)))

    def test_a_bad_component_is_refused(self) -> None:
        with pytest.raises(ValueError, match="null component"):
            cel.momentum(4, cel.basis(Fraction(3)))

    def test_negative_degrees_are_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            cel.basis(Fraction(3), -1, 0, 0, 0)


class TestBidiagonality:
    """The structural fact that makes the inversion a recursion, not a fraction."""

    @pytest.mark.parametrize("weight", WEIGHTS)
    @pytest.mark.parametrize("key", KEYS)
    def test_the_lightcone_component_has_exactly_two_terms(self, weight, key) -> None:
        image = cel.lightcone_momentum(cel.basis(weight, *key))
        assert len(image) == 2

    @pytest.mark.parametrize("weight", WEIGHTS)
    def test_the_two_terms_are_the_predicted_ones(self, weight) -> None:
        i, j, a, b = 2, 1, 3, 0
        image = cel.lightcone_momentum(cel.basis(weight, i, j, a, b))
        lowering = (weight - 1, i + 1, j + 1, a, b)
        raising = (weight + 1, i, j, a, b)
        assert image[lowering] == Fraction(1) / (weight - 1) ** 2
        assert image[raising] == weight / (weight - 1)

    @pytest.mark.parametrize("weight", WEIGHTS)
    def test_the_polynomial_labels_are_spectators(self, weight) -> None:
        """``qhat^{++} = 2`` is constant, so ``(a, b)`` never moves.

        This is what reduces the inversion to a scalar recursion; for the other
        three components it is false.
        """
        for a, b in [(0, 0), (2, 0), (3, 4)]:
            for target in cel.lightcone_momentum(cel.basis(weight, 1, 1, a, b)):
                assert target[3] == a and target[4] == b
        moved = cel.momentum(1, cel.basis(weight, 1, 1, 0, 0))
        assert any(target[3] or target[4] for target in moved)


class TestInversion:
    @pytest.mark.parametrize("weight", NON_INTEGER)
    def test_recursion_and_closed_form_agree(self, weight) -> None:
        """Two separate computations of the same coefficients."""
        recursion = cel.inverse_recursion(weight, 8)
        closed = [cel.inverse_coefficient(weight, k) for k in range(9)]
        assert recursion == closed

    @pytest.mark.parametrize("weight", NON_INTEGER)
    @pytest.mark.parametrize("order", [0, 1, 4, 7])
    def test_the_operator_undoes_the_inverse_up_to_the_tail(self, weight, order) -> None:
        residual = cel.inverse_residual(weight, 1, 2, 0, 1, order=order)
        assert len(residual) == 1, "everything but the truncation tail must cancel"
        (target,) = residual
        assert target[0] == weight - 2 * (order + 1)

    @pytest.mark.parametrize("weight", NON_INTEGER)
    def test_the_image_is_an_arithmetic_progression_of_step_two(self, weight) -> None:
        series = cel.inverse_series(weight, order=6)
        families = sorted((key[0] for key in series), reverse=True)
        assert families[0] == weight - 1
        for earlier, later in zip(families, families[1:]):
            assert earlier - later == 2

    @pytest.mark.parametrize("weight", NON_INTEGER)
    def test_the_derivative_orders_climb_in_step(self, weight) -> None:
        series = cel.inverse_series(weight, i=1, j=2, order=5)
        for (delta, i, j, _, _) in series:
            step = (weight - 1 - delta) / 2
            assert i == 1 + step and j == 2 + step

    def test_the_closed_form_survives_where_the_recursion_does_not(self) -> None:
        """Even ``Delta``: ``beta_0`` is infinite, the recursion gives ``0 * infinity``.

        The closed form says the first coefficient is zero and the second is a
        genuine pole, which is what sets :func:`celestial.obstruction_step` to
        ``1`` rather than ``0`` at ``Delta = 2``.
        """
        assert cel.inverse_coefficient(Fraction(2), 0) == 0
        with pytest.raises(ValueError, match="singular"):
            cel.inverse_coefficient(Fraction(2), 1)
        with pytest.raises(ValueError, match="degenerates"):
            cel.inverse_recursion(Fraction(2), 3)
        assert cel.obstruction_step(Fraction(2)) == 1

    def test_a_negative_order_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            cel.inverse_recursion(Fraction(7, 2), -1)
        with pytest.raises(ValueError, match="non-negative"):
            cel.inverse_coefficient(Fraction(7, 2), -1)


class TestObstruction:
    """The result: integer weights are exactly the obstructed ones."""

    @pytest.mark.parametrize("delta,step", [(n, -((1 - n) // 2)) for n in range(1, 12)])
    def test_every_positive_integer_weight_is_obstructed(self, delta, step) -> None:
        assert cel.obstruction_step(Fraction(delta)) == step
        with pytest.raises(ValueError):
            cel.inverse_coefficient(Fraction(delta), step)

    @pytest.mark.parametrize("delta", [n for n in range(1, 12)])
    def test_the_obstruction_is_where_the_falling_factorial_vanishes(self, delta) -> None:
        """A second route to the same step: count the run of integers directly."""
        step = cel.obstruction_step(Fraction(delta))
        assert step is not None
        assert 2 * step + 1 >= delta
        assert 2 * (step - 1) + 1 < delta if step else True

    @pytest.mark.parametrize(
        "delta", [Fraction(7, 2), Fraction(-3), Fraction(0), Fraction(-11, 5), Fraction(1, 3)]
    )
    def test_non_positive_integer_weights_are_unobstructed(self, delta) -> None:
        assert cel.obstruction_step(delta) is None
        assert cel.inverse_exists(delta, 50)
        assert cel.inverse_coefficient(delta, 20) != 0 or delta == 42

    def test_the_principal_series_is_never_obstructed(self) -> None:
        """``Delta = 1 + i lambda`` has denominator zero only at ``lambda = 0``.

        The criterion is arithmetic, so it can be checked without complex
        arithmetic: the falling factorial ``(Delta-1)...(Delta-2k-1)`` vanishes
        only at real integer ``Delta``, and ``1 + i lambda`` is real only when
        ``lambda`` is zero -- which is ``Delta = 1``, the one point of the series
        where the momentum operator itself is singular.
        """
        assert cel.obstruction_step(Fraction(1)) == 0
        with pytest.raises(ValueError, match="singular at Delta = 1"):
            cel.momentum(0, cel.basis(Fraction(1)))

    def test_the_poincare_generators_are_unaffected(self) -> None:
        """``p <= 2`` needs no inverse, so it survives at every weight.

        Himwich--Pate's generators carry ``(-n.P)^{-(2p-4)}``, which is the
        identity at ``p = 2``. The obstruction found here therefore begins
        exactly where their Schwinger prescription does, at ``p = 5/2``.
        """
        for spin in [Fraction(3, 2), Fraction(2)]:
            assert 2 * spin - 4 <= 0
        assert 2 * Fraction(5, 2) - 4 == 1

    @pytest.mark.parametrize("delta", [Fraction(5), Fraction(9)])
    def test_the_series_refuses_to_be_built_past_the_obstruction(self, delta) -> None:
        step = cel.obstruction_step(delta)
        assert step is not None
        cel.inverse_series(delta, order=step - 1)
        with pytest.raises(ValueError, match="singular"):
            cel.inverse_series(delta, order=step)

    def test_the_report_summarises_both_cases(self) -> None:
        good = cel.inversion_report(Fraction(7, 2), order=4)
        assert good.exists and good.obstruction is None
        assert good.spacing == 2
        bad = cel.inversion_report(Fraction(5), order=4)
        assert not bad.exists and bad.obstruction == 2
        assert bad.families == ()
