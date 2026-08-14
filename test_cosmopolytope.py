"""Referees for the cosmological polytope and the flat-space mass resummation.

Four checks that the module does not control:

* The facets computed by Normaliz from the ``3E`` generators must match an
  independent enumeration of connected subgraphs -- on loop graphs as well as
  trees, where the rule genuinely differs, since an edge left out of a subgraph
  with both endpoints inside contributes ``2 y_e``. Dropping that gives the right
  answer on every tree and the wrong one on every loop.
* The canonical forms must reproduce the published two-site, three-site and
  one-loop bubble wavefunctions, and the residue on the total-energy pole must be
  the flat-space scattering amplitude.
* The resummation constant is fitted once, at order ``m^2``, and then tested at
  ``m^4`` and ``m^6``. Nothing is adjusted between orders.
* The pole orders derived from the closed form are checked against the orders
  actually present in the polytope computation, which knows nothing about the
  closed form. The naive count -- one power per collapsing facet -- is kept as a
  test too, precisely because it fails at ``a = 3``.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
import sympy as sp

import cosmopolytope as cp

X1 = sp.Symbol("x1", positive=True)
X2 = sp.Symbol("x2", positive=True)
Y = sp.Symbol("y", positive=True)
M2 = sp.Symbol("m2", positive=True)

GRAPHS = {
    "2-site": cp.CHAIN(2),
    "3-chain": cp.CHAIN(3),
    "4-chain": cp.CHAIN(4),
    "star": cp.STAR,
    "bubble": cp.BUBBLE,
    "triangle": cp.TRIANGLE,
    "box": cp.BOX,
}


class TestFacetTheorem:
    @pytest.mark.parametrize("name", sorted(GRAPHS))
    def test_facets_are_the_subgraph_energies(self, name) -> None:
        unexpected, missing = cp.facet_theorem_residual(GRAPHS[name])
        assert unexpected == set()
        assert missing == set()

    @pytest.mark.parametrize("name", ["bubble", "triangle", "box"])
    def test_loop_graphs_need_the_double_boundary_rule(self, name) -> None:
        """A vertex-only enumeration misses facets on every loop graph.

        The extra facets carry ``2 y_e`` for an edge whose two endpoints are both
        inside the subgraph but which is not itself in it. If the correspondence
        were checked only on trees this distinction would never show up, and the
        check would be vacuous.
        """
        graph = GRAPHS[name]
        full = cp.subgraph_facets(graph)
        vertex_only = {
            facet
            for facet in full
            if all(entry <= 1 for entry in facet[graph.sites :])
        }
        assert vertex_only < full
        assert cp.support_hyperplanes(graph) == full

    def test_trees_do_not_distinguish_the_two_rules(self) -> None:
        for name in ["2-site", "3-chain", "4-chain", "star"]:
            graph = GRAPHS[name]
            full = cp.subgraph_facets(graph)
            assert all(entry <= 1 for facet in full for entry in facet[graph.sites :])

    @pytest.mark.parametrize("name", sorted(GRAPHS))
    def test_there_are_three_generators_per_edge(self, name) -> None:
        graph = GRAPHS[name]
        assert len(cp.cosmological_vertices(graph)) == 3 * len(graph.edges)

    def test_a_self_loop_is_refused(self) -> None:
        with pytest.raises(ValueError, match="self-loops"):
            cp.Graph(2, ((0, 0),))

    def test_an_out_of_range_edge_is_refused(self) -> None:
        with pytest.raises(ValueError, match="leaves the site range"):
            cp.Graph(2, ((0, 5),))


class TestCanonicalForm:
    def test_the_two_site_wavefunction(self) -> None:
        """The simplex: three facets, and the published answer."""
        x0, x1, y0 = sp.symbols("x0 x1 y0")
        got = cp.canonical_form(cp.CHAIN(2), [x0, x1, y0])
        want = 2 / ((x0 + x1) * (x0 + y0) * (x1 + y0))
        assert sp.simplify(got - want) == 0

    def test_the_three_site_wavefunction(self) -> None:
        x0, x1, x2, y0, y1 = sp.symbols("x0 x1 x2 y0 y1")
        got = cp.canonical_form(cp.CHAIN(3), [x0, x1, x2, y0, y1])
        want = (
            4
            * (x0 + 2 * x1 + x2 + y0 + y1)
            / (
                (x0 + y0)
                * (x2 + y1)
                * (x0 + x1 + x2)
                * (x0 + x1 + y1)
                * (x1 + x2 + y0)
                * (x1 + y0 + y1)
            )
        )
        assert sp.simplify(got - want) == 0

    def test_the_one_loop_bubble(self) -> None:
        """A loop graph, where the facet rule is non-trivial."""
        x0, x1, y0, y1 = sp.symbols("x0 x1 y0 y1")
        got = cp.canonical_form(cp.BUBBLE, [x0, x1, y0, y1])
        want = (
            8
            * (x0 + x1 + y0 + y1)
            / (
                (x0 + x1)
                * (x0 + x1 + 2 * y0)
                * (x0 + x1 + 2 * y1)
                * (x0 + y0 + y1)
                * (x1 + y0 + y1)
            )
        )
        assert sp.simplify(got - want) == 0

    def test_the_total_energy_residue_is_the_amplitude(self) -> None:
        """On the momentum-conserving locus this is the flat-space propagator."""
        x0, x1, y0 = sp.symbols("x0 x1 y0")
        residue = cp.total_energy_residue(cp.CHAIN(2))
        assert sp.simplify(residue - 2 / ((x0 + y0) * (x1 + y0))) == 0
        on_shell = sp.simplify(residue.subs(x1, -x0))
        assert sp.simplify(on_shell - 2 / (y0**2 - x0**2)) == 0

    def test_the_wrong_number_of_variables_is_refused(self) -> None:
        with pytest.raises(ValueError, match="kinematic variables"):
            cp.canonical_form(cp.CHAIN(2), [sp.Integer(1), sp.Integer(1)])

    def test_a_degenerate_substitution_is_refused(self) -> None:
        """All variables zero collapses every facet; the form must not pretend."""
        with pytest.raises(ValueError, match="degenerated"):
            cp.canonical_form(cp.CHAIN(2), [sp.Integer(0)] * 3)


class TestMassResummation:
    """The result: the tower sums to the undeformed form at a shifted point."""

    def test_the_closed_form_is_the_shifted_massless_one(self) -> None:
        energy = sp.sqrt(Y**2 + M2)
        shifted = 2 / ((X1 + X2) * (X1 + energy) * (X2 + energy))
        assert sp.simplify(cp.massive_two_site(X1, X2, Y, M2) - shifted) == 0

    def test_the_massless_limit_is_the_two_site_polytope(self) -> None:
        got = sp.simplify(cp.massive_two_site(X1, X2, Y, 0))
        want = cp.canonical_form(cp.CHAIN(2), [X1, X2, Y])
        assert sp.simplify(got - want) == 0

    def test_the_weight_is_fixed_at_first_order(self) -> None:
        """One constant, fitted once. Everything below is then a prediction."""
        assert cp.MASS_INSERTION_WEIGHT == Fraction(-1, 2)

    @pytest.mark.parametrize("insertions", [0, 1, 2])
    def test_the_tower_reproduces_the_resummed_answer(self, insertions) -> None:
        assert cp.resummation_residual(insertions) == 0

    def test_the_third_order_prediction(self) -> None:
        """``m^6``: the five-site polytope in ``P^8``, with fifteen facets."""
        assert cp.resummation_residual(3) == 0

    def test_the_first_two_terms_explicitly(self) -> None:
        first = sp.factor(cp.tower_term(0, X1, X2, Y))
        assert sp.simplify(first - 2 / ((X1 + X2) * (X1 + Y) * (X2 + Y))) == 0
        second = sp.factor(cp.tower_term(1, X1, X2, Y))
        want = (
            2
            * (X1 + X2 + 2 * Y)
            / (Y * (X1 + X2) * (X1 + Y) ** 2 * (X2 + Y) ** 2)
        )
        assert sp.simplify(second - want) == 0

    def test_a_negative_insertion_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            cp.subdivided_chain(-1)
        with pytest.raises(ValueError, match="non-negative"):
            cp.pole_orders(-1)

    def test_the_subdivided_chain_has_the_right_shape(self) -> None:
        for insertions in range(4):
            graph = cp.subdivided_chain(insertions)
            assert graph.sites == insertions + 2
            assert len(graph.edges) == insertions + 1


class TestPoleOrders:
    @pytest.mark.parametrize("insertions", [0, 1, 2, 3])
    def test_the_derived_orders_match_the_polytope(self, insertions) -> None:
        assert cp.measured_pole_orders(insertions) == cp.pole_orders(insertions)

    @pytest.mark.parametrize("insertions", [0, 1, 2])
    def test_the_naive_count_agrees_below_three(self, insertions) -> None:
        assert cp.collapsing_facet_counts(insertions) == cp.pole_orders(insertions)

    def test_the_naive_count_fails_at_three(self) -> None:
        """The interesting failure, kept as a test.

        Six facets collapse onto ``2y`` at ``a = 3`` and the pole is of order
        five: coincident facets bound the order of the resulting pole, they do
        not fix it. This was the module's first prediction and the polytope
        computation refuted it.
        """
        assert cp.collapsing_facet_counts(3)["y"] == 6
        assert cp.pole_orders(3)["y"] == 5
        assert cp.measured_pole_orders(3)["y"] == 5

    @pytest.mark.parametrize("insertions", [0, 1, 2, 3, 7, 20])
    def test_the_endpoint_families_saturate_the_bound(self, insertions) -> None:
        naive = cp.collapsing_facet_counts(insertions)
        derived = cp.pole_orders(insertions)
        for key in ("x1 + y", "x2 + y", "x1 + x2"):
            assert naive[key] == derived[key]
        assert naive["y"] >= derived["y"]


class TestFRWTower:
    """The de Sitter tower: where the flat-space mechanism stops."""

    def test_the_measure_exponents(self) -> None:
        """Flat space is the ``alpha = 0`` corner, de Sitter is ``alpha = 1``."""
        assert cp.frw_measure_power(0) == -1
        assert cp.frw_measure_power(1) == 1
        assert cp.frw_measure_power(2) == 3

    def test_two_routes_to_the_first_order_term(self) -> None:
        """Closed form against the polytope plus an explicit energy integral."""
        quoted = cp.frw_first_order(X1, X2, Y)
        computed = cp.frw_first_order_by_integration(X1, X2, Y)
        assert sp.simplify(sp.expand(quoted - computed)) == 0

    @pytest.mark.parametrize(
        "point",
        [
            {X1: sp.Rational(3), X2: sp.Rational(5), Y: sp.Rational(2)},
            {X1: sp.Rational(1, 2), X2: sp.Rational(7, 3), Y: sp.Rational(9, 4)},
            {X1: sp.Rational(11), X2: sp.Rational(1, 5), Y: sp.Rational(4)},
        ],
    )
    def test_against_numerical_quadrature(self, point) -> None:
        """The closed form against the integral it came from, numerically."""
        import mpmath as mp

        omega = sp.Symbol("w", positive=True)
        integrand = cp.canonical_form(cp.CHAIN(3), [X1, omega, X2, Y, Y])
        function = sp.lambdify(omega, sp.expand(omega * integrand.subs(point)), "mpmath")
        numeric = mp.quad(function, [0, mp.inf])
        closed = complex(cp.frw_first_order(X1, X2, Y).subs(point).evalf()).real
        assert abs(float(numeric) - closed) < 1e-9 * max(1.0, abs(closed))

    def test_the_letters_obey_one_relation(self) -> None:
        """``A + D = B + C`` is what makes the weight-one numerator scale free."""
        A, B, C, D = cp.subgraph_letters(X1, X2, Y)
        assert sp.simplify(A + D - B - C) == 0
        assert len({A, B, C, D}) == 4

    def test_the_numerator_is_scale_covariant(self) -> None:
        """Rescaling all letters shifts it by ``log(lambda)`` times ``A-B-C+D = 0``."""
        lam = sp.Symbol("lam", positive=True)
        scaled = cp.frw_first_order_numerator(lam * X1, lam * X2, lam * Y)
        assert sp.simplify(sp.expand(scaled - lam * cp.frw_first_order_numerator(X1, X2, Y))) == 0

    @pytest.mark.parametrize("variable", [X1, X2])
    def test_the_unphysical_quadric_branch_cancels_in_de_sitter(self, variable) -> None:
        """``x_i = y`` is not a subgraph energy, so it must not be a pole."""
        numerator = cp.frw_first_order_numerator(X1, X2, Y)
        assert cp.unphysical_branch_residual(numerator, variable, Y) == 0

    @pytest.mark.parametrize("variable", [X1, X2])
    def test_the_unphysical_quadric_branch_cancels_in_flat_space(self, variable) -> None:
        """``E = x_i`` likewise, in the resummed flat-space answer."""
        energy = sp.Symbol("E", positive=True)
        numerator = cp.flat_numerator(X1, X2, energy)
        assert cp.unphysical_branch_residual(numerator, energy, variable) == 0

    def test_the_denominator_is_a_product_of_quadrics(self) -> None:
        """Not of linear subgraph energies: the extra branches are ``x_i - y``."""
        _, denominator = sp.fraction(sp.together(cp.frw_first_order(X1, X2, Y)))
        assert sp.simplify(denominator - (X1**2 - Y**2) * (X2**2 - Y**2)) == 0

    def test_a_divergent_energy_integral_is_refused(self) -> None:
        omega = sp.Symbol("w", positive=True)
        with pytest.raises(ValueError, match="diverges"):
            cp.mellin_simple(1 / (omega + 1) ** 1, omega, 1)

    def test_a_non_simple_pole_is_refused(self) -> None:
        omega = sp.Symbol("w", positive=True)
        with pytest.raises(ValueError, match="not simple"):
            cp.mellin_simple(1 / (omega + 1) ** 3, omega, 1)

    def test_the_mellin_helper_against_a_closed_form(self) -> None:
        """``int_0^inf w dw / prod (w + p_i)`` for three simple poles."""
        omega = sp.Symbol("w", positive=True)
        p, q, r = sp.symbols("p q r", positive=True)
        got = cp.mellin_simple(1 / ((omega + p) * (omega + q) * (omega + r)), omega, 1)
        want = sum(
            a * sp.log(a) / sp.prod([b - a for b in (p, q, r) if b is not a])
            for a in (p, q, r)
        )
        assert sp.simplify(got - want) == 0


class TestReparameterisationObstruction:
    """The flat-space mechanism provably does not survive into de Sitter."""

    def test_the_first_order_term_is_not_rational(self) -> None:
        """Four logarithms, all with non-vanishing coefficients.

        A reparameterisation ``y -> f(y, m^2)`` composed with the rational
        ``psi_0`` yields a rational function at every order in ``m^2``. One
        transcendental coefficient is enough to rule the whole mechanism out.
        """
        coefficients = cp.reparameterisation_obstruction(X1, X2, Y)
        assert len(coefficients) == 4
        for name, coefficient in coefficients.items():
            assert sp.simplify(coefficient) != 0, name

    def test_the_coefficients_reproduce_the_closed_form(self) -> None:
        coefficients = cp.reparameterisation_obstruction(X1, X2, Y)
        A, B, C, D = cp.subgraph_letters(X1, X2, Y)
        rebuilt = (
            coefficients["log(x1 + x2)"] * sp.log(A)
            + coefficients["log(x1 + y)"] * sp.log(B)
            + coefficients["log(x2 + y)"] * sp.log(C)
            + coefficients["log(2y)"] * sp.log(D)
        )
        assert sp.simplify(sp.expand(rebuilt - cp.frw_first_order(X1, X2, Y))) == 0

    def test_flat_space_is_the_case_where_no_logarithm_appears(self) -> None:
        """The contrast: the flat tower term is rational, and does resum."""
        for insertions in range(3):
            term = cp.tower_term(insertions, X1, X2, Y)
            assert term.free_symbols <= {X1, X2, Y}
            assert not term.atoms(sp.log)
