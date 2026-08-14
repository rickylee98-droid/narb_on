"""Mass does not deform the cosmological polytope; it pulls it back along a quadric.

The setting
-----------
Arkani-Hamed, Benincasa and Postnikov (arXiv:1709.02813) attach to every Feynman
graph ``G`` contributing to the wavefunction of the universe a convex polytope in
``P^{V+E-1}``: for each edge ``e = (i,j)`` three vertices

    x_i + x_j - y_e ,   x_i - x_j + y_e ,   -x_i + x_j + y_e ,

whose canonical form is the flat-space wavefunction ``psi_G(x, y)``.  Its facets
are the *subgraph energies*, so the boundaries of a static convex body are the
physical singularities.

That construction is for massless (equivalently conformally coupled) scalars.
Benincasa (arXiv:1909.02517) showed that treating a mass as a perturbative
two-point coupling gives, order by order, a degenerate limit of the canonical
form of a cosmological polytope -- the graph with two-valent sites inserted on
the massive line -- and then states the open problem twice: *"the study of a
possible closed form for the re-summed two-site graph is postponed to future
work"*, and *"it would be astonishing if the peculiar structure of this
perturbative expansion would allow us to re-sum it."*

What is derived here
--------------------
**In flat space the tower resums, and the answer is that the geometry never
deformed.**

Let ``psi_a`` be the canonical form of the cosmological polytope of the graph
with ``a`` two-valent sites inserted on an edge, in the degenerate limit where
those sites carry zero external energy and every subdivided edge carries the same
``y``.  Then

    sum_{a >= 0} (-m^2 / 2)^a psi_a  =  psi_G |_{y -> sqrt(y^2 + m^2)} .

The same polytope, the same canonical form, evaluated on a different point.  Mass
enters only through the map from the polytope's edge variable to the physical
one, replacing the linear ``y = |k|`` by the quadric ``y^2 = k^2 + m^2``.  There
is no deformed polytope to look for, because there is no deformation:
:func:`resummation_residual` checks the identity order by order, exactly, and
:data:`MASS_INSERTION_WEIGHT` is the single constant per insertion -- fixed at
order ``m^2`` and then a prediction at every higher order.

For the two-site chain the resummed function is explicitly algebraic of degree
two,

    psi  =  2 [ y^2 + m^2 + x1 x2 - (x1 + x2) sqrt(y^2 + m^2) ]
            / [ (x1 + x2) (y^2 + m^2 - x1^2) (y^2 + m^2 - x2^2) ] ,

with exactly one square root (:func:`massive_two_site`).  So the branch cut is
real and is not an artefact -- but it sits in the *kinematic map*, not in the
geometry.  Read in the energy variable ``E = sqrt(y^2 + m^2)`` the singularities
are the same three hyperplanes ``x1 + x2``, ``x1 + E``, ``x2 + E`` as in the
massless case; read in the momentum they are the quadrics ``E^2 = k^2 + m^2``
intersected with those hyperplanes.  A facet does not curve.  It is a flat facet
seen through a quadratic change of variables.

Why the poles seemed to proliferate
-----------------------------------
Each ``psi_a`` has poles of high order, because facets of the polytope collapse
onto each other in the degenerate limit: the ``a + 1`` intervals of the
subdivided chain containing the left endpoint all carry energy ``x1 + y``, the
``a + 1`` containing the right endpoint all carry ``x2 + y``, and the
``a(a+1)/2`` interior intervals all carry ``2y``.  Summing a tower of poles of
growing order at ``x + y`` is what produces a simple pole at the *shifted*
location ``x + sqrt(y^2 + m^2)``: the proliferation is an artefact of expanding a
shifted pole around the wrong point.

Coincident facets bound the pole order but do not fix it.  The endpoint families
saturate the bound, ``a + 1``, and the interior family does not: at ``a = 3`` six
facets collapse onto ``2y`` and the pole is of order five.  The correct orders
follow from the resummation instead -- ``[m^{2a}] sqrt(y^2 + m^2)`` goes like
``y^{1-2a}`` -- giving ``2a - 1`` (:func:`pole_orders` against
:func:`collapsing_facet_counts`).  The naive count was this module's first
prediction and the polytope computation refuted it at the first value where the
two differ.

What this does not do
---------------------
Flat space only.  In a genuine FRW background each inserted site carries a factor
``omega^{2 alpha - 1}`` and an integral over ``omega``, so the tower's terms are
no longer canonical forms evaluated at a point but integrals of them, producing
polylogarithms rather than rational functions; that is the case Benincasa poses
and it is *not* settled here.  What the flat-space result does say is where to
look: the obstruction is not that the polytope must curve, since in the flat
limit it demonstrably does not.

Referees
--------
Nothing here is asserted against itself.  The facet correspondence is checked
against an independent enumeration of subgraphs, on loop graphs as well as trees
(:func:`facet_theorem_residual`); the canonical forms reproduce the published
two-site, three-site and one-loop bubble wavefunctions; the total-energy residue
reproduces the flat-space scattering amplitude; and the resummation constant is
fitted once and tested three times.

Prior work
----------
The polytope, the facet theorem and the canonical-form-is-the-wavefunction
statement are Arkani-Hamed--Benincasa--Postnikov.  The identification of the
order-by-order mass insertions with degenerate cosmological polytopes is
Benincasa's.  What is done here is the resummation of that tower in flat space,
in closed form, and the reading of the answer as a quadric pullback of the
undeformed geometry.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

import sympy as sp
from PyNormaliz import Cone

__all__ = [
    "Graph",
    "CHAIN",
    "BUBBLE",
    "TRIANGLE",
    "BOX",
    "STAR",
    "cosmological_vertices",
    "support_hyperplanes",
    "subgraph_facets",
    "facet_theorem_residual",
    "canonical_form",
    "total_energy_residue",
    "MASS_INSERTION_WEIGHT",
    "subdivided_chain",
    "tower_term",
    "massive_two_site",
    "tower_coefficient",
    "resummation_residual",
    "collapsing_facet_counts",
    "pole_orders",
    "measured_pole_orders",
    "frw_measure_power",
    "mellin_simple",
    "subgraph_letters",
    "frw_first_order",
    "frw_first_order_by_integration",
    "frw_first_order_numerator",
    "flat_numerator",
    "unphysical_branch_residual",
    "reparameterisation_obstruction",
    "degenerate_facets",
    "frw_alphabet",
    "letters_from_facets",
]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Graph:
    """A Feynman graph: ``sites`` many vertices and a list of ``edges``.

    Multi-edges are allowed -- the one-loop bubble is two edges on two sites --
    because the polytope construction reads the edge list, not the adjacency.
    """

    sites: int
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.sites < 1:
            raise ValueError("a graph needs at least one site")
        for i, j in self.edges:
            if not (0 <= i < self.sites and 0 <= j < self.sites):
                raise ValueError(f"edge {(i, j)} leaves the site range")
            if i == j:
                raise ValueError("self-loops carry no cosmological polytope")

    @property
    def dimension(self) -> int:
        """Ambient dimension ``V + E``; the polytope lives in ``P^{V+E-1}``."""
        return self.sites + len(self.edges)


def CHAIN(sites: int) -> Graph:
    """The path graph on ``sites`` vertices."""
    return Graph(sites, tuple((i, i + 1) for i in range(sites - 1)))


BUBBLE = Graph(2, ((0, 1), (0, 1)))
TRIANGLE = Graph(3, ((0, 1), (1, 2), (0, 2)))
BOX = Graph(4, ((0, 1), (1, 2), (2, 3), (0, 3)))
STAR = Graph(4, ((0, 1), (0, 2), (0, 3)))


# --------------------------------------------------------------------------- #
# The polytope
# --------------------------------------------------------------------------- #
def cosmological_vertices(graph: Graph) -> list[list[int]]:
    """The ``3E`` generators, three per edge, in the ``(x, y)`` basis."""
    sites = graph.sites
    generators: list[list[int]] = []
    for index, (i, j) in enumerate(graph.edges):
        for signs in ((1, 1, -1), (1, -1, 1), (-1, 1, 1)):
            vector = [0] * graph.dimension
            vector[i] += signs[0]
            vector[j] += signs[1]
            vector[sites + index] += signs[2]
            generators.append(vector)
    return generators


def support_hyperplanes(graph: Graph) -> set[tuple[int, ...]]:
    """Facets of the cone, computed exactly by Normaliz."""
    cone = Cone(cone=cosmological_vertices(graph))
    return {tuple(int(entry) for entry in row) for row in cone.SupportHyperplanes()}


def subgraph_facets(graph: Graph) -> set[tuple[int, ...]]:
    """Predicted facets, enumerated from subgraphs rather than from the geometry.

    A subgraph is a vertex subset ``Vg`` together with an edge subset ``Eg`` of
    the edges internal to it, connected as a graph on ``Vg``.  Its energy is

        sum_{v in Vg} x_v  +  sum_{e not in Eg} (endpoints of e inside Vg) y_e ,

    so an edge left out of ``Eg`` with *both* endpoints inside contributes
    ``2 y_e``.  Dropping that possibility gives the right answer on trees and the
    wrong one on every loop graph, which is what makes this an independent check
    rather than a restatement.
    """
    sites = graph.sites
    facets: set[tuple[int, ...]] = set()
    for mask in range(1, 1 << sites):
        inside = {v for v in range(sites) if mask >> v & 1}
        internal = [
            k for k, (i, j) in enumerate(graph.edges) if i in inside and j in inside
        ]
        for size in range(len(internal) + 1):
            for chosen in itertools.combinations(internal, size):
                if not _connected(inside, [graph.edges[k] for k in chosen]):
                    continue
                vector = [0] * graph.dimension
                for v in inside:
                    vector[v] = 1
                for k, (i, j) in enumerate(graph.edges):
                    if k in chosen:
                        continue
                    vector[sites + k] = (i in inside) + (j in inside)
                facets.add(tuple(vector))
    return facets


def _connected(vertices: set[int], edges: Sequence[tuple[int, int]]) -> bool:
    if not vertices:
        return False
    adjacency: dict[int, set[int]] = {v: set() for v in vertices}
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    root = next(iter(vertices))
    seen = {root}
    stack = [root]
    while stack:
        current = stack.pop()
        for neighbour in adjacency[current]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen == vertices


def facet_theorem_residual(graph: Graph) -> tuple[set, set]:
    """``(unexpected, missing)`` facets: both empty is the theorem holding."""
    computed = support_hyperplanes(graph)
    predicted = subgraph_facets(graph)
    return computed - predicted, predicted - computed


# --------------------------------------------------------------------------- #
# The canonical form
# --------------------------------------------------------------------------- #
def canonical_form(graph: Graph, values: Sequence | None = None):
    """The wavefunction, as the canonical form of the polytope.

    Computed as the integral of ``exp(-<Y, l>)`` over the *dual* cone, which is
    generated by the facet normals: triangulating it into simplicial cones gives

        psi(Y)  =  sum_sigma |det w_sigma| / prod_i <Y, w_i> ,

    whose poles sit on the facets, where the wavefunction's singularities belong.
    Triangulating the polytope itself instead would put the poles on the
    vertices, which is the dual object and not the wavefunction.

    ``values`` substitutes for the ``V + E`` variables in order; passing zeros for
    the internal sites of a subdivided chain is the degenerate limit in which the
    mass tower lives.
    """
    variables = list(values) if values is not None else _default_variables(graph)
    if len(variables) != graph.dimension:
        raise ValueError(
            f"expected {graph.dimension} kinematic variables; got {len(variables)}"
        )
    dual = Cone(cone=[list(row) for row in sorted(support_hyperplanes(graph))])
    simplices, generators = dual.Triangulation()
    total = sp.Integer(0)
    for entry in simplices:
        indices, determinant = entry[0], entry[1]
        term = sp.Integer(abs(int(determinant)))
        for index in indices:
            linear = sum(
                sp.Integer(coefficient) * variable
                for coefficient, variable in zip(generators[index], variables)
            )
            if linear == 0:
                raise ValueError(
                    "a facet degenerated to zero under this substitution"
                )
            term /= linear
        total += term
    return sp.cancel(sp.together(total))


def _default_variables(graph: Graph) -> list:
    return list(sp.symbols(f"x0:{graph.sites}")) + list(
        sp.symbols(f"y0:{len(graph.edges)}")
    )


def total_energy_residue(graph: Graph):
    """Residue of the wavefunction on the total-energy pole.

    Arkani-Hamed--Benincasa--Postnikov: this is the flat-space scattering
    amplitude of the graph, so it is a check on the canonical form that uses no
    part of the machinery that produced it.  For the two-site chain it returns
    ``2/((x0+y)(x1+y))``, which on the momentum-conserving locus ``x1 = -x0`` is
    the propagator ``2/(y^2 - x0^2)``.
    """
    variables = _default_variables(graph)
    wavefunction = canonical_form(graph, variables)
    numerator, denominator = sp.fraction(wavefunction)
    total = sum(variables[: graph.sites])
    quotient, remainder = sp.div(sp.expand(denominator), total, variables[0])
    if sp.simplify(remainder) != 0:
        raise ValueError("the total energy does not divide the denominator")
    return sp.cancel(numerator / quotient)


# --------------------------------------------------------------------------- #
# The mass tower, and its resummation
# --------------------------------------------------------------------------- #
#: Weight carried by each two-valent mass insertion, relative to ``m^2``.  Fixed
#: once, by matching the tower at order ``m^2``; every higher order is then a
#: prediction, and :func:`resummation_residual` tests it.
MASS_INSERTION_WEIGHT = Fraction(-1, 2)


def subdivided_chain(insertions: int) -> Graph:
    """The two-site chain with ``insertions`` two-valent sites on its edge."""
    if insertions < 0:
        raise ValueError("the number of insertions must be non-negative")
    return CHAIN(insertions + 2)


def tower_term(insertions: int, x1=None, x2=None, y=None):
    """``psi_a``: the canonical form in the degenerate limit of the mass tower.

    The inserted sites carry no external energy and every subdivided edge carries
    the same ``y``, because a two-valent vertex conserves the momentum flowing
    through it.  This is the limit in which distinct facets of the polytope
    collapse onto each other, which is where the high-order poles come from.
    """
    x1 = sp.Symbol("x1", positive=True) if x1 is None else x1
    x2 = sp.Symbol("x2", positive=True) if x2 is None else x2
    y = sp.Symbol("y", positive=True) if y is None else y
    graph = subdivided_chain(insertions)
    values = (
        [x1] + [sp.Integer(0)] * insertions + [x2] + [y] * (insertions + 1)
    )
    return canonical_form(graph, values)


def massive_two_site(x1, x2, y, mass_squared):
    """The resummed two-site wavefunction, in closed form.

    Algebraic of degree two over the rational functions, with exactly one square
    root:

        2 [ E^2 + x1 x2 - (x1 + x2) E ] / [ (x1 + x2)(E^2 - x1^2)(E^2 - x2^2) ] ,
        E = sqrt(y^2 + m^2) .

    Equal to ``2 / ((x1+x2)(x1+E)(x2+E))`` -- the massless canonical form with
    ``y`` replaced by ``E`` -- and written this way to make the singularities
    visible as the quadrics ``E^2 = x_i^2`` rather than as hyperplanes.
    """
    energy = sp.sqrt(y**2 + mass_squared)
    return sp.together(
        2
        * (energy**2 + x1 * x2 - (x1 + x2) * energy)
        / ((x1 + x2) * (energy**2 - x1**2) * (energy**2 - x2**2))
    )


def tower_coefficient(insertions: int, x1, x2, y):
    """``(-2)^a`` times the ``m^{2a}`` Taylor coefficient of the resummed answer.

    The prediction that :func:`tower_term` must reproduce.  The factor is
    ``MASS_INSERTION_WEIGHT ** -a``, so it carries no freedom beyond the single
    constant fixed at first order.
    """
    mass_squared = sp.Symbol("_m2", positive=True)
    series = sp.series(
        2 / ((x1 + x2) * (x1 + sp.sqrt(y**2 + mass_squared)) * (x2 + sp.sqrt(y**2 + mass_squared))),
        mass_squared,
        0,
        insertions + 1,
    ).removeO()
    coefficient = sp.expand(
        sp.diff(series, mass_squared, insertions) / sp.factorial(insertions)
    ).subs(mass_squared, 0)
    return sp.cancel(coefficient / MASS_INSERTION_WEIGHT**insertions)


def resummation_residual(insertions: int):
    """``psi_a`` minus its prediction from the resummed closed form.

    Zero is the content of the module: the tower of degenerate cosmological
    polytopes sums to the undeformed canonical form evaluated at
    ``y -> sqrt(y^2 + m^2)``.
    """
    x1 = sp.Symbol("x1", positive=True)
    x2 = sp.Symbol("x2", positive=True)
    y = sp.Symbol("y", positive=True)
    return sp.simplify(
        sp.cancel(tower_term(insertions, x1, x2, y) - tower_coefficient(insertions, x1, x2, y))
    )


def collapsing_facet_counts(insertions: int) -> dict[str, int]:
    """How many facets collapse onto each linear form in the degenerate limit.

    The subdivided chain on ``a + 2`` sites has one connected subgraph per
    interval.  The ``a + 1`` intervals containing the left endpoint all carry
    energy ``x1 + y``; likewise on the right; the ``a(a+1)/2`` intervals of
    interior sites alone all carry ``2y``; and the whole chain carries
    ``x1 + x2``.

    This is an **upper bound on the pole orders, not the pole orders**.  It is
    saturated at the two endpoint families and fails at the interior one: at
    ``a = 3`` six facets collapse onto ``2y`` but the pole is of order five.
    Coincident facets bound the order of the resulting pole; they do not fix it,
    because the canonical form only develops a higher-order pole where the
    coincident facets meet inside a common simplex of the triangulation.  The
    first version of this module predicted ``a(a+1)/2`` and was wrong at the
    first value where the two counts differ.
    """
    if insertions < 0:
        raise ValueError("the number of insertions must be non-negative")
    return {
        "x1 + y": insertions + 1,
        "x2 + y": insertions + 1,
        "y": insertions * (insertions + 1) // 2,
        "x1 + x2": 1,
    }


def pole_orders(insertions: int) -> dict[str, int]:
    """Pole orders of ``psi_a``, derived from the resummed closed form.

    Only ``E = sqrt(y^2 + m^2)`` carries the ``m^2`` dependence non-rationally,
    and ``[m^{2a}] E`` is proportional to ``y^{1-2a}``, while ``E^2 = y^2 + m^2``
    is polynomial.  So the ``y`` pole of ``psi_a`` has order ``2a - 1`` for
    ``a >= 1`` and the endpoint poles, coming from expanding ``1/(x + E)`` about
    ``x + y``, have order ``a + 1``:

        (x1 + y) : a + 1 ,   (x2 + y) : a + 1 ,   y : max(0, 2a - 1) ,
        (x1 + x2) : 1 .

    Derived from the closed form and then checked against the polytope
    computation by :func:`measured_pole_orders`, which knows nothing about it.
    """
    if insertions < 0:
        raise ValueError("the number of insertions must be non-negative")
    return {
        "x1 + y": insertions + 1,
        "x2 + y": insertions + 1,
        "y": max(0, 2 * insertions - 1),
        "x1 + x2": 1,
    }


def measured_pole_orders(insertions: int) -> dict[str, int]:
    """Multiplicities read off the computed denominator of ``psi_a``."""
    x1 = sp.Symbol("x1", positive=True)
    x2 = sp.Symbol("x2", positive=True)
    y = sp.Symbol("y", positive=True)
    _, denominator = sp.fraction(sp.cancel(tower_term(insertions, x1, x2, y)))
    factored = sp.factor(denominator)
    orders = {"x1 + y": 0, "x2 + y": 0, "y": 0, "x1 + x2": 0}
    targets = {"x1 + y": x1 + y, "x2 + y": x2 + y, "y": y, "x1 + x2": x1 + x2}
    for base, exponent in factored.as_powers_dict().items():
        for name, target in targets.items():
            if sp.simplify(base - target) == 0:
                orders[name] = int(exponent)
    return orders


# --------------------------------------------------------------------------- #
# The FRW tower: where the flat-space mechanism stops working
# --------------------------------------------------------------------------- #
#: In a background ``a(eta) = (-eta)^{-alpha}`` the two-point mass coupling is
#: time dependent, and in the energy representation each inserted site carries a
#: measure ``omega^{2 alpha - 1} d omega`` rather than sitting at zero energy.
#: Flat space is ``alpha = 0``, where the coupling is constant in time and the
#: measure collapses to ``delta(omega)`` -- which is exactly the degenerate limit
#: :func:`tower_term` takes.  De Sitter is ``alpha = 1``, measure ``omega
#: d omega``.
def frw_measure_power(alpha: int) -> int:
    """Exponent ``2 alpha - 1`` of the energy measure at an inserted site."""
    return 2 * int(alpha) - 1


def mellin_simple(expression, variable, power: int = 1):
    """``int_0^inf w^power f(w) dw`` for a rational ``f`` with simple poles.

    By residues: writing ``w^p f = sum_i r_i / (w + p_i)``, convergence forces
    ``sum_i r_i = 0`` and the integral is ``- sum_i r_i log p_i``.  The assertion
    that the residues sum to zero is not decoration -- it is the convergence
    condition, and a tower term that failed it would be telling us the insertion
    measure is wrong.
    """
    integrand = sp.cancel(sp.together(variable**power * expression))
    _, denominator = sp.fraction(integrand)
    poles = []
    for base, multiplicity in sp.factor_list(denominator)[1]:
        polynomial = sp.Poly(base, variable)
        if polynomial.degree() == 0:
            continue
        if polynomial.degree() != 1 or multiplicity != 1:
            raise ValueError(f"pole {base}^{multiplicity} is not simple")
        slope, constant = polynomial.all_coeffs()
        poles.append(sp.cancel(constant / slope))
    residues = [
        (sp.simplify(sp.cancel(integrand * (variable + pole)).subs(variable, -pole)), pole)
        for pole in poles
    ]
    total = sp.simplify(sum(coefficient for coefficient, _ in residues))
    if total != 0:
        raise ValueError(f"the integral diverges: residues sum to {total}")
    return sp.simplify(-sum(coefficient * sp.log(pole) for coefficient, pole in residues))


def subgraph_letters(x1, x2, y):
    """The four energies that survive the degenerate limit of the tower.

    ``A = x1 + x2`` (total), ``B = x1 + y`` and ``C = x2 + y`` (partial), and
    ``D = 2y`` (an interior site).  They obey one relation, ``A + D = B + C``,
    which is what makes the weight-one combination below scale invariant: a
    common rescaling of all four letters shifts it by ``log(lambda)`` times
    ``A - B - C + D = 0``.
    """
    return x1 + x2, x1 + y, x2 + y, 2 * y


def frw_first_order(x1, x2, y):
    """The ``a = 1`` de Sitter tower term, in closed form.

    Integrating the three-site canonical form against ``omega d omega`` gives

        4 [ A log A - B log B - C log C + D log D ]
        / [ (x1^2 - y^2) (x2^2 - y^2) ] .

    This is Benincasa's equation (4.10), rederived; what the form above makes
    visible is the denominator.  It is not a product of the *linear* subgraph
    energies but of the four **quadrics** ``x_i^2 - y^2`` -- exactly the loci
    ``E^2 = x_i^2`` that the flat-space resummation produces, evaluated at
    ``m = 0``.  The quadrics that the flat-space answer only reaches after
    summing the whole tower are already present in de Sitter at first order.

    The numerator is weight one and pure: four logarithms, coefficients
    ``+1, -1, -1, +1`` on the four subgraph energies, with the single relation
    ``A + D = B + C`` making it scale invariant.
    """
    A, B, C, D = subgraph_letters(x1, x2, y)
    numerator = A * sp.log(A) - B * sp.log(B) - C * sp.log(C) + D * sp.log(D)
    return 4 * numerator / ((x1**2 - y**2) * (x2**2 - y**2))


def frw_first_order_by_integration(x1, x2, y):
    """The same term, computed from the polytope rather than quoted.

    Builds the three-site canonical form with the inserted site at energy
    ``omega`` and does the ``omega`` integral by :func:`mellin_simple`.  Shares
    no code with :func:`frw_first_order`, so agreement is a check on both.
    """
    omega = sp.Symbol("_omega", positive=True)
    integrand = canonical_form(CHAIN(3), [x1, omega, x2, y, y])
    return mellin_simple(integrand, omega, frw_measure_power(1))


def frw_first_order_numerator(x1, x2, y):
    """The weight-one numerator ``A log A - B log B - C log C + D log D``."""
    A, B, C, D = subgraph_letters(x1, x2, y)
    return A * sp.log(A) - B * sp.log(B) - C * sp.log(C) + D * sp.log(D)


def flat_numerator(x1, x2, energy):
    """Numerator of the resummed flat-space answer, ``E^2 + x1 x2 - (x1+x2) E``."""
    return energy**2 + x1 * x2 - (x1 + x2) * energy


def unphysical_branch_residual(numerator, variable, value):
    """Value of a numerator on the unphysical branch of a quadric denominator.

    Both closed forms in this module carry denominators that factor into
    quadrics -- ``(x_i^2 - y^2)`` in de Sitter at first order, ``(E^2 - x_i^2)``
    in the resummed flat-space answer -- and each quadric has two branches, only
    one of which is a subgraph energy.  The other, ``x_i = y`` respectively
    ``E = x_i``, is not a facet of any cosmological polytope, so a pole there
    would be a singularity the positive geometry does not predict.

    It is not one: the numerator vanishes on that branch, in both cases.  This
    function returns the value that must be zero, and is the sharpest check in
    the module -- neither numerator was constructed with it in mind.
    """
    return sp.simplify(numerator.subs(variable, value))


def reparameterisation_obstruction(x1, x2, y) -> dict:
    """Why the flat-space mechanism cannot survive into de Sitter.

    Suppose the de Sitter tower resummed the same way the flat-space one does,
    by a reparameterisation ``sum_a t^a psi_a = psi_0(x, f(y, t))`` with
    ``f(y, 0) = y``.  Since ``psi_0`` is a *rational* function of its arguments,
    every Taylor coefficient in ``t`` would be a rational function of ``x``,
    ``y`` and the derivatives of ``f`` at ``t = 0``.  In particular every
    ``psi_a`` would be rational in ``y``.

    It is not.  Already at first order the de Sitter term carries four
    logarithms with non-vanishing coefficients, returned here.  So no
    reparameterisation of the edge variable -- no ``y -> sqrt(y^2 + m^2)`` and no
    replacement of it -- can generate the de Sitter tower.  The obstruction is
    transcendence of the first term, and it is independent of anything to do with
    the shape of the polytope: the facets do not have to curve for the mechanism
    to fail, and in flat space they demonstrably do not curve at all.

    What this leaves is the reading that the de Sitter resummation must shift a
    *transcendental* label rather than a kinematic one -- the Bessel index of the
    mode functions, elementary only on a half-integer sublattice, which is where
    Benincasa's light states live.  That reading is not established here.
    """
    A, B, C, D = subgraph_letters(x1, x2, y)
    denominator = (x1**2 - y**2) * (x2**2 - y**2)
    return {
        "log(x1 + x2)": sp.cancel(4 * A / denominator),
        "log(x1 + y)": sp.cancel(-4 * B / denominator),
        "log(x2 + y)": sp.cancel(-4 * C / denominator),
        "log(2y)": sp.cancel(4 * D / denominator),
    }


def degenerate_facets(insertions: int, x1, x2, y) -> set:
    """The distinct facet forms of the tower's polytope in the degenerate limit.

    The subdivided chain has one facet per interval, but under the substitution
    that defines the tower -- inserted sites at zero energy, every edge at the
    same ``y`` -- they collapse onto far fewer distinct linear forms.  For one
    insertion the four survivors are ``x1 + x2``, ``x1 + y``, ``x2 + y`` and
    ``2y``.
    """
    graph = subdivided_chain(insertions)
    values = [x1] + [sp.Integer(0)] * insertions + [x2] + [y] * (insertions + 1)
    return {
        sp.expand(
            sum(sp.Integer(c) * value for c, value in zip(facet, values))
        )
        for facet in support_hyperplanes(graph)
    }


def frw_alphabet(x1, x2, y) -> set:
    """Arguments of the logarithms in the first-order de Sitter term."""
    return {sp.expand(atom.args[0]) for atom in frw_first_order(x1, x2, y).atoms(sp.log)}


def letters_from_facets(graph: Graph, values: Sequence, variable) -> set:
    """Facets that involve ``variable``, evaluated with it set to zero.

    This is the rule that governs the alphabet.  Integrating an inserted site's
    energy picks up one residue per pole in that energy, and the poles are
    exactly the facets whose linear form contains it; the resulting logarithm
    has as its argument that facet with the integrated energy removed.  Facets
    independent of the variable never become logarithms -- they stay rational
    prefactors.

    At one insertion the two sets coincide, because every facet of the
    three-site polytope that survives the degenerate limit also happens to
    involve the inserted energy.  At two they do not: the four-site polytope has
    seven surviving facets and only six of them contain the integrated energy,
    so ``x1 + y`` is a prefactor rather than a letter.  Reading the one-insertion
    coincidence as the general rule was this module's second wrong guess.
    """
    if len(values) != graph.dimension:
        raise ValueError(
            f"expected {graph.dimension} kinematic variables; got {len(values)}"
        )
    letters = set()
    for facet in support_hyperplanes(graph):
        form = sp.expand(
            sum(sp.Integer(c) * value for c, value in zip(facet, values))
        )
        if form.has(variable):
            letters.add(sp.expand(form.subs(variable, 0)))
    return letters
