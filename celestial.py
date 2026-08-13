"""Where the massive w_{1+infinity} action breaks: an exact obstruction.

The setting
-----------
In four-dimensional asymptotically flat gravity an infinite tower of soft
graviton modes generates the wedge algebra of ``w_{1+infinity}``,

    [w^p_m, w^q_n]  =  [ m(q-1) - n(p-1) ] w^{p+q-2}_{m+n} ,

with ``|m| <= p-1``.  On *massless* hard particles this is the Poisson algebra of
polynomial area-preserving diffeomorphisms of a two-plane, acting on the point of
that plane which is the particle's momentum spinor.  Himwich and Pate
(arXiv:2312.08597, JHEP 07 (2024) 180) extended the action to *massive* scalars
and showed it still closes on ``w_{1+infinity}``.  Their generators for ``p > 2``
contain inverse powers of the momentum operator, which they evaluate by a
Schwinger parametrisation, describing that step as formal; the resulting action
mixes infinitely many conformal families.  Their closing question is whether a
discrete conformal primary basis at integer ``Delta`` produces any simplification.

What is derived here
--------------------
**The answer is that the inverse does not exist there, and exactly there.**

Massive celestial primaries are integrals of the hyperbolic bulk-to-boundary
propagator ``G_Delta = (-phat . qhat)^{-Delta}`` over the unit hyperboloid.
Multiplication by the momentum is a differential operator in the celestial
coordinates together with a shift of ``Delta``, and the module it acts on is
spanned by

    | Delta ; i, j ; a, b >  =  z^a zbar^b  d_z^i d_zbar^j  G_Delta ,

since the operator's derivatives fall on the propagator, not on any polynomial
prefactor.  On that module:

1. ``phat . phat = -1`` exactly, on every basis element and every ``Delta``
   (:func:`mass_shell_residual`), and the four components commute.  This is what
   pins the conventions: nothing else in the module is adjustable.

2. The distinguished component ``-n . phat``, whose inverse powers define the
   generators for ``p > 2``, is **bidiagonal**: it has exactly two terms,

       -n.phat | Delta ; i, j >  =  (Delta-1)^{-2} | Delta-1 ; i+1, j+1 >
                                  +  Delta (Delta-1)^{-1} | Delta+1 ; i, j > ,

   and raises the grading ``N = Delta + i + j - a - b`` by one.  All four
   components carry definite weights under ``N`` and under
   ``M = (a-i) - (b-j)`` -- the two Cartan gradings -- which is a second referee
   on the operator, and one it was free to fail.

3. Inverting a bidiagonal operator is a one-step recursion rather than a
   continued fraction, so the mixing HP find is an arithmetic progression:
   ``Delta - 1, Delta - 3, Delta - 5, ...``, with coefficients given in closed
   form by :func:`inverse_coefficient` as a ratio of two products.  That is the
   organising structure, and it is exact.

4. The recursion divides by ``beta_r = (Delta-1-2r) / (Delta-2-2r)``.  It
   therefore survives if and only if the progression ``Delta - 1 - 2r`` avoids
   ``0`` and ``1`` for every ``r >= 0``, which happens if and only if **Delta is
   not a positive integer** (:func:`inverse_exists`).  On the principal series
   ``Delta = 1 + i lambda`` with ``lambda != 0`` the inverse exists at every
   order.  On a discrete basis at integer ``Delta >= 1`` it fails, at the step
   :func:`obstruction_step` computes.

So the discrete integer basis does not simplify the massive action; it destroys
it.  The two forbidden values are not arbitrary: ``Delta = 1`` is the
principal-series midpoint where the ``(Delta-1)`` denominators of the momentum
operator blow up, and ``Delta = 0`` is where the operator acquires a kernel.  The
obstruction is a property of the momentum operator itself, and is invisible
before the inverse is taken, which is why the ``p <= 2`` generators -- Poincare,
which need no inverse -- are unaffected.

Prior work, and what is new
---------------------------
The algebra is Strominger's and Guevara--Himwich--Pate--Strominger's; the massive
action, its closure, and the Schwinger prescription are Himwich--Pate's; the
hyperbolic conformal primary basis is Pasterski--Shao.  Equation (6.1) of
Himwich--Pate is *derived* here rather than quoted, and reduces to the single
identity ``u u_zzbar - u_z u_zbar = 1`` for ``u = -phat . qhat``
(:func:`propagator_identity_residual`).  What is new is the module structure that
makes the operator bidiagonal, the resulting closed form for the family mixing,
and the exact statement of when the inverse exists.  The conclusion -- that
integer ``Delta`` is precisely the obstructed case -- answers the question
Himwich--Pate close on, in the negative.

Scope
-----
Massive scalars, tree level, one particle.  Everything is exact rational
arithmetic; ``Delta`` is carried as a :class:`~fractions.Fraction`, so the
principal series is reached through the criterion of :func:`inverse_exists`
rather than by complex arithmetic.  The four momentum components are carried in
the null basis ``(++, --, +-, -+)`` precisely so that no factor of ``i`` appears
and the arithmetic stays in ``Q``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, Mapping, Sequence

__all__ = [
    "in_wedge",
    "wedge_generators",
    "structure_constant",
    "wedge_bracket",
    "jacobi_residual",
    "wedge_monomial",
    "poisson_bracket",
    "poisson_realisation_residual",
    "NULL_COMPONENTS",
    "QHAT",
    "State",
    "basis",
    "momentum",
    "mass_shell_residual",
    "commutator_residual",
    "boost_weight",
    "spin_weight",
    "propagator_identity_residual",
    "lightcone_momentum",
    "inverse_recursion",
    "inverse_coefficient",
    "inverse_series",
    "inverse_residual",
    "inverse_exists",
    "obstruction_step",
]

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Part 1: the wedge algebra of w_{1+infinity}
# --------------------------------------------------------------------------- #
def in_wedge(spin: Fraction, mode: Fraction) -> bool:
    """Whether ``w^p_m`` lies in the wedge, ``|m| <= p - 1``.

    The wedge is not a convention.  Under the realisation of :func:`wedge_monomial`
    it is exactly the condition that the corresponding function on the plane is a
    *polynomial* rather than a Laurent monomial, so the wedge subalgebra is the
    part of ``w_infinity`` that acts on the plane without a singularity at the
    origin.
    """
    return abs(mode) <= spin - 1


def wedge_generators(max_spin: Fraction | int) -> Iterator[tuple[Fraction, Fraction]]:
    """All ``(p, m)`` in the wedge with ``1 <= p <= max_spin``, ``2p`` an integer.

    ``p`` runs over half-integers because the tower of conformally soft gravitons
    sits at ``Delta = 2 - k`` for ``k = 1, 0, -1, ...``; ``p`` and ``m`` are then
    simultaneously integral or simultaneously half-odd.
    """
    limit = Fraction(max_spin)
    doubled = 2
    while Fraction(doubled, 2) <= limit:
        spin = Fraction(doubled, 2)
        mode = -(spin - 1)
        while mode <= spin - 1:
            yield spin, mode
            mode += 1
        doubled += 1


def structure_constant(
    spin_a: Fraction, mode_a: Fraction, spin_b: Fraction, mode_b: Fraction
) -> Fraction:
    """``m(q-1) - n(p-1)`` for ``[w^p_m, w^q_n]``."""
    return Fraction(mode_a) * (Fraction(spin_b) - 1) - Fraction(mode_b) * (
        Fraction(spin_a) - 1
    )


def wedge_bracket(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, tuple[Fraction, Fraction]]:
    """``[w^p_m, w^q_n] = c * w^{p+q-2}_{m+n}``, returned as ``(c, (p+q-2, m+n))``."""
    spin_a, mode_a = Fraction(left[0]), Fraction(left[1])
    spin_b, mode_b = Fraction(right[0]), Fraction(right[1])
    coefficient = structure_constant(spin_a, mode_a, spin_b, mode_b)
    return coefficient, (spin_a + spin_b - 2, mode_a + mode_b)


def jacobi_residual(
    first: tuple[Fraction, Fraction],
    second: tuple[Fraction, Fraction],
    third: tuple[Fraction, Fraction],
) -> Fraction:
    """The Jacobi identity on three wedge generators, exactly.

    All three double brackets land on the same generator
    ``w^{p+q+r-4}_{m+n+l}``, so the identity reduces to a single rational number
    which must vanish.
    """
    total = Fraction(0)
    for a, b, c in ((first, second, third), (second, third, first), (third, first, second)):
        inner_coefficient, inner = wedge_bracket(b, c)
        outer_coefficient, _ = wedge_bracket(a, inner)
        total += inner_coefficient * outer_coefficient
    return total


# --------------------------------------------------------------------------- #
# Part 2: the massless realisation, as area-preserving flows of a plane
# --------------------------------------------------------------------------- #
def wedge_monomial(spin: Fraction, mode: Fraction) -> tuple[int, int]:
    """``w^p_m`` as the monomial ``lambda_0^{p-1+m} lambda_1^{p-1-m}`` on ``C^2``.

    A massless momentum is a point of that plane, so this is the whole of the
    massless action: the generators are Hamiltonians and the particle follows
    their flows.  Both exponents are non-negative exactly on the wedge.
    """
    left = Fraction(spin) - 1 + Fraction(mode)
    right = Fraction(spin) - 1 - Fraction(mode)
    if left.denominator != 1 or right.denominator != 1:
        raise ValueError(f"w^{spin}_{mode} is not a monomial: exponents {left}, {right}")
    if left < 0 or right < 0:
        raise ValueError(f"w^{spin}_{mode} lies outside the wedge")
    return int(left), int(right)


def poisson_bracket(
    left: tuple[int, int], right: tuple[int, int]
) -> tuple[Fraction, tuple[int, int]] | None:
    """``{f, g}`` of two monomials on the plane, with ``{lambda_0, lambda_1} = 1/2``.

    The half is the normalisation that lands on the standard structure constant
    rather than twice it; with ``{lambda_0, lambda_1} = 1`` every bracket would
    come out doubled.
    """
    a, b = left
    c, d = right
    coefficient = Fraction(a * d - b * c, 2)
    if coefficient == 0:
        return None
    return coefficient, (a + c - 1, b + d - 1)


def poisson_realisation_residual(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> Fraction:
    """Difference between the Poisson bracket and the abstract structure constant.

    Zero on every pair in the wedge.  This is the check that the plane really
    carries ``w_{1+infinity}`` rather than merely something with the same name.
    """
    abstract_coefficient, abstract = wedge_bracket(left, right)
    product = poisson_bracket(wedge_monomial(*left), wedge_monomial(*right))
    if product is None:
        return abstract_coefficient
    concrete_coefficient, monomial = product
    if abstract_coefficient != 0 and monomial != wedge_monomial(*abstract):
        raise ValueError(f"bracket landed on {monomial}, expected {wedge_monomial(*abstract)}")
    return concrete_coefficient - abstract_coefficient


# --------------------------------------------------------------------------- #
# Part 3: the massive module
# --------------------------------------------------------------------------- #
#: Null components of a four-vector, ``(a0 + a3, a0 - a3, a1 + i a2, a1 - i a2)``.
#: Working in this basis is what keeps every coefficient rational: the celestial
#: vector ``qhat = (1+z zbar, z+zbar, -i(z-zbar), 1-z zbar)`` has the factor of
#: ``i`` only in the Cartesian frame.
NULL_COMPONENTS: tuple[str, str, str, str] = ("++", "--", "+-", "-+")

#: ``qhat`` and its derivatives in the null basis, as polynomials in ``(z, zbar)``
#: keyed by exponent pair.  ``qhat = (2, 2 z zbar, 2 z, 2 zbar)`` and
#: ``n = d_z d_zbar qhat = (0, 2, 0, 0)``.
QHAT: tuple[dict[tuple[int, int], Fraction], ...] = (
    {(0, 0): Fraction(2)},
    {(1, 1): Fraction(2)},
    {(1, 0): Fraction(2)},
    {(0, 1): Fraction(2)},
)
_NVEC: tuple[dict[tuple[int, int], Fraction], ...] = (
    {},
    {(0, 0): Fraction(2)},
    {},
    {},
)
_DZ_QHAT: tuple[dict[tuple[int, int], Fraction], ...] = (
    {},
    {(0, 1): Fraction(2)},
    {(0, 0): Fraction(2)},
    {},
)
_DZBAR_QHAT: tuple[dict[tuple[int, int], Fraction], ...] = (
    {},
    {(1, 0): Fraction(2)},
    {},
    {(0, 0): Fraction(2)},
)

#: Minkowski pairing in the null basis:
#: ``a.b = -(a_++ b_-- + a_-- b_++)/2 + (a_+- b_-+ + a_-+ b_+-)/2``.
_PAIRING: tuple[tuple[int, int, Fraction], ...] = (
    (0, 1, Fraction(-1, 2)),
    (1, 0, Fraction(-1, 2)),
    (2, 3, Fraction(1, 2)),
    (3, 2, Fraction(1, 2)),
)

#: A basis element ``(Delta, i, j, a, b)`` stands for
#: ``z^a zbar^b d_z^i d_zbar^j G_Delta``.
Key = tuple[Fraction, int, int, int, int]
State = dict[Key, Fraction]


def basis(delta: Fraction | int, i: int = 0, j: int = 0, a: int = 0, b: int = 0) -> State:
    """The single basis element ``z^a zbar^b d_z^i d_zbar^j G_Delta``."""
    if i < 0 or j < 0 or a < 0 or b < 0:
        raise ValueError("derivative and polynomial degrees must be non-negative")
    return {(Fraction(delta), i, j, a, b): Fraction(1)}


def _accumulate(state: State, key: Key, value: Fraction) -> None:
    if value:
        total = state.get(key, Fraction(0)) + value
        if total:
            state[key] = total
        else:
            state.pop(key, None)


def _multiply(state: State, polynomial: Mapping[tuple[int, int], Fraction]) -> State:
    out: State = {}
    for (delta, i, j, a, b), coefficient in state.items():
        for (p, q), weight in polynomial.items():
            _accumulate(out, (delta, i, j, a + p, b + q), coefficient * weight)
    return out


def _d_z(state: State) -> State:
    """``d_z`` on ``z^a zbar^b d_z^i d_zbar^j G_Delta``, by the Leibniz rule."""
    out: State = {}
    for (delta, i, j, a, b), coefficient in state.items():
        if a:
            _accumulate(out, (delta, i, j, a - 1, b), coefficient * a)
        _accumulate(out, (delta, i + 1, j, a, b), coefficient)
    return out


def _d_zbar(state: State) -> State:
    out: State = {}
    for (delta, i, j, a, b), coefficient in state.items():
        if b:
            _accumulate(out, (delta, i, j, a, b - 1), coefficient * b)
        _accumulate(out, (delta, i, j + 1, a, b), coefficient)
    return out


def _momentum_on_propagator(component: int, delta: Fraction) -> State:
    """``phat^mu G_Delta`` expanded in the module, derived rather than quoted.

    Expanding ``phat`` in the frame ``(qhat, d_z qhat, d_zbar qhat, n)``, whose
    only non-zero pairings are ``qhat.n = -2`` and ``d_z qhat . d_zbar qhat = 2``,
    gives

        phat  =  (u_zzbar qhat - u_zbar d_z qhat - u_z d_zbar qhat + u n) / 2 ,

    with ``u = -phat . qhat``.  Trading ``u``, ``u_z``, ``u_zbar`` and
    ``u_zzbar`` for derivatives and weight shifts of ``G_Delta`` needs one
    identity, ``u u_zzbar - u_z u_zbar = 1``, which
    :func:`propagator_identity_residual` checks; everything else is the chain
    rule.  The result agrees with equation (6.1) of Himwich--Pate.
    """
    if delta == 1:
        raise ValueError("the momentum operator is singular at Delta = 1")
    out: State = {}
    half = Fraction(1, 2)
    for (p, q), weight in _NVEC[component].items():
        _accumulate(out, (delta - 1, 0, 0, p, q), half * weight)
    for (p, q), weight in _DZ_QHAT[component].items():
        _accumulate(out, (delta - 1, 0, 1, p, q), half * weight / (delta - 1))
    for (p, q), weight in _DZBAR_QHAT[component].items():
        _accumulate(out, (delta - 1, 1, 0, p, q), half * weight / (delta - 1))
    for (p, q), weight in QHAT[component].items():
        _accumulate(out, (delta - 1, 1, 1, p, q), half * weight / (delta - 1) ** 2)
    for (p, q), weight in QHAT[component].items():
        _accumulate(out, (delta + 1, 0, 0, p, q), half * weight * delta / (delta - 1))
    return out


def momentum(component: int, state: State) -> State:
    """Multiplication by ``phat^mu`` on the module, in the null basis.

    ``phat`` is a function of the hyperboloid variables alone, so it commutes
    with ``d_z`` and ``d_zbar`` and with multiplication by any polynomial in
    ``(z, zbar)``.  The action on a general basis element is therefore fixed by
    the action on ``G_Delta``: differentiate that ``i`` and ``j`` times, then
    multiply by ``z^a zbar^b``.  Getting this wrong -- applying the operator
    directly to the polynomial prefactor -- breaks ``phat . phat = -1``, which is
    how the ordering was found.
    """
    if component not in range(4):
        raise ValueError(f"null component index must be 0..3; got {component}")
    out: State = {}
    for (delta, i, j, a, b), coefficient in state.items():
        part = _momentum_on_propagator(component, delta)
        for _ in range(i):
            part = _d_z(part)
        for _ in range(j):
            part = _d_zbar(part)
        part = _multiply(part, {(a, b): Fraction(1)})
        for key, value in part.items():
            _accumulate(out, key, coefficient * value)
    return out


def mass_shell_residual(state: State) -> State:
    """``(phat . phat + 1) |state>``, which must vanish identically.

    The referee for the whole construction.  The operator carries three
    independent ingredients -- the frame expansion, the ``Delta`` shifts, and the
    ordering of derivative against polynomial -- and the mass shell is a single
    scalar identity that fails if any one of them is wrong.
    """
    out: State = {}
    for mu, nu, weight in _PAIRING:
        for key, value in momentum(mu, momentum(nu, state)).items():
            _accumulate(out, key, weight * value)
    for key, value in state.items():
        _accumulate(out, key, value)
    return out


def commutator_residual(first: int, second: int, state: State) -> State:
    """``[phat^mu, phat^nu] |state>``: momenta commute, so this vanishes."""
    out: State = {}
    for key, value in momentum(first, momentum(second, state)).items():
        _accumulate(out, key, value)
    for key, value in momentum(second, momentum(first, state)).items():
        _accumulate(out, key, -value)
    return out


def boost_weight(key: Key) -> Fraction:
    """``N = Delta + i + j - a - b``, the grading the momentum shifts by its weight.

    ``phat^{++}`` raises it by one, ``phat^{--}`` lowers it by one, and the two
    mixed components leave it alone -- the boost weights of a four-vector,
    recovered from a module that was built without reference to them.
    """
    delta, i, j, a, b = key
    return delta + i + j - a - b


def spin_weight(key: Key) -> int:
    """``M = (a - i) - (b - j)``, the second Cartan grading.

    ``phat^{+-}`` raises it by one, ``phat^{-+}`` lowers it, and ``phat^{++}``,
    ``phat^{--}`` leave it alone.
    """
    _, i, j, a, b = key
    return (a - i) - (b - j)


def propagator_identity_residual(
    y: Fraction, w: Fraction, wbar: Fraction, z: Fraction, zbar: Fraction
) -> Fraction:
    """``u u_zzbar - u_z u_zbar - 1`` for ``u = (y^2 + (w-z)(wbar-zbar)) / y``.

    The one identity the derivation of the momentum operator needs, evaluated
    exactly at a point.  It is the statement that ``G_Delta`` is a genuine
    hyperbolic bulk-to-boundary propagator; every other step is the chain rule.
    """
    if y == 0:
        raise ValueError("the hyperboloid coordinate y must be non-zero")
    u = (y * y + (w - z) * (wbar - zbar)) / y
    u_z = -(wbar - zbar) / y
    u_zbar = -(w - z) / y
    u_zzbar = Fraction(1) / y
    return u * u_zzbar - u_z * u_zbar - 1


# --------------------------------------------------------------------------- #
# Part 4: the distinguished component, and the obstruction to inverting it
# --------------------------------------------------------------------------- #
#: ``-n . phat`` is the ``++`` null component: with ``n = (0, 2, 0, 0)`` the
#: pairing leaves ``-n.phat = phat^{++}``, and on the hyperboloid it equals
#: ``1/y``, so it is positive and the inverse is the object the generators need.
LIGHTCONE_COMPONENT = 0


def lightcone_momentum(state: State) -> State:
    """``-n . phat``, the component whose inverse powers define ``p > 2``.

    Bidiagonal: two terms, one lowering ``Delta`` while raising both derivative
    orders, one raising ``Delta`` alone.  The polynomial labels ``(a, b)`` are
    spectators because ``qhat^{++} = 2`` is constant, which is what reduces the
    inversion to a scalar recursion.
    """
    return momentum(LIGHTCONE_COMPONENT, state)


def inverse_recursion(delta: Fraction | int, order: int) -> list[Fraction]:
    """Coefficients ``v_0 .. v_order`` of ``(-n.phat)^{-1}`` by its recursion.

    Solving ``(-n.phat) v = |Delta; i, j>`` in the grading where the operator is
    bidiagonal gives ``alpha_{r-1} v_{r-1} + beta_r v_r = delta_{r,0}`` with

        alpha_{r-1} = (Delta - 2r)^{-2} ,
        beta_r      = (Delta - 1 - 2r) / (Delta - 2 - 2r) ,

    hence ``v_0 = 1 / beta_0`` and ``v_r = -alpha_{r-1} v_{r-1} / beta_r``.  One
    step, not a continued fraction: that is what bidiagonality buys.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    weight = Fraction(delta)
    if weight == 1 or weight == 2:
        raise ValueError(f"the recursion degenerates immediately at Delta = {weight}")
    coefficients = [(weight - 2) / (weight - 1)]
    for r in range(1, order + 1):
        if weight == 2 * r or weight == 2 * r + 1 or weight == 2 * r + 2:
            raise ValueError(
                f"the recursion degenerates at Delta = {weight}, step {r}"
            )
        alpha = Fraction(1) / (weight - 2 * r) ** 2
        beta = (weight - 1 - 2 * r) / (weight - 2 - 2 * r)
        coefficients.append(-alpha * coefficients[-1] / beta)
    return coefficients


def inverse_coefficient(delta: Fraction | int, step: int) -> Fraction:
    """The same coefficient in closed form, and the definition that survives.

    Unrolling the recursion telescopes: the even offsets ``d-2, d-4, ..., d-2k``
    and the odd offsets ``d-3, d-5, ..., d-2k-1`` interleave into one run of
    consecutive integers, leaving a falling factorial of length ``2k+1``,

        v_k  =  (-1)^k (d - 2k - 2) / [ (d-1)(d-2) ... (d-2k-1) ] .

    This is not merely the recursion rewritten.  Where the recursion produces
    ``0 * infinity`` -- at even ``Delta``, where ``beta_0`` is infinite and the
    first coefficient vanishes -- the closed form is finite and says what the
    later coefficients are, and it is the form in which the poles are visible:
    the denominator vanishes exactly when ``Delta`` is an integer in
    ``[1, 2k+1]``, and the numerator, which vanishes only at ``Delta = 2k+2``,
    can never cancel one.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    weight = Fraction(delta)
    denominator = Fraction(1)
    for s in range(1, 2 * step + 2):
        denominator *= weight - s
    if denominator == 0:
        raise ValueError(
            f"coefficient {step} is singular at Delta = {weight}"
        )
    sign = -1 if step % 2 else 1
    return sign * (weight - 2 * step - 2) / denominator


def inverse_series(
    delta: Fraction | int, i: int = 0, j: int = 0, a: int = 0, b: int = 0, *, order: int = 6
) -> State:
    """``(-n.phat)^{-1} |Delta; i, j; a, b>``, truncated after ``order + 1`` terms.

    The image lives on the arithmetic progression
    ``Delta - 1, Delta - 3, Delta - 5, ...`` with the derivative orders climbing
    in step, so the infinite family mixing Himwich--Pate report is
    one-dimensional and explicit rather than unstructured.
    """
    weight = Fraction(delta)
    out: State = {}
    for r in range(order + 1):
        _accumulate(
            out, (weight - 1 - 2 * r, i + r, j + r, a, b), inverse_coefficient(weight, r)
        )
    return out


def inverse_residual(
    delta: Fraction | int, i: int = 0, j: int = 0, a: int = 0, b: int = 0, *, order: int = 6
) -> State:
    """``(-n.phat) (-n.phat)^{-1} |...> - |...>`` for the truncated series.

    Exactly one term survives, the tail the truncation left behind, sitting at
    ``Delta - 2(order+1)``.  Its presence is the check that the recursion is
    right: every other term cancels identically in rational arithmetic.
    """
    series = inverse_series(delta, i, j, a, b, order=order)
    out = lightcone_momentum(series)
    for key, value in basis(delta, i, j, a, b).items():
        _accumulate(out, key, -value)
    return out


def obstruction_step(delta: Fraction | int) -> int | None:
    """First ``r >= 0`` at which the inversion recursion fails, or ``None``.

    Read off the closed form of :func:`inverse_coefficient`, whose denominator is
    the falling factorial ``(Delta-1)(Delta-2)...(Delta-2r-1)``.  That vanishes
    exactly when ``Delta`` is an integer in ``[1, 2r+1]``, so the first failure is
    at ``r = ceil((Delta-1)/2)`` for positive integer ``Delta`` and never
    otherwise.  A non-integer weight -- in particular every principal-series
    weight ``1 + i lambda`` with ``lambda != 0`` -- misses the run of integers at
    every order.
    """
    weight = Fraction(delta)
    if weight.denominator != 1 or weight <= 0:
        return None
    numerator = int(weight)
    return -((1 - numerator) // 2)  # ceil((Delta - 1) / 2)


def inverse_exists(delta: Fraction | int, order: int) -> bool:
    """Whether ``(-n.phat)^{-1}`` is defined out to ``order`` terms at ``Delta``.

    True for every non-integer weight and every order, including the whole
    principal series ``Delta = 1 + i lambda`` with ``lambda != 0``, whose
    progression never meets ``0`` or ``1``.  False at every positive integer
    ``Delta`` once ``order`` reaches :func:`obstruction_step`.
    """
    step = obstruction_step(delta)
    return step is None or step > order


@dataclass(frozen=True)
class InversionReport:
    """What the inversion does at one weight."""

    delta: Fraction
    order: int
    exists: bool
    obstruction: int | None
    families: tuple[Fraction, ...]

    @property
    def spacing(self) -> Fraction | None:
        """Gap between consecutive conformal families in the image."""
        if len(self.families) < 2:
            return None
        return self.families[0] - self.families[1]


def inversion_report(delta: Fraction | int, *, order: int = 6) -> InversionReport:
    """Assemble the inversion data at one weight."""
    weight = Fraction(delta)
    step = obstruction_step(weight)
    exists = inverse_exists(weight, order)
    families = (
        tuple(weight - 1 - 2 * r for r in range(order + 1)) if exists else ()
    )
    return InversionReport(
        delta=weight, order=order, exists=exists, obstruction=step, families=families
    )
