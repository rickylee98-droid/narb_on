"""Which spins a discrete structure can protect.

A quantum cellular automaton, a lattice gauge theory, a spin liquid, a tensor
network -- any model built on a discrete set of cells -- has a *finite*
rotation group, not ``SO(3)``.  The question this module answers exactly is:

    for which finite ``G < SO(3)`` does the spin-``s`` multiplet survive as a
    single protected object?

The answer decides whether a massless spin-``s`` excitation can emerge without
fine tuning.  A multiplet that splits into several irreducible pieces has one
independent coupling per piece, and the pieces acquire independent velocities;
restoring isotropy then costs a tuning for every piece beyond the first.  A
multiplet that stays irreducible is protected by Schur's lemma: the invariant
Hamiltonian on it is forced to be a multiple of the identity and the degeneracy
cannot be lifted at all.

Everything here is one computation.  For a class function on ``G``, the norm

    <chi_l, chi_l>_G  =  (1/|G|) sum_g |chi_l(g)|^2

is simultaneously

  * the sum of squared multiplicities in the decomposition of ``l`` (so it is
    ``1`` exactly when ``l`` is irreducible), and
  * the dimension of the commutant of ``G`` acting on the ``(2l+1)``-dimensional
    multiplet, i.e. the number of independent invariant couplings.

So the representation theory and the fine-tuning count are literally the same
number, and `tuning_cost` is `character_norm` minus one.

What comes out
--------------

The finite subgroups of ``SO(3)`` are ``C_n``, ``D_n``, ``T``, ``O``, ``I``
(the classical ADE classification).  Computing the norm of ``chi_l`` over all
of them gives:

    s = 1  (dim 3)   irreducible under T, O and I
    s = 2  (dim 5)   irreducible under I alone
    s >= 3           irreducible under nothing

The ``s <= 2`` ceiling is not assumed.  It falls out of the fact that the
largest irreducible representation of any finite subgroup of ``SO(3)`` has
dimension ``5``, and ``5 = 2*2 + 1``.  That is the Weinberg-Witten bound on
massless helicity, arrived at from finite group theory rather than from a
Lorentz-covariant stress tensor.

The second half is the crystallographic restriction theorem: a periodic lattice
admits rotations of order ``1, 2, 3, 4, 6`` only, so no Bravais lattice has
icosahedral symmetry.  Combining the two:

    an emergent photon is protected on ordinary lattices;
    an emergent graviton is protected only on icosahedral -- hence
    quasicrystalline, hence aperiodic -- structures;
    nothing above spin two is protected anywhere.

Relevant or irrelevant: the sharpened statement
-----------------------------------------------

"One tuned parameter" undersells it.  `tuning_ladder` resolves the cost by
order in momentum, decomposing the couplings at order ``k^n`` as
``Sym^n(vector) x End(multiplet)`` and comparing the ``G``-invariant count with
the ``SO(3)``-invariant one.  The rung that matters is ``n = 0``:

    spin 1:  T splits first at n=1,  O at n=2,  I at n=4  -- all with n=0 free
    spin 2:  T and O split at n=0,   I first at n=2

An excess at ``n = 0`` is a splitting at *zero momentum*.  The pieces of the
multiplet acquire different gaps, so there is no massless spin-``s`` object at
all, and nothing suppresses the failure at low energy: it is a relevant
perturbation.  An excess first appearing at ``n > 0`` is a velocity or
dispersion anisotropy, suppressed by powers of ``k a`` -- an irrelevant
operator that flows away in the infrared.

So the no-go is not that a cubic graviton needs one knob.  It is that the cubic
failure is *relevant* and the icosahedral failure is *irrelevant*.

The spin-1 row is the control, and it is the reason to believe the method.  A
cubic lattice is isotropic through order ``k^1``, so an emergent photon's linear
dispersion is protected by symmetry and the leading correction is an irrelevant
``k^2`` term -- which is exactly the regime where emergent photons are known to
work.  The same computation, on the same lattices, says spin two fails at ``k^0``.

States or fields, and a weakening of the above
----------------------------------------------

The ladder treats the multiplet as five *states*, which is right for a gapped
quadrupolar excitation and wrong for a gauge field: a graviton has two
propagating helicities and no zero-momentum states to split.  Re-asked for a
field, the object is the elastic tensor in ``Sym^2(Sym^2(vector))`` and the
question is how many independent constants ``G`` permits.  `elastic_constant_count`
answers it, and reproduces the entire crystal-system table:

    triclinic 21, monoclinic 13, orthorhombic 9, tetragonal 7 or 6,
    trigonal 7 or 6, hexagonal 5, cubic 3, isotropic 2

and, for the icosahedral group, ``2`` -- icosahedral quasicrystals are
elastically isotropic, which is a measured fact and was not put in.  Twelve
independent numbers, all correct, from the same cyclotomic arithmetic.

This *weakens* the no-go above and the weakening is the point.  In the field
reading a cubic lattice does not gap the multiplet apart; it gives three elastic
constants where isotropy allows two, so the failure is a leading-order velocity
anisotropy -- the Zener ratio ``2 C_44 / (C_11 - C_12)`` must be tuned to one.
Real, but tunable, not a relevant-operator catastrophe.  So:

    emergent spin two as excitations   -> strong no-go on any lattice
    emergent spin two as a gauge field -> one tuned relation on a cubic lattice

Both readings leave the icosahedral group free, and they agree exactly on which
groups need no tuning at all -- a test asserts that agreement across the
classification.

Novelty
-------

Every ingredient is classical: the ADE classification of finite rotation
groups (Klein), the crystallographic restriction theorem, the character theory,
and the fact that ``l = 2`` restricted to the icosahedral group is the
five-dimensional irrep ``H``.  Solid-state texts routinely quote the cubic
splitting ``l = 2 -> E_g + T_2g``.

What is assembled here, and what I have not seen stated, is the *conjunction*
read as a no-go for emergent gravity on lattices: that the maximal irrep
dimension of finite ``SO(3)`` subgroups reproduces the massless-helicity bound,
that spin two singles out the unique non-crystallographic case, and that the
commutant dimension turns the same character norm into an explicit count of
how many parameters an emergent-graviton construction must tune.  If this is in
the literature it is a rediscovery; I could not check, because arxiv.org is
blocked from this environment and web search was rate limited.  Treat the
framing as unverified and the arithmetic as exact.

Referees
--------

The class data is not taken on trust.  Three classical facts are recomputed
from it and none of them was used to build it:

    lowest non-trivial invariant of T  at l = 3   (the tetrahedral invariant)
    lowest non-trivial invariant of O  at l = 4   (the cubic harmonic K_4)
    lowest non-trivial invariant of I  at l = 6   (the icosahedral invariant)

If the angle multiset of any group were wrong these degrees would move.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Iterator, Sequence

import sympy as sp

__all__ = [
    "RotationGroup",
    "cyclic",
    "dihedral",
    "TETRAHEDRAL",
    "OCTAHEDRAL",
    "ICOSAHEDRAL",
    "POLYHEDRAL",
    "finite_rotation_groups",
    "spin_character",
    "character_norm",
    "character_inner_product",
    "is_irreducible",
    "tuning_cost",
    "invariant_count",
    "lowest_invariant_degree",
    "LOWEST_INVARIANT_DEGREES",
    "CRYSTALLOGRAPHIC_ORDERS",
    "is_crystallographic",
    "protecting_groups",
    "maximal_protected_spin",
    "PROTECTED_SPIN_CEILING",
    "MAXIMAL_IRREP_DIMENSION",
    "graviton_requires_aperiodic",
    "polynomial_multiplet_content",
    "operator_multiplet_content",
    "isotropic_coupling_count",
    "lattice_coupling_count",
    "TuningRung",
    "tuning_ladder",
    "first_anisotropic_degree",
    "ANISOTROPY_ONSET",
    "zero_momentum_splitting",
    "elastic_constant_count",
    "ISOTROPIC_ELASTIC_CONSTANTS",
    "is_elastically_isotropic",
    "elastic_excess",
    "helicity_alias_modulus",
    "helicity_is_resolved",
    "minimal_axis_order",
    "aliasing_is_insufficient",
]


# ---------------------------------------------------------------------------
# finite subgroups of SO(3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RotationGroup:
    """A finite subgroup of ``SO(3)``, recorded by its rotation angles.

    A character of ``SO(3)`` restricted to ``G`` is a class function that
    depends on a group element only through its rotation angle, so the multiset
    of angles is all the data any computation here needs.  Angles are stored as
    exact fractions of a full turn.

    Attributes
    ----------
    name:
        Human readable label, e.g. ``"I"`` or ``"D_6"``.
    spectrum:
        Pairs ``(turn, multiplicity)``.  ``turn`` is the rotation angle divided
        by ``2*pi``, so the identity is ``turn = 0``.
    """

    name: str
    spectrum: tuple[tuple[Fraction, int], ...]

    def __post_init__(self) -> None:
        if not self.spectrum:
            raise ValueError(f"{self.name}: empty angle spectrum")
        for turn, count in self.spectrum:
            if not isinstance(turn, Fraction):
                raise TypeError(f"{self.name}: turn {turn!r} is not a Fraction")
            if not 0 <= turn < 1:
                raise ValueError(f"{self.name}: turn {turn} outside [0, 1)")
            if count <= 0:
                raise ValueError(f"{self.name}: multiplicity {count} is not positive")
        identity = [count for turn, count in self.spectrum if turn == 0]
        if sum(identity) != 1:
            raise ValueError(f"{self.name}: expected exactly one identity element")

    @property
    def order(self) -> int:
        """The number of group elements."""
        return sum(count for _, count in self.spectrum)

    @property
    def rotation_orders(self) -> frozenset[int]:
        """Orders of the rotations present, e.g. ``{1, 2, 3, 5}`` for ``I``."""
        return frozenset(turn.denominator for turn, _ in self.spectrum)

    def __str__(self) -> str:
        return self.name


def cyclic(n: int) -> RotationGroup:
    """The cyclic group ``C_n`` of rotations about a single axis."""
    if n < 1:
        raise ValueError(f"cyclic order must be positive, got {n}")
    return RotationGroup(
        name=f"C_{n}",
        spectrum=tuple((Fraction(k, n), 1) for k in range(n)),
    )


def dihedral(n: int) -> RotationGroup:
    """The dihedral group ``D_n``: ``C_n`` plus ``n`` perpendicular half turns."""
    if n < 1:
        raise ValueError(f"dihedral order must be positive, got {n}")
    rotations: dict[Fraction, int] = {}
    for k in range(n):
        rotations[Fraction(k, n)] = rotations.get(Fraction(k, n), 0) + 1
    half = Fraction(1, 2)
    rotations[half] = rotations.get(half, 0) + n
    return RotationGroup(
        name=f"D_{n}",
        spectrum=tuple(sorted(rotations.items())),
    )


#: The rotation group of the tetrahedron, isomorphic to ``A_4``.
TETRAHEDRAL = RotationGroup(
    name="T",
    spectrum=((Fraction(0), 1), (Fraction(1, 3), 8), (Fraction(1, 2), 3)),
)

#: The rotation group of the cube and octahedron, isomorphic to ``S_4``.
#: Six four-fold rotations, eight three-fold, and nine half turns
#: (three face-diagonal pairs and six edge axes).
OCTAHEDRAL = RotationGroup(
    name="O",
    spectrum=(
        (Fraction(0), 1),
        (Fraction(1, 4), 6),
        (Fraction(1, 3), 8),
        (Fraction(1, 2), 9),
    ),
)

#: The rotation group of the icosahedron and dodecahedron, isomorphic to ``A_5``.
ICOSAHEDRAL = RotationGroup(
    name="I",
    spectrum=(
        (Fraction(0), 1),
        (Fraction(1, 5), 12),
        (Fraction(1, 3), 20),
        (Fraction(2, 5), 12),
        (Fraction(1, 2), 15),
    ),
)

#: The three exceptional (polyhedral) finite rotation groups.
POLYHEDRAL: tuple[RotationGroup, ...] = (TETRAHEDRAL, OCTAHEDRAL, ICOSAHEDRAL)


def finite_rotation_groups(bound: int = 12) -> Iterator[RotationGroup]:
    """Every finite subgroup of ``SO(3)`` with cyclic/dihedral index ``<= bound``.

    The classification (Klein) is exhaustive: ``C_n``, ``D_n`` for ``n >= 1``,
    and the three polyhedral groups.  ``bound`` truncates the two infinite
    families; the polyhedral groups are always included.
    """
    if bound < 1:
        raise ValueError(f"bound must be positive, got {bound}")
    for n in range(1, bound + 1):
        yield cyclic(n)
    for n in range(1, bound + 1):
        yield dihedral(n)
    yield from POLYHEDRAL


# ---------------------------------------------------------------------------
# characters
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def spin_character(spin: int, turn: Fraction) -> sp.Expr:
    """``chi_l`` of a rotation by ``turn`` full turns, exactly.

    This is the Dirichlet kernel ``sum_{m=-l}^{l} exp(i m theta)``, written as
    ``1 + 2 sum_{m=1}^{l} cos(m theta)`` so the value is manifestly real.  The
    result is an algebraic integer in a cyclotomic field.

    Individual character values are irrational in general (``chi_1`` of a fifth
    turn is ``-1/phi``), so this is for display and spot checks.  The class sums
    that carry the argument are computed by `_root_of_unity_sum`, which stays in
    ``Z[zeta]`` and never asks a simplifier to recognise an identity.
    """
    if spin < 0:
        raise ValueError(f"spin must be non-negative, got {spin}")
    if not isinstance(turn, Fraction):
        raise TypeError(f"turn must be a Fraction, got {turn!r}")
    if spin == 0:
        return sp.Integer(1)
    angle = 2 * sp.pi * sp.Rational(turn.numerator, turn.denominator)
    total = sp.Integer(1) + 2 * sp.Add(*(sp.cos(m * angle) for m in range(1, spin + 1)))
    return sp.nsimplify(sp.expand_trig(total))


def _root_of_unity_sum(terms: dict[Fraction, int]) -> int:
    """Evaluate ``sum_t c_t * exp(2 pi i t)`` exactly, requiring a rational integer.

    Every quantity in this module is such a sum: characters are sums of roots of
    unity and so are their products, and averaging over a group leaves a
    rational integer.  Rather than hope a trigonometric simplifier closes the
    sum -- it does not, already at a seventh of a turn -- the sum is built as a
    polynomial in ``x`` and reduced modulo the ``L``-th cyclotomic polynomial,
    which is exactly arithmetic in ``Z[zeta_L]``.  A rational integer is a
    constant remainder, and anything else is a bug rather than a value.
    """
    live = {turn: coeff for turn, coeff in terms.items() if coeff}
    if not live:
        return 0
    modulus = 1
    for turn in live:
        modulus = modulus * turn.denominator // _gcd(modulus, turn.denominator)
    variable = sp.Symbol("x")
    powers: dict[int, int] = {}
    for turn, coeff in live.items():
        exponent = (turn.numerator * (modulus // turn.denominator)) % modulus
        powers[exponent] = powers.get(exponent, 0) + coeff
    polynomial = sp.Poly(
        sum(coeff * variable**exponent for exponent, coeff in powers.items()),
        variable,
        domain="ZZ",
    )
    remainder = sp.rem(
        polynomial, sp.Poly(sp.cyclotomic_poly(modulus, variable), variable, domain="ZZ")
    )
    if remainder.degree() > 0:
        raise ArithmeticError(
            f"root-of-unity sum is not a rational integer: {remainder.as_expr()}"
        )
    value = remainder.as_expr()
    if not value.is_Integer:
        raise ArithmeticError(f"root-of-unity sum did not reduce to an integer: {value}")
    return int(value)


def _gcd(left: int, right: int) -> int:
    while right:
        left, right = right, left % right
    return left


def character_norm(spin: int, group: RotationGroup) -> int:
    """``<chi_l, chi_l>_G``: sum of squared multiplicities, and commutant dimension.

    Two readings of one number.  As a character inner product it is
    ``sum_i m_i^2`` over the decomposition of the multiplet into irreducibles,
    so it equals ``1`` exactly when the multiplet stays irreducible.  As a
    dimension it counts the ``G``-invariant Hermitian operators on the
    ``(2l+1)``-dimensional multiplet, i.e. the independent couplings a model
    builder may switch on.

    Computed from ``chi * conj(chi) = sum_{m, m'} zeta^{(m - m') t}``, so the
    exponent ``d = m - m'`` carries multiplicity ``2l + 1 - |d|``.
    """
    if spin < 0:
        raise ValueError(f"spin must be non-negative, got {spin}")
    terms: dict[Fraction, int] = {}
    width = 2 * spin + 1
    for shift in range(-2 * spin, 2 * spin + 1):
        weight = width - abs(shift)
        for turn, count in group.spectrum:
            key = Fraction(shift) * turn
            key -= key.numerator // key.denominator
            terms[key] = terms.get(key, 0) + weight * count
    total = _root_of_unity_sum(terms)
    if total % group.order:
        raise ArithmeticError(
            f"character norm for spin {spin} on {group.name} is not an integer: "
            f"{total}/{group.order}"
        )
    return total // group.order


def character_inner_product(
    spin_left: int, spin_right: int, group: RotationGroup
) -> int:
    """``<chi_a, chi_b>_G``: shared irreducible content of two multiplets.

    Over ``SO(3)`` this is ``delta_ab``.  Over a finite subgroup it can exceed
    the delta, and every excess is a coupling that a lattice model may switch on
    and a rotationally invariant one may not.  That excess is the whole content
    of `tuning_ladder`.

    Computed from ``chi_a * conj(chi_b) = sum_{m, m'} zeta^{(m - m') t}`` where
    ``m`` runs over ``[-a, a]`` and ``m'`` over ``[-b, b]``; the shift ``d`` then
    carries the overlap count of the two ranges.
    """
    if spin_left < 0 or spin_right < 0:
        raise ValueError(
            f"spins must be non-negative, got {spin_left} and {spin_right}"
        )
    terms: dict[Fraction, int] = {}
    for shift in range(-(spin_left + spin_right), spin_left + spin_right + 1):
        low = max(-spin_left, shift - spin_right)
        high = min(spin_left, shift + spin_right)
        weight = high - low + 1
        if weight <= 0:
            continue
        for turn, count in group.spectrum:
            key = Fraction(shift) * turn
            key -= key.numerator // key.denominator
            terms[key] = terms.get(key, 0) + weight * count
    total = _root_of_unity_sum(terms)
    if total % group.order:
        raise ArithmeticError(
            f"inner product <{spin_left},{spin_right}> on {group.name} is not an "
            f"integer: {total}/{group.order}"
        )
    return total // group.order


def is_irreducible(spin: int, group: RotationGroup) -> bool:
    """Does the spin-``l`` multiplet stay irreducible when restricted to ``G``?"""
    return character_norm(spin, group) == 1


def tuning_cost(spin: int, group: RotationGroup) -> int:
    """Independent couplings that must be tuned to restore the full multiplet.

    ``character_norm - 1``.  Zero means Schur's lemma protects the degeneracy
    outright; ``k > 0`` means a model builder has ``k`` knobs that generically
    split the multiplet and must be set by hand.
    """
    return character_norm(spin, group) - 1


def invariant_count(spin: int, group: RotationGroup) -> int:
    """``<chi_l, 1>_G``: the number of ``G``-invariant spherical harmonics at degree ``l``."""
    if spin < 0:
        raise ValueError(f"spin must be non-negative, got {spin}")
    terms: dict[Fraction, int] = {}
    for order in range(-spin, spin + 1):
        for turn, count in group.spectrum:
            key = Fraction(order) * turn
            key -= key.numerator // key.denominator
            terms[key] = terms.get(key, 0) + count
    total = _root_of_unity_sum(terms)
    if total % group.order:
        raise ArithmeticError(
            f"invariant count for spin {spin} on {group.name} is not an integer: "
            f"{total}/{group.order}"
        )
    value = total // group.order
    if value < 0:
        raise ArithmeticError(
            f"invariant count for spin {spin} on {group.name} is negative: {value}"
        )
    return value


def lowest_invariant_degree(group: RotationGroup, search_limit: int = 16) -> int:
    """Smallest ``l > 0`` carrying a ``G``-invariant harmonic.

    An independent check on the class data: this reproduces the classical
    invariant degrees of the polyhedral groups without being told them.
    """
    for spin in range(1, search_limit + 1):
        if invariant_count(spin, group) > 0:
            return spin
    raise ValueError(
        f"{group.name}: no invariant harmonic found up to degree {search_limit}"
    )


#: Classical lowest invariant degrees of the polyhedral groups, recomputed by
#: `lowest_invariant_degree` as a referee on the angle multisets.
LOWEST_INVARIANT_DEGREES: dict[str, int] = {"T": 3, "O": 4, "I": 6}


# ---------------------------------------------------------------------------
# the crystallographic restriction
# ---------------------------------------------------------------------------

#: Rotation orders compatible with a periodic lattice in any dimension <= 3.
CRYSTALLOGRAPHIC_ORDERS: frozenset[int] = frozenset({1, 2, 3, 4, 6})


def is_crystallographic(group: RotationGroup) -> bool:
    """Can ``G`` be the point group of a periodic lattice?

    The crystallographic restriction theorem: a rotation of order ``n`` maps a
    lattice to itself only for ``n`` in ``{1, 2, 3, 4, 6}``, because the trace
    of the rotation in a lattice basis must be an integer and equals
    ``1 + 2 cos(2 pi / n)``.
    """
    return group.rotation_orders <= CRYSTALLOGRAPHIC_ORDERS


def protecting_groups(
    spin: int, bound: int = 12
) -> tuple[RotationGroup, ...]:
    """Every finite rotation group under which spin ``l`` stays irreducible."""
    return tuple(
        group for group in finite_rotation_groups(bound) if is_irreducible(spin, group)
    )


def maximal_protected_spin(group: RotationGroup, search_limit: int = 8) -> int:
    """Largest ``l`` that ``G`` keeps irreducible.

    Bounded above by ``(d_max - 1) / 2`` where ``d_max`` is the largest
    irreducible dimension of ``G``; every group here has ``d_max <= 5``.
    """
    best = 0
    for spin in range(search_limit + 1):
        if is_irreducible(spin, group):
            best = spin
    return best


#: The largest irreducible representation of any finite subgroup of ``SO(3)``.
#: Attained only by the icosahedral group, whose ``H`` irrep is the spin-2
#: multiplet itself.
MAXIMAL_IRREP_DIMENSION: int = 5

#: ``(MAXIMAL_IRREP_DIMENSION - 1) // 2``.  The largest spin any discrete
#: structure can protect -- and the Weinberg-Witten massless helicity bound,
#: reached here from finite group theory instead of from a stress tensor.
PROTECTED_SPIN_CEILING: int = (MAXIMAL_IRREP_DIMENSION - 1) // 2


def graviton_requires_aperiodic(bound: int = 12) -> bool:
    """Is every spin-2-protecting rotation group non-crystallographic?

    True is the no-go: a symmetry-protected emergent graviton cannot live on a
    periodic lattice, and the minimal structure that supports one is
    icosahedral, hence quasicrystalline.
    """
    groups = protecting_groups(2, bound)
    if not groups:
        raise ArithmeticError("no group protects spin 2; classification is broken")
    return all(not is_crystallographic(group) for group in groups)


# ---------------------------------------------------------------------------
# the tuning ladder: where anisotropy first bites, order by order in k
# ---------------------------------------------------------------------------


def polynomial_multiplet_content(degree: int) -> tuple[int, ...]:
    """Spins appearing in ``Sym^n`` of the vector representation.

    Degree-``n`` polynomials in three variables decompose into harmonics of
    degree ``n, n-2, n-4, ...`` down to ``1`` or ``0``, each once.  This is the
    momentum dependence available to a coupling at order ``n`` in ``k``.
    """
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    return tuple(range(degree, -1, -2))


def operator_multiplet_content(spin: int) -> tuple[int, ...]:
    """Spins appearing in ``End`` of the spin-``s`` multiplet: ``0`` through ``2s``."""
    if spin < 0:
        raise ValueError(f"spin must be non-negative, got {spin}")
    return tuple(range(0, 2 * spin + 1))


def isotropic_coupling_count(spin: int, degree: int) -> int:
    """Independent ``SO(3)``-invariant couplings at order ``k^degree``.

    Over ``SO(3)`` two multiplets pair only when their spins agree, so this is
    just how many spins the polynomial and operator contents share.
    """
    return len(
        set(polynomial_multiplet_content(degree)) & set(operator_multiplet_content(spin))
    )


def lattice_coupling_count(spin: int, degree: int, group: RotationGroup) -> int:
    """Independent ``G``-invariant couplings at order ``k^degree``.

    Always at least `isotropic_coupling_count`: a rotationally invariant term is
    in particular ``G`` invariant.  The difference is what must be tuned away.
    """
    return sum(
        character_inner_product(momentum, operator, group)
        for momentum in polynomial_multiplet_content(degree)
        for operator in operator_multiplet_content(spin)
    )


@dataclass(frozen=True)
class TuningRung:
    """One order in the momentum expansion.

    Attributes
    ----------
    degree:
        Power of ``k``.
    lattice:
        Couplings a ``G``-symmetric model may switch on at this order.
    isotropic:
        Couplings a rotationally invariant model may switch on.
    excess:
        ``lattice - isotropic``: parameters that must be tuned to zero (or to
        each other) for isotropy to survive at this order.
    """

    degree: int
    lattice: int
    isotropic: int
    excess: int


def tuning_ladder(
    spin: int, group: RotationGroup, max_degree: int = 6
) -> tuple[TuningRung, ...]:
    """How much isotropy costs, order by order in momentum.

    The physically decisive rung is ``degree = 0``.  An excess there is a
    splitting *at zero momentum* -- the multiplet is gapped apart, so there is no
    spin-``s`` object at all, and nothing suppresses the failure at low energy.
    Excesses at higher degree are velocity and dispersion anisotropies,
    suppressed by powers of ``k a`` and irrelevant in the infrared.

    So the reading is: onset at degree 0 is fatal, onset at degree ``n > 0`` is
    an irrelevant operator and emergent isotropy is plausible.
    """
    if max_degree < 0:
        raise ValueError(f"max_degree must be non-negative, got {max_degree}")
    rungs = []
    for degree in range(max_degree + 1):
        lattice = lattice_coupling_count(spin, degree, group)
        isotropic = isotropic_coupling_count(spin, degree)
        excess = lattice - isotropic
        if excess < 0:
            raise ArithmeticError(
                f"{group.name}: fewer lattice than isotropic couplings at degree "
                f"{degree} ({lattice} < {isotropic}), which is impossible"
            )
        rungs.append(
            TuningRung(
                degree=degree, lattice=lattice, isotropic=isotropic, excess=excess
            )
        )
    return tuple(rungs)


def first_anisotropic_degree(
    spin: int, group: RotationGroup, limit: int = 8
) -> int | None:
    """Lowest order in ``k`` at which ``G`` permits an anisotropy.

    ``None`` if none appears up to ``limit`` -- which for these groups means the
    search was too short, not that the group is fully isotropic; only ``SO(3)``
    itself is that.
    """
    for rung in tuning_ladder(spin, group, limit):
        if rung.excess:
            return rung.degree
    return None


#: Onset degree of anisotropy for the spin-2 multiplet, by group.  The gap
#: between the crystallographic groups (degree 0, a zero-momentum splitting)
#: and the icosahedral group (degree 2, a velocity anisotropy) is the
#: quantitative form of the no-go.
ANISOTROPY_ONSET: dict[str, int] = {"T": 0, "O": 0, "I": 2}


def zero_momentum_splitting(spin: int, group: RotationGroup) -> int:
    """Couplings that split the multiplet at ``k = 0``: the fatal ones.

    This is `tuning_ladder`'s degree-zero excess, isolated because it is the
    rung that decides whether a spin-``s`` object exists at all.  It equals
    `tuning_cost`, since the degree-zero couplings on a multiplet are exactly
    its commutant.
    """
    return tuning_ladder(spin, group, max_degree=0)[0].excess


# ---------------------------------------------------------------------------
# the gauge-field reading: elastic constants
# ---------------------------------------------------------------------------
#
# The ladder above treats the spin-2 multiplet as five *states*, which is the
# right reading for a gapped quadrupolar excitation but not for a gauge field.
# A graviton is a field with two propagating helicities, so the obstruction has
# to be re-asked in terms of the quadratic effective action rather than a
# spectrum.  For a symmetric rank-2 field that action is an elastic tensor,
# and the same invariant count applies to it -- with an external referee
# attached, because the answers are measured quantities.


def _shift_product(
    left: dict[int, Fraction], right: dict[int, Fraction]
) -> dict[int, Fraction]:
    """Multiply two virtual characters written as ``sum_d c_d zeta^{d t}``."""
    product: dict[int, Fraction] = {}
    for left_shift, left_coeff in left.items():
        for right_shift, right_coeff in right.items():
            key = left_shift + right_shift
            product[key] = product.get(key, Fraction(0)) + left_coeff * right_coeff
    return product


def _shift_double(source: dict[int, Fraction]) -> dict[int, Fraction]:
    """The character of ``g -> g^2``: every shift doubled."""
    doubled: dict[int, Fraction] = {}
    for shift, coeff in source.items():
        doubled[2 * shift] = doubled.get(2 * shift, Fraction(0)) + coeff
    return doubled


def _shift_symmetric_square(source: dict[int, Fraction]) -> dict[int, Fraction]:
    """``chi_{Sym^2 W}(g) = (chi_W(g)^2 + chi_W(g^2)) / 2``."""
    squared = _shift_product(source, source)
    doubled = _shift_double(source)
    combined: dict[int, Fraction] = {}
    for shift in set(squared) | set(doubled):
        value = squared.get(shift, Fraction(0)) + doubled.get(shift, Fraction(0))
        if value:
            combined[shift] = value / 2
    return combined


def _spin_shifts(spin: int) -> dict[int, Fraction]:
    """``chi_l`` as ``sum_{m=-l}^{l} zeta^{m t}``."""
    return {magnetic: Fraction(1) for magnetic in range(-spin, spin + 1)}


def _average_shifts(source: dict[int, Fraction], group: RotationGroup) -> int:
    """Average a virtual character over ``G``, requiring an integer."""
    scale = 1
    for coeff in source.values():
        scale = scale * coeff.denominator // _gcd(scale, coeff.denominator)
    terms: dict[Fraction, int] = {}
    for shift, coeff in source.items():
        integral = int(coeff * scale)
        if not integral:
            continue
        for turn, count in group.spectrum:
            key = Fraction(shift) * turn
            key -= key.numerator // key.denominator
            terms[key] = terms.get(key, 0) + integral * count
    total = _root_of_unity_sum(terms)
    divisor = scale * group.order
    if total % divisor:
        raise ArithmeticError(
            f"average over {group.name} is not an integer: {total}/{divisor}"
        )
    return total // divisor


def elastic_constant_count(group: RotationGroup) -> int:
    """Independent elastic constants of a medium with rotational symmetry ``G``.

    The elastic tensor ``C_ijkl`` is symmetric within each index pair and under
    exchange of the pairs, so it lives in ``Sym^2(Sym^2(V))`` with ``V`` the
    vector representation -- twenty-one components before any symmetry.  The
    number surviving is the dimension of the ``G``-invariant subspace.

    This is the gauge-field reading of the obstruction.  Where `tuning_ladder`
    asks whether five *states* stay degenerate, this asks how many independent
    couplings the quadratic action of a symmetric rank-2 *field* may carry.

    It also comes with a referee that no part of this module was built to
    satisfy, because the answers are measured:

        isotropic medium      2   (the Lame constants)
        cubic crystal         3   (C_11, C_12, C_44)
        icosahedral           2   -- quasicrystals are elastically isotropic

    The last line is a known property of icosahedral quasicrystals, and it is
    the same conclusion the character norm reached: the icosahedral group is the
    unique finite rotation group that cannot tell a quadrupole from a rotation.
    """
    elastic = _shift_symmetric_square(_shift_symmetric_square(_spin_shifts(1)))
    return _average_shifts(elastic, group)


#: Independent elastic constants under full rotational symmetry: the Lame pair.
ISOTROPIC_ELASTIC_CONSTANTS: int = 2


def is_elastically_isotropic(group: RotationGroup) -> bool:
    """Does ``G`` force the elastic tensor to the isotropic two-parameter form?"""
    return elastic_constant_count(group) == ISOTROPIC_ELASTIC_CONSTANTS


def elastic_excess(group: RotationGroup) -> int:
    """Elastic constants beyond the isotropic two: the anisotropy that must be tuned.

    For a cubic crystal this is ``1``, and the single relation to impose is
    ``2 C_44 = C_11 - C_12`` -- the statement that the Zener anisotropy ratio is
    one.  For the icosahedral group it is ``0``.
    """
    return elastic_constant_count(group) - ISOTROPIC_ELASTIC_CONSTANTS


# ---------------------------------------------------------------------------
# helicity aliasing on a single axis
# ---------------------------------------------------------------------------


def helicity_alias_modulus(axis_order: int) -> int:
    """Helicity is defined only modulo the order of the axis it is measured on.

    A rotation by ``2 pi / n`` about the propagation axis multiplies a helicity
    ``h`` state by ``exp(2 pi i h / n)``, which depends on ``h`` only through
    ``h mod n``.
    """
    if axis_order < 1:
        raise ValueError(f"axis order must be positive, got {axis_order}")
    return axis_order


def helicity_is_resolved(axis_order: int, spin: int) -> bool:
    """Are the helicities ``-s .. s`` pairwise distinct modulo ``n``?

    Necessary for an ``n``-fold axis to tell a spin-``s`` excitation from its
    lower-helicity neighbours.  Reduces to ``n >= 2s + 1``.
    """
    if spin < 0:
        raise ValueError(f"spin must be non-negative, got {spin}")
    residues = {h % helicity_alias_modulus(axis_order) for h in range(-spin, spin + 1)}
    return len(residues) == 2 * spin + 1


def minimal_axis_order(spin: int) -> int:
    """Smallest axis order resolving every helicity of a spin-``s`` field."""
    return 2 * spin + 1


def aliasing_is_insufficient(spin: int = 2, bound: int = 12) -> bool:
    """Does the single-axis argument understate the obstruction?

    The aliasing bound ``n >= 2s + 1`` gives ``n >= 5`` for spin two, which a
    six-fold axis satisfies -- and six-fold axes *are* crystallographic.  So the
    single-axis argument alone does not forbid a periodic graviton.  The full
    three-dimensional statement does, because no crystallographic point group
    has a five-dimensional irrep.  This returns ``True`` when some
    crystallographic group clears the aliasing bound yet still fails to protect
    the multiplet, i.e. when the weaker argument would have missed the no-go.
    """
    threshold = minimal_axis_order(spin)
    for group in finite_rotation_groups(bound):
        if not is_crystallographic(group):
            continue
        if max(group.rotation_orders) < threshold:
            continue
        if not is_irreducible(spin, group):
            return True
    return False
