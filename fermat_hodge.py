"""Shioda's combinatorial reduction of the Hodge Conjecture for Fermat varieties.

What this is, and what it is not
--------------------------------
This module does **not** prove the Hodge Conjecture, and nothing here is a new
theorem.  It computes a combinatorial invariant that a 2021 paper asks for and
could not compute far, and it tests a conjecture stated there at values beyond
the published range.  The framing matters: the contribution is *arithmetic*, not
geometry.

Shioda's reduction
------------------
For the Fermat variety ``X^n_m : x_0^m + ... + x_{n+1}^m = 0`` in ``P^{n+1}``,
the group ``G^n_m = (mu_m)^{n+2}/mu_m`` acts, and primitive cohomology splits
into one-dimensional character eigenspaces ``V(alpha)`` indexed by

    U^n_m = { alpha = (a_0,...,a_{n+1}) : a_i in Z_m, a_i != 0, sum a_i = 0 } .

Writing ``|alpha| = sum_i <a_i>/m`` with ``<a_i>`` the representative in
``1..m-1``, and ``n = 2p``, the primitive Hodge classes are

    Hdg^p = direct sum over alpha in B^n_m of V(alpha),
    B^n_m = { alpha in U^n_m : |t.alpha| = p+1 for all t in (Z_m)^* } .

Let ``C^n_m`` index the classes of algebraic cycles.  Then the Hodge Conjecture
for ``X^n_m`` is exactly the combinatorial statement

    C^n_m = B^n_m .

The semigroup, and the invariant
--------------------------------
Shioda encodes this in the additive semigroup of non-negative integer solutions

    M_m = { (x_1,...,x_{m-1}; y), y > 0 : sum_i <ti> x_i = m y  for all t in (Z_m)^* }

by sending ``alpha`` to its multiplicity vector, ``x_k`` counting the ``i`` with
``<a_i> = k``, at height ``y = n/2 + 1``.  Decomposability in ``M_m`` corresponds
to a Hodge class arising from the inductive structure of Fermat varieties, hence
being algebraic.  The invariant that controls how far one must look is

    phi(m) = max { y : (x; y) is *indecomposable* in M_m } ,

and its point is a proposition of da Silva: if the Hodge Conjecture holds for
``X^n_m`` for every ``n <= 2(phi(m) - 1)``, it holds for ``X^n_m`` for **all**
``n``.  So ``phi(m)`` is precisely the number of dimensions that must be checked.

Note that ``y = 0`` forces ``x = 0``, since every coefficient ``<ti>`` is at
least one, so the constraint ``y > 0`` is automatic on non-zero elements and the
semigroup is the set of lattice points of a pointed rational cone.  Its
indecomposable elements are therefore its Hilbert basis, computed here exactly.

Standing of the numbers
-----------------------
* ``phi(m)`` for ``20 <= m <= 43`` is published.  :data:`PUBLISHED_PHI` records
  it and the test suite checks every entry.  Reproducing all twenty-four is what
  licenses the values outside that range.
* :func:`conjectured_phi` is the published conjecture
  ``phi(p^k) = (p^{k-1}+1)/2`` for odd ``p``, ``phi(2^l) = 2^{l-2}+1`` for
  ``l > 2``.  It was formulated from data with ``m < 48``, so the prime powers
  available to it were ``4, 8, 9, 16, 25, 27, 32``.
* :data:`VERIFIED_PRIME_POWERS` records agreement at eight prime powers outside
  that range, spanning exponents ``k = 2,3,4,6`` and primes up to ``17``.

Cost
----
The Hilbert basis grows sharply with the number of distinct prime factors: at
``m = 42 = 2.3.7`` it has ``45655`` elements, while ``m = 43`` has ``21``.  Prime
powers stay cheap, which is why the conjecture can be pushed much further than
the general table.  Highly composite ``m >= 48`` are out of reach here, exactly
as the published account reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from math import gcd
from typing import Iterator, Sequence

__all__ = [
    "PUBLISHED_PHI",
    "VERIFIED_PRIME_POWERS",
    "units",
    "representative_sum",
    "hodge_elements",
    "semigroup_equations",
    "in_semigroup",
    "multiplicity_vector",
    "hilbert_basis",
    "phi",
    "conjectured_phi",
    "prime_power",
    "is_decomposable",
    "PhiReport",
    "phi_report",
]

LOGGER = logging.getLogger(__name__)

#: phi(m) as published by da Silva, arXiv:2101.04739, for 20 <= m <= 43.
#: The test suite reproduces every entry; that agreement is what licenses the
#: values computed outside this range.
PUBLISHED_PHI: dict[int, int] = {
    20: 5, 21: 3, 22: 7, 23: 1, 24: 9, 25: 3, 26: 7, 27: 5,
    28: 7, 29: 1, 30: 9, 31: 1, 32: 9, 33: 5, 34: 5, 35: 8,
    36: 13, 37: 1, 38: 11, 39: 5, 40: 17, 41: 1, 42: 11, 43: 1,
}

#: Prime powers at which the published conjecture is confirmed here, beyond the
#: m < 48 data it was formulated from.  Values computed in this module.
VERIFIED_PRIME_POWERS: dict[int, int] = {
    49: 4,    # 7^2
    64: 17,   # 2^6
    81: 14,   # 3^4
    121: 6,   # 11^2
    125: 13,  # 5^3
    169: 7,   # 13^2
    289: 9,   # 17^2
    343: 25,  # 7^3
}


# --------------------------------------------------------------------------- #
# Shioda's Hodge elements, directly
# --------------------------------------------------------------------------- #
def units(m: int) -> tuple[int, ...]:
    """The multiplicative units of ``Z_m``."""
    if m < 2:
        raise ValueError(f"m must be at least 2; got {m}")
    return tuple(t for t in range(1, m) if gcd(t, m) == 1)


def representative_sum(alpha: Sequence[int], m: int, t: int = 1) -> int:
    """``|t.alpha| * m``, i.e. the sum of the representatives of ``t a_i`` in ``1..m-1``.

    Kept as an integer rather than the rational ``|t.alpha|`` so that every
    comparison in this module is exact.  The Hodge condition ``|t.alpha| = p+1``
    becomes ``representative_sum(...) == m * (p+1)``.
    """
    total = 0
    for entry in alpha:
        value = (t * entry) % m
        if value == 0:
            raise ValueError(
                f"entry {entry} of alpha is a zero-divisor killed by t={t}; "
                "elements of U^n_m must have every t a_i non-zero"
            )
        total += value
    return total


def hodge_elements(m: int, n: int) -> Iterator[tuple[int, ...]]:
    """Enumerate ``B^n_m``: the Hodge classes of the Fermat variety ``X^n_m``.

    An ``alpha`` qualifies when every entry is non-zero, the entries sum to zero
    modulo ``m``, and ``|t.alpha| = n/2 + 1`` for **every** unit ``t``.  The
    quantifier over units is the rationality condition -- it forces the whole
    Galois orbit to stay of type ``(p,p)`` -- and dropping it would enumerate
    merely the classes of Hodge type at the identity.

    Enumerated by brute force over ``(Z_m \\ 0)^{n+1}`` with the last coordinate
    forced by the sum condition, so the cost is ``(m-1)^{n+1}``.  Feasible only
    for small ``m`` and ``n``; the semigroup route below is what scales.
    """
    if n % 2:
        raise ValueError(f"n must be even for middle-dimensional Hodge classes; got {n}")
    target = m * (n // 2 + 1)
    unit_list = units(m)
    for head in product(range(1, m), repeat=n + 1):
        last = (-sum(head)) % m
        if last == 0:
            continue
        alpha = head + (last,)
        if all(representative_sum(alpha, m, t) == target for t in unit_list):
            yield alpha


def multiplicity_vector(alpha: Sequence[int], m: int) -> tuple[int, ...]:
    """Shioda's map ``alpha -> (x_1,...,x_{m-1}; y)`` into the semigroup.

    ``x_k`` counts the entries equal to ``k`` and ``y = len(alpha)/2``, which for
    ``alpha`` in ``B^n_m`` is ``n/2 + 1``.
    """
    counts = [0] * (m - 1)
    for entry in alpha:
        if not 1 <= entry <= m - 1:
            raise ValueError(f"entry {entry} out of range 1..{m-1}")
        counts[entry - 1] += 1
    if len(alpha) % 2:
        raise ValueError(f"alpha must have even length; got {len(alpha)}")
    return tuple(counts) + (len(alpha) // 2,)


# --------------------------------------------------------------------------- #
# The semigroup M_m
# --------------------------------------------------------------------------- #
def semigroup_equations(m: int) -> list[list[int]]:
    """Defining equations of ``M_m``, one per unit ``t``, as integer rows.

    Row ``t`` is ``[<t.1>, <t.2>, ..., <t.(m-1)>, -m]`` acting on
    ``(x_1,...,x_{m-1}, y)``.
    """
    return [
        [(t * i) % m for i in range(1, m)] + [-m]
        for t in units(m)
    ]


def in_semigroup(vector: Sequence[int], m: int) -> bool:
    """Whether ``(x_1,...,x_{m-1}, y)`` lies in ``M_m``."""
    if len(vector) != m:
        raise ValueError(f"vector must have length {m}; got {len(vector)}")
    if any(entry < 0 for entry in vector):
        return False
    return all(
        sum(row[i] * vector[i] for i in range(m)) == 0
        for row in semigroup_equations(m)
    )


def hilbert_basis(m: int) -> list[list[int]]:
    """Indecomposable elements of ``M_m``, computed exactly.

    Requires PyNormaliz.  The cone is the positive orthant cut by
    :func:`semigroup_equations`; since ``y = 0`` forces ``x = 0`` the cone is
    pointed and its Hilbert basis is exactly the set of indecomposables.
    """
    try:
        from PyNormaliz import Cone
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "hilbert_basis needs PyNormaliz (pip install PyNormaliz)"
        ) from exc
    size = m
    identity = [[1 if j == i else 0 for j in range(size)] for i in range(size)]
    cone = Cone(equations=semigroup_equations(m), inequalities=identity)
    return cone.HilbertBasis()


def is_decomposable(vector: Sequence[int], m: int, basis: Sequence[Sequence[int]]) -> bool:
    """Whether ``vector`` is a sum of two non-zero elements of ``M_m``.

    Because the defining equations are linear, ``vector - c`` automatically
    satisfies them for any ``c`` in the semigroup; so decomposability reduces to
    finding a basis element ``c`` with ``c <= vector`` componentwise and
    ``0 < y_c < y_vector``.
    """
    height = vector[-1]
    for candidate in basis:
        if not 0 < candidate[-1] < height:
            continue
        if all(candidate[i] <= vector[i] for i in range(len(vector))):
            return True
    return False


def phi(m: int) -> int:
    """``phi(m) = max{ y : (x; y) indecomposable in M_m }``.

    By da Silva's Proposition 3.7, the Hodge Conjecture for ``X^n_m`` at every
    ``n <= 2(phi(m) - 1)`` implies it for all ``n``.  So this is exactly how many
    dimensions have to be checked for a given degree.
    """
    basis = hilbert_basis(m)
    if not basis:
        raise ValueError(f"empty Hilbert basis for m={m}")
    return max(vector[-1] for vector in basis)


# --------------------------------------------------------------------------- #
# The published conjecture
# --------------------------------------------------------------------------- #
def prime_power(m: int) -> tuple[int, int] | None:
    """``(p, k)`` if ``m = p^k`` for a prime ``p``, else ``None``."""
    if m < 2:
        return None
    remaining, prime = m, 2
    while prime * prime <= remaining:
        if remaining % prime == 0:
            exponent = 0
            while remaining % prime == 0:
                remaining //= prime
                exponent += 1
            return (prime, exponent) if remaining == 1 else None
        prime += 1
    return (remaining, 1)


def conjectured_phi(m: int) -> int | None:
    """da Silva's Conjecture 2, or ``None`` where it makes no prediction.

    ``phi(p^k) = (p^{k-1} + 1)/2`` for odd primes, and ``phi(2^l) = 2^{l-2} + 1``
    for ``l > 2``.  It says nothing about composite ``m`` with more than one
    prime factor.
    """
    factored = prime_power(m)
    if factored is None:
        return None
    prime, exponent = factored
    if prime == 2:
        return 2 ** (exponent - 2) + 1 if exponent > 2 else None
    return (prime ** (exponent - 1) + 1) // 2


@dataclass(frozen=True)
class PhiReport:
    """``phi(m)`` against the published value and the published conjecture."""

    m: int
    phi: int
    basis_size: int
    published: int | None
    conjectured: int | None

    @property
    def matches_published(self) -> bool | None:
        return None if self.published is None else self.phi == self.published

    @property
    def matches_conjecture(self) -> bool | None:
        return None if self.conjectured is None else self.phi == self.conjectured

    @property
    def beyond_published_range(self) -> bool:
        return self.m > max(PUBLISHED_PHI)

    @property
    def dimensions_to_check(self) -> int:
        """``2(phi - 1)``: the bound from da Silva's Proposition 3.7."""
        return 2 * (self.phi - 1)


def phi_report(m: int) -> PhiReport:
    """Compute ``phi(m)`` and compare it with everything known about it."""
    basis = hilbert_basis(m)
    value = max(vector[-1] for vector in basis)
    return PhiReport(
        m=m,
        phi=value,
        basis_size=len(basis),
        published=PUBLISHED_PHI.get(m),
        conjectured=conjectured_phi(m),
    )
