"""Numerical conformal bootstrap: exclusion functionals in two dimensions.

What this does, and what it deliberately does not
-------------------------------------------------
The bootstrap excludes a putative CFT by exhibiting a linear functional.  For
four identical scalars of dimension ``Delta_phi``, crossing symmetry says

    sum_O  p_O  F_{Delta_O, l_O}(z, zbar)  =  0,        p_O >= 0,

with the identity contributing ``p = 1``.  If a functional ``alpha`` can be found
with ``alpha[F_identity] = 1`` and ``alpha[F_{Delta,l}] >= 0`` for every operator
the assumed spectrum permits, then applying it to the sum rule gives
``0 = 1 + (something non-negative) > 0``.  The assumption is contradictory, and
the theory is excluded.  Only that direction is a proof: failing to find a
functional is not evidence that a CFT exists.

Scope.  This module works in ``d = 2``, where conformal blocks have a closed form
that factorises completely at the crossing-symmetric point.  It computes a single
correlator's bound on the leading scalar dimension.  It does **not** attempt the
3d Ising island: that needs mixed correlators, a semidefinite program over
polynomial positivity in ``Delta``, and an arbitrary-precision SDP solver.  A
linear program over sampled dimensions -- which is what this is -- gives a bound
that is rigorous only against the operators actually sampled.

That gap is not hypothetical.  A single-shot linear program here returns
functionals that dip to around ``-1e-3`` between their own sample points, which
makes them worthless as proofs; the failure is silent unless you look for it.  So
:func:`find_functional` audits every solution on a finer grid it never trained
on, feeds the violations back as new constraints, and re-solves until nothing on
the audit grid is negative.  What it reports is therefore a functional verified
non-negative on a dense grid -- stronger than the naive version by a wide margin,
and still not the continuum statement that a semidefinite program would give.

Why ``d = 2`` is the right calibration target
---------------------------------------------
The bound has a kink at the 2d Ising model, whose dimensions are known exactly:
``Delta_sigma = 1/8`` and ``Delta_epsilon = 1``.  So the pipeline can be checked
against exact algebraic numbers rather than against itself, which is the same
role Catalan numbers play for the amplituhedron module.

Blocks
------
In ``d = 2`` the block for exchanged dimension ``Delta`` and even spin ``l`` is

    g_{Delta,l}(z, zbar) = k_{Delta-l}(z) k_{Delta+l}(zbar) + (z <-> zbar),

halved when ``l = 0``, where ``k_b(x) = x^(b/2) 2F1(b/2, b/2; b; x)`` is the
SL(2) block.  Everything is then a product of one-variable functions, so
derivatives at ``z = zbar = 1/2`` never need multivariate numerical
differentiation -- they come from one-dimensional Taylor series with a bounded
tail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import mpmath as mp
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

__all__ = [
    "ISING_2D_SIGMA",
    "ISING_2D_EPSILON",
    "DerivativeBasis",
    "FunctionalResult",
    "sl2_block_derivatives",
    "sl2_block_derivatives_fast",
    "dressed_derivatives",
    "crossing_derivatives",
    "identity_crossing_derivatives",
    "unitarity_bound",
    "spectrum_samples",
    "find_functional",
    "functional_margin",
    "scalar_gap_bound",
    "casimir_residual",
]

LOGGER = logging.getLogger(__name__)

#: Factorials up to a generous derivative order, for Taylor normalisation.
_FACTORIAL = [float(mp.factorial(i)) for i in range(64)]

#: Exact dimensions of the 2d Ising model, the calibration point.
ISING_2D_SIGMA = mp.mpf(1) / 8
ISING_2D_EPSILON = mp.mpf(1)


# --------------------------------------------------------------------------- #
# SL(2) blocks and their derivatives
# --------------------------------------------------------------------------- #
def sl2_block_derivatives(
    beta: mp.mpf,
    order: int,
    *,
    point: mp.mpf | None = None,
    terms: int = 120,
) -> list[mp.mpf]:
    """Derivatives ``d^m/dx^m k_beta(x)`` at ``point``, for ``m = 0 .. order``.

    ``k_beta(x) = x^(beta/2) 2F1(beta/2, beta/2; beta; x)`` is summed term by
    term and differentiated termwise.  At ``x = 1/2`` the series converges
    geometrically, so a fixed truncation with generous working precision is
    accurate to far more digits than the linear program can use; the tail is
    reported by :func:`_series_tail` for the tests to check rather than assume.

    ``beta = 0`` is the limiting case ``k_0 = 1``, which is what a conserved
    current at the unitarity bound needs.
    """
    if order < 0:
        raise ValueError(f"order must be non-negative; got {order}")
    if terms < 1:
        raise ValueError(f"terms must be positive; got {terms}")
    x = mp.mpf(1) / 2 if point is None else mp.mpf(point)
    beta = mp.mpf(beta)

    if beta == 0:
        return [mp.mpf(1)] + [mp.mpf(0)] * order

    half = beta / 2
    derivatives = [mp.mpf(0)] * (order + 1)
    coefficient = mp.mpf(1)
    for n in range(terms):
        if n > 0:
            # c_n / c_{n-1} = ((half + n - 1)^2) / ((beta + n - 1) n)
            coefficient *= (half + n - 1) ** 2 / ((beta + n - 1) * n)
        exponent = half + n
        for m in range(order + 1):
            falling = mp.mpf(1)
            for j in range(m):
                falling *= exponent - j
            derivatives[m] += coefficient * falling * x ** (exponent - m)
    return derivatives


def sl2_block_derivatives_fast(
    beta: float, order: int, *, point: float = 0.5, terms: int = 120
) -> NDArray[np.float64]:
    """Vectorised float64 version of :func:`sl2_block_derivatives`.

    The linear program consumes float64, so summing the series at 30 decimal
    digits is wasted work -- and it is not cheap wasted work, since the mpmath
    loop dominates the whole bootstrap.  The two implementations are checked
    against each other in the tests rather than assumed to agree.

    Precision is safe here for a specific reason: the crossing function's
    antisymmetry has already been resolved analytically, so no large
    cancellation survives into these sums.  At ``x = 1/2`` successive terms fall
    off geometrically, and past roughly sixty of them they contribute below
    float64 resolution.
    """
    if order < 0:
        raise ValueError(f"order must be non-negative; got {order}")
    if terms < 1:
        raise ValueError(f"terms must be positive; got {terms}")
    if beta == 0:
        result = np.zeros(order + 1)
        result[0] = 1.0
        return result

    half = beta / 2.0
    index = np.arange(terms, dtype=np.float64)
    ratios = (half + index[:-1]) ** 2 / ((beta + index[:-1]) * (index[:-1] + 1.0))
    coefficients = np.concatenate(([1.0], np.cumprod(ratios)))
    exponents = half + index

    out = np.empty(order + 1, dtype=np.float64)
    falling = np.ones(terms, dtype=np.float64)
    for m in range(order + 1):
        if m:
            falling = falling * (exponents - (m - 1))
        out[m] = float(np.sum(coefficients * falling * point ** (exponents - m)))
    return out


def _series_tail(beta: mp.mpf, terms: int, point: mp.mpf) -> mp.mpf:
    """Magnitude of the first omitted term, as a truncation diagnostic."""
    beta = mp.mpf(beta)
    if beta == 0:
        return mp.mpf(0)
    half = beta / 2
    coefficient = mp.mpf(1)
    for n in range(1, terms + 1):
        coefficient *= (half + n - 1) ** 2 / ((beta + n - 1) * n)
    return abs(coefficient * point ** (half + terms))


def _prefactor_derivatives(delta_phi: float, order: int) -> NDArray[np.float64]:
    """Derivatives of ``(1 - z)^Delta_phi`` at ``z = 1/2``."""
    out = np.empty(order + 1, dtype=np.float64)
    falling = 1.0
    for j in range(order + 1):
        if j:
            falling *= delta_phi - (j - 1)
        out[j] = (-1) ** j * falling * 0.5 ** (delta_phi - j)
    return out


def dressed_derivatives(
    beta: float, delta_phi: float, order: int, *, terms: int = 120
) -> NDArray[np.float64]:
    """Derivatives of ``P_beta(z) = (1 - z)^Delta_phi k_beta(z)`` at ``z = 1/2``.

    This one function carries the whole crossing calculation.  Because both the
    prefactor and the block factorise between ``z`` and ``zbar``, the crossing
    function's mixed derivatives are products of these, and the reflection
    ``z -> 1 - z`` acts on them by a sign -- so no two-variable differentiation
    is ever needed.
    """
    block = sl2_block_derivatives_fast(float(beta), order, terms=terms)
    prefactor = _prefactor_derivatives(float(delta_phi), order)

    dressed = np.empty(order + 1, dtype=np.float64)
    for m in range(order + 1):
        total = 0.0
        for j in range(m + 1):
            total += float(mp.binomial(m, j)) * prefactor[j] * block[m - j]
        dressed[m] = total
    return dressed


# --------------------------------------------------------------------------- #
# The derivative basis of functionals
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DerivativeBasis:
    """Which mixed derivatives at ``z = zbar = 1/2`` a functional may use.

    The crossing function is antisymmetric under ``(z, zbar) -> (1-z, 1-zbar)``,
    so every even-order derivative vanishes identically and only ``m + n`` odd
    carries information.  It is also symmetric in ``z <-> zbar``, so ``m < n``
    covers everything without duplication.
    """

    max_order: int

    @property
    def components(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (m, n)
            for total in range(1, self.max_order + 1, 2)
            for m in range((total + 1) // 2)
            for n in [total - m]
            if m < n
        )

    @property
    def size(self) -> int:
        return len(self.components)


@lru_cache(maxsize=1 << 18)
def _crossing_row(
    delta: float, spin: int, delta_phi: float, max_order: int, terms: int
) -> tuple[float, ...]:
    """Cached crossing vector.

    Bisection re-solves at many gaps, but every spin-``l >= 2`` sample sits at
    its own unitarity bound and is therefore identical across all of them -- and
    the audit grid is regenerated verbatim each round.  Without caching the same
    hypergeometric sums are recomputed dozens of times.
    """
    return tuple(
        _crossing_vector_uncached(delta, spin, delta_phi, max_order, terms)
    )


def crossing_derivatives(
    delta: mp.mpf,
    spin: int,
    delta_phi: mp.mpf,
    basis: DerivativeBasis,
    *,
    terms: int = 120,
) -> NDArray[np.float64]:
    """Taylor components of ``F_{Delta,l}`` at the crossing-symmetric point.

    ``F = v^Delta_phi g(z, zbar) - u^Delta_phi g(1-z, 1-zbar)`` with
    ``u = z zbar`` and ``v = (1-z)(1-zbar)``.  Writing
    ``P_b(z) = (1-z)^Delta_phi k_b(z)`` the first term is
    ``P_h1(z) P_h2(zbar) + P_h2(z) P_h1(zbar)``, and the second is the same with
    ``z -> 1 - z``, which merely multiplies the ``m``-th derivative by
    ``(-1)^m``.  Hence the component is

        (1 - (-1)^(m+n)) [P_h1^(m) P_h2^(n) + P_h2^(m) P_h1^(n)] / (m! n!),

    which is exactly zero for ``m + n`` even, as the antisymmetry demands.
    Dividing by the factorials keeps the components from growing factorially,
    which is what makes the linear program conditioned at all.
    """
    if spin < 0:
        raise ValueError(f"spin must be non-negative; got {spin}")
    if spin % 2:
        raise ValueError(f"only even spins appear for identical scalars; got {spin}")

    return np.array(
        _crossing_row(float(delta), spin, float(delta_phi), basis.max_order, terms),
        dtype=np.float64,
    )


def _crossing_vector_uncached(
    delta: float, spin: int, delta_phi: float, order: int, terms: int
) -> NDArray[np.float64]:
    """The actual computation behind :func:`crossing_derivatives`."""
    basis = DerivativeBasis(max_order=order)
    low = dressed_derivatives(delta - spin, delta_phi, order, terms=terms)
    high = (
        low
        if spin == 0
        else dressed_derivatives(delta + spin, delta_phi, order, terms=terms)
    )
    weight = 1.0 if spin else 0.5  # l = 0 would otherwise double count

    values = []
    for m, n in basis.components:
        combined = low[m] * high[n] + high[m] * low[n]
        values.append(2.0 * weight * combined / (_FACTORIAL[m] * _FACTORIAL[n]))
    return np.array(values, dtype=np.float64)


def identity_crossing_derivatives(
    delta_phi: mp.mpf, basis: DerivativeBasis
) -> NDArray[np.float64]:
    """Taylor components of the identity's crossing contribution, ``g = 1``."""
    prefactor = _prefactor_derivatives(float(delta_phi), basis.max_order)
    values = [
        2.0 * prefactor[m] * prefactor[n] / (_FACTORIAL[m] * _FACTORIAL[n])
        for m, n in basis.components
    ]
    return np.array(values, dtype=np.float64)


# --------------------------------------------------------------------------- #
# The assumed spectrum
# --------------------------------------------------------------------------- #
def unitarity_bound(spin: int, dimension: int = 2) -> mp.mpf:
    """Lowest dimension a unitary primary of the given spin may have."""
    if spin < 0:
        raise ValueError(f"spin must be non-negative; got {spin}")
    if dimension < 2:
        raise ValueError(f"dimension must be at least 2; got {dimension}")
    if spin == 0:
        return mp.mpf(dimension - 2) / 2
    return mp.mpf(spin + dimension - 2)


def spectrum_samples(
    scalar_gap: mp.mpf,
    *,
    max_spin: int = 20,
    n_samples: int = 60,
    delta_max: mp.mpf | float = 40.0,
    dimension: int = 2,
) -> list[tuple[mp.mpf, int]]:
    """Sampled ``(Delta, spin)`` points the functional must be non-negative on.

    Sampling is denser near each spin's unitarity bound, where the crossing
    vectors vary fastest and where a functional is most likely to dip negative
    between samples.  The grid is geometric in ``Delta - Delta_min`` for exactly
    that reason.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2; got {n_samples}")
    scalar_gap = mp.mpf(scalar_gap)
    delta_max = mp.mpf(delta_max)

    samples: list[tuple[mp.mpf, int]] = []
    for spin in range(0, max_spin + 1, 2):
        floor = scalar_gap if spin == 0 else unitarity_bound(spin, dimension)
        if floor >= delta_max:
            continue
        span = delta_max - floor
        for i in range(n_samples):
            fraction = mp.mpf(i) / (n_samples - 1)
            samples.append((floor + span * fraction**3, spin))
    return samples


# --------------------------------------------------------------------------- #
# The exclusion functional
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FunctionalResult:
    """Outcome of searching for an exclusion functional."""

    found: bool
    coefficients: NDArray[np.float64] | None
    basis: DerivativeBasis
    scalar_gap: float
    delta_phi: float
    n_constraints: int
    status: str
    margin: float | None = None
    lp_feasible: bool = False
    converged: bool = True
    box_active: bool = False

    @property
    def excludes(self) -> bool:
        """Whether the assumed spectrum is ruled out.

        Only meaningful in one direction.  ``True`` is a proof against the
        sampled spectrum; ``False`` means no functional was found in this basis,
        which is not evidence that the theory exists.
        """
        return self.found


def find_functional(
    delta_phi: mp.mpf | float,
    scalar_gap: mp.mpf | float,
    basis: DerivativeBasis,
    *,
    max_spin: int = 20,
    n_samples: int = 60,
    delta_max: float = 40.0,
    terms: int = 120,
    audit_samples: int = 400,
    audit_delta_max: float = 60.0,
    refine_rounds: int = 40,
    tolerance: float = 1e-9,
) -> FunctionalResult:
    """Search for a functional excluding a scalar gap, by linear programming.

    Solves the feasibility problem: find ``a`` with ``a . F_identity = 1`` and
    ``a . F_{Delta,l} >= 0`` on every sampled operator.  Each constraint row is
    normalised to unit length, which changes nothing about feasibility -- the
    constraints are homogeneous inequalities -- while keeping the simplex method
    from being defeated by rows whose scales differ by many orders of magnitude.

    A single-shot linear program is not enough, and measurably so: its solution
    satisfies the sampled constraints while dipping slightly negative between
    them, which makes it useless as an exclusion proof.  So the solution is
    re-checked on a finer audit grid it never saw, the worst violators are added
    as constraints, and the program is re-solved until nothing on the audit grid
    is negative.  ``found`` is only ``True`` for a functional that survives that
    check, and ``margin`` records how much room it had to spare.
    """
    delta_phi = mp.mpf(delta_phi)
    scalar_gap = mp.mpf(scalar_gap)

    identity = identity_crossing_derivatives(delta_phi, basis)

    def rows_for(points: Iterable[tuple[mp.mpf, int]]) -> list[NDArray[np.float64]]:
        built = []
        for delta, spin in points:
            vector = crossing_derivatives(delta, spin, delta_phi, basis, terms=terms)
            norm = np.linalg.norm(vector)
            if norm > 0 and np.all(np.isfinite(vector)):
                built.append(vector / norm)
        return built

    rows = rows_for(
        spectrum_samples(
            scalar_gap, max_spin=max_spin, n_samples=n_samples, delta_max=delta_max
        )
    )
    if not rows:
        raise ValueError("no usable spectrum constraints were generated")

    # The audit grid is never used to build the functional, only to catch one
    # that dips negative between the sampled points.  Violations found here are
    # fed back as new constraints -- a cutting-plane loop -- because a functional
    # that is negative anywhere is not an exclusion proof at all, and a plain
    # single-shot linear program produces exactly that.
    audit = list(
        spectrum_samples(
            scalar_gap,
            max_spin=max_spin,
            n_samples=audit_samples,
            delta_max=audit_delta_max,
        )
    )
    audit_vectors = rows_for(audit)

    # Maximise the worst value the functional takes, rather than asking for mere
    # feasibility.  Feasibility-with-a-tolerance is the wrong question at the
    # boundary, which is precisely where the bound lives: the cutting-plane loop
    # stalls around -1e-8 while the constraint set grows without limit, and the
    # verdict then depends on the tolerance rather than on the physics.  The
    # optimal margin, by contrast, varies continuously with the assumed gap, so
    # its *sign* is a well-posed and monotone predicate.
    #
    # Variables are (a, t); the box on `a` keeps the maximisation bounded and is
    # reported through `box_active` if it ever binds.  When it does, only the
    # *sign* of the reported margin is meaningful -- an admissible functional can
    # always be scaled up, so the magnitude just reflects where the box sits.
    size = basis.size
    objective = np.zeros(size + 1)
    objective[-1] = -1.0  # maximise t
    box = 1.0e6

    result = None
    margin = -np.inf
    converged = False
    for _ in range(max(1, refine_rounds)):
        matrix = np.array(rows, dtype=np.float64)
        inequality = np.hstack([-matrix, np.ones((matrix.shape[0], 1))])
        result = linprog(
            c=objective,
            A_ub=inequality,
            b_ub=np.zeros(matrix.shape[0]),
            A_eq=np.hstack([identity, [0.0]]).reshape(1, -1),
            b_eq=np.array([1.0]),
            bounds=[(-box, box)] * size + [(None, None)],
            method="highs",
        )
        if not result.success:
            margin = -np.inf
            break
        coefficients = result.x[:size]
        if float(-result.fun) < 0.0:
            # The best achievable margin is already negative on the constraints
            # the program was given.  Adding more can only lower it, so there is
            # nothing to refine towards -- this gap is simply not excluded.
            margin = float(-result.fun)
            break
        values = np.array([coefficients @ vector for vector in audit_vectors])
        margin = float(values.min())
        if margin >= 0.0:
            converged = True
            break
        worst = np.argsort(values)[: max(1, len(values) // 10)]
        rows.extend(audit_vectors[int(i)] for i in worst)

    assert result is not None
    coefficients = result.x[:size] if result.success else None
    excluded = bool(result.success) and margin > 0.0
    return FunctionalResult(
        found=excluded,
        coefficients=coefficients if excluded else None,
        basis=basis,
        scalar_gap=float(scalar_gap),
        delta_phi=float(delta_phi),
        n_constraints=len(rows),
        status=str(result.message),
        margin=float(margin) if np.isfinite(margin) else None,
        lp_feasible=bool(result.success),
        converged=converged,
        box_active=bool(
            coefficients is not None and np.max(np.abs(coefficients)) >= box * (1 - 1e-9)
        ),
    )


def functional_margin(
    result: FunctionalResult,
    *,
    max_spin: int = 20,
    n_samples: int = 400,
    delta_max: float = 60.0,
    terms: int = 120,
) -> float:
    """Smallest value a found functional takes on an independent finer grid.

    A linear program only enforces its own constraints.  Re-testing the solution
    on a grid it never saw is the cheapest guard against a functional that dips
    negative between samples, which would invalidate the exclusion.  A negative
    return value means the exclusion is not trustworthy.
    """
    if result.coefficients is None:
        raise ValueError("no functional was found, so it has no margin")
    delta_phi = mp.mpf(result.delta_phi)
    samples = spectrum_samples(
        mp.mpf(result.scalar_gap),
        max_spin=max_spin,
        n_samples=n_samples,
        delta_max=mp.mpf(delta_max),
    )
    worst = np.inf
    for delta, spin in samples:
        vector = crossing_derivatives(delta, spin, delta_phi, result.basis, terms=terms)
        norm = np.linalg.norm(vector)
        if norm == 0 or not np.all(np.isfinite(vector)):
            continue
        worst = min(worst, float(result.coefficients @ (vector / norm)))
    return worst


def scalar_gap_bound(
    delta_phi: mp.mpf | float,
    basis: DerivativeBasis,
    *,
    low: float = 0.05,
    high: float = 6.0,
    tolerance: float = 1e-3,
    max_spin: int = 20,
    n_samples: int = 60,
    delta_max: float = 40.0,
    terms: int = 120,
    audit_samples: int = 400,
    audit_delta_max: float = 60.0,
    refine_rounds: int = 40,
) -> float:
    """Upper bound on the leading scalar dimension, by bisection.

    Above the bound a functional exists and the theory is excluded; below it none
    is found.  Bisection brackets the crossover.

    The result is an upper bound **against the sampled spectrum only**.  A true
    bound requires positivity for every real ``Delta`` in a continuum, which is a
    semidefinite rather than a linear condition; :func:`functional_margin` is the
    cheap partial substitute offered here.
    """
    if low >= high:
        raise ValueError(f"need low < high; got low={low}, high={high}")

    def excluded(gap: float) -> bool:
        return find_functional(
            delta_phi,
            gap,
            basis,
            max_spin=max_spin,
            n_samples=n_samples,
            delta_max=delta_max,
            terms=terms,
            audit_samples=audit_samples,
            audit_delta_max=audit_delta_max,
            refine_rounds=refine_rounds,
        ).found

    if excluded(low):
        LOGGER.warning("even the lowest gap %.3f is excluded; bound is below it", low)
        return low
    if not excluded(high):
        LOGGER.warning("no functional even at gap %.3f; bound is above it", high)
        return high

    while high - low > tolerance:
        middle = (low + high) / 2
        if excluded(middle):
            high = middle
        else:
            low = middle
    return (low + high) / 2


# --------------------------------------------------------------------------- #
# An independent referee for the blocks
# --------------------------------------------------------------------------- #
def casimir_residual(
    delta: mp.mpf | float,
    spin: int,
    z: mp.mpf | float,
    zbar: mp.mpf | float,
    *,
    dimension: int = 2,
) -> mp.mpf:
    """How badly a block fails the quadratic Casimir equation, at one point.

    The conformal block is by definition an eigenfunction of the quadratic
    Casimir with eigenvalue ``c2 / 2``, ``c2 = Delta(Delta - d) + l(l + d - 2)``.
    Checking that directly tests the closed form against the representation
    theory it is supposed to encode, rather than against another implementation
    of the same formula.  Returns the residual, which should vanish.
    """
    if dimension != 2:
        raise ValueError(f"only d = 2 is implemented; got {dimension}")
    delta = mp.mpf(delta)

    def block(a: mp.mpf, b: mp.mpf) -> mp.mpf:
        low, high = delta - spin, delta + spin
        value = _sl2(low, a) * _sl2(high, b) + _sl2(high, a) * _sl2(low, b)
        return value / 2 if spin == 0 else value

    def casimir_in(variable: int, a: mp.mpf, b: mp.mpf) -> mp.mpf:
        if variable == 0:
            first = mp.diff(lambda t: block(t, b), a)
            second = mp.diff(lambda t: block(t, b), a, 2)
            return a**2 * (1 - a) * second - a**2 * first
        first = mp.diff(lambda t: block(a, t), b)
        second = mp.diff(lambda t: block(a, t), b, 2)
        return b**2 * (1 - b) * second - b**2 * first

    z, zbar = mp.mpf(z), mp.mpf(zbar)
    applied = casimir_in(0, z, zbar) + casimir_in(1, z, zbar)
    c2 = delta * (delta - dimension) + spin * (spin + dimension - 2)
    return applied - c2 / 2 * block(z, zbar)


def _sl2(beta: mp.mpf, x: mp.mpf) -> mp.mpf:
    """``k_beta(x)`` in closed form, for the Casimir check only."""
    beta = mp.mpf(beta)
    if beta == 0:
        return mp.mpf(1)
    return x ** (beta / 2) * mp.hyp2f1(beta / 2, beta / 2, beta, x)
