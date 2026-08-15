"""Curvature kills supersymmetry but not the chiral pairing.

A companion to `dirac`, which showed that for the *real* simplicial Dirac
operator ``D = d + delta`` the spectrum carries exactly the information the
Hodge Laplacians carry and no more.  The standard proposal for escaping that is
to put a ``U(1)`` connection on the complex -- a magnetic Dirac operator -- on
the stated grounds that ``D^2 != Delta`` there and that this *breaks the
spectral symmetry*.

The grounds are wrong, and the conclusion is right for a different reason.

The four statements
-------------------

**One.  Chiral symmetry survives any connection.**  ``Gamma D + D Gamma = 0``
holds for every connection, flat or not, exactly and structurally.  The proof
uses only that ``d`` raises degree by one and ``delta`` lowers it by one; it
never touches ``d^2 = 0``.  So the non-zero spectrum is ``+-`` paired no matter
how much curvature is present.  Magnetic phases do not break chirality.

**Two.  What curvature breaks is ``D^2 = Delta``.**  For a connection ``sigma``,

    (d_1 d_0 c)([u,v,w]) = (sigma_uv sigma_vw - sigma_uw) c(w)

so ``d^2 = 0`` exactly when every triangle has trivial holonomy.  With curvature
present ``D^2`` acquires degree-``+-2`` off-diagonal blocks, and their size is
exactly the holonomy defect: `curvature_norm` and `dirac_square_defect` return
the same number.  Supersymmetry dies; the grading does not.

**Three.  There is a chiral asymmetry, and it is the index.**  The ``+-``
pairing holds on the non-zero spectrum but *not* on the kernel: the harmonic
spaces of even and odd degree have dimensions differing by

    index = sum_k (-1)^k beta_k = Euler characteristic

which is non-zero for a disk (1), a two-sphere (2), a path (1).  This asymmetry
is topological, it is present at zero flux, and it is exactly what persistent
homology already reports.  The one genuine chirality asymmetry in the operator
is not geometric and is not new information.

**Four.  So the magnetic Dirac operator does carry more than its Laplacians --
via curvature, not via broken chirality.**  Because ``D^2`` is no longer block
diagonal, ``spec(D_sigma)`` is not reconstructible from the twisted Laplacian
spectra, unlike the flat case proved in `dirac`.  The hope behind the magnetic
proposal is correct; its stated mechanism is not.

Status
------

A literature check (run by the repository owner; arxiv.org is unreachable from
this environment) reports that **no source states either the true or the false
version** of statement one for a non-flat connection.  The pieces are present --
Calmon, Schaub and Bianconi (arXiv:2301.10137) prove the ``+-`` pairing from the
block-off-diagonal structure alone, never invoking ``d^2 = 0``, so their
argument extends verbatim; Egidi, Gittins, Habib and Peyerimhoff
(arXiv:2211.08019) study the continuum ``d_alpha = d + i alpha wedge`` with
``d_alpha^2 = i (d alpha) wedge != 0`` -- but nobody has put them together.
Treat statements one to four as a computed assembly of known pieces whose
conjunction appears unstated, not as new mathematics.

Two cautions from the same check, recorded because they bound what may be
claimed later:

  * the diamagnetic inequality **fails** for magnetic Hodge Laplacians in
    degree above zero (Egidi et al., Cor. 4.6), so the function-case intuition
    that flux only raises the spectral gap does not survive;
  * no bottleneck- or Wasserstein-stability theorem exists for raw
    eigenvalue-valued persistence descriptors, and the obstruction is
    structural -- eigenvalues are not functorial under interleaving and the
    operators change dimension along a filtration.  At *fixed* combinatorics
    Weyl gives 1-Lipschitz dependence on the connection for free, which is what
    `weyl_bound_holds` checks and all that is claimed here.

Novelty
-------

None of the ingredients.  The assembly in statements one through three appears
to be unstated, per the check above; statement four follows immediately from
two.  Nothing here is offered as new mathematics.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

__all__ = [
    "Connection",
    "flat_connection",
    "triangle_holonomy",
    "is_flat",
    "twisted_boundary_zero",
    "twisted_boundary_one",
    "curvature_matrix",
    "curvature_norm",
    "magnetic_dirac",
    "grading_operator",
    "chiral_anticommutator_norm",
    "dirac_square_defect",
    "spectrum",
    "spectrum_asymmetry",
    "kernel_parity_split",
    "index_from_betti",
    "supersymmetry_survives",
    "chirality_survives",
    "weyl_bound_holds",
    "TOLERANCE",
]

#: Numerical tolerance.  The anticommutator is structurally zero and comes back
#: at machine epsilon; everything else is compared against this.
TOLERANCE: float = 1e-10


# ---------------------------------------------------------------------------
# connections on the filled triangle
# ---------------------------------------------------------------------------
#
# One 2-simplex is enough to separate the four statements, because a single
# triangle is the smallest place curvature can live: holonomy is defined on
# 2-cells, so nothing below dimension two can distinguish flat from curved.


#: Edges of the filled triangle, in the order used by every matrix here.
EDGES: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class Connection:
    """A ``U(1)`` connection on the filled triangle, one phase per edge.

    Attributes
    ----------
    phases:
        Angles ``theta_uv`` for the edges ``(0,1), (0,2), (1,2)`` in that order.
        The edge carries ``exp(i theta)`` in the ``u -> v`` direction and its
        conjugate in reverse, which is what makes the operators Hermitian.
    """

    phases: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.phases) != 3:
            raise ValueError(f"need three edge phases, got {len(self.phases)}")
        for phase in self.phases:
            if not isinstance(phase, (int, float)):
                raise TypeError(f"phase {phase!r} is not a real number")

    def weight(self, edge: tuple[int, int]) -> complex:
        """``exp(i theta)`` on ``edge``, conjugated if traversed backwards."""
        if edge in EDGES:
            return cmath.exp(1j * self.phases[EDGES.index(edge)])
        reverse = (edge[1], edge[0])
        if reverse in EDGES:
            return cmath.exp(-1j * self.phases[EDGES.index(reverse)])
        raise ValueError(f"{edge} is not an edge of the triangle")


def flat_connection(base: float = 0.3, second: float = 0.6) -> Connection:
    """A connection with trivial holonomy, built so ``theta_02 = theta_01 + theta_12``."""
    return Connection(phases=(base, base + second, second))


def triangle_holonomy(connection: Connection) -> complex:
    """``sigma_01 sigma_12 sigma_20``: the parallel transport around the triangle."""
    return (
        connection.weight((0, 1))
        * connection.weight((1, 2))
        * connection.weight((2, 0))
    )


def is_flat(connection: Connection, tolerance: float = TOLERANCE) -> bool:
    """Is the holonomy trivial? Equivalently, is ``d^2 = 0``?"""
    return abs(triangle_holonomy(connection) - 1) <= tolerance


# ---------------------------------------------------------------------------
# twisted operators
# ---------------------------------------------------------------------------


def twisted_boundary_zero(connection: Connection) -> np.ndarray:
    """``d_0`` from vertices to edges: ``(d c)([u,v]) = sigma_uv c(v) - c(u)``."""
    matrix = np.zeros((3, 3), dtype=complex)
    for row, (tail, head) in enumerate(EDGES):
        matrix[row, head] += connection.weight((tail, head))
        matrix[row, tail] -= 1
    return matrix


def twisted_boundary_one(connection: Connection) -> np.ndarray:
    """``d_1`` from edges to the triangle.

    ``(d f)([u,v,w]) = sigma_uv f([v,w]) - f([u,w]) + f([u,v])``.
    """
    matrix = np.zeros((1, 3), dtype=complex)
    matrix[0, EDGES.index((1, 2))] += connection.weight((0, 1))
    matrix[0, EDGES.index((0, 2))] -= 1
    matrix[0, EDGES.index((0, 1))] += 1
    return matrix


def curvature_matrix(connection: Connection) -> np.ndarray:
    """``d_1 d_0``, which is zero exactly when the connection is flat.

    Working the composition out by hand gives
    ``(d_1 d_0 c)([u,v,w]) = (sigma_uv sigma_vw - sigma_uw) c(w)``, so the single
    non-zero entry is the holonomy defect.  That identity is the content of
    statement two and `curvature_norm` measures it.
    """
    return twisted_boundary_one(connection) @ twisted_boundary_zero(connection)


def curvature_norm(connection: Connection) -> float:
    """Largest entry of ``d^2``. Zero iff flat."""
    return float(np.abs(curvature_matrix(connection)).max())


def magnetic_dirac(connection: Connection) -> np.ndarray:
    """``D = d + delta`` on vertices, edges and the triangle at once.

    Hermitian by construction: the ``delta`` blocks are the conjugate transposes
    of the ``d`` blocks, which is where the connection's unit modulus is used.
    """
    lower = twisted_boundary_zero(connection)
    upper = twisted_boundary_one(connection)
    sizes = (3, 3, 1)
    offsets = (0, 3, 6, 7)
    total = sum(sizes)
    matrix = np.zeros((total, total), dtype=complex)
    matrix[offsets[1] : offsets[2], offsets[0] : offsets[1]] = lower
    matrix[offsets[0] : offsets[1], offsets[1] : offsets[2]] = lower.conj().T
    matrix[offsets[2] : offsets[3], offsets[1] : offsets[2]] = upper
    matrix[offsets[1] : offsets[2], offsets[2] : offsets[3]] = upper.conj().T
    return matrix


def grading_operator() -> np.ndarray:
    """``Gamma``: ``+1`` on even degrees (vertices, triangle), ``-1`` on edges."""
    return np.diag([1.0] * 3 + [-1.0] * 3 + [1.0]).astype(complex)


# ---------------------------------------------------------------------------
# the four statements
# ---------------------------------------------------------------------------


def chiral_anticommutator_norm(connection: Connection) -> float:
    """``|Gamma D + D Gamma|``. Statement one: zero for every connection.

    Structurally zero, not approximately: ``Gamma`` negates exactly the blocks
    ``D`` occupies, so the sum cancels entry by entry regardless of what the
    entries are.  Curvature cannot reach it.
    """
    dirac = magnetic_dirac(connection)
    grading = grading_operator()
    return float(np.abs(grading @ dirac + dirac @ grading).max())


def dirac_square_defect(connection: Connection) -> float:
    """Largest off-block entry of ``D^2``. Statement two: equals the curvature.

    ``D^2`` is block diagonal -- a direct sum of Hodge Laplacians -- exactly when
    the connection is flat.  With curvature it acquires degree-``+-2`` blocks,
    and this returns their size.
    """
    squared = magnetic_dirac(connection) @ magnetic_dirac(connection)
    mask = np.ones_like(squared, dtype=bool)
    for start, stop in ((0, 3), (3, 6), (6, 7)):
        mask[start:stop, start:stop] = False
    return float(np.abs(squared[mask]).max()) if mask.any() else 0.0


def spectrum(connection: Connection) -> np.ndarray:
    """Eigenvalues of ``D``, sorted. Real, because ``D`` is Hermitian."""
    return np.sort(np.linalg.eigvalsh(magnetic_dirac(connection)))


def spectrum_asymmetry(connection: Connection) -> float:
    """``max_i |lambda_i + lambda_{n-i}|``. Statement one, measured on the spectrum."""
    values = spectrum(connection)
    return float(np.abs(values + values[::-1]).max())


def chirality_survives(connection: Connection, tolerance: float = TOLERANCE) -> bool:
    """Statement one: is the spectrum still ``+-`` paired under this connection?"""
    return (
        chiral_anticommutator_norm(connection) <= tolerance
        and spectrum_asymmetry(connection) <= tolerance
    )


def supersymmetry_survives(
    connection: Connection, tolerance: float = TOLERANCE
) -> bool:
    """Statement two: is ``D^2`` still a direct sum of Laplacians?

    True exactly for flat connections, so this and `chirality_survives` come
    apart the moment curvature is switched on -- which is the whole point.
    """
    return dirac_square_defect(connection) <= tolerance


def kernel_parity_split(betti: Sequence[int]) -> tuple[int, int]:
    """Harmonic dimensions in even and odd degree.

    Statement three.  These are the two chirality sectors of ``ker D``, and they
    do not match: their difference is the Euler characteristic.
    """
    if not betti:
        raise ValueError("need at least one Betti number")
    if any(number < 0 for number in betti):
        raise ValueError(f"Betti numbers must be non-negative, got {tuple(betti)}")
    even = sum(number for degree, number in enumerate(betti) if degree % 2 == 0)
    odd = sum(number for degree, number in enumerate(betti) if degree % 2 == 1)
    return even, odd


def index_from_betti(betti: Sequence[int]) -> int:
    """``sum_k (-1)^k beta_k``: the index, and the Euler characteristic.

    The one genuine chiral asymmetry of the Dirac operator.  It lives entirely
    in the kernel, it is non-zero for a disk, a sphere and a path, and it is
    present at zero flux -- so it is topological rather than geometric, and it
    is exactly what persistent homology already reports.
    """
    even, odd = kernel_parity_split(betti)
    return even - odd


def weyl_bound_holds(
    first: Connection, second: Connection, tolerance: float = TOLERANCE
) -> bool:
    """``|lambda_k(D) - lambda_k(D')| <= ||D - D'||`` at fixed combinatorics.

    All that is claimed about stability here, and it is free: Weyl's inequality
    for Hermitian matrices.  It says nothing about a *filtration*, where the
    operators change dimension and the real difficulty lies -- see the module
    docstring.
    """
    left = spectrum(first)
    right = spectrum(second)
    gap = float(
        np.linalg.norm(magnetic_dirac(first) - magnetic_dirac(second), ord=2)
    )
    return bool(np.all(np.abs(left - right) <= gap + tolerance))
