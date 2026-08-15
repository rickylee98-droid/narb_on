"""What the persistent Dirac operator can and cannot see.

The brief this answers proposes that the persistent Dirac operator -- the
"quantum square root" ``D = d + delta`` with ``D^2 = Delta`` -- can classify
geometric chirality in data where persistent homology is blind, because ``D`` is
first order and therefore "retains the sign and orientation" that the
second-order Laplacian squares away.

That conjecture is false, and it fails twice, for two independent reasons.  Both
are proved here and both are short.

Who is actually being corrected
-------------------------------

Not the topological deep learning literature.  A literature check (run by the
repository owner; arxiv.org is unreachable from this environment) established
three things, and they matter for reading everything below:

  * The brief's supporting citation, ``arXiv:2208.06456``, does not support the
    chirality claim.  The relevant work is ``arXiv:2301.10137`` and
    ``arXiv:2105.00529``.  The citation was simply wrong.
  * What the literature actually claims is that the persistent Dirac spectrum is
    strictly more informative than persistent *homology* -- which means only
    that it retains the non-harmonic spectrum.  **That claim is true**, and it
    is the one recorded under "What is actually true" below.
  * The chiral-symmetry equivalence proved here is already established, as
    standard Hodge theory.  It is not new.

So the published claim is correct and modest, and the brief inflated it into a
statement about chirality that its own citation does not make.  This module
refutes the inflation, not the literature.  Nothing below should be read as a
criticism of the cited work.

One: the Dirac spectrum carries no more than the Laplacian's
-------------------------------------------------------------

Let ``Gamma`` be the grading operator, ``+1`` on even-degree chains and ``-1`` on
odd.  Both ``d`` and ``delta`` shift degree by one, so both anticommute with
``Gamma``, and therefore

    Gamma D = - D Gamma.

``Gamma`` is a unitary involution, so if ``D psi = lambda psi`` then
``D (Gamma psi) = - lambda (Gamma psi)``.  The spectrum of ``D`` is symmetric
about zero.  Combined with ``D^2 = Delta``, this pins it completely:

    spec(D) = { +- sqrt(mu) : mu in spec(Delta) }, with the signs forced.

So ``spec(D)`` is a deterministic function of the Hodge Laplacian spectra and
vice versa: `dirac_spectrum_from_laplacian` reconstructs one from the other with
no extra input.  **They carry identical information.**  Any two complexes the
Laplacian cannot tell apart, the Dirac operator cannot tell apart either -- and
it inherits every Laplacian blindspot, including cospectrality
(`COSPECTRAL_PAIR`, minimal at six vertices).

The first-order-ness is real, but it buys expressiveness in the *eigenvectors*,
which mix degrees, not in the spectrum.  The brief's claim is about the
spectrum.

Two: nothing built from distances can see chirality
----------------------------------------------------

This one does not even need the Dirac operator.  A reflection is an isometry, so
a chiral point cloud and its mirror image have **identical** pairwise distance
matrices -- not approximately, bitwise.  Every filtration built from those
distances (Vietoris-Rips, Cech) is therefore the identical filtered complex, and
every invariant of it agrees: persistent homology, persistent Laplacian spectra,
persistent Dirac spectra, all of them.

    chirality is not a function of the distance matrix.

`chirality_is_invisible_to_distances` demonstrates it and `signed_volume` shows
what does change: the orientation determinants flip sign, and they are not
recoverable from pairwise distances.  Detecting chirality requires oriented
input -- signed volumes, a chosen embedding, a global frame -- that no
distance-based filtration provides.  This is why persistent homology is blind to
chirality, and the reason has nothing to do with squaring.

What is actually true
---------------------

One part of the brief survives, and it is worth stating because it is both the
defensible version *and* what the literature actually claims:

    the non-zero spectrum carries strictly more than persistent homology.

Homology reads only the kernel -- the Betti numbers -- and discards every
non-zero eigenvalue.  The Laplacian and Dirac spectra keep them.  That gap is
real (`homology_discards`), it is what makes spectral methods worth using, and
it is entirely a statement about Laplacian versus homology.  It says nothing
about Dirac versus Laplacian, where the answer is that they are equivalent.

Novelty
-------

**None.**  This is stated flatly rather than hedged, because the literature
check settled it.  The chiral symmetry of ``d + delta`` and the resulting
supersymmetric pairing of the spectrum is standard Hodge theory and was already
established; reflection-invariance of distance matrices is immediate; Laplacian
cospectrality is classical.  No result in this module is new mathematics, and
the one that looked most like a contribution -- the spectral equivalence in part
one -- is the one explicitly confirmed to be known.

What the module is, then, is a computed verdict on a claim that was never in the
literature to begin with: the two independent refutations separated so neither
is mistaken for the other, the surviving claim stated in the form that is
actually true, and the whole thing checked in exact arithmetic.  That is worth
having as a record.  It is not worth calling a discovery.

Referees
--------

Nothing here is asserted where it can be checked.  ``D^2`` is confirmed to be
exactly the block-diagonal Hodge Laplacian, the anticommutator ``Gamma D +
D Gamma`` is confirmed to be exactly zero, and the dimension of ``ker D`` is
confirmed to equal the sum of the Betti numbers on complexes whose homology is
known independently -- a circle gives 2, a disk 1, a 2-sphere 2.  All in exact
rational arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import sympy as sp

__all__ = [
    "SimplicialComplex",
    "complex_from_maximal_faces",
    "graph_complex",
    "boundary_matrix",
    "hodge_laplacian",
    "dirac_operator",
    "chirality_operator",
    "dirac_squared_residual",
    "chiral_anticommutator_residual",
    "dirac_spectrum",
    "laplacian_spectrum",
    "dirac_spectrum_from_laplacian",
    "spectra_are_equivalent",
    "spectrum_is_symmetric",
    "betti_numbers",
    "kernel_dimension",
    "homology_discards",
    "reflect",
    "distance_matrix",
    "rips_complex",
    "signed_volume",
    "chirality_signature",
    "chirality_is_invisible_to_distances",
    "CHIRAL_POINTS",
    "COSPECTRAL_PAIR",
    "MINIMAL_COSPECTRAL_ORDER",
    "cospectral_pair_is_a_blindspot",
    "MISCITED_REFERENCE",
    "RELEVANT_REFERENCES",
    "LITERATURE_CLAIM",
]


#: The citation the brief gave for the chirality claim.  A literature check found
#: it does not support that claim; it is recorded here so the miscitation stays
#: visible rather than being quietly dropped.
MISCITED_REFERENCE: str = "arXiv:2208.06456"

#: What the brief should have cited, per the same literature check.  These are
#: reported second hand -- arxiv.org is unreachable from this environment, so
#: they were supplied by the repository owner and not verified here.
RELEVANT_REFERENCES: tuple[str, ...] = ("arXiv:2301.10137", "arXiv:2105.00529")

#: What the topological deep learning literature actually claims, which is true
#: and is the statement `homology_discards` measures.
LITERATURE_CLAIM: str = (
    "the persistent Dirac spectrum is strictly more informative than persistent "
    "homology, because it retains the non-harmonic spectrum"
)


# ---------------------------------------------------------------------------
# simplicial complexes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimplicialComplex:
    """An abstract simplicial complex, closed under taking faces.

    Simplices are sorted vertex tuples.  "Abstract" is the operative word: a
    complex records which vertices span which faces and nothing about where they
    sit, which is already most of the reason chirality is invisible.
    """

    simplices: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.simplices:
            raise ValueError("a complex needs at least one simplex")
        present = set(self.simplices)
        for simplex in self.simplices:
            if tuple(sorted(simplex)) != simplex:
                raise ValueError(f"simplex {simplex} is not sorted")
            if len(set(simplex)) != len(simplex):
                raise ValueError(f"simplex {simplex} repeats a vertex")
            for size in range(1, len(simplex)):
                for face in combinations(simplex, size):
                    if face not in present:
                        raise ValueError(
                            f"complex is not closed: {simplex} is present but its "
                            f"face {face} is not"
                        )

    @property
    def top_dimension(self) -> int:
        return max(len(simplex) for simplex in self.simplices) - 1

    def of_dimension(self, degree: int) -> tuple[tuple[int, ...], ...]:
        """Simplices of the given degree, in a fixed order."""
        if degree < 0:
            raise ValueError(f"degree must be non-negative, got {degree}")
        return tuple(
            simplex for simplex in self.simplices if len(simplex) == degree + 1
        )

    @property
    def dimensions(self) -> tuple[int, ...]:
        """Count of simplices in each degree, from zero to the top."""
        return tuple(
            len(self.of_dimension(degree))
            for degree in range(self.top_dimension + 1)
        )


def complex_from_maximal_faces(
    maximal: Iterable[Sequence[int]],
) -> SimplicialComplex:
    """Close a list of maximal faces downward into a complex."""
    collected: set[tuple[int, ...]] = set()
    for face in maximal:
        ordered = tuple(sorted(face))
        if not ordered:
            raise ValueError("a face must have at least one vertex")
        for size in range(1, len(ordered) + 1):
            collected.update(combinations(ordered, size))
    ordered_simplices = tuple(
        sorted(collected, key=lambda simplex: (len(simplex), simplex))
    )
    return SimplicialComplex(simplices=ordered_simplices)


def graph_complex(vertices: int, edges: Iterable[Sequence[int]]) -> SimplicialComplex:
    """A graph as a one-dimensional complex, isolated vertices included."""
    if vertices < 1:
        raise ValueError(f"need at least one vertex, got {vertices}")
    faces: list[Sequence[int]] = [(vertex,) for vertex in range(vertices)]
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"edge {edge} does not have two endpoints")
        faces.append(tuple(edge))
    return complex_from_maximal_faces(faces)


# ---------------------------------------------------------------------------
# the operators
# ---------------------------------------------------------------------------


def boundary_matrix(complex_: SimplicialComplex, degree: int) -> sp.Matrix:
    """``partial_k`` from degree-``k`` chains to degree-``(k-1)`` chains.

    Entry signs are ``(-1)^i`` for dropping the ``i``-th vertex, which is what
    makes ``partial^2 = 0`` and what the whole construction rests on.
    """
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    rows = complex_.of_dimension(degree - 1) if degree > 0 else ()
    columns = complex_.of_dimension(degree)
    matrix = sp.zeros(len(rows), len(columns))
    if degree == 0:
        return matrix
    index = {simplex: position for position, simplex in enumerate(rows)}
    for column, simplex in enumerate(columns):
        for position in range(len(simplex)):
            face = simplex[:position] + simplex[position + 1 :]
            matrix[index[face], column] = (-1) ** position
    return matrix


def hodge_laplacian(complex_: SimplicialComplex, degree: int) -> sp.Matrix:
    """``Delta_k = partial_{k+1} partial_{k+1}^T + partial_k^T partial_k``."""
    lower = boundary_matrix(complex_, degree)
    upper = boundary_matrix(complex_, degree + 1)
    size = len(complex_.of_dimension(degree))
    result = sp.zeros(size, size)
    if upper.rows == size and upper.cols:
        result += upper * upper.T
    if lower.cols == size and lower.rows:
        result += lower.T * lower
    return result


def dirac_operator(complex_: SimplicialComplex) -> sp.Matrix:
    """``D = d + delta`` on the whole chain complex at once.

    Block off-diagonal: the ``(k-1, k)`` block is ``partial_k`` and the
    ``(k, k-1)`` block is its transpose.  Acting on every degree simultaneously
    is the point -- this is what mixes node, edge and triangle data.
    """
    sizes = complex_.dimensions
    offsets = [sum(sizes[:degree]) for degree in range(len(sizes) + 1)]
    total = sum(sizes)
    matrix = sp.zeros(total, total)
    for degree in range(1, len(sizes)):
        block = boundary_matrix(complex_, degree)
        for row in range(block.rows):
            for column in range(block.cols):
                value = block[row, column]
                if value:
                    matrix[offsets[degree - 1] + row, offsets[degree] + column] = value
                    matrix[offsets[degree] + column, offsets[degree - 1] + row] = value
    return matrix


def chirality_operator(complex_: SimplicialComplex) -> sp.Matrix:
    """``Gamma``: ``+1`` on even-degree chains, ``-1`` on odd.

    The grading involution.  It anticommutes with ``D``, and that single fact is
    the whole of the first refutation.
    """
    diagonal: list[int] = []
    for degree, size in enumerate(complex_.dimensions):
        diagonal.extend([(-1) ** degree] * size)
    return sp.diag(*diagonal)


def dirac_squared_residual(complex_: SimplicialComplex) -> sp.Matrix:
    """``D^2`` minus the block-diagonal Hodge Laplacian. Must be exactly zero."""
    squared = dirac_operator(complex_) ** 2
    blocks = [
        hodge_laplacian(complex_, degree)
        for degree in range(complex_.top_dimension + 1)
    ]
    assembled = sp.diag(*[block for block in blocks if block.rows])
    return sp.simplify(squared - assembled)


def chiral_anticommutator_residual(complex_: SimplicialComplex) -> sp.Matrix:
    """``Gamma D + D Gamma``. Must be exactly zero, and that forces the symmetry."""
    dirac = dirac_operator(complex_)
    grading = chirality_operator(complex_)
    return sp.simplify(grading * dirac + dirac * grading)


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------


def _spectrum(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    """Eigenvalues with multiplicity, exact, sorted by numeric value."""
    if matrix.rows == 0:
        return ()
    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.nsimplify(eigenvalue)] * int(multiplicity))
    return tuple(sorted(values, key=lambda value: complex(sp.N(value)).real))


def dirac_spectrum(complex_: SimplicialComplex) -> tuple[sp.Expr, ...]:
    """Exact eigenvalues of ``D``."""
    return _spectrum(dirac_operator(complex_))


def laplacian_spectrum(complex_: SimplicialComplex) -> tuple[sp.Expr, ...]:
    """Exact eigenvalues of ``Delta``, pooled over every degree."""
    values: list[sp.Expr] = []
    for degree in range(complex_.top_dimension + 1):
        values.extend(_spectrum(hodge_laplacian(complex_, degree)))
    return tuple(sorted(values, key=lambda value: complex(sp.N(value)).real))


def dirac_spectrum_from_laplacian(
    spectrum: Sequence[sp.Expr],
) -> tuple[sp.Expr, ...]:
    """Reconstruct ``spec(D)`` from ``spec(Delta)`` alone.

    The constructive form of the first refutation.  Each non-zero Laplacian
    eigenvalue ``mu`` contributes a pair ``+-sqrt(mu)`` -- the chiral symmetry
    forces both signs and forbids any other multiplicity -- and each zero
    contributes one zero.  No information about the complex is used beyond the
    Laplacian spectrum, so if this reproduces ``spec(D)`` then ``spec(D)`` held
    nothing the Laplacian did not.
    """
    values: list[sp.Expr] = []
    for eigenvalue in spectrum:
        simplified = sp.nsimplify(eigenvalue)
        if simplified == 0:
            values.append(sp.Integer(0))
            continue
        root = sp.sqrt(simplified)
        values.append(-root)
        values.append(root)
    # Each non-zero Laplacian eigenvalue is counted once per degree it appears
    # in, and D pairs those degrees, so the +- pair above double counts by two.
    paired = [value for value in values if value != 0]
    zeros = [value for value in values if value == 0]
    kept = _halve_pairs(paired)
    return tuple(
        sorted(kept + zeros, key=lambda value: complex(sp.N(value)).real)
    )


def _halve_pairs(values: Sequence[sp.Expr]) -> list[sp.Expr]:
    """Keep one copy of each duplicated ``+-`` pair produced by the doubling."""
    counts: dict[sp.Expr, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    kept: list[sp.Expr] = []
    for value, count in counts.items():
        kept.extend([value] * (count // 2))
    return kept


def spectrum_is_symmetric(complex_: SimplicialComplex) -> bool:
    """Is ``spec(D)`` symmetric about zero, as the chiral symmetry demands?"""
    spectrum = dirac_spectrum(complex_)
    negated = sorted(
        (-value for value in spectrum), key=lambda value: complex(sp.N(value)).real
    )
    return all(
        sp.simplify(left - right) == 0 for left, right in zip(spectrum, negated)
    )


def spectra_are_equivalent(complex_: SimplicialComplex) -> bool:
    """Does the Laplacian spectrum alone reproduce the Dirac spectrum?

    ``True`` is the first refutation: the Dirac operator adds no spectral
    information.
    """
    reconstructed = dirac_spectrum_from_laplacian(laplacian_spectrum(complex_))
    actual = dirac_spectrum(complex_)
    if len(reconstructed) != len(actual):
        return False
    return all(
        sp.simplify(left - right) == 0 for left, right in zip(reconstructed, actual)
    )


def kernel_dimension(complex_: SimplicialComplex) -> int:
    """``dim ker D``: the number of zero eigenvalues."""
    return sum(1 for value in dirac_spectrum(complex_) if sp.simplify(value) == 0)


def betti_numbers(complex_: SimplicialComplex) -> tuple[int, ...]:
    """``dim ker Delta_k`` in each degree: the Betti numbers.

    An external referee for the whole construction -- these are known
    independently for the standard test complexes, so if the boundary signs were
    wrong they would come out wrong.
    """
    return tuple(
        len(complex_.of_dimension(degree))
        - hodge_laplacian(complex_, degree).rank()
        for degree in range(complex_.top_dimension + 1)
    )


def homology_discards(complex_: SimplicialComplex) -> int:
    """How many non-zero eigenvalues persistent homology throws away.

    The one part of the brief that survives.  Homology reads only the kernel, so
    every non-zero eigenvalue of the Laplacian is information it discards and a
    spectral method keeps.  That gap is real -- but it is a statement about
    homology versus the Laplacian, not about the Laplacian versus Dirac.
    """
    return len(dirac_spectrum(complex_)) - kernel_dimension(complex_)


# ---------------------------------------------------------------------------
# chirality
# ---------------------------------------------------------------------------

#: A five-point configuration in three dimensions with non-zero signed volumes,
#: so that its mirror image is genuinely a different embedded object.
CHIRAL_POINTS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.3, 0.4, 0.9),
)


def reflect(
    points: Sequence[Sequence[float]], axis: int = 2
) -> tuple[tuple[float, ...], ...]:
    """Mirror a point cloud through the hyperplane ``coordinate[axis] = 0``."""
    if not points:
        raise ValueError("need at least one point")
    dimension = len(points[0])
    if not 0 <= axis < dimension:
        raise ValueError(f"axis {axis} outside {dimension} dimensions")
    return tuple(
        tuple(
            -value if position == axis else value
            for position, value in enumerate(point)
        )
        for point in points
    )


def distance_matrix(points: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Pairwise Euclidean distances, squared then rooted in plain arithmetic."""
    if not points:
        raise ValueError("need at least one point")
    return tuple(
        tuple(
            sum((left - right) ** 2 for left, right in zip(first, second)) ** 0.5
            for second in points
        )
        for first in points
    )


def rips_complex(
    points: Sequence[Sequence[float]], radius: float
) -> SimplicialComplex:
    """The Vietoris-Rips complex at scale ``radius``.

    A simplex is included when every pair of its vertices is within ``radius``.
    Note what the input is: distances, and nothing else.  That is the whole
    content of the second refutation.
    """
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    distances = distance_matrix(points)
    count = len(points)
    faces: list[Sequence[int]] = [(vertex,) for vertex in range(count)]
    for size in range(2, count + 1):
        for candidate in combinations(range(count), size):
            if all(
                distances[first][second] <= radius
                for first, second in combinations(candidate, 2)
            ):
                faces.append(candidate)
    return complex_from_maximal_faces(faces)


def signed_volume(
    points: Sequence[Sequence[float]], indices: Sequence[int]
) -> float:
    """Orientation determinant of four points: the thing that flips under mirroring.

    Not a function of the pairwise distances, which is exactly why a
    distance-based filtration cannot recover it.
    """
    if len(indices) != 4:
        raise ValueError(f"need four indices, got {len(indices)}")
    origin = points[indices[0]]
    rows = [
        [points[index][axis] - origin[axis] for axis in range(3)]
        for index in indices[1:]
    ]
    return float(sp.Matrix(rows).det())


def chirality_signature(points: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Signs of every four-point orientation determinant."""
    count = len(points)
    if count < 4:
        raise ValueError(f"need at least four points, got {count}")
    signs = []
    for candidate in combinations(range(count), 4):
        volume = signed_volume(points, candidate)
        signs.append(0 if volume == 0 else (1 if volume > 0 else -1))
    return tuple(signs)


def chirality_is_invisible_to_distances(
    points: Sequence[Sequence[float]] = CHIRAL_POINTS,
    radius: float = 1.5,
) -> bool:
    """The second refutation, demonstrated.

    ``True`` when the configuration is genuinely chiral -- its orientation
    signature flips under reflection -- and yet its distance matrix, its
    Vietoris-Rips complex, and therefore its Dirac spectrum are all identical to
    the mirror image's.
    """
    mirrored = reflect(points)
    genuinely_chiral = chirality_signature(points) != chirality_signature(mirrored)
    same_distances = distance_matrix(points) == distance_matrix(mirrored)
    same_complex = rips_complex(points, radius) == rips_complex(mirrored, radius)
    same_spectrum = dirac_spectrum(rips_complex(points, radius)) == dirac_spectrum(
        rips_complex(mirrored, radius)
    )
    return genuinely_chiral and same_distances and same_complex and same_spectrum


# ---------------------------------------------------------------------------
# the inherited blindspot
# ---------------------------------------------------------------------------

#: A Laplacian-cospectral, non-isomorphic pair of graphs on six vertices, found
#: by exhaustive search.  The Dirac operator inherits the blindspot: identical
#: spectra, different complexes.
COSPECTRAL_PAIR: tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] = (
    ((0, 2), (0, 3), (0, 4), (0, 5), (1, 4), (1, 5), (2, 3)),
    ((0, 2), (0, 4), (0, 5), (1, 2), (1, 4), (1, 5), (2, 3)),
)

#: Six.  Exhaustive search finds no cospectral non-isomorphic pair on five
#: vertices or fewer, so this is the smallest register where the blindspot bites.
MINIMAL_COSPECTRAL_ORDER: int = 6


def cospectral_pair_is_a_blindspot() -> bool:
    """Do two non-isomorphic complexes share a Dirac spectrum?

    ``True``.  Since ``spec(D)`` is a function of ``spec(Delta)``, every
    Laplacian cospectrality is a Dirac cospectrality -- the "square root" cannot
    separate what the Laplacian merges.
    """
    first, second = (
        graph_complex(MINIMAL_COSPECTRAL_ORDER, edges) for edges in COSPECTRAL_PAIR
    )
    if first == second:
        raise ArithmeticError("the cospectral pair collapsed to one complex")
    left = dirac_spectrum(first)
    right = dirac_spectrum(second)
    if len(left) != len(right):
        return False
    return all(
        sp.simplify(one - other) == 0 for one, other in zip(left, right)
    )
