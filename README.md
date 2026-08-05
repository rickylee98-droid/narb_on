# Spectral Analysis of Tetrahedral Point Clouds

Builds a large cluster of unit-edge regular tetrahedra, merges coincident vertices,
constructs the unit-distance graph, assembles the sparse combinatorial Laplacian
`L = D - A`, and analyses the low end of its spectrum for degeneracies, spacing
structure, and spectral gaps.

## The central finding

**A maximum-density tetrahedron packing and a vertex-sharing graph are mutually
exclusive, and this is what determines whether a spectral fingerprint exists at all.**

In a dense packing, tetrahedra sit in *generic* position: they touch face-to-face at
incommensurate offsets and essentially never share a vertex or land at unit separation.
Merging at `atol=1e-5` and connecting at distance exactly 1 therefore produces one
disjoint `K4` per tetrahedron. The pipeline confirms this directly — 1372 tetrahedra
give **1372 components, all of size 4**, every vertex of degree 3, algebraic
connectivity exactly 0, and a spectrum that is nothing but the kernel. There is no
fingerprint to find, and that is a property of dense packings rather than a defect of
the method.

The rich, degenerate spectrum lives instead in the **interlocking** structure: the
tetrahedral–octahedral honeycomb on the FCC lattice, where tetrahedra genuinely share
vertices and edges. Both geometries are implemented, so the contrast is reproducible.

### Honeycomb result (1136 tetrahedra, 767 distinct vertices)

| Quantity | Value |
| --- | --- |
| Distinct vertices | 767 (from 4544 raw; up to 8 tetrahedra per vertex) |
| Graph | connected, 3900 edges, max degree **12** (the FCC kissing number) |
| λ₀ | exactly 0, multiplicity 1 = component count |
| λ₁ (Fiedler value) | 0.306588497197396, **multiplicity 3** |
| Distinct levels in the first 100 eigenvalues | 43 |
| Multiplicities observed | **{1, 2, 3} only**, max = 3 |
| Eigenvalues in degenerate levels | 90% |

Every multiplicity is an irreducible-representation dimension of the octahedral group
`O_h` (which has irreps of dimension 1, 1, 2, 3, 3), and triplets are the most common
level — 24 of 43. That is the algebraic fingerprint, and it is exactly the one the
geometry predicts.

**The symmetry is the finite point group `O_h` of order 48 — not a Lie group.** A
continuous symmetry would produce multiplicities growing without bound (a Lie group has
arbitrarily large irreps); the hard ceiling at 3 is the signature of a finite group. The
spherical cluster cut-off is used precisely because a ball is `O_h`-invariant, so the
cluster inherits the lattice point group instead of having it broken by the boundary.

Two further cautions the script prints for itself:

- The mean adjacent-gap ratio ⟨r⟩ ≈ 0.316 sits near the Poisson value (0.386), not the
  GOE value (0.536). Independent symmetry sectors superposing without interacting is
  itself a symmetry signature — not evidence of anything exotic.
- The spectral gap of a finite cluster cut from an infinite lattice scales as O(1/L²)
  with cluster diameter. It is a **finite-size effect**, not an intrinsic mass gap.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Default: honeycomb backend, the configuration with a fingerprint
python tetra_spectral_analysis.py --min-tetrahedra 1000 --num-eigenvalues 100

# The contrast case: a genuine dense packing (slower; runs a Monte Carlo search)
python tetra_spectral_analysis.py --backend packing --asc-cycles 1500 --asc-restarts 4

# Export tables for downstream work
python tetra_spectral_analysis.py --csv-prefix results/run --json-summary results/summary.json

# Certify that no tetrahedra interpenetrate (exact separating-axis test)
python tetra_spectral_analysis.py --verify-packing
```

`--help` lists every option. Key ones: `--merge-atol` (vertex merge tolerance, default
`1e-5`), `--edge-atol` (unit-distance tolerance, default `1e-6`), `--degeneracy-rtol`
(level-grouping tolerance), `--dense-threshold` (dense/sparse solver crossover),
`--threads` (BLAS thread count).

The honeycomb backend runs end to end in well under a second. The packing backend is
dominated by the Monte Carlo search — roughly 90 seconds at 4 restarts × 1500 cycles.

## Layout

| File | Contents |
| --- | --- |
| `tetra_geometry.py` | Canonical tetrahedron, exact SAT overlap test, honeycomb and packing generators |
| `tetra_spectral.py` | Vertex merging, graph construction, Laplacian, eigensolver, degeneracy analysis |
| `tetra_spectral_analysis.py` | CLI, reporting, export |
| `test_tetra_spectral.py` | Test suite (95 tests) |

## Method notes

**Canonical tetrahedron.** Four alternating corners of a cube. Every pair is a face
diagonal, so all six edges are equal *by construction* rather than by optimisation —
the maximum edge-length deviation in a 1136-tetrahedron honeycomb is 4.4e-16.

**Vertex merging** is single-linkage via a KD-tree — the metric-space analogue of
`np.isclose(rtol=0, atol=atol)`, in O(n log n) rather than O(n²). Single-linkage can
chain, so the largest cluster radius is measured and a warning is emitted if it exceeds
10× the tolerance. In the honeycomb, coincident vertices are bitwise identical, so the
observed radius is < 1e-15 and no chaining occurs.

**Eigensolver** tries three paths in order and reports which was used: dense LAPACK when
the matrix is small enough to be exact and cheap; ARPACK shift-invert at a small
*negative* sigma (a shift of exactly zero would factorise the singular `L`); then direct
ARPACK in `which="SA"` mode. A partial ARPACK result is preferred over raising — if
convergence stalls, the converged subset is returned and the method is reported as
`arpack-partial`. Sparse and dense paths are tested to agree to 1e-8.

The kernel dimension equals the component count, but that can only be *checked* when the
computed window reaches past the kernel; when `k` is smaller than the number of
components the report says the kernel is unresolved rather than flagging a false
mismatch.

**Packing search** is adaptive-shrinking-cell Monte Carlo on hard particles. Three
design points matter, and each was established by measurement rather than assumption:

1. Lattice moves use a *traceless* (volume-preserving) shear plus an explicit
   compression factor. A naive random symmetric strain changes volume by ±17% while the
   compression bias is 0.4%; since expansion never creates overlaps, those moves always
   pass and the cell random-walks *outward*. Making every accepted lattice move strictly
   densifying is what gives monotone convergence.
2. Periodic reheating of the step sizes is required. Without it the search fully stalls
   (0.5573 → 0.5579 for 2.7× more cycles); with it, 0.5573 → 0.5932.
3. Dimer seeding is available (`seed_dimers=True`) but **defaults to off because it
   measurably does not help**: 0.42–0.52 across three seeds versus 0.49–0.72 for random
   initialisation, since reheating disassembles the seeded pairs well before jamming.
   Seed variance dominates, which is why the builder restarts and keeps the densest.

Periodic image ranges are computed from the cell's perpendicular widths rather than
assuming a 3×3×3 minimum-image scheme, which silently misses collisions once the cell
shrinks below the particle diameter — as it must at high density. A cell too anisotropic
to certify is rejected rather than under-tested.

**On the achieved density.** The search reaches roughly 0.6–0.75 at default settings,
against the best packing known in the literature — the Chen–Engel–Glotzer dimer double
lattice at 4000/4671 ≈ 0.856347. This script does **not** reproduce that optimum and
makes no claim to; approaching it needs far longer runs and larger cells than a
demonstration default should impose. Every density reported is one that was measured,
and `--verify-packing` runs an exact separating-axis check that no tetrahedra
interpenetrate.

## Tests

```bash
python -m pytest test_tetra_spectral.py -v
```

Coverage includes closed-form checks (K4's Laplacian spectrum is exactly {0, 4, 4, 4};
tetrahedron volume and circumradius), the FCC kissing number, sparse/dense solver
agreement, kernel dimension versus component count, monotonicity of the packing search,
rejection of degenerate periodic cells, and the `O_h` degeneracy claim itself — that no
multiplicity exceeds 3 and that triplets dominate.
