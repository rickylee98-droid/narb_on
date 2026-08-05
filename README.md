# Spectral Analysis of Tetrahedral Point Clouds

Builds a large cluster of unit-edge regular tetrahedra, merges coincident vertices,
constructs the unit-distance graph, assembles the sparse combinatorial Laplacian
`L = D - A`, and analyses the low end of its spectrum for degeneracies, spacing
structure, and spectral gaps.

## The central finding

**Maximising packing density and building a richly connected graph are different
objectives, and which one you pick decides whether a spectral fingerprint exists.**

The densest known packing of regular tetrahedra — Chen–Engel–Glotzer at
φ = 4000/4671 ≈ 0.856347 — is implemented exactly (`--backend ceg`) from the paper's
own rationals, and certified here by separating-axis test rather than trusted. Its
spectral answer is **complete and exact**, and it is not the one you might expect.

CEG is a *dimer* crystal: two tetrahedra sharing a face. That face contact is genuine
vertex sharing (8 vertices → 5), so each dimer forms one 5-vertex component — the
triangular dipyramid, `K₅` minus an edge — while **distinct dimers share nothing**. For
1372 tetrahedra the pipeline finds 3430 distinct vertices, **686 components of size 5**,
6174 edges, degrees only 3 and 4. The entire Laplacian spectrum is therefore

> **{0, 3, 5, 5, 5} repeated once per dimer** — here exactly 686 zeros, 686 threes, 1428 fives.

Algebraic connectivity is exactly 0. Every degeneracy is *repetition of identical
components*, not a hidden symmetry group. That is the honest answer for a maximum-density
packing.

### Density trades off against connectivity, inside the family

The paper's construction is a **three-parameter family** (eq. 6), and the whole family is
implemented — so the trade-off is directly measurable. Three named members reproduce their
published densities exactly and all pass the separating-axis test:

| `--ceg-variant` | φ | Components | Max degree | Structure |
| --- | --- | --- | --- | --- |
| `optimal` | **4000/4671** = 0.856348 | exactly 1 per dimer | 4 | bounded |
| `torquato-jiao` | 12250/14319 = 0.855507 | exactly 1 per dimer | 4 | bounded |
| `kallus-elser-gravel` | 100/117 = 0.854701 | **exactly 12(reps−1)** | **6** | **extended** |

The *least* dense of the three has by far the richest graph. KEG sits at the symmetric
origin `(u,v,w) = (0,0,0)` and is transitive on individual tetrahedra, and that extra
symmetry produces unit-distance coincidences **between** dimers — not shared vertices, but
distinct vertices exactly 1 apart. Its component count grows only *linearly* with box size
while the dimer count grows cubically, so mean component size grows as ~reps²: the
components are extended networks, not isolated dipyramids. Its Fiedler value is 0.157,
not 3.

So buying the last 0.2% of density costs you the entire connected structure. Both facts
are locked in by tests at several box sizes.

The richest spectrum of all still lives in the **interlocking** structure: the
tetrahedral–octahedral honeycomb on the FCC lattice, where tetrahedra genuinely share
vertices and edges. All backends are implemented, so every contrast is reproducible.

*(An earlier version of this README claimed a dense packing gives one disjoint `K₄` per
tetrahedron. That is true only of the stochastic `packing` backend, whose tetrahedra land
in fully generic position; the real CEG optimum shares faces within dimers and is one
step richer.)*

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

# The densest known packing, phi = 4000/4671, exact and certified.
# Ask for more eigenvalues than there are components to see past the kernel:
python tetra_spectral_analysis.py --backend ceg --min-tetrahedra 1000 --num-eigenvalues 2800

# A stochastic density search instead (slower; reaches ~0.5-0.75, never the optimum)
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
| `tetra_geometry.py` | Canonical tetrahedron, exact SAT overlap test, honeycomb / CEG / ASC generators |
| `tetra_spectral.py` | Vertex merging, graph construction, Laplacian, eigensolver, degeneracy analysis |
| `tetra_spectral_analysis.py` | CLI, reporting, export |
| `test_tetra_spectral.py` | Test suite (120 tests) |

## Method notes

**Canonical tetrahedron.** Four alternating corners of a cube. Every pair is a face
diagonal, so all six edges are equal *by construction* rather than by optimisation —
the maximum edge-length deviation in a 1136-tetrahedron honeycomb is 4.4e-16.

**Vertex merging** is single-linkage via a KD-tree — the metric-space analogue of
`np.isclose(rtol=0, atol=atol)`, in O(n log n) rather than O(n²). Single-linkage can
chain, so the largest cluster radius is measured and a warning is emitted if it exceeds
10× the tolerance. In the honeycomb, coincident vertices are bitwise identical, so the
observed radius is < 1e-15 and no chaining occurs.

**The CEG family** is implemented as equation (6) of Chen, Engel & Glotzer (2010): the
three-parameter linear space `(u, v, w)` of double dimer configurations satisfying their
nine linear incidence conditions, carried in exact `Fraction` arithmetic. The packing
places positive dimers on the even sublattice `L⁺ = ⟨a+b, b+c, c+a⟩` and negative
(inverted) dimers on the coset `L⁻ = L⁺ + (d+a)`, giving 4 tetrahedra per cell.
Coordinates are rescaled from the paper's edge length of 3√2 to unit edge.

Cross-checks that the implementation is right, all of them tested:

- eq. (6) at `(3/160, 3/64, 0)` reproduces the Theorem 1 vectors to 1e-15 — two
  independent statements in the paper agreeing.
- The closed form `φ = 100/(117 + 60u² − 80uv − 80v²)` (eq. 11) matches the geometric
  `4·V_tet/|det L|` at every tested point.
- `w` is a pure lattice shear, so density is provably independent of it — verified.
- `ceg_in_restricted_space` implements the four half-spaces of `P″` (eq. 9). These are
  *sufficient* conditions for a packing, not necessary, so construction runs the
  separating-axis test regardless rather than trusting the predicate.

**Eigensolver.** When the graph is disconnected the Laplacian is block diagonal and the
spectrum is assembled from the blocks. This is a correctness requirement, not an
optimisation: a graph with 686 components has a 686-fold degenerate kernel, and no
Krylov method can resolve a degeneracy of that order — ARPACK converges to an arbitrary
subset of the invariant subspace and silently returns eigenvalues that are *not* the
smallest. On the CEG packing, whole-matrix shift-invert reported 53 zeros and 7 threes
where the true answer is 60 zeros. The block path returns it exactly. The mismatch
between kernel dimension and component count is what exposed the bug, and that check is
still printed on every run.

Within a single block the solver tries dense LAPACK when the block is small enough to be
exact and cheap; then ARPACK shift-invert at a small *negative* sigma (a shift of exactly
zero would factorise the singular `L`); then direct ARPACK in `which="SA"` mode. A
partial ARPACK result is preferred over raising. Sparse and dense paths are tested to
agree to 1e-8, and the block path against dense on the full spectrum to 1e-10.

The kernel dimension equals the component count, but that can only be *checked* when the
computed window reaches past the kernel; when `k` is smaller than the number of
components the report says the kernel is unresolved rather than flagging a false
mismatch.

**Statistics are suppressed when undersampled.** The mean adjacent-gap ratio is reported
only from at least 8 ratios. On the CEG packing there are 3 distinct levels and hence 2
spacings; quoting ⟨r⟩ = 0.667 there and calling it "near GOE" would be reading a symmetry
class out of noise.

**Packing search** is adaptive-shrinking-cell Monte Carlo on hard particles. Three
design points matter, and each was established by measurement rather than assumption:

1. Lattice moves use a *traceless* (volume-preserving) shear plus an explicit
   compression factor. A naive random symmetric strain changes volume by ±17% while the
   compression bias is 0.4%; since expansion never creates overlaps, those moves always
   pass and the cell random-walks *outward*. Making every accepted lattice move strictly
   densifying is what gives monotone convergence.
2. Periodic reheating of the step sizes is required. Without it the search fully stalls
   (0.5573 → 0.5579 for 2.7× more cycles); with it, 0.5573 → 0.5932.
3. Neither dimer idea helped the stochastic search, and both were kept only because the
   measurements say so. *Seeding* random tetrahedra as dimer pairs gave 0.42–0.52 across
   three seeds versus 0.49–0.72 for random initialisation, because reheating disassembles
   the pairs before jamming. Making the dimer a *rigid* motif (`--motif dimer`, halving
   the degrees of freedom) did no better: best-of-four 0.5325 versus 0.7238 for free
   tetrahedra. Seed variance dominates either way, which is why the builder restarts and
   keeps the densest. The exact optimum comes from `--backend ceg`, not from this search.

Periodic image ranges are computed from the cell's perpendicular widths rather than
assuming a 3×3×3 minimum-image scheme, which silently misses collisions once the cell
shrinks below the particle diameter — as it must at high density. A cell too anisotropic
to certify is rejected rather than under-tested.

**On the achieved density.** The stochastic search reaches roughly 0.5–0.75 depending on
seed, and makes no claim to reach the optimum — for that, use `--backend ceg`, which is
exact. Every density reported by the search is one that was measured, and
`--verify-packing` runs an exact separating-axis check that no tetrahedra interpenetrate.

## Tests

```bash
python -m pytest test_tetra_spectral.py -v
```

Coverage includes closed-form checks (K4's Laplacian spectrum is exactly {0, 4, 4, 4};
tetrahedron volume and circumradius), the FCC kissing number, sparse/dense/block solver
agreement, kernel dimension versus component count, monotonicity of the packing search,
rejection of degenerate periodic cells, and the two structural claims themselves: that
the honeycomb's multiplicities never exceed 3 with triplets dominating (`O_h`), and that
the CEG packing's spectrum is exactly {0, 3, 5, 5, 5} per dimer with V, φ and
non-overlap all matching the paper.
