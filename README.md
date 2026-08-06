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

### The connected set is exactly the plane u = 0

Sweeping the family and counting *exact* inter-dimer unit contacts gives a sharp answer:
they exist **only when u = 0**, for any v and any w, and vanish everywhere else. That is
not a numerical accident — it reproduces the paper's own equation (10), where the
vertex-to-edge incidence condition `H_{a−b}` holds precisely on `−u ≤ 0 ∧ +u ≤ 0`. The
graph observation and their incidence geometry are the same fact, arrived at
independently.

So connectivity is a **codimension-1 condition**, and the "moment the Fiedler value
drops" is a genuine discontinuity at u = 0, not a curve. Along the straight path from the
KEG origin to the density optimum, the 156 inter-dimer contacts drift linearly,
`|d − 1| = 0.0437 t`, so a run with edge tolerance `atol` appears to keep them until
`t ≈ atol/0.0437` — **the apparent transition width is the tolerance, nothing else.**
The figure shows it shifting a decade per decade of tolerance. The collapse itself has
two stages: 24 components (largest 20 vertices) → 27 (largest 10) → 54 (one per dimer,
largest 5), with λ₁⁺ going 0.517 → 0.586 → 3.

Because connectivity only needs `u = 0`, density can still be maximised *within* that
plane. On it, φ = 100/(117 − 80v²), so the problem is to maximise |v|; adding the two
binding `P″` constraints `2v − w ≤ 33/320` and `v + w ≤ 3/64` gives **v ≤ 1/20**. The
result is `(0, 1/20, −1/320)` with **φ = 125/146** — which turns out to be the paper's
own `C3+cen` entry, derived here from the constraints rather than looked up. It is
available as `--ceg-variant densest-connected`.

That prices the trade-off exactly:

> **Requiring a connected unit-distance graph costs 4000/4671 − 125/146 = 125/681966,
> or 0.0214% of the optimal density.**

### Density trades off against connectivity, inside the family

The paper's construction is a **three-parameter family** (eq. 6), and the whole family is
implemented — so the trade-off is directly measurable. Three named members reproduce their
published densities exactly and all pass the separating-axis test:

| `--ceg-variant` | φ | Components | Max degree | Structure |
| --- | --- | --- | --- | --- |
| `optimal` | **4000/4671** = 0.856348 | exactly 1 per dimer | 4 | bounded |
| `densest-connected` | 125/146 = 0.856164 | 4 (of 128 dimers) | 6 | **extended** |
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

# The densest member whose graph is actually connected
python tetra_spectral_analysis.py --backend ceg --ceg-variant densest-connected --num-eigenvalues 2800

# A stochastic density search instead (slower; reaches ~0.5-0.75, never the optimum)
python tetra_spectral_analysis.py --backend packing --asc-cycles 1500 --asc-restarts 4

# The N = 3 phase at exactly 2/3, reconstructed
python tetra_spectral_analysis.py --backend n3 --min-tetrahedra 500 --num-eigenvalues 2200 --verify-packing

# The three-fold screw family (the orbit reading, which caps at 0.5956)
python tetra_spectral_analysis.py --backend p3 --p3-screw 2 --asc-cycles 3000 --verify-packing

# Map connectivity across the family and render the figure
python explore_connectivity.py --reps 3 --out connectivity

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
| `explore_connectivity.py` | Family sweep: connectivity vs density, CSV + figure |
| `tetra_fastsat.py` | Optional compiled (numba) periodic overlap kernel; NumPy fallback |
| `tetra_lift.py` | Higher-dimensional lift obstructions: Z-module rank, root-system angles |
| `test_tetra_spectral.py` | Test suite (256 tests) |

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

**Packing search** is adaptive-shrinking-cell Monte Carlo on hard particles. Lattice
moves use a *traceless* (volume-preserving) shear plus an explicit compression factor: a
naive random symmetric strain changes volume by ±17% while the compression bias is 0.4%,
and since expansion never creates overlaps those moves always pass and the cell
random-walks *outward*. Making every accepted lattice move strictly densifying is what
gives monotone convergence.

> **Correction — a bug invalidated every density this search previously reported.**
> Fractional coordinates were never wrapped back into the cell, so a particle could
> random-walk several cells away. The periodic image range is derived from the lattice
> widths and is only valid for particles *inside* the cell, so a drifted particle's true
> neighbours were never tested and overlaps went unseen. It surfaced at N = 3, seed 1007,
> as φ = 0.982 — a physically impossible density that an independent tiling check showed
> to be **581 overlapping pairs with penetration depth 0.457**, nearly half an edge.
>
> Fixed by wrapping coordinates each move (a lattice translation, exactly neutral for a
> periodic packing) and widening the image span by one cell to cover the fractional
> offset between particles. `build_dense_packing` now also runs an independent
> `find_overlapping_pairs` check on the assembled cloud, so a flaw in the image-range
> reasoning cannot hide again.
>
> **CEG and honeycomb were never affected** — both are verified by independent tiling
> checks and reproduce their published densities exactly. Only the stochastic backend's
> numbers moved.

With the bug fixed the honest picture is more modest, and two claims made earlier do not
survive:

- The search reaches roughly **0.43–0.61**, not the 0.5–0.75 previously reported. Seed 11
  at N = 4 gave 0.7238 before the fix and **0.6055** after; the difference was overlap.
- **Reheating is not demonstrated to help.** The earlier figure (0.5573 → 0.5932) came
  from invalid packings. Re-measured at seed 7: 0.5338 without reheating at both 1500 and
  4000 cycles, 0.5188 / 0.5198 with. Both plateau; reheating is marginally *worse* here.
  It is kept as an option, no longer as a recommendation.
- Rigid dimer motifs still do not help, and that direction does survive: best-of-four
  **0.5325** (dimer) versus **0.6055** (free tetrahedra).

Periodic image ranges are computed from the cell's perpendicular widths rather than
assuming a 3×3×3 minimum-image scheme, which silently misses collisions once the cell
shrinks below the particle diameter. A cell too anisotropic to certify is rejected rather
than under-tested.

**On the achieved density.** The stochastic search makes no claim to reach the optimum —
for that, use `--backend ceg`, which is exact. Every density it reports is measured, and
`--verify-packing` runs an exact separating-axis check.

## Where a higher-dimensional lift *does* exist

The rank test above returns 3 for every crystal in this repo, which is a negative result —
and a test that can only ever return a negative is worth very little. So it was pointed at a
structure whose lift is real.

The paper's largest entry is a **dodecagonal quasicrystal approximant** (N = 8×82,
φ = 0.850267), and dodecagonal quasicrystals are cut-and-project sets from **five
dimensions**: four for the in-plane module `Z[ζ₁₂]` — whose minimal polynomial `x⁴ − x² + 1`
has degree 4 — plus one for the periodic stacking axis. `tetra_lift.dodecagonal_quasilattice`
builds one: integer points of `Z⁴` accepted when their image under the *conjugate* star
(Galois conjugation √3 → −√3) lands in the acceptance window, then projected by the physical
star.

| structure | Z-module rank | lift? |
| --- | --- | --- |
| **dodecagonal quasicrystal** | **5** (in-plane 4 = `Z[ζ₁₂]`) | **yes** |
| CEG optimal / KEG / densest-connected | 3 | no |
| N = 3 phase | 3 | no |

The machinery detects it, and the resulting point set is verified 12-fold symmetric in its
core and genuinely discrete (nearest-neighbour spacing bounded away from zero — the
acceptance window is what makes it a quasicrystal rather than a dense module).

**So the original higher-dimensional intuition was not wrong — it was aimed at the wrong
phase.** The dimer crystals are rational and three-dimensional, and no amount of searching
will find a lift in them. The quasicrystal in the same paper genuinely is a shadow of a
five-dimensional lattice. This is the symmetry *class*, not a reconstruction of the
82-tetrahedron approximant, whose coordinates are in the same dead data file as the rest.

## No higher-dimensional lift exists in the crystals

`tetra_lift.py` tests whether these structures could be 3D shadows of something more
symmetric — a cut-and-project quasicrystal, or a graph drawn from a root system such as
`D₆` or `E₈`. Two independent obstructions both fire, and either alone is decisive.

**1. The coordinate Z-module has rank 3.** A cut-and-project set from dimension *n* has
coordinate rank *n* — icosahedral quasicrystals have rank 6, dodecagonal ones rank 5 —
because their coordinates live in `Q(√5)` or `Q(√3)` with rationally independent
components. Every CEG family member has **exactly rational** coordinates (common
denominators 5, 160, 320 for the three presets), hence rank 3, hence it *is* a
three-dimensional lattice structure with no higher-dimensional space to descend from.

The test is built to be able to say yes: it takes coordinates as coefficient vectors over
an algebraic basis, and a regression test feeds it icosahedron vertices over `[1, φ]` and
confirms it returns **rank 6**. A test that could only ever deny a lift would be worthless.

**2. The edge angles are not root-system angles.** Two roots of norm 2 have integer inner
product, so minimal vectors in *any* root lattice meet only at 60°, 90°, 120° or 180°. The
`densest-connected` graph shows **15 distinct angles, of which only 60° is legal**. Among
the offenders:

> **109.471221° = arccos(−1/3) — the regular tetrahedron's own vertex angle.**

That angle is intrinsic to the shape, not to the packing, so *no* arrangement of regular
tetrahedra embeds in a root lattice, in any dimension. This is the same incommensurability
that stops tetrahedra from tiling space.

The Laplacian multiplicities tell the same story: `{4: 122, 24: 1, 128: 1}` — every value
divisible by 4, the number of identical components. They are component repetition, not
irrep dimensions. `E₈`'s smallest non-trivial irrep is 248-dimensional.

## The N = 2 phase: solved to nine figures

Table I gives N = 2 at φ₂ = 9/(139 − 40√10) ≈ 0.719488, "2 monomers, transitive". Getting
there took two ideas, one about the frame and one about the optimiser.

**Choose the frame so the algebra is visible.** In the unit-edge frame the target
determinant is (139√2 − 80√5)/54 — two mixed radicals, and nothing is recognisable. In the
*integer* frame (tetrahedron at alternating cube corners, edge 2√2) it becomes

> |det A| = **16(139 − 40√10)/27** — pure `Q(√10)`, no √2.

**Monte Carlo cannot finish this.** Near jamming the accessible moves are smaller than any
sensible step size, and the search plateaus at 0.7155 — 99.4% of target, and it looks
converged from the inside. Three attempts to fix it *as* a Monte Carlo all failed to close
the gap: adaptive compression helped a little (0.7037 → 0.7145), refinement from the
unconstrained optimum reached 0.7155, and an **isobaric** version that allows volume
increases under a pressure penalty did *worse* (0.6786), so the monotone-descent hypothesis
was wrong.

What works is dropping Monte Carlo entirely and treating it as a constrained programme:
minimise cell volume subject to every contact depth staying at or below zero, rebuilding the
active set each round because which images touch changes as the cell shrinks.

| method | φ | of target |
| --- | --- | --- |
| unconstrained ASC | 0.711822 | 98.94% |
| double lattice, adaptive compression | 0.714462 | 99.30% |
| MC refinement from the ASC optimum | 0.715488 | 99.44% |
| **constrained refinement** | **0.7194880889** | **99.999999%** |

The optimum is **jammed** — 14 of 39 neighbour pairs in contact — and the safety margin is
the whole story near the end: holding contacts 1e-6 apart leaves the answer 6e-6 short,
exactly as it should.

Finally, `tetra_lift.recognise_quadratic` reads the converged determinant back as an exact
algebraic number:

> **2224/27 − (640/27)√10 = 16(139 − 40√10)/27**

which is the target, recovered rather than assumed. The recogniser returns `None` at a
tolerance of 1e-8 because the numerics are good to 4.5e-8 — the height bound reporting the
truth rather than fitting noise, which is what makes the positive result meaningful.

**What is still not done:** the individual lattice *entries* are not yet in closed form. The
determinant is a rotation- and basis-invariant, so it is recoverable from the numerics as it
stands; pinning the entries needs a canonical frame (LLL-reduced basis, rotation gauge fixed)
and another few digits. The structure, the field and the density are settled; the coordinates
are not.

## Recovering the N = 3 phase, φ = 2/3

Table I lists an N = 3 phase at **exactly φ = 2/3**, "3 monomers, three-fold symmetric".
Its coordinates are not in the paper: ref. [26] is an external data file that is **no longer
online** (the Glotzer group URL 404s), and ref. [27] Appendix D is likewise out of reach. So
the structure was reconstructed instead — and the reconstruction succeeded.

### The ansatz that worked

"Three-fold symmetric" admits more than one reading, and the first one tried was wrong. The
natural guess is that the rotation *permutes* the three monomers — one orbit of a screw axis.
That family was implemented (`--backend p3`), searched to convergence, and **caps at 0.5956**,
89% of target. Raising the budget tenfold moved it +0.2%, so it was converged, not starved.

The right reading is the other one: the rotation maps each monomer **to itself**. A hexagonal
cell has *three distinct* three-fold axes — Wyckoff sites 1a, 1b, 1c at `(0,0,z)`, `(1/3,2/3,z)`,
`(2/3,1/3,z)` — and a regular tetrahedron has its own C₃ axis. Put one monomer on each,
C₃ axis aligned, and the crystal is three-fold symmetric with three monomers and no orbit at all.

Two further readings were eliminated by arithmetic before any code was written:

- all three monomers on one shared axis needs `c ≥ 3 × 0.8165`, forcing `a ≤ 0.5` and putting
  columns 0.29 apart against a base radius of 0.577 — impossible;
- a rhombohedral R-centred cell puts one tetrahedron in the primitive cell, making it a
  *lattice* packing, capped at 18/49 = 0.367 (Hoylman) — Table I's own N = 1 entry.

### The structure

The search hit 0.6658 (99.87% of 2/3) with `a ≈ 0.8665`, `c ≈ 0.8167`, heights ≈ [0, 0, ½].
Those are recognisable, and the closed forms are exact:

| | |
| --- | --- |
| **a = √3/2** | the height of a unit equilateral triangle |
| **c = √(2/3)** | *the tetrahedron's own height* — each column is exactly one tetrahedron tall |
| V = (√3/2)a²c | = **3√2/8** |
| φ = 3·V_tet / V | = **2/3, exactly** |

Monomers sit at axial offsets {0, ½}c with azimuths from {0, π/3} and **mixed** apex
directions — all three aligned collapses the density to 0.43. Twenty-four symmetry-equivalent
configurations pack; `--backend n3` builds one, verified on construction: unit edges to 1e-15,
density against 2/3 to 1e-14, and zero overlaps under the separating-axis test both
periodically and across a tiled 648-tetrahedron block.

This matches Table I's description and density exactly. Without the authors' coordinates it
cannot be *proved* identical to theirs, but the agreement is not loose: an exact rational
density, the stated motif, and the stated symmetry.

### Its graph answers the original question

The N = 3 phase is the only dense packing here with genuinely rich connectivity:

| | N = 3 phase | CEG optimum | P3 search |
| --- | --- | --- | --- |
| φ | 2/3 | 4000/4671 | 0.5956 |
| raw → distinct vertices | 2592 → **2142** | no sharing between dimers | no sharing at all |
| components (648 tetrahedra) | **16** | 324 | 648 |
| degrees | **3–10**, mean 6.44 | 3–4 | 3 |
| λ₁⁺ | **0.2203** | 3 | 4 |

So the answer to "smooth connectivity curve or codimension-1 lock" is a third thing again:
the N = 3 phase is *natively* well connected. Its vertex sharing does not depend on tuning a
parameter to a measure-zero set, which is exactly what the dimer family's u = 0 plane required.

### The N = 3 phase's algebraic fingerprint is C₃

The connectivity turns out to be structured, not merely abundant. The unit-distance graph
splits into **exactly 16 components at every system size** — 90 vertices or 3360, always 16,
each growing with the block, and all 16 reaching the deep interior of an 8³ tiling. They are
not boundary fragments: they are **sixteen interpenetrating infinite networks**.

Reading their symmetry needs the same care the honeycomb did. A rectangular block in
hexagonal coordinates destroys the three-fold symmetry, and so does a cut that selects whole
*cells* — rotation carries the 1b monomer of cell `(0,0)` into cell `(-1,-1)`, so an
invariant set of cell *positions* is still not an invariant set of tetrahedra. Selecting
individual tetrahedra by distance from the axis is exactly invariant, and that is what
`build_n3_cluster` does. Cutting a symmetric structure with an asymmetric boundary destroys
precisely the degeneracies you are looking for.

With a genuinely C₃-invariant cluster, the decomposition is exact:

| network type | count | multiplicities |
| --- | --- | --- |
| **fixed by C₃** | 4 | **{1, 2} only** — exactly C₃'s real irrep dimensions |
| in a 3-orbit | 12 (4 orbits) | **{1} only** — no internal symmetry; the orbit supplies ×3 |

C₃ acts on the sixteen networks with orbit structure **4 × (fixed) + 4 × (size 3) = 16**, and
the whole spectrum is the sum of those pieces: multiplicities {1, 2, 3, 6}, nothing else. A
network the rotation *moves* cannot be symmetric, and its spectrum is correspondingly simple;
a network it *fixes* carries C₃ and shows singlets and doublets, which over the reals is the
complete list of C₃ irrep dimensions.

So the fingerprint is real but modest: **C₃, cyclic of order 3**. Set against the honeycomb's
`O_h` of order 48 with dimensions {1, 1, 2, 3, 3}, this is a much smaller group — and, as
there, a *finite* one. Nothing in either structure supports a continuous symmetry.

One artefact worth naming, since the script would otherwise report it as signal: with an
**even** number of layers the sixteen networks pair into isomorphic partners and every
multiplicity doubles, giving {2, 4, 6}. That is a property of the finite cluster, not the
crystal. Use an odd `layers`.

## The P3 orbit family (the ansatz that did not work)

The best P3₂ packing has **no vertex sharing at all**: 648 tetrahedra give 2592 raw vertices
and 2592 distinct ones, a compression ratio of exactly 1.0. The graph is 648 disjoint `K₄`,
every degree 3, spectrum `{0, 4, 4, 4}` repeated, λ₁⁺ = 4.

That places the three families on a clean ladder of decreasing connectivity:

| structure | sharing | components | λ₁⁺ |
| --- | --- | --- | --- |
| FCC honeycomb | vertices *and* edges | 1 (connected) | 0.3066 |
| CEG dimers, u = 0 | faces, plus inter-dimer contacts | extended networks | 0.033–0.28 |
| CEG dimers, u ≠ 0 | faces only | 1 per dimer (`K₅−e`) | 3 |
| **P3 screw packings** | **none** | **1 per tetrahedron (`K₄`)** | **4** |

So for the three-fold family the answer to "smooth curve or codimension-1 lock" is *neither*:
connectivity is simply **absent**, with no parameter region producing contacts at all. The
caveat is real, though — this is the packing the search found at φ = 0.537–0.594, not the
φ = 2/3 phase, and the true one could behave differently.

## The overlap test is the whole cost, so it got rebuilt

Every Monte Carlo move calls the periodic overlap check once, so the search speed *is* the
speed of that function. Profiling — not guessing — drove three changes worth **13.7×** end
to end (`_p3_search(400)`: 4.59s → 0.33s; the test suite went 43s → 7.1s):

| change | effect |
| --- | --- |
| per-pair image boxes | ~730 global images × 9 ordered pairs → 6 unordered pairs × ~125 |
| staged SAT + fast cross/matmul | 8 face normals first; 36 edge crosses only for survivors |
| compiled kernel (numba) | early exit, zero temporaries — the remaining 5× |

The first change is a correctness improvement too. The old design sized **one global image
box** to cover the whole cell, so it depended on how far apart the cell's contents were —
which is what made a spread-out configuration either uncertifiable or, in the version before
that, silently mis-certified. Centring each pair's box on the shift that pair actually needs
removes the dependence completely: validity is now provably invariant under translating any
particle by whole lattice vectors, and that invariance is tested at ±13 cells.

Profiling also showed where *not* to optimise. After the first two changes, further
vectorisation stopped paying — at 6 pairs × 125 candidates, NumPy's per-call dispatch
overhead dominates the arithmetic outright. That is what the compiled kernel is for, and why
it is worth a dependency.

**numba is optional.** When it is absent, `HAVE_NUMBA` is `False` and the array
implementation runs instead. The two are cross-checked on random configurations, and both
were validated against a deliberately naive brute-force reference enumerating a 15³ image
range — 0 disagreements over 119 configurations, with both verdicts exercised.

## Tests

```bash
python -m pytest test_tetra_spectral.py -v
```

Coverage includes closed-form checks (K4's Laplacian spectrum is exactly {0, 4, 4, 4};
tetrahedron volume and circumradius), the FCC kissing number, sparse/dense/block solver
agreement, kernel dimension versus component count, rejection of degenerate periodic
cells, and the structural claims themselves: the honeycomb's multiplicities never
exceeding 3 with triplets dominating (`O_h`), the CEG spectrum being exactly
{0, 3, 5, 5, 5} per dimer with V, φ and non-overlap all matching the paper, and
connectivity localising to the plane u = 0.

Two groups exist specifically to keep the project honest:

- `TestPeriodicOverlapRegression` pins the wrapping bug — particles inside the cell,
  image spans wide enough for the fractional offset, results surviving an *independent*
  tiling check rather than the image-range reasoning that failed, and `build_dense_packing`
  refusing to return a non-packing.
- `TestZModuleRank` includes a **positive control**: icosahedron vertices over `[1, φ]`
  must return rank 6. Without it, the lift test could only ever say no, and would pass
  even if it were broken.
- `test_compiled_and_array_paths_agree` checks the numba kernel against the NumPy fallback
  and asserts both verdicts actually occur, so it cannot pass by trivially agreeing on
  "everything is valid".
