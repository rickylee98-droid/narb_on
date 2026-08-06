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
not 3. (The quotient graph later pins the exponent down exactly: those components are
two-dimensional **sheets** — see "Why exactly 16".)

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
| `amplituhedron.py` | Exact `k = 1` amplituhedron tilings: integer SAT in `R^m`, enumeration, flip graph |
| `amplituhedron_analysis.py` | CLI for the amplituhedron pipeline |
| `test_tetra_spectral.py` | Tetrahedron test suite (300 tests) |
| `test_amplituhedron.py` | Amplituhedron test suite (116 tests) |

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

## Finishing the searches: what constrained optimisation rescues

`refine_packing` generalises the N = 2 result — minimise cell volume subject to every
contact depth staying at or below zero, rebuilding the active set each round. Applied
across every family here, the outcome is **not** uniform, and the pattern is the
interesting part:

| family | parameters | MC plateau | refined | |
| --- | --- | --- | --- | --- |
| N = 2 double lattice | 12 | 0.715488 | **0.7194880889** | hits the published φ₂ |
| P3 orbit family | 7 | 0.595562 | **0.5968995** | new family optimum |
| trimer, free lattice | 13 | 0.5242 | 0.5176 | no gain on the best |
| N = 4 general ASC | 37 | 0.6055 | 0.6055 | **exactly zero change** |

**A claim from earlier in this work does not survive.** After the N = 2 success I wrote
that "every one of those plateaus is probably an MC artifact." That is wrong. Refinement
rescues a search that landed in the *right basin* and merely could not finish — N = 2 and
P3. Where the search landed in a poor basin, the plateau is a genuine local optimum:
on the N = 4 cell SLSQP finds no feasible descent direction at all, because 147 contacts
have jammed 37 parameters. A local method cannot move between basins, and 0.6055 versus
the true 4000/4671 is a different structure, not an unfinished one.

The P3 result is worth stating separately because it strengthens an earlier conclusion
rather than weakening it. Both screw groups, from many independent seeds, converge to the
same **φ = 0.5968995291** (a = 0.86399, c = 0.91623). That is the single-orbit family's
true optimum, and it is still **10.5% below 2/3** — so the N = 3 phase genuinely is not a
single C₃ orbit, and the shortfall was never a search artefact. No low-height closed form
was found for these parameters; the recogniser offers candidates with large height and
residuals sitting exactly at the numerical precision, which is overfitting, and unlike the
N = 2 determinant there is no independent target to check them against. Reported as
numerical.

Two bugs surfaced while building this, both worth naming:

- `refine_packing` originally took its starting volume on trust, so an infeasible start was
  reported at the *search's* density while the configuration actually interpenetrated. It
  now refuses an overlapping start outright.
- `PackingResult` did not carry its own parameters, forcing callers to recover orientations
  by Kabsch alignment — and my recovery was transposed, silently producing overlapping
  configurations that looked fine. Searches now return the parameters they used.

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

### Why exactly 16: the quotient graph with voltages

"16 at every block size" is an observation, not an explanation, and a finite block can never
supply one — its component count mixes genuine networks with pieces the boundary happened to
sever. The infinite graph answers it exactly, and cheaply, via `periodic_graph_components`.

Collapse the structure onto a single unit cell, so vertices become **orbits** under lattice
translation, and label each edge with the translation it crosses — its *voltage*. Fix a
spanning tree, and every remaining edge closes a cycle carrying a net voltage in `Z³`. Those
voltages generate a subgroup `L ⊆ Z³`, and a quotient component unrolls into exactly
**`[Z³ : L]`** infinite networks. The intuition is direct: if every cycle in a component sums
to an *even* translation, no path inside that component can ever reach an odd cell, so the odd
cells must be served by a different copy.

For the N = 3 phase:

| | value |
| --- | --- |
| raw vertices per cell | 12 (3 tetrahedra × 4) |
| distinct orbits | **9** — three vertex pairs coincide under translation |
| quotient components | **2**, of 4 and 5 orbits |
| cycle-voltage lattice, each | **`2Z³`** — Hermite basis `(2,2,0), (0,2,0), (0,0,2)`, Smith invariants `[2,2,2]` |
| index, each | **8** |

So the count factors, and the factors are meaningful:

> **16 = 2 × 2³**

Two vertex families that never touch each other, each closing its cycles only on even
translations, hence each interleaving into one copy per **parity class** of `Z³`. Not sixteen
arbitrary pieces — two structures, each eight-fold interpenetrating. The arithmetic is exact
integer Hermite reduction (`sublattice_index`), with no floating-point determinant that could
round a singular matrix into an invertible one, and no symbolic dependency. It agrees with
direct `build_unit_distance_graph` counts at every block size tested, and is independent of
the translation-search span.

The same machinery sharpens an earlier claim about the dimer family, and not in its favour.
The **rank** of the voltage lattice is the *dimensionality* of a component: rank 0 means it
closes into a bounded cluster, rank 1 an infinite chain, rank 2 an infinite sheet, rank 3 a
network filling all three directions. Only rank 3 yields a finite component count.

| structure | quotient components | rank | what a component actually is |
| --- | --- | --- | --- |
| CEG `optimal` | 2 | **0** | bounded dipyramids — infinitely many, one per dimer |
| CEG `torquato-jiao` | 2 | **0** | same |
| CEG `kallus-elser-gravel` | 1 | **2** | an infinite **sheet** |
| CEG `densest-connected` (u = 0) | 1 | **2** | an infinite **sheet** |
| **N = 3 phase** | 2 | **3** | genuine 3D networks — **16** of them |

The u = 0 connectivity plane does not produce a three-dimensional network. It produces
**stacked two-dimensional layers with no bonds between them**, which a finite block confirms
directly: tiling `densest-connected` into an `L³` block gives exactly `L` components, one per
layer, growing without bound. Earlier this README called those "extended networks"; the honest
description is *extended within a plane*. The N = 3 phase is the only structure here whose
connectivity is genuinely three-dimensional.

## The P3 orbit family (the ansatz that did not work)

The best P3₂ packing has **no vertex sharing at all**: 648 tetrahedra give 2592 raw vertices
and 2592 distinct ones, a compression ratio of exactly 1.0. The graph is 648 disjoint `K₄`,
every degree 3, spectrum `{0, 4, 4, 4}` repeated, λ₁⁺ = 4.

That places the three families on a clean ladder of decreasing connectivity:

| structure | sharing | components | λ₁⁺ |
| --- | --- | --- | --- |
| FCC honeycomb | vertices *and* edges | 1 (connected) | 0.3066 |
| CEG dimers, u = 0 | faces, plus inter-dimer contacts | 2D sheets, one per layer | 0.033–0.28 |
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

## A second target: exact tilings of the k = 1 amplituhedron

The same three tools — exact arithmetic, a separating-axis overlap test, a graph
Laplacian — transfer to the amplituhedron, but only where the geometry is genuinely
polytopal. Three things had to be corrected before any code was written.

**The tree-level `m = 4` tiling problem is not open.** BCFW cells were proven to tile the
`m = 4` tree amplituhedron (Even-Zohar–Lakrec–Parisi–Sherman-Bennett–Tessler–Williams).
What remains open is the *loop* case, and *counting* tilings. This module targets counting.

**SAT does not apply at `k ≥ 2`.** Those tiles are images of positroid cells: curved
semialgebraic sets, not polytopes. There is no separating hyperplane to find, so a
"generalised SAT" there would return confident nonsense.

**"Multiplicities matching a Yangian symmetry" is not well posed.** The Yangian is
infinite dimensional and does not act on a finite tiling complex.

What *is* exactly true is that at `k = 1` the amplituhedron is a polytope:

> **A(n, 1, m) = C(n, m)**, the cyclic polytope.

A point of `Gr₊(1,n)` is a positive row vector `c`, so `Y = cZ` is a positive combination
of the rows of `Z`, and positivity of `Z`'s minors says exactly that those rows sit in
convex position like points on the moment curve. Hence **tilings of `A(n,1,m)` are
triangulations of `C(n,m)`** — integer vertices, integer volumes, no floating point
anywhere in the pipeline.

### Everything is pinned against classical results

The module is never allowed to check itself. Every number it produces has an external
referee that it has no way to know about:

| Claim | Independent referee | Result |
| --- | --- | --- |
| facets of `C(n,m)` | Gale's evenness condition, and McMullen's upper-bound theorem | agree |
| volume of `C(n,2)` | shoelace formula | agree |
| volume, general route vs facet route | two independent algorithms | agree, 7 cases |
| volume of `Δ(k,n)` | Eulerian number `A(n-1,k-1)` (Laplace) | agree, 7 cases |
| tilings of `A(n,1,2)` | Catalan `C(n-2)` | 2, 5, 14, 42, 132 — exact |
| flip graph of `A(n,1,2)` | associahedron 1-skeleton | `(n-3)`-regular, right edge count |
| flip-connectivity | Rambau's theorem | confirmed, never assumed |

Flip-connectivity is worth singling out. The enumeration never uses flips — it is an
exhaustive include/exclude search over exact geometry — so finding the flip graph connected
afterwards is a real check on both the enumeration and the flip criterion, not a tautology.

### The physical case, m = 4

| | n=6 | n=7 | n=8 | n=9 | n=10 |
| --- | --- | --- | --- | --- | --- |
| candidate simplices | 6 | 21 | 56 | 126 | 252 |
| **tilings** | **2** | **7** | **40** | **357** | **4824** |
| simplices per tiling | 3 | 6 | 10 | 15 | 21 |
| flip graph edges | 1 | 7 | 64 | 825 | 15120 |
| flip degrees | 1 | 2 | 3–4 | 4–7 | 5–10 |
| connected | yes | yes | yes | yes | yes |

Every tiling has exactly `binom(n-3, 2)` simplices — the count is constant across the whole
tiling space, so tilings differ in shape but never in size. And unlike `m = 2`, the `m = 4`
flip graph is **not regular**: at `n = 9` its degrees run 4 to 7. Tilings genuinely differ
in how many flips they admit.

### The symmetry is dihedral, and the degeneracies are exact

Replacing the ill-posed Yangian test with the group that demonstrably does act:

> **|Aut(flip graph)| = 2n**, exactly, for `n = 5,6,7,8` at `m = 2` and `n = 7,8,9` at
> `m = 4`. That is the dihedral group `D_n`: the cyclic symmetry of the amplituhedron
> together with reversal of the moment curve.

One case is deliberately excluded, and it is worth stating rather than hiding. `A(6,1,4)`
has only **2** tilings, so its flip graph is `K₂` and its automorphism group has order 2,
not 12. `D_6` still acts on those two tilings — it simply cannot act *faithfully* on a
two-element set, so the graph cannot see most of the group. The rule is that `|Aut|` equals
`2n` once there are enough tilings for the action to be faithful; `n = 6` at `m = 4` is
below that threshold. The test suite parametrisation excludes exactly this case.

Two multiplicities exceed anything `D_n` can force — its real irreps are only dimensions 1
and 2 — so they were checked from the **characteristic polynomial**, not from a tolerance:

| configuration | eigenvalue | multiplicity | status |
| --- | --- | --- | --- |
| `A(8,1,2)` | **6** | **8** | exact integer |
| `A(8,1,4)` | **3** | **6** | exact integer |

These are real, not artefacts — and they are *not* explained by the symmetry group, since
`2n = 16` has no 6- or 8-dimensional irrep. The next section pins that down exactly.

### Forced degeneracy versus accidental coincidence

A multiplicity is only evidence of anything when the eigenspace carries a **single**
irreducible representation of the symmetry group. If several unrelated representations
happen to land on one eigenvalue, the multiplicity means nothing. Eigenvalue counting alone
cannot tell these apart — but character theory can, exactly:

> For the character `χ` of the group acting on an eigenspace, `⟨χ,χ⟩ = Σ mᵢ²`.
> So **`⟨χ,χ⟩ = 1` iff the degeneracy is forced by the symmetry.**

`tetra_spectral.eigenspace_symmetry` implements this, with both controls:

| graph | λ | dim | ⟨χ,χ⟩ | forced? |
| --- | --- | --- | --- | --- |
| `K₄` | 4 | 3 | 1 | **yes** |
| Petersen | 2 | 5 | 1 | **yes** |
| cube `Q₃` | 2 | 3 | 1 | **yes** |
| `K₄ ⊔ K₂,₂` (negative control) | 4 | 4 | 2 | no |
| `A(8,1,4)` | 2 | 1 | 1 | **yes** |
| **`A(8,1,4)`** | **3** | **6** | **4** | **no** |
| **`A(8,1,2)`** | **6** | **8** | **8** | **no** |

The negative control matters as much as the positive ones: without it the test could only
ever say yes, and would pass even if it were broken.

So the verdict on the two big degeneracies is negative and exact. For `λ = 6` the character
is `8` at the identity, **`−8` at exactly one element**, and `0` at the other fourteen. A
character of `−dim` means that element acts as exactly `−I`, so the eigenspace sits entirely
in the odd part of the central rotation `r⁴` — the shift by `n/2` along the moment curve.
That is genuine structure, and `⟨χ,χ⟩ = 8` still says it is two 2-dimensional irreps
appearing twice each, not one 8-dimensional one. `D_n` accounts for at most 2 of the 8.

### Three explanations that were tested and failed

Having ruled out symmetry, the natural next question is what *does* produce them. Three
mechanisms were tried and none survived, which is worth recording rather than quietly
dropping:

| hypothesis | mechanism | verdict |
| --- | --- | --- |
| **twin vertices** | twins force λ = degree (or degree+1 when adjacent), and 5+1 = 6 exactly | **refuted** — the flip graphs have no twin pairs at all, of either kind |
| **degree-3 independent set** | an independent set of degree-`d` vertices forces λ = `d` | **refuted** — `A(8,1,4)` has 32 degree-3 vertices but they are *not* independent, and the incidence rank predicts multiplicity 25, not 6 |
| **perfect matching** | a matched pair antisymmetric in a `d`-regular graph forces λ = `d+1` | **refuted** — the minimum-support vector found for `A(8,1,2)` has support 104 of 132 and induces 208 edges, nothing like a matching |

What did survive is a localisation result, and only on the `m = 4` side. For `A(8,1,4)`,
**16 of the 40 coordinates are identically zero** across the whole `λ = 3` eigenspace — no
eigenvector touches those 16 tilings — and the minimum-support eigenvector has support
exactly **8**, on a set inducing **zero edges**. An independent set of degree-3 vertices does
force λ = 3, so that vector is explained. The remaining multiplicity is not.

`A(8,1,2)` shows the opposite behaviour: its `λ = 6` eigenvectors are **delocalised**, spread
over at least 104 of the 132 tilings. Whatever produces multiplicity 8 there is global.

**Bottom line: these degeneracies are real, exact, and unexplained.** They are not symmetry
fingerprints, which is the claim worth being careful about — the `O_h` and `C₃` results
earlier in this project *are* forced degeneracies, and conflating the two would be the
easiest way to overstate what any of this shows.

```bash
# Calibration: must reproduce Catalan
python amplituhedron_analysis.py --n 8 --m 2 --automorphisms

# The physical case, with symbolic eigenvalues
python amplituhedron_analysis.py --n 8 --m 4 --exact-spectrum --automorphisms

# The hypersimplex, refereed by an Eulerian number
python amplituhedron_analysis.py --hypersimplex 2,5 --volume-only
```

### The overlap test, again profiled rather than guessed

Generalising SAT to `R^m` needs the right axis set. The completeness argument is that the
origin misses the interior of `P + (-Q)`, whose facets are sums `F + (-G)` of faces with
`dim F + dim G = m - 1`. A first implementation enumerated `(m-1)`-subsets of the combined
*edge* directions — a valid superset, but a loose one.

Profiling (not guessing) showed ~435 axes tried per pair. A first fix — caching per-simplex
facet normals — produced **no measurable speedup**, because those are only 10 of the 435.
Enumerating **face pairs** directly, which is what the completeness argument actually says,
cuts `R^4` from 910 axes to 210 and gives:

| | before | after |
| --- | --- | --- |
| `A(8,1,4)` full enumeration | 19.0s | **4.1s** |
| `Δ(2,5)` volume | 127.6s | **33.2s** |

Verdicts were confirmed identical on all 2,535 simplex pairs across four configurations
before the speedup was accepted. In `R^3` the face-pair set reduces to exactly the familiar
4 + 4 face normals and 36 edge crosses; in `R^2` to edge normals alone.

## Tests

```bash
python -m pytest test_tetra_spectral.py -v      # tetrahedron packings
python -m pytest test_amplituhedron.py -v       # amplituhedron tilings
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
