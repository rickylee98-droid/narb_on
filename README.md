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
| `bootstrap.py` | 2d conformal bootstrap: blocks, crossing, exclusion functionals |
| `bootstrap_analysis.py` | CLI for the bootstrap |
| `test_amplituhedron.py` | Amplituhedron test suite (116 tests) |
| `arithmetic_que.py` | LPS Ramanujan graphs, Hecke operators, thin-set equidistribution |
| `test_bootstrap.py` | Bootstrap test suite (60 tests) |
| `test_arithmetic_que.py` | Arithmetic QUE test suite (84 tests) |
| `spinfoam.py` | Exact Wigner 3j/6j, tetrahedron geometry, Ponzano-Regge limit |
| `test_spinfoam.py` | Spin-foam test suite (45 tests) |
| `fractal_stokes.py` | Hölder forms on Koch curves: resolution window, coherence exponent, (d, α) sweep |
| `test_fractal_stokes.py` | Fractal Stokes test suite (174 tests) |
| `selberg.py` | Ihara zeta, prime geodesics, graph trace formula, RH analogue |
| `test_selberg.py` | Selberg test suite (142 tests) |
| `detection.py` | Community detection via the non-backtracking spectrum |
| `test_detection.py` | Detection test suite (59 tests) |
| `adm.py` | Einstein constraint rank loss, KIDs, linearisation instability |
| `test_adm.py` | ADM test suite (154 tests) |
| `fermat_hodge.py` | Shioda's reduction of the Hodge Conjecture; phi(m) via Hilbert bases |
| `test_fermat_hodge.py` | Fermat/Hodge test suite (78 tests) |

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
`2n = 16` has no 6- or 8-dimensional irrep. The next section pins that down exactly, and the
one after it **solves the `m = 4` case**.

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

### Solved: λ = 3 in `A(8,1,4)` is a degree eigenvalue

The clue was the localisation. The eigenspace lives on exactly the **24 degree-3 vertices**,
and `λ = 3` **is** their degree. That turns out to be the whole mechanism.

Let `S` be the set of degree-`d` vertices. A vector `v` vanishing off `S` satisfies `Lv = dv`
exactly when

> `Σ_{w ∈ N(u) ∩ S} v_w = 0` for **every** vertex `u` — inside `S` and outside it alike.

So the multiplicity is `|S| − rank(M)`, with `M` the `|V| × |S|` incidence matrix. My earlier
refuted attempt imposed the condition only at vertices *outside* `S`, which predicts **25**
where the answer is **6** — the error that made this look unexplainable.

| configuration | `\|S\|` | predicted | observed | |
| --- | --- | --- | --- | --- |
| `A(8,1,4)`, d=3 | 32 | **6** | **6** | ✅ |
| `A(8,1,4)`, d=4 | 8 | **1** | **1** | ✅ (a level I had not even looked at) |
| `A(7,1,4)`, d=2 | 7 | 0 | 0 | ✅ |
| `A(9,1,4)`, all degrees | — | 0 | 0 | ✅ |

The zeros matter as much as the hits: a mechanism that merely reproduced whatever
multiplicity was present would be worthless. And this is **containment, not just matching
dimensions** — the observed eigenvectors vanish identically off `S`, checked exactly, so the
constructed space *is* the eigenspace.

The rank is taken modulo two large primes rather than numerically, because the entire content
of the answer is a rank deficiency and a floating-point rank would need a threshold to see one.

### Still open: λ = 6 in `A(8,1,2)`

The same mechanism cannot touch it, and the reason is clean: `A(8,1,2)` is **5-regular**, so
`S` is every vertex, `M` is the adjacency matrix, and the prediction collapses to the nullity
of `A` — i.e. tautologically the multiplicity of `λ = 5`. The anomaly sits at `λ = 6`, one
above the degree, where nothing here applies. It is also **sporadic**: across `n = 6,7,8,9`
at `m = 2` the integer levels are `{0,1,3,4}`, `{0}`, `{0,6}`, `{0}` — no pattern.

So one of the two is solved and one is not, and the boundary between them is a stated
property rather than a gap.

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

## A third target: the conformal bootstrap in two dimensions

Same scoping discipline, three corrections before any code:

**The 3d Ising island is out of reach here, and saying otherwise would be dishonest.**
It needs mixed correlators, a semidefinite program enforcing polynomial positivity in `Δ`
over a continuum, and an arbitrary-precision SDP solver (SDPB). None of that is a weekend's
work on top of this repo.

**A separating-axis engine is the wrong tool.** "Testing for the existence of positive
linear functionals" is linear/semidefinite programming, not SAT. There is no geometry to
separate.

**But `d = 2` has exactly the phenomenon of interest.** The bound has a kink at the 2d Ising
model, whose dimensions are known *exactly*: `Δσ = 1/8`, `Δε = 1`. That is a boundary point
locking onto exact algebraic values, and it is checkable — the same role Catalan numbers
play for the amplituhedron.

### The blocks are refereed by the Casimir equation

In `d = 2` the conformal block has a closed form that factorises completely, so derivatives
at the crossing-symmetric point come from one-variable Taylor series with a bounded tail —
no multivariate numerical differentiation anywhere. The correctness check is independent of
the implementation: a block *is* an eigenfunction of the quadratic Casimir, and

> residuals come out at **1e-31** across `Δ ∈ {0.25 … 7}`, `ℓ ∈ {0, 2, 4, 6}`.

The float64 fast path (which carries the whole computation) is checked against a 40-digit
mpmath path: agreement to **1.3e-15**, i.e. six times machine epsilon.

### The result

| Δσ | bound on Δε | local slope |
| --- | --- | --- |
| 0.0625 | 0.3428 | — |
| 0.0900 | 0.6090 | 9.7 |
| 0.1100 | 0.8425 | 11.7 |
| **0.1250** | **1.0023** ← 2d Ising, exact value **1** | **10.7** |
| 0.1400 | 1.0533 | **3.4** |
| 0.1700 | 1.1478 | 3.1 |
| 0.2200 | 1.3010 | 3.1 |

Two things to read off. First, a 15-component functional lands within **0.23%** of an
exactly known dimension. Second, and more to your original question about boundaries
locking onto special values: **the slope drops by a factor of ~3.4 precisely at Δσ = 1/8**
and then stays flat. That discontinuity in slope is the kink, and the 2d Ising model sits
at it — the bound is not a smooth curve that happens to pass near a physical theory, it
changes character there.

```bash
python bootstrap_analysis.py --delta-phi 0.125            # the calibration point
python bootstrap_analysis.py --delta-phi 0.125 --gap 1.4  # is one assumption excluded?
python bootstrap_analysis.py --scan 0.0625,0.45,9 --csv bound.csv
```

### Two bugs that a less suspicious pipeline would have shipped

**A single-shot linear program produces functionals that are not exclusion proofs.** The LP
satisfies its own 660 sampled constraints and then dips to about **−1e-3 between them**. An
exclusion argument needs `α[F] ≥ 0` for *every* operator, so a functional that goes negative
anywhere proves nothing at all. The failure is completely silent — the LP reports success.
The fix is an audit grid the program never trains on, with violations fed back as new
constraints until nothing on it is negative.

**Feasibility-with-a-tolerance is the wrong question, and it corrupted the bisection.** The
audit loop would stall near margin `−3e-8` while the constraint set ballooned to 18,260 rows,
so a gap got reported "not excluded" while gaps on *both sides of it* were excluded. That
breaks monotonicity, and bisection on a non-monotone predicate returns garbage: it put the
bound at `Δσ = 0.0625` at **1.85** when the answer is **0.34**.

The fix is to stop asking a yes/no question at the boundary. Maximising the *worst value* the
functional takes gives a quantity that varies continuously with the assumed gap, so its
**sign** is a well-posed, monotone predicate — no tolerance in the decision at all. A test
now pins the monotonicity directly, because that property is what the whole bisection rests
on.

Caching the constraint rows (every spin-`ℓ ≥ 2` sample sits at its own unitarity bound and is
therefore identical at every gap) took a bound from **217s to 27s**, same answer.

### What this is, precisely

An exclusion is a proof **against the sampled spectrum**, verified on an independent dense
grid. It is not the continuum statement a semidefinite program would give. And the converse
direction is not evidence of anything: failing to find a functional means this derivative
basis cannot rule a theory out, not that the theory exists. The CLI prints that caveat rather
than burying it.

## A fourth target: quantum unique ergodicity on thin sets

Scoping corrections first, as before.

**AQUE on the modular surface is a theorem, not an open problem.** Lindenstrauss settled
the compact and Hecke cases, Soundararajan completed the modular surface. What is open is
the *thin set* case — restriction to a lower-dimensional or sparse subset.

**Computing real Maass forms needs Hejhal's algorithm**, a specialist hyperbolic-PDE method,
not a sparse eigenproblem.

So this uses the standard discrete model: **Lubotzky–Phillips–Sarnak Ramanujan graphs**
`X^{p,q}`, Cayley graphs of `PSL(2,F_q)` or `PGL(2,F_q)` built from the four-square
parametrisation. Exact integer arithmetic mod `q`, and — crucially — an exact theorem to
check against.

### The construction, refereed

| Claim | Referee | Result |
| --- | --- | --- |
| `p+1` normalised four-square solutions | Jacobi's theorem | exact for p = 5…41 |
| generator determinants | must equal `p` mod `q` | exact |
| vertex count | `\|PSL(2,F_q)\|` or `\|PGL(2,F_q)\|`, chosen by the Legendre symbol | exact |
| **non-trivial eigenvalues** | **Ramanujan bound `\|λ\| ≤ 2√p`** | holds, and tightly |

`X^{5,29}`: largest non-trivial `|λ| = 4.44202` against a bound of `4.47214`. That is a real
Ramanujan graph, not something that merely compiles.

A bug worth recording: quotienting the matrices by only `±1` instead of by *all* scalars
gives `X^{5,13}` **4368 vertices instead of 2184**, and its largest non-trivial eigenvalue
is **5.677 — violating the Ramanujan bound**. The theorem caught it immediately. Without
that check the graph would have looked perfectly healthy.

### The obstruction: this model cannot answer the question, and that is a theorem

The naive experiment is ill-posed, and the obvious repair is vacuous. Both are *measured*
here, not asserted:

1. **Per-eigenvector mass is basis-dependent.** These eigenspaces reach multiplicity **81**
   at `X^{5,13}`. Inside a degenerate eigenspace any orthonormal basis is as good as any
   other, so a "scar" can be manufactured by rotating the basis.

2. **The basis-free repair is constant by symmetry.** The spectral projector's diagonal
   `Π_λ(v,v)` is canonical — but a Cayley graph is vertex-transitive and `Π_λ` commutes with
   every automorphism, so that diagonal *cannot vary*. Measured spread: **< 1e-13**. The
   observable is identically 1 and carries no information whatsoever.

3. **Hecke operators do not rescue it.** This is what puts the *arithmetic* in AQUE, and the
   discrete Hecke operators are genuine: `[A_p, A_p'] = 0` **exactly**, in integer
   arithmetic, for distinct primes of the same quadratic character mod `q`. But they are
   right convolutions, so they commute with the entire *left* regular action, and their
   joint eigenspaces cannot drop below the group's irrep dimensions. On `PSL(2,F_13)`, three
   Hecke operators cut the maximum multiplicity only from **112 to 42**, and the joint
   multiplicities come out as exactly `{1} ∪ {12,13,14} × {1,2,3}` — the irrep dimensions.

> The homogeneity that makes LPS graphs a clean arithmetic object is exactly what blinds
> them to this question. The modular surface is *not* homogeneous, and that is the property
> the model fails to capture.

I built the whole pipeline before noticing this, ran the thin-set experiment, and got
deviations of 0.0009–0.004 that looked like a beautiful equidistribution result. They were
window-edge rounding noise on a quantity that is constant by symmetry.

### Where the question does have content

Random `d`-regular graphs keep the spectral gap (Friedman: almost-Ramanujan) and drop the
fatal homogeneity. Their spectra are **simple** — measured multiplicity 1 at every size
tested — so eigenvectors are canonical and `|ψ(v)|²` is well posed.

| model | `\|S\|` | r.m.s. deviation | Gaussian baseline `√(2/\|S\|)` | ratio |
| --- | --- | --- | --- | --- |
| random 6-regular, N=2000 | 10 | 0.4382 | 0.4472 | **0.98** |
| random 6-regular, N=2000 | 45 | 0.2047 | 0.2108 | **0.97** |
| random 6-regular, N=2000 | 205 | 0.0953 | 0.0988 | **0.97** |
| **barbell (negative control)** | 12 | 0.6206 | 0.4082 | **1.52** |

And the sharp statement — fix `|S| = 30` and grow the graph:

| N | 500 | 1000 | 2000 | 4000 |
| --- | --- | --- | --- | --- |
| r.m.s. deviation | 0.2486 | 0.2525 | 0.2544 | 0.2638 |

**Flat.** The fluctuation is governed by `|S|` alone, not by `N`, and it sits within a few
percent of the Gaussian value at every size. So in this model: thin-set mass equidistributes
exactly when `|S| → ∞`, at precisely the random-matrix rate — **no scarring, and no
arithmetic enhancement either**. The barbell control confirms the statistic can detect
scarring when it is there (ratio 1.5), so the null result is a measurement rather than a
blind spot.

### Pulling the thread: a model that is Ramanujan *and* inhomogeneous

The obstruction says exactly what to give up. The modular surface is a **quotient** `Γ\ℍ`,
not a group — so the faithful discrete analogue is a **Schreier graph**, not a Cayley graph.
Taking the LPS generators acting on the projective line `P¹(F_q)` gives one, and all three
properties that matter survive:

| property | why it survives | measured |
| --- | --- | --- |
| **Ramanujan** | the Schreier operator is the *restriction* of the Cayley one to stabiliser-invariant functions, so its spectrum is a sub-multiset | `4.3861 ≤ 4.4721` at q=509 |
| **Hecke structure** | `[A_p, A_p'] = 0` is a group-algebra identity, so it survives any action | all six commutators **exactly 0** |
| **Inhomogeneity** | right multiplication no longer acts by automorphisms — it would have to normalise the generator set | projector spread **order 1**, versus `1e-13` on the Cayley graph |

Ramanujan comes for free, with no new proof. And the graph has `q+1` vertices rather than
`~q³`, so `q` reaches the thousands cheaply.

**The null model is what makes the answer real.** The raw projector spread *grows* with the
graph — 1.93 → 4.11 as q goes 101 → 2053 — which looks like scarring and is not. The maximum
is simply taken over more vertices. Comparing against a uniformly random subspace of matched
dimension absorbs that entirely:

| q | n | observed | random-subspace null | **ratio** |
| --- | --- | --- | --- | --- |
| 101 | 102 | 1.9286 | 4.0407 | **0.477** |
| 257 | 258 | 2.8590 | 4.7738 | **0.599** |
| 509 | 510 | 3.1869 | 5.2846 | **0.603** |
| 1013 | 1014 | 3.6675 | 5.7814 | **0.634** |
| 2053 | 2054 | 4.1051 | 6.3084 | **0.651** |

**Ratio below 1 at every size**: the eigenspace mass is *more* equidistributed than a random
subspace, not less. That is a genuine QUE-type positive result — and unlike the Cayley case
it is a measurement rather than an identity, because the observable is no longer frozen by
symmetry. Reading the raw spread alone would have suggested the exact opposite conclusion.

## A fifth target: the semiclassical limit of spin networks

Two corrections to the usual framing, before any code.

**The 10j symbol's uniform-scaling asymptotics is not open.** Baez–Christensen–Egan showed
the Barrett–Crane vertex is dominated by *degenerate* configurations, so the oscillatory term
carrying the Regge action is subdominant and the amplitude does **not** approach classical
geometry. Monitoring that phase numerically means chasing a signal known to be buried.

**The model that does work is EPRL/FK** (Barrett–Dowdall–Fairbairn–Gomes–Hellmann), which
needs `SL(2,C)` booster functions — a different and much heavier computation.

What *is* exactly computable and genuinely about convergence to classical geometry is the
**Ponzano–Regge** limit of the `6j` symbol:

> `{6j} ~ cos(S_R + π/4) / √(12πV)`,  `S_R = Σ lᵢθᵢ`,  `lᵢ = jᵢ + ½`

The Regge action — the classical action of discrete gravity — appears explicitly in the phase.

### Exactness and referees

The Racah formula gives `{6j} = S·√R` with `S, R` rational, so a symbol is stored as that
exact pair. Spins are doubled internally, so half-integers never produce half-integer
factorials. The engine is refereed against identities it cannot fake:

| referee | result |
| --- | --- |
| sympy cross-check (3j and 6j) | agree to `1e-14` |
| vanishing when a triad fails | 0 violations |
| orthogonality relation | 0 failures |
| Regge symmetry | exact, to the last bit |
| **Biedenharn–Elliott pentagon** | 0 failures, worst `5.6e-17` |

### The measurement

| shape | α | trustworthy |
| --- | --- | --- |
| (1,1,1,1,1,1) | −1.095 | yes |
| (2,2,2,3,3,3) | −1.097 | yes |
| (3,3,3,2,2,2) | −1.055 | yes |
| (2,3,4,4,3,2) | −1.071 | yes |
| (2,3,3,4,4,5) | −1.085 | yes |
| (3,4,5,5,4,3) | −1.109 | yes |
| (3,5,4,2,4,6) | +0.013 | **no — aliased** |

The envelope-normalised error `|exact − PR|·√(12πV)` scales as **λ^(−1)**, and this survives
**non-uniform** edge scaling. In the classically forbidden regime the symbol instead decays
**exponentially with no sign change**: for (1,3,4,2,2,2) an exponential fit beats a power law
by 14× in residual.

### Two bugs, one caught by refusing to average away an outlier

**The edge map was wrong.** The four triads of a `6j` symbol must be the four *faces*. I had
`(j1,j2,j3)` on the three edges meeting at one vertex — a vertex star. That reproduces correct
asymptotics on every symmetric shape (the regular tetrahedron, anything palindromic), so it
passed the first checks; it fails only on shapes like `(2,2,2,3,3,3)`, which showed up as a
lone α ≈ 0 outlier. There is now a structural test asserting triads = faces, and the fix
changed which shapes are classically forbidden — an earlier decay result had to be redone.

**`(-1)**n` returns a float for negative `n`.** Python evaluates `(-1)**-1` as `-1.0`, silently
contaminating exact rational arithmetic. The `3j` sign exponent goes negative for ordinary
arguments. Replaced with a parity helper.

**And one artifact that is not a bug.** `(3,5,4,2,4,6)` gives α ≈ 0 no matter how far the scan
runs. Its Regge phase advances **6.99745 cycles per unit scaling** — 0.0026 from an integer, a
beat period of 391 — so integer spin sampling is nearly phase-locked and no envelope is
measurable. Well-behaved shapes sit 0.17–0.41 away. `aliasing_diagnostic` now detects this in
advance and `is_trustworthy` refuses to report the exponent rather than leaving a spurious
zero to be read as an absence of convergence.

## A sixth target: Stokes' theorem on jagged boundaries

`fractal_stokes.py` — line integrals of Hölder forms over Koch curves, and where Young's
bound actually bites.

### The question, sharpened

"Stokes' theorem fails on a fractal boundary" is, as usually stated, false for the case
people reach for. At any finite prefractal stage the boundary is a polygon, so Stokes holds
exactly; and for a *smooth* form the limit converges without difficulty, because the integral
of a differential form is not the integral with respect to arclength. The snowflake's
arclength diverges like (4/3)ⁿ while ∮ x dy converges to an area known in closed form. Both
are checked, the arclength against (4r)ⁿ to twelve figures and the area against the exact
finite-level formula A_n = T(1 + (3/5)(1 − (4/9)ⁿ)) at every level.

The failure is real at *limited regularity*. If the curve has finite p-variation with p equal
to its box dimension d, and the form is α-Hölder, Young's condition 1/p + α/p > 1 reads

> **α > d − 1**, so α_c = log 4 / log 3 − 1 = 0.26186 for the standard Koch curve.

The same estimate predicts a rate, not just a threshold: refining level n touches 4ⁿ segments
of length rⁿ, each contributing O(r^{n(1+α)}), so the increment is bounded by (4 r^{1+α})ⁿ.
Writing the measured decay as (4^β r^{1+α})ⁿ defines a **coherence exponent** β — Young's
bound is β = 1, and β = ½ is what independent signs would give. Young's bound is an upper
bound, so β is a measurement.

### The apex angle makes the dimension a dial

Replacing each segment by a symmetric bump with apex angle θ forces 2r + 2r cos θ = 1, so
r = 1/(2 + 2cos θ) and d = log 4 / log(1/r) sweeps continuously from 1 to 2. The measurement
is therefore over the (d, α) *plane*, not a line through one curve.

### The result: two regimes, and a threshold an order of magnitude below Young's

The primary instrument is `measure_increments`: the RMS refinement increment per level,
averaged in quadrature over independent phases, with a bootstrap interval on the tail ratio.
No model of the decay is assumed. For the standard curve (d = 1.262, α_c = 0.262):

| α | 0.02 | 0.05 | 0.15 | 0.262 = α_c | 0.35 | 0.45 | 0.55 | 0.65 | 0.95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tail ratio | 0.985 | 0.934 | 0.804 | 0.669 | 0.578 | 0.497 | 0.453 | 0.4445 | 0.4438 |

**Above α ≈ 0.52 the rate is 4r² = 0.4444 exactly and does not depend on α at all.** At
α = 0.95 the successive ratios are 0.445, 0.444, 0.445, 0.443, 0.444, 0.445, 0.443 across
seven refinements; the smooth control gives 0.444 to four places at every level. The curve's
own second-order geometry sets the rate, and the form's regularity stops mattering once it is
smooth enough. This half is exact and it is the solid result.

**Below that, the rate rises smoothly toward 1 with no feature whatsoever at α_c.** The low
branch fits ln(rate) = 0.017 − 1.607 α, which reaches 1 at α ≈ 0.011 — but that is an
extrapolation through points that are themselves marginal, so the honest statement is

> the edge of convergence lies **below α ≈ 0.05, consistent with zero**, against Young's 0.262

At α = 0.02 the tail ratio is 0.985 with interval [0.909, 1.060], straddling 1. At α = 0.05
the answer depends on the sampling — 0.934 [0.874, 0.994] at eight levels and 32 phases,
1.024 at seven levels and 12 phases — so that point is **not resolved** by the accessible
level range, and the table entry should be read as "≈ 1". By α = 0.15, still well under
Young's threshold, the ratio is 0.804 [0.749, 0.858] with the whole interval clear of 1. And
at α_c itself the ratio is 0.669 [0.620, 0.714]: comfortably convergent, no kink, nothing
happening.

So the sums do converge well inside the region Young's condition leaves open. That does not
contradict Young — his condition is *sufficient*, so failing it proves nothing, and the
theorem simply goes silent there. The mechanism is sign cancellation that an absolute-value
estimate necessarily discards.

### The threshold across dimension: Young's law with d → βd, β ≈ ¾

The apex angle makes this testable rather than a story about one curve. Extrapolating each
low branch to rate = 1:

| d | Young α_c = d − 1 | measured threshold | 4r² (geometric floor) | implied β = (1+α_thr)/d |
| --- | --- | --- | --- | --- |
| 1.135 | 0.135 | −0.178 → converges for every α > 0 | 0.3474 | 0.724 |
| 1.262 | 0.262 | ≲ 0.05, consistent with 0 | 0.4444 | ≈ 0.80 |
| 1.631 | 0.631 | **+0.156** | 0.7306 | 0.709 |

At d = 1.631, α = 0.05 the tail ratio is 1.0856 with interval [1.028, 1.168] — entirely above
1, so that is genuine non-convergence, not a marginal case. There *is* a phase boundary; it
simply is not where Young's condition puts it.

Setting 4^β r^(1+α) = 1 gives β = (1 + α_thr)/d, and the three dimensions return 0.724, ≈0.80,
0.709. So the convergence condition is Young's with the box dimension replaced by an effective
one:

> **α > βd − 1 with β ≈ 0.75 ± 0.05**

β = 1 would recover Young exactly; β = ½ is the random-walk value. It is neither. Independent
signs are rejected at 15σ by a 48-phase measurement, and the low branch is not even of the
form 4^β r^(1+α) for constant β — its slope in α is −1.607 at d = 1.262 where that law forces
−1.099, and −1.444 vs −1.222 at d = 1.135. The implied β drifts from 0.80 at α = 0.02 to 0.65
at α = 0.45 before the geometric branch takes over.

The geometric floor is confirmed at every dimension: at d = 1.135, α = 0.90 the tail ratio is
0.3471 [0.345, 0.349] against 4r² = 0.3474. The crossover into it is at α ≈ 0.52 for the
standard curve, observed between 0.45 (0.497, still above the floor) and 0.55 (0.4525,
essentially on it) — higher than the 1 − d/2 = 0.369 that `predicted_crossover` returns, since
that formula inherits the refuted random-walk assumption for the branch below.

### The measurement had to be rebuilt twice

**A fitted decay rate cannot see this, and the first sweep was wrong.** Fitting
|I_{n+1} − I_n| ~ C·rateⁿ gave a table with β ≈ 0.6 at small α that looked like a clean
result. It was measuring the truncation. `truncation_stability` is the check: at α = 0.1 the
fitted rate runs from 0.33 at 12 terms to 0.69 at 38, while at α = 0.5 and 0.75 it holds to
ten percent.

**And the trap is not escapable by tuning.** A Weierstrass truncation must be both resolved by
the partition (K ≤ L log(1/r)/log b, since a mode faster than the segment length is aliased
by the midpoint rule) and small in its tail (b^{−αK} ≤ tail). Multiplying the two, **log b
cancels**:

> α · L · log(1/r) ≥ log(1/tail)

Raising the base buys nothing — it caps the truncation exactly as fast as it deepens each
term. Only the level count helps, and L is capped by the 4ᴸ points of the curve. The honest
floor is α ≥ 0.63 at L = 10, *far above* the threshold 0.26 one wants to probe, and the ratio
α_floor/α_c = log(1/tail)/(L(log 4 − log(1/r))) falls only like 1/L. **A fixed Weierstrass
form can never reach the sub-threshold regime.** `resolution_window` reports this and
`measure_coherence` flags whether its own answer is admissible.

The fix is `matched_weierstrass_form`: cut the spectrum exactly where the partition stops
resolving it. Every retained mode is resolved, every discarded one would have been aliased,
so any α becomes measurable. The price is that it is a *sequence* of forms, and it is the
lacunary construction that **saturates** Young's bound — so it measures how bad a form can
be, not how bad a typical one is. Results from it are labelled accordingly.

**Then β itself turned out not to be an exponent.** The per-level β drifts monotonically
downward — 0.813 → 0.572 over eight levels at α = 0.05, no flattening — so quoting it at any
one level measures the level. What is scale-free is the growth of |Σ|/‖·‖₂: coherent addition
doubles it every level, independent signs leave it flat. β = ½ + log(growth)/log 4 inverts
that, and the smooth control pins the calibration exactly, returning growth = 1.9999 and
β = 1.0000 with residual 5.6 × 10⁻⁵.

**And a regression cannot notice that its own model is wrong.** `decay_rate` returns
exp(slope) of a log-linear fit, which is meaningful only if the sequence *is* geometric.
Below the crossover it is not: the fit reported convergence with rate 0.75 while the
increments were flattening out. `RateMeasurement` now carries a `tail_rate` and a `geometric`
flag, and `converges` tests the tail rather than the whole-sequence slope. The first published
version of this section drew the wrong exponent (β = ½) from that fit, and a subsequent
12-phase run then over-corrected in the other direction — reading a genuine marginal case at
α = 0.05 as a plateau, because RMS over 12 phases has 20% scatter per level. Neither is
right; the bootstrap interval on the tail ratio is what settles it, and it is now what
`converges` uses.

**Base 3, not base 2.** With base 2 the matched truncation grows by one or two modes per level
depending on the level (K = 6, 7, 9, 11, 12, 14, 15), injecting a systematic wobble into any
cross-level fit — visible as a reproducible dip at level six at every α. Base 3 gives K = L
exactly for the standard curve, one new mode per level, matching the curve's own lacunarity.
The χ²/dof of the coherence fit drops from 16.7 to 0.04 above the crossover.

### Referees

- The per-segment terms must sum to I_fine − I_coarse exactly, or the child-to-parent
  grouping is wrong and every coherence number is meaningless.
- `sampled_alpha` recovers the Hölder exponent the contributions actually exhibit, from how
  the mean term magnitude shrinks with level. It tracks the requested α to within 0.02 across
  the whole plane — the aliasing referee.
- `peakiness` = ℓ¹/(√count·ℓ²) stays near 0.84, so the terms are comparable in size and β is
  a cancellation exponent rather than one term dominating.
- The smooth control satisfies |Σ| = ℓ¹ *exactly*, to the last bit, over 262,144 segments:
  every contribution carries the same sign. That is what coherent addition literally means,
  and no fitted rate could produce it — the old estimator could only manage 1.005.
- The partial sums themselves, as the last word on convergence. At α = 0.95 they spread by
  5.6 × 10⁻⁵ over the final four levels; at α = 0.05, by 9.9 × 10⁻² — of order the value.

```bash
# Model-free: RMS increments per level with a bootstrap interval on the tail
python -c "import fractal_stokes as f; m=f.measure_increments(0.65); print(m.ratios, m.tail_interval())"

# Coherence exponents across the (d, alpha) plane
python -c "import fractal_stokes as f; print(f.phase_diagram([0.8, f.KOCH_ANGLE, 1.4], [0.2, 0.6, 0.9]))"
```

## A seventh target: the Selberg trace formula on graphs

`selberg.py` — the Ihara zeta function, prime geodesics, and the Riemann hypothesis
analogue, in exact integer arithmetic.

### The object

For a hyperbolic surface, Selberg's trace formula couples the Laplacian spectrum to the
lengths of closed geodesics. A finite graph has an exact analogue in which every term is a
finite integer computable two independent ways — which is the only reason to do this
numerically. The geodesics are non-backtracking tailless closed walks, counted by the
Hashimoto edge operator on the 2|E| directed edges:

> B[(u,v),(v,w)] = 1 iff w ≠ u, and N_m = tr(Bᵐ), with ζ(u) = 1/det(I − uB)

Bass's theorem re-expresses that 2|E| × 2|E| determinant through the |V| × |V| adjacency
matrix: det(I − uB) = (1−u²)^(r−1) det(I − Au + qu²I) for a (q+1)-regular graph.

### Everything is refereed, nothing is asserted

Since arXiv is unreachable from this sandbox (see below), no formula here is taken on
authority. Each is computed along two routes sharing no code:

- **tr(Bᵐ) vs brute-force enumeration** of closed non-backtracking tailless walks — exact
  integers, four graphs, lengths 3–8. The wrap-around condition is what makes a walk
  *tailless*, and a naive path enumeration misses it.
- **Bass's determinant vs det(I − uB)** — exact integer polynomials, coefficient by
  coefficient, on six graphs including the 144×144 edge operator of the 24-cell against its
  24×24 adjacency matrix.
- **Euler product ∏(1 − u^len)^(−π) vs the ζ power series** — formal power series identity.
- **Spectral side vs geodesic side** of the trace formula.
- **1/ζ(K₄) against its classical closed form** (1−u²)²(1−u)(1−2u)(1+u+2u²)³ — an anchor
  outside the module's own machinery.

Independent combinatorial anchors: π(girth) = 2 × (number of shortest cycles), one per
orientation. K₄ gives 8 = 2×4 triangles, Petersen 24 = 2×12 pentagons, Heawood 56 = 2×28
hexagons, K₃,₃ 18 = 2×9 squares. All hold.

### The result: the RH analogue, quantitatively

Substituting u = q^(−s) turns ζ's pole structure into a critical-line statement. A
(q+1)-regular graph satisfies it exactly when it is **Ramanujan** — every non-trivial
adjacency eigenvalue obeying |λ| ≤ 2√q. Each λ gives poles at the roots of x² − λx + q, which
have modulus exactly √q when the Ramanujan bound holds and split into a real pair exceeding
it when it fails.

The sharp consequence is about the graph prime number theorem π(m) ~ qᵐ/m. Normalising the
error,

> R(m) = |π(m) − main(m)| · m / q^(m/2)

R is **bounded** iff the graph is Ramanujan, and otherwise grows at rate |α_max|/√q. The
prediction has no free parameter: the growth comes from diagonalising the adjacency matrix,
the error from counting closed walks.

| graph | Ramanujan | measured growth | predicted |
| --- | --- | --- | --- |
| Petersen | yes | 1.0025 | 1.0000 |
| Heawood | yes | 0.9868 | 1.0000 |
| K₅ | yes | 1.0119 | 1.0000 |
| Pappus | yes | 0.9845 | 1.0000 |
| 24-cell / 2T Cayley | yes | 0.9521 | 1.0000 |
| Circular ladder 20 | **no** | 1.2241 | 1.2558 |
| Circulant C₂₀(1,2) | **no** | 1.1761 | 1.1968 |
| Circular ladder 30 | **no** | 1.3045 | 1.3493 |
| Circulant C₂₄(1,2) | **no** | 1.4233 | 1.4022 |

Every case agrees within 0.08, and the two classes separate cleanly.

### Two instrument failures, one of them a real bug

**A fitted error exponent does not work, for the third time in this project.** The error is a
sum of terms αᵐ with |α| = √q, so it *oscillates* and passes near zero at some lengths.
Regressing log|error| on m gave residuals of 0.8–1.0 — factor-of-three scatter — and missed
the predicted exponent at every graph tested, including ones where the prediction is exact by
construction. Boundedness of R(m) is the right test; a slope is not. `prime_geodesic_fit` is
retained only so the failure stays visible, with a test asserting its residual is bad.

**The 24-cell Cayley graph was disconnected, and it took four digits to notice.** The obvious
generating set {±i, ±j, ±k} generates the quaternion group of order 8, not the binary
tetrahedral group of order 24 — no product of i, j, k has order 3. The resulting graph looks
entirely healthy: 6-regular on 24 vertices, passes the Ramanujan test. But the prime geodesic
theorem's main term is *per component*, so with three components it counts 3qᵐ/m and the
missing two copies masquerade as an error growing like √q per step. Measured growth 2.2372
against √5 = 2.2360 — the four-digit match is what identified the cause. Fixed by generating
with ω = (−1+i+j+k)/2, which has order 3, and by verifying the generated subgroup rather than
assuming it. `riemann_hypothesis_test` now refuses a disconnected graph outright.

The 24 units are the vertices of the 24-cell and form the binary tetrahedral group, the
double cover of the tetrahedron's rotation group — which is where this meets the packing
problem the project started from.

### Literature access, and what it confirmed

This sandbox initially blocked `arxiv.org`, `wikipedia.org` and the journal hosts, so the
module was written to take no formula on authority — every identity verified in code against
an independent computation. Access has since been opened, and the standard statements were
then checked against it. All of them hold as used:

- **Ihara's formula** ζ_G(u) = 1/[(1−u²)^(r−1)·det(I − Au + qu²I)] — matches exactly,
  including the exponent r−1. Ihara (1966); the graph-theoretic reformulation is Sunada
  (1986), following a suggestion of Serre; the determinant formula over the adjacency matrix
  is Bass (1992); the edge operator is Hashimoto's.
- **Ramanujan ⟺ RH** for the Ihara zeta — the equivalence this whole paper measures — is due
  to **Sunada**, an attribution the first draft did not have.
- **r−1 = (q−1)n/2** for connected (q+1)-regular graphs on n vertices, which the literature
  states and which we verify on six graphs as an extra consistency check.

Two things worth recording about the access itself. The proxy answers blocked hosts with
**HTTP 200 and an error page in the body**, so a status-code reachability probe reports
success for hosts that are still blocked — the body has to be inspected. And the proxy serves
HTTPS only, so arXiv's documented `http://export.arxiv.org` API endpoint fails while the
`https://` form works; that produced a second wrong diagnosis before the first was corrected.

```bash
python -c "import networkx as nx, selberg as sb; print(sb.riemann_hypothesis_test(nx.petersen_graph()).growth)"

```

## An application: community detection as a failure of the Ramanujan bound

`detection.py` — turning the proved dichotomy into a detector with a predicted threshold.

### The idea

`selberg.py` proves that each adjacency eigenvalue λ contributes two non-backtracking
eigenvalues, the roots of x² − λx + q, and that

- |λ| ≤ 2√q ⟹ they are a conjugate pair of modulus **exactly** √q — on a circle;
- |λ| > 2√q ⟹ they are real and the larger **escapes** the circle.

So "Ramanujan" means *nothing outside the circle*. The application is to read that
backwards: **an escaping eigenvalue is structure**, and how far outside it sits measures how
much. Planted communities are a graph failing the Ramanujan bound.

For a sparse graph of mean degree c the bulk sits at radius √(c−1), and a two-group block
model with assortativity ε = (a−b)/(a+b) puts its signal eigenvalue near cε. The two cross at

> ε\* = √(c−1)/c — the Kesten–Stigum threshold, with **no fitted constant**

### The prediction, tested

n = 4000, c = 5, so ε\* = 0.400. Overlap is scaled so chance = 0 and perfect = 1.

| ε | 0.00 | 0.20 | 0.35 | 0.40 | 0.45 | 0.50 | 0.60 | 0.70 | 0.80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \|λ₂\| | 2.263 | 2.260 | 2.261 | 2.287 | 2.281 | 2.469 | 3.015 | 3.500 | 4.000 |
| c·ε | 0.00 | 1.00 | 1.75 | 2.00 | 2.25 | 2.50 | 3.00 | 3.50 | 4.00 |
| non-backtracking | 0.003 | 0.010 | 0.015 | 0.039 | 0.065 | **0.350** | 0.648 | 0.800 | 0.887 |
| adjacency | 0.010 | 0.017 | 0.020 | 0.039 | 0.069 | 0.096 | 0.382 | 0.781 | 0.897 |

Two things happen exactly as predicted:

**λ₂ = max(bulk edge, c·ε).** Below the threshold the second eigenvalue is *pinned* — it does
not move with the signal at all, reading 2.26 whether ε is 0.00 or 0.35. Above it, it detaches
and tracks c·ε to under a percent: 3.015 vs 3.00, 3.500 vs 3.50, 4.000 vs 4.00.

**Detection turns on at the crossing.** The overlap sits at chance until the eigenvalue
detaches, then rises immediately.

**And the non-backtracking operator beats the adjacency matrix exactly where it should** — near
the threshold, 0.350 against 0.096 at ε = 0.50, a 3.6× advantage. Far above, both work and the
advantage vanishes (0.887 vs 0.897 at ε = 0.80). That is the expected shape: the adjacency
spectrum of a sparse graph is spoiled by degree fluctuations whose eigenvectors localise on
high-degree vertices, and a non-backtracking walk cannot linger on one.

### The finite-size correction, and why it matters practically

The radius √(c−1) is asymptotic. Measured at c = 5 with no planted signal at all, the bulk
edge sits at **2.26–2.30 across n = 500 to 8000** — flat to one percent, and nowhere near
2.00. It does not converge over any size one can build.

That has a real consequence: a detector comparing |λ₂| against √(c−1) reports structure in
structureless graphs, at every size. `empirical_bulk_edge` supplies the honest null by
measuring the same statistic at ε = 0, and `finite_size_threshold` returns 0.452 against the
asymptotic 0.400 — which is where detection was actually observed to begin, between 0.45 and
0.50. A test asserting the edge *descends* toward the limit was written first and failed;
there is no descent to see.

### Two bugs the referees caught

**The sparse operator was permutation-scrambled.** Built from `graph.edges()` in insertion
order it is permutation-similar to `selberg.hashimoto_operator` — same spectrum, different
matrix — so a spectral check would pass while an entrywise one failed. Fixed by reusing
`selberg.directed_edges` outright instead of restating its convention, which makes the
entrywise referee meaningful.

**The block model sampled one group twice.** `for b in (a, 1)` visits the (1,1) block twice,
so group one's internal edges were drawn at double rate. Mean degree came out 6.22 for a
requested 5 — and since the threshold is a function of mean degree, the experiment was testing
a prediction for a graph it wasn't generating. It also left the two groups with different
densities, which is structure nobody asked for and which a detector could find while appearing
to recover the planted partition. There are tests for both the mean degree and the group
symmetry now.

```bash
python -c "import detection as dt; [print(p.assortativity, round(p.nonbacktracking_overlap,3)) for p in dt.threshold_sweep([0.2,0.5,0.8], size=2000)]"
```

## A seventh target: rank loss in the Einstein constraints

`adm.py` — where the linearised constraint operator stops being surjective, and why a
perturbation can then solve the linear problem without solving the real one.

### The phenomenon

Vacuum initial data is a pair (g, K) satisfying the constraints

> H = R(g) − |K|² + (tr K)² = 0,  M^i = ∇_j(K^ij − g^ij tr K) = 0

Write Φ(g,K) = (H, M). Where DΦ is *surjective*, the implicit function theorem says every
solution of DΦ(h,k) = 0 integrates — it is tangent to an actual curve of solutions.
Surjectivity fails exactly when the adjoint DΦ\* has a kernel, and those kernel elements are
**KIDs** (Killing Initial Data), pairs (N, X) generating a spacetime Killing field. So
**symmetry of the solution is the same thing as rank loss of the constraint map**, and where
the rank drops the linear theory stops predicting the nonlinear one. That is linearisation
instability, and the obstruction is second order: for each KID, ∫ N·Q(h,k) must vanish.

### Why the flat torus makes it exact

On T³ with the flat metric and K = 0, Fourier modes are indexed by **integer** vectors, so
every mode matrix of DΦ is integral and its rank is a combinatorial fact — not a decision
about how small a singular value must be before it counts as zero. Near a rank-loss point
that distinction is the entire question.

### The result

Sweeping every mode in [−3,3]³, as exact integer ranks:

| | modes | rank | deficiency | KID dimension |
| --- | --- | --- | --- | --- |
| k ≠ 0 | 342 | 4 (full) | 0 | 0 |
| k = 0 | 1 | 0 | **4** | **4** |

**Deficiency equals KID dimension at every single mode** — the Fischer–Marsden correspondence
as an integer identity. The 4 is one constant lapse plus three translations.

And rank loss is confined *entirely to the zero mode*. That is precisely why the obstruction
is an **integral** over the torus rather than a pointwise condition — there is nowhere else
for it to live.

### The instability, exhibited

For each k ≠ 0 there are exactly **two** transverse-traceless polarisations. With h = 0 they
solve the linearised constraints *exactly* — trace-free and transverse kills both DH and DM
identically. And the second-order obstruction on them reduces to

> ∫[(tr k)² − |k|²] = −|k|² < 0

strictly negative, never zero: −2, −4, −6, −7, −28/9 at k = (1,0,0), (1,1,0), (1,1,1),
(2,−1,3), (3,1,−2). So these perturbations solve the linear problem and provably fail to be
tangent to any solution. The linear theory predicts a deformation the nonlinear theory does
not admit.

The sign is genuinely about the TT sector, not an artefact: a pure-trace perturbation gives a
*positive* value, and there is a test for that.

### Two conventions that had to be pinned, not assumed

**Gauge invariance is the sharp check.** A pure-gauge perturbation h = Lie_X δ must lie in the
kernel of DΦ. Get any sign or index wrong and it doesn't. This passes for every mode and every
shift vector, and there's a test confirming a generic perturbation is *not* in the kernel, so
the check isn't vacuous.

**The adjoint is not the transpose.** I first wrote `kid_matrix` as the naive transpose and
the pairing identity failed with 900 mismatches. The reason is real: the natural inner product
on symmetric tensors weights an off-diagonal slot by 2, since it stands for two entries, and
the constraint matrix folds that weight into its own entries. The identity is

> Σ_r (DΦh)_r u_r = Σ_slot mult(slot)·h_slot·(DΦ\*u)_slot

with multiplicities explicit. A test asserts the naive transpose *fails*, so the distinction
stays visible.

Also fixed: `mode_report` originally transposed the adjoint before taking its kernel, computing
the null space in the 12-dimensional target instead of the 4-dimensional source — the wrong
space entirely.

### Closing the second-order term

The obstruction's metric half needs R⁽²⁾(h). The first version of this module declined to
implement it, on the grounds that an expansion with no independent referee shouldn't be used.
It now has one.

Expanding the curvature of δ + εA·cos(k·x) symbolically and averaging over the torus gives,
with residual **identically zero** against the four available quadratic invariants:

> ⟨R⁽²⁾⟩ = −⅛|k|²|A|² − ⅛|k|²(tr A)² + ¼|Ak|²

(No (tr A)(kAk) term — not obvious in advance, and what makes four invariants enough.)

**The referee:** the same symbolic pipeline taken to *first* order must reproduce DH from
`linearised_constraint_matrix` — and does, as an identical polynomial. Since DH is
independently pinned by the gauge check, the second-order coefficient inherits that standing.

**A normalisation this exposed.** Both terms are torus *averages*, and ⟨cos²⟩ = ½ means the
extrinsic-curvature term carries a half that the first version omitted. Immaterial to a claim
about the sign of one term — a positive rescaling can't flip a sign — but wrong the moment the
two are added, which is exactly what the full obstruction does.

### The full result: rigidity

With both halves, the obstruction is a quadratic form on the 8-dimensional solution space.
Gauge there is 4-dimensional: three shifts, plus a lapse direction h = 0, k̂_ij = −k_i k_j N̂
(which solves the momentum constraint identically, since ∂_j(N_,ij − δ_ij ΔN) telescopes).

> **At every k ≠ 0 the inertia is (negative, zero, positive) = (4, 4, 0), and the four null
> directions are exactly the gauge span.**

So the form is negative definite modulo gauge: **every** non-gauge linearised solution is
strictly obstructed and none integrate — metric perturbations included, not just the
transverse-traceless ones. The TT result is now the special case h = 0.

Inertia is computed by exact congruence over ℚ (symmetric Gaussian elimination, Sylvester's
law), not from floating-point eigenvalues whose signature would depend on a tolerance — the
wrong instrument for a form whose whole interest is where it degenerates.

One subtlety worth recording: Q·g ≠ 0 as a vector in ℝ¹². The radical is a statement about the
*restriction* to ker DΦ, and checking it on the full space reports a spurious failure. I hit
that before pairing gauge against kernel vectors instead.

```bash
python -c "import adm; print([ (r.wave, r.rank, r.kid_dimension) for r in adm.sweep_modes(1) if r.loses_rank])"

```

## An eighth target: the Hodge Conjecture, via Shioda's reduction

`fermat_hodge.py` — **not** a proof of anything. It computes a combinatorial invariant that a
published paper explicitly asks for and could not push far, and tests a conjecture stated
there at values outside its range.

### The reduction

For the Fermat variety Xⁿₘ, Shioda showed the Hodge Conjecture becomes a *finite combinatorial
statement*. Primitive cohomology splits into 1-dimensional character eigenspaces V(α), and the
primitive Hodge classes are indexed by

> Bⁿₘ = { α ∈ (ℤₘ)^(n+2) : all aᵢ ≠ 0, Σaᵢ ≡ 0, and |t·α| = p+1 for **all** t ∈ ℤ*ₘ }

with Cⁿₘ ⊆ Bⁿₘ indexing the algebraic ones. **HC for Xⁿₘ ⟺ Cⁿₘ = Bⁿₘ.** The quantifier over
all units is the rationality condition — at t = 1 alone you'd only get Hodge *type*.

Shioda encodes this in the semigroup M_m = {(x₁…x_{m−1}; y) ≥ 0 : Σᵢ⟨ti⟩xᵢ = my ∀t ∈ ℤ*ₘ},
where decomposability means the class comes from the inductive structure of Fermat varieties,
hence is algebraic. The controlling invariant is

> **φ(m) = max{ y : (x; y) indecomposable in M_m }**

and da Silva's Prop 3.7 makes it load-bearing: if HC holds for Xⁿₘ at every n ≤ 2(φ(m)−1), it
holds for **all** n. So φ(m) is exactly how many dimensions must be checked at degree m. Since
y = 0 forces x = 0, the cone is pointed and the indecomposables are its Hilbert basis.

### Three referees, none of them self-referential

- **Fermat surface Picard numbers.** |B²ₘ| must equal ρ − 1. Computed: 6, 19, 36, 85, 90 for
  m = 3…7, against classical ρ = 7, 20, 37, 86, 91. The m = 4 case is the maximal K3 with
  ρ = 20; m = 3 is the cubic surface, a plane blown up at six points.
- **Two constructions that share no code.** Bⁿₘ enumerated from the character criterion must
  land inside M_m — cut out by different equations — at height exactly n/2+1. It does.
- **The published table.** φ(m) for 20 ≤ m ≤ 43 is in the literature; all **24 values
  reproduced exactly**. That agreement is what licenses anything computed outside the range.

### What was computed

The published data stops at m < 48 ("computations become more and more time consuming"). The
conjecture stated there is φ(pᵏ) = (p^(k−1)+1)/2 for odd p, φ(2ˡ) = 2^(l−2)+1 for l > 2 —
formulated from the prime powers available below 48, namely 4, 8, 9, 16, 25, 27, 32.

**Confirmed at eight prime powers outside that range:**

| m | 49 | 64 | 81 | 121 | 125 | 169 | 289 | 343 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | 7² | 2⁶ | 3⁴ | 11² | 5³ | 13² | 17² | 7³ |
| φ(m) computed | 4 | 17 | 14 | 6 | 13 | 7 | 9 | 25 |
| conjecture | 4 | 17 | 14 | 6 | 13 | 7 | 9 | 25 |

Spanning five primes and exponents k = 2, 3, 4, 6 — a single prime or a single exponent would
have been weak evidence. Every prime tested from 47 to 109 gives φ = 1, consistent with
Shioda's theorem that prime degree implies HC.

**Honest limits.** This is verification of someone else's conjecture, not a new theorem, and it
says nothing about HC beyond what Shioda and da Silva already established. The general table
past m = 47 stayed out of reach: the Hilbert basis explodes with distinct prime factors
(45,655 elements at m = 42 = 2·3·7 versus 21 at m = 43), which is why prime powers could be
pushed to 343 while m = 48 = 2⁴·3 could not be finished at all.

```bash
python -c "import fermat_hodge as f; r=f.phi_report(121); print(r.phi, r.conjectured, r.matches_conjecture)"
```

## Tests

```bash
python -m pytest test_tetra_spectral.py -v      # tetrahedron packings
python -m pytest test_amplituhedron.py -v       # amplituhedron tilings
python -m pytest test_bootstrap.py -v           # conformal bootstrap
python -m pytest test_arithmetic_que.py -v      # arithmetic QUE
python -m pytest test_spinfoam.py -v            # spin networks
python -m pytest test_fractal_stokes.py -v      # fractal Stokes
python -m pytest test_selberg.py -v             # graph Selberg trace formula
python -m pytest test_detection.py -v           # community detection
python -m pytest test_adm.py -v                 # ADM constraint rank loss
python -m pytest test_fermat_hodge.py -v        # Hodge conjecture / Fermat
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
