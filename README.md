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
| `platycosm.py` | Linearisation instability on the compact flat 3-manifolds |
| `test_platycosm.py` | Platycosm test suite (51 tests) |

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

## A ninth target: linearisation instability beyond the torus

`platycosm.py` — quantum gravity on compact flat slices that are **not** the 3-torus.

### The question

Fischer–Marsden–Moncrief: a spacetime with a compact Cauchy surface and a Killing field is
linearisation unstable — a solution of the linearised constraints must satisfy one integral
condition per KID to extend to a real solution. Moncrief worked this out in detail for flat
spacetime with **toroidal** spatial sections, and showed quantum-mechanically that physical
states must be invariant under the symmetries those conditions generate. Group averaging
reproduces it.

But T³ is only one of six orientable compact flat 3-manifolds — the *platycosms* — and the
others carry strictly less symmetry. A search for "Bieberbach" + "general relativity" returns
nothing; the case appears unexamined.

It matters because the standard slogan is *symmetry ⟹ instability*, and the quantum version is
*states must be symmetry-invariant*. Both suggest the effect should weaken as symmetry is
removed. The platycosms test that, because holonomy progressively kills the translations.

### What was computed

| | manifold | \|point group\| | Killing fields | KIDs | b₁ | rigid |
| --- | --- | --- | --- | --- | --- | --- |
| G1 | 3-torus | 1 | 3 | 4 | 3 | yes |
| G2 | half-turn (dicosm) | 2 | 1 | 2 | 1 | yes |
| G4 | quarter-turn (tetracosm) | 4 | 1 | 2 | 1 | yes |
| G6 | **Hantzsche–Wendt** | 4 | **0** | **1** | **0** | **yes** |

The Hantzsche–Wendt manifold has holonomy ℤ₂×ℤ₂ fixing no direction, so **no Killing vector
fields at all** and b₁ = 0. Yet the constant lapse is a KID on *every* flat compact slice,
since Hess N − (ΔN)δ vanishes for constant N. So G6 is linearisation unstable with **zero
spatial symmetry** — nothing for a symmetry group to average over — and one stability
condition where the torus has four.

And it is still **rigid**. The obstruction form is negative definite modulo gauge on the
covering torus (`adm.py`'s (4,4,0)); restricting a negative semi-definite form to the invariant
subspace keeps it so, and a single condition then forces any solution satisfying it to be pure
gauge. **The torus's three translational KIDs are redundant for rigidity — only the lapse does
work.**

### Three referees

- **Betti numbers.** For compact flat manifolds b₁ = dim Fix(holonomy). The module computes
  the invariant subspace from the point group and never mentions homology, yet returns
  3, 1, 1, 0 — the classical b₁ of these four platycosms.
- **The Bieberbach condition is checked, not assumed.** `is_bieberbach` verifies the action is
  free. It **rejected my first Hantzsche–Wendt presentation**: a half-turn about x whose
  translation part had no x-component is a rotation about a shifted axis rather than a screw,
  so it has genuine fixed points and the quotient is an orbifold. The corrected generators are
  screws about their own axes. There is a test asserting the bad presentation fails.
- **G1 must reproduce `adm.py`** exactly, inertia (4,4,0) included.

**Limits.** G3 and G5 need a hexagonal lattice, so their point groups aren't integer matrices
on ℤ³ and the exact-integer mode analysis doesn't apply unchanged; their KID dimension is
nonetheless 2 by the same fixed-subspace argument.

**Correction.** The rigidity above is a statement about *one mode at a time*, at k ≠ 0. The
actual stability condition is a single integral over the whole slice, and summing across the
origin changes the answer — see the next section, which is where that was found.

```bash
python -c "import platycosm as p; [print(p.analyse(s)) for s in p.PLATYCOSMS]"
```

## A tenth target: what the obstruction actually is

`graviton.py` — identifying the second-order obstruction term by term.

### The gap

`adm.py` computes a quadratic form and finds it negative definite modulo gauge. `platycosm.py`
shows that survives to manifolds with no symmetry. Neither says what the form *is*. The
previous section closed with the admission that the physical consequence was "stated, not
derived." This derives it.

### Step one: the inhomogeneous modes are gravitons, and they are massless

On the transverse-traceless sector at a mode k, with h = A cos(k·x) and K = B cos(k·x),

    Q_k = −(1/8)|k|²|A|² − (1/2)|B|²

ADM at unit lapse gives ḣ = 2K, so B = Ȧ/2 and −Q_k = (1/8)(Ȧ² + |k|²A²): a harmonic
oscillator. Reading its frequency off the **exact rational form** gives

| k | (1,0,0) | (1,1,0) | (1,1,1) | (2,0,0) | (2,2,1) | (2,−1,3) | (3,1,−2) | (4,−3,1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ω² | 1 | 2 | 3 | 4 | 9 | 14 | 14 | 26 |
| \|k\|² | 1 | 2 | 3 | 4 | 9 | 14 | 14 | 26 |

**ω = \|k\| exactly, at every mode and both polarisations.** The massless dispersion is an
*output* of the constraint algebra, not an input — nothing in `adm.py` mentions propagation.

Two controls make the extraction mean something. The pure-trace mode returns
ω² = −(5/3)\|k\|² with a *negative* kinetic coefficient — the conformal factor problem, not a
graviton — so the method does not hand back \|k\|² for every tensor. And dropping the ḣ = 2K
factor gives ω = \|k\|/2, so the one physical input is load-bearing rather than decorative.

### Step two: the coefficient is Isaacson's

−Q_k is not merely proportional to the gravitational wave energy. Against the Isaacson
effective density ρ = (1/64π)⟨ḣ_ij ḣ^ij + ∂_l h_ij ∂^l h^ij⟩,

    −Q_k = 16π ρ   exactly, at every mode and every amplitude.

`adm.py` knows nothing about 1/32π; the factor arrives from the second-order expansion of the
scalar curvature.

### Step three: the homogeneous mode pays for it — and the answer is Friedmann

The obstruction is **not** negative definite overall. At k = 0 the metric term vanishes and the
momentum term has inertia **(5, 0, 1)** — exactly one positive direction, the isotropic one.
Writing B₀ = −Hδ + σ,

    Q₀ = 6H² − |σ|²

Summing over all modes, Q = 0 reads

    6H² − |σ|² = Σ_{k≠0} (−Q_k) = 16πρ    ⟺    3H² = 8πρ + ½ σ_ij σ^ij

**the Friedmann constraint with shear, sourced by the graviton energy.** Every coefficient in
it — the 6, the 16π, the ½ on the shear — is computed output, not supplied.

So linearisation instability on a compact flat slice is *not* a prohibition on gravitational
waves. It is the statement that their energy must be paid for by expansion at precisely the
Friedmann rate. A linearised solution with waves and no expansion does not integrate; one with
the matching expansion does, at this order.

### What this corrected

Read per-mode, Q_k < 0 at every k ≠ 0 and the obvious conclusion — the one this README drew in
the previous section — is that the condition kills the graviton sector outright. That is wrong,
and the reason is **a factor of two**: the averaging weight is ⟨cos²(k·x)⟩ = ½ away from the
origin but ⟨1⟩ = 1 at it. The distinction is invisible to any single-mode statement (a positive
rescaling moves no sign and no signature) and it is exactly what sets the relative weight once
the two sectors are added. `adm.obstruction_value` now carries it; every per-mode result stands
unchanged, which is precisely why the bug survived.

### The platycosms, sharpened

δ is invariant under every holonomy group, so the direction that pays for the waves exists on
every compact flat manifold. What holonomy cuts is the **shear** budget:

| | manifold | invariant homogeneous momenta | of which trace-free |
| --- | --- | --- | --- |
| G1 | 3-torus | 6 | 5 |
| G2 | dicosm | 4 | 3 |
| G4 | tetracosm | 2 | 1 |
| G6 | Hantzsche–Wendt | 3 | 2 |

Shear enters with the sign *opposite* to expansion, so less holonomy means more ways to fail
the condition, not more ways to satisfy it. And on Hantzsche–Wendt, with no Killing fields at
all, the Friedmann balance is the **only** second-order condition there is — the torus's three
extra conditions constrain wave momenta, not energy. The balance is what survives when every
symmetry that could be averaged over is gone.

### A remark on the quantum condition, not a derivation

Moncrief's proposal imposes the condition as an operator equation on physical states. Read
per-mode it says Ĥ_graviton|ψ⟩ = 0 — before normal ordering no state satisfies it, after normal
ordering only the Fock vacuum does. Read globally it says 6Ĥ² − |σ̂|² = Ĥ_graviton, relating the
graviton number operator to the momenta conjugate to the flat moduli, whose spectrum is
continuous and unbounded above. Every graviton state then has a partner expansion rate rather
than being excluded. What does not survive is the zero-point sum: on the un-normal-ordered
vacuum the right side diverges and no finite expansion pays for it, so the ordering
prescription is doing real work rather than being a convention. This is a reading of the
classical identity, not something computed here.

### Referees

- **The moduli spaces.** The invariant homogeneous momenta count 6, 4, 2, 3 — the classical
  dimensions of the moduli spaces of flat metrics on these four platycosms. The computation
  builds the point-group action on symmetric tensors and never mentions moduli.
- **A raw sum over modes.** `balance()` splits the homogeneous momentum and halves velocities
  itself; the test recomputes the same number by calling `adm.obstruction_value` mode by mode
  with none of that bookkeeping, and the two agree exactly.
- **Gauge invariance at second order.** h_ij = k_i ξ_j + k_j ξ_i returns exactly zero potential
  energy at every k and every ξ. `metric_obstruction_term` was matched against four quadratic
  invariants without any reference to gauge, so this was free to fail.
- **Normalisation independence.** `transverse_traceless_modes` returns an unnormalised rational
  basis, so the individual coefficients differ mode to mode; only their ratio is meaningful,
  and it is invariant under rescaling.

**Prior work.** None of the physical ingredients are new: the linearisation-stability framework
is Fischer–Marsden and Moncrief, the necessity of Killing fields is Arms–Marsden, the effective
energy density is Isaacson's, and second-order back-reaction of waves on a homogeneous
background is standard cosmological perturbation theory. What is done here is to obtain the
identity between them as exact computed output of one constraint calculation, with no
coefficient supplied by hand, and to push it onto the flat manifolds where the shear budget
shrinks and the symmetry conditions disappear.

```bash
python -c "import graviton as g; [print(c) for c in g.check_dispersion((2,-1,3))]"
```

## An eleventh target: where the massive w₁₊∞ action breaks

`celestial.py` — celestial holography, and an exact obstruction at integer conformal weight.

### The state of the problem

An infinite tower of soft graviton modes generates the wedge algebra of w₁₊∞,
[w^p_m, w^q_n] = [m(q−1) − n(p−1)] w^{p+q−2}_{m+n}. On **massless** hard particles this is the
Poisson algebra of polynomial area-preserving diffeomorphisms of a two-plane, acting on the
point of that plane which is the particle's momentum spinor.

The massive case is **not open**. Himwich and Pate ([arXiv:2312.08597](https://arxiv.org/abs/2312.08597),
JHEP 07 (2024) 180) derived the action on massive scalars from the soft theorems and proved it
closes on w₁₊∞. Their generators for p > 2 contain inverse powers of the momentum operator,
evaluated by a Schwinger parametrisation they describe as formal; the action mixes infinitely
many conformal families. They close by asking whether **a discrete basis at integer Δ produces
any simplification.** That is the question attacked here.

### The answer: the inverse does not exist there, and exactly there

Massive celestial primaries are integrals of the hyperbolic bulk-to-boundary propagator
G_Δ = (−p̂·q̂)^{−Δ}. Multiplication by momentum is a differential operator in the celestial
coordinates plus a shift of Δ, and the module it acts on is spanned by
**|Δ; i, j; a, b⟩ = z^a z̄^b ∂_z^i ∂_z̄^j G_Δ** — the derivatives fall on the propagator, not on
any polynomial prefactor.

**Step 1 — the operator, derived not quoted.** Expanding p̂ in the frame (q̂, ∂_zq̂, ∂_z̄q̂, n)
and trading u = −p̂·q̂ for weight shifts needs exactly one identity,

> **u·u_zz̄ − u_z·u_z̄ = 1**

which is the statement that G_Δ is a genuine hyperbolic propagator. Everything else is the
chain rule. The result agrees with Himwich–Pate's equation (6.1).

**Step 2 — the referee.** On this module **p̂·p̂ = −1 exactly**, on every basis element and every
Δ, and the four components commute. The operator carries a frame expansion, two weight shifts
and an ordering of derivative against prefactor; the mass shell is one scalar identity that
fails if any of them is wrong — and it did fail on the first version, which applied the
operator to the prefactor as well. The four components also come out carrying the **boost and
spin weights of a four-vector** under two gradings, N = Δ+i+j−a−b and M = (a−i)−(b−j), that the
module was built without reference to.

**Step 3 — bidiagonality.** The component whose inverse powers define the p > 2 generators is
−n·p̂, and it has **exactly two terms**:

    −n·p̂ |Δ; i, j⟩ = (Δ−1)^{−2} |Δ−1; i+1, j+1⟩ + Δ(Δ−1)^{−1} |Δ+1; i, j⟩

with (a, b) spectators, because q̂^{++} = 2 is constant. Inverting a bidiagonal operator is a
one-step recursion, not a continued fraction.

**Step 4 — the organising structure.** The image is the arithmetic progression
Δ−1, Δ−3, Δ−5, … with derivative orders climbing in lockstep, and the coefficients telescope
into a falling factorial:

> **v_k = (−1)^k (Δ − 2k − 2) / [(Δ−1)(Δ−2)···(Δ−2k−1)]**

So the infinite mixing is one-dimensional and explicit. Applying the operator back to the
truncated series leaves **exactly one term** — the truncation tail — everything else cancelling
in exact rational arithmetic.

**Step 5 — the obstruction.** That denominator vanishes exactly when Δ is an integer in
[1, 2k+1], and the numerator (zero only at Δ = 2k+2) can never cancel it. So:

> **(−n·p̂)^{−1} exists if and only if Δ is not a positive integer**, and at positive integer Δ
> it fails at step ⌈(Δ−1)/2⌉.

| Δ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| first singular k | 0 | 1 | 1 | 2 | 2 | 3 | 3 | 4 |

The whole principal series Δ = 1 + iλ, λ ≠ 0, is unobstructed — the progression never meets an
integer. **The discrete integer basis does not simplify the massive action; it destroys it.**

The two forbidden values are not arbitrary: Δ = 1 is the principal-series midpoint where the
(Δ−1) denominators of the momentum operator itself blow up, and Δ = 0 is where it acquires a
kernel. The obstruction is invisible before the inverse is taken, which is why the p ≤ 2
generators — Poincaré, carrying (−n·P)^{−(2p−4)} with a non-positive exponent — are unaffected.
The failure begins at p = 5/2, exactly where Himwich–Pate's Schwinger prescription does.

### Referees

- **The mass shell**, above: one scalar identity over three independent ingredients.
- **Four-vector weights** under two gradings, free to fail and didn't.
- **Two routes to the coefficients.** The recursion and the closed form are separate
  computations that must agree wherever both are defined — and they disagree in a specific way
  where they do not. At even Δ the recursion produces 0 × ∞ (β₀ infinite, v₀ zero) while the
  closed form is finite and locates the true pole one step later. That is what sets the
  obstruction at Δ = 2 to k = 1 rather than k = 0; the naive reading was wrong.
- **The wedge is the polynomial condition.** |m| ≤ p−1 is exactly the statement that
  λ₀^{p−1+m}λ₁^{p−1−m} has no pole at the origin of the plane — where a zero-energy massless
  particle sits. Jacobi holds exactly on 6000 triples; the Poisson realisation reproduces the
  structure constants exactly, and doubling the bracket normalisation breaks it.

**Prior work.** The algebra is Strominger's and Guevara–Himwich–Pate–Strominger's; the massive
action, its closure and the Schwinger prescription are Himwich–Pate's; the hyperbolic primary
basis is Pasterski–Shao. New here: the module structure that makes the operator bidiagonal, the
closed form for the family mixing, and the exact statement of when the inverse exists — a
negative answer, with a precise obstruction, to the question that paper closes on.

```bash
python -c "import celestial as c; from fractions import Fraction as F; print(c.inversion_report(F(7,2))); print(c.inversion_report(F(5)))"
```

## A twelfth target: mass and the cosmological polytope

`cosmopolytope.py` — the wavefunction of the universe as the volume of a static shape, and
what mass actually does to it.

### The setting

Arkani-Hamed, Benincasa and Postnikov ([arXiv:1709.02813](https://arxiv.org/abs/1709.02813))
attach to every Feynman graph a convex polytope in P^{V+E−1}: three vertices per edge,
x_i + x_j − y_e and its two sign flips. Its **canonical form is the flat-space wavefunction**,
and its **facets are the subgraph energies** — the boundaries of a static convex body are the
physical singularities. No time, no evolution.

The construction is for massless (conformally coupled) scalars. The usual statement of the wall
is that mass introduces branch cuts, curves the facets, and wrecks the triangulations.

**That is not quite where the frontier is.** Benincasa
([arXiv:1909.02517](https://arxiv.org/abs/1909.02517)) already showed in 2019 that treating mass
as a perturbative two-point coupling gives, order by order, a *degenerate limit of the canonical
form of a cosmological polytope* — the graph with two-valent sites inserted on the massive line.
What he leaves open, twice in the same paper, is the **resummation**: *"the study of a possible
closed form for the re-summed two-site graph is postponed to future work"*, and *"it would be
astonishing if the peculiar structure of this perturbative expansion would allow us to re-sum
it."* That is the target here.

### The result: in flat space it resums, and the geometry never deformed

Let ψ_a be the canonical form of the polytope of the chain with a two-valent sites inserted, in
the degenerate limit where those sites carry zero external energy and every subdivided edge
carries the same y. Then

> **Σ_{a≥0} (−m²/2)^a ψ_a = ψ_G evaluated at y → √(y² + m²)**

The same polytope, the same canonical form, a different point. **Mass does not deform the
geometry at all** — it changes only the map from the polytope's edge variable to the physical
one, replacing the linear y = |k| by the quadric y² = k² + m². Verified exactly through m⁶,
with one constant fitted at order m² and then predicted at m⁴ and m⁶.

For the two-site chain the resummed function is explicitly algebraic of degree two, with exactly
one square root:

    ψ = 2[E² + x₁x₂ − (x₁+x₂)E] / [(x₁+x₂)(E² − x₁²)(E² − x₂²)],   E = √(y² + m²)

So the branch cut is real — but it lives in the **kinematic map, not the geometry**. In the
energy variable E the singularities are the same three hyperplanes as in the massless case; in
the momentum variable they are those hyperplanes seen through E² = k² + m². **A facet does not
curve. It is a flat facet seen through a quadratic change of variables.**

### Why the poles looked like they were proliferating

Each ψ_a has high-order poles because facets collapse onto each other in the degenerate limit:
the a+1 intervals of the subdivided chain touching the left endpoint all carry x₁+y, likewise on
the right, and the a(a+1)/2 interior intervals all carry 2y. Summing a tower of poles of growing
order at x+y is what produces a **simple pole at the shifted location** x + √(y²+m²). The
proliferation is an artefact of expanding a shifted pole around the wrong point.

**A prediction that failed.** I first predicted the pole orders as the count of collapsing
facets. That is only an upper bound. At a = 3 six facets collapse onto 2y and the pole is of
order **five**, not six — coincident facets bound the order but do not fix it, because a higher
pole only appears where the coincident facets meet inside a common simplex. The correct orders
follow from the resummation instead, since [m^{2a}]√(y²+m²) ∝ y^{1−2a}:

| a | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| collapsing facets on 2y | 0 | 1 | 3 | **6** |
| actual pole order | 0 | 1 | 3 | **5** |

The polytope computation refuted the guess at the first value where the two counts differ. Both
functions are kept, and the disagreement is a test.

### Referees

- **The facet theorem, on loops.** Facets from Normaliz vs. an independent subgraph enumeration,
  matching exactly on 7 graphs — including the bubble, triangle and box, where the rule genuinely
  differs (an edge left out of a subgraph with *both* endpoints inside contributes 2y_e).
  Checking only trees would have been vacuous; there's a test asserting that.
- **Published wavefunctions.** The 2-site, 3-site and one-loop bubble canonical forms come out
  right, and the residue on the total-energy pole is the flat-space amplitude 2/(y²−x²).
- **One constant, three tests.** The insertion weight −1/2 is fitted at m² and then predicts m⁴
  and m⁶, the last requiring the five-site polytope in P⁸ with fifteen facets.

**Scope.** Flat space only. In a genuine FRW background each inserted site carries ω^{2α−1} and
an integral over ω, so the tower's terms are integrals of canonical forms rather than canonical
forms, and produce polylogarithms; that is the case Benincasa poses and it is **not** settled
here. What the flat-space result does say is where to look: the obstruction is not that the
polytope must curve, because in the flat limit it demonstrably does not.

```bash
python -c "import cosmopolytope as c; print(c.facet_theorem_residual(c.BOX)); print(c.resummation_residual(2))"
```

### The FRW case: why the same trick provably cannot work

Flat space is the corner where the mass coupling is constant in time, so in the energy
representation each inserted site sits at ω = 0 and the tower term is a canonical form
evaluated at a point. In FRW with a(η) = (−η)^{−α} the coupling is time dependent and each
inserted site carries **ω^{2α−1} dω** instead. De Sitter is α = 1.

Doing that integral at first order gives, in closed form,

> **ψ₁^{dS} = 4[A ln A − B ln B − C ln C + D ln D] / [(x₁²−y²)(x₂²−y²)]**

with A = x₁+x₂, B = x₁+y, C = x₂+y, D = 2y — the four subgraph energies that survive the
degenerate limit — obeying the single relation **A + D = B + C**, which is exactly what makes
the weight-one numerator scale free. (This term is Benincasa's eq. 4.10, rederived; verified
here two independent ways and against numerical quadrature to 12 digits at three points.)

Two things are worth reading off it.

**The quadrics are already there.** The denominator is not a product of linear subgraph
energies but of the quadrics x_i² − y² — the very loci E² = x_i² that the flat-space
resummation only reaches after summing the entire tower, here at m = 0 and at first order. In
both cases each quadric has an unphysical branch (x_i = y in dS, E = x_i in flat space) that is
**not** a facet of any cosmological polytope, and in both cases the numerator vanishes there
exactly, so no unpredicted singularity appears. Neither numerator was built with that in mind.

**And the mechanism is dead.** Suppose the dS tower resummed the way the flat one does, by a
reparameterisation Σ t^a ψ_a = ψ₀(x, f(y,t)). Since ψ₀ is *rational* in its arguments, every
Taylor coefficient in t would be rational in y. But ψ₁^{dS} carries four logarithms with
non-vanishing coefficients. So:

> **No reparameterisation of the edge variable can generate the FRW tower.** The obstruction is
> transcendence of the first term — nothing to do with facets curving, which in flat space they
> demonstrably do not do at all.

**But the geometry does not vanish — it changes job.** Integrating an inserted site's energy
picks up one residue per pole, and the poles are exactly the **facets containing that energy**;
the resulting logarithm's argument is that facet with the energy removed. So:

> the polylogarithmic **alphabet is the facet set**, while facets not carrying the integrated
> energy stay rational prefactors.

At one insertion this gives {x₁+x₂, x₁+y, x₂+y, 2y} — matching ψ₁^dS exactly. At two it gives
six letters, matching the six poles the actual integration produced. In FRW the polytope stops
computing the answer and starts supplying the letters: the result is no longer a canonical form,
is not even rational, and its singularity structure is still dictated by the same convex body.

*(A guess I had to correct: at one insertion every surviving facet happens to contain the
integrated energy, so "letters = surviving facets" and "letters = facets carrying the energy"
agree. At two they don't — seven surviving facets, six letters, with x₁+y staying a prefactor.
The one-insertion coincidence is not the rule.)*

At second order the first energy integral leaves six independent ω-dependent logarithmic
letters, so the next integration is genuinely dilogarithmic: the transcendentality is not
bounded. What that leaves is the reading that the FRW resummation must shift a *transcendental*
label rather than a kinematic one — the Bessel index of the mode functions, elementary only on a
half-integer sublattice, which is where Benincasa's light states live. **That reading is not
established here**, and the FRW resummation remains open.

```bash
python -c "import cosmopolytope as c, sympy as sp; x1,x2,y=sp.symbols('x1 x2 y',positive=True); print(sp.simplify(c.frw_first_order(x1,x2,y)))"
```

## A thirteenth target: the shell-model route to Navier-Stokes

`shell.py` — the parameter geometry of the current best blow-up candidate.

### Where the problem actually stands

Global regularity for 3D Navier–Stokes is open, and the standard route to a counterexample is
Tao's two-step program: build a shell model that blows up, then embed it in the true equations.
Step 1 has a short and very recent history:

| model | inviscid | viscous (3D parameters) |
|---|---|---|
| Katz–Pavlović | blow-up | **regular** |
| Obukhov, exponential shells N_k = λ^k | **regular** | **regular** |
| Tao's model | blow-up | blow-up, but interactions have no Euler counterpart |
| **Obukhov, super-exponential N_k = N₀^{b^k}** | blow-up (α ≥ 1) | blow-up (α > 2, smooth forcing) |

That last line is [Palasek, arXiv:2605.13827](https://arxiv.org/abs/2605.13827), **May 2026** —
three months old, and now the leading candidate for Step 2 because its interactions
(`u_k·∇u_{k−1}` and `P_k div u_{k+1}⊗u_{k+1}`) do appear in the real Euler nonlinearity.

**I am not contributing to the Navier–Stokes problem.** The open step is the embedding, which
is a PDE construction and is untouched here. What follows is exact arithmetic on that model's
parameter geometry.

### The cascade exponent, for arbitrary shell separation

The model has an exact power-law stationary state X_k = c·N_k^{−γ}, and solving the fixed-point
equation gives

> **γ = α / (2b + 1)**

verified as an identity between *exponents* in rational arithmetic, not numerically. At b = 1
(exponential shells) this is α/3, and at α = 1 that is **Kolmogorov's 1/3** — an anchor the
derivation was not fitted to.

The consequence is the point: **γ decreases in b.** Wider shell separation flattens the cascade
state that regularises the model, and as b → ∞ it flattens away entirely. That is the mechanism
by which super-exponential separation buys blow-up, compressed into one exponent. I have not
found this written down for b > 1.

### The trapping region sits above the cascade

Palasek's barriers are A_k = N_k^β in the rescaled variable, where the cascade state sits at
N_k^{2αb/(2b+1)}. His viscous constraint β > 2b clears it exactly when

    2b + 1 ≥ α

which is automatic for b > 1 whenever α ≤ 3 — hence throughout the three-dimensional window,
and unconditionally in the inviscid case. So the blow-up is an **escape above** the regularising
cascade, not a competition with it.

### A tension in the parameters worth naming

The viscous theorem needs b ∈ (1, α/2), which is non-empty **exactly when α > 2** — recovering
the sharpness of his hypothesis by counting an interval, where the paper gets it independently
from energy criticality. But the physically relevant intermittency range in 3D is α ∈ [1, 5/2],
so the candidate window is α ∈ (2, 5/2] and there

> **b < 5/4.**

The shells are super-exponential but *barely*. Section 4 of the paper argues that wide
separation is precisely what would make an embedding tractable, since it suppresses cross-scale
errors — and the physically relevant window is where that margin is thinnest. The module states
the trade-off exactly. It does not resolve it.

### Referees

- **Kolmogorov's 1/3** falls out of the cascade formula at α = b = 1, with a negative control
  showing a perturbed exponent breaks the fixed-point identity.
- **The energy identity telescopes to exactly zero** in rational arithmetic, for arbitrary
  amplitudes and arbitrary shells — including 10^{2^k} — since no property of N_k is used.
- **Truncation invariance** (Palasek's Remark 1.10, and why his blow-up is unstable in every
  C^s) checked as an exact property of the vector field, with a control showing untruncated
  high shells do move.
- **The α > 2 threshold** obtained twice by unrelated routes.

```bash
python -c "import shell; from fractions import Fraction as F; print(shell.kolmogorov_exponent(1,1), shell.viscous_window(F(5,2)), shell.separation_budget(F(5,2)))"
```

## A fourteenth target: the open step — the amplifier gate in real Euler

`embedding.py` — Step 2 of Tao's program, for one gate.

### The question

Palasek's case for his model being embeddable rests on its two interactions being ones the real
Euler nonlinearity contains. One, `P_k div(u_{k+1}⊗u_{k+1})`, he calls harnessable by convex
integration. The other he flags as the hard one:

> *"The other Obukhov interaction, N_{k−1}^α X_{k−1}X_k, is **more difficult to harness**, but
> nonetheless is easily understood as the nonlinear interaction u_k·∇u_{k−1}."*

The worry is a coefficient mismatch. A triad k₁+k₂+k₃ = 0 with one low mode and two high ones
carries interaction coefficients of size |k_high| — for widely separated shells, enormous
compared with the |k_low| the Obukhov gate asks for. If that survived, the gate would be
unrealisable.

### It doesn't survive: the transport cancels identically

In the helical basis the Euler nonlinearity gives a₁′ = c₁ ā₂ā₃ cyclically, with
**c_j = −½·g·(s_{j+1}|k_{j+1}| − s_{j+2}|k_{j+2}|)** and a single geometric factor g shared by
all three. Energy conservation gives c₂ + c₃ = −c₁, so the **high pair's** energy moves at a
rate governed by c₁ alone — and c₁ sees the two high wavenumbers **only through their
difference**. The O(|k_high|) parts cancel identically (verified as an exact symbolic residual,
not asserted). Since k₂ + k₃ = −k₁, what remains is bounded by |k₁|.

That cancellation *is* the statement that the leading non-local interaction is pure transport —
it moves a small eddy without amplifying it. The residue is the low mode's **strain**, which is
exactly what the Obukhov gate needs.

### The gate law

In the scale-separated limit with the high pair carrying the **same helicity**:

> **|c₁| / |k₁| = |sin θ·cos θ| = |sin 2θ| / 2**,  maximised at **θ = 45°, efficiency exactly ½**

θ is the angle between the low mode and the high pair. Both factors are forced and pull against
each other: the wavenumber difference → |k₁||cos θ|, largest when the triad is collinear, while
the geometric factor → 2|sin θ| and **vanishes** for collinear triads — the classical fact that
collinear triads don't interact, since a divergence-free mode is orthogonal to its own
wavevector. Neither factor alone locates the optimum; only the product does.

Verified: the finite-wavenumber coefficient converges to the law (errors strictly decreasing at
Q = 10³, 10⁴, 10⁵), and the maximiser is located by search rather than read off the formula.

### What this settles, and what it doesn't

**Settled:** the coefficient question for one gate. There is no |k_high| mismatch to overcome;
the |k_low| the Obukhov model wants is exactly what Euler supplies, with a computable constant
of ½ and an explicit optimal geometry. The remaining N_{k−1}^α rather than N_{k−1} comes from
intermittency — the volume fraction N_k^{−2(α−1)} of Palasek's §2 — not from the triad.

**Not settled:** anything else. A real embedding needs this gate to act coherently across
infinitely many shells at once, with the errors from every *other* triad — the ones this module
deliberately examines one at a time — controlled. That is the open problem and nothing here
closes it. There is a test asserting the module contains no `embed` and no `blowup`, so the
scope stays attached to the code.

### Referees

- **Energy and helicity** conserved exactly and symbolically on every triad and every helicity
  assignment — the two identities that pin the coefficient convention, since nothing else here
  fixes it.
- **Collinear triads are inert** (g = 0), with a control confirming non-collinear ones aren't.
- **The transport cancellation** stated as a residual that must vanish, not as an assertion.
- **|‖k₂‖ − ‖k₃‖| ≤ ‖k₁‖** — the triangle inequality is what caps the gate at the low wavenumber.

```bash
python -c "import embedding as e, math; print(e.gate_efficiency(math.pi/4, 1e5))"
```

## A fifteenth target: coherence — many gates at once

`coherence.py` — an explicit mode architecture whose *entire* triad list is the Obukhov graph.

### The problem

`embedding.py` settles one gate. The reason Step 2 is hard is that a shell model prescribes a
very sparse **interaction graph** — shell j talks to j−1 and j+1 and nothing else — while the
true nonlinearity couples every triple of modes that closes. An embedding must suppress:

- **local triads** (j,j,j) — these drive the turbulent cascade that makes the Katz–Pavlović and
  exponential-shell Obukhov models globally regular;
- **long-range triads** (i,j,j) with i ≤ j−2 — pure error.

### The reduction

Reality forces S_j = −S_j, so S_j + S_j = S_j − S_j =: D_j, and all three requirements become
statements about **one difference set**:

| requirement | condition |
|---|---|
| no local triads | D_j ∩ S_j = ∅ |
| the gate exists | D_j ∩ S_{j−1} ≠ ∅ |
| no long-range triads | D_j ∩ S_i = ∅ for i ≤ j−2 |

The middle line supplies the model's *other* interaction free of charge. A triad
k_{j−1} + k_j + k′_j = 0 is simultaneously the amplification of the high pair and the drain of
the low mode — **Palasek's two Obukhov terms are the two ends of one triad**, which is why they
conserve energy against each other. There is nothing separate to arrange.

### The architecture

Two directions per shell suffice. Take d, e integer, orthogonal, equal length, and iterate

    p = A(d + e),   q = p − d,   (d, e) ← (p, A(e − d))

The recursion is closed in **closed form** — (d+e)·(e−d) = |e|² − |d|² = 0 — so no search for a
perpendicular partner is ever needed. Then S_j = {±p, ±q}, and by construction p − q = d is a
mode of shell j−1 (the gate), and angle(p, d) = **45° exactly** (the optimal efficiency).

Seeded with d = (3,4,0), e = (0,0,5), exhaustive search over all fourteen modes of a four-shell
instance finds **six closing triads, all six nearest-neighbour gates**:

> **local: 0.  long-range: 0.  gate: 2 per adjacent pair.**

The interaction graph is exactly the Obukhov graph. Nothing is filtered — every unordered triple
is examined, so an unwanted triad could not hide.

*(One overclaim I had to walk back: not every gate is at exactly 45°. One high mode is; the
other is q = p − d and misses by O(N_{j−1}/N_j) — 4.4° at separation ratio 10, 0.004° at 10⁴.
The two high modes differ by the low one, so they cannot both meet it at the same angle. Since
the model needs super-exponential separation anyway, the gap closes faster than any requirement
on it.)*

### How wide may a shell be? Sharp answer: 30°

Two modes per shell is space-filling, α = 1 — the wrong end of the intermittency range. Widening
a shell reintroduces local triads, and the threshold is **not** the obvious one.

For |a| = |b| = N, the sum a+b returns to the shell exactly at a **120°** opening. A cap of
half-angle φ spans pairwise angles [0, 2φ], suggesting φ < 60°. **That's wrong** — reality adds
the antipodal cap, contributing [180−2φ, 180], which reaches 120° already at φ = 30°. So:

> **a shell inside a cap of half-angle < 30° carries no local triads, and 30° is sharp.**

The reasoning that gives 60° was my first answer and it is wrong for a reason worth keeping: a
real velocity field cannot populate a cap without populating its antipode. Both are tests.

### The mode budget saturates exactly at α = 5/2

Concentrating a field onto volume fraction μ costs ~1/μ Fourier modes, and the intermittency
dictionary sets μ_k = N_k^{−2(α−1)}, so a shell needs **M_k ~ N_k^{2(α−1)}** modes. A dyadic
shell holds ~N_k³ lattice points and a 30° cap keeps a fixed fraction, so there is room exactly
when 2(α−1) < 3 — that is **α < 5/2**, saturating at 5/2. That is the top of the
three-dimensional intermittency range, which the model derives instead from the uncertainty
principle. Two unrelated routes to one endpoint.

### What this settles, and what it doesn't

**Settled:** the interaction graph. An explicit, exhaustively verified family of integer mode
sets whose complete triad list is the Obukhov graph and nothing else, every gate at the optimal
angle up to O(N_{j−1}/N_j), plus a sharp rule for how wide a shell may be.

**Not settled: everything analytic.** Which triads *exist* is not what the amplitudes *do*. Time
dependence, the Leray projection acting on products of non-monochromatic fields, the errors from
a solution not being a finite set of exact modes, and whether the resulting system reproduces
the Obukhov coefficients with the right signs throughout the evolution — none of it is here.
This is the combinatorial half of the coherence problem. The analytic half is the hard one.

```bash
python -c "import coherence as c; a=c.architecture((3,4,0),(0,0,5),[70]*3); print(c.triad_census(a))"
```

## A sixteenth target: the analytic half — what the amplitudes do

`cascade.py` — the exact Euler ODE system on the architecture, integrated and measured.

### Why this is answerable at all

`coherence.py` says which triads *exist*. The shell model is a statement about what the
amplitudes *do*. But because the triad list is finite and exactly known, **the Euler equations
restricted to the architecture are a finite ODE system** — writable, integrable, measurable.
Divergence-free condition, reality, Leray projection, true nonlinearity, nothing imposed.

### Three measurements

**1. The construction is sound.** Energy conserved to **1.8 × 10⁻¹²** over hundreds of turnover
times, and transport runs up the chain from shell 0 outward.

**2. The chain is what makes the cascade one-way.** A single gate is a triad, and triads are
integrable — they oscillate and give the energy back. Adding shells lets energy escape before it
can return. Fraction of exported energy recovered:

| shells | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| recovered | **0.66** | 0.26 | 0.16 | **0.14** |

So the cascade is a property of the **chain**, not of any gate. That's the mechanism the shell
model abstracts, seen here in the true equations.

**3. But the flux is not coherent — and that's the obstruction.** The instantaneous flux through
the bottom gate reverses sign after **0.8** time units against a turnover time of **0.22**, and
it reverses for *every* seed tried. Matched initial data diverges from the Obukhov trajectory by
5% within about one turnover.

### The surprise

Optimising the initial phases to maximise exported energy gains **0.08%**. The net transport is
essentially phase-independent. That cuts both ways and both are worth saying:

- **Against the shell model:** the Obukhov system describes *coherent* transfer and the embedded
  system does not transfer coherently. **The shell model is not a trajectory-wise description of
  the embedded dynamics.**
- **For the embedding:** the transport is **robust, not fine-tuned**. Over ~90 turnover times the
  bottom shell exports 10% of its energy regardless of how the phases are set. The cascade is a
  slow drift riding on a fast oscillation.

### Where that leaves Step 2

The three pieces together now say: the coefficient is right (`embedding`), the interaction graph
is right (`coherence`), and the transport is real and robust but **not pointwise Obukhov**. So an
embedding cannot proceed by matching trajectories to the shell model —

> it has to control a **time-averaged** flux, treating the oscillation as something to average
> over rather than something to suppress.

That is a sharper statement of the open problem than "the analytic half is hard", and it is the
useful output. It is **not** a solution: averaging arguments need error control over the
averaging window, uniformly in the shell index, and nothing here supplies that.

One further point the numerics make concrete: even with phases locked, a single gate reverses
once its low mode is exhausted — the amplitude equation drives the low amplitude through zero
and the signs flip. Sustained cascade needs the bottom shell **replenished**. That is exactly
why Palasek's viscous theorem carries an external force, and why his Remark 1.4 records that
forcing is *necessary*. Here it shows up as a property of the true Euler dynamics rather than of
the model.

**Scope.** Finite Galerkin truncations of Euler conserve energy and cannot blow up, so nothing
here tests blow-up — it tests transport: direction, reversibility, phase sensitivity. Time
integration is floating point, unlike the exact arithmetic elsewhere in this repo, because the
object measured is a trajectory; the conservation check at 10⁻¹² is what licenses reading
anything off it.

```bash
python -c "import cascade as c; print({n: round(c.return_fraction(n,600.0),3) for n in (2,3,4,5)})"
```

## A seventeenth target: the averaging estimate — half closed

`averaging.py` — the shell-index dependence is eliminated exactly; the phase tail is not.

### The gap that was left

`cascade.py` ended with an explicit admission: *"averaging arguments need error control over the
averaging window, uniformly in the shell index, and nothing in this module supplies that."* This
supplies it.

### The quantity

Define the **coherence penalty** at a gate as measured rate ÷ shell-model rate:

    ρ = (measured growth rate of X_j) / ( ½ · N_{j−1} X_{j−1} / √2 )

with the ½ the gate efficiency from `embedding` and the √2 sharing the shell's energy between
its two modes. What matters is not ρ's value but whether it stays bounded away from zero **as
the gate climbs the chain**.

### Uniformity comes from a symmetry

Euler is scale invariant, and the architecture is built by multiplying the previous shell by an
integer — so **gate j is a rescaled copy of gate 1** with the same separation ratio. The penalty
cannot depend on where in the chain the gate sits, only on the ratio. That's a theorem, and it's
checkable. Placing one gate geometry at absolute scales 1, 2, 4, 8, 16:

> **ρ = 0.089517 in every case** — spread 6 × 10⁻¹¹, i.e. solver-limited zero.

So ρ_j = ρ(r_j): one function of one variable, the same at every shell.

*(A mistake worth recording: amplitudes are normalised by 1/|k|, so the rate |k||u| is scale
free and the time window must be held **fixed**. Shrinking it in proportion to the scale — the
natural-looking thing — makes ρ appear to grow with absolute scale by 2×. That was my first
result and it was an artefact of measuring a different part of the trajectory. There's a test
pinning it.)*

### The separation limit

Measuring ρ(r) across separations 2.8 → 141 gives a clean two-parameter fit, residual 1.8 × 10⁻³:

> **ρ(r) = ρ∞ + c/r,  ρ∞ = 0.0869,  c = 0.0201**

ρ **decreases** toward its limit, so ρ(r) > ρ∞ at every finite ratio — the limit is a genuine
lower bound, not just an asymptote. And it is **positive**.

### What this gives, and what it doesn't

The penalty depends on two things: **ρ_j = ρ(r_j, φ_j)** — the separation ratio and the phase
configuration.

**The ratio dependence is eliminated, exactly.** Scale invariance makes it one function, the same
at every shell, approaching a positive limit from above. The shell index drops out of the
problem — which is precisely what "uniform in j" was asking for, and it is a theorem, not a fit.

**The phase dependence is not.** I first reported this as closed. It isn't, and the measurement
that settles it is the one I'd flagged as the thing to watch. Running the transfer-time
estimator — which cannot go negative, unlike the rate fit — over eight phase draws:

| separation | draws reaching target | min | median |
|---|---|---|---|
| 11.3 | 4/8 | 0.0632 | 0.0874 |
| 28.3 | 7/8 | 0.0423 | 0.0660 |
| 84.9 | 7/8 | 0.0433 | 0.0684 |
| **212.1** | **8/8** | **0.0078** | 0.0670 |

Medians are stable across a 20× range of separation — consistent with scale invariance, typical
behaviour is scale free. **Minima are not.** In the only sample where *every* draw reached the
target, the minimum is 0.0078: five times below what the incomplete samples showed, and an order
of magnitude below its own median.

That gap is exactly the bias `transfer_time_penalty` returns `None` to expose — the draws that
fail to reach the gain are the slow ones, so a sample that drops them reports a floor that
doesn't exist. **Reading the incomplete rows would have given me a uniform bound that isn't
there.**

So the estimate reads

> **Σ_j τ_j ≤ Σ_j τ_j^coherent / ρ(r_j, φ_j)**

with the r-dependence removed and the φ-dependence outstanding. Finite *provided* the phase
configurations at successive gates don't repeatedly land in the low tail. Controlling that tail —
not the typical value, which is fine — is what remains.

### Proved / measured / refuted

- **Proved:** ρ depends only on the separation ratio. Euler's scale invariance on a self-similar
  architecture; residual 6 × 10⁻¹¹, not a small number needing interpretation.
- **Measured:** ρ(r) and its limit ρ∞ ≈ 0.0869, at fixed phase draw.
- **Refuted:** that ρ is bounded below uniformly over phases. The complete-sample minimum is
  0.0078 and the apparent floor at 0.043 was an artefact of dropped runs. The
  separation-independence survives untouched, since it holds draw by draw — but the sum bound is
  now conditional on the phases rather than unconditional.

**Scope.** Two modes per shell (α = 1, the inviscid regime of Palasek's Theorem 1.8, which needs
only α ≥ 1). Galerkin truncation, not the PDE. Nothing here exhibits a blow-up — finite
truncations of Euler conserve energy and cannot — it bounds the *rate* against the model whose
blow-up is a theorem.

```bash
python -c "import averaging as a; print(a.scale_invariance_residual(), a.fit_penalty(), a.slowdown_factor())"
```

## An eighteenth target: the quantum coordination boundary in markets

`quantumcoord.py` — what's real, how big it is, and who can see it.

### Where the literature actually is

Two of the three usually-stated open problems have papers from **this year**:

- **Latency vs decoherence.** [arXiv:2604.07451](https://arxiv.org/abs/2604.07451) (Li, Kikura,
  Goban, Yamasaki, Sunami — April 2026) gives operational criteria for quantum advantage in
  latency-constrained tacit coordination, with finite operation times, finite entanglement rates,
  statistical certification, and hardware numbers: microsecond latency, 8×10³ decisions/s, 50 km
  metro network. The "critical latency threshold" is a framework, not a gap.
- **Many entangled agents.** [arXiv:2602.06367](https://arxiv.org/abs/2602.06367) (Hymas et al. —
  February 2026) builds a quantum stock market with RL agents and finds entanglement **stabilises**
  prices, removing the pathological pure-strategy Nash equilibrium that drives speculative
  collapse. Opposite to the usual guess that markets go chaotic.
- **Forensic detectability** is the one still open. That's what this attacks.

*(The Abushaqra "Quantum Time Dilation" reference is a March 2025 SSRN preprint, not a 2026
framework. And QAOA for triangular arbitrage can't beat classical "in real time" — negative-cycle
detection is exactly solvable in polynomial time by Bellman–Ford.)*

### A correction to the conjecture itself

"QCE ⊋ CCE" is **false** for complete-information games, in one line: a correlated equilibrium is
a distribution over action profiles obeying incentive constraints; a quantum device produces a
distribution over action profiles; if it obeys the constraints it *is* a classical CE, realisable
by a mediator who samples it and whispers recommendations. So QCE ⊆ CCE.

The advantage is real but lives elsewhere — **Bayesian games with private types and no mediator**,
where the classical resource is shared randomness and gives exactly the local (Bell) polytope. The
two-servers-see-a-local-signal story *is* that setting, so the mechanism survives; the label
doesn't.

Computed exactly: the CE polytope of the Prisoner's Dilemma has **1 vertex** (defect–defect,
uniquely) and a coordination game has 5.

**Corollary — your problem B dissolves.** A "Quantum Price of Anarchy" is asked for as though it
needs defining. For complete-information games it doesn't exist as a separate quantity: the two
equilibrium sets coincide, so every welfare ratio over them agrees identically. The Prisoner's
Dilemma has correlated PoA **exactly 1/3**, quantum or classical alike. The question only becomes
substantive once the mediator is removed and types are private. The local polytope, built from nothing but the 16
deterministic strategies, comes out with **24 facets in dimension 8** — 16 positivity + the 8 CHSH
inequalities, the textbook answer.

### How big the rent is: exactly √2, or exactly nothing

CHSH classical **3/4** (enumerated), quantum **(2+√2)/4 ≈ 0.8536** (derived from the distribution).
But the useful statement is across *all* such games. A two-input XOR game is a sign matrix M;
classical bias maximises over sign vectors, quantum bias is Tsirelson's. Over all 16:

> **8 of 16 admit advantage, every one at ratio exactly √2. The other 8 admit none, ratio exactly
> 1. The split is precisely rank-2 vs rank-1.**

There is **no continuum of quantum rents.** A market payoff is either rank-2 — in which case it's
CHSH in disguise and the bias improves by √2 — or it's worth exactly zero. Whether real order flow
presents rank-2 payoffs is an empirical question about markets, not about quantum mechanics, and
it's the question that decides whether any of this pays.

**And a bigger game does not buy a bigger rent.** Enumerating all 512 sign matrices at three
inputs gives a maximum ratio of exactly **6/5**, attained at rank 3 — strictly *below* the
two-input maximum of √2. The spectrum is {1, 1.0102, 1.2}: still discrete, no longer a dichotomy.

| inputs | max ratio |
|---|---|
| 2 | **√2 ≈ 1.4142** |
| 3 | **6/5 = 1.2** |
| any, weighted payoffs | ≤ K_G ≈ 1.7822 |

So the largest rent in this family sits at the *smallest* game — enlarging the coordination
problem shrinks the edge. Larger ratios need weighted rather than sign payoffs, and even then
Grothendieck's constant caps every XOR game at any size. There is a **universal ceiling** on the
quantum rent, and CHSH already sits within 25% of it. (The optimiser is licensed by reproducing
CHSH's 2√2 exactly.)

*(A bug the tests caught: I first used the nuclear norm as the quantum bias. It agrees at rank 2 —
giving CHSH's 2√2 — and fails at rank 1, returning **less** than classical, which is impossible
since quantum strategies include classical ones. The correct closed form is
max_c Σ_y √(2 + 2 s_y c) with s_y = M_{0y}M_{1y}.)*

### The forensic result

**From outcomes alone: detection is impossible, provably.** No-signalling forces the quantum
marginals to be independent of the other party's input, and for optimal CHSH every single-party
marginal is exactly **1/2**. A classical pair using one shared fair coin — "both act on the same
random bit" — wins 3/4 with *the same* marginals, exactly 1/2. The KL divergence between what a
one-sided observer sees is **exactly zero**. Not small. Zero. No quantity of order-book data
separates them; it's an identity, not a statistical difficulty.

**From joint statistics: easy.** With the private signals reconstructible, the divergence from the
quantum point to the whole local polytope is **0.0321 nats/round**, so decisive evidence needs
~**215 rounds** at 1e-3 and ~**431** at 1e-6 — under a second of tape at thousands of events/s.

> So the regulatory problem is not "can quantum collusion be detected". It is **"can the private
> inputs be reconstructed"** — an ordinary question about market data with no quantum content. If
> the trigger is a public event both servers saw, detection is easy. If genuinely private,
> impossible in principle.

### And a second threshold, which bites harder

A regulator doesn't see every decision — only those that reach the tape. Model that as a
visibility η, unseen events recorded as a default. Then a violation is certifiable only above a
critical visibility, and **below it a local model reproduces the record exactly** — the evidence
isn't weak, it's absent. For the maximally entangled strategy the threshold is exactly

> **η\* = 2(√2 − 1) = 0.82842712…**

bisected to eight digits against the closed form. Over partially entangled states it falls toward
Eberhard's 2/3 — but the advantage falls with it:

| θ | π/4 | 0.5 | 0.3 | 0.1 | 0.02 |
|---|---|---|---|---|---|
| threshold η\* | 0.8284 | 0.7597 | 0.7183 | 0.6824 | 0.6697 |
| advantage | +0.1036 | +0.0767 | +0.0371 | +0.0049 | +0.0002 |

The two move together, **monotonically**. A regulator cannot trade effect size against
observation quality: certifying the ten-percentage-point strategy needs **82.8%** of the
coordination events visible, and an observer limited to 2/3 can certify nothing whose advantage
isn't already negligible. That — not anything about quantum mechanics — is the binding constraint
on forensic finance here.

**Scope.** Two parties, two inputs, two outputs — where the polytopes are exactly computable and
the XOR classification is complete. Polytope, game values and the marginal identity are exact; the
detection divergence is a numerical convex optimisation and is labelled as such. Nothing here
models an order book, and no claim is made that real markets present rank-2 payoffs.

```bash
python -c "import quantumcoord as q; print(q.marginal_divergence(), q.detection_divergence(), q.rounds_to_detect(1e-3))"
```

## A nineteenth target: which spins a discrete structure can protect

`qca.py` asks the one exact question underneath "can a quantum cellular automaton
produce gravity": for which finite `G < SO(3)` does the spin-`s` multiplet survive
restriction as a single irreducible object? A multiplet that splits acquires an
independent coupling — and an independent velocity — per piece, so restoring
isotropy costs a tuned parameter for every piece beyond the first. A multiplet
that stays irreducible is protected outright by Schur's lemma.

One number answers both halves. The character norm

    <chi_l, chi_l>_G = (1/|G|) sum_g |chi_l(g)|^2

is simultaneously the sum of squared multiplicities (so it is 1 exactly when the
multiplet is irreducible) and the dimension of the commutant — the number of
independent invariant couplings. `tuning_cost` is that norm minus one.

Computing it over the full ADE classification `C_n, D_n, T, O, I`:

| spin | dimension | protected by | tuning cost on a cube |
| --- | --- | --- | --- |
| 1 (photon) | 3 | T, O, I | 0 |
| 2 (graviton) | 5 | **I alone** | 1 |
| 3 and above | 7+ | nothing | 2+ |

The `s <= 2` ceiling is derived, not assumed: the largest irrep of any finite
subgroup of `SO(3)` has dimension 5, and `5 = 2*2 + 1`. That is the
Weinberg–Witten massless-helicity bound, reached from finite group theory rather
than from a Lorentz-covariant stress tensor.

The second half is the crystallographic restriction theorem — periodic lattices
admit rotations of order 1, 2, 3, 4, 6 only, so none is icosahedral. Together:

> An emergent photon is symmetry-protected on ordinary lattices. An emergent
> graviton is protected only on icosahedral — hence quasicrystalline, hence
> aperiodic — structures. Nothing above spin two is protected anywhere.

Exactness is not decorative here: sympy's trigonometric simplifier fails to close
these cyclotomic sums already at a seventh of a turn. Every character sum is built
as a polynomial in `x` and reduced modulo the `L`-th cyclotomic polynomial, which
is arithmetic in `Z[zeta_L]` and needs no simplifier to be clever.

**Referees.** The polyhedral class data is confirmed by recomputing three classical
invariant degrees that were never fed in — lowest invariant at `l = 3` for the
tetrahedral group, `l = 4` for the octahedral (the cubic harmonic `K_4`), and
`l = 6` for the icosahedral. Every exact result is independently recomputed in
floating point, and the cyclic-group norms a third time by counting residues.

**A weaker argument that does not suffice, kept as a test.** Helicity on an
`n`-fold axis is defined only mod `n`, which gives `n >= 2s+1`, so `n >= 5` for
spin two — but a six-fold axis clears that and *is* crystallographic. The
single-axis argument alone leaves a periodic graviton open;
`aliasing_is_insufficient` asserts the gap.

### Relevant or irrelevant: the sharpened statement

"One tuned parameter" undersells the obstruction. `tuning_ladder` resolves the
cost by order in momentum, decomposing the couplings at order `k^n` as
`Sym^n(vector) x End(multiplet)` and comparing the `G`-invariant count against
the `SO(3)`-invariant one. The decisive rung is `n = 0`.

| | spin 1 (photon) | spin 2 (graviton) |
| --- | --- | --- |
| `T` | `k=0` free, first anisotropy at `n=1` | **splits at `k=0`**, 2 parameters |
| `O` | `k=0` free, first anisotropy at `n=2` | **splits at `k=0`**, 1 parameter |
| `I` | `k=0` free, first anisotropy at `n=4` | `k=0` free, first anisotropy at `n=2` |

An excess at `n = 0` is a splitting at *zero momentum*: the pieces of the
multiplet acquire different gaps, so no massless spin-2 object exists and
nothing suppresses the failure at low energy. That is a relevant perturbation.
An excess first appearing at `n > 0` is a velocity or dispersion anisotropy,
suppressed by powers of `k a` — an irrelevant operator that flows away in the
infrared. So:

> The cubic failure is **relevant**. The icosahedral failure is **irrelevant**.

**The spin-1 row is the control, and it is why the method is believable.** A
cubic lattice is isotropic through order `k^1`, so an emergent photon's linear
dispersion is protected by symmetry and the leading correction is an irrelevant
`k^2` term. That is precisely the regime in which emergent photons are known to
work. The same computation, on the same lattices, says spin two fails at `k^0`.

Two further checks fell out. `zero_momentum_splitting` and `tuning_cost` reach
the same integer by disjoint code paths — one from the character norm of the
multiplet, one from the degree-zero rung of the polynomial-operator
decomposition — and are asserted equal for every group and spin. And the
tetrahedral group permits a linear-in-`k` vector–quadrupole coupling that the
octahedral group forbids, because `l = 2` restricted to `T` contains the same
three-dimensional irrep as `l = 1` while under `O` they land in `T_1` versus
`E + T_2`. That is a known feature of non-centrosymmetric structures, and it was
not put in.

### States or fields — the strongest referee, and a weakening

The ladder treats the multiplet as five *states*. That is right for a gapped
quadrupolar excitation and wrong for a gauge field: a graviton has two
propagating helicities and no zero-momentum states to split. Re-asked for a
field, the object is the elastic tensor in `Sym^2(Sym^2(vector))`, and
`elastic_constant_count` reproduces the entire crystal-system table:

| system | group | count |
| --- | --- | --- |
| triclinic | `C_1` | 21 |
| monoclinic | `C_2` | 13 |
| orthorhombic | `D_2` | 9 |
| trigonal / tetragonal (low) | `C_3`, `C_4` | 7 |
| trigonal / tetragonal (high) | `D_3`, `D_4` | 6 |
| hexagonal | `C_6`, `D_6` | 5 |
| cubic | `T`, `O` | 3 |
| **icosahedral** | `I` | **2** |
| isotropic | `SO(3)` | 2 |

Twelve independent measured numbers, all correct, none of them put in — and the
last row is the known elastic isotropy of icosahedral quasicrystals.

**This weakens the no-go above, and the weakening is the point.** In the field
reading a cubic lattice does not gap the multiplet apart; it gives three elastic
constants where isotropy allows two, so the failure is a leading-order velocity
anisotropy — the Zener ratio `2 C_44 / (C_11 - C_12)` must be tuned to one.
Real, but tunable, and not the relevant-operator catastrophe the state reading
gives. So:

- emergent spin two **as excitations** → strong no-go on any lattice
- emergent spin two **as a gauge field** → one tuned relation on a cubic lattice

Both readings leave the icosahedral group free, and a test asserts they agree
across the whole classification on *which* groups need no tuning at all.

### Does the fracton route escape? No.

Symmetric tensor gauge theory is the known way around Weinberg–Witten: replace
`delta h_ij = d_(i xi_j)` with `delta A_ij = d_i d_j phi` (Pretko), and pay for
it with mobility restrictions on the charges. It does not escape this
obstruction, and the argument is one line:

> The obstruction is a property of the **field**, and both theories use the same
> field, `Sym^2(vector) = l0 + l2`. A different gauge parameter changes which
> polarisations survive; it cannot fuse two distinct point-group irreducibles
> back into one.

`field_splitting_is_theory_independent` asserts the equality across the whole
classification, and `fracton_evades_obstruction` returns `False` on every group.
Splitting values are identical for both theories: `T` → 2, `O` → 1, `I` → 0.

What the gauge parameter *does* change is the surviving helicity content:

| theory | gauge parameter | field − parameter | helicities |
| --- | --- | --- | --- |
| linearised gravity | `xi_i` (`l=1`), 1 derivative | 6 − 3 = 3 | `{0, ±2}` |
| scalar-charge fracton | `phi` (`l=0`), 2 derivatives | 6 − 1 = 5 | `{0, ±1, ±2}` |

Gravity's gauge parameter removes the helicity `±1` content entirely; the
fracton's does not. That is the rigorous distinction.

**A function that oversold, corrected.** The first version asserted the gauge
quotient isolates the `±2` pair — and it returned `False` for *gravity*, because
the quotient leaves `{0, ±2}` and the residual helicity-zero mode is the
Newtonian piece removed by the Hamiltonian constraint, not by gauge. Constraint
structure is not computed here. `carries_only_helicity_two` is now named for what
it measures: gauge orbits, not propagating modes.

**What this does not claim.** Protection is necessary, not sufficient.
Icosahedral symmetry keeps the multiplet whole and forces elastic isotropy; it
supplies neither a massless dispersion, nor diffeomorphism invariance, nor any
nonlinear structure. And fine-tuning is not impossibility — a cubic
model can still be tuned, at a cost of exactly one parameter.

**Novelty.** Every ingredient is classical: Klein's classification, the
crystallographic restriction, the character theory, and the textbook cubic
splitting `l = 2 -> E_g + T_2g`. What is assembled is the conjunction read as a
no-go, plus the commutant reading that turns the same norm into a fine-tuning
count. I could not run the search myself (arxiv.org is egress-blocked here and web
search was rate limited); the literature check was carried out by the repository
owner, who reports no prior statement of the conjunction. The novelty claim
therefore rests on someone else's search, not mine. The arithmetic is exact
independently of that.

## A twentieth target: what logic actually costs

`thermo.py` answers the proposed "Goedel–Landauer–Prigogine trilemma" — that a
physical computer cannot be simultaneously consistent, efficient and thermally
stable, with heat diverging exactly when the logic becomes complete. No such
conjecture exists in the literature. Underneath it there are two real and
**separable** effects, and conflating them is where the framing goes wrong.

| claim | verdict |
| --- | --- |
| erasing perfectly costs infinite heat | **false** — bounded by `ln 2` |
| there is a dissipation singularity | **true, twice**, neither where the brief puts it |
| it is a phase transition | true of one, false of the other |
| the trilemma is the TUR | partly — the TUR says nothing about logic |

**Maintenance is logarithmic, not singular.** A bit held at error `eps` against
a bath of rate `gamma` is an exactly solvable two-channel Markov jump process.
In the reliable limit the demon flux saturates at the noise rate, `|J| -> gamma`,
while the affinity grows as `2 ln(1/eps)`, giving

    Sdot  ->  2 * gamma * ln(1/eps)

Consistency does cost, and the cost does diverge — but logarithmically, with a
coefficient of exactly **twice the noise rate**. Every decade of reliability
costs the same fixed increment. Nothing here deserves the word "transition".
The coefficient `2` is checked by confirming that 1, 3 and 4 all *fail*, so it
is derived rather than fitted.

**Erasure is bounded, so the headline claim is false.** `W(eps) = ln 2 - H(eps)`
rises monotonically to `ln 2` and stops. Perfect erasure is finitely priced. The
brief's error is to attribute the maintenance divergence to erasure.

**The real phase transition is the fault-tolerance threshold.** Majority-vote
concatenation gives `p' = 3p^2 - 2p^3`, fixed points `0`, `1/2`, `1`. The
unstable one at `p = 1/2` is a genuine critical point:

| `p` | reachable? | levels | gates | dissipation |
| --- | --- | --- | --- | --- |
| 0.100 | yes | 4 | 81 | 1.1e2 `kT` |
| 0.400 | yes | 8 | 6561 | 9.1e3 `kT` |
| 0.490 | yes | 14 | 4782969 | 6.6e6 `kT` |
| 0.499 | yes | 19 | 1162261467 | 1.6e9 `kT` |
| **0.500** | **no** | — | — | **unattainable** |
| 0.510 | no | — | — | unattainable |

Finite on one side of a sharp parameter value, unattainable on the other, with
overhead exponent `log 3 / log 2 = 1.585`. That is the dissipation singularity,
and it sits at a **noise** value, not at "logical completeness".

**Critical slowing down, derived.** Linearising the map at the fixed point gives
`d/dp (3p^2 - 2p^3) = 6p(1-p)`, exactly `3/2` at `p = 1/2`. So distance from
threshold grows geometrically with ratio `3/2` and escape takes
`ln(1/delta)/ln(3/2)` levels — 5.68 per decade, measured 6, 5, 6. The test checks
the *slope*, since the absolute count carries an offset from the doubly
exponential phase that follows.

**Referees.** The mean current read off the cumulant generating function of the
tilted generator must equal the steady-state cycle flux — two disjoint paths,
agreeing to 1e-12. The TUR is checked to hold *and* separately checked not to be
vacuous (it approaches 2 near equilibrium). The floating-point orbit used near
the threshold is checked against the exact rational map.

**A test that failed and was right to.** `test_flux_saturates_at_the_noise_rate`
originally used a fixed tolerance and passed at `gamma = 1` while failing at
`gamma = 3`: saturation is asymptotic in the affinity and the residual carries a
factor `gamma/gamma_D`. It now checks that the gap *shrinks*, which is the claim
that was meant.

**What this does not claim.** No Gödel statement enters at any point, and a test
asserts it. The maintenance law, the erasure bound, the TUR and the threshold
are all statements about noise, current and gate count. The incompleteness half
of the conjecture contributes nothing, and pretending otherwise would be the
overclaim.

## A twenty-first target: exact circuit complexity of stabilizer states

`complexity.py` answers the Brown–Susskind brief — does optimising an algorithm
shrink the wormhole? **No, and it needs no computation.** Complexity is *defined*
as the minimum gate count over all circuits preparing a state. An optimised
circuit prepares the same state and was already in the set the minimum ranged
over. Nothing shrinks. (The random-circuit side is also less open than the brief
suggests: Haferkamp, Faist, Kothakonda, Eisert and Yunger Halpern proved linear
growth of exact complexity there.)

The question that survives is about *states*, and on stabilizer states it is
decidable rather than estimable. Breadth-first search from `|0...0>` under
`{H, S, CNOT}` returns the exact minimal gate count for every reachable state —
a true minimum over all circuits, not a bound from one construction.

**The referee.** The number of `n`-qubit stabilizer states is known in closed
form, `2^n prod (2^k + 1)` = 6, 60, 1080, 36720. The search must enumerate
exactly that many, and does. Nothing in the tableau, canonical form or gate
rules was built to make that come out — a sign error in the Pauli-product
bookkeeping or a canonical form that failed to identify two descriptions of one
state would both show up as a miscount. One number validates the apparatus.

**What came out, which is not what I expected.**

| quantity | law | n=1 | n=2 | n=3 | n=4 |
| --- | --- | --- | --- | --- | --- |
| diameter | `3n + 1` | 4 | 7 | 10 | 13 |
| mean | `~2.4n` | 2.17 | 4.45 | 6.86 | 9.44 |
| `|+>^n` | `n` | 1 | 2 | 3 | 4 |
| GHZ | `n` | — | 2 | 3 | 4 |
| line graph | `2n - 1` | — | 3 | 5 | 7 |
| complete graph | `3(n - 1)` | — | 3 | 6 | 9 |

Confirmed at `n = 5` by a full search of all **2 423 520** states (921 s):
diameter 16, plus 5, GHZ 5, line graph 9, complete graph 12, mean 12.1871. Every
law holds, and the state count matches exactly.

I built this expecting structured states to sit far below typical ones with the
gap widening in `n`. **The search refutes that.** GHZ and the complete-graph
state are both maximally structured — each is a one-line rule — yet they differ
by a factor of `3 - 3/n`, and the complete-graph state sits *exactly four gates*
below the diameter at every size measured, essentially saturating it. So
`structure_gap` **closes** (+1.17, +1.45, +0.86, +0.44, +0.19) instead of widening.

### Where the observed laws stop being true

The measured trends invited a concentration claim — `mean - 3(n-1)` running
1.45, 0.86, 0.44, 0.19 and `diameter - mean` running 1.83, 2.55, 3.14, 3.56,
3.81, apparently converging to 4. **Both are small-`n` artifacts, and counting
proves it.**

There are `N(n) ~ 2^(n^2/2)` stabilizer states and only `|G| = n^2 + n` gates,
so a ball of radius `L` holds at most `|G|^L` states and the diameter must grow
like `n^2 / (4 log_2 n)` — superlinear. Therefore:

- **`diameter = 3n + 1` is false.** It is exactly right at every size the search
  reaches and cannot hold in general. At **`n = 72`** the counting bound alone
  gives 219 against `3n+1 = 217`, and the curves never cross back.
- **`diameter - mean -> 4` is false** for the same reason: the diameter grows
  and the structured states do not, so the gap must diverge.

Both statements coexist without contradiction — the counting bound sits far
*below* `3n+1` at small `n`, which is exactly why five points proved nothing.

**What survives, now proved rather than observed:**

- **GHZ complexity is exactly `n`, for every `n`.** At least one Hadamard is
  needed, since `CNOT` and `S` map computational basis states to computational
  basis states and cannot create a superposition. At least `n-1` controlled-nots
  are needed, since the two-qubit interaction graph must be connected or the
  output factorises along a disconnected cut. The two counts are disjoint, and
  `ghz_state` attains the bound.
- **The concentration is real and stronger than measured.**
  `cheap_fraction_bound` bounds the fraction of states with complexity `<= n` by
  `|B(n)|/N(n)` with no search at all: **7e-18** at `n = 20`, **2e-131** at
  `n = 40`. Cheap states vanish superexponentially.

So the mechanism I claimed was right and its arithmetic was wrong. Almost every
stabilizer state does sit far above the structured ones, leaving no room below —
but the separation **grows without bound** rather than saturating at four. The
conclusion is strengthened by the correction, not weakened.

> Optimising the **algorithm** changes nothing, by definition. Choosing a
> structured **state** buys nothing in general — some structured states are as
> hard as anything there is.

**Two tests that failed and were right to.**
`test_structure_does_not_imply_low_complexity` first asserted the
complete-graph/GHZ ratio exceeds 2. It is `3 - 3/n`, which is *exactly* 2 at
`n = 3`. And `test_this_is_why_structure_buys_nothing` used a fixed 2% cutoff on
the cheap-state fraction, set without looking at `n = 3` where it is 3.3%. Both
claims were about a *trend*; both tests had been written as thresholds.

**What this does not claim.** The laws above are read off five points, and two
of them are now known false in general (see above) — the tests are the claim,
not the formulas. And by Gottesman–Knill every
state here is classically simulable, so this measures exact minimal *Clifford*
complexity and says nothing about quantum advantage or about the states Shor's
algorithm produces. No volume, no tensor network, no bulk is constructed.

## A twenty-second target: what the persistent Dirac operator can and cannot see

`dirac.py` answers the conjecture that the persistent Dirac operator
`D = d + delta` (with `D^2 = Delta`) classifies geometric **chirality** where
persistent homology is blind, because `D` is first-order and "retains sign and
orientation" that the second-order Laplacian squares away.

**The conjecture is false, and it fails twice, for two independent reasons** —
but the target is the brief, not the literature. A literature check (run by the
repository owner; arxiv.org is unreachable from this environment) established:

- The brief's citation `arXiv:2208.06456` **does not support the chirality
  claim**. The relevant work is `arXiv:2301.10137` and `arXiv:2105.00529`.
- What the topological deep learning literature actually claims is that the
  persistent Dirac spectrum beats persistent *homology*, by retaining the
  non-harmonic spectrum. **That claim is true**, and it is exactly the one
  recorded under "what actually survives" below.
- The chiral-symmetry equivalence proved here is **already established**, as
  standard Hodge theory.

So the published claim is correct and modest, and the brief inflated it into a
statement about chirality its own citation never made. This module refutes the
inflation. Nothing here is a criticism of the cited work, and **nothing here is
novel** — the result that looked most like a contribution, the spectral
equivalence, is the one confirmed to be known.

### One: the Dirac spectrum carries no more than the Laplacian's

Let `Gamma` be the grading operator, `+1` on even-degree chains and `-1` on odd.
Both `d` and `delta` shift degree by one, so both anticommute with it:

    Gamma D = - D Gamma        (verified exactly zero on every test complex)

`Gamma` is a unitary involution, so `D psi = lambda psi` implies
`D (Gamma psi) = -lambda (Gamma psi)`. The spectrum is symmetric about zero, and
with `D^2 = Delta` that pins it completely:

    spec(D) = { +- sqrt(mu) : mu in spec(Delta) },  signs forced

`dirac_spectrum_from_laplacian` reconstructs `spec(D)` from `spec(Delta)` with no
other input, and it agrees on every complex tested. **The two carry identical
information.** The first-order-ness is real, but it buys expressiveness in the
*eigenvectors*, which mix degrees — not in the spectrum, which is what the
conjecture is about.

Consequence: the Dirac operator inherits *every* Laplacian blindspot. An
exhaustive search finds a Laplacian-cospectral non-isomorphic pair of graphs at
six vertices (none exist at five), and the two share a Dirac spectrum exactly.

### Two: nothing built from distances can see chirality

This one does not involve the Dirac operator at all. A reflection is an isometry,
so a chiral point cloud and its mirror have **bitwise identical** distance
matrices — the measured difference is exactly 0.0, not small. Every
distance-based filtration (Vietoris–Rips, Čech) is therefore the *same filtered
complex*, and every invariant of it agrees: persistent homology, persistent
Laplacian, persistent Dirac.

| | original | mirrored |
| --- | --- | --- |
| orientation signature | `(1, 1, -1, 1, 1)` | `(-1, -1, 1, -1, -1)` |
| distance matrix | — | **identical** |
| Rips complex (r = 1.1, 1.5, 2.0) | — | **identical** |
| Dirac spectrum | — | **identical** |

So chirality is not a function of the distance matrix. What *does* flip is the
signed volume — and it is not recoverable from pairwise distances, which is
precisely why no distance-based filtration can reach it. Persistent homology is
blind to chirality for this reason, and it has nothing to do with squaring.

### What actually survives

One part of the brief is true and worth keeping — and it is what the literature
actually claims: **the non-zero spectrum carries strictly more than persistent
homology.** Homology reads only the kernel — the
Betti numbers — and discards every non-zero eigenvalue (`homology_discards`
counts them: 4 for a circle, 12 for a 2-sphere). That gap is real and is what
makes spectral methods worth using. But it is a statement about *homology versus
the Laplacian*, not about *the Laplacian versus Dirac*, where the answer is that
they are equivalent.

**Referees.** `partial . partial = 0` on every complex — without correct boundary
signs nothing else would mean anything. `dim ker D` equals the sum of the Betti
numbers, and the Betti numbers themselves come out right on complexes whose
homology is known independently: circle `(1,1)`, disk `(1,0,0)`, 2-sphere
`(1,0,1)`. All in exact arithmetic; the shared cospectral spectrum prints as
exact algebraic numbers like `-sqrt(sqrt(5) + 3)`.

**What this does not claim.** No persistence module is built, because neither
refutation needs one — the second says every complex in the filtration is
identical, which is stronger than any statement about the module above it. And
only the *spectra* are shown equivalent: the Dirac eigenvectors genuinely mix
degrees in a way the Laplacian's do not, and nothing here says otherwise.

## A twenty-third target: curvature kills supersymmetry but not chirality

`magnetic.py` follows `dirac.py`. Having shown the *real* simplicial Dirac
operator carries exactly what the Hodge Laplacians carry, the standard escape is
a `U(1)` connection — a magnetic Dirac operator — on the stated grounds that
`D^2 != Delta` there and **that this breaks the spectral symmetry**.

The grounds are wrong. The conclusion is right for a different reason.

**One. Chiral symmetry survives any connection.** `Gamma D + D Gamma = 0` holds
for every connection, flat or curved, *exactly* — asserted at literal zero, not
a tolerance. The proof uses only that `d` raises degree by one and `delta`
lowers it; it never touches `d^2 = 0`. Magnetic phases do not break chirality.

**Two. What curvature breaks is `D^2 = Delta`.** Working out the composition,

    (d_1 d_0 c)([u,v,w]) = (sigma_uv sigma_vw - sigma_uw) c(w)

so `d^2 = 0` exactly when every triangle has trivial holonomy. Three separately
computed quantities turn out to be one number:

| | flat | curved |
| --- | --- | --- |
| holonomy defect \|hol − 1\| | 0 | **0.397339** |
| \|d²\| | 0 | **0.397339** |
| `D²` off-block norm | 0 | **0.397339** |
| `{Gamma, D}` | **0** | **0 exactly** |
| spectrum asymmetry | 1e-15 | 1e-15 |

Statements one and two come apart *exactly* at curvature. Supersymmetry dies;
the grading does not.

**Three. There is a chiral asymmetry, and it is the index.** The `±` pairing
holds on the non-zero spectrum but not on the kernel — the even and odd harmonic
spaces differ in dimension by `sum_k (-1)^k beta_k`, the Euler characteristic:
disk 1, sphere 2, path 1, circle 0. This asymmetry is **topological, present at
zero flux, and exactly what persistent homology already reports.** The one
genuine chirality asymmetry in the operator is not geometric and not new
information.

**Four.** So the magnetic Dirac operator *does* carry more than its Laplacians —
because `D²` is no longer block diagonal, `spec(D_sigma)` is not reconstructible
from the twisted Laplacian spectra, unlike the flat case. Curvature does change
the spectrum: the flat triple `±sqrt(3)` splits into `±1.6133, ±1.7321, ±1.8432`.
The hope behind the magnetic proposal is correct; its stated mechanism is not.

### Statement five — and this one is not assembly

The literature check named the exact obstruction to a stability theorem for
spectral persistence: *filtration change alters the operator's dimension, and
interleaving only controls kernels.* The nearest existing result, for the real
persistent Laplacian, is a **Lipschitz** bound under one-simplex insertion
(Anh–Dik–Anh, arXiv:2506.21352).

A Lipschitz bound is weaker than what is available. Inserting a simplex adds one
basis element, and **the new simplex has no cofaces** — nothing above it can
already contain it, by closure — so the Dirac matrix gains exactly one row and
one column and *no existing entry changes*. That is a bordered Hermitian matrix,
and bordered Hermitian matrices interlace:

    lambda_i(D')  <=  lambda_i(D)  <=  lambda_{i+1}(D')

Three consequences:

- **The connection is irrelevant.** Interlacing constrains where the new
  eigenvalues land, not what the new entries are — and the connection only
  touches the entries. So the bound is **uniform over all connections** and
  curvature cannot degrade it. That answers the stated worry that a holonomy
  error term might destabilise the descriptor: for this descriptor, it cannot.
- **It survives the dimension change**, which is precisely what blocks
  interleaving arguments.
- **The counting function moves by at most one.** If `lambda_k <= t <
  lambda_{k+1}`, interlacing traps the new count in `{k, k+1}`, so
  `|N'(t) - N(t)| <= 1` off the spectrum — the spectral counting function is
  1-Lipschitz in insertions along a filtration.

Verified on **308 randomised complexes** with random connections: **zero**
interlacing failures, and **zero** counting violations across 12 320
off-spectrum thresholds. Curvature *anti*-correlates with the eigenvalue shift
(−0.35) — it damps rather than amplifies.

**A test artifact worth recording.** A first sweep reported 9 counting
violations. Every one sat at exactly `t = 0`, where the harmonic modes lie at
machine epsilon with mixed signs, and none occurred away from a tie. The test
was wrong, not the theorem — and the ambiguity lives exactly where statement
three located the operator's one genuine asymmetry. `counting_is_stable` now
returns `None` on a tie rather than a coin-flip boolean.

### Statement six — the useful form of five

`lambda_i(D') <= lambda_i(D)` is not merely a bound. It is **monotonicity**:
along a filtration each eigenvalue index traces a monotone curve, with *no
stability constant needed* and no genericity assumption to keep the indexing
well defined. Composing over `m` insertions:

    lambda_i(D^(m))  <=  lambda_i(D)  <=  lambda_{i+m}(D^(m))

so the spectral counting function is **`m`-Lipschitz over a filtration segment
adding `m` simplices, uniformly in the connection.**

Combined with statement one, the spectrum **spreads symmetrically**. Measured
over a 21-step filtration with a random connection, the two ends stayed exact
mirrors at every step while the spread grew monotonically:

| step | size | λ_min | λ_max | spread |
| --- | --- | --- | --- | --- |
| 1 | 7 | −1.41421 | 1.41421 | 2.82843 |
| 5 | 11 | −2.44949 | 2.44949 | 4.89898 |
| 12 | 18 | −2.67731 | 2.67731 | 5.35463 |
| 21 | 27 | −2.83573 | 2.83573 | 5.67145 |

Zero monotonicity failures, zero interlacing failures across all 21 steps.

Monotonicity is the part that matters practically: a descriptor built from "the
k-th eigenvalue" is ill-defined if indices can swap, and the flat spectra here
are heavily degenerate. Monotonicity in the index removes the question entirely.

**Scope, stated because it is easy to overclaim.** This is stability under
**combinatorial** change — inserting a simplex — not under **metric**
perturbation of an underlying point cloud. The metric case remains open and
nothing here touches it.

**Status.** A literature check found **no source stating either the true or the
false version** of statement one for a non-flat connection. The pieces exist —
Calmon–Schaub–Bianconi (arXiv:2301.10137) prove the `±` pairing from
block-off-diagonal structure alone, never invoking `d^2 = 0`, so their argument
extends verbatim; Egidi–Gittins–Habib–Peyerimhoff (arXiv:2211.08019) study the
continuum `d_alpha = d + i alpha wedge` with `d_alpha^2 != 0` — but nobody has
put them together. Treat statements one to four as a computed assembly of known pieces whose
conjunction appears unstated. Statement five is the original part: Cauchy
interlacing is classical and simplex insertion is elementary, but putting them
together to get a connection-uniform stability statement that survives the
dimension change is not in the literature, which offers a Lipschitz bound for
the real case and nothing for the magnetic one.

**Two cautions recorded from the same check**, bounding what may be claimed:
the diamagnetic inequality **fails** for magnetic Hodge Laplacians above degree
zero, so the intuition that flux only raises the spectral gap does not survive;
and no bottleneck- or Wasserstein-stability theorem exists for raw
eigenvalue-valued persistence descriptors, the obstruction being structural —
eigenvalues are not functorial under interleaving and operators change dimension
along a filtration. At *fixed* combinatorics Weyl gives 1-Lipschitz dependence
for free, which is all `weyl_bound_holds` claims.

## A twenty-fourth target: local spectral measures, girth, and Wilson action

`insertion.py`. **A correction first, because the first version overclaimed.**
The object is *not* new — it is the **local density of states** of Savostianov,
Guglielmi, Schaub and Tudisco (arXiv:2502.07558, Def 4.1), whose Chebyshev
moments `2[T_m(H)]_jj` are already walk moments at a simplex. That claim is
withdrawn and a test asserts the withdrawal.

What appears unclaimed is narrower: taking the measure **at the moment a simplex
enters a filtration** (a two-parameter (simplex, scale) object — the literature
computes it at fixed scale), taking it for the **Dirac** operator rather than a
Hodge Laplacian, and the moment results below.

**One. Every odd moment vanishes**, any connection — `Gamma` acts on `e_tau` by
a sign and anticommutes with `D`.

**Two. At zero flux the measure is known completely.** `M_2j = (k+1)^j` exactly
— integers `2,4,8`; `3,9,27`; `4,16,64`; `5,25,125`; `6,36,216`. With vanishing
odd moments that determines it outright:

    mu_tau = ½ delta_{+sqrt(k+1)} + ½ delta_{-sqrt(k+1)}

A symmetric Bernoulli measure supported on the square root of the face count.

**Three — the girth law. Flux enters the local moments at order exactly `2g`**,
where `g` is the shortest bounding cycle through the simplex:

| structure | girth | first flux-bearing moment |
| --- | --- | --- |
| tree | ∞ | **none** (blind at M2–M10) |
| 2-simplex | 2 | M4 |
| edge, graph girth 3 | 3 | M6 |
| edge, graph girth 4 | 4 | M8 |

This is the local form of a Kesten–McKay fact: a tree is simply connected, so
every connection on it is gauge-trivial and the measure at its root cannot depend
on phases. **A closed walk sees flux only once it is long enough to enclose
something.** The law subsumes statement two — `g >= 2` always, so `M_2` is
flux-blind everywhere and equals the face count.

**Four — the fourth-moment law, in every dimension.** For any simplex, any
dimension, any connection, any ambient complex:

    M_4(tau)  =  M_2(tau)²  +  S(tau)  +  sum_g |1 - omega_g|²

Three terms with disjoint meanings, and **the separation is the result**:

| term | meaning | sees the connection? |
| --- | --- | --- |
| `M_2²` | the two-point Bernoulli baseline | no |
| `S(tau)` | sibling count: facets shared with another same-dimension simplex | no |
| `sum_g \|1 - omega_g\|²` | Wilson action over the Hasse plaquettes at `tau` | **yes** |

Each codim-2 face `g` lies in exactly two facets `f_1, f_2`, so
`tau -> f_1 -> g -> f_2 -> tau` is a canonical Hasse plaquette; `omega_g` is its
holonomy, normalised so `omega_g = 1` at zero flux — **the normalising sign being
exactly the `d² = 0` cancellation.** Each is gauge-invariant (checked to `1e-16`
under vertex gauge transformations). Verified to `1e-15` on lone simplices of
dimension 2–4, triangle fans, a tetrahedron shell, and pairs of tetrahedra.

So **the fourth moment separates combinatorics from curvature exactly.** The
first two terms are blind to the connection; the third is the only place it
enters.

**Retraction.** An earlier version reported this identity as dimension-two only.
That was two mistakes, not an obstruction: the sum was indexed over the simplex's
**triangular** faces when the plaquettes live on its **codimension-two** faces —
for a tetrahedron, its 6 edges, not its 4 triangles — and the sibling term was
missing. `identity_fails_above_dimension_two` is kept in the API, now returning
`False`, so the retraction stays visible instead of vanishing.

**Naming.** The right-hand side is a Wilson plaquette action, not a squared
curvature — `|1-omega|²` and `2(1 - Re omega)` are identical for unitary holonomy,
but `|F|²` is recovered only in the continuum small-flux limit.

Two structural precedents: **Kenyon** (Ann. Probab. 39, 2011) weights
cycle-rooted spanning forests by `2 - tr(hol)`, exactly this summand for `U(1)` —
the same quantity in a determinant identity rather than a moment one. And
**Chamseddine–Connes** (hep-th/9606001) show the fourth heat-expansion
coefficient of `Tr F(D/Lambda)` contains Yang–Mills; this is a discrete
*localised* analogue, localised being the operative word since existing discrete
work (arXiv:2509.04311) takes global traces.

### Five — the filtration invariant

Sum the fourth-moment law over an entire filtration. Individually the terms move
with the insertion order — a simplex entering early sees fewer siblings and a
smaller subcomplex. **The totals do not.**

    sum_tau M_4(tau)  =  B(K)  +  P(K)  +  W(K)

with `B` the sum of squared facet counts, `P` the facet-sharing pairs (each
counted once, when the second arrives), and `W` the **total Wilson action of the
complex**. All three are independent of which linear extension of the face poset
is used — verified across six random orders per complex, spread `0` to `2.8e-14`,
residual `0` to `7e-15`.

Rearranged, it is a recovery statement:

    W(K)  =  sum_tau M_4(tau)  -  B(K)  -  P(K)

**A global gauge-theoretic quantity from strictly local spectral data** — each
moment computed on a subcomplex, at the moment one simplex entered, with no
global operator ever formed. Chamseddine–Connes obtain Yang–Mills from the fourth
heat-expansion coefficient of a *global* trace `Tr F(D/Lambda)`, and the existing
discrete work (arXiv:2509.04311) also takes global traces. This assembles the
same order of the same expansion from local, filtration-adapted pieces.

**A dimension-zero correction the sum surfaced.** Statement two originally read
`M_2 = dim + 1` for every simplex. It is **wrong for a vertex**: a 0-simplex's
only facet is the empty face, which is not a simplex, so `M_2 = 0`. Every earlier
check used `dimension >= 1`, so the case was untested until vertices became
unavoidable in a filtration. The same oversight made `sibling_count` treat every
other vertex as a sibling, since `combinations(tau, 0)` is the empty tuple and
the empty set is contained in everything. Both fixed; the corrected statement is
**`M_2 = number of facets present`**.

## Reference document

`docs/REFERENCE.md` is the standing write-up: every result this repository
establishes, what it refutes, and what it leaves open, with the formulas and
citations in one place. It covers all ten investigation threads, the ledger of
solved versus open problems, and the failure catalogue.

Read §12 first if you read nothing else — twelve wrong guesses that died in
computation, the mechanical traps that produced them, and the one pattern behind
two of them (claims about a *trend* written as tests against a *threshold*).

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
python -m pytest test_platycosm.py -v           # flat 3-manifolds / instability
python -m pytest test_graviton.py -v            # obstruction = Friedmann constraint
python -m pytest test_celestial.py -v           # massive w_1+inf / celestial obstruction
python -m pytest test_cosmopolytope.py -v       # cosmological polytope / mass resummation
python -m pytest test_shell.py -v               # Obukhov shell model / blow-up window
python -m pytest test_embedding.py -v           # Euler triad gate / Tao's step 2
python -m pytest test_coherence.py -v           # mode architecture / interaction graph
python -m pytest test_cascade.py -v             # Euler transport on the architecture
python -m pytest test_averaging.py -v           # the averaging estimate
python -m pytest test_quantumcoord.py -v        # quantum coordination / Bell polytopes
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
