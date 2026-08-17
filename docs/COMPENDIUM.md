# The Narb Notebook — Compendium of Equations and Literature

*Every formula this repository derives or uses, and every work it cites, in one
file.*

Companion to `docs/REFERENCE.md`, which gives the arguments. This gives the
statements and the sources. Each equation is labelled with its module and its
status: **P** proved here, **M** measured/computed here, **C** classical (not
mine), **R** retracted.

**Status at time of writing:** 4321 tests passing, branch
`claude/tetrahedron-packing-spectral-u7029m`.

**On citations.** Author, title and year are reliable. arXiv identifiers and
novelty claims were verified by the repository owner through external searches;
`arxiv.org` is unreachable from the build environment, so nothing here was
checked by the code that produced it. Two of those sweeps overturned a framing
rather than confirming one.

---

# PART I — EQUATIONS

## 1. ADM obstruction and the Friedmann constraint
*Modules: `adm.py`, `graviton.py`, `platycosm.py`*

**Taub obstruction on a flat compact slice with `K = 0`** — C

```
Q = ⟨R⁽²⁾(h)⟩ + ⟨(tr K)² − |K|²⟩ = 0
```

**Transverse-traceless sector** — P. With `h = A cos(k·x)`, `A` TT, `ḣ = 2K`:

```
−Q_k = ⅛(Ȧ² + |k|²A²)     ⟹     ω² = |k|²
```

Massless, two polarisations. Controls: pure-trace gives `ω² = −(5/3)|k|²` (the
conformal factor problem); linearised diffeomorphisms `h_ij = k_i ξ_j + k_j ξ_i`
give exactly zero.

**Isaacson normalisation** — M. `−Q_k = 16πρ` exactly, every mode and amplitude.

**The homogeneous mode.** At `k = 0` the momentum term has inertia `(5,0,1)`.
With `B₀ = −Hδ + σ`, summing all modes and imposing `Q = 0` — P:

```
3H² = 8πρ + ½ σ_ij σ^ij
```

The Friedmann constraint with shear, sourced by graviton energy. Every
coefficient — the 6, the 16π, the ½ — is computed output.

**Averaging weights** — the trap that hid an error: `⟨cos²(k·x)⟩ = ½` for `k ≠ 0`
but `⟨1⟩ = 1` at `k = 0`.

**Platycosm shear budget** (trace-free invariant momenta) — M: torus 5, dicosm 3,
tetracosm 1, Hantzsche–Wendt 2. Matches classical moduli-space dimensions.

## 2. Massive w₁₊∞ — an obstruction at integer Δ
*Module: `celestial.py`*

**Wedge algebra** — C:

```
[w^p_m, w^q_n] = [m(q−1) − n(p−1)] w^{p+q−2}_{m+n},   |m| ≤ p−1
```

**The collapsing identity** — P. With bulk-to-boundary propagator
`G_Δ = (−p̂·q̂)^{−Δ}`, the whole derivation reduces to:

```
u·u_{zz̄} − u_z·u_z̄ = 1
```

**Bidiagonality** — P:

```
−n·p̂ |Δ; i,j⟩ = (Δ−1)⁻² |Δ−1; i+1,j+1⟩ + Δ(Δ−1)⁻¹ |Δ+1; i,j⟩
```

**Inversion, closed form** — P:

```
v_k = (−1)^k (Δ − 2k − 2) / [(Δ−1)(Δ−2)···(Δ−2k−1)]
```

Denominator is a falling factorial of length `2k+1`, vanishing exactly when Δ is
an integer in `[1, 2k+1]`; the numerator vanishes only at `Δ = 2k+2` and never
cancels it.

**The obstruction** — P:

```
(−n·p̂)⁻¹ exists  ⟺  Δ is NOT a positive integer
first failure at  r* = ⌈(Δ−1)/2⌉
```

Principal series `Δ = 1 + iλ` untouched. Poincaré generators (`p ≤ 2`)
unaffected; failure begins at `p = 5/2`. **Answers Himwich–Pate's closing
question in the negative.**

## 3. Cosmological polytopes and mass
*Module: `cosmopolytope.py`*

**Facet theorem** — M, verified on 7 graphs including loops. For connected
subgraph `g`, the facet is the vanishing of:

```
Σ_{v∈V_g} x_v + Σ_{e∉E_g} (#endpoints in V_g) · y_e
```

An edge left out with *both* endpoints inside contributes `2y_e`. Trees never see
this clause, which is why the theorem is easy to state wrongly.

**Flat-space mass resummation** — M (Benincasa's stated open problem, flat
corner):

```
Σ_{a≥0} (−m²/2)^a ψ_a  =  ψ_G |_{y → √(y²+m²)}
```

Fitted once at `m²`, then *predicted* at `m⁴` and `m⁶` (five-site polytope in P⁸,
15 facets).

**FRW obstruction** — P. The dS first-order term:

```
ψ₁^dS = 4[A ln A − B ln B − C ln C + D ln D] / [(x₁²−y²)(x₂²−y²)]
```

Four logarithms with non-vanishing coefficients. If `Σ tᵃψ_a = ψ₀(x, f(y,t))`
with ψ₀ rational, every Taylor coefficient would be rational in `y`. **⟹ no
reparameterisation of the edge variable generates the FRW tower.**

## 4. Navier–Stokes: five modules
*Modules: `shell.py`, `embedding.py`, `coherence.py`, `cascade.py`,
`averaging.py`*

**Cascade exponent** — P. For Obukhov shells `N_k = N₀^{b^k}`:

```
γ = α / (2b + 1)
```

`b = 1` gives `α/3`; `α = 1, b = 1` gives Kolmogorov's `1/3`. γ *decreases* in
`b`: wider separation flattens the regularising cascade.

**Euler amplifier gate** — P. In Waleffe's helical decomposition energy
conservation gives `c₂ + c₃ = −c₁`, so the high pair's exchange depends on `c₁`
alone, which sees the high wavenumbers only through their *difference*. The
`O(|k_high|)` transport cancels identically:

```
|c₁|/|k₁| = |sin 2θ|/2      max = ½ at θ = 45°
```

**The ℤ³ architecture** — M. With `d, e` orthogonal and equal length:

```
p = A(d+e),  q = p − d,  (d,e) ← (p, A(e−d))
```

Closes in the integers. Seeded with `d = (3,4,0)`, `e = (0,0,5)`: exhaustive
search over all 14 modes gives **6 closing triads, all nearest-neighbour, zero
local, zero long-range** — the Obukhov graph realised in ℤ³.

**Cap rule** — P. For `|a| = |b| = N`, `a+b` returns to the shell at a 120°
opening; reality adds the antipodal cap, so the threshold is **30°, sharp**.

**Mode budget** — P. `μ_k = N_k^{−2(α−1)}` needs `N_k^{2(α−1)}` modes; a 30° cap
of a dyadic shell keeps a fixed fraction of `N_k³`. Room iff:

```
2(α−1) < 3      ⟺      α < 5/2
```

The top of the 3D intermittency range, reached by counting lattice points rather
than by the uncertainty principle. **Two unrelated arguments, one boundary.**

**Averaging estimate** — M / R. Scale invariance proved: one geometry at absolute
scales 1–16 returns `ρ = 0.089517` with spread `6×10⁻¹¹`. Separation fit
`ρ(r) = 0.0869 + 0.0201/r` eliminates the shell-ratio dependence exactly. **The
uniform phase floor was announced and then refuted by a complete sample.**

## 5. Quantum coordination in markets
*Module: `quantumcoord.py`*

**The correction** — P. A quantum device produces a distribution; if it satisfies
the incentive constraints a mediator can sample it. **⟹ QCE ⊆ CCE for
complete-information games.**

**XOR dichotomy (two inputs)** — M. Over all 16 sign matrices: 8 have advantage,
every one at ratio exactly `√2`; 8 have none, ratio exactly 1. The split is
exactly rank-2 versus rank-1.

**The rent shrinks** — M. All 512 three-input sign matrices give spectrum
`{1, 1.0102, 1.2}` — maximum **6/5**, strictly below `√2`. Universal ceiling
`K_G ≤ 1.7822` (Grothendieck).

**Detection is an identity** — M. A shared-coin classical strategy wins 3/4 with
marginals exactly ½, matching quantum marginals exactly, so
`marginal_divergence() = 0.0`. Joint distributions do separate:
`min KL = 0.0321` nats/round → 215 rounds at `δ = 10⁻³`.

**Visibility threshold** — P. With efficiency `η`, no-click assigned the default
outcome, at maximal entanglement:

```
2√2 η² + 2(1−η)² > 2  ⟺  η > 2/(1+√2) = 0.82842712…
```

The Eberhard floor 2/3 is approached only as entanglement → 0.

**Price of anarchy** — P. `quantum_price_of_anarchy == correlated_price_of_anarchy`
identically.

## 6. Which spins a discrete structure can protect
*Module: `qca.py`*

**The one number** — P. The character norm is simultaneously the sum of squared
multiplicities and the dimension of the commutant:

```
⟨χ_l, χ_l⟩_G = Σ m_i²  = dim(commutant) = # independent invariant couplings
tuning_cost  = ⟨χ_l, χ_l⟩_G − 1
```

Representation theory and fine-tuning cost are the same computation.

| spin | dim | protected by | tuning cost on a cube |
| --- | --- | --- | --- |
| 1 (photon) | 3 | T, O, I | 0 |
| 2 (graviton) | 5 | **I alone** | 1 |
| ≥ 3 | 7+ | nothing | 2+ |

**The `s ≤ 2` ceiling, derived from finite group theory** — P. The largest irrep
of any finite `SO(3)` subgroup has dimension 5, and:

```
5 = 2·2 + 1      ⟹      s ≤ 2
```

That is the Weinberg–Witten massless helicity bound, from ADE classification
rather than a stress tensor. With the crystallographic restriction (rotation
orders ∈ {1,2,3,4,6}): **an emergent graviton is symmetry-protected only on
icosahedral — hence quasicrystalline — structures.**

**Relevant vs irrelevant** — M. Resolving by order in `k`, the decisive rung is
`n = 0`:

- spin 1: `k=0` free on all three; first anisotropy at `n = 1` (T), 2 (O), 4 (I)
- spin 2: T and O split at **n = 0**; I first at `n = 2`

An excess at `n = 0` is a *gap* splitting — relevant, unsuppressed in the IR. At
`n > 0` it is a velocity anisotropy — irrelevant.

**Elastic tensor** — M. `C_ijkl ∈ Sym²(Sym²V)`, 21 components. The machinery
reproduces the entire crystal-system table — 21, 13, 9, 7, 6, 5, 3 — and **2 for
icosahedral**, the measured elastic isotropy of icosahedral quasicrystals. Twelve
independent numbers, none put in.

```
spin 2 as excitations   → strong no-go on any lattice
spin 2 as a gauge field → one tuned relation on a cubic lattice
```

**Fractons do not escape** — P. Both linearised gravity and the scalar-charge
fracton theory use the same field `Sym²(V) = ℓ0 ⊕ ℓ2`. A different gauge
parameter changes which polarisations survive; it cannot fuse two distinct
point-group irreps. Splitting values identical: T → 2, O → 1, I → 0.

**Cyclotomic arithmetic note.** sympy's trig simplifier fails to close cyclotomic
sums at a seventh of a turn; reduce polynomials mod `Φ_L(x)` in `ℤ[ζ_L]` instead.

## 7. What logic actually costs
*Module: `thermo.py`*

| claim | verdict |
| --- | --- |
| erasing perfectly costs infinite heat | **false** — bounded by `ln 2` |
| there is a dissipation singularity | **true, twice**, neither as described |
| it is a phase transition | true of one, false of the other |
| the trilemma is the TUR | partly — the TUR says nothing about logic |

**Maintenance is logarithmic** — P. In the reliable limit the demon flux
saturates at the noise rate and the affinity grows as `2 ln(1/ε)`:

```
Σ̇ → 2γ ln(1/ε)
```

Coefficient 2 derived, not fitted — 1, 3 and 4 all fail by >10%.

**Erasure is bounded** — P:

```
W(ε) = ln 2 − H(ε)   ↗   ln 2
```

**Thermodynamic uncertainty relation** — C:

```
Var(J)/⟨J⟩² · Σ ≥ 2
```

**The real transition is the fault-tolerance threshold** — P:

```
p' = 3p² − 2p³,   unstable fixed point at p = 1/2
```

Finite dissipation below (polylog overhead, exponent `log 3 / log 2`),
unattainable at or above. **It sits at a noise value, not at logical
completeness.**

**Critical slowing down, derived** — P:

```
6p(1−p) = 3/2   exactly at threshold
escape takes ln(1/δ)/ln(3/2) levels — 5.68 per decade, measured 6, 5, 6
```

**No Gödel statement enters anywhere**, and a test asserts it.

## 8. Exact stabilizer complexity
*Module: `complexity.py`*

**The algorithm-side claim needs no computation** — P. Complexity is defined as
the minimum over all circuits; an optimised circuit prepares the same state and
was already in the set the minimum ranged over. Nothing shrinks.

**The referee is one number** — M. BFS from `|0…0⟩` under `{H, S, CNOT}` must
enumerate exactly:

```
N(n) = 2ⁿ ∏_{k=1}^{n} (2ᵏ + 1)  =  6, 60, 1080, 36720, 2423520
```

Verified up to `n = 5` (921 s).

| quantity | law (n ≤ 5) | status |
| --- | --- | --- |
| diameter | `3n + 1` | **R — false from n = 72** |
| mean | ≈ `2.4n` | fit |
| GHZ, `\|+⟩ⁿ` | `n` | **P for every n** |
| line graph | `2n − 1` | fit, suspect |
| complete graph | `3(n − 1)` | fit, suspect |

**The counting bound that kills two laws** — P. With `N(n) ~ 2^(n²/2)` states and
`|G| = n²+n` gates, a radius-`L` ball holds at most `Σ|G|^i`, so:

```
diameter  ~  n² / (4 log₂ n)
```

Therefore `3n+1` is **dead from n = 72** (bound 219 vs 217), and
`diameter − mean → 4` fails likewise.

**GHZ complexity is exactly n** — P. At least one Hadamard (CNOT and S map
computational basis states to computational basis states); at least `n−1` CNOTs
(the interaction graph must be connected or the output factorises). The counts
are disjoint and the construction attains the bound.

**Concentration** — P. The fraction of states with complexity ≤ n is at most
`|B(n)|/N(n)`:

```
7×10⁻¹⁸ at n = 20,    2×10⁻¹³¹ at n = 40
```

**Structure does not imply low complexity** — M. GHZ and the complete-graph state
are both one-line rules yet differ by `3 − 3/n`.

## 9. What the persistent Dirac operator can and cannot see
*Module: `dirac.py`*

**Refutation one** — P. With `Γ` the grading operator, both `d` and `δ` shift
degree by one, so `ΓD = −DΓ`. `Γ` is a unitary involution, so the spectrum is
symmetric about zero; with `D² = Δ` that pins it completely:

```
D = d + δ,   D² = Δ,   Γ = (−1)^k,   ΓD = −DΓ
spec(D) = {±√μ : μ ∈ spec(Δ)},  signs forced
```

`spec(D)` and `spec(Δ)` carry **identical information**. A Laplacian-cospectral
non-isomorphic graph pair exists at 6 vertices (none at 5) and shares a Dirac
spectrum exactly.

**Refutation two** — P. Reflection is an isometry, so a chiral point cloud and its
mirror have **bitwise identical** distance matrices. Every distance-based
filtration is the *same filtered complex*, so persistent homology, Laplacian and
Dirac all agree. What flips is the signed volume, which is not a function of
pairwise distances.

**What survives** — the non-zero spectrum carries strictly more than persistent
*homology*, which reads only the kernel. That is Laplacian vs homology, not
Laplacian vs Dirac.

**Provenance.** The brief's citation `arXiv:2208.06456` (Rui–Wang–Wei) makes the
weaker true claim and does not support the chirality framing. **This module
refutes an inflation of the literature, not the literature.**

## 10. Curvature kills supersymmetry but not chirality
*Module: `magnetic.py`*

**Chirality survives any connection** — P. `chiral_anticommutator_norm` is exactly
`0.0`; supersymmetry does not survive.

**Interlacing under simplex insertion** — P (Cauchy interlacing for bordered
Hermitian matrices). A newly inserted simplex has no cofaces, so the Dirac matrix
gains exactly one row and column and no existing entry changes:

```
λ_i(D') ≤ λ_i(D) ≤ λ_{i+1}(D')
```

**Composed over `m` insertions** — P:

```
λ_i(D⁽ᵐ⁾) ≤ λ_i(D) ≤ λ_{i+m}(D⁽ᵐ⁾)
```

**Monotone spectral curves** — M. Over a 21-step filtration the ends stayed exact
mirrors while the spread grew monotonically `2.83 → 5.67`.

**Curvature and holonomy** — C. Wilson plaquette action; curvature
`= |holonomy − 1|`.

**Scope.** Stability under *combinatorial* change, not *metric* perturbation. The
metric case is open.

## 11. Local spectral measures: girth, Wilson action, filtration invariant
*Module: `insertion.py`. Statements 3–5 original.*

**The object** — C, and the novelty claim on it was **withdrawn**. It is the
*local density of states* of Savostianov–Guglielmi–Schaub–Tudisco
(arXiv:2502.07558, Def. 4.1):

```
mu_j(λ | A) = Σ_i |e_j^T q_i|² δ(λ − λ_i)
M_n(τ) = ⟨e_τ, Dⁿ e_τ⟩
```

**Odd moments vanish** — P (the grading again).

**The flat measure is two-point Bernoulli** — P:

```
mu_tau = ½ delta_{+sqrt(k+1)} + ½ delta_{-sqrt(k+1)}
M_n(flat, dim k) = (k+1)^{n/2}
```

**The girth law** — P/M. Flux enters the local moments at order **exactly `2g`**,
where `g` is the length of the shortest cycle through `τ` that bounds:

```
first flux-bearing moment = M_{2g}
```

| structure | girth | first flux |
| --- | --- | --- |
| tree | ∞ | never (M₂…M₁₀ all blind) |
| 2-simplex | 2 | `M₄` |
| edge, graph girth 3 | 3 | `M₆` |
| edge, graph girth 4 | 4 | `M₈` |

Local form of a fact about trees: a simply connected structure carries no
holonomy. Verified over 150 random connections with zero exceptions.

**The three-term fourth-moment law, in every dimension** — P:

```
M_4(tau) = M_2(tau)² + S(tau) + Σ_g |1 − ω_g|²
```

Three terms with disjoint meanings:

- `M_2²` — the two-point Bernoulli baseline (`M_2 = dim(τ) + 1` for `dim ≥ 1`,
  and `0` at dimension zero);
- `S(τ)` — the **sibling count**: pairs `(σ, f)` with `f` a facet of `τ` shared
  with another same-dimension `σ`. Purely combinatorial and flux-blind, because
  `τ → f → σ → f → τ` sends each phase against its own conjugate;
- `Σ_g |1 − ω_g|²` — the local **Wilson plaquette action**, over Hasse plaquettes
  indexed by **codimension-two** faces. (`|F|²` only in the continuum limit — the
  "squared curvature" naming was **retracted**.)

A codim-two face `g` of `τ` lies in exactly two facets `f₁, f₂`, so
`τ → f₁ → g → f₂ → τ` is a canonical four-cycle in the Hasse diagram.

**The filtration invariant** — P:

```
Σ_τ M_4(τ) = B(K) + P(K) + W(K)
W(K) = Σ_τ M_4(τ) − B(K) − P(K)
```

The **total Wilson action is recoverable from strictly local spectral data**, each
moment computed on a subcomplex with no global operator ever formed. Verified
across six random linear extensions per complex, spread `0` to `2.8e-14`.

**Two retractions kept visible in the API.** The definition (above); and a
reported dimension-two ceiling on the fourth-moment law, which was two mistakes —
the sum was indexed over *triangular* faces when the plaquettes live on
*codimension-two* faces (for a tetrahedron, 6 edges against 4 triangles), and the
sibling term was missing. `identity_fails_above_dimension_two` now returns
`False`.

## 12. Flux chirality: the one bit spectra cannot see
*Module: `rigidity.py`. Original.*

**The spectral signature.** Insertion moments `M_0 … M_8` collected at every
simplex. `SIGNATURE_ORDER = 8`, `TOLERANCE = 1e-8`.

**Statement one — reversing every flux is invisible** — P, exactly:

```
D_conj(σ) = conj(D_σ)   entrywise
⟨e_τ, conj(D)^j e_τ⟩ = conj(⟨e_τ, D^j e_τ⟩) = ⟨e_τ, D^j e_τ⟩
```

Every moment, every simplex, every order, every complex. Measured residual
**exactly `0.0`** — not a tolerance. The cancellation is entry by entry, so a test
asserts equality with zero rather than smallness.

**Statement two — that is the only ambiguity** — P:

```
ambiguity group = Z/2
```

*The argument.* Write `ω_j = exp(i θ_j)`. Every moment is real and of the form
`cos(a · θ)`, because a walk and its reverse contribute conjugate terms. The
girth law puts each plaquette into `M₄` alone (a 2-simplex has `g = 2`), giving
`cos θ_j`. Higher moments give the pair terms, and:

```
cos(θ_j − θ_k) − cos(θ_j + θ_k) = 2 sin(θ_j) sin(θ_k)
```

`cos θ_j` pins each angle up to sign; `sin θ_j sin θ_k` pins the *relative* signs,
since flipping one alone would negate the product. The signs move together, so
the ambiguity is one global bit.

**The degeneracy the argument predicts** — M. The constraint is vacuous exactly
when `sin θ_j = 0`, i.e. `ω_j = ±1` is **real** and therefore its own conjugate:

| holonomies | matches on a 36×36 sweep |
| --- | --- |
| both non-real | 2 — both coordinates flip |
| exactly one real | 2 — only the non-real one flips |
| both real | **1 — conjugation is the identity** |

**The general count for a possibly disconnected complex** — P:

```
|ambiguity| = 2^k,  k = # components carrying a non-real holonomy
```

**Scope.** On a **connected** complex the ambiguity group is `Z/2`, acting
faithfully iff some plaquette holonomy is non-real. The proof needs the pair
terms to exist, which needs closed walks crossing two plaquettes.

**Three chiralities, one pattern — and its demotion.**

| module | chirality | mechanism |
| --- | --- | --- |
| §9 `dirac` | geometric — point cloud vs mirror | reflection is an isometry |
| §10 `magnetic` | operator — the grading `Γ` | `Γ` needs only degree parity |
| §12 `rigidity` | **flux** — sign of the holonomy | diagonal moments are real |

§13 explained the third row as the **Weyl group** — a mechanism belonging to that
row alone. A shared group with one row spoken for privately is *weaker* evidence
of a common cause than a shared group with nothing explained. **The pattern is
now an open question with one row answered, not two-thirds of a theorem.**

## 13. Local spectral moments are characters, and the Fricke cubic
*Module: `character.py`. Original in its assembly; the cubic is Fricke's.*

**Step one — the moments are integral characters** — P. Label each edge by an
integer vector, so a connection is a point `θ ∈ T^r` and each Dirac entry is a
monomial:

```
M_n(τ) = Σ_a c_a(n, τ) x^a,     every c_a ∈ ℤ,     c_{−a} = c_a
       = c_0 + Σ_{a > 0} 2 c_a cos(a · θ)
```

Integrality because `M_n` is a signed count of closed `n`-walks; symmetry because
a walk and its reverse are in bijection with inverse monomials. **This is the
character form of §12's statement one.** Verified two ways sharing no code: exact
matrix power over `ℤ[x₁^±, …, x_r^±]`, and an FFT of the float moments.

**Step two — the support law is an INEQUALITY** — P (the inequality), R (the
equality):

```
first order carrying class a  ≥  shortest closed Hasse walk at τ carrying x^a
```

The inequality is a theorem — no walk, no term. **Equality is false**, because the
count is *signed* and same-length walks cancel.

| class | at vertex (2) | at vertex (0) | at the shared edge (0,1) |
| --- | --- | --- | --- |
| (1, 0) | 4 | 6 | never |
| (0, 1) | 10 | 6 | never |
| (1, −1) | 8 | 8 | never |
| (2, 0) | 8 | 10 | never |

The shared edge is **permanently flux-blind**: moments `4, 16, 64, 256, 1024` =
`4^{n/2}` for every connection — a two-point measure at `±2`, so `e_τ` is an
eigenvector of the magnetic Hodge Laplacian at its Hasse degree whatever the field
does. Walks reach every class from length 6 (10 for the difference class) and all
of them cancel.

**This fixes `SIGNATURE_ORDER = 8`.** Order 6 cannot see any relation between two
plaquettes, and relations between plaquettes are the whole content of the rigidity
proof. Eight is the first order that can. The constant was picked by guessing
generously; it is exactly tight.

**Step 2½ — the basepoint, and a scope correction** — M. `general_dirac` puts a
simplex's whole transport on its leading edge, i.e. picks each simplex's **minimal
vertex** as basepoint. Any choice is gauge-covariant; two choices are conjugate by
a diagonal unitary only if re-basing inside a simplex is path-independent, which
fails inside a *curved* simplex:

| relabel two triangles so the shared edge is (2,3) | result |
| --- | --- |
| flat connection (pure gauge or zero) | identical spectra, identical moments |
| curved connection | **different spectra** |

**So the Dirac operator here — and every local spectral measure in §§10–12 — is an
invariant of the ORDERED complex, not of the complex.** What survives relabelling
is exactly the character layer.

**Step three — the cubic** — P (the composite), C (the identity). For any two
classes `a`, `b` and their difference, with `u = cos(a·θ)`, `v = cos(b·θ)`,
`w = cos((a−b)·θ)`:

```
w − uv = sin(a·θ) sin(b·θ)
(w − uv)² = (1 − u²)(1 − v²)
```

```
u² + v² + w² − 2uvw = 1
```

Measured residual on real moments: **4×10⁻¹⁵**.

**Step four — the pillowcase.** The map `θ ↦ (u, v, w)` is the quotient
`T² → T²/(θ ~ −θ)`, realised as the Cayley cubic: 2:1 away from the four fixed
points of the inversion, 1:1 at them.

- the `Z/2` is the deck group;
- the degeneracy locus of §12 is exactly the **four two-torsion points**, mapping
  bijectively to the **four nodes**, where `grad(u²+v²+w²−2uvw) = 2(u−vw, v−uw,
  w−uv)` vanishes;
- solving the gradient gives entries `±1` with `uvw = 1`: **four** sign patterns
  out of eight.

```
CAYLEY_CUBIC_NODES = (1,1,1), (1,−1,−1), (−1,1,−1), (−1,−1,1)
```

### Where the cubic comes from — a framing correction

**The `U(1)` framing was WRONG.** `U(1)` is abelian, so its character variety is
`Hom(π₁, U(1)) = H¹(K; U(1))` — a **flat torus**, no relations, no cubic. Fricke,
Cayley and the pillowcase are `SL₂` objects, existing because `SL₂` has
Cayley–Hamilton `Z² − tr(Z)Z + I = 0`, which `U(1)` has nothing like.

What is really happening. The moments are *forced* to be `cos(a·θ)`, and:

```
2 cos θ = tr diag(e^{iθ}, e^{−iθ})  ∈ SU(2)
```

> The local spectral moments of a `U(1)` magnetic Dirac operator are
> **automatically Weyl-invariant functions on a maximal torus of `SU(2)`**.

Walk reversal *is* the Weyl group. Substituting `X = 2u`:

```
X² + Y² + Z² − XYZ − 4 = 0
```

the Cayley cubic in standard normalisation — the `κ = 2` level set of the
commutator trace, i.e. the **commuting** locus of `X(F₂, SU(2))`, and `(T×T)/W` is
the pillowcase. Fricke relation:

```
tr[A,B] = x² + y² + z² − xyz − 2
```

**The cubic needs no invariant theory at all.** It is the trigonometric identity:

```
cos²A + cos²B + cos²C − 2 cosA cosB cosC = 1     whenever C = ±(A + B)
```

**The cubic is the relation `θ₃ = θ₁ + θ₂`, and nothing more.**

**What the pillowcase IS here.** The character variety of the complex is still the
torus. The pillowcase is `T/W` with `W` the residual symmetry of the
**observable**:

> The pillowcase is the moduli space of the spectral **signature**, not of the
> connection.

The quotient is by the **diagonal** inversion, matching the Weyl element
conjugating both commuting matrices at once — not a per-factor `(Z/2)²`, which is
the natural wrong guess and is exactly what the *truncated* regime gives.

**The image is compact semi-algebraic, not the whole complex surface.** Every
coordinate lies in `[−1,1]` since an `SU(2)` element has trace `2cos θ`. Those
range restrictions are **Loll's inequalities**: Mandelstam identities alone do not
cut out the `SU(2)` locus.

**The pillowcase needs dimension two.** For a graph `π₁` is free, so
`X(Γ, SU(2)) = SU(2)^b / SU(2)` — a ball in trace coordinates. The commuting
relation cutting it to `(T×T)/W` is imposed by attaching a **2-cell**. Graphs give
balls; 2-complexes give pillowcases — the same flatness-on-2-simplices condition
the basepoint correction turns on, from the opposite direction.

### The Burnside counts

**The group is inversion alone, `Z/2`** — not `Aut(K) × Z/2`. The apex swap *is*
visible to the signature, so it does not quotient.

**Full signature (order ≥ 8)** — P, matched exactly by brute force:

```
distinct signatures = ( N^r + gcd(2, N)^r ) / 2
```

**Truncated below the coupling order** — P. No moment joins two classes, so the
signs are independent and the group is the full `(Z/2)^r`:

```
distinct signatures = ( (N + gcd(2,N)) / 2 )^r
```

| `N` | rank | order ≥ 8 | order < 8 | measured |
| --- | --- | --- | --- | --- |
| 11 | 2 | 61 | — | 61 |
| 12 | 2 | 74 | — | 74 |
| 36 | 2 | 650 | — | 650 |
| 7 | 2 | 25 | 16 | both |
| 8 | 2 | 34 | 25 | both |
| 5 | 3 | 63 | 27 | both |
| 6 | 3 | 112 | 64 | both |

**`rigidity_sweep` at order 6 returns FOUR matches**, at order 8 two — the four
being exactly the independent sign flips `(6,13), (6,23), (30,13), (30,23)`. So
`SIGNATURE_ORDER = 8` is the **threshold**, not a margin.

**The general-rank theorem** — P, conditional:

> If every standard basis class and every pairwise difference appears with
> non-zero coefficient at some simplex and some order, then the signature
> determines the connection modulo gauge and one global reflection, and the
> ambiguity group is `Z/2`.

The hypothesis is **not automatic** — the moment is a signed count and the
flux-blind edge is a simplex where all walks cancel. `classes_are_resolved`
discharges it per complex. **Proving it in general is open.**

### Local data beats the global spectrum

The published obstruction — the spectrum does not determine the magnetic potential
(Fabila-Carrasco–Lledó–Post) — **bites, exactly where it claims to**. Measured at
`N = 12`:

| invariant | classes |
| --- | --- |
| global Dirac spectrum | 40 |
| local signature | **74** — exactly the Burnside count |

It obstructs the **global** invariant and not the **local** one.

**The exhibit is a curve, not a pair.** The entire line `θ₁ + θ₂ = π` is
isospectral to `2×10⁻¹⁵` while the local moments vary continuously and separate
every point. A generic level `θ₁ + θ₂ = c` is not (variation `10⁻¹`).

### The half-flux mechanism, derived in exact integers

```
tr(D^k) = Σ_τ M_k(τ)
```

The global spectrum is the *sum* of the local measures, and the power sums fix the
characteristic polynomial. The total expansion is **swap-symmetric** — the
coefficient at `(a₁,a₂)` equals that at `(a₂,a₁)`, exactly, because exchanging the
apexes fixes every minimal-vertex basepoint.

On the line `θ₂ = s − θ₁`, class `a` contributes at frequency `a₁ − a₂` with
**phase** `e^{i a₂ s}`. At `s = π` that phase is `(−1)^{a₂}`. At `k = 8`, in
integers:

| frequency | contributions | total |
| --- | --- | --- |
| 1 | `(1,0) = −256` at phase `+1`; `(0,−1) = −256` at phase `−1` | **0** |
| 2 | `(2,0) + (0,−2) = 8` at phase `+1`; `(1,−1) = 8` at phase `−1` | **0** |

Every order tested (`k = 4, 6, 8, 10`) leaves `{0: constant}` and nothing else. So
every power sum is constant on the line and so is the characteristic polynomial.

**The diagnostic answers NO.** At `s = 0` — the *other* central element, `+I` —
every phase is `+1`, the swap-symmetric pairs **add**, and the trace is not
constant:

```
θ₁+θ₂ = π   (−I)    deviation = 2.44×10⁻¹⁵
θ₁+θ₂ = 0   (+I)    deviation = 8.77×10⁻¹
θ₁+θ₂ = 2π  (+I)    deviation = 1.38
```

**So the mechanism is not "the holonomy is central" — it is specifically the
non-trivial central element**, whose `(−1)^{a₂}` does the cancelling. That
distinguishes half-flux from flat the way **Lieb's flux phase theorem** does, and
the natural home is Kasteleyn theory — a Kasteleyn orientation being a `±1`
connection with holonomy `−1` around every even face, and Kenyon's CRSF
determinant weight `2 − tr(hol)` being maximal at `−I`. **Whether the cancellation
is already known in determinant form is not settled here.**

### The flux torus, refined

For a *graph* every connection is flat — no 2-cells — so flux torus and character
variety coincide. On a complex they do not:

```
1 → H¹(K; U(1)) → U(1)^E / U(1)^V  --F-->  U(1)^F
```

The flux torus of the 1-skeleton is `T^{|E|−|V|+1}`; the character variety is the
**zero-curvature fibre** of the curvature map, and the non-flat connections are
the other fibres. Berkolaiko's Morse theory lives on the *whole* flux torus, with
the character variety a distinguished submanifold where Hodge theory switches on.
**Morse theory relative to that submanifold appears to be unasked.**

---

# PART II — CONSTANTS AND NAMED LAWS

| constant / law | value | module |
| --- | --- | --- |
| Weinberg–Witten ceiling from ADE | `5 = 2·2+1 ⟹ s ≤ 2` | `qca` |
| Crystallographic rotation orders | `{1,2,3,4,6}` | `qca` |
| Elastic component counts | 21, 13, 9, 7, 6, 5, 3, **2** | `qca` |
| Landauer ceiling | `ln 2` | `thermo` |
| Maintenance coefficient | `2` | `thermo` |
| Fault-tolerance threshold | `p = 1/2` | `thermo` |
| Critical multiplier | `6p(1−p) = 3/2` | `thermo` |
| Overhead exponent | `log 3 / log 2` | `thermo` |
| TUR bound | `≥ 2` | `thermo` |
| Stabilizer state count | `2ⁿ∏(2ᵏ+1)` | `complexity` |
| Diameter law | `3n+1`, **false from n = 72** | `complexity` |
| Asymptotic diameter | `~ n²/(4 log₂ n)` | `complexity` |
| Concentration width | `4` (retracted as a limit) | `complexity` |
| Cap threshold | `30°`, sharp | `cascade` |
| Intermittency budget | `α < 5/2` | `shell` |
| Cascade exponent | `γ = α/(2b+1)` | `cascade` |
| Gate maximum | `½` at `45°` | `embedding` |
| Visibility threshold | `2/(1+√2) = 0.82842712…` | `quantumcoord` |
| Grothendieck ceiling | `K_G ≤ 1.7822` | `quantumcoord` |
| Three-input maximum | `6/5` | `quantumcoord` |
| Girth law | first flux at `M_{2g}` | `insertion` |
| Second moment | `M₂ = dim + 1` (`0` at dim 0) | `insertion` |
| Signature order | `8` — the **threshold** | `rigidity` |
| Ambiguity group order | `2` = the **Weyl group** | `rigidity` |
| Cayley cubic nodes | `4` = `2^rank` two-torsion | `character` |
| Coupling order | `8` | `character` |
| Spectrum vs signature classes | `40` vs `74` at `N = 12` | `character` |

---

# PART III — LITERATURE

Organised by thread. Annotations say what each work is used **for** — background,
the result being refuted, or the classical fact being named.

## Relativity, ADM, linearisation instability
- **Taub** (1970) — the second-order obstruction.
- **Fischer & Marsden** (1973) — linearisation instability.
- **Moncrief**, *J. Math. Phys.* **16** (1975) 493 — Killing initial data.
- **Isaacson**, *Phys. Rev.* **166** (1968) 1272 — gravitational-wave effective
  stress tensor; supplies the `16π` normalisation.
- **Brill & Deser**, *Comm. Math. Phys.* **32** (1973) — positivity.

## Celestial holography / w₁₊∞
- **Strominger**, arXiv:2105.14346 — `w₁₊∞` in celestial CFT.
- **Guevara, Himwich, Pate & Strominger**, arXiv:2103.03961 — holographic
  symmetry algebras.
- **Bakas**, *Phys. Lett. B* **228** (1989) — the wedge algebra.
- **Pope, Romans & Shen**, *Nucl. Phys. B* **339** (1990) — `W_∞` structure.
- arXiv:2312.08597, *JHEP* **07** (2024) 180 — extension to *massive* scalars;
  the paper whose closing question §2 answers.

## Cosmological polytopes
- **Arkani-Hamed, Benincasa & Postnikov**, arXiv:1709.02813 — cosmological
  polytopes.
- **Benincasa**, arXiv:1909.02517 — cosmological-polytope combinatorics; states
  the mass problem §3 addresses.
- ⚠ **arXiv:1711.09102 is the ABHY associahedron paper, not the cosmological
  polytope paper** — a miscitation corrected during the work.

## Navier–Stokes / turbulence
- **Obukhov** (1971) — the shell model.
- **Tao**, arXiv:1402.0290 — averaged Navier–Stokes blow-up.
- **Waleffe**, *Phys. Fluids A* **4** (1992) 350 — helical decomposition.
- **Caffarelli, Kohn & Nirenberg**, *CPAM* **35** (1982) — partial regularity.
- **Palasek**, arXiv:2605.13827 (2026), Thms 1.3 & 1.8 — finite-time blow-up in a
  Navier–Stokes-like model; the target §4 tests.
- **Kolmogorov** — the `1/3` recovered as `α = 1, b = 1`.

## Bell / quantum coordination
- **Bell** (1964); **CHSH** (1969); **Tsirelson** (1980).
- **Brunner et al.**, arXiv:1303.2849 — Bell nonlocality review.
- **Eberhard**, *Phys. Rev. A* **47** (1993) R747 — the 2/3 detection floor.
- **Aumann** (1974) — correlated equilibrium.
- **Krivine** (1979) — Grothendieck-constant bound `K_G ≤ 1.7822`.
- arXiv:2604.07451 (2026) — operational criteria for quantum advantage.
- **Hymas et al.**, arXiv:2602.06367 (2026) — many entangled agents.

## Point groups, spin protection, fractons
- **Klein** — classification of finite `SO(3)` subgroups (ADE).
- **Crystallographic restriction theorem** — rotation orders `{1,2,3,4,6}`.
- **Weinberg & Witten** — the massless helicity bound recovered here from finite
  group theory.
- **Pretko**, arXiv:1707.03838 — fracton gauge theory; the escape route §6 closes.

## Thermodynamics of computation
- **Landauer** (1961) — `ln 2` per erased bit.
- **Bennett** (1982) — reversible computation.
- **Barato & Seifert** (2015) — the thermodynamic uncertainty relation.
- **Gingrich, Horowitz, Perunov & England** (2016) — dissipation bounds.
- **Aharonov & Ben-Or**; **Knill, Laflamme & Zurek** — fault-tolerance thresholds.
- **Prigogine**; **Gödel** — named in the brief §7 dismantles; neither enters the
  mathematics.

## Stabilizer complexity
- **Gottesman** — the stabilizer formalism.
- **Aaronson & Gottesman** — the tableau representation and `_g`/`_rowsum`.
- **Haferkamp, Faist, Kothakonda, Eisert & Yunger Halpern**, arXiv:2106.05305 —
  linear growth of quantum circuit complexity.
- **Brown & Susskind** — the complexity-growth conjecture §8 addresses.
- **Nielsen & Chuang** — standard reference for the gate set.

## Persistent Dirac operators / topological signal processing
- **Calmon, Schaub & Bianconi**, arXiv:2301.10137 — Dirac signal processing.
- **Bianconi**, arXiv:2106.02929 and arXiv:2309.07851 — the topological Dirac
  operator; the magnetic/complex Dirac line.
- **Egidi, Gittins, Habib & Peyerimhoff**, arXiv:2211.08019 — ⚠ note: the
  **diamagnetic inequality fails above degree zero**.
- **Wei & Wei**, arXiv:2112.10906 — persistent Dirac.
- **Anh, Dik & Anh**, arXiv:2506.21352 — persistent Dirac / spectral bounds.
- **Jung, Kang & Park**, arXiv:2512.05463 — recent persistent-Dirac work.
- arXiv:2105.00529 — the relevant work for the claim the brief actually wanted.
- ⚠ **arXiv:2208.06456 (Rui, Wang & Wei)** — the brief's citation. It makes the
  weaker *true* claim and does **not** support the chirality framing. §9 refutes
  an inflation of the literature, not the literature.

## Local density of states / discrete spectral geometry
- **Savostianov, Guglielmi, Schaub & Tudisco**, arXiv:2502.07558, Def. 4.1 —
  **the local density of states**. §11's "new definition" claim was withdrawn in
  favour of this. Their Thm 4.2 reads the off-kernel mass as generalised
  effective resistance; their Chebyshev moments `2[T_m(H)]_jj` are already walk
  moments at a simplex.
- **Kenyon**, *Ann. Probab.* **39** (2011) — cycle-rooted spanning forests;
  determinant weight `2 − tr(hol)`, vanishing at trivial holonomy and maximal at
  `−I`.
- **Chamseddine & Connes**, hep-th/9606001 — Yang–Mills from the fourth heat
  coefficient of a *global* trace. §11 assembles the same order locally.
- **Najem, Mrad & Elsayed**, arXiv:2509.04311 — discrete spectral action, global
  traces.
- **Preciado & Jadbabaie**, arXiv:1107.5676 — the walk-moment lemma.
- **Kesten**; **McKay** — the tree/root spectral measure behind the girth law.
- **Lieb & Loss**, "Fluxes, Laplacians, and Kasteleyn's theorem," *Duke Math. J.*
  **71**(2) (1993) 337–363 — the **flux phase theorem**; half-flux as the
  distinguished value.
- **Kasteleyn** — orientations as `±1` connections with holonomy `−1` around every
  even face.

## Character varieties, trace coordinates, the pillowcase
- **Fricke & Klein** (1897) — the Fricke relation
  `x²+y²+z²−xyz−2 = tr[A,B]`; the Cayley cubic; `X(F₂, SL₂) ≅ ℂ³`.
- **Cayley** — the nodal cubic surface.
- **Procesi** (1976) — trace functions generate; with Cayley–Hamilton, words of
  length ≤ 2 suffice, so **all moments are polynomials in finitely many trace
  coordinates**.
- **Loll**, hep-th/9309056 — beyond the Mandelstam constraints there are
  **inequalities** restricting the range of Wilson loops; these are what make the
  `SU(2)` locus semi-algebraic with corners.
- **Mandelstam** — the trace identities.
- **Giles**, "Reconstruction of gauge potentials from Wilson loops," *Phys. Rev.
  D* **24** (1981) 2160–2168 — holonomy → connection is **solved**, up to gauge:
  `𝒜/𝒢 ≃ {W(γ)}/ℳ`. Reviewed in arXiv:2502.20928.
- **Fock & Rosly**, *Theor. Math. Phys.* (1992) — the moduli space of **graph
  connections** `G^E/G^V`, characterised by a ciliated fat graph. The correct
  discrete home for this line of work.
- **Alekseev, Grosse & Schomerus**; **Buffenoir & Roche** — quantisation of the
  Fock–Rosly structure.
- arXiv:2206.14183 — short modern proof of the `F₂` generation and the
  `[-2,2]` reality condition for `SU(2)`.
- **Hedden, Herald & Kirk**, *Geom. Topol.* **18**(1) (2014) 211–287 — "The
  pillowcase and perturbations of traceless representations of knot groups."
- Part II: arXiv:1501.00028 — `ℤ/4`-graded Lagrangian-Floer complex.
- **Hedden, Herald, Hogancamp & Kirk**, arXiv:1808.06957 — Fukaya-categorical
  factorisation of Bar-Natan's functor.
- **Herald & Kirk**, arXiv:2407.11247 — holonomy-perturbed traceless character
  varieties.
- **Sivek & Zentner** — `SU(2)`-cyclic surgeries and the pillowcase; the image is
  semi-algebraic over the real algebraic numbers.
- arXiv:2607.26095 (2026) — consolidation with explicit representation data;
  `ℤ/2` orbifold points.
- **Kronheimer & Mrowka** — singular instanton theory, the traceless boundary
  condition.
- **Bar-Natan** — the functor being factored.

## Inverse spectral problems on graphs / the flux torus
- **Berkolaiko**, *Anal. PDE* **6** (2013) 1213–1233, arXiv:1110.5373 — Morse
  index of `λ_n` at zero field equals the **nodal surplus**.
- **Berkolaiko & Weyand**, *Phil. Trans. R. Soc. A* **372** (2014) 20120522,
  arXiv:1212.4475 — metric-graph version, Morse index `= φ − (n−1)`. **The space
  of fluxes is a `β`-dimensional torus** — which is `H¹(Γ; U(1))`.
- **Colin de Verdière**, *Anal. PDE* **6** (2013) 1235–1242 — magnetic
  interpretation of the nodal defect.
- **Alon, Band & Berkolaiko**, *Comm. Math. Phys.* **362** (2018) 909–948; *Exp.
  Math.* (2022) — nodal statistics as a distribution over the flux torus.
- **arXiv:2212.00830** — "Morse theory for discrete magnetic operators and nodal
  count distribution for graphs." **Spectral data as a Morse function on the
  `U(1)` character variety of a graph, in everything but name.**
- **Berkolaiko & Zelenko**, arXiv:2304.04331 — Morse theory *through* eigenvalue
  crossings; diabolical/Dirac/Weyl points and conical intersections. The
  non-smooth contribution depends only on multiplicity and relative position.
- **Fabila-Carrasco, Lledó & Post**, "A geometric construction of isospectral
  magnetic graphs," *Anal. Math. Phys.* **13**:64 (2023) — **the spectrum does
  not determine the magnetic potential.** The central negative datum.
- **Kurasov**, "Inverse problem for Aharonov–Bohm rings," *Math. Proc. Camb.
  Phil. Soc.* **148** (2010) 331–362 — flux recovery for rings.
- **Gutkin & Smilansky**, "Can one hear the shape of a graph?", *J. Phys. A* **34**
  (2001) 6061.
- **Kurasov & Nowaczyk**, *J. Phys. A* **38** (2005) 4901.
- **Parzanchevski & Band**, *J. Geom. Anal.* **20** (2010) 439 — isospectrality
  with boundary conditions.

## Twisted torsion — the nearest existing framework above degree 0
- **García López**, arXiv:1407.0301 — twisted Reidemeister torsion from a
  `ℤ₂`-graded twisted cochain complex `(C^•(K,E), ∂ + ϑ∪)`. Structurally the same
  object as the chiral/Dirac setup here.
- **Mathai & Wu** — the earlier definition it extends.
- **Bénard**, arXiv:1711.08781 — the torsion function on character varieties.
- **Kitano, Morifuji & Tran**, arXiv:1904.08026 — local constancy.

## Other threads in the repository
- **Ihara** (1966); **Sunada** (1986); **Bass** (1992); **Hashimoto** — the graph
  zeta function and the Ramanujan ⟺ RH equivalence (`selberg.py`; classical).
- **Bordenave, Lelarge & Massoulié**; **Krzakala et al.** — non-backtracking
  spectrum and community detection (`detection.py`; **definitively not novel**).
- **Ponzano & Regge**; **Roberts** — 6j asymptotics (`spinfoam.py`; a theorem).
- **da Silva**, arXiv:2101.04739 — `φ(m)` values used by `fermat_hodge.py`.

---

# PART IV — STATUS

## Proved here
Friedmann identification with computed coefficients · the massive `w₁₊∞`
integer-Δ obstruction · the flat-space polytope mass resummation · the FRW
reparameterisation obstruction · the cascade exponent · the Euler gate law · the
ℤ³ Obukhov architecture and the sharp 30° cap · the `α < 5/2` budget, twice ·
scale invariance of the coherence penalty · the quantum-coordination package ·
the spin-protection classification and its RG sharpening · the elastic table ·
the fracton no-escape · the thermodynamic verdict · exact stabilizer complexity
with GHZ `= n` and the concentration bound · both Dirac refutations · interlacing
and monotonicity · the girth law · the three-term fourth-moment law · the
filtration invariant · **both halves of the flux-chirality rigidity theorem with
its degeneracy and disconnected count** · the character expansion's integrality
and symmetry · the per-class support inequality · **the Cayley-cubic identity on
measured moments** · the Burnside counts, full and truncated, ranks 1–3 · **the
local signature strictly dominating the global spectrum** · **the half-flux
isospectral curve derived per order in exact integers**.

## Classical, named, not mine
Fricke–Klein · the Cayley cubic · the pillowcase · Procesi's generation theorem ·
Loll's inequalities · Giles' reconstruction · Fock–Rosly · Berkolaiko's Morse
theory on the flux torus · Berkolaiko–Zelenko on collisions ·
Fabila-Carrasco–Lledó–Post on isospectral magnetic graphs · Burnside's lemma ·
the local density of states · Lieb–Loss · Kenyon · Cauchy interlacing · Hodge
theory · Klein's ADE classification · the crystallographic restriction.

## Retracted or withdrawn
1. The uniform phase floor (§4) — refuted by a complete sample, after being
   announced.
2. `diameter = 3n+1` and `diameter − mean → 4` (§8) — dead from `n = 72`.
3. The insertion measure as a new definition (§11) — it is the published LDoS.
4. "Excess kurtosis = squared curvature" (§11) — it is the Wilson plaquette
   action; `|F|²` only in the continuum limit.
5. A dimension-two ceiling on the fourth-moment law (§11) — two mistakes, not an
   obstruction.
6. The per-class support law stated as an **equality** (§13).
7. **The cubic as a `U(1)` phenomenon** (§13) — it is `SU(2)`, inherited by
   `U(1)` sitting inside as a maximal torus.
8. **The four nodes read as eigenvalue degeneracies** (§13) — two of the four
   two-torsion points carry repeated eigenvalues and two carry none.
9. **"The apparatus does not transfer"** — over-withdrawal; Berkolaiko–Zelenko
   handles the two points where the loci do coincide.
10. **"The obstruction doesn't bite"** — it bites exactly where it claims.

## Open, and stated as open
1. The FRW resummation beyond reparameterisations.
2. The analytic embedding for Navier–Stokes; the phase tail (*refuted*).
3. Whether real order flow presents rank-2 payoff structure.
4. Three stabilizer laws still five-point fits — suspect, since `3n+1` looked
   equally solid.
5. The metric case for spectral persistence stability.
6. **The non-cancellation hypothesis** of the general-rank rigidity theorem —
   checkable per complex, checked on every complex used, not proved, and the
   flux-blind edge shows it cannot hold unconditionally.
7. **Whether the half-flux cancellation is known in determinant form** —
   Lieb–Loss and Kenyon's `2 − tr(hol)` weight are where to look.
8. **Morse theory relative to the zero-curvature fibre** of the curvature map —
   apparently unasked, and the best-posed question left.
9. **Whether the three-chirality `Z/2` is a pattern at all** — one row is now
   explained by a mechanism the other two do not share, which weakens the case.

## Gaps found by external literature sweep
| question | status |
| --- | --- |
| Magnetic Dirac on complexes ↔ character variety | **Empty.** Fock–Rosly is the right substrate; connection never made. |
| LDoS ↔ Fricke / Cayley cubic | **Empty.** Every link exists; the chain is short and writable. |
| Inverse spectral, `U(1)`, graphs | **Rich.** Flux torus = character variety; Morse theory established; non-uniqueness proven. |
| Inverse spectral, `U(1)`, complexes (`k ≥ 1`) | **Empty** except twisted Reidemeister torsion. |
| Pillowcase ↔ discrete gauge field | **Empty.** |

---

*Generated from the repository at branch
`claude/tetrahedron-packing-spectral-u7029m`. The code is the authority; this is
the index.*
