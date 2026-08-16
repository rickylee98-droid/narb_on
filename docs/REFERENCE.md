# The Narb Notebook — Reference

A standing record of what this repository establishes, what it refutes, and
what it leaves open. Every result here is backed by a module and a test suite;
the code is the authority and this document is the map.

**Status at time of writing:** 4264 tests passing, branch
`claude/tetrahedron-packing-spectral-u7029m`.

**On citations:** author, title and year are reliable. arXiv identifiers were
verified by the repository owner through an external search; `arxiv.org` is
unreachable from the build environment, so nothing here was checked by the
code that produced it.

---

## Method

Three rules, in the order they matter.

1. **Literature first, to rule the problem out.** Eight briefs in this
   repository described a frontier that had already moved, or cited a paper
   that did not say what the brief claimed. Finding that out took minutes.
   Building on the stale version would have taken days.
2. **Never select a problem by what can be verified.** That criterion selects
   for solved problems. It is the most corrosive habit available to a
   computational agent.
3. **Every identity computed two ways that share no code.** Exact arithmetic
   wherever the problem admits it — `Fraction`, `sympy`, cyclotomic integers.

The failure catalogue in §15 is the most transferable artifact here.

---

## 1. The Friedmann constraint from the ADM obstruction

*Module: `graviton.py`, building on `adm.py`.*

Vacuum ADM constraints on a flat compact slice with `K = 0` admit a Killing
initial data set, so linearisation instability follows and second-order
perturbations must satisfy the Taub obstruction:

```
Q = ⟨R⁽²⁾(h)⟩ + ⟨(tr K)² − |K|²⟩ = 0
```

**Averaging weights.** `⟨cos²(k·x)⟩ = ½` for `k ≠ 0` but `⟨1⟩ = 1` at `k = 0`.
Applying ½ uniformly is wrong and leaves every per-mode signature unchanged —
which is why the error survived until sectors were summed.

**TT sector.** With `h = A cos(k·x)`, `A` transverse-traceless, and ADM giving
`ḣ = 2K`:

```
−Q_k = ⅛(Ȧ² + |k|²A²)     ⟹     ω² = |k|²
```

Massless, two polarisations. Controls: the pure-trace mode gives
`ω² = −(5/3)|k|²` (the conformal factor problem); linearised diffeomorphisms
`h_ij = k_i ξ_j + k_j ξ_i` give exactly zero.

**Isaacson normalisation.** `−Q_k = 16πρ` exactly, every mode and amplitude.

**The homogeneous mode.** At `k = 0` the momentum term has inertia `(5,0,1)`.
With `B₀ = −Hδ + σ`, summing all modes and imposing `Q = 0`:

```
3H² = 8πρ + ½ σ_ij σ^ij
```

The Friedmann constraint with shear, sourced by graviton energy. Every
coefficient — the 6, the 16π, the ½ — is computed output.

**Platycosm shear budget** (trace-free invariant momenta): torus 5, dicosm 3,
tetracosm 1, Hantzsche–Wendt 2. Totals match the classical moduli-space
dimensions, an external referee.

*Taub (1970); Fischer–Marsden (1973); Moncrief, J. Math. Phys. 16 (1975) 493;
Isaacson, Phys. Rev. 166 (1968) 1272; Brill–Deser, CMP 32 (1973).*

---

## 2. Massive w₁₊∞ — an obstruction at integer Δ

*Module: `celestial.py`.*

The wedge algebra `[w^p_m, w^q_n] = [m(q−1) − n(p−1)] w^{p+q−2}_{m+n}` with the
wedge condition `|m| ≤ p−1` being exactly polynomiality of the monomial
realisation.

Bulk-to-boundary propagator `G_Δ = (−p̂·q̂)^{−Δ}`, and the entire derivation
collapses to one identity:

```
u·u_{zz̄} − u_z·u_z̄ = 1
```

**Bidiagonality.** `−n·p̂` acts as

```
−n·p̂ |Δ; i,j⟩ = (Δ−1)⁻² |Δ−1; i+1,j+1⟩ + Δ(Δ−1)⁻¹ |Δ+1; i,j⟩
```

**Inversion, closed form:**

```
v_k = (−1)^k (Δ − 2k − 2) / [(Δ−1)(Δ−2)···(Δ−2k−1)]
```

The denominator is a falling factorial of length `2k+1`, vanishing exactly when
Δ is an integer in `[1, 2k+1]`; the numerator vanishes only at `Δ = 2k+2` and
never cancels it.

```
(−n·p̂)⁻¹ exists  ⟺  Δ is NOT a positive integer;  first failure at r* = ⌈(Δ−1)/2⌉
```

Principal series `Δ = 1 + iλ` is untouched. Poincaré generators (`p ≤ 2`) are
unaffected; failure begins at `p = 5/2`, exactly where the Schwinger prescription
does. **Answers Himwich–Pate's closing question in the negative.**

*Strominger, arXiv:2105.14346; Guevara–Himwich–Pate–Strominger,
arXiv:2103.03961; Bakas, Phys. Lett. B 228 (1989); Pope–Romans–Shen,
Nucl. Phys. B 339 (1990).*

---

## 3. Cosmological polytopes and mass

*Module: `cosmopolytope.py`.*

**Facet theorem** (verified on 7 graphs including loops): for connected subgraph
`g`, the facet is the vanishing of

```
Σ_{v∈V_g} x_v + Σ_{e∉E_g} (#endpoints in V_g) · y_e
```

An edge left out with *both* endpoints inside contributes `2y_e`. Trees never
see this clause, which is why the theorem is easy to state wrongly.

**Flat-space mass resummation** — Benincasa's stated open problem, flat corner:

```
Σ_{a≥0} (−m²/2)^a ψ_a  =  ψ_G |_{y → √(y²+m²)}
```

Fitted once at `m²`, then *predicted* at `m⁴` and `m⁶` (needing the five-site
polytope in P⁸ with 15 facets).

**FRW obstruction theorem.** The dS first-order term

```
ψ₁^dS = 4[A ln A − B ln B − C ln C + D ln D] / [(x₁²−y²)(x₂²−y²)]
```

contains four logarithms with non-vanishing coefficients. If
`Σ tᵃψ_a = ψ₀(x, f(y,t))` with ψ₀ rational, every Taylor coefficient would be
rational in y. **⟹ no reparameterisation of the edge variable generates the FRW
tower.**

*Arkani-Hamed–Benincasa–Postnikov, arXiv:1709.02813; Benincasa,
arXiv:1909.02517. Note: arXiv:1711.09102 is the ABHY associahedron paper, not
the cosmological polytope paper — a miscitation corrected during the work.*

---

## 4. Navier–Stokes: five modules

*Modules: `shell.py`, `embedding.py`, `coherence.py`, `cascade.py`,
`averaging.py`.*

**Cascade exponent.** For Obukhov shells `N_k = N₀^{b^k}`, the stationary
fixed-point condition gives

```
γ = α / (2b + 1)
```

`b = 1` gives `α/3`; `α = 1, b = 1` gives Kolmogorov's `1/3`. γ *decreases* in b:
wider separation flattens the regularising cascade, which is how
super-exponential shells buy blow-up.

**Euler amplifier gate.** In Waleffe's helical decomposition, energy conservation
gives `c₂ + c₃ = −c₁`, so the high pair's exchange depends on `c₁` alone, which
sees the high wavenumbers only through their *difference*. The `O(|k_high|)`
transport cancels identically. In the scale-separated limit:

```
|c₁|/|k₁| = |sin 2θ|/2      max = ½ at θ = 45°
```

**The architecture.** With `d, e` orthogonal and equal length,

```
p = A(d+e),  q = p − d,  (d,e) ← (p, A(e−d))
```

closes in the integers. Seeded with `d = (3,4,0)`, `e = (0,0,5)`: exhaustive
search over all 14 modes gives **6 closing triads, all nearest-neighbour gates,
zero local, zero long-range** — the Obukhov graph realised in ℤ³.

**Cap rule.** For `|a| = |b| = N`, `a+b` returns to the shell at a 120° opening.
Reality adds the antipodal cap, so the threshold is **30°, and it is sharp**.

**Mode budget.** `μ_k = N_k^{−2(α−1)}` needs `N_k^{2(α−1)}` modes; a 30° cap of a
dyadic shell keeps a fixed fraction of `N_k³`. Room iff `2(α−1) < 3`, i.e.
**α < 5/2** — the top of the 3D intermittency range, reached here by counting
lattice points rather than by the uncertainty principle. Two unrelated arguments,
one boundary.

**The averaging estimate — half closed, half refuted.** Scale invariance is
proved: one geometry at absolute scales 1–16 returns ρ = 0.089517 with spread
6×10⁻¹¹. The separation fit `ρ(r) = 0.0869 + 0.0201/r` eliminates the shell-ratio
dependence exactly. **The phase dependence is not eliminated, and I wrongly
announced it was:** a complete 8/8 sample gives min ρ = 0.0078, five times below
the floor the incomplete samples suggested. The earlier floor was survivorship —
slow draws never reached the target and so were absent from the statistic.

*Obukhov (1971); Tao, arXiv:1402.0290; Waleffe, Phys. Fluids A 4 (1992) 350;
Caffarelli–Kohn–Nirenberg, CPAM 35 (1982).*

---

## 5. Quantum coordination in markets

*Module: `quantumcoord.py`.*

**The correction.** A quantum device produces a distribution; if it satisfies the
incentive constraints a mediator can sample it. **⟹ QCE ⊆ CCE for
complete-information games.** The advantage lives in Bayesian games with private
types and no mediator, where the classical resource is shared randomness.

**XOR dichotomy (two inputs).** Over all 16 sign matrices: 8 have advantage,
every one at ratio exactly √2; 8 have none, ratio exactly 1. The split is exactly
rank-2 versus rank-1.

**The rent shrinks.** All 512 three-input sign matrices give spectrum
`{1, 1.0102, 1.2}` — maximum **6/5**, strictly below √2. Bigger games, smaller
advantage. Universal ceiling `K_G ≤ 1.7822`.

**Detection is an identity, not a difficulty.** A shared-coin classical strategy
wins 3/4 with marginals exactly ½, matching the quantum marginals exactly, so
`marginal_divergence() = 0.0`. Joint distributions do separate:
`min KL = 0.0321` nats/round → 215 rounds at δ = 10⁻³.

**Visibility threshold.** With efficiency η and no-click assigned the default
outcome, at maximal entanglement:

```
2√2 η² + 2(1−η)² > 2  ⟺  η > 2/(1+√2) = 0.82842712…
```

The frontier is monotone: advantage and threshold move together, and the
Eberhard floor 2/3 is approached only as entanglement → 0.

**Price of anarchy.** `quantum_price_of_anarchy == correlated_price_of_anarchy`
identically. For complete information there is nothing left to define.

*Bell (1964); CHSH (1969); Tsirelson (1980); Brunner et al., arXiv:1303.2849;
Eberhard, Phys. Rev. A 47 (1993) R747; Aumann (1974); Krivine (1979).*

---

## 6. Which spins a discrete structure can protect

*Module: `qca.py`.*

The question underneath "can a QCA produce gravity": for which finite
`G < SO(3)` does the spin-s multiplet stay irreducible?

One number answers both halves. The character norm `⟨χ_l, χ_l⟩_G` is
simultaneously the sum of squared multiplicities (so `= 1` iff irreducible) and
the dimension of the commutant — the count of independent invariant couplings.
Representation theory and fine-tuning cost are the same computation.

| spin | dim | protected by | tuning cost on a cube |
| --- | --- | --- | --- |
| 1 (photon) | 3 | T, O, I | 0 |
| 2 (graviton) | 5 | **I alone** | 1 |
| ≥ 3 | 7+ | nothing | 2+ |

**The `s ≤ 2` ceiling is derived.** The largest irrep of any finite `SO(3)`
subgroup has dimension 5, and `5 = 2·2+1`. That is the Weinberg–Witten massless
helicity bound, from finite group theory rather than a stress tensor. With the
crystallographic restriction: **an emergent graviton is symmetry-protected only
on icosahedral — hence quasicrystalline — structures.**

**Relevant vs irrelevant.** Resolving by order in k, the decisive rung is n = 0:

- spin 1: `k=0` free on all three; first anisotropy at n = 1 (T), 2 (O), 4 (I)
- spin 2: T and O split at **n = 0**; I first at n = 2

An excess at n = 0 is a *gap* splitting — relevant, unsuppressed in the IR. At
n > 0 it is a velocity anisotropy — irrelevant. The spin-1 row is the control:
cubic lattices are isotropic through `k¹`, which is exactly why emergent photons
work.

**The elastic reading, and a self-weakening.** For a gauge field the object is
`C_ijkl ∈ Sym²(Sym²V)`. The machinery reproduces the entire crystal-system
table — 21, 13, 9, 7, 6, 5, 3 — and **2 for icosahedral**, the measured elastic
isotropy of icosahedral quasicrystals. Twelve independent numbers, none put in.
But it weakens the no-go: in the field reading the cubic failure is one tunable
Zener relation, not a catastrophe.

```
spin 2 as excitations   → strong no-go on any lattice
spin 2 as a gauge field → one tuned relation on a cubic lattice
```

**Fractons do not escape.** The obstruction is a property of the *field*, and
both linearised gravity and the scalar-charge fracton theory use the same field
`Sym²(V) = ℓ0 ⊕ ℓ2`. A different gauge parameter changes which polarisations
survive; it cannot fuse two distinct point-group irreps. Splitting values are
identical: T → 2, O → 1, I → 0.

*Klein's classification; crystallographic restriction theorem; Pretko,
arXiv:1707.03838.*

---

## 7. What logic actually costs

*Module: `thermo.py`.*

The proposed "Gödel–Landauer–Prigogine trilemma" is three names in a trenchcoat.
Underneath it are two real and **separable** effects.

| claim | verdict |
| --- | --- |
| erasing perfectly costs infinite heat | **false** — bounded by `ln 2` |
| there is a dissipation singularity | **true, twice**, neither as described |
| it is a phase transition | true of one, false of the other |
| the trilemma is the TUR | partly — the TUR says nothing about logic |

**Maintenance is logarithmic.** In the reliable limit the demon flux saturates at
the noise rate and the affinity grows as `2 ln(1/ε)`:

```
Σ̇ → 2γ ln(1/ε)
```

Every decade of reliability costs the same fixed increment. The coefficient 2 is
derived, not fitted — 1, 3 and 4 all fail by >10%.

**Erasure is bounded**, so the headline claim fails: `W(ε) = ln 2 − H(ε)` rises
monotonically to `ln 2` and stops.

**The real transition is the fault-tolerance threshold.** `p' = 3p² − 2p³`,
unstable fixed point at `p = 1/2`: finite dissipation below (polylog overhead,
exponent `log 3 / log 2`), unattainable at or above. **It sits at a noise value,
not at logical completeness.**

**Critical slowing down, derived:** `6p(1−p) = 3/2` exactly at threshold, so
escape takes `ln(1/δ)/ln(3/2)` levels — 5.68 per decade, measured 6, 5, 6.

**No Gödel statement enters anywhere**, and a test asserts it.

*Landauer (1961); Bennett (1982); Barato–Seifert (2015);
Gingrich–Horowitz–Perunov–England (2016); Aharonov–Ben-Or; Knill–Laflamme–Zurek.*

---

## 8. Exact stabilizer complexity

*Module: `complexity.py`.*

The Brown–Susskind brief's algorithm-side claim needs no computation:
**complexity is defined as the minimum over all circuits**, an optimised circuit
prepares the same state, and it was already in the set the minimum ranged over.
Nothing shrinks.

The state-side question is decidable on stabilizer states. BFS from `|0…0⟩` under
`{H, S, CNOT}` gives true minima. **The referee is one number:** the search must
enumerate exactly `2ⁿ∏(2ᵏ+1)` = 6, 60, 1080, 36720, 2423520 states — and does, up
to n = 5 (921 s).

| quantity | law (n ≤ 5) |
| --- | --- |
| diameter | `3n + 1` |
| mean | ≈ `2.4n` |
| GHZ, `|+⟩ⁿ` | `n` |
| line graph | `2n − 1` |
| complete graph | `3(n − 1)` |

**Structure does not imply low complexity.** GHZ and the complete-graph state are
both one-line rules yet differ by `3 − 3/n`, and the complete graph sits *exactly
four gates* below the diameter at every size.

**Two of those laws are false, and counting proves it.** With `N(n) ~ 2^(n²/2)`
states and `|G| = n²+n` gates, a radius-L ball holds at most `|G|^L`, so the
diameter must grow like `n²/(4 log₂ n)`. Therefore `3n+1` is **dead from n = 72**
(bound 219 vs 217), and `diameter − mean → 4` fails likewise.

**What survives, proved for every n:**

- **GHZ complexity is exactly n.** At least one Hadamard (CNOT and S map
  computational basis states to computational basis states); at least `n−1`
  CNOTs (the interaction graph must be connected or the output factorises). The
  counts are disjoint and the construction attains the bound.
- **Concentration, proved:** the fraction of states with complexity ≤ n is at
  most `|B(n)|/N(n)` — **7×10⁻¹⁸ at n = 20, 2×10⁻¹³¹ at n = 40.**

The mechanism claimed was right; its arithmetic was wrong. The separation grows
without bound rather than saturating at four.

*Gottesman; Aaronson–Gottesman; Haferkamp–Faist–Kothakonda–Eisert–Yunger
Halpern, arXiv:2106.05305.*

---

## 9. What the persistent Dirac operator can and cannot see

*Module: `dirac.py`.*

**Refutation one.** With `Γ` the grading operator, both `d` and `δ` shift degree
by one, so `ΓD = −DΓ`. `Γ` is a unitary involution, so the spectrum is symmetric
about zero; with `D² = Δ` that pins it completely:

```
spec(D) = {±√μ : μ ∈ spec(Δ)},  signs forced
```

`dirac_spectrum_from_laplacian` rebuilds `spec(D)` from `spec(Δ)` with no other
input. **They carry identical information**, and the Dirac operator inherits
every Laplacian blindspot — a Laplacian-cospectral non-isomorphic graph pair
exists at 6 vertices (none at 5) and shares a Dirac spectrum exactly.

**Refutation two.** Reflection is an isometry, so a chiral point cloud and its
mirror have **bitwise identical** distance matrices. Every distance-based
filtration is the *same filtered complex*, so persistent homology, Laplacian and
Dirac all agree. What flips is the signed volume, which is not a function of
pairwise distances.

**What survives** — and it is what the literature actually claims: the non-zero
spectrum carries strictly more than persistent *homology*, which reads only the
kernel. That is Laplacian vs homology, not Laplacian vs Dirac.

**Provenance.** The brief's citation `arXiv:2208.06456` (Rui–Wang–Wei) makes the
weaker true claim and does not support the chirality framing; the relevant work
is `arXiv:2301.10137` and `arXiv:2105.00529`. The chiral-symmetry equivalence is
standard Hodge theory. **This module refutes an inflation of the literature, not
the literature.**

---

## 10. Curvature kills supersymmetry but not chirality

*Module: `magnetic.py`. Statements 5–6 are the original mathematics of this
repository.*

**One. Chiral symmetry survives any connection.** `ΓD + DΓ = 0` holds for every
connection, flat or curved, at *literal zero* — the cancellation is structural.
The proof uses only degree shifting and never touches `d² = 0`. **Magnetic phases
do not break chirality**, contrary to the standard framing.

**Two. What curvature breaks is `D² = Δ`.** Since
`(d₁d₀c)([u,v,w]) = (σ_uv σ_vw − σ_uw) c(w)`, `d² = 0` iff every triangle has
trivial holonomy. Three separately computed quantities are one number:

| | flat | curved |
| --- | --- | --- |
| holonomy defect | 0 | 0.397339 |
| `\|d²\|` | 0 | 0.397339 |
| `D²` off-block | 0 | 0.397339 |

**Three. The one real chiral asymmetry is the index.** The ± pairing holds on the
non-zero spectrum but not on the kernel: even and odd harmonic dimensions differ
by `Σ(−1)ᵏβ_k` — disk 1, sphere 2, path 1, circle 0. **Topological, present at
zero flux, and exactly what persistent homology already reports.**

**Four.** So magnetic Dirac *does* beat its Laplacians — via curvature, not broken
chirality. The flat triple `±√3` splits into `±1.6133, ±1.7321, ±1.8432`.

### Five — interlacing under simplex insertion *(original)*

The obstruction to spectral-persistence stability is that filtration change
alters the operator's dimension. The nearest existing result is a **Lipschitz**
bound (Anh–Dik–Anh, arXiv:2506.21352).

One structural observation does better: **a newly inserted simplex has no
cofaces** — nothing above it can already contain it, by closure — so `D` gains
exactly one row and column and no existing entry changes. That is a **bordered
Hermitian matrix**, and bordered Hermitian matrices interlace:

```
λ_i(D') ≤ λ_i(D) ≤ λ_{i+1}(D')
```

- **The connection is irrelevant.** Interlacing constrains where new eigenvalues
  land, not what the new entries are — and the connection only touches the
  entries. So the bound is **uniform over all connections; curvature cannot
  degrade it.**
- **It survives the dimension change**, which is what blocks interleaving.
- **The counting function moves by at most one** off the spectrum.

Verified on 308 randomised complexes: zero interlacing failures, zero counting
violations across 12 320 off-spectrum thresholds. Curvature *anti*-correlates
with the eigenvalue shift (−0.35).

### Six — monotone spectral curves *(original)*

`λ_i(D') ≤ λ_i(D)` is not a bound but **monotonicity**: each eigenvalue index
traces a monotone curve along a filtration, with no stability constant and **no
genericity assumption** — which matters, because a "k-th eigenvalue" descriptor
is ill-defined if indices can swap, and these spectra are heavily degenerate.

Composing over m insertions:

```
λ_i(D⁽ᵐ⁾) ≤ λ_i(D) ≤ λ_{i+m}(D⁽ᵐ⁾)
```

so the counting function is **m-Lipschitz over a segment adding m simplices,
uniformly in the connection**. With statement one, the spectrum spreads
symmetrically: over a 21-step filtration the ends stayed exact mirrors while the
spread grew monotonically 2.83 → 5.67.

**Scope.** This is stability under *combinatorial* change, not *metric*
perturbation. A small point-cloud perturbation can insert many simplices at once,
so m-Lipschitz does not convert to metric-Lipschitz without bounding m. **The
metric case is open and nothing here touches it.**

*Calmon–Schaub–Bianconi, arXiv:2301.10137; Bianconi, arXiv:2106.02929,
arXiv:2309.07851; Egidi–Gittins–Habib–Peyerimhoff, arXiv:2211.08019 (note: the
diamagnetic inequality **fails** above degree zero); Wei–Wei, arXiv:2112.10906;
Anh–Dik–Anh, arXiv:2506.21352; Jung–Kang–Park, arXiv:2512.05463.*

---

## 11. Local spectral measures: girth, Wilson action, and a filtration invariant

*Module: `insertion.py`. Statements 3–5 are original.*

**The object is not new.** It is the *local density of states* of Savostianov,
Guglielmi, Schaub and Tudisco (arXiv:2502.07558, Def 4.1) — the spectral measure
of an operator at one simplex's basis vector, whose Chebyshev moments
`2[T_m(H)]_jj` are already walk moments at a simplex. An earlier draft proposed
it as a new definition; that claim is withdrawn and a test asserts the
withdrawal. What is narrower and appears unclaimed: taking it **at the moment a
simplex enters a filtration**, taking it for the **Dirac** operator, and the
moment results below.

**Why this measure.** §10 established that insertion is a *bordering*, which is
exactly the situation where a Hermitian matrix has a distinguished last basis
vector — and that vector carries its spectral measure. Its Stieltjes transform
`m(z) = <e_tau, (D-z)^-1 e_tau>` is the rank-one Donoghue M-function of
`(D, e_tau)`. Standard in degree zero (Post 2009; Pankrashkin math-ph/0512090,
where magnetic phases already enter the boundary functionals), never pushed
above vertices nor composed with a filtration.

**One. Every odd moment vanishes**, any connection — `Gamma` acts on `e_tau` by
a sign and anticommutes with `D`. Inherited from §10.

**Two. At zero flux the measure is known completely.** `M_2j = (k+1)^j` exactly
— integers `2,4,8`; `3,9,27`; `4,16,64`; `5,25,125`; `6,36,216`. With vanishing
odd moments this determines it outright:

```
mu_tau = ½ delta_{+sqrt(k+1)} + ½ delta_{-sqrt(k+1)}
```

A symmetric Bernoulli measure supported on the square root of the facet count.

**Three — the girth law.** *Flux enters the local moments at order exactly `2g`*,
where `g` is the shortest bounding cycle through the simplex:

| structure | girth | first flux-bearing moment |
| --- | --- | --- |
| tree | ∞ | **none** (blind M2–M10) |
| 2-simplex | 2 | M4 |
| edge, graph girth 3 | 3 | M6 |
| edge, graph girth 4 | 4 | M8 |

The local form of a Kesten–McKay fact: a tree is simply connected, so every
connection on it is gauge-trivial and the measure at its root cannot depend on
phases. **A closed walk sees flux only once it is long enough to enclose
something.**

**Four — the fourth-moment law, every dimension.**

```
M_4(tau) = M_2(tau)² + S(tau) + sum_g |1 - omega_g|²
```

Three terms with disjoint meanings, and **the separation is the result**: `M_2²`
is the Bernoulli baseline, `S(tau)` counts facets shared with a same-dimension
simplex (combinatorial, flux-blind), and the Wilson sum is the only place the
connection enters. Each codim-2 face `g` lies in exactly two facets, so
`tau -> f_1 -> g -> f_2 -> tau` is a canonical Hasse plaquette and `omega_g` is
its holonomy — **normalised by the sign that is exactly the `d²=0`
cancellation.** So curvature is precisely the failure of that cancellation,
measured at fourth order. Each `omega_g` is gauge-invariant to `1e-16`.

The right-hand side is a **Wilson plaquette action**, not a squared curvature:
`|1-omega|²` and `2(1 - Re omega)` are identical for unitary holonomy, but
`|F|²` is recovered only in the continuum small-flux limit. Kenyon
(Ann. Probab. 39, 2011) weights cycle-rooted spanning forests by `2 - tr(hol)` —
exactly this summand for `U(1)`, the same quantity in a determinant identity.

**Five — the filtration invariant.** Summed over an entire filtration:

```
sum_tau M_4(tau) = B(K) + P(K) + W(K)
W(K) = sum_tau M_4(tau) - B(K) - P(K)
```

The individual terms move with the insertion order; **the totals do not** —
verified across six random linear extensions per complex, spread `0` to
`2.8e-14`. So the **total Wilson action is recoverable from strictly local
spectral data**, each moment computed on a subcomplex with no global operator
ever formed. Chamseddine–Connes (hep-th/9606001) obtain Yang–Mills from the
fourth heat coefficient of a *global* trace, and the existing discrete work
(arXiv:2509.04311) also takes global traces; this assembles the same order of
the same expansion locally.

**Two retractions, both kept visible in the API.** The definition (above), and a
reported dimension-two ceiling on statement four. The ceiling was two mistakes,
not an obstruction — the sum was indexed over *triangular* faces when the
plaquettes live on *codimension-two* faces (for a tetrahedron, 6 edges against 4
triangles, index sets of different size), and the sibling term was missing.
`identity_fails_above_dimension_two` now returns `False`.

*Savostianov–Guglielmi–Schaub–Tudisco, arXiv:2502.07558; Kenyon, Ann. Probab. 39
(2011); Chamseddine–Connes, hep-th/9606001; Najem–Mrad–Elsayed, arXiv:2509.04311;
Preciado–Jadbabaie, arXiv:1107.5676 (the walk-moment lemma); Anh–Dik–Anh,
arXiv:2506.21352.*

---

## 12. Flux chirality: the one bit spectra cannot see

*Module: `rigidity.py`. Original.*

Every result above computes spectra *forward*. This asks the inverse question,
which the moment results provoke and do not answer:

> How much of a gauge field does its local spectral data determine?

Collecting the insertion moments over all simplices gives the **spectral
signature** of a connection.

**One — proved. Reversing every flux is invisible.** `D` of the conjugate
connection is the entrywise conjugate of `D`, and a diagonal moment at a real
basis vector is real, so

```
<e_tau, conj(D)^j e_tau> = conj(<e_tau, D^j e_tau>) = <e_tau, D^j e_tau>
```

Every moment, every simplex, every order, every complex. The measured residual is
**exactly `0.0`** — not a tolerance. The cancellation is entry by entry, so a test
asserts equality with zero rather than smallness: machine noise there would mean
the argument is wrong.

**Two — proved. That is the only ambiguity, and it degenerates predictably.** On
two triangles sharing an edge, gauge-fixed so the plaquette holonomies are free
coordinates, an exhaustive sweep of 1296 connections finds exactly two sharing any
given signature — the connection and its global conjugate. Reversing a *single*
plaquette is visible by order 1 against a `1e-8` threshold.

```
ambiguity group = Z/2
```

*The argument.* Write `omega_j = exp(i theta_j)` for the plaquette holonomies. A
closed walk and its reverse contribute conjugate terms, so every moment is real
and of the form `cos(a · theta)` summed over the exponent vectors `a` that closed
walks realise. The fourth moments supply `cos(theta_j)` for each plaquette
separately — that is the girth law of §11 doing the work, since a 2-simplex has
`g = 2` and so flux first appears at `M_{2g} = M_4`, one plaquette at a time.
Higher moments supply the *pair* terms `cos(theta_j + theta_k)` and
`cos(theta_j − theta_k)`, and their difference is

```
cos(theta_j - theta_k) - cos(theta_j + theta_k) = 2 sin(theta_j) sin(theta_k)
```

Now `cos(theta_j)` pins each `theta_j` up to sign, and the product
`sin(theta_j) sin(theta_k)` pins the *relative* signs: flipping `theta_j` alone
would negate that product, so it is invisible only if the product is zero. The
signs must therefore move together, and the ambiguity is one global bit rather
than one bit per plaquette. That is exactly what the sweep measures.

*The degeneracy, which the argument also predicts.* The relative-sign constraint
is vacuous precisely when `sin(theta_j) = 0` — when `omega_j = ±1` is **real**. A
real holonomy is its own conjugate, so reversing it does nothing. Measured on a
`36 × 36` sweep, and agreeing with `conjugation_acts_faithfully` in every case:

| reference indices | holonomies | matches |
| --- | --- | --- |
| (6, 13) | both non-real | 2 — both coordinates flip |
| (18, 13) | one real | 2 — only the non-real one flips |
| (6, 18) | one real | 2 |
| (18, 18) | both real (`−1`, `−1`) | **1 — conjugation is the identity** |
| (0, 13) | one real | 2 |
| (0, 0) | both real (`+1`, `+1`) | **1** |

So the sharp statement is: the spectral signature is a complete invariant of the
connection modulo gauge and one global reflection, and that reflection **acts
faithfully exactly when some plaquette holonomy is non-real**. When every holonomy
is real the signature determines the connection outright.

**The name.** The surviving bit is the **flux chirality**: whether the fluxes run
one way or the other. Gauge invariant, spectrally undetectable, and the unique
non-trivial symmetry of the signature.

**Why that word — three chiralities, one pattern.** This repository has now found
three distinct chiralities, in three modules, none of them looked for. Every one
is invisible to a spectrum:

| module | chirality | mechanism |
| --- | --- | --- |
| §9 `dirac` | geometric — point cloud vs mirror | reflection is an isometry, so distance matrices are identical |
| §10 `magnetic` | operator — the grading `Gamma` | `Gamma` needs only degree parity, so curvature cannot reach it |
| §12 `rigidity` | **flux** — sign of the holonomy | diagonal moments are real, so conjugation cancels |

Three unrelated mechanisms, one pattern: **a spectrum is built from `|.|²`-type
data and cannot resolve an orientation.** In each case the invisible thing is
exactly a `Z/2`. The first two were refutations of external briefs; the third
fell out of asking the inverse question. That they land on the same group is
either meaningful or a very tidy accident, and which is not established.

**Scope, and the hypothesis the second proof carries.** Statement one is general
in every argument. Statement two now has a proof as well as a sweep, but the proof
uses one thing worth naming: the moment data must actually *contain* the pair
terms `cos(theta_j ± theta_k)`. Those come from closed walks traversing two
plaquettes, which exist once the complex connects them. A complex whose plaquettes
sit in different connected components supplies no such walk, and there the signs
really are independent. So the theorem reads:

> On a **connected** complex the ambiguity group is `Z/2`, acting faithfully iff
> some plaquette holonomy is non-real. In general it is one `Z/2` per connected
> component that carries a non-real holonomy.

`AMBIGUITY_IS_PROVED` is `True` and `component_ambiguity_order` returns the
general count — `2^k` for `k` faithful components. Nothing here reconstructs a
connection from a signature; the theorem says only how many share one, and a test
asserts that no `reconstruct_connection` exists.

---

## 13. Local spectral moments are characters, and they obey the Fricke cubic

*Module: `character.py`. Original in its assembly; the cubic is Fricke's.*

§12 proved the rigidity theorem in trigonometry. Rewritten in the right language
it becomes a statement about a quotient of a torus — and then it stops being a
fact about Dirac operators and becomes a fact about a classical algebraic
surface, which is what makes it usable for anything but itself.

**Step one — the moments are integral characters.** Label each edge by an integer
vector, so a connection is a point `theta` of `T^r` and each Dirac entry is a
monomial `x^v`. Then

```
M_n(tau) = sum_a c_a(n, tau) x^a,   every c_a in Z,   c_{-a} = c_a
         = c_0 + sum_{a > 0} 2 c_a cos(a . theta)
```

The integrality is because `M_n` is a signed count of closed `n`-walks; the
symmetry is because a walk and its reverse are in bijection with inverse
monomials. **This is the character-level form of §12's statement one:**
conjugating the connection sends `x^a` to `x^{-a}` and therefore fixes every
moment. Verified two ways sharing no code — an exact matrix power over
`Z[x_1^±,…,x_r^±]`, and an FFT of the float moments over a grid on the torus.

**Step two — the support law is an inequality, and the obvious equality is
false.** §11 found flux entering `M_n` at order `2g`; that is the `a`-summed
statement. Per class:

> The class `a` appears in `M_n(tau)` **no earlier** than the shortest closed
> Hasse walk at `tau` carrying `x^a` — usually exactly there, not always.

The inequality is a theorem (no walk, no term). Equality is not, because the sum
is *signed* and same-length walks of the same class can cancel. They do. Checked
against breadth-first search in the `Z^r`-cover of the Hasse diagram — a genuinely
different algorithm from a matrix power. Orders on two triangles glued along an
edge:

| class | at vertex (2) | at vertex (0) | at the shared edge (0,1) |
| --- | --- | --- | --- |
| (1, 0) | 4 | 6 | never |
| (0, 1) | 10 | 6 | never |
| (1, −1) | 8 | 8 | never |
| (2, 0) | 8 | 10 | never |

The shared edge is **permanently flux-blind**: moments `4, 16, 64, 256, 1024` =
`4^{n/2}` for every connection, i.e. a two-point measure at `±2`, i.e. `e_tau` is
an eigenvector of the magnetic Hodge Laplacian at its Hasse degree whatever the
field does. Walks carrying every class reach it from length 6 on, and all of them
cancel.

**This also fixes `rigidity.SIGNATURE_ORDER = 8`.** Order 6 cannot see any
relation between two plaquettes, and relations between plaquettes are the whole
content of the rigidity proof. Eight is the first order that can. The constant had
been picked by guessing generously; it is exactly tight.

**Step two and a half — the basepoint, and a scope correction it forces.**
Chasing the flux-blind edge turned up something that has to be said out loud.
`magnetic.general_dirac` puts a simplex's whole transport on its leading edge,
which is the same as choosing each simplex's **minimal vertex** as a basepoint.
Any basepoint choice gives a gauge-covariant operator — verified, the moments are
unchanged by a vertex gauge transformation. But two *different* choices are
conjugate by a diagonal unitary only if re-basing inside a simplex is path
independent, and inside a curved simplex it is not:

| relabel two triangles so the shared edge is (2,3) | result |
| --- | --- |
| flat connection (pure gauge or zero) | identical spectra, identical moments |
| curved connection | **different spectra** |

So the Dirac operator here, and every local spectral measure built from it, is an
invariant of the **ordered** complex, not of the complex. That is a real limit on
§11 and §12 and it is stated rather than left to be found. Which simplex is
flux-blind is likewise a fact about the ordering.

What survives is exactly the character layer: **the recovered cosines and the
Fricke identity come out identical in both labellings**, because they are
functions of loop holonomies and a loop holonomy does not know what the vertices
are called. That is a reason to work at the character level, not a footnote about
it.

**Step three — the cosines satisfy a cubic.** Peel the moments in order of first
appearance and every `cos(a . theta)` comes out. For any two classes `a`, `b`, set
`u = cos(a.θ)`, `v = cos(b.θ)`, `w = cos((a−b).θ)`. Then `w − uv = sin(a.θ)
sin(b.θ)`, so `(w − uv)² = (1−u²)(1−v²)`, which expands to

```
u² + v² + w² − 2uvw = 1
```

the **Cayley cubic**, and classically the Fricke relation for the inversion
quotient of a torus. Stated here it is an exact identity among *measured local
spectral moments*: three moments at three simplices of any complex give three
numbers lying on one fixed cubic surface, whatever the connection. Measured
residual `4e-15`.

**Step four — and now §12 is a picture.** The map `theta ↦ (u, v, w)` is exactly
the quotient `T² → T²/(θ ~ −θ)`, realised as the Cayley cubic. That quotient is
the **pillowcase**: two-to-one away from the four fixed points of the inversion,
one-to-one at them. So

- the `Z/2` of §12 is the deck group of this double cover;
- its degeneracy locus — both holonomies real — is exactly the four **two-torsion
  points**, and their images are exactly the four **nodes** of the cubic, where
  `grad(u²+v²+w²−2uvw) = 2(u−vw, v−uw, w−uv)` vanishes;
- so the singular points of a surface Fricke wrote down in 1897 and the collapse
  of a spectral ambiguity found by sweeping a grid **are the same four points**.

Solving the gradient gives entries `±1` with `uvw = 1`: four sign patterns out of
eight, not eight.

**A door this opens: an exact count of distinguishable connections.** If the
signature separates inversion orbits — which is §12 — then counting distinct
signatures is counting orbits, and Burnside gives it in closed form. On the
`N`-torsion grid of rank `r`,

```
distinct signatures = ( N^r + gcd(2, N)^r ) / 2
```

with no spectral computation at all. Against brute force: `N = 11 → 61`,
`N = 12 → 74`, `N = 36 → 650`, exact. This is a test of the *theorem*, not of
arithmetic — if the signature failed to separate orbits the measured count would
come in strictly lower.

**And it reaches past what was proved.** §12 argued the ambiguity at rank two. On
a fan of three triangles — rank three, fifteen simplices — the count predicts
`63` at `N = 5` and `112` at `N = 6`, and brute force returns `63` and `112`. The
ambiguity is still exactly `Z/2` one rank up, established by *counting* rather
than by redoing the argument. That is what a key is for.

**And the general-rank theorem, with its hypothesis made checkable.** The
rank-two argument needs exactly three things from the moments: `cos(θⱼ)` for each
coordinate, and `cos(θⱼ − θₖ)` for each pair. At general rank it needs the same
list and nothing else, so:

> **Theorem.** If every standard basis class and every pairwise difference
> appears with non-zero coefficient at some simplex and some order, the signature
> determines the connection modulo gauge and one global reflection, and the
> ambiguity group is `Z/2`.

The proof is the rank-two one verbatim. What is *not* automatic is the
hypothesis: every class is carried by some closed Hasse walk on a connected
complex, but the moment is a **signed** count, and the flux-blind edge above is a
simplex where all the walks cancel. So `classes_are_resolved` discharges the
condition for a given complex rather than assuming it — it holds for fans of one,
two and three plaquettes, which is *why* the rank-three count comes out at `Z/2`,
and it fails, correctly, when the signature is truncated at order 6.

**Proving the hypothesis in general is open**, and it is the honest residue of
this section. It would need a non-cancellation lemma for signed closed-walk counts
on a Hasse diagram, and the flux-blind edge shows such a lemma cannot be
unconditional.

**And a truncation law falls out of the support law.** Below the coupling order
no moment contains a term joining two classes, so the signs are independent and
the ambiguity group is the **full** `(Z/2)^r`, not the diagonal. The same
Burnside argument then gives a different closed form:

| truncation | ambiguity group | distinct signatures |
| --- | --- | --- |
| order < 8 | `(Z/2)^r` | `((N + g)/2)^r` |
| order ≥ 8 | diagonal `Z/2` | `(N^r + g^r)/2` |

with `g = gcd(2, N)`. Both measured against brute force, and they differ: `N = 8`
at rank two gives `25` truncated against `34` full; rank three at `N = 6` gives
`64` against `112`. The sharpest form is a sweep — at order 6
`rigidity.rigidity_sweep` returns **four** connections sharing a signature and at
order 8 it returns two, the four being exactly the independent sign flips
`(6,13), (6,23), (30,13), (30,23)`.

So `SIGNATURE_ORDER = 8` is not a safety margin, it is the threshold, and one
notch below it the rigidity theorem is false. That constant was originally chosen
by guessing generously; it turns out to be exactly tight, and nothing in §12
could have told us so.

**Two more.** The Fricke residual is a consistency check on local spectral data
that needs no ground truth, since the cubic is a constraint the data satisfies by
itself. And the inverse problem now has a normal form: asking what a spectrum
determines about a connection is asking for the fibres of a map into the
coordinate ring of `T^r/±`, generated by the `cos(a.θ)` with the Fricke cubics as
relations — a coordinate system fixed before any operator is written down.

### Where the cubic comes from — a framing correction

The first version of this section said the cubic was a fact about `U(1)`
connections. **That is wrong, and the correction is worth more than the error
was.**

`U(1)` is abelian, so its character variety on a complex is `Hom(π₁, U(1)) =
H¹(K; U(1))` — a **torus**, flat, with no relations and no cubic surface anywhere.
The Fricke identity, the Cayley cubic and the pillowcase are `SL₂` objects: they
exist because `SL₂` has a Cayley–Hamilton relation `Z² − tr(Z)Z + I = 0`, and
`U(1)` has nothing of the kind. Taken at face value, "`U(1)` moments satisfy a
cubic" is a category error.

What is really happening is better. The moments are *forced* to be `cos(a·θ)` —
a closed walk and its reverse contribute conjugate terms, so nothing else
survives. And `2 cos θ` is exactly the trace of `diag(e^{iθ}, e^{−iθ}) ∈ SU(2)`.
So:

> The local spectral moments of a `U(1)` magnetic Dirac operator are
> automatically **Weyl-invariant functions on a maximal torus of `SU(2)`**.

They land in trace coordinates whether or not anyone asks. Substituting `X = 2u`
turns the identity proved above into `X² + Y² + Z² − XYZ − 4 = 0`, the Cayley
cubic in standard normalisation — the `κ = 2` level set of the commutator trace,
i.e. the *commuting* locus of `X(F₂, SU(2))`. And `(T × T)/W` is the pillowcase.

Three things the correct framing hands over and the wrong one hid:

- **The `Z/2` is the Weyl group.** `W = N(T)/T = Z/2` for `SU(2)`. Not a
  coincidence, not an artifact of `|·|²`-type data. That is the entire content of
  the rigidity ambiguity, and it explains why all three chiralities in §9, §10,
  §12 land on the same group only insofar as this one does — the other two need
  their own explanations and do not get one from here.
- **The four nodes are orbifold points** — the points of `T²` with non-trivial
  `W`-stabiliser, i.e. the two-torsion. Singular on the cubic and degenerate for
  `rigidity` are the same fact stated twice. They are *not* the
  eigenvalue-collision points — but that withdrawal says only "my nodes are not
  those points", **not** that the apparatus for collisions is irrelevant. Conical
  intersections live over this flux torus on their own account, and at two of the
  four two-torsion points the loci do coincide, which is precisely the case
  Berkolaiko–Zelenko (arXiv:2304.04331) is built to handle.
- **The image is compact semi-algebraic, not the whole complex surface.** Every
  coordinate lies in `[−1,1]` because an `SU(2)` element has trace `2cos θ`. Those
  range restrictions are **Loll's inequalities** (hep-th/9309056): the Mandelstam
  identities alone do not cut out the `SU(2)` locus, and the extra inequalities
  are what make a pillowcase a pillowcase rather than a variety.

**And the cubic needs no invariant theory at all.** It is the trigonometric
identity `cos²A + cos²B + cos²C − 2cosA cosB cosC = 1` whenever `C = ±(A+B)`. So
`(θ₁,θ₂) ↦ (2cosθ₁, 2cosθ₂, 2cos(θ₁+θ₂))` lands on the Cayley cubic
*automatically*, because the third loop is the product of the first two. **The
cubic is the relation `θ₃ = θ₁ + θ₂`, and nothing more** — one line implying all
three bullets at once, and the honest level at which to state the result.

**What the pillowcase is here — the same error one level down, avoided.** The
character variety of the complex is still the *torus*. The pillowcase is `T/W`
where `W` is the residual symmetry of the **observable**. So:

> The pillowcase is the moduli space of the spectral **signature**, not of the
> connection.

It sits downstream of the connection moduli, and the `Z/2` is a statement about
what the measurement cannot see, not about what the connection is. The 40-vs-74
gap below is exactly a measurement of that.

One check that came out right without being aimed at: the quotient is by the
**diagonal** inversion, not a `Z/2` on each factor — because the Weyl element
conjugates both commuting matrices simultaneously, and walk reversal inverts all
holonomies at once. Same action. A per-factor `(Z/2)²` was the natural wrong
guess, and it is exactly what the *truncated* regime gives, so the distinction is
checkable rather than rhetorical.

A structural remark that follows, and that §12 had no way to see: **the pillowcase
needs dimension two.** For a graph, `π₁` is free, so `X(Γ, SU(2)) = SU(2)^b/SU(2)`
— a ball in trace coordinates, no pillowcase. The commuting relation cutting the
ball down to `(T × T)/W` is imposed by attaching a **2-cell**. Graphs give balls;
2-complexes give pillowcases. Which is the same flatness-on-2-simplices condition
the basepoint correction turns on, reached from the opposite direction.

### What the local measure buys over the global spectrum

The published obstruction is that **the spectrum does not determine the magnetic
potential** — Fabila-Carrasco, Lledó and Post construct isospectral magnetic
graphs (*Anal. Math. Phys.* 13:64, 2023).

**It bites, exactly where it claims to.** The global Dirac spectrum here loses a
third of the classes — that *is* the obstruction, measured, not evaded. What the
measurement shows is narrower and better: it obstructs the **global** invariant
and not the **local** one, because the signature is one measure per simplex while
the spectrum is their sum. On the running example at `N = 12`:

| invariant | classes |
| --- | --- |
| global Dirac spectrum | 40 |
| local signature | **74** — exactly the Burnside count |

The spectrum fuses connections the signature separates, and the signature
separates every orbit there is.

**And the exhibit is a curve, not a pair.** On two triangles glued along an edge
the entire line `θ₁ + θ₂ = π` is isospectral: the global Dirac spectrum is
constant along it to `2e-15`, while the local moments vary continuously and the
recovered cosines separate every point. A generic level `θ₁ + θ₂ = c` is *not*
isospectral (variation `1e-1`), so the curve is picked out by its half-flux
condition — the outer four-cycle carries holonomy `−1`.

**The mechanism, derived in exact integers.** `tr(D^k) = Σ_τ M_k(τ)`, so the
global spectrum is the *sum* of the local measures and the power sums fix the
characteristic polynomial. Summing the exact local expansions gives `tr(D^k)` as
an integer character expansion, and it comes out **swap-symmetric** — the
coefficient at `(a₁,a₂)` equals the one at `(a₂,a₁)`, exactly, because exchanging
the two apexes is an automorphism fixing every simplex's minimal-vertex basepoint.

On the line `θ₂ = s − θ₁` the class `a` contributes at frequency `a₁ − a₂` with
**phase** `e^{i a₂ s}`. At `s = π` that phase is `(−1)^{a₂}`, and the entire
non-constant part cancels. At `k = 8`, in integers:

| frequency | contributions | total |
| --- | --- | --- |
| 1 | `(1,0) = −256` at phase `+1`; `(0,−1) = −256` at phase `−1` | **0** |
| 2 | `(2,0) + (0,−2) = 8` at phase `+1`; `(1,−1) = 8` at phase `−1` | **0** |

Every order tested (`k = 4, 6, 8, 10`) leaves `{0: constant}` and nothing else. So
each power sum is constant on the line and so is the characteristic polynomial —
isospectrality *proved per order*, not observed in eigenvalues.

**And the diagnostic answers no.** At `s = 0` — the *other* central element, `+I`
— every phase `e^{i a₂ · 0}` is `+1`, the swap-symmetric pairs **add** instead of
cancelling, and the trace is not constant. Measured: deviation `9e-1` along
`θ₁ + θ₂ = 0` against `2e-15` along `θ₁ + θ₂ = π`. So the mechanism is **not**
"the holonomy is central" — it is specifically the *non-trivial* central element,
whose `(−1)^{a₂}` does the cancelling.

That distinguishes half-flux from flat the way **Lieb's flux phase theorem** does,
and the natural place to look for a combinatorial account is Kasteleyn theory — a
Kasteleyn orientation being a `±1` connection with holonomy `−1` around every even
face (Lieb–Loss, *Duke Math. J.* 71:337, 1993; Kenyon's cycle-rooted spanning
forests, whose determinant weight `2 − tr(hol)` is maximal at `−I`). **Whether the
cancellation above is already known in determinant form is not settled here.**

**Novelty, flatly.** *Classical, not mine:* the Fricke identity and the Cayley
cubic (Fricke–Klein, 1897); `(T × T)/W` as the pillowcase and its role in
traceless character varieties (Hedden–Herald–Kirk, *Geom. Topol.* 18:211, 2014);
that trace functions generate, hence that all moments are polynomials in finitely
many of them (Procesi, 1976); the range inequalities (Loll, hep-th/9309056);
reconstruction of a connection from its holonomies, which is solved and is not a
spectral statement (Giles, *Phys. Rev. D* 24:2160, 1981); the moduli space of
graph connections (Fock–Rosly, 1992); the flux torus as the `U(1)` character
variety with Morse theory on it (Berkolaiko, *Anal. PDE* 6:1213, 2013;
Berkolaiko–Weyand, *Phil. Trans. R. Soc. A* 372:20120522, 2014; arXiv:2212.00830)
and the treatment of its eigenvalue-collision loci (Berkolaiko–Zelenko,
arXiv:2304.04331); isospectral magnetic graphs (Fabila-Carrasco–Lledó–Post, 2023);
Burnside's lemma; the local density of states itself (arXiv:2502.07558 — see §11
for the withdrawal).

*Mine, and still unverified:* that local spectral moments of a magnetic Dirac
operator are **integral** characters with symmetric support; the per-class support
inequality and its cancellation exceptions; the basepoint scope correction; the
recovery procedure; the composite that measured moments therefore satisfy the
Cayley cubic; the Burnside count and its truncated form; and the measurement that
the local signature strictly dominates the global spectrum.

*Withdrawn:* that the cubic is a `U(1)` phenomenon — it is an `SU(2)` phenomenon
`U(1)` inherits by sitting inside as a maximal torus. And any suggestion that the
four nodes are eigenvalue degeneracies: measured, two of the four two-torsion
points carry repeated eigenvalues and two carry none, so the Weyl-degeneracy locus
and the eigenvalue-collision locus are **different sets**.

**On the flux torus, refined.** For a *graph* every connection is flat — no
2-cells — so the flux torus and the `U(1)` character variety coincide outright.
On a complex they do not, and the right statement is an exact sequence:

```
1 → H¹(K; U(1)) → U(1)^E / U(1)^V --F--> U(1)^F
```

The flux torus of the 1-skeleton is `T^{|E|−|V|+1}`; the character variety is the
**zero-curvature fibre** of the curvature map, and the non-flat connections this
repository is actually about are the other fibres. That is better than a bare
identification: Berkolaiko's Morse theory lives on the *whole* flux torus, so it
applies to the entire parameter space here, with the character variety a
distinguished submanifold inside it where Hodge theory and cohomology switch on.
**Morse theory relative to that submanifold** appears to be unasked, and is the
natural next question.

*Status of the check:* a literature sweep on all four questions was run externally.
It found no paper joining a magnetic Dirac operator on a complex to a character
variety, none joining a local density of states to the Fricke relation, a rich and
directly adjacent literature on `U(1)` inverse spectral problems for **graphs** in
which the flux torus and the `U(1)` character variety are the same space without
either community saying so, and nothing on the discrete side of the pillowcase.

---

## 14. Ledger

**Solved.** The graviton/Friedmann identification with computed coefficients.
The massive w₁₊∞ integer-Δ obstruction. The flat-space polytope mass resummation.
The FRW reparameterisation obstruction. The cascade exponent. The Euler gate law.
The ℤ³ Obukhov architecture and the sharp 30° cap. The α < 5/2 budget, twice.
Scale invariance of the coherence penalty. The full quantum-coordination package.
The spin-protection classification and its RG sharpening. The elastic table. The
fracton no-escape. The thermodynamic verdict. Exact stabilizer complexity and its
concentration proof. Both Dirac refutations. **Interlacing and monotonicity for
the magnetic Dirac operator.** The girth law, the three-term fourth-moment law and
the filtration invariant. **Both halves of the flux-chirality rigidity theorem,
including the degeneracy characterisation and the disconnected count.** The
character expansion with its integrality and symmetry, the per-class support
inequality, the basepoint scope correction, **the Cayley-cubic identity satisfied
by measured local spectral moments, and the Burnside count of distinguishable
signatures.**

**Open, and stated as open.** The FRW resummation beyond reparameterisations. The
analytic embedding for Navier–Stokes. The phase tail in the averaging estimate
(*refuted*, not merely unproven). Whether real order flow presents rank-2 payoff
structure. Three stabilizer laws still five-point fits (`|+⟩ⁿ = n`, line graph
`2n−1`, complete graph `3(n−1)`) — treat as suspect, since `3n+1` looked equally
solid and died at n = 72. **The metric case for spectral persistence stability.**
**The non-cancellation hypothesis of the general-rank rigidity theorem** — it is
checkable per complex and checked on every complex used here, but not proved, and
the flux-blind edge shows it cannot hold unconditionally.

---

## 15. The failure catalogue

The most transferable artifact in this repository.

**Wrong guesses that died in computation, each preserved as a test:**

1. "The stability condition kills the graviton sector" → it balances it.
2. Pole order 6 at a = 3 → measured 5. Coincident facets *bound* multiplicity but
   do not fix it.
3. "Letters = surviving facets" → true at a = 1 by coincidence, false at a = 2.
4. "Every gate at exactly 45°" → off by `O(N_{j−1}/N_j)` for the second mode.
5. Cap threshold 60° → 30°, because reality supplies the antipodal cap.
6. Rescaled time window in the coherence penalty — normalised amplitudes make the
   rate scale-free, so shrinking the window with the scale doubled ρ spuriously.
7. **Uniform phase floor → refuted by a complete sample.** The worst, because it
   had already been announced. Survivorship: slow draws never reached the target.
8. CE polytope missing non-negativity → collapsed every game to one vertex.
9. Nuclear norm as quantum XOR bias → returns *less than classical* at rank 1.
10. **`diameter = 3n+1`** → exact at every reachable size, false from n = 72.
11. **`diameter − mean → 4`** → same cause.
12. `helicity_two_is_isolated` → returned `False` for *gravity itself*; the gauge
    quotient leaves `{0, ±2}` and the residual zero mode is killed by the
    Hamiltonian constraint, not by gauge.

13. **The insertion measure as a new definition** → it is the published local
    density of states. Withdrawn.
14. **"Excess kurtosis = squared curvature"** → it is the Wilson plaquette
    action; `|F|²` only in the continuum limit.
15. **A dimension-two ceiling on the fourth-moment law** → two mistakes, not an
    obstruction: the sum was indexed over *triangular* faces instead of
    *codimension-two* faces, and the sibling term was missing.
16. **`M_2 = dim + 1` for every simplex** → false at dimension zero. A vertex's
    only facet is the empty face, which is not a simplex, so `M_2 = 0`. Every
    earlier check used `dimension >= 1`.
17. **A guard that rejected sweep index 0** on the stated grounds that its
    conjugate would not be a distinct grid point — but index `N/2` is equally
    self-conjugate and was always accepted. The justification was wrong even
    though the guard looked harmless, and it was hiding half the degeneracy.
18. **"The class appears at exactly the shortest walk length"** → an equality
    where only the inequality is a theorem. The moment is a *signed* count, and
    at the shared edge of two triangles every walk of every class cancels, at
    every order, forever. Stated as a law it would have been false at one
    simplex out of eleven.
19. **The Dirac operator taken for an invariant of the complex** → it is an
    invariant of the **ordered** complex. Putting the transport on the leading
    edge picks each simplex's minimal vertex as a basepoint; re-basing inside a
    curved simplex is path dependent, so relabelling changes the spectrum. Flat
    connections are unaffected, which is what identifies the cause. Found by
    chasing a flux-blind edge that turned out not to be geometric.
20. **"The moments satisfy a cubic, therefore `U(1)` connections have trace
    coordinates"** → category error. `U(1)` is abelian; its character variety is
    a flat torus with no relations. The cubic is an `SL₂` fact, available only
    because the moments are forced into `2cos θ = tr diag(e^{iθ}, e^{−iθ})`. The
    *result* survives; the *reason* was wrong, and the right reason identifies
    the `Z/2` as the **Weyl group** rather than leaving it a coincidence.
21. **The four nodes read as eigenvalue degeneracies** → they are not. Measured:
    two of the four two-torsion points carry repeated eigenvalues and two carry
    none. The Weyl-degeneracy locus and the eigenvalue-collision locus are
    different sets, and conflating them would have imported the wrong apparatus.

**The pattern behind 13–16, and the one worth carrying.** *Every time a result
was aggregated, the aggregation found a boundary case the local tests had
structurally excluded.* Interlacing over a filtration found the `t = 0` tie
where the harmonic modes sit. Generalising the moment law found the wrong index
set. Summing over a whole filtration made vertices unavoidable and found
dimension zero. Local tests share the assumptions of the local result; only
aggregation crosses them.

**A pattern worth naming.** In `complexity.py` two tests failed the same way: the
complete-graph/GHZ ratio asserted `> 2` when it is `3 − 3/n`, exactly 2 at n = 3;
and a cheap-fraction cutoff fixed at 2% when n = 3 gives 3.3%. **Both claims were
about a trend; both tests had been written as thresholds.** The measurements were
right each time — a constant had been guessed instead of read off the formula.

**Mechanical traps:**

- **sympy ordering.** `cancel((f·(w+P)).subs(w,−P))` gives `0/0 → nan`. Cancel
  *then* substitute.
- **sympy cyclotomic sums** do not close under `simplify` — already at a seventh
  of a turn. Reduce mod `Φ_L` in `ℤ[ζ_L]` instead of hoping.
- **Nelder–Mead under-converging** a constrained KL (0.0428 vs 0.0321). SLSQP
  with an explicit simplex constraint.
- **Averaging weights at the zero mode** — invisible per-mode, fatal on sums.
- **Exact `Fraction` orbits near a critical point** blow up: each level squares
  the denominator, and critical slowing down demands many levels.
- **Fixed tolerances on asymptotic quantities.** `flux → γ` saturates in the
  affinity with a residual carrying `γ/γ_D`, so one tolerance passed at γ = 1 and
  failed at γ = 3. Test the *convergence*, not a threshold.
- **Float ties at zero.** Every apparent counting-function violation in
  `magnetic.py` sat at exactly `t = 0`, where harmonic modes lie at machine
  epsilon with mixed signs. The test was wrong, not the theorem.
- **Background job hygiene.** `pkill -f "slowdown"` killed the shell running it.
- **A function stored as a class attribute becomes a bound method.**
  `phase_function` returns a closure; `self.WEIGHT(u, v)` then passes `self` as a
  third argument. Python semantics, not mathematics — move it to module level.
- **`combinations(x, 0)` is the empty tuple, and the empty set is contained in
  everything.** That silently turned "facets of a vertex" into "all other
  vertices".

---

## 16. How to read this

The results I would defend hardest are not the most impressive-looking.

**The three chiralities** of §9, §10 and §12 — three unrelated mechanisms
converging on the same `Z/2`, none of them looked for — and **the α < 5/2
coincidence** in §4 — two unrelated arguments landing on the same boundary and
saturating there — are evidence that the geometry tracks something real rather
than something arranged.

**The retraction in §4** (the phase floor), **the refutation in §8** (my own
`3n+1` law), **the two withdrawals in §11** and **the basepoint correction in
§13** are the reason to believe the first. A repository that only ever confirms
its own guesses is measuring nothing.

### On §13 in particular

It is the one section that reaches backwards. Three of its findings are about the
sections before it:

- `SIGNATURE_ORDER = 8` in §12 was chosen by guessing generously. It is exactly
  the threshold, and one notch below it the rigidity theorem is **false**. §12
  could not have discovered that about itself.
- The Dirac operator of §10–§12 is an invariant of the **ordered** complex, not
  of the complex. Nothing proved becomes false; claims of geometric invariance
  would have been wrong.
- The per-class support law, stated as the equality one expects, would have been
  false at one simplex out of eleven.

None of the three came from re-checking those sections. All three came from
changing language — writing the same theorem in terms of characters instead of
trigonometry — and then finding that the new language had opinions about the old
results. That is the transferable part, and it is worth more than the cubic.

### What is open here, stated plainly

The non-cancellation hypothesis of the general-rank theorem. It is checkable per
complex, checked on every complex used, and not proved — and the flux-blind edge
shows it cannot hold unconditionally. Everything else in §13 is either a proof, a
classical result named as such, or an exact integer count matched by brute force.
