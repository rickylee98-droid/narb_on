"""Local spectral moments are characters, and they obey the Fricke cubic.

What this module claims
-----------------------

`rigidity` proved that the spectral signature of a ``U(1)`` connection
determines it up to one global reflection, and that the reflection acts
trivially exactly at the connections whose holonomies are all real.  That proof
was written in trigonometry.  Rewritten in the right language it becomes a
statement about a **quotient of a torus**, and then it stops being a fact about
Dirac operators and starts being a fact about a classical algebraic surface --
which is what makes it useful for anything other than itself.

The translation, in four steps.

**Step one -- the moments are integral characters.**  Fix a complex and label
each edge by an integer vector, so the connection is a point ``theta`` of the
torus ``T^r`` and each Dirac entry is a monomial ``x^v``.  Then

    M_n(tau)  =  sum_a  c_a(n, tau) x^a         with every c_a in **Z**,

because ``M_n`` is a signed count of closed ``n``-walks and each walk
contributes one monomial.  Moreover ``c_{-a} = c_a``, because a walk and its
reverse are in bijection and carry inverse monomials.  So

    M_n(tau)  =  c_0  +  sum_{a > 0} 2 c_a cos(a . theta),

an integer combination of cosines.  This is verified two ways that share no
code: an exact Laurent-polynomial matrix power over ``Z[x_1^+-, ..., x_r^+-]``,
and a fast Fourier transform of the float moments over a grid on the torus.
They agree to ``1e-9`` on every case tested, and the FFT coefficients come out
integral to the same precision.

**Step two -- the support law, which is the girth law sharpened, and which is an
inequality.**  `insertion` found that flux enters ``M_n`` at order ``2g``.  That
is the ``a``-summed statement.  The refinement here is per class, and the first
thing to say about it is that the obvious form of it is false:

    the class ``a`` appears in ``M_n(tau)`` no earlier than the length of the
    shortest closed Hasse walk at ``tau`` whose monomial is ``x^a`` -- and
    usually exactly there, but not always.

The inequality is a theorem: ``M_n`` is a sum over closed ``n``-walks, so a class
no ``n``-walk carries cannot appear.  Equality is not, because the sum is
*signed* and walks of the same length and class can cancel.  They do.  In the
running example the shared edge ``(0, 1)`` has walks reaching the single-plaquette
classes at length six and the difference class at length ten, and every single
one of them cancels: its moments are ``4, 16, 64, 256, 1024`` -- exactly
``4^{n/2}`` -- for **every** connection.  Its insertion measure is the two-point
measure at ``+-2`` and the field never touches it.  `is_flux_blind` reports such
simplices, and the correct statement is the inequality plus that exceptional set.

Checked against a breadth-first search in the ``Z^r``-cover of the Hasse
diagram, which is an algorithm with nothing in common with a matrix power.  The
order depends on the simplex as well as the class, which is the whole point of a
*local* measure.  For two triangles glued along an edge:

    class          at vertex (2)      at vertex (0)
    (1, 0)              4                  6
    (0, 1)             10                  6
    (1, -1)             8                  8
    (2, 0)              8                 10

Vertex ``2`` sits on the first plaquette and sees it at ``4``, but has to walk
all the way across the complex and back for the second, so it does not see that
one until ``10``.  Vertex ``0`` lies on both and sees both at ``6``.  The
ordering is not by class size and not by simplex: it is walk length, class by
class and simplex by simplex.  The *difference* class -- the four-cycle running
around the outside -- appears at ``8`` at both, because that is the length of
that walk from anywhere on it.  And that is why `rigidity.SIGNATURE_ORDER` is
``8`` and could not have been ``6``: order ``8`` is the first order at which the
signature sees a relation *between* two plaquettes, and relations between
plaquettes are the entire content of the rigidity proof.  The constant was chosen
by guessing generously; it turns out to be exactly tight.

**Step two and a half -- the basepoint, and a scope correction it forces.**
Chasing the flux-blind edge turned up something that has to be said out loud.
`magnetic.general_dirac` puts a simplex's whole parallel transport on its leading
edge, which amounts to choosing each simplex's **minimal vertex** as a basepoint.
Any such choice gives a gauge-covariant operator -- verified, the moments are
unchanged by a vertex gauge transformation to ``1e-9``.  But two *different*
choices are conjugate by a diagonal unitary only if re-basing inside a simplex is
path independent, and inside a curved simplex it is not.  Consequence, measured:

    relabel two triangles so the shared edge is (2,3) instead of (0,1)
      flat connection      -> identical spectra, identical moments
      curved connection    -> **different spectra**

So the Dirac operator here, and every local spectral measure built from it, is an
invariant of the **ordered** complex, not of the complex.  That is a real limit
on the earlier modules and it is stated here rather than left to be found.  Which
simplex is flux-blind is likewise a fact about the ordering.

What survives is exactly the layer this module is about: the recovered cosines
and the Fricke identity come out **identical in both labellings**, to ``1e-10``,
because they are functions of loop holonomies and a loop holonomy does not know
what the vertices are called.  The character data is intrinsic even though the
moments carrying it are not -- which is a reason to work at the character level
and not a footnote about it.

**Step three -- the cosines are trace coordinates, and they satisfy a cubic.**
Peel the moments in order of first appearance and every ``cos(a . theta)`` comes
out (`recover_cosines`).  Now take any two classes ``a``, ``b`` and set

    u = cos(a . theta),   v = cos(b . theta),   w = cos((a - b) . theta).

Then ``w - uv = sin(a.theta) sin(b.theta)``, so ``(w - uv)^2 = (1-u^2)(1-v^2)``,
which expands to

    **u^2 + v^2 + w^2 - 2uvw  =  1.**

That is the **Cayley cubic**, and it is the classical Fricke relation for the
inversion quotient of a torus.  Stated here it is an *exact identity among local
spectral moments of a magnetic Dirac operator*: measure three moments at three
simplices of a complex, and the three numbers you get lie on a fixed cubic
surface in ``R^3``, whatever the connection and whatever the complex.  Measured
residual: ``4e-15``.

**Step four -- and now the rigidity theorem is a picture.**  The map

    theta  |-->  (cos, cos, cos)  =  (u, v, w)

is exactly the quotient ``T^2 -> T^2 / (theta ~ -theta)``, realised as the
Cayley cubic surface.  That quotient is the **pillowcase**.  It is two-to-one
away from the four fixed points of the inversion, and one-to-one at them.  So:

  * the ``Z/2`` of `rigidity` is the deck group of this double cover;
  * the degeneracy locus measured there -- both holonomies real -- is exactly
    the **four two-torsion points**, and their images are exactly the **four
    nodes of the Cayley cubic**, the points where its gradient vanishes;
  * the *singularities of a classical cubic surface* and the *collapse of a
    spectral ambiguity* are the same four points.

That last line is the reason to write this module.  The degeneracy was found by
sweeping a grid and noticing that two cases gave one match instead of two.  It
is not a special case; it is the singular locus of a surface Fricke wrote down
in 1897.

Two warnings on that paragraph, both earned the hard way and both expanded
below.  The cubic is **not** a ``U(1)`` phenomenon -- see the framing correction
-- and the four nodes are **not** eigenvalue degeneracies: measured on the
running example, two of the four two-torsion points carry repeated eigenvalues
and two carry none, so the Weyl-degeneracy locus and the eigenvalue-collision
locus are different sets.

Doors this opens
----------------

Five, and they are the point of the exercise -- each is a question about
spectra that becomes a finite computation once the geometry is named.  The first
three are consequences that fall out; the last two are what the framework is
for.

**One, an exact count of distinguishable connections.**  If the signature
separates the orbits of the inversion -- which is the rigidity theorem -- then
counting distinct signatures is counting orbits, and Burnside's lemma gives it
in closed form.  On the ``N``-torsion grid of rank ``r``,

    distinct signatures  =  ( N^r  +  gcd(2, N)^r ) / 2.

No spectral computation required.  Measured against brute force: ``N = 11``
gives ``61``, ``N = 12`` gives ``74``, ``N = 36`` gives ``650``, all exact.  A
lower bound on how much a connection can hide, from a nineteenth-century
counting lemma.

And it is a *test that reaches past what was proved*.  `rigidity` argued the
ambiguity at rank two.  On a fan of three triangles -- rank three, fifteen
simplices -- the count predicts ``63`` at ``N = 5`` and ``112`` at ``N = 6``,
and brute force returns ``63`` and ``112``.  So the ambiguity is still exactly
``Z/2`` one rank up, and that was established by counting rather than by
redoing the argument.  If a rank-three complex admitted an extra coincidence
the measured count would come in low, and it does not.

**Two, the general-rank theorem, with its hypothesis made checkable.**  The
rank-two argument needs exactly three things from the moments:
``cos(theta_j)`` for each coordinate, to pin each angle up to sign, and
``cos(theta_j - theta_k)`` for each pair, to pin the relative signs.  At general
rank it needs the same list -- a basis and its pairwise differences -- and
nothing else.  So:

    **Theorem.**  If every standard basis class and every pairwise difference
    appears with non-zero coefficient at some simplex and some order, then the
    signature determines the connection modulo gauge and one global reflection,
    and the ambiguity group is ``Z/2``.

The proof is the rank-two one verbatim.  What is *not* automatic is the
hypothesis: every class is carried by some closed Hasse walk on a connected
complex, but the moment is a **signed** count, and step two exhibits a simplex
where the walks all cancel.  So the hypothesis is a real condition rather than a
formality, and `classes_are_resolved` discharges it for a given complex instead
of assuming it.  It holds for fans of one, two and three plaquettes -- which is
why the rank-three count above comes out at ``Z/2`` -- and it fails, correctly,
when the signature is truncated at order six.

Proving the hypothesis in general is open.  It would need a non-cancellation
lemma for signed closed-walk counts on a Hasse diagram, and the flux-blind edge
shows such a lemma cannot be unconditional.

**Three, the truncation law, which falls straight out of the support law.**
Below `COUPLING_ORDER` no moment contains a term joining two classes, so
the signs are independent, the ambiguity group is the *full* ``(Z/2)^r`` rather
than the diagonal, and the same Burnside argument gives a different closed form:

    order  < 8    orbits = ( (N + g) / 2 )^r          g = gcd(2, N)
    order >= 8    orbits = ( N^r + g^r ) / 2

Measured, and they are genuinely different -- ``N = 8`` at rank two gives ``25``
truncated against ``34`` full; rank three at ``N = 6`` gives ``64`` against
``112``.  The sharpest form is a sweep: `rigidity.rigidity_sweep` at order six
returns **four** connections sharing a signature, at order eight it returns two,
and the four are exactly the independent sign flips ``(6,13), (6,23), (30,13),
(30,23)``.  So `rigidity.SIGNATURE_ORDER = 8` is not a safety margin, it is the
threshold, and one notch below it the theorem is false.

**Four, a consistency test for local spectral data that needs no ground truth.**
The Fricke residual is computable from measured moments alone.  It does not
compare against a known connection, because there is nothing to compare to: the
cubic is a constraint the data must satisfy by itself.  Any pipeline computing
local densities of states on a complex with a magnetic field can check it, and a
failure localises to a genuine error rather than to a modelling choice.

**Five, the right normal form for the inverse problem.**  Asking what a
spectrum determines about a connection is asking for the fibres of a map into
the coordinate ring of ``T^r/W``.  In rank ``r`` that ring is generated by the
``cos(a . theta)`` with the Fricke cubics as relations, so the inverse problem
has a coordinate system and a known set of relations before any operator is
written down.  The rank-two case is a surface with four nodes; the general case
is the same statement with more generators.

Where the cubic actually comes from -- a framing correction
-----------------------------------------------------------

The first version of this module said the cubic was a fact about ``U(1)``
connections.  **That is wrong and the correction is worth more than the error.**

``U(1)`` is abelian, so its character variety on a complex is
``Hom(pi_1, U(1)) = H^1(K; U(1))``, a **torus** -- flat, with no relations and no
cubic surface anywhere.  The Fricke identity, the Cayley cubic and the pillowcase
are ``SL_2`` objects: they exist because ``SL_2`` has a Cayley-Hamilton relation
``Z^2 - tr(Z) Z + I = 0``, and ``U(1)`` has nothing of the kind.  Taken at face
value the claim "``U(1)`` moments satisfy a cubic" is a category error.

What is actually happening is better.  The moments are forced to be
``cos(a . theta)`` -- a closed walk and its reverse contribute conjugate terms,
so nothing else can survive.  But ``2 cos(theta)`` is exactly the **trace of the
diagonal ``SU(2)`` matrix** ``diag(e^{i theta}, e^{-i theta})``.  So:

    the local spectral moments of a ``U(1)`` magnetic Dirac operator are
    automatically Weyl-invariant functions on a maximal torus of ``SU(2)``.

They land in the trace coordinates whether or not anyone asks them to, because
walk reversal *is* the Weyl group.  With ``X = 2u``, ``Y = 2v``, ``Z = 2w`` the
identity proved above becomes ``X^2 + Y^2 + Z^2 - XYZ - 4 = 0``, which is the
Cayley cubic in the standard normalisation -- the ``kappa = 2`` level set of the
commutator trace, i.e. the **commuting** locus of ``X(F_2, SU(2))``.  And
``(T x T)/W`` is precisely the pillowcase.

**And the cubic needs no invariant theory at all.**  It is a trigonometric
identity: ``cos^2 A + cos^2 B + cos^2 C - 2 cos A cos B cos C = 1`` whenever
``C = +-(A + B)``.  So ``(theta_1, theta_2) -> (2cos theta_1, 2cos theta_2,
2cos(theta_1 + theta_2))`` lands on the Cayley cubic *automatically*, because the
third loop is the product of the first two.  **The cubic is the relation
``theta_3 = theta_1 + theta_2``, and nothing more.**  That one line implies all
three bullets below at once, and it is the honest level at which to state the
result.

**What the pillowcase is here -- the same error one level down, avoided.**  The
character variety of the complex is still the *torus*.  The pillowcase is
``T/W`` where ``W`` is the residual symmetry of the **observable**.  So:

    the pillowcase is the moduli space of the spectral *signature*, not of the
    connection.

It is downstream of the connection moduli, and the ``Z/2`` is a statement about
what the measurement cannot see -- not about what the connection is.  The ``40``
against ``74`` below is exactly a measurement of that gap.

One check that came out right without being aimed at: the pillowcase quotient is
by the **diagonal** inversion ``(theta_1, theta_2) -> (-theta_1, -theta_2)``, not
by a ``Z/2`` on each factor, because the Weyl element conjugates both commuting
matrices simultaneously.  Walk reversal inverts *all* holonomies at once.  Same
action -- which is evidence the identification is structural and not
numerological, since a per-factor ``Z/2`` would have been the natural wrong guess
and is what the truncated regime actually gives.

Three consequences that the correct framing hands over and the wrong one hid:

  * **The ``Z/2`` is the Weyl group.**  Not a coincidence, not an accident of
    ``|.|^2``-type data.  ``W = N(T)/T = Z/2`` for ``SU(2)``, and that is the
    whole content of the rigidity theorem's ambiguity.
  * **The four nodes are orbifold points**, the points of ``T^2`` with
    non-trivial ``W``-stabiliser -- the two-torsion.  Their being singular on the
    cubic and their being the degeneracy locus of `rigidity` are the same fact
    stated twice.  They are *not* the eigenvalue-collision points, and the
    withdrawal below says only that -- **not** that the Morse-theoretic apparatus
    for collisions is irrelevant.  Conical intersections exist over this flux
    torus regardless of where the orbifold points are, and at two of the four
    two-torsion points the two loci happen to coincide, which is precisely the
    case Berkolaiko-Zelenko (arXiv:2304.04331) is built to handle.
  * **The image is compact and semi-algebraic, not the whole complex surface.**
    Every coordinate lies in ``[-1, 1]`` because an ``SU(2)`` element has trace
    ``2 cos theta``.  Those range restrictions are Loll's inequalities
    (hep-th/9309056): the Mandelstam identities alone do not cut out the
    ``SU(2)`` locus, and the extra inequalities are exactly what makes a
    pillowcase a pillowcase rather than a variety.

A structural remark that follows and that I had no way to see before: **the
pillowcase needs dimension two.**  For a graph, ``pi_1`` is free, so
``X(Gamma, SU(2)) = SU(2)^b / SU(2)``, a ball in trace coordinates -- no
pillowcase.  The commuting relation that cuts the ball down to ``(T x T)/W`` is
imposed by attaching a **2-cell**.  Graphs give balls; 2-complexes give
pillowcases.  Which is the same flatness-on-2-simplices condition that the
basepoint correction above turns on, arrived at from the opposite direction.

What the local measure buys over the global spectrum
----------------------------------------------------

The published obstruction is that **the spectrum does not determine the magnetic
potential**: Fabila-Carrasco, Lledo and Post construct isospectral magnetic
graphs (Anal. Math. Phys. 13:64, 2023).

It bites, and it bites exactly where it claims to.  The global Dirac spectrum
here loses a third of the classes -- that *is* the obstruction, measured, not
evaded.  What the measurement shows is narrower and better: it is an obstruction
to the **global** invariant and not to the **local** one, because the signature
is one measure per simplex while the spectrum is their sum.  On the running
example at ``N = 12``:

    global Dirac spectrum   ->  40 classes
    local signature         ->  74 classes  = the Burnside count exactly

So the spectrum fuses connections the signature separates, and the signature
separates every orbit there is.  The obstruction is real and it is strictly
weaker than the local statement.

The exhibit is sharper than a pair.  On two triangles glued along an edge the
**entire curve** ``theta_1 + theta_2 = pi`` is isospectral: the global Dirac
spectrum is constant along it to ``2e-15``, while the local moments vary
continuously and the recovered cosines separate every point of it.  A generic
level ``theta_1 + theta_2 = c`` is not isospectral, so the curve is picked out by
its half-flux condition -- the outer four-cycle carries holonomy ``-1``.

**The mechanism, derived.**  ``tr(D^k) = sum_tau M_k(tau)``, so the global
spectrum is the *sum* of the local measures and the power sums fix the
characteristic polynomial.  Summing the exact local expansions gives ``tr(D^k)``
as an integer character expansion (`trace_expansion`), and it comes out
**swap-symmetric**: the coefficient at ``(a_1, a_2)`` equals the one at
``(a_2, a_1)``, exactly, because exchanging the two apexes is an automorphism of
the complex that fixes every simplex's minimal-vertex basepoint.

On the line ``theta_2 = s - theta_1`` the class ``a`` contributes at frequency
``a_1 - a_2`` with **phase** ``exp(i a_2 s)``.  At ``s = pi`` that phase is
``(-1)^{a_2}``, and the whole non-constant part cancels.  At ``k = 8``, in
integers:

    frequency 1:   (1,0) gives -256 at phase +1
                   (0,-1) gives -256 at phase -1     ->  0
    frequency 2:   (2,0) and (0,-2) give 4 + 4 at phase +1
                   (1,-1) gives 8 at phase -1        ->  0

Every order tested (``k = 4, 6, 8, 10``) leaves ``{0: constant}`` and nothing
else, in exact integer arithmetic -- so every power sum is constant on the line
and the characteristic polynomial is too.  That is the isospectrality, proved per
order rather than observed in eigenvalues (`trace_is_constant_on_line`).

**And the sign is the whole story.**  At ``s = 0`` -- the *other* central element,
``+I`` -- every phase ``exp(i a_2 . 0)`` is ``+1``, so the swap-symmetric pairs
**add** instead of cancelling and the trace is not constant.  Measured: deviation
``9e-1`` along ``theta_1 + theta_2 = 0`` against ``2e-15`` along
``theta_1 + theta_2 = pi``.  So the mechanism is **not** "the holonomy is
central".  It is specifically the *non-trivial* central element, whose
``(-1)^{a_2}`` is what does the cancelling.

That distinguishes half-flux from flat in exactly the way Lieb's flux phase
theorem does, and the natural place to look for a combinatorial account is
Kasteleyn theory -- a Kasteleyn orientation being a ``+-1`` connection with
holonomy ``-1`` around every even face (Lieb and Loss, "Fluxes, Laplacians, and
Kasteleyn's theorem," Duke Math. J. 71:337, 1993; Kenyon's cycle-rooted spanning
forests, whose determinant weight ``2 - tr(hol)`` is maximal at ``-I``).  Whether
the cancellation above is an instance of something already known in determinant
form is **not settled here**.

Novelty, stated flatly
----------------------

**Classical, not mine:** the Fricke identity and the Cayley cubic (Fricke-Klein,
1897); ``(T x T)/W`` as the pillowcase and its role in traceless character
varieties (Hedden-Herald-Kirk, Geom. Topol. 18:211, 2014); that trace functions
generate, hence that all moments are polynomials in finitely many of them
(Procesi, 1976); the range inequalities (Loll, hep-th/9309056); reconstruction of
a connection from its holonomies, which is solved and not a spectral statement
(Giles, Phys. Rev. D 24:2160, 1981); the moduli space of graph connections
(Fock-Rosly, 1992); the flux torus as the ``U(1)`` character variety together
with Morse theory on it (Berkolaiko, Anal. PDE 6:1213, 2013; Berkolaiko-Weyand,
Phil. Trans. R. Soc. A 372:20120522, 2014; arXiv:2212.00830) and the treatment of
its eigenvalue-collision loci (Berkolaiko-Zelenko, arXiv:2304.04331); isospectral
magnetic graphs (Fabila-Carrasco-Lledo-Post, Anal. Math. Phys. 13:64, 2023);
Burnside's lemma; the local density of states itself (Savostianov et al.,
arXiv:2502.07558, and see `insertion` for the withdrawal of my claim on it).

**Mine, and still unverified:** that the local spectral moments of a magnetic
Dirac operator are *integral* characters with symmetric support; the per-class
support inequality and its cancellation exceptions; the basepoint scope
correction; the recovery procedure; the composite that measured local spectral
moments therefore satisfy the Cayley cubic; the Burnside count of distinguishable
signatures and its truncated form; and the measurement that the local signature
strictly dominates the global spectrum, with the half-flux isospectral curve as
the exhibit.

**On the flux torus, refined.**  For a *graph* every connection is flat -- no
2-cells -- so the flux torus and the ``U(1)`` character variety coincide outright.
On a complex they do not, and the correct statement is an exact sequence:

    1 -> H^1(K; U(1)) -> U(1)^E / U(1)^V --F--> U(1)^F

The flux torus of the 1-skeleton is ``T^{|E|-|V|+1}``; the character variety is
the **zero-curvature fibre** of the curvature map, and the non-flat connections
this repository is actually about are the other fibres.  That is better than a
bare identification: Berkolaiko's Morse theory lives on the *whole* flux torus,
so it applies to the entire parameter space here, with the character variety a
distinguished submanifold inside it where Hodge theory and cohomology switch on.
Morse theory *relative* to that submanifold is a question nobody appears to have
asked, and it is the natural next thing to ask.

**Status of the check.**  A literature sweep on all four questions was run
externally (arxiv.org is unreachable from this environment).  It returned: no
paper joining a magnetic Dirac operator on a complex to a character variety; no
paper joining a local density of states to the Fricke relation; a rich and
directly adjacent literature on inverse spectral problems for ``U(1)`` on
*graphs*, in which the flux torus and the ``U(1)`` character variety are the same
space without either community saying so; and nothing at all on the discrete side
of the pillowcase.  So the assembly appears unclaimed and the framing was wrong,
which is the outcome this section now reflects.

**Withdrawn:** the claim that the cubic is a ``U(1)`` phenomenon.  It is an
``SU(2)`` phenomenon that ``U(1)`` inherits by sitting inside as a maximal torus.
Also withdrawn: any suggestion that the four nodes are eigenvalue degeneracies.
Measured on the running example, two of the four two-torsion points carry
repeated eigenvalues and two carry none, so the ``W``-degeneracy locus and the
eigenvalue-collision locus are **different sets** and must not be conflated.
The withdrawal is "my nodes are not those points", not "that apparatus does not
apply" -- conical intersections live over this flux torus on their own account,
and at the two points where the loci do coincide Berkolaiko-Zelenko is exactly
what handles them.
"""

from __future__ import annotations

import cmath
from collections import deque
from itertools import combinations
from typing import Callable, Sequence

import numpy as np

import insertion
import magnetic

__all__ = [
    "IntegerConnection",
    "TWO_TRIANGLES",
    "TWO_TRIANGLE_CONNECTION",
    "angles_from",
    "canonical_class",
    "character_expansion",
    "character_expansion_by_transform",
    "expansions_agree",
    "expansion_is_symmetric",
    "expansion_is_integral",
    "first_order_carrying",
    "shortest_walk_carrying",
    "support_order_is_at_least_walk_length",
    "support_law_holds",
    "is_flux_blind",
    "BASEPOINT_IS_THE_MINIMAL_VERTEX",
    "relabel_complex",
    "relabel_angles",
    "relabelling_changes_the_spectrum",
    "pure_gauge",
    "recover_cosines",
    "fricke_residual",
    "fricke_identity_holds",
    "fricke_coordinates",
    "CAYLEY_CUBIC_NODES",
    "is_cayley_node",
    "node_count",
    "plaquette_fan",
    "COUPLING_ORDER",
    "WEYL_GROUP_ORDER",
    "trace_coordinate",
    "standard_cayley_residual",
    "global_spectrum",
    "spectra_agree",
    "HALF_FLUX_IS_ISOSPECTRAL",
    "half_flux_curve",
    "local_data_beats_the_spectrum",
    "trace_expansion",
    "trace_restricted_to_line",
    "trace_is_constant_on_line",
    "resolving_simplex",
    "classes_are_resolved",
    "moments_couple_the_signs",
    "truncated_ambiguity_order",
    "distinguishable_signature_count",
    "measured_signature_count",
    "TOLERANCE",
]

#: Comparisons unless stated otherwise.  The Fricke residual measures around
#: ``1e-15``; the FFT-versus-exact agreement around ``1e-11``.
TOLERANCE: float = 1e-9

#: An **integral connection**: each sorted edge carries an integer vector, and
#: the ``U(1)`` weight of that edge at the torus point ``theta`` is
#: ``exp(i v . theta)``.  Edges absent from the mapping are flat.
IntegerConnection = dict[tuple[int, int], tuple[int, ...]]

#: The running example: two triangles glued along the edge ``(0, 1)``.
TWO_TRIANGLES: tuple[tuple[int, ...], ...] = insertion.close_under_faces(
    [(0, 1, 2), (0, 1, 3)]
)

#: Gauge-fixed so the two plaquette holonomies are the torus coordinates.  The
#: three edges through vertex ``0`` are flat, which is a choice of gauge and not
#: a restriction: every connection on this complex is gauge equivalent to one of
#: these.
TWO_TRIANGLE_CONNECTION: IntegerConnection = {(1, 2): (1, 0), (1, 3): (0, 1)}


# ---------------------------------------------------------------------------
# integral connections
# ---------------------------------------------------------------------------


def _validate(connection: IntegerConnection, rank: int) -> None:
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    for edge, vector in connection.items():
        if len(edge) != 2 or edge[0] >= edge[1]:
            raise ValueError(f"edge key {edge} must be a sorted pair")
        if len(vector) != rank:
            raise ValueError(
                f"edge {edge} carries a vector of length {len(vector)}, "
                f"expected rank {rank}"
            )


def angles_from(
    connection: IntegerConnection, thetas: Sequence[float]
) -> dict[tuple[int, int], float]:
    """Evaluate an integral connection at a torus point: ``v -> v . theta``."""
    _validate(connection, len(thetas))
    return {
        edge: float(sum(component * angle for component, angle in zip(vector, thetas)))
        for edge, vector in connection.items()
    }


def canonical_class(vector: Sequence[int]) -> tuple[int, ...]:
    """Pick a representative of ``{a, -a}``: the lexicographically larger one.

    The expansion is symmetric under negation, so a class and its negative carry
    the same information.  Reporting both would double every support set and
    make the counts harder to read, not easier.
    """
    forward = tuple(int(component) for component in vector)
    backward = tuple(-component for component in forward)
    return max(forward, backward)


# ---------------------------------------------------------------------------
# the exact expansion: a matrix power over Z[x^+-]
# ---------------------------------------------------------------------------

_Laurent = dict[tuple[int, ...], int]


def _zero(rank: int) -> tuple[int, ...]:
    return (0,) * rank


def _add(target: _Laurent, vector: tuple[int, ...], coefficient: int) -> None:
    if coefficient == 0:
        return
    total = target.get(vector, 0) + coefficient
    if total == 0:
        target.pop(vector, None)
    else:
        target[vector] = total


def _multiply(
    left: Sequence[Sequence[_Laurent]], right: Sequence[Sequence[_Laurent]]
) -> list[list[_Laurent]]:
    size = len(left)
    result: list[list[_Laurent]] = [[{} for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for middle in range(size):
            entry = left[row][middle]
            if not entry:
                continue
            for column in range(size):
                other = right[middle][column]
                if not other:
                    continue
                cell = result[row][column]
                for vector, coefficient in entry.items():
                    for shift, weight in other.items():
                        _add(
                            cell,
                            tuple(a + b for a, b in zip(vector, shift)),
                            coefficient * weight,
                        )
    return result


def _dirac_over_laurent(
    simplices: Sequence[tuple[int, ...]],
    connection: IntegerConnection,
    rank: int,
) -> list[list[_Laurent]]:
    """`magnetic.general_dirac`, but with entries in ``Z[x_1^+-, ..., x_r^+-]``.

    Same incidence structure and the same ``(-1)^position`` signs; the only
    change is that a phase becomes a monomial instead of a complex number.  Kept
    separate from `magnetic` on purpose: the whole value of the exact expansion
    is that it is arrived at without touching the float path.
    """
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    index = {simplex: position for position, simplex in enumerate(ordered)}
    size = len(ordered)
    matrix: list[list[_Laurent]] = [[{} for _ in range(size)] for _ in range(size)]
    for simplex in ordered:
        if len(simplex) < 2:
            continue
        column = index[simplex]
        edge = (simplex[0], simplex[1])
        for position in range(len(simplex)):
            face = simplex[:position] + simplex[position + 1 :]
            if face not in index:
                raise ValueError(f"{simplex} has a face {face} outside the complex")
            row = index[face]
            vector = connection.get(edge, _zero(rank)) if position == 0 else _zero(rank)
            sign = (-1) ** position
            _add(matrix[row][column], tuple(vector), sign)
            _add(matrix[column][row], tuple(-component for component in vector), sign)
    return matrix


def character_expansion(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    connection: IntegerConnection,
    order: int,
    rank: int,
) -> dict[tuple[int, ...], int]:
    """``M_order(tau)`` as an exact integer combination of characters ``x^a``.

    Computed by raising the Laurent-coefficient Dirac matrix to the given power
    and reading the diagonal entry.  Exact integer arithmetic throughout; there
    is no floating point anywhere in this path.

    Returns the full support including both ``a`` and ``-a``; use
    `canonical_class` to fold it.
    """
    if order < 0:
        raise ValueError(f"moment order must be non-negative, got {order}")
    _validate(connection, rank)
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    if simplex not in ordered:
        raise ValueError(f"{simplex} is not in the complex")
    position = ordered.index(simplex)
    size = len(ordered)
    if order == 0:
        return {_zero(rank): 1}
    matrix = _dirac_over_laurent(ordered, connection, rank)
    powered = matrix
    for _ in range(order - 1):
        powered = _multiply(powered, matrix)
    return dict(powered[position][position])


def character_expansion_by_transform(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    connection: IntegerConnection,
    order: int,
    rank: int,
    resolution: int = 16,
) -> dict[tuple[int, ...], float]:
    """The same expansion, obtained instead by a Fourier transform of floats.

    Evaluate the float moment on a ``resolution^rank`` grid on the torus and
    transform.  This shares no code with `character_expansion`: it goes through
    `insertion.insertion_moment` and hence `magnetic.general_dirac`, in complex
    arithmetic, while the exact path is integer arithmetic on monomials.  Their
    agreement is the referee for the whole module.

    ``resolution`` must exceed twice the largest exponent present or the
    transform aliases; the default of ``16`` is ample for the orders used here.
    """
    if resolution < 4:
        raise ValueError(f"resolution must be at least four, got {resolution}")
    _validate(connection, rank)
    step = 2 * np.pi / resolution
    shape = (resolution,) * rank
    samples = np.zeros(shape, dtype=float)
    for flat in range(resolution**rank):
        cell = np.unravel_index(flat, shape)
        thetas = [step * component for component in cell]
        weight = insertion.phase_function(angles_from(connection, thetas))
        samples[cell] = insertion.insertion_moment(simplices, simplex, weight, order)
    transformed = np.fft.fftn(samples) / samples.size
    result: dict[tuple[int, ...], float] = {}
    for flat in range(transformed.size):
        cell = np.unravel_index(flat, shape)
        value = transformed[cell]
        if abs(value) <= TOLERANCE:
            continue
        if abs(value.imag) > TOLERANCE:
            raise ArithmeticError(
                f"transform coefficient at {tuple(int(c) for c in cell)} is not "
                f"real: {value}; the moments are real, so this means the grid "
                "aliased -- raise the resolution"
            )
        vector = tuple(
            int(component)
            if component <= resolution // 2
            else int(component) - resolution
            for component in cell
        )
        result[vector] = float(value.real)
    return result


def expansions_agree(
    exact: dict[tuple[int, ...], int],
    transformed: dict[tuple[int, ...], float],
    tolerance: float = TOLERANCE,
) -> bool:
    """Do the integer path and the float path give the same character expansion?"""
    if set(exact) != set(transformed):
        return False
    return all(
        abs(exact[vector] - transformed[vector]) <= tolerance for vector in exact
    )


def expansion_is_symmetric(expansion: dict[tuple[int, ...], int]) -> bool:
    """``c_{-a} = c_a`` for every class.

    True because reversing a closed walk is an involution on closed walks that
    inverts the monomial and preserves the sign -- each step contributes its own
    conjugate entry, and the Dirac matrix is Hermitian with real incidence
    signs.  This is the character-level form of statement one of `rigidity`:
    conjugating the connection sends ``x^a`` to ``x^{-a}`` and therefore fixes
    every moment.
    """
    for vector, coefficient in expansion.items():
        opposite = tuple(-component for component in vector)
        if expansion.get(opposite) != coefficient:
            return False
    return True


def expansion_is_integral(
    transformed: dict[tuple[int, ...], float], tolerance: float = TOLERANCE
) -> bool:
    """Are the transform's coefficients rational integers?

    They must be: each is a signed count of closed walks.  Checking it on the
    float path is a test of the float path, not of the theorem.
    """
    return all(
        abs(value - round(value)) <= tolerance for value in transformed.values()
    )


# ---------------------------------------------------------------------------
# the support law
# ---------------------------------------------------------------------------


def first_order_carrying(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    connection: IntegerConnection,
    target: Sequence[int],
    upto: int,
    rank: int,
) -> int | None:
    """Least order whose expansion contains the class ``target``, or ``None``.

    Searched by expanding, which is the definition.  `shortest_walk_carrying`
    answers the same question by walking, which is not.
    """
    if upto < 0:
        raise ValueError(f"upto must be non-negative, got {upto}")
    wanted = tuple(int(component) for component in target)
    for order in range(upto + 1):
        expansion = character_expansion(simplices, simplex, connection, order, rank)
        if wanted in expansion:
            return order
    return None


def shortest_walk_carrying(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    connection: IntegerConnection,
    target: Sequence[int],
    upto: int,
    rank: int,
) -> int | None:
    """Length of the shortest closed Hasse walk at ``tau`` with monomial ``x^target``.

    Breadth-first search in the ``Z^r``-cover of the Hasse diagram: a state is a
    simplex together with the accumulated exponent vector, and the search stops
    on returning to ``tau`` with the accumulated vector equal to ``target``.

    Nothing here multiplies a matrix, which is the point -- it is an independent
    route to the same number, and their agreement is the support law.  Exponent
    vectors are unbounded in principle and bounded in practice by ``upto``,
    since a walk of length ``n`` cannot accumulate more than ``n`` monomials.
    """
    if upto < 0:
        raise ValueError(f"upto must be non-negative, got {upto}")
    _validate(connection, rank)
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    if simplex not in ordered:
        raise ValueError(f"{simplex} is not in the complex")
    wanted = tuple(int(component) for component in target)
    if len(wanted) != rank:
        raise ValueError(f"target {target} has length {len(wanted)}, expected {rank}")

    matrix = _dirac_over_laurent(ordered, connection, rank)
    index = {face: position for position, face in enumerate(ordered)}
    steps: list[list[tuple[int, tuple[int, ...]]]] = [
        [
            (column, vector)
            for column in range(len(ordered))
            for vector in matrix[row][column]
        ]
        for row in range(len(ordered))
    ]

    start = index[simplex]
    frontier: deque[tuple[int, tuple[int, ...], int]] = deque(
        [(start, _zero(rank), 0)]
    )
    seen = {(start, _zero(rank))}
    while frontier:
        node, accumulated, length = frontier.popleft()
        if length == upto:
            continue
        for neighbour, shift in steps[node]:
            moved = tuple(a + b for a, b in zip(accumulated, shift))
            # The goal is tested here rather than on pop.  The start state is
            # marked seen before the search begins, so a walk *returning* to it
            # -- which is what a closed walk carrying the trivial class is --
            # would never be popped a second time.
            if neighbour == start and moved == wanted:
                return length + 1
            state = (neighbour, moved)
            if state in seen:
                continue
            seen.add(state)
            frontier.append((neighbour, moved, length + 1))
    return None


def support_order_is_at_least_walk_length(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    connection: IntegerConnection,
    target: Sequence[int],
    upto: int,
    rank: int,
) -> bool:
    """The inequality, which always holds: no class before a walk carries it.

    A moment of order ``n`` is a sum over closed ``n``-walks, so a class absent
    from every ``n``-walk is absent from ``M_n``.  ``None`` on the expansion side
    counts as infinite; ``None`` on the walk side would make the class
    unreachable and is treated as vacuously satisfied.
    """
    order = first_order_carrying(simplices, simplex, connection, target, upto, rank)
    walk = shortest_walk_carrying(simplices, simplex, connection, target, upto, rank)
    if walk is None:
        return order is None
    return order is None or order >= walk


def support_law_holds(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    connection: IntegerConnection,
    target: Sequence[int],
    upto: int,
    rank: int,
) -> bool:
    """Equality: expansion order equals shortest-walk length.

    True at most simplices and **false at some**, because the moment is a
    *signed* count: walks of the same length carrying the same class can cancel.
    In the running example the shared edge ``(0, 1)`` is the counterexample --
    walks reaching each class exist at length ``6``, and every one of them
    cancels, at every order, forever.  See `is_flux_blind`.

    Stated as a law it would be wrong.  It is the inequality that is a law; this
    predicate reports whether the slack is zero.
    """
    return first_order_carrying(
        simplices, simplex, connection, target, upto, rank
    ) == shortest_walk_carrying(simplices, simplex, connection, target, upto, rank)


def is_flux_blind(
    simplices: Sequence[tuple[int, ...]],
    simplex: tuple[int, ...],
    connection: IntegerConnection,
    rank: int,
    upto: int = 10,
) -> bool:
    """Does this simplex's local measure ignore the connection entirely?

    True when every expansion up to ``upto`` is a single constant term, so the
    moments are the same for every connection.  Such simplices exist: in the
    running example the shared edge ``(0, 1)`` has moments ``4, 16, 64, 256,
    1024`` -- exactly ``4^{n/2}``, the two-point measure at ``+-2`` -- whatever
    the field does, which says ``e_tau`` is an eigenvector of the magnetic Hodge
    Laplacian at its Hasse degree for every connection.

    **This is a property of the ordering, not of the complex.**  Relabel the same
    two triangles so the shared edge is ``(2, 3)`` and it is no longer blind,
    while two other edges become so.  See `relabelling_changes_the_spectrum`.
    """
    for order in range(2, upto + 1, 2):
        expansion = character_expansion(simplices, simplex, connection, order, rank)
        if set(expansion) - {_zero(rank)}:
            return False
    return True


# ---------------------------------------------------------------------------
# the basepoint, and what depends on it
# ---------------------------------------------------------------------------

#: `magnetic.general_dirac` puts the whole parallel transport of a simplex onto
#: its leading edge, which is the same as choosing each simplex's **minimal
#: vertex** as a basepoint: the entry from ``sigma`` to the face missing
#: ``sigma[0]`` carries ``exp(i A(sigma[0] -> sigma[1]))`` and every other entry
#: carries ``1``.  Any basepoint choice gives a gauge-covariant operator; the
#: minimal-vertex choice is the one this repository has been using throughout.
BASEPOINT_IS_THE_MINIMAL_VERTEX: bool = True


def relabel_complex(
    simplices: Sequence[tuple[int, ...]], permutation: dict[int, int]
) -> tuple[tuple[int, ...], ...]:
    """Apply a vertex permutation and re-sort."""
    relabelled = [
        tuple(sorted(permutation[vertex] for vertex in simplex))
        for simplex in simplices
    ]
    return tuple(sorted(relabelled, key=lambda s: (len(s), s)))


def relabel_angles(
    angles: dict[tuple[int, int], float], permutation: dict[int, int]
) -> dict[tuple[int, int], float]:
    """Transport a connection along a vertex permutation.

    Angles are stored on sorted edges and mean the phase in the increasing
    direction, so an edge whose image reverses orientation must have its angle
    negated.  Forgetting that negation makes the relabelled connection a
    different connection, and then every comparison below is meaningless.
    """
    moved: dict[tuple[int, int], float] = {}
    for (tail, head), angle in angles.items():
        image_tail, image_head = permutation[tail], permutation[head]
        key = (min(image_tail, image_head), max(image_tail, image_head))
        moved[key] = angle if image_tail < image_head else -angle
    return moved


def relabelling_changes_the_spectrum(
    simplices: Sequence[tuple[int, ...]],
    angles: dict[tuple[int, int], float],
    permutation: dict[int, int],
    tolerance: float = 1e-8,
) -> bool:
    """Do the two labellings of one complex give different Dirac spectra?

    They should not, and they do -- unless the connection is flat.  The reason is
    the basepoint: relabelling moves each simplex's minimal vertex, so the
    implicit basepoint moves, and re-basing means transporting inside the
    simplex.  That transport is path independent exactly when the simplex has no
    curvature, so:

        flat connection      -> spectra agree, the operator is an invariant of
                                the abstract complex;
        curved connection    -> spectra differ, and the operator is an invariant
                                of the **ordered** complex only.

    Verified both ways on two triangles glued along an edge.  This is a scope
    statement for everything in `insertion`, `rigidity` and this module: those
    results are theorems about a labelled complex.  What survives relabelling is
    the *character* data -- the recovered cosines and the Fricke identity are
    identical in both labellings, because they are functions of loop holonomies
    and loop holonomies do not know about vertex names.
    """
    first = np.linalg.eigvalsh(
        magnetic.general_dirac(
            sorted(simplices, key=lambda s: (len(s), s)),
            insertion.phase_function(angles),
        )
    )
    second = np.linalg.eigvalsh(
        magnetic.general_dirac(
            relabel_complex(simplices, permutation),
            insertion.phase_function(relabel_angles(angles, permutation)),
        )
    )
    return bool(np.abs(np.sort(first) - np.sort(second)).max() > tolerance)


def pure_gauge(
    vertices: int, potentials: Sequence[float]
) -> dict[tuple[int, int], float]:
    """A flat connection: ``A(u -> v) = lambda(v) - lambda(u)``, every holonomy one."""
    if len(potentials) != vertices:
        raise ValueError(
            f"need one potential per vertex: got {len(potentials)} for {vertices}"
        )
    return {
        (tail, head): float(potentials[head] - potentials[tail])
        for tail, head in combinations(range(vertices), 2)
    }


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------


def recover_cosines(
    simplices: Sequence[tuple[int, ...]],
    connection: IntegerConnection,
    thetas: Sequence[float],
    upto: int = 8,
    rank: int | None = None,
) -> dict[tuple[int, ...], float]:
    """Recover ``cos(a . theta)`` for every class the moments reach, by peeling.

    The expansions are a property of the *complex*; the moments are what a
    measurement returns.  Walk up in order, and at each simplex subtract the
    contributions of the classes already recovered.  If exactly one new class
    remains with a non-zero coefficient, its cosine follows by division.  That
    triangularity is not an accident: `first_order_carrying` says classes enter
    in order of walk length, so each new order introduces the classes it is long
    enough to enclose and no others.

    Nothing here uses ``thetas`` except to evaluate the moments, which is what a
    measurement would supply; the recovered values are then compared against the
    truth in the tests rather than being read from it.
    """
    width = rank if rank is not None else len(thetas)
    _validate(connection, width)
    if len(thetas) != width:
        raise ValueError(f"thetas has length {len(thetas)}, expected rank {width}")
    if upto < 0:
        raise ValueError(f"upto must be non-negative, got {upto}")
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    weight = insertion.phase_function(angles_from(connection, thetas))
    known: dict[tuple[int, ...], float] = {}
    for order in range(2, upto + 1, 2):
        for simplex in ordered:
            expansion = character_expansion(
                ordered, simplex, connection, order, width
            )
            folded: dict[tuple[int, ...], int] = {}
            for vector, coefficient in expansion.items():
                folded[canonical_class(vector)] = coefficient
            constant = float(folded.pop(_zero(width), 0))
            unknown = [
                class_
                for class_, coefficient in folded.items()
                if class_ not in known and coefficient != 0
            ]
            if len(unknown) != 1:
                continue
            target = unknown[0]
            measured = insertion.insertion_moment(ordered, simplex, weight, order)
            residual = measured - constant
            for class_, coefficient in folded.items():
                if class_ != target and class_ in known:
                    residual -= 2 * coefficient * known[class_]
            known[target] = residual / (2 * folded[target])
    return known


# ---------------------------------------------------------------------------
# the Fricke cubic
# ---------------------------------------------------------------------------


def fricke_residual(first: float, second: float, composite: float) -> float:
    """``|u^2 + v^2 + w^2 - 2uvw - 1|`` for the cosines of ``a``, ``b``, ``a - b``.

    Zero identically.  With ``u = cos A``, ``v = cos B``, ``w = cos(A - B)``,

        w - uv = sin A sin B,   so   (w - uv)^2 = (1 - u^2)(1 - v^2),

    and expanding gives the cubic.  Classical -- this is the Fricke relation,
    and the surface it cuts out in ``R^3`` is the Cayley cubic.  What is being
    asserted here is that the three numbers come out of *measured local spectral
    moments*, so a magnetic Dirac operator on any complex produces points of a
    fixed cubic surface and nothing else.
    """
    return abs(
        first * first
        + second * second
        + composite * composite
        - 2 * first * second * composite
        - 1.0
    )


def fricke_identity_holds(
    first: float, second: float, composite: float, tolerance: float = TOLERANCE
) -> bool:
    """Does the triple lie on the Cayley cubic?"""
    return fricke_residual(first, second, composite) <= tolerance


def fricke_coordinates(
    simplices: Sequence[tuple[int, ...]] = TWO_TRIANGLES,
    connection: IntegerConnection | None = None,
    thetas: Sequence[float] = (0.7, 1.3),
    upto: int = 8,
) -> tuple[float, float, float]:
    """``(u, v, w)`` for the two plaquettes and their difference, from moments only.

    The recovery is `recover_cosines`; this only names the three classes that
    make the cubic.  Defaults are the running example, whose two plaquettes are
    the classes ``(1, 0)`` and ``(0, 1)`` with difference ``(1, -1)``.
    """
    live = TWO_TRIANGLE_CONNECTION if connection is None else connection
    recovered = recover_cosines(simplices, live, thetas, upto)
    missing = [
        class_
        for class_ in ((1, 0), (0, 1), (1, -1))
        if class_ not in recovered
    ]
    if missing:
        raise ValueError(
            f"moments up to order {upto} do not reach the classes {missing}; "
            "raise upto or use a complex whose loops are shorter"
        )
    return (recovered[(1, 0)], recovered[(0, 1)], recovered[(1, -1)])


#: The four nodes of the Cayley cubic: the sign patterns with product ``+1``.
#: Solving ``grad(u^2+v^2+w^2-2uvw) = 0`` gives ``u = vw``, ``v = uw``,
#: ``w = uv``, hence entries ``+-1`` with ``uvw = 1``, and the cubic itself then
#: reads ``3 - 2 = 1``.  These are the images of the four two-torsion points of
#: the torus -- the connections whose holonomies are all real -- and they are
#: exactly the degeneracy locus measured in `rigidity`.
CAYLEY_CUBIC_NODES: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, -1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
)


def is_cayley_node(
    first: float, second: float, composite: float, tolerance: float = TOLERANCE
) -> bool:
    """Is this triple one of the four nodes, i.e. a point where the cover collapses?"""
    return any(
        abs(first - node[0]) <= tolerance
        and abs(second - node[1]) <= tolerance
        and abs(composite - node[2]) <= tolerance
        for node in CAYLEY_CUBIC_NODES
    )


def node_count(rank: int) -> int:
    """``2^rank``: the two-torsion of ``T^rank``, hence the branch points.

    For ``rank = 2`` this is four, and four is the number of nodes on the Cayley
    cubic and the number of corners of a pillowcase.  The coincidence is not
    one.
    """
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    return 2**rank


# ---------------------------------------------------------------------------
# the count
# ---------------------------------------------------------------------------


#: The order at which the moments first couple the signs of two plaquettes on a
#: fan of triangles: the difference class needs a walk around two plaquettes, and
#: on triangles that walk has length eight.  Below this order the signs are
#: independent and the ambiguity is ``(Z/2)^rank``; at or above it they move
#: together and the ambiguity is a single ``Z/2``.
COUPLING_ORDER: int = 8


def moments_couple_the_signs(order: int) -> bool:
    """Is this truncation deep enough for the rigidity theorem to hold?"""
    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}")
    return order >= COUPLING_ORDER


#: A ``U(1)`` holonomy ``exp(i theta)`` enters the moments only through
#: ``2 cos(theta)``, which is the trace of ``diag(exp(i theta), exp(-i theta))``
#: in ``SU(2)``.  The moments are therefore Weyl-invariant functions on a maximal
#: torus whether or not anyone intended it, and the rigidity ambiguity is the
#: Weyl group ``W = N(T)/T = Z/2``.
WEYL_GROUP_ORDER: int = 2


def trace_coordinate(cosine: float) -> float:
    """``2 cos(theta)``: the ``SU(2)`` trace whose Weyl-invariance moments inherit."""
    return 2.0 * cosine


def standard_cayley_residual(first: float, second: float, composite: float) -> float:
    """The cubic in the usual ``SU(2)`` normalisation, from the same three cosines.

    `fricke_residual` measures ``u^2+v^2+w^2-2uvw-1`` in cosines; substituting
    ``X = 2u`` and friends turns it into ``X^2+Y^2+Z^2-XYZ-4``, the ``kappa = 2``
    level set of the commutator trace and the standard equation of the Cayley
    cubic.  The two differ by a factor of four and nothing else, which is the
    arithmetic behind the framing correction in the module docstring.
    """
    one, two, three = (
        trace_coordinate(first),
        trace_coordinate(second),
        trace_coordinate(composite),
    )
    return abs(one * one + two * two + three * three - one * two * three - 4.0)


def global_spectrum(
    simplices: Sequence[tuple[int, ...]],
    connection: IntegerConnection,
    thetas: Sequence[float],
) -> np.ndarray:
    """Eigenvalues of the whole Dirac operator: the *sum* of the local measures.

    Kept beside the local machinery because the comparison between the two is the
    answer to the published obstruction -- the spectrum does not determine the
    magnetic potential, and the local measures do.
    """
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    weight = insertion.phase_function(angles_from(connection, thetas))
    return np.sort(np.linalg.eigvalsh(magnetic.general_dirac(ordered, weight)))


def spectra_agree(
    first: np.ndarray, second: np.ndarray, tolerance: float = 1e-9
) -> bool:
    """Are two connections isospectral?"""
    if first.shape != second.shape:
        raise ValueError(
            f"spectra have different lengths {first.shape} and {second.shape}"
        )
    return bool(np.abs(first - second).max() <= tolerance)


#: On two triangles glued along an edge, the whole line ``theta_1 + theta_2 =
#: pi`` is isospectral: the global Dirac spectrum is constant along it to
#: ``2e-15`` while the local moments vary continuously.  The condition says the
#: outer four-cycle carries holonomy ``-1``.  **Why half-flux produces an
#: isospectral family is not explained here** -- the fact is measured and the
#: mechanism is open.
HALF_FLUX_IS_ISOSPECTRAL: bool = True


def half_flux_curve(parameter: float) -> tuple[float, float]:
    """A point of the isospectral curve ``theta_1 + theta_2 = pi``."""
    return (float(parameter), float(np.pi - parameter))


def local_data_beats_the_spectrum(
    resolution: int = 12,
    simplices: Sequence[tuple[int, ...]] | None = None,
    connection: IntegerConnection | None = None,
    order: int = COUPLING_ORDER,
    digits: int = 7,
) -> tuple[int, int]:
    """``(spectrum classes, signature classes)`` on a torsion grid.

    The measured answer to the isospectrality obstruction.  Isospectral magnetic
    graphs are published and real (Fabila-Carrasco, Lledo and Post, Anal. Math.
    Phys. 13:64, 2023), and if the spectrum were all one had, the inverse problem
    would be dead.  It is not: the signature is one measure *per simplex* and the
    spectrum is their sum, so the spectrum fuses what the signature separates.

    At ``resolution = 12`` on the running example this returns ``(40, 74)``, and
    ``74`` is exactly `distinguishable_signature_count` -- the local data
    separates every orbit while the spectrum loses a third of them.
    """
    if resolution < 1:
        raise ValueError(f"resolution must be positive, got {resolution}")
    live_simplices = TWO_TRIANGLES if simplices is None else simplices
    live = TWO_TRIANGLE_CONNECTION if connection is None else connection
    ordered = sorted(live_simplices, key=lambda s: (len(s), s))
    step = 2 * np.pi / resolution
    spectra: set[tuple[float, ...]] = set()
    signatures: set[tuple[float, ...]] = set()
    for first in range(resolution):
        for second in range(resolution):
            thetas = (step * first, step * second)
            spectra.add(
                tuple(
                    round(value, digits)
                    for value in global_spectrum(ordered, live, thetas)
                )
            )
            weight = insertion.phase_function(angles_from(live, thetas))
            signatures.add(
                tuple(
                    round(
                        insertion.insertion_moment(ordered, simplex, weight, moment),
                        digits,
                    )
                    for simplex in ordered
                    for moment in range(order + 1)
                )
            )
    return (len(spectra), len(signatures))


def trace_expansion(
    simplices: Sequence[tuple[int, ...]],
    connection: IntegerConnection,
    order: int,
    rank: int,
) -> dict[tuple[int, ...], int]:
    """``tr(D^order)`` as an exact integer character expansion.

    The bridge between the local and the global.  ``tr(D^k) = sum_tau M_k(tau)``,
    so the *global* spectrum is nothing but the sum of the local measures -- and
    the power sums determine the characteristic polynomial.  Summing the local
    expansions therefore gives the global invariant in the same coordinates,
    exactly, with no floating point anywhere.
    """
    total: dict[tuple[int, ...], int] = {}
    for simplex in sorted(simplices, key=lambda s: (len(s), s)):
        for vector, coefficient in character_expansion(
            simplices, simplex, connection, order, rank
        ).items():
            _add(total, vector, coefficient)
    return total


def trace_restricted_to_line(
    expansion: dict[tuple[int, ...], int], shift: float
) -> dict[int, complex]:
    """Restrict a rank-two expansion to the line ``theta_1 + theta_2 = shift``.

    On that line ``theta_2 = shift - theta_1``, so the class ``a`` contributes at
    frequency ``a_1 - a_2`` with phase ``exp(i a_2 shift)``.  The result is keyed
    by frequency; a support of ``{0: c}`` alone means the trace is *constant*
    along the line, which is isospectrality at that order.
    """
    collected: dict[int, complex] = {}
    for vector, coefficient in expansion.items():
        if len(vector) != 2:
            raise ValueError(f"class {vector} is not rank two")
        frequency = vector[0] - vector[1]
        collected[frequency] = collected.get(frequency, 0j) + coefficient * cmath.exp(
            1j * vector[1] * shift
        )
    return {
        frequency: value
        for frequency, value in collected.items()
        if abs(value) > TOLERANCE
    }


def trace_is_constant_on_line(
    shift: float,
    upto: int = 10,
    simplices: Sequence[tuple[int, ...]] | None = None,
    connection: IntegerConnection | None = None,
) -> bool:
    """Is every power sum constant along ``theta_1 + theta_2 = shift``?

    If so the whole characteristic polynomial is constant there, and the line is
    an **isospectral family**.  This is the derivation of the half-flux curve, and
    it is done in exact integer arithmetic rather than by comparing eigenvalues.
    """
    live_simplices = TWO_TRIANGLES if simplices is None else simplices
    live = TWO_TRIANGLE_CONNECTION if connection is None else connection
    for order in range(2, upto + 1, 2):
        restricted = trace_restricted_to_line(
            trace_expansion(live_simplices, live, order, 2), shift
        )
        if set(restricted) - {0}:
            return False
    return True


def resolving_simplex(
    simplices: Sequence[tuple[int, ...]],
    connection: IntegerConnection,
    target: Sequence[int],
    rank: int,
    upto: int = COUPLING_ORDER,
) -> tuple[tuple[int, ...], int] | None:
    """A simplex and order where the class ``target`` appears with non-zero weight.

    ``None`` when no simplex resolves it within ``upto`` -- which is a real
    possibility and not a formality, since the shared edge of two triangles
    resolves nothing at any order.  What matters for the theorem below is whether
    *some* simplex resolves each needed class, not whether every one does.
    """
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    wanted = tuple(int(component) for component in target)
    for order in range(2, upto + 1, 2):
        for simplex in ordered:
            expansion = character_expansion(
                ordered, simplex, connection, order, rank
            )
            if expansion.get(wanted, 0) != 0:
                return (simplex, order)
    return None


def classes_are_resolved(
    simplices: Sequence[tuple[int, ...]],
    connection: IntegerConnection,
    rank: int,
    upto: int = COUPLING_ORDER,
) -> bool:
    """The hypothesis of the general-rank theorem, checked for one complex.

    The rank-two argument needs three things from the moments: ``cos(theta_j)``
    for each coordinate, to pin each angle up to sign, and ``cos(theta_j -
    theta_k)`` for each pair, to pin the relative signs.  At general rank it
    needs exactly the same list -- a basis and its pairwise differences -- so:

        **Theorem.**  If every standard basis class and every pairwise difference
        appears with non-zero coefficient at some simplex and some order, then
        the spectral signature determines the connection modulo gauge and one
        global reflection, and the ambiguity group is ``Z/2``.

    The proof is the rank-two one verbatim: the basis cosines pin each angle to
    ``+-theta_j``, and ``cos(theta_j - theta_k) = cos theta_j cos theta_k +
    sin theta_j sin theta_k`` then forces the signs to move together, since
    flipping one alone would negate ``sin theta_j sin theta_k``.

    What is *not* automatic is the hypothesis.  Every class is carried by some
    closed Hasse walk on a connected complex, but the moment is a **signed**
    count and the walks can cancel -- `is_flux_blind` exhibits a simplex where
    they all do, at every order.  So the hypothesis is a genuine condition, and
    this function is how one discharges it for a given complex rather than
    assuming it.  Measured: it holds for fans of one, two and three plaquettes,
    which is why the rank-three count comes out at ``Z/2``.
    """
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    needed = [
        tuple(1 if index == axis else 0 for index in range(rank))
        for axis in range(rank)
    ]
    for first in range(rank):
        for second in range(first + 1, rank):
            needed.append(
                tuple(
                    (1 if index == first else 0) - (1 if index == second else 0)
                    for index in range(rank)
                )
            )
    return all(
        resolving_simplex(simplices, connection, target, rank, upto) is not None
        for target in needed
    )


def truncated_ambiguity_order(order: int, rank: int = 2) -> int:
    """Size of the ambiguity group at truncation ``order``.

    ``2^rank`` below `COUPLING_ORDER`, ``2`` at or above it.  The drop is the
    whole reason the rigidity theorem needs order eight: truncate at six and the
    two plaquette signs are genuinely independent, so a sweep returns **four**
    connections sharing a signature rather than two.  Measured, on a ``36 x 36``
    grid: four at orders four and six, two at order eight.
    """
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    return 2 if moments_couple_the_signs(order) else 2**rank


def distinguishable_signature_count(
    resolution: int, rank: int = 2, order: int = COUPLING_ORDER
) -> int:
    """How many distinct spectral signatures a torsion grid of connections carries.

    Burnside applied to the ambiguity group acting on ``(Z/N)^rank``, and *which*
    group that is depends on how deep the signature is truncated.  Write
    ``g = gcd(2, N)`` for the two-torsion of ``Z/N``.

    **At or above `COUPLING_ORDER`** the group is the diagonal ``Z/2``, global
    inversion.  The identity fixes ``N^rank`` points, the inversion fixes the
    two-torsion, so

        orbits  =  ( N^rank + g^rank ) / 2.

    **Below it** the moments have not yet produced a term joining two classes, so
    each sign flips independently and the group is the full ``(Z/2)^rank``.  Each
    of its ``2^rank`` elements fixes ``g^k N^{rank-k}`` points for ``k`` flipped
    coordinates, and the binomial sum collapses:

        orbits  =  ( (N + g) / 2 )^rank.

    Both are counts of *signatures* only because `rigidity` says the signature
    separates orbits, so both are predictions with content -- a failure to
    separate would make the measured count come in lower.  Neither does.  The gap
    between the two is what makes `rigidity.SIGNATURE_ORDER` load-bearing: at
    ``N = 8, rank = 2`` the truncated count is ``25`` and the full one is ``34``,
    and a sweep at order six really does return four matches where a sweep at
    order eight returns two.
    """
    if resolution < 1:
        raise ValueError(f"resolution must be positive, got {resolution}")
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    torsion = 2 if resolution % 2 == 0 else 1
    if moments_couple_the_signs(order):
        return (resolution**rank + torsion**rank) // 2
    return ((resolution + torsion) // 2) ** rank


def plaquette_fan(count: int) -> tuple[tuple[tuple[int, ...], ...], IntegerConnection]:
    """``count`` triangles sharing the edge ``(0, 1)``, gauge-fixed to rank ``count``.

    The rank-two case is the running example; higher counts are how the rigidity
    theorem gets tested beyond the rank it was proved at.  Apex ``k`` is vertex
    ``k + 2``, and the class of its plaquette is the ``k``-th standard basis
    vector.
    """
    if count < 1:
        raise ValueError(f"need at least one plaquette, got {count}")
    simplices = insertion.close_under_faces(
        [(0, 1, apex + 2) for apex in range(count)]
    )
    connection: IntegerConnection = {
        (1, apex + 2): tuple(1 if index == apex else 0 for index in range(count))
        for apex in range(count)
    }
    return simplices, connection


def measured_signature_count(
    resolution: int = 12,
    simplices: Sequence[tuple[int, ...]] | None = None,
    connection: IntegerConnection | None = None,
    order: int = 8,
    digits: int = 7,
    rank: int = 2,
) -> int:
    """Brute force: compute every signature on the grid and count distinct ones.

    Rounded to ``digits`` before hashing.  The margin is wide -- distinct
    signatures differ by order one, and equal ones agree to machine precision --
    so the rounding is bookkeeping rather than a threshold to tune.

    Defaults to the running example at rank two.  Pass a `plaquette_fan` and its
    rank to test the count where the rigidity theorem was never proved.
    """
    if resolution < 1:
        raise ValueError(f"resolution must be positive, got {resolution}")
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    live_simplices = TWO_TRIANGLES if simplices is None else simplices
    live = TWO_TRIANGLE_CONNECTION if connection is None else connection
    ordered = sorted(live_simplices, key=lambda s: (len(s), s))
    step = 2 * np.pi / resolution
    seen: set[tuple[float, ...]] = set()
    for flat in range(resolution**rank):
        cell = np.unravel_index(flat, (resolution,) * rank)
        weight = insertion.phase_function(
            angles_from(live, [step * int(component) for component in cell])
        )
        seen.add(
            tuple(
                round(
                    insertion.insertion_moment(ordered, simplex, weight, moment),
                    digits,
                )
                for simplex in ordered
                for moment in range(order + 1)
            )
        )
    return len(seen)
