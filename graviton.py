"""The second-order obstruction is the Friedmann constraint, sourced by gravitons.

What is derived here
--------------------
:mod:`adm` computes, as exact rational linear algebra, the second-order
obstruction that a solution of the linearised Einstein constraints on a compact
flat slice must satisfy in order to be tangent to a curve of genuine solutions.
For the constant-lapse Killing initial datum it is the torus average

    Q  =  < R^(2)(h) >  +  < (tr K)^2 - |K|^2 >  =  0 .

That is a quadratic form on Fourier amplitudes and carries no physics on its
face.  This module identifies it, term by term, and the identification is the
result.

**Step one: the inhomogeneous modes are gravitons.**  Restricted to the
transverse-traceless sector at a mode ``k != 0``, with ``h = A cos(k.x)`` and
``K = B cos(k.x)``,

    Q_k  =  -(1/8) |k|^2 |A|^2  -  (1/2) |B|^2 .

ADM evolution at unit lapse and zero shift gives ``hdot = 2K``, so ``B = Adot/2``
and

    -Q_k  =  (1/8) ( Adot^2 + |k|^2 A^2 ) ,

a harmonic oscillator of frequency ``omega`` with ``omega^2 = |k|^2``.  The
massless dispersion is an *output* of the constraint algebra, not an input:
:func:`dispersion` extracts it from the exact form and returns a rational number,
and it equals ``|k|^2`` at every mode and both polarisations.

**Step two: the coefficient is Isaacson's.**  The quantity ``-Q_k`` is not merely
proportional to the gravitational wave energy, it is ``16 pi`` times the Isaacson
effective energy density

    rho  =  (1/(64 pi)) < hdot_ij hdot^ij  +  d_l h_ij d^l h^ij >

exactly.  :func:`isaacson_density` computes ``rho`` and
:data:`ISAACSON_TO_OBSTRUCTION` is the ratio, which is an integer.  Nothing in
:mod:`adm` knows about ``1/(32 pi)``; the factor arrives from the second-order
expansion of the scalar curvature.

**Step three: the homogeneous mode pays for it.**  The obstruction is *not*
negative definite overall.  At ``k = 0`` the metric term vanishes identically and
the momentum term survives with inertia ``(5, 0, 1)`` -- exactly one positive
direction, the isotropic one.  Writing the homogeneous momentum as
``B_0 = -H delta + sigma`` with ``sigma`` trace-free,

    Q_0  =  6 H^2  -  |sigma|^2 .

Summing over all modes, ``Q = 0`` reads

    6 H^2  -  |sigma|^2  =  sum_{k != 0} ( -Q_k )  =  16 pi rho ,

which is the Friedmann constraint with shear,

    3 H^2  =  8 pi rho  +  (1/2) sigma_ij sigma^ij ,

with the gravitational wave energy as its source.  Every coefficient in it --
the ``6``, the ``16 pi``, the ``1/2`` on the shear -- is computed output.
:func:`balance` assembles it and :func:`friedmann_residual` checks it as an exact
rational identity.

So the linearisation-stability condition on a compact flat slice is not a
prohibition on gravitational waves.  It is the statement that their energy must
be paid for by a homogeneous expansion at precisely the Friedmann rate.  A
linearised solution with waves and no expansion does not integrate; one with the
matching expansion does, at this order.

What this corrects
------------------
Read per-mode, ``Q_k < 0`` at every ``k != 0`` and the natural conclusion is that
the stability condition kills the graviton sector outright.  That conclusion is
wrong, and the reason is a factor of two: the averaging weight is
``<cos^2(k.x)> = 1/2`` away from the origin but ``<1> = 1`` at it.  The
distinction is invisible to any statement about one mode -- a positive rescaling
moves no sign and no signature -- and it is exactly what sets the relative weight
once homogeneous and inhomogeneous contributions are added against each other.
:func:`adm.obstruction_value` now carries that weight; before this module it did
not, and the per-mode results in :mod:`adm` and :mod:`platycosm` are unaffected
because they never added across the origin.

The point about the platycosms
------------------------------
The positive direction that pays for the waves is the isotropic one, and
``delta`` is invariant under every holonomy group, so the balance survives on
every compact flat manifold.  What the holonomy cuts is the *shear* budget: the
trace-free homogeneous momenta invariant under the point group number 5 on the
three-torus, 3 on the dicosm, 1 on the tetracosm and 2 on the Hantzsche--Wendt
manifold (:func:`shear_budget`).  Shear enters the balance with the opposite
sign to expansion, so a manifold with less holonomy has more ways to fail the
condition, not more ways to satisfy it.

The Hantzsche--Wendt manifold sharpens this further.  By :mod:`platycosm` it has
no Killing vector fields at all, so the constant lapse is its only Killing
initial datum and the Friedmann balance is its *only* second-order condition.
On the three-torus there are three more, from the translational Killing fields,
and they constrain the wave momenta rather than the energy.  The balance is
therefore the irreducible content of linearisation stability on a compact flat
slice: what survives when every symmetry that could be averaged over is gone.

A remark on the quantum condition, not a derivation
---------------------------------------------------
Moncrief's proposal is to impose the stability condition as an operator equation
on physical states.  Read per-mode it says ``Hhat_graviton |psi> = 0``, which
before normal ordering no state satisfies and after normal ordering only the Fock
vacuum does.  Read globally it says instead

    6 Hhat^2  -  |sigma hat|^2  =  Hhat_graviton ,

relating the graviton number operator to the momenta conjugate to the flat
moduli, whose spectrum is continuous and unbounded above.  Every graviton state
then has a partner expansion rate rather than being excluded, and the condition
becomes a normalisation on the homogeneous sector.  The one thing it does not
survive is the zero-point sum: on the un-normal-ordered vacuum the right-hand
side diverges and no finite expansion rate pays for it, so an ordering
prescription is doing real work here rather than being a convention.  This
paragraph is a reading of the classical identity above, not a result computed by
this module.

Prior work
----------
None of the physical ingredients are new.  The linearisation-stability framework
is Fischer--Marsden and Moncrief; that the absence of Killing fields is necessary
for stability is Arms--Marsden; the effective energy density of a short-wave
gravitational field is Isaacson's; second-order back-reaction of gravitational
waves on a homogeneous background is standard cosmological perturbation theory.
What is done here is to obtain the identity between them as exact computed
output of one constraint calculation, with no coefficient supplied by hand, and
to push it onto the flat manifolds with holonomy where the shear budget shrinks
and the symmetry conditions disappear.

Scope
-----
Vacuum, one flat compact slice, second order.  Amplitudes are real, one
representative per ``+-k`` pair.  The gravitons are treated in the
transverse-traceless sector, which is where :func:`dispersion` and
:func:`mode_energy` are defined; :func:`balance` accepts any amplitudes and
reports the exact obstruction regardless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

import adm
import platycosm

__all__ = [
    "KINETIC_TO_VELOCITY",
    "ISAACSON_TO_OBSTRUCTION",
    "kinetic_coefficient",
    "potential_coefficient",
    "dispersion",
    "DispersionCheck",
    "check_dispersion",
    "mode_energy",
    "isaacson_density",
    "Homogeneous",
    "split_homogeneous",
    "BalanceReport",
    "balance",
    "friedmann_residual",
    "required_hubble_squared",
    "shear_budget",
]

LOGGER = logging.getLogger(__name__)

#: ADM at unit lapse and zero shift gives ``hdot_ij = 2 K_ij``, so the momentum
#: amplitude is half the velocity amplitude and the kinetic coefficient picks up
#: a factor ``(1/2)^2``.  Everything hinges on this factor: without it the
#: extracted frequency comes out as ``|k|/2`` and the derivation fails.
KINETIC_TO_VELOCITY = Fraction(1, 4)

#: ``-Q_k = ISAACSON_TO_OBSTRUCTION * pi * rho`` on the transverse-traceless
#: sector, with ``rho`` the Isaacson effective energy density.  An integer, and
#: the reason the balance below comes out as the Friedmann constraint with its
#: standard normalisation rather than something proportional to it.
ISAACSON_TO_OBSTRUCTION = 16

_ZERO_WAVE = (0, 0, 0)


def _key(wave: Sequence[int]) -> tuple[int, int, int]:
    if len(wave) != 3:
        raise ValueError(f"a wave vector needs three components; got {len(wave)}")
    return (int(wave[0]), int(wave[1]), int(wave[2]))


def _slots(tensor: Sequence[Fraction | int]) -> list[Fraction]:
    if len(tensor) != 6:
        raise ValueError(f"a symmetric tensor needs six slots; got {len(tensor)}")
    return [Fraction(entry) for entry in tensor]


def _zero() -> list[Fraction]:
    return [Fraction(0)] * 6


def _tensor_norm(tensor: Sequence[Fraction | int]) -> Fraction:
    """``t_ij t^ij``, with off-diagonal slots counted twice."""
    values = _slots(tensor)
    return sum(
        adm._multiplicity(slot) * values[slot] * values[slot] for slot in range(6)
    )


# --------------------------------------------------------------------------- #
# Step one: the inhomogeneous modes are oscillators of frequency |k|
# --------------------------------------------------------------------------- #
def kinetic_coefficient(
    wave: Sequence[int], tensor: Sequence[Fraction | int]
) -> Fraction:
    """Coefficient of the momentum square in ``-Q``: the obstruction at ``h = 0``."""
    key = _key(wave)
    return -adm.obstruction_value({key: _slots(tensor)}, {key: _zero()})


def potential_coefficient(
    wave: Sequence[int], tensor: Sequence[Fraction | int]
) -> Fraction:
    """Coefficient of the gradient square in ``-Q``: the obstruction at ``K = 0``."""
    key = _key(wave)
    return -adm.obstruction_value({key: _zero()}, {key: _slots(tensor)})


def dispersion(wave: Sequence[int], tensor: Sequence[Fraction | int]) -> Fraction:
    """``omega^2`` read off the obstruction form, as an exact rational.

    Writing ``-Q = kin * B^2 + pot * A^2`` and substituting ``B = Adot/2`` gives
    ``-Q = (kin/4) Adot^2 + pot A^2``, an oscillator with
    ``omega^2 = pot / (kin/4)``.  Nothing about wave propagation is assumed; the
    frequency is extracted from the constraint algebra.
    """
    kinetic = kinetic_coefficient(wave, tensor)
    if kinetic == 0:
        raise ValueError(
            f"mode {_key(wave)} has no kinetic term, so no frequency is defined"
        )
    return potential_coefficient(wave, tensor) / (kinetic * KINETIC_TO_VELOCITY)


@dataclass(frozen=True)
class DispersionCheck:
    """``omega^2`` against ``|k|^2`` at one mode and polarisation."""

    wave: tuple[int, int, int]
    omega_squared: Fraction
    wave_squared: int
    kinetic: Fraction
    potential: Fraction

    @property
    def massless(self) -> bool:
        """Whether the extracted frequency is exactly the massless one."""
        return self.omega_squared == self.wave_squared

    @property
    def energy_positive(self) -> bool:
        """Both oscillator coefficients positive, so ``-Q`` is a positive form."""
        return self.kinetic > 0 and self.potential > 0


def check_dispersion(wave: Sequence[int]) -> list[DispersionCheck]:
    """Extract ``omega^2`` for both transverse-traceless polarisations at ``k``."""
    key = _key(wave)
    if key == _ZERO_WAVE:
        raise ValueError("the homogeneous mode carries expansion, not gravitons")
    checks = []
    for tensor in adm.transverse_traceless_modes(key):
        checks.append(
            DispersionCheck(
                wave=key,
                omega_squared=dispersion(key, tensor),
                wave_squared=sum(component * component for component in key),
                kinetic=kinetic_coefficient(key, tensor),
                potential=potential_coefficient(key, tensor),
            )
        )
    return checks


def mode_energy(
    wave: Sequence[int],
    amplitude: Sequence[Fraction | int],
    velocity: Sequence[Fraction | int],
) -> Fraction:
    """``-Q_k`` for ``h = A cos(k.x)`` with ``hdot = Adot cos(k.x)``.

    The velocity, not the momentum, is the natural argument once the form is read
    as an energy; ``K = hdot / 2`` is applied here so that callers never have to
    remember the factor.
    """
    key = _key(wave)
    if key == _ZERO_WAVE:
        raise ValueError("the homogeneous mode carries expansion, not gravitons")
    momentum = [entry / 2 for entry in _slots(velocity)]
    return -adm.obstruction_value({key: momentum}, {key: _slots(amplitude)})


def isaacson_density(
    wave: Sequence[int],
    amplitude: Sequence[Fraction | int],
    velocity: Sequence[Fraction | int],
) -> Fraction:
    """Isaacson effective energy density, returned as ``rho * pi``.

    ``rho = (1/(64 pi)) < hdot_ij hdot^ij + d_l h_ij d^l h^ij >``, and with
    ``h = A cos(k.x)`` the two spatial averages contribute ``|Adot|^2 / 2`` and
    ``|k|^2 |A|^2 / 2``.  The factor of ``pi`` is stripped so the value stays
    rational; :data:`ISAACSON_TO_OBSTRUCTION` compares it against
    :func:`mode_energy` with the ``pi`` restored.
    """
    key = _key(wave)
    square = sum(component * component for component in key)
    kinetic = _tensor_norm(velocity)
    gradient = square * _tensor_norm(amplitude)
    return (kinetic + gradient) / 128


# --------------------------------------------------------------------------- #
# Step three: the homogeneous mode, split into expansion and shear
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Homogeneous:
    """The ``k = 0`` momentum written as ``B = -H delta + sigma``."""

    momentum: tuple[Fraction, ...]
    hubble: Fraction
    shear_squared: Fraction

    @property
    def value(self) -> Fraction:
        """``Q_0 = 6 H^2 - |sigma|^2``, the only positive direction in the form."""
        return 6 * self.hubble * self.hubble - self.shear_squared


def split_homogeneous(momentum: Sequence[Fraction | int]) -> Homogeneous:
    """Separate a homogeneous momentum into expansion rate and shear.

    The sign convention is the cosmological one, ``K_ij = -H g_ij`` for pure
    expansion, so ``H = -(tr B) / 3``; the shear is what is left after removing
    the trace and is trace-free by construction.
    """
    values = _slots(momentum)
    trace = sum(values[adm.symmetric_index(i, i)] for i in range(3))
    hubble = -trace / 3
    shear = list(values)
    for i in range(3):
        shear[adm.symmetric_index(i, i)] += hubble
    return Homogeneous(
        momentum=tuple(values), hubble=hubble, shear_squared=_tensor_norm(shear)
    )


# --------------------------------------------------------------------------- #
# The balance
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BalanceReport:
    """The second-order condition, assembled from its two halves."""

    hubble: Fraction
    shear_squared: Fraction
    graviton_energy: Fraction
    isaacson_pi_density: Fraction

    @property
    def homogeneous_value(self) -> Fraction:
        """``Q_0 = 6 H^2 - |sigma|^2``."""
        return 6 * self.hubble * self.hubble - self.shear_squared

    @property
    def obstruction(self) -> Fraction:
        """The full ``Q``: homogeneous contribution minus the wave energy."""
        return self.homogeneous_value - self.graviton_energy

    @property
    def satisfied(self) -> bool:
        """Whether the linearised solution passes the second-order condition."""
        return self.obstruction == 0

    @property
    def friedmann_residual(self) -> Fraction:
        """``6 H^2 - |sigma|^2 - 16 pi rho``, with ``pi`` cancelled.

        Zero exactly when the balance holds, which is the Friedmann constraint
        ``3 H^2 = 8 pi rho + (1/2) sigma_ij sigma^ij``.
        """
        return (
            self.homogeneous_value
            - ISAACSON_TO_OBSTRUCTION * self.isaacson_pi_density
        )


def balance(
    homogeneous: Sequence[Fraction | int],
    modes: Mapping[
        tuple[int, int, int],
        tuple[Sequence[Fraction | int], Sequence[Fraction | int]],
    ],
) -> BalanceReport:
    """Assemble the second-order condition for one linearised solution.

    ``homogeneous`` is the ``k = 0`` extrinsic curvature amplitude; ``modes`` maps
    each non-zero wave vector to its ``(amplitude, velocity)`` pair.  The
    obstruction is the exact sum over all of them, and it vanishes precisely when
    the expansion supplies the wave energy.
    """
    split = split_homogeneous(homogeneous)
    energy = Fraction(0)
    density = Fraction(0)
    for wave, (amplitude, velocity) in modes.items():
        key = _key(wave)
        if key == _ZERO_WAVE:
            raise ValueError("the homogeneous mode belongs in the first argument")
        energy += mode_energy(key, amplitude, velocity)
        density += isaacson_density(key, amplitude, velocity)
    return BalanceReport(
        hubble=split.hubble,
        shear_squared=split.shear_squared,
        graviton_energy=energy,
        isaacson_pi_density=density,
    )


def friedmann_residual(
    homogeneous: Sequence[Fraction | int],
    modes: Mapping[
        tuple[int, int, int],
        tuple[Sequence[Fraction | int], Sequence[Fraction | int]],
    ],
) -> Fraction:
    """``6 H^2 - |sigma|^2 - 16 pi rho`` for a linearised solution, exactly."""
    return balance(homogeneous, modes).friedmann_residual


def required_hubble_squared(
    modes: Mapping[
        tuple[int, int, int],
        tuple[Sequence[Fraction | int], Sequence[Fraction | int]],
    ],
    shear_squared: Fraction | int = 0,
) -> Fraction:
    """The ``H^2`` that makes the second-order condition hold, given the waves.

    Solving ``6 H^2 = |sigma|^2 + sum_k (-Q_k)``.  Always non-negative, since the
    wave energy is, so the required expansion exists for any wave content: the
    stability condition is a normalisation on the background, not a prohibition
    on the perturbation.
    """
    energy = Fraction(0)
    for wave, (amplitude, velocity) in modes.items():
        energy += mode_energy(_key(wave), amplitude, velocity)
    return (Fraction(shear_squared) + energy) / 6


# --------------------------------------------------------------------------- #
# The flat manifolds: how much shear the holonomy leaves
# --------------------------------------------------------------------------- #
def _symmetric_action(matrix: Sequence[Sequence[int]]) -> list[list[int]]:
    """The point-group action on the six slots of a symmetric tensor.

    ``(A . t)_ij = A_ik A_jl t_kl``, written in the slot basis of :mod:`adm`.
    """
    rows: list[list[int]] = []
    for i, j in adm.SYMMETRIC_PAIRS:
        row = [0] * 6
        for k in range(3):
            for l in range(3):
                row[adm.symmetric_index(k, l)] += matrix[i][k] * matrix[j][l]
        rows.append(row)
    return rows


def _invariant_dimension(
    group: Iterable[Sequence[Sequence[int]]], *, trace_free: bool
) -> int:
    """Dimension of the invariant symmetric tensors, optionally trace-free."""
    rows: list[list[int]] = []
    for matrix in group:
        action = _symmetric_action(matrix)
        for slot in range(6):
            rows.append(
                [action[slot][other] - (1 if slot == other else 0) for other in range(6)]
            )
    if trace_free:
        trace_row = [0] * 6
        for i in range(3):
            trace_row[adm.symmetric_index(i, i)] = 1
        rows.append(trace_row)
    if not rows:
        return 6
    return 6 - adm.integer_rank(rows)


def shear_budget(space: platycosm.Platycosm) -> tuple[int, int]:
    """``(homogeneous momenta, of which trace-free)`` invariant under the holonomy.

    The isotropic direction is invariant under every orthogonal matrix, so the
    first entry always exceeds the second by exactly one and the expansion that
    pays for the gravitons is available on every flat manifold.  The shear count
    is what the holonomy cuts: 5 on the torus down to 2 on the Hantzsche--Wendt
    manifold.  Shear enters the balance with the sign opposite to expansion, so
    this is a count of the ways a solution can fail the condition, not of ways to
    satisfy it.
    """
    group = platycosm.point_group(space)
    total = _invariant_dimension(group, trace_free=False)
    free = _invariant_dimension(group, trace_free=True)
    return total, free
