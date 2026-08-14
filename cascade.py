"""What the amplitudes actually do on the architecture: the analytic half.

The question
------------
:mod:`coherence` settles the combinatorial half of Tao's Step 2: there is an
explicit family of integer mode sets whose complete list of closing triads is the
Obukhov interaction graph and nothing else.  That says which triads *exist*.  It
says nothing about what the amplitudes *do*, and the shell model is a statement
about amplitudes.

Because the triad list is finite and known, the Euler equations restricted to the
architecture are a finite ODE system that can be written down and integrated
exactly.  This module does that and measures three things the shell model needs.

What the measurements say
-------------------------
**1. The construction is sound.**  The Galerkin system conserves energy to
``10^-12`` over hundreds of turnover times (:func:`energy_drift`), and transport
runs up the chain, from shell ``0`` outward.  Nothing is imposed: the
divergence-free condition, reality and the Leray projection are all applied to
the true nonlinearity.

**2. The chain is what makes the cascade one-way.**  A single gate is a triad,
and a triad is integrable: it oscillates and gives its energy back.  Adding
shells lets energy escape onward before it can return.  Measured as the fraction
of the exported energy that comes back (:func:`return_fraction`):

    shells       2       3       4       5
    recovered    0.66    0.26    0.16    0.14

So the cascade is a property of the *chain*, not of any gate.  This is the
mechanism the shell model abstracts, seen in the true equations.

**3. But the flux is not coherent, and that is the obstruction.**  The
instantaneous flux through the bottom gate reverses sign within a few turnover
times -- ``0.4`` to ``2.6`` against a turnover of ``0.22``
(:func:`first_flux_reversal`) -- and it does so for *every* choice of initial
phases.  Optimising the phases to maximise the exported energy improves it by
``0.08%`` (:func:`net_export`).

That last number cuts both ways, and both ways are worth saying.

* Against the shell model: the Obukhov system describes coherent transfer, and the
  embedded system does not transfer coherently.  Matched initial data diverges
  from the Obukhov trajectory by ``5%`` within about one turnover time.  **The
  shell model is not a trajectory-wise description of the embedded dynamics.**
* For the embedding: the net transport is nearly independent of the phases, so it
  is *robust* rather than fine-tuned.  Over ``90`` turnover times the bottom shell
  exports ``10%`` of its energy regardless of how the phases are set.  The cascade
  proceeds as a slow drift riding on a fast oscillation.

Where that leaves Step 2
------------------------
The three pieces now say: the coefficient is right (:mod:`embedding`), the
interaction graph is right (:mod:`coherence`), and the transport is real and
robust but *not* pointwise Obukhov.  An embedding therefore cannot proceed by
matching trajectories to the shell model.  It has to control a **time-averaged**
flux, with the oscillation treated as something to average over rather than
something to suppress.

That is a sharper statement of the open problem than "the analytic half is hard",
and it is the useful output here.  It is not a solution: averaging arguments need
error control over the averaging window, uniformly in the shell index, and
nothing in this module supplies that.

One further point, which the numerics make concrete.  Even with phases locked, a
single gate reverses once its low mode is exhausted -- the amplitude equation
drives the low amplitude through zero and the signs flip.  Sustained cascade
needs the bottom shell replenished.  That is exactly why Palasek's viscous
theorem carries an external force and why his Remark 1.4 records that forcing is
*necessary* for blow-up in this model.  The requirement is visible here as a
property of the true Euler dynamics rather than of the model.

Scope
-----
Finite Galerkin truncations of Euler conserve energy and cannot blow up, so
nothing here tests blow-up.  What is tested is transport: direction, reversibility
and phase sensitivity.  Time integration is floating point, unlike the exact
arithmetic elsewhere in this repository, because the object being measured is a
trajectory; the exact statements live in :mod:`coherence` and :mod:`embedding`,
and the conservation check is what pins the integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

import coherence as co

__all__ = [
    "GalerkinSystem",
    "build_system",
    "initial_data",
    "evolve",
    "energy_drift",
    "return_fraction",
    "turnover_time",
    "first_flux_reversal",
    "net_export",
    "SEED_VECTOR",
    "SEED_PARTNER",
]

LOGGER = logging.getLogger(__name__)

#: The seed pair used throughout: orthogonal, equal length, integer.
SEED_VECTOR = (3, 4, 0)
SEED_PARTNER = (0, 0, 5)


@dataclass(frozen=True)
class GalerkinSystem:
    """The Euler equations restricted to an architecture's modes.

    ``targets``, ``left`` and ``right`` index the quadratic terms: the mode
    ``targets[n]`` receives a contribution from the pair
    ``(left[n], right[n])``.  Because the architecture's triad list is exactly
    the Obukhov graph, this index set is small and every term in it is a gate.
    """

    modes: tuple[co.Vector, ...]
    wavevectors: np.ndarray
    targets: np.ndarray
    left: np.ndarray
    right: np.ndarray
    norms: np.ndarray
    shell: np.ndarray

    @property
    def shell_count(self) -> int:
        return int(self.shell.max()) + 1


def build_system(arch: co.Architecture) -> GalerkinSystem:
    """Assemble the finite ODE system from an architecture."""
    modes = [mode for shell in arch.shells for mode in shell]
    index = {mode: position for position, mode in enumerate(modes)}
    targets, left, right = [], [], []
    for first in modes:
        for second in modes:
            total = tuple(a + b for a, b in zip(first, second))
            if total in index:
                targets.append(index[total])
                left.append(index[first])
                right.append(index[second])
    wavevectors = np.array(modes, dtype=float)
    return GalerkinSystem(
        modes=tuple(modes),
        wavevectors=wavevectors,
        targets=np.array(targets, dtype=int),
        left=np.array(left, dtype=int),
        right=np.array(right, dtype=int),
        norms=(wavevectors * wavevectors).sum(axis=1),
        shell=np.array([j for j, shell in enumerate(arch.shells) for _ in shell]),
    )


def _rhs(system: GalerkinSystem, viscosity: float):
    wavevectors = system.wavevectors
    norms = system.norms

    def rhs(_time, state):
        velocity = state.view(np.complex128).reshape(-1, 3)
        derivative = np.zeros_like(velocity)
        advection = (
            -1j
            * np.einsum("ij,ij->i", velocity[system.left], wavevectors[system.right])
        )[:, None] * velocity[system.right]
        np.add.at(derivative, system.targets, advection)
        derivative -= (
            wavevectors * (np.einsum("ij,ij->i", wavevectors, derivative) / norms)[:, None]
        )
        if viscosity:
            derivative -= viscosity * norms[:, None] * velocity
        return derivative.ravel().view(np.float64)

    return rhs


def initial_data(
    system: GalerkinSystem,
    *,
    seed: int = 3,
    upper_fraction: float = 0.1,
    phases: Sequence[float] | None = None,
) -> np.ndarray:
    """Divergence-free, real-field initial data with energy concentrated at the bottom.

    Each conjugate pair is filled once and mirrored, so the velocity field is real
    by construction; each mode is projected transverse to its own wavevector.
    ``phases`` rotates each independent amplitude, which is the freedom
    :func:`net_export` searches over.
    """
    generator = np.random.default_rng(seed)
    velocity = np.zeros((len(system.modes), 3), dtype=complex)
    index = {mode: position for position, mode in enumerate(system.modes)}
    filled: set[int] = set()
    counter = 0
    for position, mode in enumerate(system.modes):
        if position in filled:
            continue
        partner = index[tuple(-component for component in mode)]
        draw = generator.normal(size=3) + 1j * generator.normal(size=3)
        wavevector = system.wavevectors[position]
        draw -= wavevector * (wavevector @ draw) / (wavevector @ wavevector)
        if phases is not None:
            draw *= np.exp(1j * phases[counter])
        velocity[position] = draw
        velocity[partner] = np.conj(draw)
        filled |= {position, partner}
        counter += 1
    for position in range(len(system.modes)):
        scale = 1.0 if system.shell[position] == 0 else upper_fraction
        velocity[position] *= scale / np.sqrt(system.norms[position])
    return velocity


def _shell_energies(system: GalerkinSystem, state: np.ndarray) -> np.ndarray:
    velocity = state.view(np.complex128).reshape(-1, 3)
    density = 0.5 * np.real(np.einsum("ij,ij->i", velocity, np.conj(velocity)))
    return np.bincount(system.shell, weights=density, minlength=system.shell_count)


def evolve(
    shells: int,
    duration: float,
    *,
    samples: int = 200,
    seed: int = 3,
    growth: int = 2,
    upper_fraction: float = 0.1,
    viscosity: float = 0.0,
    phases: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the exact system and return ``(times, shell energies)``."""
    from scipy.integrate import solve_ivp

    if shells < 2:
        raise ValueError("a cascade needs at least two shells")
    arch = co.architecture(SEED_VECTOR, SEED_PARTNER, [growth] * (shells - 1))
    system = build_system(arch)
    start = initial_data(
        system, seed=seed, upper_fraction=upper_fraction, phases=phases
    )
    solution = solve_ivp(
        _rhs(system, viscosity),
        [0.0, duration],
        start.ravel().view(np.float64),
        rtol=1e-11,
        atol=1e-14,
        dense_output=True,
        method="DOP853",
    )
    if solution.status != 0:
        raise RuntimeError(f"integration failed: {solution.message}")
    times = np.linspace(0.0, duration, samples)
    energies = np.array(
        [_shell_energies(system, solution.sol(time)) for time in times]
    )
    return times, energies


def energy_drift(shells: int = 3, duration: float = 200.0, **kwargs) -> float:
    """Relative change in total energy: the check on the integration.

    Euler conserves energy exactly and so does any Galerkin truncation of it, so
    this measures the integrator rather than the physics.  It is what licenses
    reading anything else off a trajectory.
    """
    _, energies = evolve(shells, duration, **kwargs)
    totals = energies.sum(axis=1)
    return float(abs(totals[-1] - totals[0]) / totals[0])


def return_fraction(shells: int, duration: float = 600.0, **kwargs) -> float:
    """Fraction of the exported energy that comes back to the bottom shell.

    ``(E_end - E_min) / (E_start - E_min)``.  Near one means the transfer is
    reversible, which is what a single triad does; small means the chain carried
    the energy away before it could return.
    """
    _, energies = evolve(shells, duration, **kwargs)
    bottom = energies[:, 0]
    exported = bottom[0] - bottom.min()
    if exported <= 0:
        raise ValueError("the bottom shell never exported any energy")
    return float((bottom[-1] - bottom.min()) / exported)


def turnover_time(shells: int = 3, **kwargs) -> float:
    """``1 / (N_0 X_0)``: the eddy turnover time of the bottom shell.

    The unit against which every time below should be read.
    """
    arch = co.architecture(
        SEED_VECTOR, SEED_PARTNER, [kwargs.get("growth", 2)] * (shells - 1)
    )
    system = build_system(arch)
    start = initial_data(
        system,
        seed=kwargs.get("seed", 3),
        upper_fraction=kwargs.get("upper_fraction", 0.1),
    )
    energies = _shell_energies(system, start.ravel().view(np.float64))
    return float(1.0 / (arch.radius(0) * np.sqrt(2 * energies[0])))


def first_flux_reversal(
    shells: int = 3, duration: float = 200.0, samples: int = 2000, **kwargs
) -> float:
    """Time at which the up-chain flux first changes sign.

    The flux is ``-dE_0/dt``.  A coherent cascade would keep one sign; what
    happens instead is a reversal after a few turnover times, for every phase
    choice tried.  Returns ``duration`` if no reversal occurs.
    """
    times, energies = evolve(shells, duration, samples=samples, **kwargs)
    flux = -np.gradient(energies[:, 0], times)
    sign = np.sign(flux[1])
    for position in range(2, len(times)):
        if np.sign(flux[position]) != sign and abs(flux[position]) > 1e-12:
            return float(times[position])
    return float(duration)


def net_export(
    shells: int = 3,
    duration: float = 20.0,
    *,
    phases: Sequence[float] | None = None,
    **kwargs,
) -> float:
    """Energy the bottom shell has lost by ``duration``.

    The quantity that turns out to be almost phase-independent: optimising the
    initial phases to maximise it gains ``0.08%`` over an unrotated draw.  That
    is the reason to call the transport robust rather than fine-tuned, and also
    the reason it cannot be matched to a coherent shell-model trajectory.
    """
    _, energies = evolve(shells, duration, samples=2, phases=phases, **kwargs)
    return float(energies[0, 0] - energies[-1, 0])
