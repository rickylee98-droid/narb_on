"""Exact circuit complexity of stabilizer states, by breadth-first search.

The brief this answers asks whether optimising an algorithm shrinks the
wormhole it corresponds to under Brown-Susskind complexity = volume.  It does
not, and the reason needs no computation: complexity is *defined* as the
minimum gate count over all circuits preparing a state.  Optimising a circuit
does not change the state it prepares, and the minimum was already taken over
every circuit including the optimised one.  Nothing shrinks.

The question that survives is different and sharper.  Complexity = volume is
supported by results about *random* circuits -- Haferkamp, Faist, Kothakonda,
Eisert and Yunger Halpern proved linear growth of exact complexity for those.
The open side is structured states.  So:

    do structured states have smaller exact complexity than typical ones,
    and by how much?

On stabilizer states that is decidable rather than estimable.  The set is
finite, the gate set ``{H, S, CNOT}`` generates it, and breadth-first search
from ``|0...0>`` returns the exact minimal gate count for every reachable state
-- a true minimum over all circuits, not a bound from some particular
construction.

What comes out
--------------

I expected structured states to sit far below typical ones, with the gap
widening in ``n``.  That is not what the search returns, and the correction is
the result.  Measured exactly for ``n = 1..4``:

    diameter          3n + 1        4,  7, 10, 13
    mean               ~2.4n      2.17, 4.45, 6.86, 9.44
    |+>^n                  n        1,  2,  3,  4
    GHZ                    n        --,  2,  3,  4
    line graph        2n - 1        --,  3,  5,  7
    complete graph  3(n - 1)        --,  3,  6,  9

Confirmed at ``n = 5`` by a full search of all 2423520 states (921 s): diameter
16, plus 5, GHZ 5, line graph 9, complete graph 12, mean 12.1871.  Every law
holds.

**Structure does not imply low complexity.**  The GHZ state and the complete
graph state are both maximally structured -- each is specified by a one-line
rule -- yet they differ by a factor of ``3 - 3/n``, and the complete graph state
sits *exactly four gates* below the diameter at every size measured, essentially
saturating it.

**And the distribution concentrates there.**  The stronger statement, which the
``n = 5`` mean exposed: ``mean - 3(n-1)`` runs 1.45, 0.86, 0.44, 0.19, so the
mean complexity converges to the complete graph state's exactly, and
``diameter - mean`` runs 1.83, 2.55, 3.14, 3.56, 3.81, converging to 4.  Typical
and maximal complexity differ by ``O(1)``, not by anything that grows.

That is why structure buys nothing here: almost every stabilizer state already
has essentially maximal complexity, so there is no room below for a structured
state to occupy.  The ones that are cheap -- GHZ at ``n``, against a diameter of
``3n+1`` -- are a vanishing fraction, and being a one-line rule is not what puts
them there.

So `structure_gap`, which measures the mean minus the *hardest* named
structured state, closes with ``n`` (+1.17, +1.45, +0.86, +0.44) instead of
widening.  That closing is the finding, not a defect: it refutes the intuition
the module was built to test.  The honest summary is:

    optimising the *algorithm*   ->  changes nothing, by definition
    choosing a structured *state* ->  buys nothing in general; some structured
                                      states are as hard as anything there is

Referee
-------

The number of ``n``-qubit stabilizer states is known in closed form,

    2^n * prod_{k=1}^{n} (2^k + 1)   =   6, 60, 1080, 36720, 2423520, ...

and the search is required to enumerate exactly that many.  Nothing in the
tableau representation, the canonical form or the gate rules was built to make
that come out; if the sign bookkeeping in `_rowsum` were wrong, or the canonical
form failed to identify two descriptions of one state, the count would miss.
It is the single check that validates the whole apparatus.

Scope
-----

Stabilizer states only.  They are the states for which exact complexity is
computable at all, and they are not a fair sample of Hilbert space: by the
Gottesman-Knill theorem they are classically simulable, so none of them is
computationally hard in the sense Shor's algorithm is.  What is computed here is
exact minimal *Clifford* complexity, and a test asserts that boundary rather
than letting it blur into a claim about quantum advantage.

Novelty
-------

Nothing here is new mathematics.  The stabilizer formalism is Gottesman's, the
tableau update rules and the ``Theta(n^2 / log n)`` bound on Clifford circuit
size are Aaronson-Gottesman, the state count is classical, and exact BFS over
small stabilizer sets has certainly been done.  The contribution is a computed,
exact answer to a specific claim -- with the refutation of the algorithm-side
version stated plainly, and the state-side version quantified instead of waved
at.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

__all__ = [
    "StabilizerState",
    "zero_state",
    "apply_hadamard",
    "apply_phase",
    "apply_cnot",
    "gate_set",
    "neighbours",
    "stabilizer_state_count",
    "explore",
    "complexity_distribution",
    "diameter",
    "complexity_of",
    "ghz_state",
    "plus_state",
    "line_graph_state",
    "complete_graph_state",
    "structured_complexities",
    "typical_complexity",
    "structure_gap",
    "predicted_diameter",
    "diameter_law_holds",
    "STRUCTURED_LAWS",
    "predicted_structured_complexities",
    "CONCENTRATION_WIDTH",
    "mean_gap_to_complete_graph",
    "diameter_minus_mean",
    "MAX_EXACT_QUBITS",
]

#: Largest register the search is expected to complete on in reasonable time.
#: The state count grows as ``2^n prod (2^k + 1)``: 36720 at ``n = 4`` and
#: 2423520 at ``n = 5``, so five is the practical ceiling for a full sweep.
MAX_EXACT_QUBITS: int = 5


# ---------------------------------------------------------------------------
# stabilizer tableaux
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilizerState:
    """A stabilizer state, as a canonical tableau of Pauli generators.

    Each generator is a triple ``(x, z, sign)`` where ``x`` and ``z`` are
    ``n``-bit masks selecting the qubits on which the Pauli has an ``X`` or
    ``Z`` component (both set means ``Y``), and ``sign`` is ``1`` for a leading
    minus.  The rows are kept in reduced row echelon form over ``GF(2)``, which
    makes the representation a *canonical* one: two tableaux describe the same
    state exactly when their canonical forms are equal.  That is what lets the
    search deduplicate, and it is what the state-count referee checks.
    """

    qubits: int
    rows: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if self.qubits < 1:
            raise ValueError(f"need at least one qubit, got {self.qubits}")
        if len(self.rows) != self.qubits:
            raise ValueError(
                f"a {self.qubits}-qubit state needs {self.qubits} generators, "
                f"got {len(self.rows)}"
            )
        limit = 1 << self.qubits
        for x, z, sign in self.rows:
            if not 0 <= x < limit or not 0 <= z < limit:
                raise ValueError(f"generator ({x}, {z}) exceeds {self.qubits} qubits")
            if sign not in (0, 1):
                raise ValueError(f"sign must be 0 or 1, got {sign}")


def _g(x1: int, z1: int, x2: int, z2: int) -> int:
    """Phase exponent contributed by one qubit when multiplying two Paulis.

    The Aaronson-Gottesman ``g`` function: returns the power of ``i`` picked up
    on this qubit when the Pauli ``(x1, z1)`` is multiplied onto ``(x2, z2)``.
    """
    if x1 == 0 and z1 == 0:
        return 0
    if x1 == 1 and z1 == 1:
        return z2 - x2
    if x1 == 1 and z1 == 0:
        return z2 * (2 * x2 - 1)
    return x2 * (1 - 2 * z2)


def _rowsum(
    target: tuple[int, int, int], source: tuple[int, int, int], qubits: int
) -> tuple[int, int, int]:
    """Multiply ``source`` onto ``target``, tracking the sign exactly.

    Stabilizer generators commute, so the accumulated power of ``i`` is always
    even and the product is again real-signed.  An odd total means the two rows
    anticommute, which cannot happen inside a stabilizer group -- so it is
    raised rather than rounded away.
    """
    total = 2 * target[2] + 2 * source[2]
    for position in range(qubits):
        total += _g(
            (source[0] >> position) & 1,
            (source[1] >> position) & 1,
            (target[0] >> position) & 1,
            (target[1] >> position) & 1,
        )
    total %= 4
    if total not in (0, 2):
        raise ArithmeticError(
            f"anticommuting generators in a stabilizer group (phase {total}); "
            "the tableau is not a valid stabilizer state"
        )
    return (target[0] ^ source[0], target[1] ^ source[1], total // 2)


def _canonicalise(
    rows: list[tuple[int, int, int]], qubits: int
) -> tuple[tuple[int, int, int], ...]:
    """Reduced row echelon form over ``GF(2)``, giving a unique representative.

    Columns are ordered ``x_0 .. x_{n-1}`` then ``z_0 .. z_{n-1}``.  Row
    operations are Pauli multiplications, so `_rowsum` carries the signs.
    """
    working = list(rows)
    pivot = 0
    for column in range(2 * qubits):
        if pivot >= qubits:
            break
        which = 0 if column < qubits else 1
        bit = column if column < qubits else column - qubits
        found = None
        for index in range(pivot, len(working)):
            if (working[index][which] >> bit) & 1:
                found = index
                break
        if found is None:
            continue
        working[pivot], working[found] = working[found], working[pivot]
        for index in range(len(working)):
            if index != pivot and ((working[index][which] >> bit) & 1):
                working[index] = _rowsum(working[index], working[pivot], qubits)
        pivot += 1
    return tuple(working)


def zero_state(qubits: int) -> StabilizerState:
    """``|0...0>``, stabilized by ``Z_0, ..., Z_{n-1}``."""
    if qubits < 1:
        raise ValueError(f"need at least one qubit, got {qubits}")
    rows = [(0, 1 << position, 0) for position in range(qubits)]
    return StabilizerState(qubits=qubits, rows=_canonicalise(rows, qubits))


# ---------------------------------------------------------------------------
# Clifford gates on the tableau
# ---------------------------------------------------------------------------


def _require_qubit(state: StabilizerState, target: int, name: str) -> None:
    if not 0 <= target < state.qubits:
        raise ValueError(
            f"{name} target {target} outside a {state.qubits}-qubit register"
        )


def apply_hadamard(state: StabilizerState, target: int) -> StabilizerState:
    """``H`` on ``target``: swap the ``X`` and ``Z`` components, sign ``^= x z``."""
    _require_qubit(state, target, "hadamard")
    mask = 1 << target
    rows = []
    for x, z, sign in state.rows:
        has_x = (x >> target) & 1
        has_z = (z >> target) & 1
        sign ^= has_x & has_z
        new_x = (x & ~mask) | (has_z << target)
        new_z = (z & ~mask) | (has_x << target)
        rows.append((new_x, new_z, sign))
    return StabilizerState(state.qubits, _canonicalise(rows, state.qubits))


def apply_phase(state: StabilizerState, target: int) -> StabilizerState:
    """``S`` on ``target``: ``z ^= x``, sign ``^= x z``."""
    _require_qubit(state, target, "phase")
    rows = []
    for x, z, sign in state.rows:
        has_x = (x >> target) & 1
        has_z = (z >> target) & 1
        sign ^= has_x & has_z
        rows.append((x, z ^ (has_x << target), sign))
    return StabilizerState(state.qubits, _canonicalise(rows, state.qubits))


def apply_cnot(state: StabilizerState, control: int, target: int) -> StabilizerState:
    """``CNOT`` from ``control`` to ``target``."""
    _require_qubit(state, control, "cnot control")
    _require_qubit(state, target, "cnot target")
    if control == target:
        raise ValueError(f"cnot control and target must differ, both were {control}")
    rows = []
    for x, z, sign in state.rows:
        control_x = (x >> control) & 1
        control_z = (z >> control) & 1
        target_x = (x >> target) & 1
        target_z = (z >> target) & 1
        sign ^= control_x & target_z & (target_x ^ control_z ^ 1)
        rows.append(
            (
                x ^ (control_x << target),
                z ^ (target_z << control),
                sign,
            )
        )
    return StabilizerState(state.qubits, _canonicalise(rows, state.qubits))


def gate_set(qubits: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Every generator available at this register size.

    ``n`` Hadamards, ``n`` phase gates and ``n(n-1)`` controlled-nots.
    """
    if qubits < 1:
        raise ValueError(f"need at least one qubit, got {qubits}")
    gates: list[tuple[str, tuple[int, ...]]] = []
    for target in range(qubits):
        gates.append(("h", (target,)))
        gates.append(("s", (target,)))
    for control in range(qubits):
        for target in range(qubits):
            if control != target:
                gates.append(("cnot", (control, target)))
    return tuple(gates)


def neighbours(state: StabilizerState) -> Iterator[StabilizerState]:
    """Every state one gate away."""
    for name, operands in gate_set(state.qubits):
        if name == "h":
            yield apply_hadamard(state, operands[0])
        elif name == "s":
            yield apply_phase(state, operands[0])
        else:
            yield apply_cnot(state, operands[0], operands[1])


# ---------------------------------------------------------------------------
# exhaustive search
# ---------------------------------------------------------------------------


def stabilizer_state_count(qubits: int) -> int:
    """``2^n prod_{k=1}^{n} (2^k + 1)``: the closed form, for the referee."""
    if qubits < 1:
        raise ValueError(f"need at least one qubit, got {qubits}")
    total = 1 << qubits
    for power in range(1, qubits + 1):
        total *= (1 << power) + 1
    return total


@lru_cache(maxsize=None)
def explore(qubits: int) -> dict[StabilizerState, int]:
    """Exact minimal gate count for every stabilizer state, by breadth-first search.

    Breadth-first order guarantees the first time a state is reached is by a
    shortest path, so these are true minima over all circuits -- not upper
    bounds from any particular construction.  That is the whole reason this
    module can say anything about complexity rather than about one compiler.

    Cached, because every summary below needs the same full search and it is by
    far the expensive step.  The returned mapping must not be mutated.
    """
    if qubits < 1:
        raise ValueError(f"need at least one qubit, got {qubits}")
    if qubits > MAX_EXACT_QUBITS:
        raise ValueError(
            f"exhaustive search is impractical above {MAX_EXACT_QUBITS} qubits; "
            f"asked for {qubits}, which has {stabilizer_state_count(qubits)} states"
        )
    start = zero_state(qubits)
    distances: dict[StabilizerState, int] = {start: 0}
    frontier: deque[StabilizerState] = deque([start])
    while frontier:
        current = frontier.popleft()
        depth = distances[current] + 1
        for successor in neighbours(current):
            if successor not in distances:
                distances[successor] = depth
                frontier.append(successor)
    return distances


def complexity_distribution(qubits: int) -> dict[int, int]:
    """How many stabilizer states sit at each exact complexity."""
    counts: dict[int, int] = {}
    for depth in explore(qubits).values():
        counts[depth] = counts.get(depth, 0) + 1
    return dict(sorted(counts.items()))


def diameter(qubits: int) -> int:
    """Complexity of the hardest stabilizer state: the Clifford state diameter."""
    return max(explore(qubits).values())


def complexity_of(state: StabilizerState) -> int:
    """Exact minimal gate count for one state."""
    distances = explore(state.qubits)
    if state not in distances:
        raise ValueError(
            "state is not reachable from |0...0> under {H, S, CNOT}; "
            "either it is not a stabilizer state or the gate set is incomplete"
        )
    return distances[state]


# ---------------------------------------------------------------------------
# structured states
# ---------------------------------------------------------------------------


def plus_state(qubits: int) -> StabilizerState:
    """``|+>^n``: one Hadamard per qubit, so complexity at most ``n``."""
    state = zero_state(qubits)
    for target in range(qubits):
        state = apply_hadamard(state, target)
    return state


def ghz_state(qubits: int) -> StabilizerState:
    """``(|0...0> + |1...1>)/sqrt(2)``: one Hadamard and ``n-1`` controlled-nots."""
    if qubits < 2:
        raise ValueError(f"a GHZ state needs at least two qubits, got {qubits}")
    state = apply_hadamard(zero_state(qubits), 0)
    for target in range(1, qubits):
        state = apply_cnot(state, 0, target)
    return state


def line_graph_state(qubits: int) -> StabilizerState:
    """Graph state on a path: Hadamards everywhere, then ``CZ`` along the line."""
    if qubits < 2:
        raise ValueError(f"a graph state needs at least two qubits, got {qubits}")
    state = plus_state(qubits)
    for position in range(qubits - 1):
        state = _controlled_z(state, position, position + 1)
    return state


def complete_graph_state(qubits: int) -> StabilizerState:
    """Graph state on the complete graph: ``CZ`` on every pair."""
    if qubits < 2:
        raise ValueError(f"a graph state needs at least two qubits, got {qubits}")
    state = plus_state(qubits)
    for first in range(qubits):
        for second in range(first + 1, qubits):
            state = _controlled_z(state, first, second)
    return state


def _controlled_z(
    state: StabilizerState, control: int, target: int
) -> StabilizerState:
    """``CZ = (I x H) CNOT (I x H)``, built from the generating set."""
    state = apply_hadamard(state, target)
    state = apply_cnot(state, control, target)
    return apply_hadamard(state, target)


def structured_complexities(qubits: int) -> dict[str, int]:
    """Exact complexity of the named structured states at this register size."""
    distances = explore(qubits)
    named = {
        "zero": zero_state(qubits),
        "plus": plus_state(qubits),
    }
    if qubits >= 2:
        named["ghz"] = ghz_state(qubits)
        named["line-graph"] = line_graph_state(qubits)
        named["complete-graph"] = complete_graph_state(qubits)
    return {name: distances[state] for name, state in named.items()}


def typical_complexity(qubits: int) -> float:
    """Mean exact complexity over all stabilizer states.

    The comparison point for the structured ones.  Not a random-circuit proxy --
    an average over the exact distribution.
    """
    distances = explore(qubits)
    return sum(distances.values()) / len(distances)


def predicted_diameter(qubits: int) -> int:
    """``3n + 1``: the diameter observed for ``n = 1..4``.

    A law read off four points, not a theorem.  `diameter_law_holds` tests it,
    and the test is worth more than the formula: if it fails at ``n = 5`` the
    formula is wrong and the failure is the interesting output.
    """
    if qubits < 1:
        raise ValueError(f"need at least one qubit, got {qubits}")
    return 3 * qubits + 1


def diameter_law_holds(qubits: int) -> bool:
    """Does the measured diameter match ``3n + 1`` at this size?"""
    return diameter(qubits) == predicted_diameter(qubits)


#: Closed forms observed for the named structured states, as functions of ``n``.
#: Read off ``n = 1..4`` and tested, not derived.
STRUCTURED_LAWS: dict[str, str] = {
    "zero": "0",
    "plus": "n",
    "ghz": "n",
    "line-graph": "2n - 1",
    "complete-graph": "3(n - 1)",
}


def predicted_structured_complexities(qubits: int) -> dict[str, int]:
    """`STRUCTURED_LAWS` evaluated at ``qubits``."""
    if qubits < 1:
        raise ValueError(f"need at least one qubit, got {qubits}")
    predictions = {"zero": 0, "plus": qubits}
    if qubits >= 2:
        predictions["ghz"] = qubits
        predictions["line-graph"] = 2 * qubits - 1
        predictions["complete-graph"] = 3 * (qubits - 1)
    return predictions


#: ``diameter - mean`` appears to converge to this, and it is also the constant
#: gap between the diameter and the complete graph state at every measured size.
CONCENTRATION_WIDTH: int = 4


def mean_gap_to_complete_graph(qubits: int) -> float:
    """``mean - 3(n-1)``: how far typical complexity sits above the complete graph.

    Runs 1.45, 0.86, 0.44, 0.19 for ``n = 2..5`` -- shrinking toward zero, so
    the mean converges to the complete graph state's complexity exactly.
    """
    if qubits < 2:
        raise ValueError(f"needs at least two qubits, got {qubits}")
    return typical_complexity(qubits) - 3 * (qubits - 1)


def diameter_minus_mean(qubits: int) -> float:
    """How far the hardest state sits above the typical one.

    Runs 1.83, 2.55, 3.14, 3.56, 3.81 for ``n = 1..5``, converging to
    `CONCENTRATION_WIDTH`.  Bounded, so typical and maximal complexity differ by
    ``O(1)`` -- the distribution concentrates just below the diameter.
    """
    return diameter(qubits) - typical_complexity(qubits)


def structure_gap(qubits: int) -> float:
    """Mean complexity minus the hardest named structured state's.

    Built to confirm that structured states are cheaper than typical ones, and
    it does the opposite: the value *closes* with ``n`` (+1.17, +1.45, +0.86,
    +0.44) because the complete graph state tracks the diameter.  Kept because
    the closing is the result -- structure buys nothing in general.
    """
    return typical_complexity(qubits) - max(structured_complexities(qubits).values())
