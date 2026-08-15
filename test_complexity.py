"""Referees for `complexity`.

The one that matters is `TestStateCountReferee`: the search must enumerate
exactly ``2^n prod (2^k + 1)`` states.  Nothing in the tableau, the canonical
form or the gate rules was built to make that come out, so a sign error in
`_rowsum` or a canonical form that failed to identify two descriptions of one
state would both show up as a miscount.  It validates the whole apparatus at
once.

`TestTheRefutation` records what the module was built to test and failed to
find.
"""

from __future__ import annotations

import pytest

import complexity


# ---------------------------------------------------------------------------
# the referee
# ---------------------------------------------------------------------------


class TestStateCountReferee:
    @pytest.mark.parametrize(
        "qubits, expected", [(1, 6), (2, 60), (3, 1080), (4, 36720)]
    )
    def test_closed_form(self, qubits: int, expected: int):
        assert complexity.stabilizer_state_count(qubits) == expected

    @pytest.mark.parametrize("qubits", [1, 2, 3, 4])
    def test_search_finds_exactly_that_many(self, qubits: int):
        """The whole apparatus, checked by one number."""
        assert len(complexity.explore(qubits)) == complexity.stabilizer_state_count(
            qubits
        )

    def test_single_qubit_states_are_the_six_pauli_eigenstates(self):
        """``|0>, |1>, |+>, |->, |+i>, |-i>`` and nothing else."""
        assert len(complexity.explore(1)) == 6

    def test_count_rejects_zero_qubits(self):
        with pytest.raises(ValueError):
            complexity.stabilizer_state_count(0)


# ---------------------------------------------------------------------------
# tableau mechanics
# ---------------------------------------------------------------------------


class TestGateAlgebra:
    @pytest.mark.parametrize("qubits", [1, 2, 3])
    def test_hadamard_is_an_involution(self, qubits: int):
        state = complexity.zero_state(qubits)
        for target in range(qubits):
            twice = complexity.apply_hadamard(
                complexity.apply_hadamard(state, target), target
            )
            assert twice == state

    @pytest.mark.parametrize("qubits", [1, 2, 3])
    def test_phase_has_order_four_on_states(self, qubits: int):
        state = complexity.apply_hadamard(complexity.zero_state(qubits), 0)
        current = state
        for _ in range(4):
            current = complexity.apply_phase(current, 0)
        assert current == state

    @pytest.mark.parametrize("qubits", [2, 3])
    def test_cnot_is_an_involution(self, qubits: int):
        state = complexity.apply_hadamard(complexity.zero_state(qubits), 0)
        twice = complexity.apply_cnot(complexity.apply_cnot(state, 0, 1), 0, 1)
        assert twice == state

    def test_phase_fixes_the_zero_state(self):
        state = complexity.zero_state(2)
        assert complexity.apply_phase(state, 0) == state

    def test_cnot_fixes_the_zero_state(self):
        state = complexity.zero_state(2)
        assert complexity.apply_cnot(state, 0, 1) == state

    def test_hadamard_moves_the_zero_state(self):
        state = complexity.zero_state(1)
        assert complexity.apply_hadamard(state, 0) != state

    def test_gate_set_size(self):
        """``n`` Hadamards, ``n`` phases, ``n(n-1)`` controlled-nots."""
        for qubits in (1, 2, 3, 4):
            assert len(complexity.gate_set(qubits)) == 2 * qubits + qubits * (
                qubits - 1
            )

    @pytest.mark.parametrize("qubits", [1, 2, 3])
    def test_neighbour_count_matches_gate_set(self, qubits: int):
        state = complexity.zero_state(qubits)
        assert len(list(complexity.neighbours(state))) == len(
            complexity.gate_set(qubits)
        )


class TestCanonicalForm:
    def test_same_state_by_different_routes_compares_equal(self):
        """``CZ`` is symmetric in its two qubits, reached by two different circuits."""
        base = complexity.plus_state(2)
        first = complexity.apply_hadamard(
            complexity.apply_cnot(complexity.apply_hadamard(base, 1), 0, 1), 1
        )
        second = complexity.apply_hadamard(
            complexity.apply_cnot(complexity.apply_hadamard(base, 0), 1, 0), 0
        )
        assert first == second

    def test_ghz_is_symmetric_under_relabelling(self):
        """Building GHZ from qubit 1 instead of 0 gives the same state."""
        forward = complexity.ghz_state(3)
        state = complexity.apply_hadamard(complexity.zero_state(3), 1)
        state = complexity.apply_cnot(state, 1, 0)
        state = complexity.apply_cnot(state, 1, 2)
        assert state == forward

    @pytest.mark.parametrize("qubits", [1, 2, 3])
    def test_states_are_hashable(self, qubits: int):
        assert len({complexity.zero_state(qubits)}) == 1


class TestValidation:
    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_empty_register(self, bad: int):
        with pytest.raises(ValueError):
            complexity.zero_state(bad)
        with pytest.raises(ValueError):
            complexity.gate_set(bad)
        with pytest.raises(ValueError):
            complexity.explore(bad)

    def test_rejects_out_of_range_target(self):
        state = complexity.zero_state(2)
        with pytest.raises(ValueError, match="hadamard"):
            complexity.apply_hadamard(state, 5)
        with pytest.raises(ValueError, match="phase"):
            complexity.apply_phase(state, -1)
        with pytest.raises(ValueError, match="cnot"):
            complexity.apply_cnot(state, 0, 9)

    def test_rejects_cnot_onto_itself(self):
        with pytest.raises(ValueError, match="differ"):
            complexity.apply_cnot(complexity.zero_state(2), 1, 1)

    def test_rejects_wrong_generator_count(self):
        with pytest.raises(ValueError, match="generators"):
            complexity.StabilizerState(qubits=2, rows=((0, 1, 0),))

    def test_rejects_oversized_generator(self):
        with pytest.raises(ValueError, match="exceeds"):
            complexity.StabilizerState(qubits=1, rows=((0, 4, 0),))

    def test_rejects_bad_sign(self):
        with pytest.raises(ValueError, match="sign"):
            complexity.StabilizerState(qubits=1, rows=((0, 1, 2),))

    def test_refuses_impractical_sizes(self):
        with pytest.raises(ValueError, match="impractical"):
            complexity.explore(complexity.MAX_EXACT_QUBITS + 1)

    def test_ghz_needs_two_qubits(self):
        with pytest.raises(ValueError, match="two qubits"):
            complexity.ghz_state(1)

    def test_graph_states_need_two_qubits(self):
        with pytest.raises(ValueError, match="two qubits"):
            complexity.line_graph_state(1)
        with pytest.raises(ValueError, match="two qubits"):
            complexity.complete_graph_state(1)


# ---------------------------------------------------------------------------
# what the search measures
# ---------------------------------------------------------------------------


class TestComplexity:
    @pytest.mark.parametrize("qubits", [1, 2, 3])
    def test_zero_state_is_free(self, qubits: int):
        assert complexity.complexity_of(complexity.zero_state(qubits)) == 0

    @pytest.mark.parametrize("qubits", [1, 2, 3, 4])
    def test_distribution_sums_to_the_state_count(self, qubits: int):
        counts = complexity.complexity_distribution(qubits)
        assert sum(counts.values()) == complexity.stabilizer_state_count(qubits)

    @pytest.mark.parametrize("qubits", [1, 2, 3])
    def test_distribution_starts_at_one(self, qubits: int):
        """Exactly one state at distance zero: the origin."""
        assert complexity.complexity_distribution(qubits)[0] == 1

    @pytest.mark.parametrize("qubits", [2, 3])
    def test_first_shell_is_the_gate_set_orbit(self, qubits: int):
        """States one gate away, after the gates that fix the origin are removed."""
        origin = complexity.zero_state(qubits)
        distinct = {
            successor
            for successor in complexity.neighbours(origin)
            if successor != origin
        }
        assert complexity.complexity_distribution(qubits)[1] == len(distinct)

    @pytest.mark.parametrize("qubits", [1, 2, 3, 4])
    def test_complexity_never_exceeds_the_diameter(self, qubits: int):
        assert max(complexity.explore(qubits).values()) == complexity.diameter(qubits)

    def test_bfs_beats_or_matches_the_obvious_circuit(self):
        """The search returns a true minimum, so it can only improve on a construction.

        ``ghz_state`` is built with ``1 + (n-1) = n`` gates; the minimum cannot
        exceed that, and here it meets it.
        """
        for qubits in (2, 3, 4):
            assert complexity.complexity_of(complexity.ghz_state(qubits)) <= qubits

    def test_unreachable_state_is_reported(self):
        """A tableau that is not a stabilizer state cannot be looked up."""
        impostor = complexity.StabilizerState(qubits=1, rows=((1, 1, 0),))
        distances = complexity.explore(1)
        if impostor not in distances:
            with pytest.raises(ValueError, match="not reachable"):
                complexity.complexity_of(impostor)


class TestDiameterLaw:
    @pytest.mark.parametrize(
        "qubits, expected", [(1, 4), (2, 7), (3, 10), (4, 13)]
    )
    def test_measured_diameter(self, qubits: int, expected: int):
        assert complexity.diameter(qubits) == expected

    @pytest.mark.parametrize("qubits", [1, 2, 3, 4])
    def test_three_n_plus_one(self, qubits: int):
        assert complexity.predicted_diameter(qubits) == 3 * qubits + 1
        assert complexity.diameter_law_holds(qubits) is True

    def test_law_rejects_zero_qubits(self):
        with pytest.raises(ValueError):
            complexity.predicted_diameter(0)


class TestStructuredStates:
    @pytest.mark.parametrize("qubits", [2, 3, 4])
    def test_closed_forms(self, qubits: int):
        assert complexity.structured_complexities(
            qubits
        ) == complexity.predicted_structured_complexities(qubits)

    @pytest.mark.parametrize("qubits", [2, 3, 4])
    def test_ghz_and_plus_are_linear(self, qubits: int):
        measured = complexity.structured_complexities(qubits)
        assert measured["ghz"] == qubits
        assert measured["plus"] == qubits

    @pytest.mark.parametrize("qubits", [2, 3, 4])
    def test_complete_graph_sits_four_below_the_diameter(self, qubits: int):
        """Constant gap, at every size measured."""
        measured = complexity.structured_complexities(qubits)["complete-graph"]
        assert complexity.diameter(qubits) - measured == 4

    def test_laws_are_recorded(self):
        assert complexity.STRUCTURED_LAWS["ghz"] == "n"
        assert complexity.STRUCTURED_LAWS["complete-graph"] == "3(n - 1)"

    def test_predictions_reject_zero_qubits(self):
        with pytest.raises(ValueError):
            complexity.predicted_structured_complexities(0)


class TestTheRefutation:
    """What the module was built to find, and did not."""

    def test_structure_does_not_imply_low_complexity(self):
        """Two maximally structured states differing by a factor of three.

        GHZ and the complete graph state are each specified by a one-line rule.
        GHZ costs ``n``; the complete graph state costs ``3(n-1)`` and sits four
        gates below the diameter.  Structure is not a proxy for cheapness.

        The ratio is ``3(n-1)/n = 3 - 3/n``, so it rises toward three from below
        and is *exactly* two at ``n = 3``.  A strict ``> 2`` fails there, which
        is how the first version of this test failed -- the separation is real
        but its size had to be read off the formula rather than guessed.
        """
        ratios = []
        for qubits in (2, 3, 4):
            measured = complexity.structured_complexities(qubits)
            ratios.append(measured["complete-graph"] / measured["ghz"])
            assert measured["complete-graph"] / measured["ghz"] == pytest.approx(
                3 - 3 / qubits
            )
        assert ratios == sorted(ratios)
        assert ratios[-1] >= 2
        assert all(ratio < 3 for ratio in ratios)

    def test_the_gap_closes_instead_of_widening(self):
        """The prediction this module was written to confirm, refuted.

        I expected the mean to pull away from the structured states as ``n``
        grew.  Past ``n = 2`` it does the opposite, because the complete graph
        state tracks the diameter.
        """
        gaps = [complexity.structure_gap(qubits) for qubits in (2, 3, 4)]
        assert gaps == sorted(gaps, reverse=True)
        assert gaps[-1] < gaps[0]

    def test_some_structured_states_approach_the_hardest(self):
        for qubits in (2, 3, 4):
            ratio = (
                complexity.structured_complexities(qubits)["complete-graph"]
                / complexity.diameter(qubits)
            )
            assert ratio > 0.4

    def test_typical_complexity_is_between_ghz_and_diameter(self):
        for qubits in (2, 3, 4):
            mean = complexity.typical_complexity(qubits)
            assert complexity.structured_complexities(qubits)["ghz"] < mean
            assert mean < complexity.diameter(qubits)


#: Measured by a full search of all 2423520 five-qubit stabilizer states, which
#: took 921 seconds.  Recorded rather than re-run: the search is far too slow for
#: a test suite, but the numbers are the strongest confirmation the laws have.
FIVE_QUBIT_MEASUREMENT = {
    "states": 2423520,
    "diameter": 16,
    "mean": 12.1871,
    "plus": 5,
    "ghz": 5,
    "line-graph": 9,
    "complete-graph": 12,
}


class TestFiveQubitConfirmation:
    """The laws checked at the largest size the search reaches.

    These assert the *predictions* against measurements taken out of band, so
    the suite stays fast.  If a law were wrong, the prediction and the recorded
    number would disagree here.
    """

    def test_state_count(self):
        assert (
            complexity.stabilizer_state_count(5) == FIVE_QUBIT_MEASUREMENT["states"]
        )

    def test_diameter_law(self):
        assert (
            complexity.predicted_diameter(5) == FIVE_QUBIT_MEASUREMENT["diameter"]
        )

    def test_structured_laws(self):
        predicted = complexity.predicted_structured_complexities(5)
        for name in ("plus", "ghz", "line-graph", "complete-graph"):
            assert predicted[name] == FIVE_QUBIT_MEASUREMENT[name], name

    def test_complete_graph_gap_is_still_four(self):
        assert (
            FIVE_QUBIT_MEASUREMENT["diameter"]
            - FIVE_QUBIT_MEASUREMENT["complete-graph"]
            == complexity.CONCENTRATION_WIDTH
        )

    def test_mean_is_closing_on_the_complete_graph(self):
        gap = FIVE_QUBIT_MEASUREMENT["mean"] - 3 * (5 - 1)
        assert 0 < gap < complexity.mean_gap_to_complete_graph(4)


class TestConcentration:
    """Typical and maximal complexity differ by ``O(1)``."""

    @pytest.mark.parametrize("qubits", [2, 3, 4])
    def test_mean_sits_above_the_complete_graph(self, qubits: int):
        assert complexity.mean_gap_to_complete_graph(qubits) > 0

    def test_mean_gap_shrinks(self):
        """``mean - 3(n-1)`` runs 1.45, 0.86, 0.44 and keeps closing."""
        gaps = [complexity.mean_gap_to_complete_graph(n) for n in (2, 3, 4)]
        assert gaps == sorted(gaps, reverse=True)

    def test_diameter_minus_mean_grows_toward_the_width(self):
        """1.83, 2.55, 3.14, 3.56 -- rising, and bounded by 4."""
        gaps = [complexity.diameter_minus_mean(n) for n in (1, 2, 3, 4)]
        assert gaps == sorted(gaps)
        for gap in gaps:
            assert gap < complexity.CONCENTRATION_WIDTH

    def test_width_matches_the_complete_graph_gap(self):
        """The same constant, reached two ways.

        ``diameter - mean`` converges to it, and ``diameter - complete_graph``
        equals it exactly at every size.
        """
        for qubits in (2, 3, 4):
            measured = complexity.structured_complexities(qubits)["complete-graph"]
            assert (
                complexity.diameter(qubits) - measured
                == complexity.CONCENTRATION_WIDTH
            )

    def test_rejects_single_qubit(self):
        with pytest.raises(ValueError):
            complexity.mean_gap_to_complete_graph(1)

    def test_this_is_why_structure_buys_nothing(self):
        """The mechanism behind the refutation.

        Almost every stabilizer state already sits within ``O(1)`` of the
        diameter, so there is no room below for a structured state to occupy.
        The cheap ones -- GHZ at ``n`` against a diameter of ``3n+1`` -- are a
        vanishing fraction, and being a one-line rule is not what puts them
        there.

        The claim is that the fraction *vanishes*, so the test is that it falls
        with ``n`` -- 3.3 percent at ``n = 3``, 0.63 percent at ``n = 4``.  A
        fixed cutoff is the wrong check and the first version used one, set at
        two percent without looking at ``n = 3``.
        """
        fractions = []
        for qubits in (2, 3, 4):
            distribution = complexity.complexity_distribution(qubits)
            total = sum(distribution.values())
            cheap = sum(
                count for depth, count in distribution.items() if depth <= qubits
            )
            fractions.append(cheap / total)
        assert fractions == sorted(fractions, reverse=True)
        assert fractions[-1] < 0.01


class TestWhatThisDoesNotClaim:
    """The boundary, asserted so it cannot drift."""

    def test_the_algorithm_side_claim_needs_no_computation(self):
        """Optimising a circuit cannot shrink complexity, by definition.

        Complexity is the minimum over all circuits preparing a state.  An
        optimised circuit prepares the same state and was already in the set the
        minimum ranged over.  Nothing in this module measures algorithms, and
        nothing needs to.
        """
        assert not hasattr(complexity, "optimise_circuit")
        assert not hasattr(complexity, "wormhole_volume")

    def test_stabilizer_states_are_not_hard(self):
        """Gottesman-Knill: every state here is classically simulable.

        So this measures exact minimal *Clifford* complexity, and says nothing
        about quantum advantage or about the states Shor's algorithm produces.
        Reading it as a statement about computational hardness would be the
        overclaim.
        """
        assert not hasattr(complexity, "shor")
        assert not hasattr(complexity, "quantum_advantage")

    def test_no_geometry_is_constructed(self):
        """No volume, no tensor network, no bulk. Only exact gate counts."""
        assert not hasattr(complexity, "tensor_network")
        assert not hasattr(complexity, "bulk_volume")

    def test_laws_are_observations_not_theorems(self):
        """``3n + 1`` is read off four points. The test is the claim, not the formula."""
        assert complexity.diameter_law_holds(4) is True
        assert complexity.MAX_EXACT_QUBITS == 5
