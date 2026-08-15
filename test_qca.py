"""Referees for `qca`.

The load-bearing checks are the ones that could have failed:

  * the class data of the polyhedral groups is confirmed by recomputing three
    classical invariant degrees (3, 4, 6) that were never fed in;
  * every exact cyclotomic result is recomputed by an independent
    floating-point path that shares no code with it;
  * the cyclic-group norms are recomputed a third way, by counting residues.

`TestWhatThisDoesNotClaim` records the boundary: this module is a statement
about symmetry protection, not a construction of an emergent graviton.
"""

from __future__ import annotations

import cmath
from fractions import Fraction

import pytest

import qca


# ---------------------------------------------------------------------------
# independent referees
# ---------------------------------------------------------------------------


def _numeric_character(spin: int, turn: Fraction) -> float:
    """``chi_l`` by direct complex exponentials. Shares no code with `qca`."""
    total = sum(
        cmath.exp(2j * cmath.pi * order * float(turn))
        for order in range(-spin, spin + 1)
    )
    assert abs(total.imag) < 1e-9, "character of a real representation must be real"
    return total.real


def _numeric_norm(spin: int, group: qca.RotationGroup) -> float:
    return (
        sum(
            count * _numeric_character(spin, turn) ** 2
            for turn, count in group.spectrum
        )
        / group.order
    )


def _numeric_invariants(spin: int, group: qca.RotationGroup) -> float:
    return (
        sum(count * _numeric_character(spin, turn) for turn, count in group.spectrum)
        / group.order
    )


ALL_GROUPS = list(qca.finite_rotation_groups(bound=8))


class TestAgainstIndependentArithmetic:
    @pytest.mark.parametrize("group", ALL_GROUPS, ids=lambda g: g.name)
    @pytest.mark.parametrize("spin", range(5))
    def test_norm_matches_floating_point(self, spin: int, group: qca.RotationGroup):
        assert qca.character_norm(spin, group) == pytest.approx(
            _numeric_norm(spin, group), abs=1e-6
        )

    @pytest.mark.parametrize("group", ALL_GROUPS, ids=lambda g: g.name)
    @pytest.mark.parametrize("spin", range(5))
    def test_invariants_match_floating_point(
        self, spin: int, group: qca.RotationGroup
    ):
        assert qca.invariant_count(spin, group) == pytest.approx(
            _numeric_invariants(spin, group), abs=1e-6
        )

    @pytest.mark.parametrize("order", range(1, 10))
    @pytest.mark.parametrize("spin", range(5))
    def test_cyclic_norm_by_residue_counting(self, spin: int, order: int):
        """A third path: over ``C_n`` every ``m`` gives a one-dimensional rep.

        The multiplet decomposes into characters labelled by ``m mod n``, so the
        sum of squared multiplicities is a pure counting problem.
        """
        residues: dict[int, int] = {}
        for magnetic in range(-spin, spin + 1):
            residues[magnetic % order] = residues.get(magnetic % order, 0) + 1
        expected = sum(count * count for count in residues.values())
        assert qca.character_norm(spin, qca.cyclic(order)) == expected

    @pytest.mark.parametrize("group", ALL_GROUPS, ids=lambda g: g.name)
    def test_results_are_exact_integers(self, group: qca.RotationGroup):
        for spin in range(4):
            assert type(qca.character_norm(spin, group)) is int
            assert type(qca.invariant_count(spin, group)) is int


class TestClassDataReferees:
    """Three classical facts, recomputed rather than assumed."""

    @pytest.mark.parametrize(
        "group, degree",
        [
            (qca.TETRAHEDRAL, 3),
            (qca.OCTAHEDRAL, 4),
            (qca.ICOSAHEDRAL, 6),
        ],
        ids=["tetrahedral", "octahedral", "icosahedral"],
    )
    def test_lowest_invariant_degree(self, group: qca.RotationGroup, degree: int):
        assert qca.lowest_invariant_degree(group) == degree
        assert qca.LOWEST_INVARIANT_DEGREES[group.name] == degree

    @pytest.mark.parametrize(
        "group, order",
        [(qca.TETRAHEDRAL, 12), (qca.OCTAHEDRAL, 24), (qca.ICOSAHEDRAL, 60)],
        ids=["T", "O", "I"],
    )
    def test_group_orders(self, group: qca.RotationGroup, order: int):
        assert group.order == order

    @pytest.mark.parametrize("group", ALL_GROUPS, ids=lambda g: g.name)
    def test_trivial_representation(self, group: qca.RotationGroup):
        assert qca.character_norm(0, group) == 1
        assert qca.invariant_count(0, group) == 1
        assert qca.is_irreducible(0, group)

    @pytest.mark.parametrize("group", ALL_GROUPS, ids=lambda g: g.name)
    def test_vector_invariants_count_the_fixed_axis(self, group: qca.RotationGroup):
        """``<chi_1, 1>_G`` is the dimension of the ``G``-fixed subspace of ``R^3``.

        Three degenerate cases matter, and the first version of this test got
        two of them wrong by assuming the generic one:

          * ``C_1`` is the trivial group and fixes all of ``R^3``  -> 3
          * ``C_n`` for ``n >= 2`` fixes its rotation axis          -> 1
          * ``D_1`` is a single half turn, i.e. ``C_2`` relabelled  -> 1
          * every other group has axes in more than one direction   -> 0
        """
        axial = group.name == "D_1" or (
            group.name.startswith("C_") and group.name != "C_1"
        )
        if group.name == "C_1":
            expected = 3
        elif axial:
            expected = 1
        else:
            expected = 0
        assert qca.invariant_count(1, group) == expected

    def test_rotation_orders(self):
        assert qca.ICOSAHEDRAL.rotation_orders == frozenset({1, 2, 3, 5})
        assert qca.OCTAHEDRAL.rotation_orders == frozenset({1, 2, 3, 4})
        assert qca.TETRAHEDRAL.rotation_orders == frozenset({1, 2, 3})


# ---------------------------------------------------------------------------
# the classification
# ---------------------------------------------------------------------------


class TestProtection:
    def test_spin_one_is_protected_by_all_three_polyhedral_groups(self):
        names = {group.name for group in qca.protecting_groups(1)}
        assert names == {"T", "O", "I"}

    def test_spin_two_is_protected_by_the_icosahedral_group_alone(self):
        protectors = qca.protecting_groups(2)
        assert [group.name for group in protectors] == ["I"]

    @pytest.mark.parametrize("spin", [3, 4, 5])
    def test_nothing_above_spin_two_is_protected(self, spin: int):
        assert qca.protecting_groups(spin) == ()

    def test_ceiling_is_the_weinberg_witten_bound(self):
        assert qca.MAXIMAL_IRREP_DIMENSION == 5
        assert qca.PROTECTED_SPIN_CEILING == 2
        assert 2 * qca.PROTECTED_SPIN_CEILING + 1 == qca.MAXIMAL_IRREP_DIMENSION

    @pytest.mark.parametrize("group", ALL_GROUPS, ids=lambda g: g.name)
    def test_no_group_exceeds_the_ceiling(self, group: qca.RotationGroup):
        assert qca.maximal_protected_spin(group) <= qca.PROTECTED_SPIN_CEILING

    def test_only_the_icosahedral_group_reaches_the_ceiling(self):
        reaching = [
            group.name
            for group in qca.finite_rotation_groups(8)
            if qca.maximal_protected_spin(group) == qca.PROTECTED_SPIN_CEILING
        ]
        assert reaching == ["I"]

    @pytest.mark.parametrize(
        "group, spin, cost",
        [
            (qca.ICOSAHEDRAL, 2, 0),
            (qca.OCTAHEDRAL, 2, 1),
            (qca.TETRAHEDRAL, 2, 2),
            (qca.OCTAHEDRAL, 1, 0),
        ],
        ids=["icosahedral-free", "cubic-one-knob", "tetrahedral-two", "photon-free"],
    )
    def test_tuning_cost(self, group: qca.RotationGroup, spin: int, cost: int):
        """Schur's lemma is worth exactly `character_norm - 1` tuned parameters."""
        assert qca.tuning_cost(spin, group) == cost

    def test_cubic_splitting_is_the_textbook_one(self):
        """``l = 2`` on a cube splits in two pieces: ``E_g + T_2g``."""
        assert qca.character_norm(2, qca.OCTAHEDRAL) == 2


class TestCrystallographicRestriction:
    @pytest.mark.parametrize(
        "group, expected",
        [
            (qca.TETRAHEDRAL, True),
            (qca.OCTAHEDRAL, True),
            (qca.ICOSAHEDRAL, False),
            (qca.cyclic(6), True),
            (qca.cyclic(5), False),
            (qca.cyclic(7), False),
            (qca.dihedral(4), True),
            (qca.dihedral(5), False),
        ],
        ids=["T", "O", "I", "C6", "C5", "C7", "D4", "D5"],
    )
    def test_is_crystallographic(self, group: qca.RotationGroup, expected: bool):
        assert qca.is_crystallographic(group) is expected

    def test_allowed_orders(self):
        assert qca.CRYSTALLOGRAPHIC_ORDERS == frozenset({1, 2, 3, 4, 6})
        assert 5 not in qca.CRYSTALLOGRAPHIC_ORDERS

    def test_the_no_go(self):
        """Every spin-2 protector is aperiodic. This is the whole result."""
        assert qca.graviton_requires_aperiodic() is True

    def test_the_photon_has_no_such_obstruction(self):
        """Spin one is protected on structures that *are* crystallographic."""
        periodic = [
            group.name
            for group in qca.protecting_groups(1)
            if qca.is_crystallographic(group)
        ]
        assert set(periodic) == {"T", "O"}


class TestTuningLadder:
    """Where anisotropy first bites, order by order in momentum."""

    @pytest.mark.parametrize(
        "degree, expected",
        [
            (0, (0,)),
            (1, (1,)),
            (2, (2, 0)),
            (3, (3, 1)),
            (4, (4, 2, 0)),
            (5, (5, 3, 1)),
        ],
    )
    def test_polynomial_content(self, degree: int, expected: tuple[int, ...]):
        assert qca.polynomial_multiplet_content(degree) == expected

    @pytest.mark.parametrize("degree", range(8))
    def test_polynomial_content_has_right_dimension(self, degree: int):
        """``Sym^n`` of a 3-dim space has dimension ``(n+1)(n+2)/2``."""
        total = sum(
            2 * spin + 1 for spin in qca.polynomial_multiplet_content(degree)
        )
        assert total == (degree + 1) * (degree + 2) // 2

    @pytest.mark.parametrize("spin", range(4))
    def test_operator_content_has_right_dimension(self, spin: int):
        """``End(W)`` has dimension ``(2s+1)^2``."""
        total = sum(2 * j + 1 for j in qca.operator_multiplet_content(spin))
        assert total == (2 * spin + 1) ** 2

    @pytest.mark.parametrize("group", ALL_GROUPS, ids=lambda g: g.name)
    @pytest.mark.parametrize("spin", [1, 2])
    def test_lattice_never_below_isotropic(self, spin: int, group: qca.RotationGroup):
        """A rotationally invariant coupling is in particular ``G`` invariant."""
        for rung in qca.tuning_ladder(spin, group, max_degree=4):
            assert rung.lattice >= rung.isotropic
            assert rung.excess == rung.lattice - rung.isotropic
            assert rung.excess >= 0

    @pytest.mark.parametrize("group", ALL_GROUPS, ids=lambda g: g.name)
    @pytest.mark.parametrize("spin", range(4))
    def test_degree_zero_excess_is_the_commutant(
        self, spin: int, group: qca.RotationGroup
    ):
        """Two independent routes to the same number, and they must agree.

        `tuning_cost` comes from the character norm of the multiplet;
        `zero_momentum_splitting` comes from the degree-zero rung of the
        polynomial-times-operator decomposition.  They share no code path.
        """
        assert qca.zero_momentum_splitting(spin, group) == qca.tuning_cost(spin, group)

    @pytest.mark.parametrize(
        "group, onset",
        [(qca.TETRAHEDRAL, 0), (qca.OCTAHEDRAL, 0), (qca.ICOSAHEDRAL, 2)],
        ids=["T", "O", "I"],
    )
    def test_spin_two_onset(self, group: qca.RotationGroup, onset: int):
        assert qca.first_anisotropic_degree(2, group) == onset
        assert qca.ANISOTROPY_ONSET[group.name] == onset

    @pytest.mark.parametrize(
        "group, onset",
        [(qca.TETRAHEDRAL, 1), (qca.OCTAHEDRAL, 2), (qca.ICOSAHEDRAL, 4)],
        ids=["T", "O", "I"],
    )
    def test_spin_one_onset(self, group: qca.RotationGroup, onset: int):
        """The photon control.

        A cubic lattice is isotropic through order ``k^1``, so the linear
        dispersion of an emergent photon is protected and the first correction
        is an irrelevant ``k^2`` term.  That is the regime in which emergent
        photons are known to work, and the method reproduces it.
        """
        assert qca.first_anisotropic_degree(1, group) == onset

    def test_the_relevant_versus_irrelevant_split(self):
        """The sharpened no-go, and the reason it is stronger than a parameter count.

        On a crystallographic group the spin-2 multiplet splits at zero
        momentum: the pieces acquire different gaps, so no massless spin-2
        object exists and nothing suppresses the failure in the infrared.  On
        the icosahedral group the zero-momentum degeneracy is protected and the
        first anisotropy is a velocity term at order ``k^2`` -- an irrelevant
        operator.
        """
        for group in qca.finite_rotation_groups(8):
            splitting = qca.zero_momentum_splitting(2, group)
            if qca.is_crystallographic(group):
                assert splitting > 0, f"{group.name} unexpectedly protects spin 2"
            else:
                assert (splitting == 0) == qca.is_irreducible(2, group)
        assert qca.zero_momentum_splitting(2, qca.ICOSAHEDRAL) == 0
        assert qca.first_anisotropic_degree(2, qca.ICOSAHEDRAL) == 2

    def test_photon_survives_where_graviton_does_not(self):
        """Same lattice, same method, opposite verdicts. This is the control."""
        assert qca.zero_momentum_splitting(1, qca.OCTAHEDRAL) == 0
        assert qca.zero_momentum_splitting(2, qca.OCTAHEDRAL) == 1

    def test_tetrahedral_permits_a_linear_coupling(self):
        """``T`` allows an order-``k`` anisotropy that ``O`` forbids.

        ``l = 2`` restricted to ``T`` contains the same three-dimensional irrep
        as ``l = 1``, so a linear-in-``k`` vector-quadrupole coupling is allowed;
        under ``O`` the two land in different irreps (``T_1`` versus ``E + T_2``)
        and it is forbidden.
        """
        assert qca.character_inner_product(1, 2, qca.TETRAHEDRAL) == 1
        assert qca.character_inner_product(1, 2, qca.OCTAHEDRAL) == 0

    def test_ladder_reports_none_when_search_is_too_short(self):
        assert qca.first_anisotropic_degree(2, qca.ICOSAHEDRAL, limit=1) is None

    def test_ladder_rejects_bad_input(self):
        with pytest.raises(ValueError):
            qca.tuning_ladder(2, qca.OCTAHEDRAL, max_degree=-1)
        with pytest.raises(ValueError):
            qca.polynomial_multiplet_content(-1)
        with pytest.raises(ValueError):
            qca.operator_multiplet_content(-1)

    def test_rung_fields(self):
        rung = qca.tuning_ladder(2, qca.OCTAHEDRAL, max_degree=0)[0]
        assert (rung.degree, rung.lattice, rung.isotropic, rung.excess) == (0, 2, 1, 1)


class TestInnerProduct:
    @pytest.mark.parametrize("group", ALL_GROUPS, ids=lambda g: g.name)
    @pytest.mark.parametrize("left", range(4))
    @pytest.mark.parametrize("right", range(4))
    def test_matches_floating_point(
        self, left: int, right: int, group: qca.RotationGroup
    ):
        expected = (
            sum(
                count
                * _numeric_character(left, turn)
                * _numeric_character(right, turn)
                for turn, count in group.spectrum
            )
            / group.order
        )
        assert qca.character_inner_product(left, right, group) == pytest.approx(
            expected, abs=1e-6
        )

    @pytest.mark.parametrize("group", ALL_GROUPS, ids=lambda g: g.name)
    @pytest.mark.parametrize("spin", range(4))
    def test_agrees_with_character_norm(self, spin: int, group: qca.RotationGroup):
        assert qca.character_inner_product(spin, spin, group) == qca.character_norm(
            spin, group
        )

    @pytest.mark.parametrize("group", ALL_GROUPS, ids=lambda g: g.name)
    @pytest.mark.parametrize("spin", range(4))
    def test_agrees_with_invariant_count(self, spin: int, group: qca.RotationGroup):
        assert qca.character_inner_product(spin, 0, group) == qca.invariant_count(
            spin, group
        )

    @pytest.mark.parametrize("group", ALL_GROUPS, ids=lambda g: g.name)
    def test_symmetric(self, group: qca.RotationGroup):
        for left in range(4):
            for right in range(4):
                assert qca.character_inner_product(
                    left, right, group
                ) == qca.character_inner_product(right, left, group)

    def test_rejects_negative_spin(self):
        with pytest.raises(ValueError):
            qca.character_inner_product(-1, 2, qca.OCTAHEDRAL)


class TestHelicityAliasing:
    @pytest.mark.parametrize("axis_order", [1, 2, 3, 4, 5, 6, 7])
    def test_modulus(self, axis_order: int):
        assert qca.helicity_alias_modulus(axis_order) == axis_order

    @pytest.mark.parametrize("spin", [0, 1, 2, 3])
    def test_minimal_axis_order(self, spin: int):
        threshold = qca.minimal_axis_order(spin)
        assert threshold == 2 * spin + 1
        assert qca.helicity_is_resolved(threshold, spin)
        if threshold > 1:
            assert not qca.helicity_is_resolved(threshold - 1, spin)

    def test_photon_needs_a_three_fold_axis(self):
        assert qca.minimal_axis_order(1) == 3
        assert not qca.helicity_is_resolved(2, 1)

    def test_graviton_needs_a_five_fold_axis(self):
        assert qca.minimal_axis_order(2) == 5
        assert not qca.helicity_is_resolved(4, 2)

    def test_six_fold_axes_clear_the_aliasing_bound(self):
        """The weaker argument does not forbid a periodic graviton -- 6 >= 5."""
        assert qca.helicity_is_resolved(6, 2)
        assert 6 in qca.CRYSTALLOGRAPHIC_ORDERS

    def test_single_axis_argument_is_strictly_weaker(self):
        """Recorded because I reached for it first and it is not enough.

        Aliasing gives ``n >= 5``, which a six-fold axis satisfies, so on its own
        it leaves a periodic graviton open.  The three-dimensional statement --
        no crystallographic point group has a five-dimensional irrep -- is what
        actually closes it.
        """
        assert qca.aliasing_is_insufficient(spin=2) is True


# ---------------------------------------------------------------------------
# construction and validation
# ---------------------------------------------------------------------------


class TestConstruction:
    @pytest.mark.parametrize("order", range(1, 10))
    def test_cyclic_order(self, order: int):
        assert qca.cyclic(order).order == order

    @pytest.mark.parametrize("order", range(1, 10))
    def test_dihedral_order(self, order: int):
        assert qca.dihedral(order).order == 2 * order

    def test_classification_is_exhaustive(self):
        names = [group.name for group in qca.finite_rotation_groups(3)]
        assert names == ["C_1", "C_2", "C_3", "D_1", "D_2", "D_3", "T", "O", "I"]

    @pytest.mark.parametrize("bad", [0, -1, -7])
    def test_rejects_non_positive_orders(self, bad: int):
        with pytest.raises(ValueError):
            qca.cyclic(bad)
        with pytest.raises(ValueError):
            qca.dihedral(bad)
        with pytest.raises(ValueError):
            list(qca.finite_rotation_groups(bad))

    def test_rejects_missing_identity(self):
        with pytest.raises(ValueError, match="identity"):
            qca.RotationGroup(name="bad", spectrum=((Fraction(1, 2), 3),))

    def test_rejects_duplicate_identity(self):
        with pytest.raises(ValueError, match="identity"):
            qca.RotationGroup(name="bad", spectrum=((Fraction(0), 2),))

    def test_rejects_empty_spectrum(self):
        with pytest.raises(ValueError, match="empty"):
            qca.RotationGroup(name="bad", spectrum=())

    def test_rejects_turn_outside_unit_interval(self):
        with pytest.raises(ValueError, match="outside"):
            qca.RotationGroup(
                name="bad", spectrum=((Fraction(0), 1), (Fraction(3, 2), 1))
            )

    def test_rejects_non_positive_multiplicity(self):
        with pytest.raises(ValueError, match="multiplicity"):
            qca.RotationGroup(
                name="bad", spectrum=((Fraction(0), 1), (Fraction(1, 2), 0))
            )

    def test_rejects_float_turns(self):
        with pytest.raises(TypeError):
            qca.RotationGroup(name="bad", spectrum=((Fraction(0), 1), (0.5, 1)))

    @pytest.mark.parametrize("bad", [-1, -3])
    def test_rejects_negative_spin(self, bad: int):
        with pytest.raises(ValueError):
            qca.character_norm(bad, qca.OCTAHEDRAL)
        with pytest.raises(ValueError):
            qca.invariant_count(bad, qca.OCTAHEDRAL)
        with pytest.raises(ValueError):
            qca.spin_character(bad, Fraction(0))
        with pytest.raises(ValueError):
            qca.helicity_is_resolved(3, bad)

    def test_rejects_bad_axis_order(self):
        with pytest.raises(ValueError):
            qca.helicity_alias_modulus(0)

    def test_spin_character_rejects_floats(self):
        with pytest.raises(TypeError):
            qca.spin_character(2, 0.25)

    def test_lowest_invariant_degree_reports_failure(self):
        with pytest.raises(ValueError, match="no invariant"):
            qca.lowest_invariant_degree(qca.ICOSAHEDRAL, search_limit=5)

    def test_group_str(self):
        assert str(qca.ICOSAHEDRAL) == "I"


class TestSpinCharacter:
    def test_identity_gives_the_dimension(self):
        for spin in range(5):
            assert qca.spin_character(spin, Fraction(0)) == 2 * spin + 1

    def test_scalar_is_constant(self):
        for turn in (Fraction(0), Fraction(1, 3), Fraction(2, 5)):
            assert qca.spin_character(0, turn) == 1

    @pytest.mark.parametrize("spin", range(4))
    @pytest.mark.parametrize(
        "turn", [Fraction(1, 2), Fraction(1, 3), Fraction(1, 4), Fraction(1, 5)]
    )
    def test_matches_numeric(self, spin: int, turn: Fraction):
        assert float(qca.spin_character(spin, turn)) == pytest.approx(
            _numeric_character(spin, turn), abs=1e-9
        )

    def test_half_turn_alternates(self):
        """``chi_l(pi) = (-1)^l``."""
        for spin in range(6):
            assert qca.spin_character(spin, Fraction(1, 2)) == (-1) ** spin


class TestWhatThisDoesNotClaim:
    """The boundary of the result, asserted so it cannot quietly drift."""

    def test_no_automaton_is_constructed(self):
        assert not hasattr(qca, "automaton")
        assert not hasattr(qca, "graviton")
        assert not hasattr(qca, "emergent_metric")

    def test_protection_is_necessary_not_sufficient(self):
        """Icosahedral symmetry protects the multiplet. It does not supply a field.

        `graviton_requires_aperiodic` says every structure that could carry a
        symmetry-protected spin-2 multiplet is aperiodic.  It does not say an
        icosahedral quasicrystal has an emergent graviton -- only that nothing
        periodic can, and that anything aperiodic still has to produce the
        massless dispersion and the gauge redundancy on its own.
        """
        assert qca.is_irreducible(2, qca.ICOSAHEDRAL)
        assert not qca.is_crystallographic(qca.ICOSAHEDRAL)

    def test_fine_tuning_is_not_impossibility(self):
        """A cubic model can still be tuned; the cost is one parameter, not infinity."""
        assert qca.tuning_cost(2, qca.OCTAHEDRAL) == 1
