"""Tests for linearisation instability across the compact flat 3-manifolds.

Three referees that are not restatements of the module:

* For a compact flat manifold the first Betti number equals the dimension of the
  holonomy-invariant subspace, so the *classical* ``b_1`` of each platycosm --
  3, 1, 1, 0 -- must come back out of a computation that never mentions it.
* The Bieberbach condition is checked, not assumed. A first presentation of the
  Hantzsche--Wendt manifold was rejected by it for having genuine fixed points,
  which is what a correct test should do.
* ``G1`` is the three-torus, so it must reproduce :mod:`adm` exactly, including
  the inertia ``(4,4,0)``.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

import adm
import platycosm as pc

#: Classical first Betti numbers of the orientable platycosms handled here.
KNOWN_BETTI = {"G1": 3, "G2": 1, "G4": 1, "G6": 0}
#: Classical point-group orders.
KNOWN_ORDER = {"G1": 1, "G2": 2, "G4": 4, "G6": 4}


def by_label(label: str) -> pc.Platycosm:
    for space in pc.PLATYCOSMS:
        if space.label == label:
            return space
    raise KeyError(label)


class TestPointGroups:
    @pytest.mark.parametrize("label,order", sorted(KNOWN_ORDER.items()))
    def test_orders_are_the_classical_ones(self, label, order) -> None:
        assert len(pc.point_group(by_label(label))) == order

    @pytest.mark.parametrize("space", pc.PLATYCOSMS, ids=lambda s: s.label)
    def test_group_is_closed_and_contains_identity(self, space) -> None:
        group = pc.point_group(space)
        assert pc._IDENTITY in group
        for a in group:
            for b in group:
                assert pc._multiply(a, b) in group

    @pytest.mark.parametrize("space", pc.PLATYCOSMS, ids=lambda s: s.label)
    def test_every_element_is_orientation_preserving(self, space) -> None:
        """These are the *orientable* platycosms, so determinants are ``+1``."""
        for m in pc.point_group(space):
            det = (
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            )
            assert det == 1


class TestBettiReferee:
    """``b_1`` of a compact flat manifold is the holonomy-invariant dimension.

    The module computes the invariant subspace from the point group and never
    refers to homology; the classical Betti numbers therefore act as an
    independent check on the group presentations.
    """

    @pytest.mark.parametrize("label,betti", sorted(KNOWN_BETTI.items()))
    def test_fixed_subspace_reproduces_betti_one(self, label, betti) -> None:
        assert pc.fixed_subspace_dimension(by_label(label)) == betti

    @pytest.mark.parametrize("space", pc.PLATYCOSMS, ids=lambda s: s.label)
    def test_declared_betti_matches_the_computation(self, space) -> None:
        assert pc.fixed_subspace_dimension(space) == space.betti_one


class TestBieberbach:
    @pytest.mark.parametrize("space", pc.PLATYCOSMS, ids=lambda s: s.label)
    def test_every_platycosm_acts_freely(self, space) -> None:
        assert pc.is_bieberbach(space)

    def test_a_bad_hantzsche_wendt_presentation_is_rejected(self) -> None:
        """The check earns its keep: this presentation was tried and is wrong.

        A half-turn about the x-axis whose translation part has no x-component
        is a rotation about a shifted axis, not a screw, and it fixes points.
        The quotient is then an orbifold rather than a manifold.
        """
        bad = pc.Platycosm(
            name="broken Hantzsche-Wendt",
            label="bad",
            generators=(pc._HALF_Z, pc._HALF_X),
            shifts=(
                (Fraction(1, 2), Fraction(0), Fraction(1, 2)),
                (Fraction(0), Fraction(1, 2), Fraction(1, 2)),
            ),
            betti_one=0,
        )
        assert not pc.is_bieberbach(bad)

    def test_a_pure_rotation_is_rejected(self) -> None:
        """Zero translation part means the axis itself is fixed pointwise."""
        bad = pc.Platycosm(
            name="pure rotation",
            label="bad",
            generators=(pc._HALF_Z,),
            shifts=((Fraction(0), Fraction(0), Fraction(0)),),
            betti_one=1,
        )
        assert not pc.is_bieberbach(bad)

    def test_mismatched_generators_and_shifts_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="generators against"):
            pc.Platycosm(
                name="x", label="x", generators=(pc._HALF_Z,), shifts=(), betti_one=0
            )


class TestKillingInitialData:
    @pytest.mark.parametrize("space", pc.PLATYCOSMS, ids=lambda s: s.label)
    def test_kid_is_lapse_plus_killing_fields(self, space) -> None:
        assert pc.kid_dimension(space) == 1 + pc.fixed_subspace_dimension(space)

    def test_the_torus_has_four(self) -> None:
        """Three translations and the constant lapse, matching :mod:`adm`."""
        assert pc.kid_dimension(by_label("G1")) == 4
        assert adm.mode_report((0, 0, 0)).kid_dimension == 4

    def test_hantzsche_wendt_has_exactly_one(self) -> None:
        """No spatial symmetry at all, yet still a KID.

        Its holonomy fixes no direction, so there are no Killing vector fields;
        the constant lapse survives because ``Hess N - (Delta N) delta`` vanishes
        for constant ``N`` on any flat slice. This is the case that separates
        "instability needs symmetry" from "instability needs a compact Cauchy
        surface".
        """
        space = by_label("G6")
        assert pc.fixed_subspace_dimension(space) == 0
        assert pc.kid_dimension(space) == 1

    def test_kid_count_decreases_with_symmetry(self) -> None:
        counts = [pc.kid_dimension(s) for s in pc.PLATYCOSMS]
        assert counts == [4, 2, 2, 1]
        assert min(counts) >= 1, "every compact flat slice keeps the lapse KID"


class TestModeOrbits:
    @pytest.mark.parametrize("space", pc.PLATYCOSMS, ids=lambda s: s.label)
    def test_orbit_size_divides_the_group_order(self, space) -> None:
        order = len(pc.point_group(space))
        for wave in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, -1, 3)]:
            assert order % len(pc.mode_orbit(space, wave)) == 0

    def test_the_torus_orbits_are_singletons(self) -> None:
        space = by_label("G1")
        for wave in [(1, 0, 0), (2, -1, 3)]:
            assert pc.mode_orbit(space, wave) == (tuple(wave),)

    def test_the_zero_mode_is_always_fixed(self) -> None:
        for space in pc.PLATYCOSMS:
            assert pc.mode_orbit(space, (0, 0, 0)) == ((0, 0, 0),)

    @pytest.mark.parametrize("space", pc.PLATYCOSMS, ids=lambda s: s.label)
    def test_invariant_modes_cover_the_box_exactly_once(self, space) -> None:
        limit = 2
        representatives = list(pc.invariant_modes(space, limit))
        covered: set[tuple[int, ...]] = set()
        for wave in representatives:
            orbit = set(pc.mode_orbit(space, wave))
            assert not (covered & orbit), "orbits must not overlap"
            covered |= orbit
        span = range(-limit, limit + 1)
        box = {(a, b, c) for a in span for b in span for c in span}
        assert box <= covered


class TestRigidity:
    @pytest.mark.parametrize("space", pc.PLATYCOSMS, ids=lambda s: s.label)
    def test_every_platycosm_is_rigid(self, space) -> None:
        """The obstruction is negative definite modulo gauge at each ``k != 0``.

        Restricting a negative semi-definite form to the invariant subspace keeps
        it negative semi-definite, so one condition already forces any linearised
        solution satisfying it *mode by mode* to be pure gauge. That much
        descends to every quotient regardless of how much symmetry was removed.

        The global condition is weaker, because the homogeneous mode contributes
        a positive direction; see :mod:`graviton` and
        :class:`test_graviton.TestBalance`.
        """
        report = pc.analyse(space)
        assert report.rigid
        assert report.kid_dimension >= 1

    def test_the_torus_reproduces_adm(self) -> None:
        report = pc.analyse(by_label("G1"))
        assert report.torus_inertia == adm.physical_signature((1, 1, 1))
        assert report.torus_inertia == (4, 4, 0)

    def test_rigidity_does_not_require_spatial_symmetry(self) -> None:
        """The substantive point.

        ``G6`` has no Killing vector fields, so nothing for a symmetry group to
        average over, and a single linearisation-stability condition rather than
        four. It is rigid anyway. The three translational KIDs of the torus are
        redundant for rigidity; only the lapse is doing work.
        """
        hw = pc.analyse(by_label("G6"))
        torus = pc.analyse(by_label("G1"))
        assert hw.has_no_spatial_symmetry
        assert not torus.has_no_spatial_symmetry
        assert hw.kid_dimension == 1 < torus.kid_dimension == 4
        assert hw.rigid and torus.rigid

    def test_a_form_with_a_positive_direction_would_not_be_rigid(self) -> None:
        """A negative control on the rigidity criterion itself."""
        report = pc.PlatycosmReport(
            label="x", name="x", point_group_order=1, killing_fields=0,
            kid_dimension=1, betti_one=0, free_action=True,
            torus_inertia=(3, 4, 1),
        )
        assert not report.rigid
