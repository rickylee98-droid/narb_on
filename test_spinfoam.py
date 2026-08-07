"""Tests for the exact recoupling algebra and the Ponzano-Regge limit.

The recoupling coefficients are pinned against identities they cannot fake --
Biedenharn-Elliott, orthogonality, the Regge symmetry -- rather than only against
another implementation of the same Racah formula.  The semiclassical results are
pinned against the Ponzano-Regge formula in both the allowed and the forbidden
regime.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import numpy as np
import pytest
from sympy import N, Rational
from sympy.physics.wigner import wigner_3j as sympy_3j, wigner_6j as sympy_6j

import spinfoam as sf

SPINS = [Fraction(k, 2) for k in range(0, 9)]


# --------------------------------------------------------------------------- #
# Exact arithmetic
# --------------------------------------------------------------------------- #
class TestSurd:
    def test_float_of_a_simple_surd(self) -> None:
        value = sf.Surd(Fraction(3), Fraction(2))
        assert float(value) == pytest.approx(3 * math.sqrt(2))

    def test_negative_rational_keeps_its_sign(self) -> None:
        assert float(sf.Surd(Fraction(-3), Fraction(4))) == pytest.approx(-6.0)

    def test_rejects_a_negative_radicand(self) -> None:
        with pytest.raises(ValueError, match="radicand"):
            sf.Surd(Fraction(1), Fraction(-1))

    def test_square_is_exact_and_rational(self) -> None:
        assert sf.Surd(Fraction(3, 5), Fraction(7, 2)).square() == Fraction(63, 50)

    def test_huge_parts_do_not_overflow(self) -> None:
        """Regression: the parts overflow float long before the value does.

        At ``j = 60`` the rational part and the radicand are separately outside
        float range while the symbol itself is near ``1e-3``; converting each and
        multiplying raised ``OverflowError``.
        """
        for j in (60, 120, 200):
            value = sf.wigner_6j(j, j, j, j, j, j)
            assert math.isfinite(value)
            assert abs(value) < 1.0

    def test_conversion_keeps_full_double_precision(self) -> None:
        for j in (5, 10, 30):
            mine = sf.wigner_6j(j, j, j, j, j, j)
            reference = float(N(sympy_6j(j, j, j, j, j, j), 30))
            assert abs(mine - reference) <= 1e-15 * abs(reference)


class TestTriangleConditions:
    @pytest.mark.parametrize(
        "triple, expected",
        [
            ((1, 1, 1), True),
            ((1, 1, 3), False),
            ((Fraction(1, 2), Fraction(1, 2), 1), True),
            ((Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)), False),
            ((2, 3, 5), True),
            ((2, 3, 6), False),
        ],
    )
    def test_cases(self, triple, expected: bool) -> None:
        assert sf.triangle_holds(*triple) is expected

    def test_half_integer_perimeter_fails(self) -> None:
        """Not a triangle-inequality failure but an integrality one."""
        assert not sf.triangle_holds(1, 1, Fraction(1, 2))

    def test_rejects_a_quarter_spin(self) -> None:
        with pytest.raises(ValueError, match="half-integer"):
            sf.triangle_holds(Fraction(1, 4), 1, 1)


# --------------------------------------------------------------------------- #
# Recoupling coefficients against external statements
# --------------------------------------------------------------------------- #
class TestAgainstSympy:
    def test_6j_matches_where_all_triads_close(self) -> None:
        rng = random.Random(3)
        compared = 0
        for _ in range(4000):
            js = [rng.choice(SPINS) for _ in range(6)]
            triads = [
                (js[0], js[1], js[2]),
                (js[0], js[4], js[5]),
                (js[3], js[1], js[5]),
                (js[3], js[4], js[2]),
            ]
            if not all(sf.triangle_holds(*t) for t in triads):
                continue
            mine = sf.wigner_6j(*js)
            reference = float(N(sympy_6j(*[Rational(x) for x in js]), 25))
            assert abs(mine - reference) < 1e-14
            compared += 1
        assert compared > 30

    def test_6j_vanishes_when_a_triad_fails(self) -> None:
        rng = random.Random(4)
        checked = 0
        for _ in range(2000):
            js = [rng.choice(SPINS) for _ in range(6)]
            triads = [
                (js[0], js[1], js[2]),
                (js[0], js[4], js[5]),
                (js[3], js[1], js[5]),
                (js[3], js[4], js[2]),
            ]
            if all(sf.triangle_holds(*t) for t in triads):
                continue
            assert sf.wigner_6j_exact(*js).is_zero
            checked += 1
        assert checked > 100

    def test_3j_matches(self) -> None:
        rng = random.Random(5)
        compared = 0
        for _ in range(4000):
            j1, j2, j3 = (rng.choice(SPINS) for _ in range(3))
            if not sf.triangle_holds(j1, j2, j3):
                continue
            m1 = -j1 + rng.randrange(int(2 * j1) + 1)
            m2 = -j2 + rng.randrange(int(2 * j2) + 1)
            m3 = -m1 - m2
            if abs(m3) > j3 or (Fraction(j3) - m3).denominator != 1:
                continue
            mine = sf.wigner_3j(j1, j2, j3, m1, m2, m3)
            reference = float(
                N(sympy_3j(*[Rational(x) for x in (j1, j2, j3, m1, m2, m3)]), 25)
            )
            assert abs(mine - reference) < 1e-14
            compared += 1
        assert compared > 50


class TestExactIdentities:
    """Identities a wrong 6j implementation cannot satisfy by accident."""

    def test_orthogonality(self) -> None:
        rng = random.Random(11)
        grid = [Fraction(k, 2) for k in range(0, 17)]
        checked = 0
        for _ in range(400):
            a, b, c, d = (rng.choice(SPINS[:7]) for _ in range(4))
            for p in grid:
                if not (sf.triangle_holds(a, d, p) and sf.triangle_holds(c, b, p)):
                    continue
                for q in grid:
                    if not (sf.triangle_holds(a, d, q) and sf.triangle_holds(c, b, q)):
                        continue
                    total = sum(
                        float(2 * x + 1)
                        * sf.wigner_6j(a, b, x, c, d, p)
                        * sf.wigner_6j(a, b, x, c, d, q)
                        for x in grid
                    ) * float(2 * p + 1)
                    assert total == pytest.approx(1.0 if p == q else 0.0, abs=1e-9)
                    checked += 1
                    if checked >= 60:
                        return
        assert checked > 0

    def test_regge_symmetry(self) -> None:
        """A symmetry beyond the 24 tetrahedral ones, and exact to the last bit."""
        rng = random.Random(7)
        checked = 0
        for _ in range(20000):
            js = [rng.choice(SPINS) for _ in range(6)]
            j1, j2, j3, j4, j5, j6 = js
            triads = [
                (j1, j2, j3),
                (j1, j5, j6),
                (j4, j2, j6),
                (j4, j5, j3),
            ]
            if not all(sf.triangle_holds(*t) for t in triads):
                continue
            s = (j2 + j3 + j5 + j6) / 2
            image = (j1, s - j2, s - j3, j4, s - j5, s - j6)
            if any(x < 0 for x in image) or any(
                Fraction(x).denominator > 2 for x in image
            ):
                continue
            assert sf.wigner_6j_exact(*js).square() == sf.wigner_6j_exact(*image).square()
            checked += 1
            if checked >= 120:
                break
        assert checked >= 50

    def test_biedenharn_elliott_pentagon(self) -> None:
        """The deep identity: a buggy Racah sum fails it immediately."""
        rng = random.Random(5)
        grid = [Fraction(k, 2) for k in range(0, 25)]
        small = [Fraction(k, 2) for k in range(0, 6)]
        checked = 0
        for _ in range(400000):
            a, b, c, d, e, f, p, q, r = (rng.choice(small) for _ in range(9))
            total = 0.0
            for x in grid:
                first = sf.wigner_6j(a, b, x, c, d, p)
                if first == 0.0:
                    continue
                second = sf.wigner_6j(c, d, x, e, f, q)
                if second == 0.0:
                    continue
                third = sf.wigner_6j(e, f, x, b, a, r)
                if third == 0.0:
                    continue
                phase = a + b + c + d + e + f + p + q + r + x
                if Fraction(phase).denominator != 1:
                    continue
                total += ((-1) ** int(phase)) * float(2 * x + 1) * first * second * third
            right = sf.wigner_6j(p, q, r, e, a, d) * sf.wigner_6j(p, q, r, f, b, c)
            if total == 0.0 and right == 0.0:
                continue
            assert total == pytest.approx(right, abs=1e-10)
            checked += 1
            if checked >= 40:
                break
        assert checked >= 20


# --------------------------------------------------------------------------- #
# The tetrahedron, and the bug the edge map hides
# --------------------------------------------------------------------------- #
class TestEdgeMap:
    def test_every_triad_is_a_face(self) -> None:
        """The structural check that catches a wrong edge assignment.

        The four triads of a ``6j`` symbol must be the four triangular faces.
        The natural-looking alternative puts ``j1, j2, j3`` on the edges meeting
        at one vertex -- a vertex star, not a face -- and that assignment
        reproduces correct asymptotics on every symmetric test case, failing only
        on a shape like ``(2,2,2,3,3,3)``.
        """
        faces = {
            frozenset(
                i for i, pair in enumerate(sf.SIXJ_EDGE_VERTICES) if vertex not in pair
            )
            for vertex in range(4)
        }
        triads = {
            frozenset((0, 1, 2)),
            frozenset((0, 4, 5)),
            frozenset((3, 1, 5)),
            frozenset((3, 4, 2)),
        }
        assert faces == triads

    def test_opposite_pairs_share_no_vertex(self) -> None:
        pairs = [(0, 3), (1, 4), (2, 5)]
        for a, b in pairs:
            assert not set(sf.SIXJ_EDGE_VERTICES[a]) & set(sf.SIXJ_EDGE_VERTICES[b])

    def test_every_vertex_pair_is_used_once(self) -> None:
        assert len({frozenset(p) for p in sf.SIXJ_EDGE_VERTICES}) == 6


class TestGeometry:
    def test_regular_tetrahedron_volume(self) -> None:
        """Six equal spins give the regular tetrahedron of edge ``j + 1/2``."""
        j = 10
        geometry = sf.tetrahedron_geometry(j, j, j, j, j, j)
        edge = j + 0.5
        assert geometry.volume == pytest.approx(edge**3 / (6 * math.sqrt(2)))

    def test_regular_tetrahedron_dihedral_angles(self) -> None:
        geometry = sf.tetrahedron_geometry(5, 5, 5, 5, 5, 5)
        expected = math.pi - math.acos(1 / 3)
        assert geometry.dihedral_angles is not None
        for angle in geometry.dihedral_angles:
            assert angle == pytest.approx(expected)

    def test_forbidden_shape_has_negative_volume_squared(self) -> None:
        geometry = sf.tetrahedron_geometry(1, 3, 4, 2, 2, 2)
        assert not geometry.is_euclidean
        assert geometry.volume_squared < 0
        with pytest.raises(ValueError, match="not bound a Euclidean"):
            _ = geometry.volume

    def test_asymmetric_shape_is_orientation_sensitive(self) -> None:
        """(2,2,2,3,3,3) and its swap are genuinely different tetrahedra.

        This is the pair that distinguishes the correct edge map from the
        vertex-star one; on symmetric shapes the two agree.
        """
        one = sf.tetrahedron_geometry(2, 2, 2, 3, 3, 3)
        other = sf.tetrahedron_geometry(3, 3, 3, 2, 2, 2)
        assert one.volume != pytest.approx(other.volume)

    def test_rejects_wrong_edge_count(self) -> None:
        with pytest.raises(ValueError, match="six edges"):
            sf.cayley_menger_volume_squared([1.0, 1.0, 1.0])


# --------------------------------------------------------------------------- #
# The semiclassical limit
# --------------------------------------------------------------------------- #
class TestPonzanoRegge:
    @pytest.mark.parametrize("j", [20, 40, 80])
    def test_regular_tetrahedron_is_reproduced(self, j: int) -> None:
        comparison = sf.compare_to_ponzano_regge(j, j, j, j, j, j)
        envelope = math.sqrt(
            12 * math.pi * sf.tetrahedron_geometry(j, j, j, j, j, j).volume
        )
        assert abs(comparison.exact - comparison.predicted) * envelope < 0.2

    def test_regge_action_uses_exterior_angles(self) -> None:
        """Interior angles give a different action and visibly worse agreement."""
        j = 40
        geometry = sf.tetrahedron_geometry(j, j, j, j, j, j)
        action = sf.regge_action(geometry)
        interior = sum(
            l * (math.pi - t) for l, t in zip(geometry.lengths, geometry.dihedral_angles)
        )
        exact = sf.wigner_6j(j, j, j, j, j, j)
        good = math.cos(action + math.pi / 4) / math.sqrt(
            12 * math.pi * geometry.volume
        )
        bad = math.cos(interior + math.pi / 4) / math.sqrt(
            12 * math.pi * geometry.volume
        )
        assert abs(exact - good) < abs(exact - bad)

    def test_forbidden_regime_raises(self) -> None:
        with pytest.raises(ValueError, match="classically forbidden"):
            sf.ponzano_regge(1, 3, 4, 2, 2, 2)

    def test_forbidden_regime_decays_exponentially(self) -> None:
        """The other half of Ponzano-Regge: damping, and no oscillation."""
        shape = (1, 3, 4, 2, 2, 2)
        scales, logs = [], []
        for scale in range(1, 26):
            value = sf.wigner_6j(*[scale * s for s in shape])
            if value == 0.0:
                continue
            scales.append(scale)
            logs.append(math.log(abs(value)))
        x = np.array(scales, dtype=float)
        y = np.array(logs)
        mask = x >= 8
        exponential = np.polyfit(x[mask], y[mask], 1)
        power = np.polyfit(np.log(x[mask]), y[mask], 1)
        residual_exp = np.std(y[mask] - np.polyval(exponential, x[mask]))
        residual_pow = np.std(y[mask] - np.polyval(power, np.log(x[mask])))
        assert residual_exp < 0.25 * residual_pow

    def test_forbidden_regime_does_not_oscillate(self) -> None:
        shape = (1, 3, 4, 2, 2, 2)
        values = [sf.wigner_6j(*[s * x for x in shape]) for s in range(4, 25)]
        signs = {v > 0 for v in values if v != 0.0}
        assert len(signs) == 1, "an evanescent regime must not change sign"

    def test_comparison_reports_the_forbidden_case(self) -> None:
        comparison = sf.compare_to_ponzano_regge(1, 3, 4, 2, 2, 2)
        assert not comparison.classically_allowed
        assert comparison.predicted is None
        assert comparison.absolute_error is None


class TestConvergenceRate:
    """The rate of approach to classical geometry, and the trap in measuring it."""

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1, 1, 1, 1, 1),
            (2, 2, 2, 3, 3, 3),
            (3, 3, 3, 2, 2, 2),
            (2, 3, 4, 4, 3, 2),
            (2, 3, 3, 4, 4, 5),
        ],
    )
    def test_exponent_is_minus_one(self, shape) -> None:
        """Ponzano-Regge leads an asymptotic series in 1/j, so the rate is j^-1."""
        measurement = sf.convergence_exponent(shape, max_scale=120)
        assert measurement.is_trustworthy
        assert -1.35 < measurement.exponent < -0.85

    def test_aliased_shape_is_flagged_not_reported(self) -> None:
        """The trap: spins are quantised, so the sampling can beat with the phase.

        For ``(3,5,4,2,4,6)`` the Regge phase advances 6.997 cycles per unit
        scaling, so integer sampling is nearly phase-locked and the apparent
        exponent is ~0.  That is a sampling artefact, and it must be reported as
        untrustworthy rather than as an absence of convergence.
        """
        measurement = sf.convergence_exponent((3, 5, 4, 2, 4, 6), max_scale=120)
        assert not measurement.is_trustworthy
        assert measurement.diagnostic.beat_period > 100

    def test_well_sampled_shapes_have_short_beats(self) -> None:
        for shape in [(1, 1, 1, 1, 1, 1), (2, 2, 2, 3, 3, 3), (2, 3, 4, 4, 3, 2)]:
            diagnostic = sf.aliasing_diagnostic(shape)
            assert not diagnostic.is_aliased
            assert diagnostic.beat_period < 20

    def test_forbidden_shape_cannot_be_measured(self) -> None:
        with pytest.raises(ValueError, match="forbidden"):
            sf.convergence_exponent((1, 3, 4, 2, 2, 2))

    def test_rejects_a_malformed_shape(self) -> None:
        with pytest.raises(ValueError, match="six spins"):
            sf.aliasing_diagnostic((1, 1, 1))
