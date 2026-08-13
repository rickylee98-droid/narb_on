"""Tests for Shioda's combinatorial reduction of the Hodge Conjecture.

Three independent referees, none of which restates the module's own machinery:

* Shioda's Hodge set ``B^n_m``, enumerated by brute force over characters, must
  land inside the semigroup ``M_m`` at the correct height. Two constructions
  that share no code.
* For Fermat *surfaces* the count ``|B^2_m|`` must equal the Picard number minus
  one, and those Picard numbers are classical -- the Fermat quartic is the
  maximal K3 with rho = 20, the cubic surface has rho = 7.
* ``phi(m)`` must reproduce every published value for ``20 <= m <= 43``. That
  agreement is what licenses the values computed outside the range.
"""

from __future__ import annotations

import pytest

import fermat_hodge as fh

pytest.importorskip("PyNormaliz", reason="Hilbert basis computation needs PyNormaliz")

#: Picard numbers of the Fermat surfaces X^2_m, classical.
FERMAT_SURFACE_PICARD = {3: 7, 4: 20, 5: 37, 6: 86, 7: 91}

#: Published phi values cheap enough to recompute in the default suite.
FAST_PUBLISHED = [20, 21, 22, 23, 25, 26, 27, 29, 31, 32, 33, 34, 35, 37, 39, 41, 43]


# --------------------------------------------------------------------------- #
# Arithmetic helpers
# --------------------------------------------------------------------------- #
class TestUnits:
    def test_units_of_a_prime_are_everything_below_it(self) -> None:
        assert fh.units(7) == (1, 2, 3, 4, 5, 6)

    def test_units_of_a_prime_power(self) -> None:
        assert fh.units(9) == (1, 2, 4, 5, 7, 8)

    def test_rejects_m_below_two(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            fh.units(1)

    def test_representative_sum_is_exact_integers(self) -> None:
        # alpha = (1,1,1) mod 3: each representative is 1, so the sum is 3
        assert fh.representative_sum((1, 1, 1), 3) == 3

    def test_representative_sum_respects_the_twist(self) -> None:
        assert fh.representative_sum((1, 2), 5, t=2) == 2 + 4
        assert fh.representative_sum((1, 2), 5, t=3) == 3 + 1

    def test_rejects_an_entry_killed_by_the_twist(self) -> None:
        with pytest.raises(ValueError, match="zero-divisor"):
            fh.representative_sum((2, 1), 4, t=2)


class TestPrimePower:
    @pytest.mark.parametrize(
        "m,expected",
        [(2, (2, 1)), (4, (2, 2)), (8, (2, 3)), (9, (3, 2)), (49, (7, 2)),
         (343, (7, 3)), (13, (13, 1))],
    )
    def test_recognises_prime_powers(self, m, expected) -> None:
        assert fh.prime_power(m) == expected

    @pytest.mark.parametrize("m", [6, 12, 20, 33, 42, 100])
    def test_rejects_non_prime_powers(self, m) -> None:
        assert fh.prime_power(m) is None


class TestConjecture:
    def test_matches_the_published_statement(self) -> None:
        assert fh.conjectured_phi(25) == 3      # (5+1)/2
        assert fh.conjectured_phi(27) == 5      # (9+1)/2
        assert fh.conjectured_phi(32) == 9      # 2^3+1
        assert fh.conjectured_phi(343) == 25    # (49+1)/2

    def test_makes_no_prediction_off_prime_powers(self) -> None:
        for m in (6, 20, 33, 42):
            assert fh.conjectured_phi(m) is None

    def test_makes_no_prediction_for_small_powers_of_two(self) -> None:
        """The published statement is restricted to ``l > 2``."""
        assert fh.conjectured_phi(2) is None
        assert fh.conjectured_phi(4) is None
        assert fh.conjectured_phi(8) == 3


# --------------------------------------------------------------------------- #
# Referee 1: Hodge elements land in the semigroup
# --------------------------------------------------------------------------- #
class TestHodgeElements:
    @pytest.mark.parametrize("m,n", [(3, 2), (4, 2), (5, 2), (6, 2), (4, 4)])
    def test_every_hodge_element_maps_into_the_semigroup(self, m, n) -> None:
        """Two constructions sharing no code must agree.

        ``B^n_m`` is enumerated from Shioda's character criterion; ``M_m`` is cut
        out by a different set of linear equations. The multiplicity map has to
        carry one into the other, at height exactly ``n/2 + 1``.
        """
        height = n // 2 + 1
        elements = list(fh.hodge_elements(m, n))
        assert elements
        for alpha in elements:
            vector = fh.multiplicity_vector(alpha, m)
            assert fh.in_semigroup(vector, m)
            assert vector[-1] == height

    @pytest.mark.parametrize("m,n", [(3, 2), (5, 2), (4, 4)])
    def test_elements_satisfy_the_defining_conditions(self, m, n) -> None:
        target = m * (n // 2 + 1)
        for alpha in fh.hodge_elements(m, n):
            assert len(alpha) == n + 2
            assert all(a != 0 for a in alpha)
            assert sum(alpha) % m == 0
            for t in fh.units(m):
                assert fh.representative_sum(alpha, m, t) == target

    def test_the_unit_quantifier_is_not_vacuous(self) -> None:
        """Dropping the quantifier over units admits strictly more elements.

        The condition at ``t = 1`` alone is Hodge *type*; requiring it at every
        unit is rationality. If the two agreed the criterion would be doing no
        work.
        """
        m, n = 7, 2
        target = m * (n // 2 + 1)
        type_only = 0
        for head in fh.product(range(1, m), repeat=n + 1):
            last = (-sum(head)) % m
            if last == 0:
                continue
            alpha = head + (last,)
            if fh.representative_sum(alpha, m, 1) == target:
                type_only += 1
        assert type_only > len(list(fh.hodge_elements(m, n)))

    def test_rejects_odd_dimension(self) -> None:
        with pytest.raises(ValueError, match="even"):
            list(fh.hodge_elements(5, 3))


# --------------------------------------------------------------------------- #
# Referee 2: Picard numbers of Fermat surfaces
# --------------------------------------------------------------------------- #
class TestPicardNumbers:
    @pytest.mark.parametrize("m,picard", sorted(FERMAT_SURFACE_PICARD.items()))
    def test_hodge_count_is_the_picard_number_less_one(self, m, picard) -> None:
        """An anchor outside this module entirely.

        For a surface the middle Hodge classes are the Picard group, and the
        primitive part drops the hyperplane class. The Fermat quartic is the
        maximal K3 with ``rho = 20``; the cubic surface, a blow-up of the plane
        at six points, has ``rho = 7``.
        """
        assert len(list(fh.hodge_elements(m, 2))) == picard - 1


# --------------------------------------------------------------------------- #
# Referee 3: the published table
# --------------------------------------------------------------------------- #
class TestPublishedTable:
    @pytest.mark.parametrize("m", FAST_PUBLISHED)
    def test_phi_reproduces_the_published_value(self, m) -> None:
        assert fh.phi(m) == fh.PUBLISHED_PHI[m]

    def test_the_published_table_covers_twenty_to_forty_three(self) -> None:
        assert set(fh.PUBLISHED_PHI) == set(range(20, 44))

    def test_primes_have_phi_one(self) -> None:
        """Shioda's theorem: for prime ``m`` every indecomposable sits at ``y = 1``.

        Equivalently the Hodge Conjecture holds for ``X^n_m`` at all ``n``, which
        is the classical prime-degree case.
        """
        for m in (23, 29, 31, 37, 41, 43, 47):
            assert fh.phi(m) == 1

    def test_report_flags_agreement_and_range(self) -> None:
        report = fh.phi_report(25)
        assert report.matches_published is True
        assert report.matches_conjecture is True
        assert not report.beyond_published_range
        assert report.dimensions_to_check == 2 * (report.phi - 1)


# --------------------------------------------------------------------------- #
# Beyond the published range
# --------------------------------------------------------------------------- #
class TestBeyondPublished:
    """The conjecture at prime powers it was never tested against.

    It was formulated from data with ``m < 48``, where the available prime powers
    were ``4, 8, 9, 16, 25, 27, 32``. These are outside that range.
    """

    @pytest.mark.parametrize(
        "m,expected", sorted(fh.VERIFIED_PRIME_POWERS.items())[:5]
    )
    def test_conjecture_holds_at_new_prime_powers(self, m, expected) -> None:
        report = fh.phi_report(m)
        assert report.beyond_published_range
        assert report.phi == expected
        assert report.matches_conjecture is True

    def test_the_recorded_values_agree_with_the_conjecture(self) -> None:
        for m, value in fh.VERIFIED_PRIME_POWERS.items():
            assert fh.conjectured_phi(m) == value

    def test_the_recorded_values_span_several_primes_and_exponents(self) -> None:
        """A single prime, or a single exponent, would be weak evidence."""
        factored = [fh.prime_power(m) for m in fh.VERIFIED_PRIME_POWERS]
        assert len({p for p, _ in factored}) >= 5
        assert len({k for _, k in factored}) >= 3


class TestSemigroup:
    @pytest.mark.parametrize("m", [5, 7, 8, 9, 12])
    def test_equations_have_one_row_per_unit(self, m) -> None:
        assert len(fh.semigroup_equations(m)) == len(fh.units(m))

    @pytest.mark.parametrize("m", [5, 7, 9])
    def test_hilbert_basis_elements_are_in_the_semigroup(self, m) -> None:
        for vector in fh.hilbert_basis(m):
            assert fh.in_semigroup(list(vector), m)

    @pytest.mark.parametrize("m", [5, 7, 9, 16])
    def test_basis_elements_are_indecomposable(self, m) -> None:
        """The defining property, checked directly rather than trusted."""
        basis = fh.hilbert_basis(m)
        for vector in basis:
            assert not fh.is_decomposable(list(vector), m, basis)

    def test_zero_height_forces_the_zero_vector(self) -> None:
        """Why no ``y > 0`` inequality is needed: the cone is already pointed."""
        m = 7
        assert fh.in_semigroup([0] * m, m)
        for i in range(m - 1):
            probe = [0] * m
            probe[i] = 1
            assert not fh.in_semigroup(probe, m)

    def test_rejects_a_wrong_length_vector(self) -> None:
        with pytest.raises(ValueError, match="length"):
            fh.in_semigroup([1, 2], 7)
