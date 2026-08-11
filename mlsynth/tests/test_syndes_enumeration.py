"""Enumerating treated sets inside the restrictions, not around them.

The exact two-way backend used to generate every ``C(N, K)`` subset and test each
one against the restrictions. That made the enumeration limit a statement about
the unrestricted problem: an instance with a handful of admissible designs was
refused whenever ``C(N, K)`` was large, because the cost of reaching those designs
was the cost of walking past all the others.

Three of the restrictions are structural and can be honoured while candidates are
built. Forced units are in every design, forbidden units are in none, and a
stratum quota bounds how many members of a disjoint group a design may hold. What
survives is a product of per-group choices, which this module both counts and
walks.

The remaining restrictions -- conflict pairs, a cost budget, and the pool's
no-good sets -- stay predicates on a finished candidate. They only ever remove
designs, so the count here is an upper bound on the admissible designs and an
exact count of what the walk emits.

The contract these tests pin:

* the walk emits exactly ``size`` subsets, and ``size`` is computed without
  walking;
* the walk emits exactly the subsets a brute-force filter would keep;
* an unrestricted space reproduces ``combinations(range(N), K)`` verbatim,
  including order, so removing the old path changes no existing result;
* an unsatisfiable restriction set gives ``size == 0`` and an empty walk, which
  is how the search reports that no design exists.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import pytest
from hypothesis import given, settings, strategies as st

from mlsynth.utils.syndes_helpers.enumeration import SearchSpace


# ----------------------------------------------------------------------
# brute-force reference: what the old generate-then-filter path kept
# ----------------------------------------------------------------------


def _structural(subset, forced, forbidden, strata):
    """Whether ``subset`` satisfies the three structural restrictions."""
    chosen = set(subset)
    if not set(forced) <= chosen:
        return False
    if chosen & set(forbidden):
        return False
    for members, lo, hi in strata:
        inside = len(chosen.intersection(members))
        if lo is not None and inside < lo:
            return False
        if hi is not None and inside > hi:
            return False
    return True


def _brute(N, K, forced=(), forbidden=(), strata=()):
    """Every admissible size-``K`` subset, by exhaustive test."""
    return {s for s in combinations(range(N), K)
            if _structural(s, forced, forbidden, strata)}


def _walk(space):
    return [tuple(s) for s in space.iter_subsets()]


# ----------------------------------------------------------------------
# smoke
# ----------------------------------------------------------------------


class TestSmoke:
    def test_builds_and_walks(self):
        space = SearchSpace.build(5, 2)
        walked = _walk(space)
        assert space.size == comb(5, 2)
        assert len(walked) == comb(5, 2)
        assert all(len(s) == 2 for s in walked)

    def test_forced_unit_appears_in_every_design(self):
        space = SearchSpace.build(6, 3, forced=(1,))
        assert all(1 in s for s in _walk(space))

    def test_forbidden_unit_appears_in_none(self):
        space = SearchSpace.build(6, 3, forbidden=(4,))
        assert all(4 not in s for s in _walk(space))


# ----------------------------------------------------------------------
# the count is exact, and computed without walking
# ----------------------------------------------------------------------


class TestSizeIsExact:
    CASES = [
        dict(N=6, K=3),
        dict(N=6, K=3, forced=(0,)),
        dict(N=6, K=3, forced=(0, 5)),
        dict(N=7, K=3, forbidden=(1, 2)),
        dict(N=7, K=3, forced=(0,), forbidden=(6,)),
        dict(N=8, K=4, strata=(((0, 1, 2), 1, 2), ((3, 4, 5), None, 1))),
        dict(N=8, K=4, forced=(0,), strata=(((0, 1, 2), 1, 2),)),
        dict(N=8, K=3, forbidden=(0,), strata=(((0, 1, 2), None, 1),)),
        dict(N=7, K=3, strata=(((4, 5, 6), None, 2),)),
        dict(N=9, K=4, strata=(((5, 6), None, 1), ((0, 1), 1, None))),
    ]

    @pytest.mark.parametrize("case", CASES)
    def test_size_equals_walk_length(self, case):
        space = SearchSpace.build(**case)
        assert space.size == len(_walk(space))

    @pytest.mark.parametrize("case", CASES)
    def test_walk_equals_brute_force(self, case):
        space = SearchSpace.build(**case)
        assert set(_walk(space)) == _brute(**case)

    @pytest.mark.parametrize("case", CASES)
    def test_walk_has_no_duplicates(self, case):
        walked = _walk(space := SearchSpace.build(**case))
        assert len(walked) == len(set(walked)) == space.size

    @pytest.mark.parametrize("case", CASES)
    def test_every_subset_is_sorted_and_sized(self, case):
        K = case["K"]
        for s in _walk(SearchSpace.build(**case)):
            assert list(s) == sorted(s)
            assert len(s) == K
            assert len(set(s)) == K

    def test_size_never_exceeds_the_unrestricted_count(self):
        for case in self.CASES:
            assert SearchSpace.build(**case).size <= comb(case["N"], case["K"])


# ----------------------------------------------------------------------
# the unrestricted space is the old path, verbatim
# ----------------------------------------------------------------------


class TestUnrestrictedIsUnchanged:
    @pytest.mark.parametrize("N,K", [(5, 2), (6, 3), (7, 1), (7, 6)])
    def test_matches_combinations_including_order(self, N, K):
        """No restrictions must reproduce the previous generator exactly.

        Order is part of this: the search breaks ties by which design it meets
        first, so a reordering would be free to move a reported design.
        """
        assert _walk(SearchSpace.build(N, K)) == list(combinations(range(N), K))

    def test_size_is_the_binomial(self):
        assert SearchSpace.build(9, 4).size == comb(9, 4)


# ----------------------------------------------------------------------
# restrictions shrink the space, which is the point
# ----------------------------------------------------------------------


class TestRestrictionsShrinkTheSpace:
    def test_forced_units_collapse_the_ground_set(self):
        """``C(200, 6)`` is past any enumeration limit; forcing five is not."""
        space = SearchSpace.build(200, 6, forced=(0, 1, 2, 3, 4))
        assert space.size == 195
        assert comb(200, 6) > 8e10

    def test_forbidding_most_units_collapses_the_ground_set(self):
        space = SearchSpace.build(200, 6, forbidden=tuple(range(10, 200)))
        assert space.size == comb(10, 6)

    def test_a_full_quota_of_forced_units_leaves_one_design(self):
        space = SearchSpace.build(50, 3, forced=(7, 11, 13))
        assert space.size == 1
        assert _walk(space) == [(7, 11, 13)]

    def test_stratum_ceiling_removes_designs(self):
        capped = SearchSpace.build(8, 4, strata=(((0, 1, 2, 3), None, 1),))
        assert capped.size < comb(8, 4)
        assert set(_walk(capped)) == _brute(8, 4, strata=(((0, 1, 2, 3), None, 1),))


# ----------------------------------------------------------------------
# edge cases
# ----------------------------------------------------------------------


class TestEdges:
    @pytest.mark.parametrize("K", [1, 7])
    def test_extreme_k(self, K):
        space = SearchSpace.build(8, K)
        assert space.size == comb(8, K) == len(_walk(space))

    def test_k_equal_to_the_forced_count(self):
        space = SearchSpace.build(6, 2, forced=(1, 4))
        assert _walk(space) == [(1, 4)]

    def test_stratum_with_no_selectable_member(self):
        """Every member forbidden: admissible only when the floor is zero."""
        strata = (((0, 1), None, 2),)
        space = SearchSpace.build(6, 2, forbidden=(0, 1), strata=strata)
        assert set(_walk(space)) == _brute(6, 2, forbidden=(0, 1), strata=strata)

    def test_open_ended_quotas(self):
        """``None`` bounds are no bound, not a bound of zero."""
        strata = (((0, 1, 2), None, None),)
        space = SearchSpace.build(6, 3, strata=strata)
        assert space.size == comb(6, 3)

    def test_forced_unit_inside_a_capped_stratum_consumes_quota(self):
        strata = (((0, 1, 2), None, 1),)
        space = SearchSpace.build(6, 3, forced=(0,), strata=strata)
        assert set(_walk(space)) == _brute(6, 3, forced=(0,), strata=strata)
        assert all(len(set(s) & {0, 1, 2}) == 1 for s in _walk(space))

    def test_groups_out_of_order_still_emit_sorted_sets(self):
        """The leftover group lands last whatever it holds.

        A stratum over the high units leaves the low ones to the leftover group,
        so concatenating one pick per group runs downhill. Treating that
        concatenation as sorted would emit tuples the search reads as a treated
        set and the design would name the wrong units.
        """
        strata = (((4, 5, 6), None, 2),)
        walked = _walk(SearchSpace.build(7, 3, strata=strata))
        assert all(list(s) == sorted(s) for s in walked)
        assert set(walked) == _brute(7, 3, strata=strata)

    def test_one_stratum_covering_every_unit(self):
        """A single group with no leftover, so a quota is the only thing bounding it.

        The unrestricted space is also a single group, and it is the space the
        walk is specialised for. A specialisation that read ``k_free`` without
        checking the quota would still be correct there and wrong here.
        """
        strata = (((0, 1, 2, 3, 4, 5), 2, 3),)
        space = SearchSpace.build(6, 3, strata=strata)
        assert set(_walk(space)) == _brute(6, 3, strata=strata)
        assert space.size == len(_walk(space)) == comb(6, 3)

    def test_one_stratum_whose_ceiling_excludes_k(self):
        strata = (((0, 1, 2, 3, 4, 5), None, 2),)
        space = SearchSpace.build(6, 3, strata=strata)
        assert space.size == 0 and _walk(space) == []

    def test_one_stratum_whose_floor_excludes_k(self):
        strata = (((0, 1, 2, 3, 4, 5), 4, None),)
        space = SearchSpace.build(6, 3, strata=strata)
        assert space.size == 0 and _walk(space) == []

    def test_overlapping_strata_still_enumerate_correctly(self):
        """Disjointness is exploited when present and not assumed when absent.

        Restrictions built by ``design_restrictions`` key each unit to one
        stratum, so the groups partition the units. A caller may hand over
        overlapping sets, and the walk must stay correct there -- covering a
        superset and leaving the rest to the candidate test.
        """
        strata = (((0, 1, 2), None, 1), ((2, 3, 4), 1, None))
        space = SearchSpace.build(7, 3, strata=strata)
        walked = _walk(space)
        assert set(walked) >= _brute(7, 3, strata=strata)
        assert space.size == len(walked)
        # Treating overlapping sets as a partition would let unit 2 be drawn by
        # both groups, emitting a repeated unit and the same design twice.
        assert len(walked) == len(set(walked))
        assert all(len(set(s)) == 3 for s in walked)


# ----------------------------------------------------------------------
# unsatisfiable restrictions: an empty space, not an exception
# ----------------------------------------------------------------------


class TestUnsatisfiable:
    """The search reports "no admissible design" from a count of zero.

    Raising here would move that message, so an impossible space is a space of
    size zero.
    """

    def test_forced_and_forbidden_collide(self):
        space = SearchSpace.build(6, 2, forced=(1,), forbidden=(1,))
        assert space.size == 0 and _walk(space) == []

    def test_more_forced_than_k(self):
        space = SearchSpace.build(6, 2, forced=(0, 1, 2))
        assert space.size == 0 and _walk(space) == []

    def test_too_few_units_left(self):
        space = SearchSpace.build(6, 4, forbidden=(0, 1, 2, 3))
        assert space.size == 0 and _walk(space) == []

    def test_stratum_floor_exceeds_k(self):
        space = SearchSpace.build(6, 2, strata=(((0, 1, 2, 3), 3, None),))
        assert space.size == 0 and _walk(space) == []

    def test_stratum_ceilings_cannot_reach_k(self):
        """Two groups capped at one apiece cover every unit, so K=3 is impossible."""
        strata = (((0, 1, 2), None, 1), ((3, 4, 5), None, 1))
        space = SearchSpace.build(6, 3, strata=strata)
        assert space.size == 0 and _walk(space) == []

    def test_forced_unit_breaks_a_stratum_ceiling(self):
        space = SearchSpace.build(6, 3, forced=(0, 1), strata=(((0, 1, 2), None, 1),))
        assert space.size == 0 and _walk(space) == []


# ----------------------------------------------------------------------
# properties
# ----------------------------------------------------------------------


_UNITS = st.integers(min_value=0, max_value=7)


@st.composite
def _instance(draw):
    N = draw(st.integers(min_value=3, max_value=8))
    K = draw(st.integers(min_value=1, max_value=N - 1))
    forced = draw(st.frozensets(st.integers(0, N - 1), max_size=2))
    forbidden = draw(st.frozensets(st.integers(0, N - 1), max_size=2))
    members = draw(st.frozensets(st.integers(0, N - 1), min_size=1, max_size=4))
    lo = draw(st.one_of(st.none(), st.integers(0, 2)))
    hi = draw(st.one_of(st.none(), st.integers(0, 3)))
    use_stratum = draw(st.booleans())
    strata = ((tuple(sorted(members)), lo, hi),) if use_stratum else ()
    return dict(N=N, K=K, forced=tuple(sorted(forced)),
                forbidden=tuple(sorted(forbidden)), strata=strata)


class TestProperties:
    @settings(max_examples=250, deadline=None)
    @given(case=_instance())
    def test_walk_equals_brute_force(self, case):
        """The only property that matters: the same designs, however reached."""
        assert set(_walk(SearchSpace.build(**case))) == _brute(**case)

    @settings(max_examples=250, deadline=None)
    @given(case=_instance())
    def test_size_counts_the_walk(self, case):
        space = SearchSpace.build(**case)
        assert space.size == len(_walk(space))

    @settings(max_examples=250, deadline=None)
    @given(case=_instance())
    def test_size_bounds_the_admissible_count(self, case):
        """A safe gate needs the count to bound what the restrictions allow."""
        space = SearchSpace.build(**case)
        assert space.size >= len(_brute(**case))
        assert space.size <= comb(case["N"], case["K"])

    @settings(max_examples=200, deadline=None)
    @given(case=_instance())
    def test_every_emitted_subset_is_a_valid_design(self, case):
        for s in _walk(SearchSpace.build(**case)):
            assert len(s) == case["K"]
            assert list(s) == sorted(set(s))
            assert set(case["forced"]) <= set(s)
            assert not set(s) & set(case["forbidden"])
