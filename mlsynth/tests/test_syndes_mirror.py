"""Tests for the two-way design's mirror rule (#838).

Contract. The two-way objective is unchanged when treatment and control trade
places: ``u -> -u`` maps the designs that treat ``S`` onto those that treat its
complement, so ``f(S) = f(S^c)``. When every restriction admits a treated set
exactly when it admits the complement (:meth:`SubsetFilter.is_mirrored`), the
exact backend searches half of the designs:

- with ``K=None``, the sizes ``1 .. N // 2`` only, so the design returned is the
  one with fewer treated units;
- at ``K = N / 2``, the designs that treat :data:`MIRROR_UNIT` only;
- the pool at ``K = N / 2`` excludes a design's complement with the design.

A restriction that names a side (forced and forbidden units, conflict pairs, a
budget, a quota that does not map to itself, an excluded set whose complement is
admitted) turns the rule off, and the restricted optimum is found as before.

The levels:

- a design and its complement score the same, with the weights exchanged (unit,
  property);
- the filter reads as mirrored only when it admits a set exactly when it admits
  the complement (unit, property);
- with ``K=None`` only the lower half of the sizes is searched and the optimum
  is found, by the enumeration and by the branch and bound (smoke, property);
- at ``K = N / 2`` the design treats unit 0, and a seed that does not reaches
  the search as its complement (unit, edge);
- every asymmetric restriction turns the rule off, on an instance where leaving
  it on returns a worse design (edge);
- restrictions no design satisfies still raise (failure).
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.syndes_helpers import exact
from mlsynth.utils.syndes_helpers.branch import RESTARTS, branch_and_bound
from mlsynth.utils.syndes_helpers.exact import (
    MIRROR_UNIT,
    SubsetFilter,
    _settle,
    solve_synthetic_design_exact,
    solve_synthetic_design_exact_pool,
    with_complements,
)
from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem, maxcut_candidates
from mlsynth.utils.syndes_helpers.restrictions import DesignRestrictions

SETTINGS = dict(max_examples=20, deadline=None,
                suppress_health_check=[HealthCheck.too_slow])


def _panel(N=8, T=14, rank=3, noise=0.35, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, rank)) @ rng.standard_normal((rank, N))
            + noise * rng.standard_normal((T, N)))


def _engine(Y, lam=None):
    problem = TwoWayProblem.from_panel(Y, lam=lam)
    return problem, BoundEngine.from_problem(problem)


def _value(Y, S, lam=None):
    problem, engine = _engine(Y, lam)
    return _settle(problem, engine, np.asarray(S, dtype=np.int64))[0]


def _complement(N, S):
    return tuple(u for u in range(N) if u not in set(int(i) for i in S))


def _every_design(Y, filt=None, lam=None):
    """``{S: f(S)}`` over every admissible treated set of every size."""
    problem, engine = _engine(Y, lam)
    filt = filt or SubsetFilter()
    out = {}
    for k in range(1, problem.N):
        for S in combinations(range(problem.N), k):
            if filt.accepts(S):
                out[S] = _settle(problem, engine, np.asarray(S, dtype=np.int64))[0]
    return out


def _solve(Y, K, limit, **kwargs):
    return solve_synthetic_design_exact(Y, K, candidate_limit=limit, **kwargs)


#: The enumeration prices every design; the branch and bound searches all of them.
ENUMERATE, BRANCH = 10 ** 9, 1


# ----------------------------------------------------------------------
# the symmetry
# ----------------------------------------------------------------------


class TestTheSymmetry:
    @given(seed=st.integers(0, 10 ** 6), N=st.integers(3, 9),
           scale=st.sampled_from([1e-3, 1.0, 1e3]), level=st.sampled_from([0.0, 1e4]),
           lam=st.sampled_from([None, 0.05, 10.0]), data=st.data())
    @settings(**SETTINGS)
    def test_a_design_and_its_complement_score_the_same(self, seed, N, scale, level,
                                                        lam, data):
        Y = scale * _panel(N=N, seed=seed) + level
        k = data.draw(st.integers(1, N - 1))
        S = tuple(sorted(data.draw(st.permutations(range(N)))[:k]))
        problem, engine = _engine(Y, lam)
        v, a, b, _ = _settle(problem, engine, np.asarray(S))
        w, c, d, _ = _settle(problem, engine, np.asarray(_complement(N, S)))
        assert w == pytest.approx(v, rel=1e-7, abs=1e-12 * scale ** 2)
        # the treated weights of one are the control weights of the other
        assert np.allclose(c, b, atol=1e-6) and np.allclose(d, a, atol=1e-6)


# ----------------------------------------------------------------------
# the filter
# ----------------------------------------------------------------------


class TestIsMirrored:
    def test_no_restriction_is_mirrored(self):
        assert SubsetFilter().is_mirrored(8)

    @pytest.mark.parametrize("filt", [
        SubsetFilter(forced_in=frozenset({2})),
        SubsetFilter(forbidden=frozenset({2})),
        SubsetFilter(conflict_pairs=((1, 2),)),
        SubsetFilter(costs=np.ones(8), budget=3.0),
        SubsetFilter(strata=(((0, 1, 2), None, 1),)),
        SubsetFilter(strata=(((0, 1, 2), 1, None),)),
        SubsetFilter(forbidden_sets=(frozenset({0, 1, 2, 3}),)),
    ], ids=["forced", "forbidden", "conflict", "budget", "quota-at-most",
            "quota-at-least", "an-excluded-set-alone"])
    def test_a_restriction_that_names_a_side_is_not(self, filt):
        assert not filt.is_mirrored(8)

    @pytest.mark.parametrize("filt", [
        SubsetFilter(strata=(((0, 1, 2, 3), 1, 3),)),
        SubsetFilter(strata=(((0, 1, 2, 3), None, None),)),
        SubsetFilter(strata=(((0, 1, 2), 0, 3),)),
        SubsetFilter(strata=(((0, 1, 2, 3), 2, 2), ((4, 5), 1, 1))),
        SubsetFilter(forbidden_sets=(frozenset({0, 1, 2, 3}), frozenset({4, 5, 6, 7}))),
    ], ids=["quota-1-to-3-of-4", "quota-unbounded", "quota-0-to-3-of-3",
            "two-exact-halves", "an-excluded-set-and-its-complement"])
    def test_a_restriction_that_maps_to_itself_is(self, filt):
        assert filt.is_mirrored(8)

    def test_a_quota_past_the_stratum_reads_as_the_whole_stratum(self):
        """``hi`` above the stratum's size admits what ``hi = |M|`` admits."""
        assert SubsetFilter(strata=(((0, 1, 2), 0, 9),)).is_mirrored(8)
        assert SubsetFilter(strata=(((0, 1, 2), -2, 3),)).is_mirrored(8)

    @given(N=st.integers(2, 6), data=st.data())
    @settings(**SETTINGS)
    def test_a_mirrored_filter_admits_a_set_exactly_when_it_admits_the_complement(
            self, N, data):
        units = st.integers(0, N - 1)
        members = st.lists(units, min_size=1, max_size=N, unique=True).map(tuple)
        bound = st.one_of(st.none(), st.integers(-1, N + 1))
        strata = data.draw(st.lists(st.tuples(members, bound, bound), max_size=2))
        sets = data.draw(st.lists(st.lists(units, min_size=1, max_size=N - 1,
                                           unique=True).map(frozenset), max_size=3))
        if data.draw(st.booleans()):
            sets += [frozenset(range(N)) - s for s in sets]
        filt = SubsetFilter(strata=tuple(strata), forbidden_sets=tuple(sets))
        if not filt.is_mirrored(N):
            return
        for k in range(1, N):
            for S in combinations(range(N), k):
                assert filt.accepts(S) == filt.accepts(_complement(N, S)), (S, filt)


# ----------------------------------------------------------------------
# seeds and their complements
# ----------------------------------------------------------------------


class TestWithComplements:
    def test_half_size_rows_gain_their_complements(self):
        got = with_complements(np.array([[3, 0], [1, 2]]), 4)
        assert got.tolist() == [[0, 3], [1, 2], [1, 2], [0, 3]]

    def test_other_sizes_are_only_sorted(self):
        assert with_complements(np.array([[2, 0, 1]]), 8).tolist() == [[0, 1, 2]]

    @pytest.mark.parametrize("row", [[1, 1], [0, 9], [-1, 2]],
                             ids=["repeated", "past-the-panel", "negative"])
    def test_a_malformed_row_gains_no_complement(self, row):
        assert with_complements(np.array([row, [0, 1]]), 4).tolist() == [
            sorted(row), [0, 1], [2, 3]]

    @given(N=st.integers(2, 12), data=st.data())
    @settings(**SETTINGS)
    def test_each_complement_partitions_the_units_with_its_row(self, N, data):
        k = N // 2 if N % 2 == 0 else data.draw(st.integers(1, N - 1))
        rows = data.draw(st.lists(st.permutations(range(N)).map(lambda p: p[:k]),
                                  min_size=1, max_size=4))
        got = with_complements(np.array(rows), N)
        assert got.shape == ((2 if 2 * k == N else 1) * len(rows), k)
        assert (np.diff(got, axis=1) > 0).all()
        if 2 * k == N:
            for S, C in zip(got[:len(rows)], got[len(rows):]):
                assert set(S.tolist()) | set(C.tolist()) == set(range(N))
                assert not set(S.tolist()) & set(C.tolist())


# ----------------------------------------------------------------------
# every size
# ----------------------------------------------------------------------


class TestEverySize:
    @pytest.mark.parametrize("N", [2, 3, 7, 8])
    def test_only_the_lower_half_of_the_sizes_is_searched(self, N, monkeypatch):
        searched = []
        real = exact._search

        def spy(problem, K, *args, **kwargs):
            searched.append(K)
            return real(problem, K, *args, **kwargs)

        monkeypatch.setattr(exact, "_search", spy)
        solve_synthetic_design_exact(_panel(N=N, seed=1), None)
        assert sorted(searched) == list(range(1, N // 2 + 1))

    @pytest.mark.parametrize("limit", [ENUMERATE, BRANCH], ids=["enumerate", "branch"])
    @pytest.mark.parametrize("seed", [0, 3, 6])
    def test_the_optimum_with_fewer_treated_units(self, seed, limit):
        Y = _panel(seed=seed)
        best = min(_every_design(Y).values())
        got = _solve(Y, None, limit)
        S = tuple(int(i) for i in got.selected_unit_indices)
        assert got.objective_value == pytest.approx(best, rel=1e-9)
        assert len(S) <= 4
        assert len(S) < 4 or MIRROR_UNIT in S
        assert got.raw_results["certified"] is True
        assert got.raw_results["mirrored"] is True

    @given(seed=st.integers(0, 10 ** 6), N=st.integers(3, 8),
           scale=st.sampled_from([1e-3, 1.0, 1e3]))
    @settings(**SETTINGS)
    def test_the_optimum_over_every_size_is_found(self, seed, N, scale):
        Y = scale * _panel(N=N, seed=seed)
        best = min(_every_design(Y).values())
        got = solve_synthetic_design_exact(Y, None)
        S = tuple(int(i) for i in got.selected_unit_indices)
        assert got.objective_value == pytest.approx(best, rel=1e-7)
        assert 2 * len(S) < N or (2 * len(S) == N and MIRROR_UNIT in S)

    def test_the_branch_and_bound_and_the_enumeration_agree(self):
        Y = _panel(N=10, seed=2)
        got = _solve(Y, None, BRANCH)
        want = _solve(Y, None, ENUMERATE)
        assert tuple(got.selected_unit_indices) == tuple(want.selected_unit_indices)
        assert got.raw_results["certified"] is True

    def test_two_units(self):
        got = solve_synthetic_design_exact(_panel(N=2, seed=4), None)
        assert tuple(got.selected_unit_indices) == (MIRROR_UNIT,)


# ----------------------------------------------------------------------
# half the units
# ----------------------------------------------------------------------


class TestHalfTheUnits:
    @pytest.mark.parametrize("limit", [ENUMERATE, BRANCH], ids=["enumerate", "branch"])
    @pytest.mark.parametrize("seed", [0, 1, 5])
    def test_the_design_treats_unit_zero(self, seed, limit):
        Y = _panel(seed=seed)
        best = min(v for S, v in _every_design(Y).items() if len(S) == 4)
        got = _solve(Y, 4, limit)
        assert MIRROR_UNIT in tuple(got.selected_unit_indices)
        assert got.objective_value == pytest.approx(best, rel=1e-9)
        assert got.raw_results["certified"] is True

    def test_half_the_designs_are_searched(self):
        got = _solve(_panel(seed=0), 4, ENUMERATE)
        assert got.raw_results["n_designs"] == comb(7, 3)

    def test_other_sizes_search_every_design(self):
        got = _solve(_panel(seed=0), 3, ENUMERATE)
        assert got.raw_results["n_designs"] == comb(8, 3)
        assert got.raw_results["mirrored"] is True

    def test_a_seed_without_unit_zero_seeds_its_complement(self):
        """At ``K = N / 2`` a seed's complement is a design of the same value,
        the one that treats unit 0. A search stopped at once returns it."""
        N = 20
        Y = _panel(N=N, seed=3)
        problem, engine = _engine(Y)
        seed = np.arange(1, 11)
        filt = SubsetFilter(forced_in=frozenset({MIRROR_UNIT}))
        out = branch_and_bound(problem, engine, 10, filt, incumbent_sets=seed[None, :],
                               restarts=0, work_limit=1e-12)
        assert not out.closed
        assert tuple(out.treated) == _complement(N, seed)
        assert out.value == pytest.approx(_value(Y, seed), rel=1e-9)

    def test_the_swap_searchs_best_design_seeds_its_size_from_either_side(self):
        """The swap search can find the best design only on the side that does
        not treat unit 0; its complement, of the same value, is the incumbent."""
        N = 14
        problem, engine = _engine(_panel(N=N, seed=6))
        found = maxcut_candidates(engine, 7, n_restarts=RESTARTS, seed=0)
        values = [_settle(problem, engine, S)[0] for S in found]
        best = min(values)
        # the instance has power: every candidate at the best value lacks unit 0
        assert all(MIRROR_UNIT not in S for S, v in zip(found, values) if v <= best * (1 + 1e-12))
        filt = SubsetFilter().mirrored_at(N, 7, True)
        out = exact._swap_incumbent(problem, {7: filt}, 1, 0)
        assert MIRROR_UNIT in tuple(out.treated)
        assert out.value == pytest.approx(best, rel=1e-12)

    def test_the_swap_searchs_designs_reach_the_size_as_their_complements(self):
        """With ``K=None`` the swap search seeds every size past the limit; at
        ``N / 2`` the design returned still treats unit 0."""
        Y = _panel(N=12, seed=8)
        got = _solve(Y, None, BRANCH)
        S = tuple(int(i) for i in got.selected_unit_indices)
        assert len(S) < 6 or MIRROR_UNIT in S
        assert got.objective_value == pytest.approx(
            _solve(Y, None, ENUMERATE).objective_value, rel=1e-9)


# ----------------------------------------------------------------------
# restrictions that name a side
# ----------------------------------------------------------------------


def _optimum(Y):
    got = solve_synthetic_design_exact(Y, None)
    return tuple(int(i) for i in got.selected_unit_indices), got.objective_value


def _side_restrictions(S):
    """Restrictions under which the complement of ``S`` is admissible and ``S``
    is not, each naming a side. ``S`` treats unit 0 and half the units, so with
    the rule left on the search would look only among designs that treat unit 0
    and miss the complement."""
    out_ = next(u for u in range(8) if u not in S)
    costs = np.zeros(8)
    costs[list(S)] = 1.0
    return {
        "forbidden": dict(restrictions=DesignRestrictions(forbidden=[S[0]])),
        "forced": dict(restrictions=DesignRestrictions(forced_in=[out_])),
        "conflict": dict(restrictions=DesignRestrictions(conflict_pairs=[(S[0], S[1])])),
        "budget": dict(costs=costs, budget=0.5),
        "quota": dict(restrictions=DesignRestrictions(strata=[(S, None, 0)])),
        "an-excluded-set": dict(forbidden_sets=[np.asarray(S)]),
    }


class TestASideNamedTurnsItOff:
    @pytest.mark.parametrize("limit", [ENUMERATE, BRANCH], ids=["enumerate", "branch"])
    @pytest.mark.parametrize("kind", ["forbidden", "forced", "conflict", "budget",
                                      "quota", "an-excluded-set"])
    @pytest.mark.parametrize("K", [None, 4], ids=["every-size", "half"])
    def test_the_complement_is_the_restricted_optimum(self, kind, K, limit):
        Y = _panel(seed=0)
        S, value = _optimum(Y)
        assert len(S) == 4 and MIRROR_UNIT in S           # the instance has power
        got = _solve(Y, K, limit, **_side_restrictions(S)[kind])
        assert tuple(got.selected_unit_indices) == _complement(8, S)
        assert got.objective_value == pytest.approx(value, rel=1e-9)
        assert got.raw_results["mirrored"] is False

    def test_a_quota_that_maps_to_itself_keeps_it_on(self, monkeypatch):
        Y = _panel(seed=0)
        rules = DesignRestrictions(strata=[((0, 1, 2, 3), 1, 3)])
        filt = SubsetFilter.build(rules, N=8)
        best = min(_every_design(Y, filt).values())
        got = solve_synthetic_design_exact(Y, None, restrictions=rules)
        assert got.objective_value == pytest.approx(best, rel=1e-9)
        assert got.raw_results["mirrored"] is True
        assert len(got.selected_unit_indices) <= 4


# ----------------------------------------------------------------------
# the pool
# ----------------------------------------------------------------------


class TestPool:
    @pytest.mark.parametrize("limit", [ENUMERATE, BRANCH], ids=["enumerate", "branch"])
    def test_no_design_is_anothers_complement(self, limit):
        Y = _panel(seed=1)
        pool = solve_synthetic_design_exact_pool(Y, 4, top_K=3, candidate_limit=limit)
        sets = [tuple(int(i) for i in d.selected_unit_indices) for d in pool]
        values = sorted({round(v, 12) for S, v in _every_design(Y).items()
                         if len(S) == 4})
        assert len(sets) == 3
        assert all(MIRROR_UNIT in S for S in sets)
        assert len({frozenset(S) for S in sets}
                   | {frozenset(_complement(8, S)) for S in sets}) == 6
        assert [d.objective_value for d in pool] == pytest.approx(values[:3], rel=1e-9)

    def test_every_size(self):
        """With ``K=None`` the pool keeps to the lower half of the sizes."""
        Y = _panel(seed=1)
        pool = solve_synthetic_design_exact_pool(Y, None, top_K=4)
        for d in pool:
            S = tuple(int(i) for i in d.selected_unit_indices)
            assert len(S) < 4 or MIRROR_UNIT in S


# ----------------------------------------------------------------------
# failure
# ----------------------------------------------------------------------


class TestFailure:
    def test_a_mirrored_quota_no_design_meets_still_raises(self):
        """At least three and at most one of four units: the quota maps to
        itself, so the rule is on, and no design satisfies it."""
        rules = DesignRestrictions(strata=[((0, 1, 2, 3), 3, 1)])
        assert SubsetFilter.build(rules, N=8).is_mirrored(8)
        with pytest.raises(MlsynthEstimationError):
            solve_synthetic_design_exact(_panel(seed=0), None, restrictions=rules)
