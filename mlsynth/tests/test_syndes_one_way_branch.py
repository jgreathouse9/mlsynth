"""SYNDES one-way above the candidate limit: a certified branch and bound (#820).

Contract. Above ``candidate_limit`` the one-way backend searches treated sets on
the two-way backend's tree (:mod:`~mlsynth.utils.syndes_helpers.branch`). A
node's two-way bound bounds one-way too, since one-way is two-way with the
treated weights fixed at ``1/K``; the sets the search reaches are priced as the
enumeration prices them, by the ridge bound and then a fit. The search returns
the optimum the enumeration would, certified, under every treated-side
restriction, and at its work limit returns its best design, a valid bound and a
warning.

The levels:

- the leaves: a block is bounded by the ridge form, its best-bounded set is
  fitted, and only the sets under the incumbent stay open; a set whose donor
  rules leave it no control is never open; the open sets are fitted in bound
  order up to the incumbent, or up to the limit once there is a design (unit);
- the cost model's calibration points (unit);
- the search returns the brute-force optimum, certified, under any cost model,
  with every restriction, at any scale, level and labelling of the units
  (smoke, property, edge);
- a work limit returns a valid bracket and warns, a limit reached before any
  design raises, and an invalid limit is refused (failure).

The anchor is brute force: every admissible treated set priced from a fresh fit.
"""
from __future__ import annotations

import warnings
from itertools import combinations
from math import comb

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.solvers.active_set import solve_simplex_qp
from mlsynth.utils.syndes_helpers.branch import branch_and_bound
from mlsynth.utils.syndes_helpers.exact import SubsetFilter
from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem
from mlsynth.utils.syndes_helpers.one_way_exact import (
    OneWayFits,
    OneWayLeaves,
    fit_seconds,
    ridge_bound,
    ridge_seconds,
    solve_one_way_exact,
    solve_one_way_exact_pool,
)
from mlsynth.utils.syndes_helpers.optimization import (SYNDESSolveWarning,
                                                        estimate_lambda)
from mlsynth.utils.syndes_helpers.restrictions import DesignRestrictions

SETTINGS = dict(max_examples=20, deadline=None,
                suppress_health_check=[HealthCheck.too_slow])

#: Every node is bounded and every bounded node that survives the trust region
#: is lifted: the costs that make the search do the most bounding.
EAGER = dict(price_seconds=1.0, bound_seconds=lambda n: 0.0,
             lift_seconds=lambda n: 0.0)


def _panel(N=10, T=14, seed=0, rank=3, noise=0.35):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, rank)) @ rng.standard_normal((rank, N))
            + noise * rng.standard_normal((T, N)))


def _noise(N=10, T=14, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, 1)) + rng.standard_normal((T, N))


def _excluded(pairs):
    out = {}
    for i, j in pairs:
        out.setdefault(i, set()).add(j)
    return out


def _f(Y, S, lam, excluded=None):
    """``f(S)`` from a fresh fit on ``Y`` as given; infinite on an empty pool."""
    T, N = Y.shape
    banned = set(S)
    for i in S:
        banned |= set((excluded or {}).get(i, ()))
    pool = [j for j in range(N) if j not in banned]
    if not pool:
        return np.inf
    B = np.vstack([Y[:, pool] / np.sqrt(T), np.sqrt(lam) * np.eye(len(pool))])
    A = np.concatenate([Y[:, list(S)].mean(axis=1) / np.sqrt(T), np.zeros(len(pool))])
    c = solve_simplex_qp(B, A)
    r = A - B @ c
    return float(r @ r) + lam / len(S)


def _brute_force(Y, K, lam, accepts=lambda S: True, excluded=None):
    best, best_set = np.inf, None
    for S in combinations(range(Y.shape[1]), K):
        if accepts(S):
            value = _f(Y, S, lam, excluded)
            if value < best:
                best, best_set = value, S
    return best_set, best


def _treated(design):
    return tuple(int(i) for i in design.selected_unit_indices)


def _branched(Y, K, **kwargs):
    return solve_one_way_exact(Y, K, candidate_limit=1, **kwargs)


def _setup(Y, K, excluded=None, lam=None):
    lam = estimate_lambda(Y) if lam is None else lam
    problem = TwoWayProblem.from_panel(Y, lam=lam)
    engine = BoundEngine.from_problem(problem)
    fits = OneWayFits(Y, lam, {i: frozenset(js) for i, js in (excluded or {}).items()})
    return problem, engine, OneWayLeaves(fits, engine, K), lam


def _search(Y, K, filt=None, excluded=None, **kwargs):
    problem, engine, leaves, lam = _setup(Y, K, excluded)
    out = branch_and_bound(problem, engine, K, filt or SubsetFilter(), leaves=leaves,
                           **kwargs)
    return out, lam


def _every_set(N, K):
    return np.array(list(combinations(range(N), K)), dtype=np.int64)


# --------------------------------------------------------------------------- #
# the leaves
# --------------------------------------------------------------------------- #


class TestLeaves:
    def test_a_block_fits_its_best_bounded_set_and_keeps_those_under_it(self):
        Y = _panel(N=8, seed=1)
        _, engine, leaves, lam = _setup(Y, 3)
        sets = _every_set(8, 3)
        bounds = ridge_bound(engine, sets)
        j = int(np.argmin(bounds))
        value, S, open_sets, open_bounds, seconds = leaves.price(sets, np.inf)
        assert tuple(S) == tuple(sets[j])
        assert value == pytest.approx(_f(Y, tuple(sets[j]), lam), rel=1e-9)
        expected = {tuple(s) for k, (s, b) in enumerate(zip(sets, bounds))
                    if b < value and k != j}
        assert {tuple(s) for s in open_sets} == expected
        assert np.all(open_bounds < value)
        assert seconds == leaves.seconds_per_fit and leaves.fits.n_fits == 1
        assert leaves.kept == len(expected)

    def test_a_block_whose_best_bound_reaches_the_incumbent_fits_nothing(self):
        Y = _panel(N=8, seed=2)
        _, engine, leaves, _ = _setup(Y, 3)
        sets = _every_set(8, 3)
        floor = float(ridge_bound(engine, sets).min())
        value, S, open_sets, _, seconds = leaves.price(sets, floor)
        assert S is None and value == np.inf and len(open_sets) == 0
        assert seconds == 0.0 and leaves.fits.n_fits == 0

    def test_a_set_whose_donor_rules_leave_no_control_is_never_open(self):
        Y = _panel(N=4, seed=3)
        _, _, leaves, _ = _setup(Y, 2, excluded={0: {2, 3}})
        sets = _every_set(4, 2)
        bounds = leaves.bounds(sets)
        empty = [k for k, s in enumerate(sets) if tuple(s) == (0, 1)]
        assert np.isinf(bounds[empty]).all()
        assert np.isfinite(np.delete(bounds, empty)).all()
        _, _, open_sets, _, _ = leaves.price(sets, np.inf)
        assert (0, 1) not in {tuple(s) for s in open_sets}

    def test_with_every_pool_empty_nothing_is_fitted_or_open(self):
        Y = _panel(N=4, seed=4)
        pairs = [(i, j) for i in range(4) for j in range(4) if i != j]
        _, _, leaves, _ = _setup(Y, 2, excluded=_excluded(pairs))
        value, S, open_sets, _, seconds = leaves.price(_every_set(4, 2), np.inf)
        assert S is None and len(open_sets) == 0 and seconds == 0.0

    def test_resolution_fits_in_bound_order_up_to_the_incumbent(self):
        Y = _panel(N=8, seed=5)
        _, engine, leaves, lam = _setup(Y, 3)
        sets = _every_set(8, 3)
        bounds = ridge_bound(engine, sets)
        values = np.array([_f(Y, tuple(s), lam) for s in sets])
        value, S, n_fit, seconds, unresolved = leaves.resolve(
            sets, bounds, np.inf, None, budget=np.inf)
        assert value == pytest.approx(values.min(), rel=1e-9)
        assert tuple(S) == tuple(sets[int(np.argmin(values))])
        # every set whose bound is under the optimum was fitted, and at most one
        # other: the optimum's own, when its bound rounds to its value
        below = int(np.sum(bounds < value))
        assert below <= n_fit <= below + 1
        assert seconds == n_fit * leaves.seconds_per_fit
        assert unresolved == np.inf

    def test_resolution_stops_at_the_limit_with_the_least_bound_left(self):
        Y = _panel(N=8, seed=6)
        _, engine, leaves, lam = _setup(Y, 3)
        sets = _every_set(8, 3)
        bounds = ridge_bound(engine, sets)
        order = np.argsort(bounds, kind="stable")
        best = _f(Y, tuple(sets[order[-1]]), lam)
        value, S, n_fit, seconds, unresolved = leaves.resolve(
            sets, bounds, best, sets[order[-1]], budget=0.0)
        assert n_fit == 0 and seconds == 0.0 and value == best
        assert unresolved == bounds[order[0]]

    def test_resolution_without_a_design_fits_until_it_has_one(self):
        """``f(S) >= 0``, so zero bounds every set: past its budget the
        resolution fits the first set in order and stops there."""
        Y = _panel(N=8, seed=7)
        _, _, leaves, lam = _setup(Y, 3)
        sets = _every_set(8, 3)
        value, S, n_fit, _, unresolved = leaves.resolve(
            sets, np.zeros(len(sets)), np.inf, None, budget=-1.0)
        assert n_fit == 1 and tuple(S) == tuple(sets[0]) and unresolved == 0.0
        assert value == pytest.approx(_f(Y, tuple(sets[0]), lam), rel=1e-9)

    def test_a_settled_design_has_equal_treated_weights_and_its_fit(self):
        Y = _panel(N=8, seed=8)
        _, _, leaves, lam = _setup(Y, 3)
        value, a, b, gap = leaves.settle(np.array([1, 4, 6]))
        assert value == pytest.approx(_f(Y, (1, 4, 6), lam), rel=1e-9)
        assert np.allclose(a[[1, 4, 6]], 1 / 3) and a.sum() == pytest.approx(1.0)
        assert b[[1, 4, 6]].sum() == 0.0 and b.sum() == pytest.approx(1.0)
        assert gap == 0.0

    def test_a_seed_is_its_own_set(self):
        Y = _panel(N=8, seed=9)
        _, _, leaves, _ = _setup(Y, 4)
        block = np.array([[0, 1, 2, 3]])
        assert np.array_equal(leaves.seeds(block), block)


class TestTheCostModel:
    """Calibrated on one thread at 20 to 100 units and 12 to 120 periods (#820)."""

    @pytest.mark.parametrize("K,low,high", [(2, 0.37e-6, 0.40e-6), (10, 2.85e-6, 3.23e-6),
                                            (20, 10.4e-6, 12.3e-6), (40, 37e-6, 43e-6)])
    def test_the_ridge_bound_per_set_is_within_the_measured_range(self, K, low, high):
        assert 0.9 * low <= ridge_seconds(K) <= 1.1 * high

    @pytest.mark.parametrize("P,low,high", [(18, 0.78e-3, 0.85e-3), (38, 2.2e-3, 3.0e-3),
                                            (58, 4.6e-3, 8.2e-3), (98, 7.8e-3, 8.8e-3)])
    def test_a_fit_is_within_twice_the_measured_range(self, P, low, high):
        assert low / 2 <= fit_seconds(P) <= 2 * high

    @given(st.integers(1, 200))
    @settings(max_examples=30, deadline=None)
    def test_both_grow_with_their_size(self, n):
        assert 0 < ridge_seconds(n) < ridge_seconds(n + 1)
        assert 0 < fit_seconds(n) < fit_seconds(n + 1)


# --------------------------------------------------------------------------- #
# the search
# --------------------------------------------------------------------------- #


def test_past_the_limit_the_search_returns_the_certified_optimum():
    Y = _panel(N=10, seed=30)
    lam = estimate_lambda(Y)
    S, optimum = _brute_force(Y, 4, lam)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SYNDESSolveWarning)
        d = _branched(Y, 4)
    assert _treated(d) == S
    assert d.objective_value == pytest.approx(optimum, rel=1e-9)
    out = d.solve_outcome
    assert out.status == "optimal" and not out.stopped_at_limit
    assert optimum * (1 - 1e-8) <= out.dual_bound <= d.objective_value * (1 + 1e-12)
    raw = d.raw_results
    assert raw["certified"] is True and raw["search"] == "branch_and_bound"
    assert raw["n_designs"] == comb(10, 4) and raw["work"] > 0.0
    assert raw["n_nodes"] >= 0 and raw["n_lifts"] >= 0 and raw["n_priced"] >= 1


def test_below_the_limit_the_search_is_the_enumeration():
    d = solve_one_way_exact(_panel(N=8, seed=31), 3)
    assert d.raw_results["search"] == "enumeration"
    assert d.raw_results["n_nodes"] == 0 and d.raw_results["work"] == 0.0


panels = st.builds(
    lambda N, seed, kind: (_noise if kind else _panel)(N=N, seed=seed),
    st.integers(6, 9), st.integers(0, 10 ** 6), st.booleans())


@given(panels, st.data(), st.booleans())
@settings(**SETTINGS)
def test_under_any_cost_model_the_search_returns_the_optimum(Y, data, eager):
    N = Y.shape[1]
    K = data.draw(st.integers(1, N - 1))
    out, lam = _search(Y, K, **(EAGER if eager else {}))
    _, optimum = _brute_force(Y, K, lam)
    assert out.certified and out.closed
    assert out.value == pytest.approx(optimum, rel=1e-9)
    assert out.lower <= optimum * (1 + 1e-9)


def test_the_swap_search_only_seeds_the_incumbent():
    Y = _panel(N=10, seed=32)
    seeded, lam = _search(Y, 4)
    cold, _ = _search(Y, 4, restarts=0)
    _, optimum = _brute_force(Y, 4, lam)
    assert seeded.value == pytest.approx(optimum, rel=1e-9)
    assert cold.value == pytest.approx(optimum, rel=1e-9) and cold.certified


def test_a_node_whose_two_way_bound_clears_the_incumbent_is_discarded():
    """The two-way bound bounds one-way: under eager bounding the search prunes
    and still returns the optimum."""
    Y = _panel(N=12, seed=33)
    out, lam = _search(Y, 4, **EAGER)
    assert out.pruned > 0 and out.priced < comb(12, 4)
    assert out.value == pytest.approx(_brute_force(Y, 4, lam)[1], rel=1e-9)


#: Bound the larger nodes and price the smaller: on the panel below the
#: subtrees priced hold 21 to 70 sets, and a block of fifty fills from two.
SMALL_SUBTREES = dict(price_seconds=1e-4, bound_seconds=lambda n: 1e-3,
                      lift_seconds=lambda n: np.inf)


@pytest.mark.parametrize("costs,chunk", [({}, 7), (SMALL_SUBTREES, 50)],
                         ids=["the-root-streams", "small-subtrees-buffer"])
def test_the_block_size_changes_nothing(monkeypatch, costs, chunk):
    """A subtree larger than a block streams in blocks, and small subtrees are
    buffered until a block fills; neither changes the design or its bound."""
    from mlsynth.utils.syndes_helpers import branch
    Y = _panel(N=10, seed=35)
    ref, lam = _search(Y, 4, **costs)
    monkeypatch.setattr(branch, "CHUNK", chunk)
    got, _ = _search(Y, 4, **costs)
    assert got.certified and got.priced == ref.priced
    assert got.value == pytest.approx(ref.value, rel=1e-12)
    assert got.value == pytest.approx(_brute_force(Y, 4, lam)[1], rel=1e-9)


@pytest.mark.parametrize("K", [1, 9])
def test_the_extreme_sizes(K):
    Y = _panel(N=10, seed=34)
    lam = estimate_lambda(Y)
    d = _branched(Y, K)
    assert d.objective_value == pytest.approx(_brute_force(Y, K, lam)[1], rel=1e-9)
    assert d.raw_results["certified"] is True


# --------------------------------------------------------------------------- #
# restrictions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rules", [
    dict(forced_in=[1]),
    dict(forbidden=[0, 2]),
    dict(conflict_pairs=[(0, 1), (1, 3), (2, 5)]),
    dict(strata=[((0, 1, 2, 3), 1, 1)]),
    dict(forced_in=[4], forbidden=[0], conflict_pairs=[(4, 5)]),
])
def test_treated_side_restrictions_hold_past_the_limit(rules):
    Y = _panel(seed=40)
    lam = estimate_lambda(Y)
    restrictions = DesignRestrictions(**rules)
    filt = SubsetFilter.build(restrictions, N=10)
    S, optimum = _brute_force(Y, 4, lam, accepts=filt.accepts)
    d = _branched(Y, 4, restrictions=restrictions)
    assert _treated(d) == S and filt.accepts(S)
    assert d.objective_value == pytest.approx(optimum, rel=1e-9)
    assert d.raw_results["certified"] is True


def test_a_cost_budget_holds_past_the_limit():
    Y = _panel(seed=41)
    lam = estimate_lambda(Y)
    costs = np.arange(1.0, 11.0)
    filt = SubsetFilter.build(None, costs, 14.0, N=10)
    S, optimum = _brute_force(Y, 4, lam, accepts=filt.accepts)
    d = _branched(Y, 4, costs=costs, budget=14.0)
    assert _treated(d) == S and costs[list(S)].sum() <= 14.0
    assert d.objective_value == pytest.approx(optimum, rel=1e-9)


def test_donor_exclusions_apply_past_the_limit():
    Y = _panel(seed=42)
    lam = estimate_lambda(Y)
    pairs = [(i, j) for i in range(10) for j in range(10) if i != j and (i + j) % 4 == 0]
    S, optimum = _brute_force(Y, 4, lam, excluded=_excluded(pairs))
    d = _branched(Y, 4, restrictions=DesignRestrictions(donor_exclusion=pairs))
    assert _treated(d) == S
    assert d.objective_value == pytest.approx(optimum, rel=1e-9)
    assert all(d.control_weights[j] == 0.0 for i, j in pairs if i in S)


@given(st.integers(0, 10 ** 6), st.data())
@settings(**SETTINGS)
def test_any_restriction_gives_the_admissible_optimum(seed, data):
    rng = np.random.default_rng(seed)
    N, K = 9, 3
    Y = _panel(N=N, seed=seed % 1000)
    lam = estimate_lambda(Y)
    units = rng.permutation(N)
    rules = dict(forced_in=[int(units[0])] if data.draw(st.booleans()) else [],
                 forbidden=[int(units[1])] if data.draw(st.booleans()) else [],
                 conflict_pairs=[(int(units[2]), int(units[3]))]
                 if data.draw(st.booleans()) else [])
    pairs = ([(int(units[4]), int(units[5]))] if data.draw(st.booleans()) else [])
    restrictions = DesignRestrictions(donor_exclusion=pairs, **rules)
    filt = SubsetFilter.build(DesignRestrictions(**rules), N=N)
    S, optimum = _brute_force(Y, K, lam, accepts=filt.accepts, excluded=_excluded(pairs))
    d = _branched(Y, K, restrictions=restrictions)
    assert d.objective_value == pytest.approx(optimum, rel=1e-9)
    assert filt.accepts(_treated(d)) and d.raw_results["certified"] is True


def test_the_pool_past_the_limit_is_the_enumerations():
    Y = _panel(N=8, seed=43)
    enumerated = solve_one_way_exact_pool(Y, 3, top_K=4)
    branched = solve_one_way_exact_pool(Y, 3, top_K=4, candidate_limit=1)
    assert [_treated(d) for d in branched] == [_treated(d) for d in enumerated]
    for a, b in zip(branched, enumerated):
        assert a.objective_value == pytest.approx(b.objective_value, rel=1e-9)


@pytest.mark.parametrize("rules", [
    dict(forced_in=[0, 1, 2, 3, 4]),               # more forced than K
    dict(forbidden=[0, 1, 2, 3, 4, 5, 6]),         # fewer treatable than K
    dict(forced_in=[0], conflict_pairs=[(0, j) for j in range(1, 10)]),
])
def test_restrictions_that_leave_no_design_are_reported_past_the_limit(rules):
    with pytest.raises(MlsynthEstimationError, match="no feasible design"):
        _branched(_panel(), 4, restrictions=DesignRestrictions(**rules))


def test_donor_exclusions_that_empty_every_pool_are_reported_past_the_limit():
    pairs = [(i, j) for i in range(6) for j in range(6) if i != j]
    with pytest.raises(MlsynthEstimationError, match="no feasible design"):
        _branched(_panel(N=6), 2, restrictions=DesignRestrictions(donor_exclusion=pairs))


# --------------------------------------------------------------------------- #
# invariances
# --------------------------------------------------------------------------- #


@given(st.floats(1e-3, 1e3), st.integers(0, 10 ** 6))
@settings(**SETTINGS)
def test_a_panel_in_other_units_gets_the_same_design(s, seed):
    Y = _panel(N=9, seed=seed % 1000)
    ref = _branched(Y, 3)
    got = _branched(Y * s, 3)
    assert _treated(got) == _treated(ref)
    assert got.objective_value / s ** 2 == pytest.approx(ref.objective_value, rel=1e-8)


@given(st.floats(-1e4, 1e4), st.integers(0, 10 ** 6))
@settings(**SETTINGS)
def test_a_common_level_changes_nothing(level, seed):
    Y = _panel(N=9, seed=seed % 1000)
    ref = _branched(Y, 3)
    got = _branched(Y + level, 3, lam=ref.lambda_value)
    assert _treated(got) == _treated(ref)
    assert got.objective_value == pytest.approx(ref.objective_value, rel=1e-7)


@given(st.permutations(range(9)), st.integers(0, 10 ** 6))
@settings(**SETTINGS)
def test_relabelling_the_units_relabels_the_design(perm, seed):
    perm = np.asarray(perm)
    Y = _panel(N=9, seed=seed % 1000)
    ref = _branched(Y, 3)
    got = _branched(Y[:, perm], 3, lam=ref.lambda_value)
    assert tuple(sorted(int(perm[i]) for i in _treated(got))) == _treated(ref)
    assert got.objective_value == pytest.approx(ref.objective_value, rel=1e-9)


# --------------------------------------------------------------------------- #
# the work limit
# --------------------------------------------------------------------------- #


def test_a_search_stopped_at_its_limit_returns_its_best_design_and_warns():
    Y = _panel(N=12, seed=50)
    lam = estimate_lambda(Y)
    _, optimum = _brute_force(Y, 5, lam)
    with pytest.warns(SYNDESSolveWarning, match="work limit"):
        d = _branched(Y, 5, work_limit=1e-9)
    out = d.solve_outcome
    assert out.status == "work_limit" and out.stopped_at_limit
    assert d.raw_results["certified"] is False
    assert out.dual_bound <= optimum * (1 + 1e-9)
    assert optimum <= d.objective_value * (1 + 1e-9)
    assert out.gap > 1e-8 and out.gap == pytest.approx(
        (d.objective_value - out.dual_bound) / d.objective_value, rel=1e-12)


@given(st.floats(1e-7, 1e-2), st.integers(0, 10 ** 6))
@settings(**SETTINGS)
def test_at_any_limit_the_design_and_bound_bracket_the_optimum(limit, seed):
    Y = _noise(N=10, seed=seed % 1000)
    out, lam = _search(Y, 4, work_limit=limit, **EAGER)
    _, optimum = _brute_force(Y, 4, lam)
    assert out.lower <= optimum * (1 + 1e-9)
    assert out.value >= optimum * (1 - 1e-9)
    if out.closed:
        assert out.value == pytest.approx(optimum, rel=1e-9)


def test_each_set_bounded_and_each_fit_is_charged_at_the_leaves_cost():
    """Priced at the root, without seeds, the search's work is its sets at the
    ridge bound's cost and its fits at a fit's."""
    Y = _panel(N=10, seed=53)
    problem, engine, leaves, _ = _setup(Y, 4)
    out = branch_and_bound(problem, engine, 4, SubsetFilter(), leaves=leaves,
                           restarts=0, price_seconds=0.0)
    assert out.nodes == 0 and out.priced == comb(10, 4)
    assert leaves.seconds_per_set == ridge_seconds(4)
    assert leaves.seconds_per_fit == fit_seconds(6)
    assert out.work == pytest.approx(out.priced * ridge_seconds(4)
                                     + leaves.fits.n_fits * fit_seconds(6), rel=1e-12)


def test_every_fit_counts_toward_the_limit():
    """A fit priced at a second against a limit of half that: the root's
    best-bounded set is fitted, its donor rules leave it one poor control, so
    other sets stay open under it, and the resolution stops with the search's
    one design and leaves them open."""
    Y = _panel(N=10, seed=51)
    lam = estimate_lambda(Y)
    engine = BoundEngine.from_problem(TwoWayProblem.from_panel(Y, lam=lam))
    sets = _every_set(10, 4)
    first = tuple(int(i) for i in sets[int(np.argmin(ridge_bound(engine, sets)))])
    keep = next(j for j in range(10) if j not in first)
    excluded = {first[0]: {j for j in range(10) if j not in first and j != keep}}
    problem, engine, leaves, _ = _setup(Y, 4, excluded=excluded)
    leaves.seconds_per_fit = 1.0
    out = branch_and_bound(problem, engine, 4, SubsetFilter(), leaves=leaves,
                           restarts=0, work_limit=0.5, price_seconds=0.0)
    _, optimum = _brute_force(Y, 4, lam, excluded=excluded)
    assert leaves.fits.n_fits == 1 and out.work >= 1.0
    assert not out.closed and not out.certified
    assert out.lower <= optimum * (1 + 1e-9)
    assert optimum <= out.value * (1 + 1e-9)


def test_a_limit_reached_before_any_design_is_reported():
    Y = _panel(N=10, seed=52)
    problem, engine, leaves, _ = _setup(Y, 4)
    with pytest.raises(MlsynthEstimationError, match="work limit"):
        branch_and_bound(problem, engine, 4, SubsetFilter(), leaves=leaves,
                         restarts=0, work_limit=1e-12, **EAGER)


@pytest.mark.parametrize("limit", [0, -3.0, np.inf, np.nan, True, "1"])
def test_an_invalid_limit_is_refused(limit):
    with pytest.raises(MlsynthConfigError, match="work_limit"):
        solve_one_way_exact(_panel(), 3, work_limit=limit)
