"""MAREX's exact backend under a total-cost budget across clusters
(jgreathouse9/mlsynth#952).

``market_cost_col`` with ``max_total_cost`` caps the summed cost of the treated
markets, sum_j c_j z_j <= B. With one cluster the search drops every treated set
over budget. With several, the budget ties the clusters together: the best
affordable design can need a cluster's 50th-best treated set because it is
cheap. The exact backend now keeps each cluster's best designs in order with
their costs and picks one per cluster by a branch and bound over clusters (a
multiple-choice knapsack). A combination that uses a design beyond a cluster's
list scores at least that list's last value plus every other cluster's best;
while that bound does not clear the answer, the lists are lengthened.

The oracle enumerates every treated set of every cluster, fits both composites
with Clarabel, and searches every combination; it imports nothing from the
search.

Levels: smoke, unit invariants of the three pieces (the cheapest admissible
cost, the combination step, the bound outside the lists), the design against
the oracle and the MIQP, edge cases, failures, and hypothesis properties.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.config_models import MAREXConfig
from mlsynth.tests.test_marex_exact_backend import _cfg, _fit, _panel, _treated
from mlsynth.tests.test_marex_exact_oracle import _predictors, _simplex_fit
from mlsynth.utils.marex_helpers import exact as exact_mod
from mlsynth.utils.marex_helpers.exact import (cheapest_admissible_cost,
                                               k_best_affordable, outside_bound)
from mlsynth.utils.syndes_helpers.exact import SubsetFilter

CARRIED = 1e-8


# ---------------------------------------------------------------- the oracle
def _with_costs(df, costs):
    out = df.copy()
    labels = list(dict.fromkeys(out["unit"]))
    out["cost"] = out["unit"].map(dict(zip(labels, map(float, costs))))
    return out


def _cluster_designs(X, members, m):
    """Every admissible treated set of one cluster: (value, set as rows of X)."""
    A = X[members].mean(axis=0)
    out = []
    for S in itertools.combinations(members, m):
        g, w = _simplex_fit(X[list(S)].T, A, weights=True)
        if np.any(w <= CARRIED):
            continue
        C = [j for j in members if j not in S]
        out.append((g + _simplex_fit(X[C].T, A), S))
    return out


def _oracle(df, T0, costs, budget, m, region_sizes, k=1):
    """The ``k`` smallest objectives over one admissible set per cluster whose
    costs sum to at most ``budget``, with each one's treated labels."""
    labels, X = _predictors(df, T0)
    starts = np.cumsum([0, *region_sizes])
    per = [_cluster_designs(X, list(range(a, b)), m)
           for a, b in zip(starts[:-1], starts[1:])]
    combos = []
    for pick in itertools.product(*per):
        rows = [j for _, S in pick for j in S]
        if sum(costs[j] for j in rows) <= budget + 1e-9:
            combos.append((sum(v for v, _ in pick), frozenset(labels[j] for j in rows)))
    combos.sort(key=lambda t: t[0])
    return combos[:k]


def _budgeted(df, T0, costs, budget, backend="exact", m=1, **over):
    return _fit(_with_costs(df, costs), T0, backend, m_eq=m, cluster="region",
                market_cost_col="cost", max_total_cost=budget, **over)


def _spend(res, df, costs):
    by = dict(zip(dict.fromkeys(df["unit"]), costs))
    return sum(by[u] for u in _treated(res))


REGIONS = (5, 5)


@pytest.fixture(scope="module")
def panel():
    return _panel(J=10, seed=3, region_sizes=REGIONS)


@pytest.fixture(scope="module")
def free(panel):
    df, T0 = panel
    return _fit(df, T0, "exact", m_eq=1, cluster="region")


@pytest.fixture(scope="module")
def costs(panel, free):
    """The free design's markets cost 5 each, every other market 1."""
    df, _ = panel
    chosen = _treated(free)
    return [5.0 if u in chosen else 1.0 for u in dict.fromkeys(df["unit"])]


# ---------------------------------------------------------------- smoke
def test_smoke_a_budget_across_clusters_gives_a_design_within_it(panel, costs):
    df, T0 = panel
    res = _budgeted(df, T0, costs, budget=6.0)
    assert len(_treated(res)) == 2
    assert _spend(res, df, costs) <= 6.0
    assert np.isfinite(res.globres.objective)


# ---------------------------------------------------------------- the cheapest admissible cost
def test_the_cheapest_cost_is_the_cheapest_markets():
    c = np.array([4.0, 1.0, 3.0, 2.0])
    assert cheapest_admissible_cost(c, [2], SubsetFilter()) == 3.0


def test_the_cheapest_cost_takes_the_cheapest_size():
    c = np.array([4.0, 1.0, 3.0, 2.0])
    assert cheapest_admissible_cost(c, [3, 1, 2], SubsetFilter()) == 1.0


def test_a_forced_market_is_always_paid():
    c = np.array([4.0, 1.0, 3.0, 2.0])
    assert cheapest_admissible_cost(c, [2], SubsetFilter(forced_in=frozenset({0}))) == 5.0


def test_a_forbidden_market_is_never_paid():
    c = np.array([4.0, 1.0, 3.0, 2.0])
    assert cheapest_admissible_cost(c, [2], SubsetFilter(forbidden=frozenset({1}))) == 5.0


def test_a_size_the_rules_cannot_fill_costs_nothing_it_is_never_taken():
    c = np.array([4.0, 1.0, 3.0])
    filt = SubsetFilter(forbidden=frozenset({0, 1}))
    assert cheapest_admissible_cost(c, [2], filt) == np.inf
    assert cheapest_admissible_cost(c, [2, 1], filt) == 3.0


def test_more_forced_markets_than_the_size_admit_nothing():
    c = np.array([4.0, 1.0, 3.0])
    filt = SubsetFilter(forced_in=frozenset({0, 1}))
    assert cheapest_admissible_cost(c, [1], filt) == np.inf


@settings(max_examples=60, deadline=None)
@given(c=st.lists(st.sampled_from([0.0, 0.5, 1.0, 2.0, 3.0, 7.0]), min_size=3, max_size=7),
       data=st.data())
def test_the_cheapest_cost_is_a_floor_under_every_admissible_set(c, data):
    """Over costs and rules: no admissible set costs less, and with only forced
    and forbidden markets the floor is attained."""
    n = len(c)
    c = np.asarray(c)
    sizes = data.draw(st.lists(st.integers(1, n - 1), min_size=1, max_size=3, unique=True))
    forced = data.draw(st.sets(st.integers(0, n - 1), max_size=2))
    forbidden = data.draw(st.sets(st.integers(0, n - 1), max_size=2)) - forced
    conflicts = tuple(data.draw(st.lists(
        st.tuples(st.integers(0, n - 1), st.integers(0, n - 1)), max_size=2)))
    plain = SubsetFilter(forced_in=frozenset(forced), forbidden=frozenset(forbidden))
    ruled = SubsetFilter(forced_in=frozenset(forced), forbidden=frozenset(forbidden),
                         conflict_pairs=conflicts)
    floor = cheapest_admissible_cost(c, sizes, ruled)
    paid = [float(c[list(S)].sum()) for s in sizes
            for S in itertools.combinations(range(n), s) if ruled.accepts(S)]
    assert all(p >= floor - 1e-12 for p in paid)
    attained = [float(c[list(S)].sum()) for s in sizes
                for S in itertools.combinations(range(n), s) if plain.accepts(S)]
    assert floor == (min(attained) if attained else np.inf)


# ---------------------------------------------------------------- the combination step
def _brute(lists, budget, k):
    out = []
    for pick in itertools.product(*[range(len(lst)) for lst in lists]):
        cost = sum(lists[c][i][1] for c, i in enumerate(pick))
        if cost <= budget + 1e-9:
            out.append((sum(lists[c][i][0] for c, i in enumerate(pick)), pick))
    out.sort()
    return out[:k]


def test_a_slack_budget_picks_each_clusters_best():
    lists = [[(1.0, 5.0), (2.0, 1.0)], [(0.5, 5.0), (3.0, 1.0)]]
    assert k_best_affordable(lists, 10.0, 1) == [(1.5, (0, 0))]


def test_a_binding_budget_takes_a_worse_design_where_it_costs_least():
    lists = [[(1.0, 5.0), (2.0, 1.0)], [(0.5, 5.0), (3.0, 1.0)]]
    assert k_best_affordable(lists, 6.0, 1) == [(2.5, (1, 0))]


def test_a_budget_no_combination_meets_returns_none():
    lists = [[(1.0, 5.0)], [(0.5, 5.0)]]
    assert k_best_affordable(lists, 9.0, 3) == []


def test_a_cost_exactly_at_the_budget_is_affordable():
    lists = [[(1.0, 2.5)], [(0.5, 3.5)]]
    assert k_best_affordable(lists, 6.0, 1) == [(1.5, (0, 0))]


def test_the_k_best_are_distinct_and_in_order():
    lists = [[(1.0, 1.0), (2.0, 1.0), (4.0, 1.0)], [(0.0, 1.0), (1.5, 1.0)]]
    got = k_best_affordable(lists, 10.0, 4)
    assert [t for t, _ in got] == [1.0, 2.0, 2.5, 3.5]
    assert len({p for _, p in got}) == 4


@settings(max_examples=150, deadline=None)
@given(data=st.data(), budget=st.sampled_from([0.0, 1.0, 2.0, 3.5, 5.0, 8.0]),
       k=st.integers(1, 5))
def test_the_combination_step_is_the_brute_force(data, budget, k):
    """Over lists, costs, budgets and pool sizes: the step returns the k best
    affordable combinations, by value."""
    K = data.draw(st.integers(1, 4))
    lists = []
    for _ in range(K):
        vals = sorted(data.draw(st.lists(st.floats(0, 10, allow_nan=False),
                                         min_size=1, max_size=5)))
        lists.append([(v, data.draw(st.sampled_from([0.0, 0.5, 1.0, 2.0, 4.0])))
                      for v in vals])
    got = k_best_affordable(lists, budget, k)
    want = _brute(lists, budget, k)
    assert [t for t, _ in got] == pytest.approx([t for t, _ in want])
    for total, pick in got:
        assert sum(lists[c][i][1] for c, i in enumerate(pick)) <= budget + 1e-9
        assert total == pytest.approx(sum(lists[c][i][0] for c, i in enumerate(pick)))


# ---------------------------------------------------------------- the bound outside the lists
def test_the_bound_is_the_last_value_plus_the_other_clusters_best():
    values = [[1.0, 2.0, 4.0], [0.5, 3.0]]
    assert outside_bound(values, [False, False]) == pytest.approx(min(4.0 + 0.5, 3.0 + 1.0))


def test_a_complete_list_has_nothing_outside_it():
    values = [[1.0, 2.0, 4.0], [0.5, 3.0]]
    assert outside_bound(values, [False, True]) == pytest.approx(4.5)
    assert outside_bound(values, [True, True]) == np.inf


@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_nothing_outside_the_lists_scores_under_the_bound(data):
    """Over clusters' full design lists cut to any lengths: every combination
    using a design past some cluster's cut scores at least the bound."""
    K = data.draw(st.integers(1, 3))
    full = [sorted(data.draw(st.lists(st.floats(0, 10, allow_nan=False),
                                      min_size=1, max_size=5))) for _ in range(K)]
    cut = [data.draw(st.integers(1, len(f))) for f in full]
    bound = outside_bound([f[:n] for f, n in zip(full, cut)],
                          [n == len(f) for f, n in zip(full, cut)])
    for pick in itertools.product(*[range(len(f)) for f in full]):
        if any(i >= n for i, n in zip(pick, cut)):
            assert sum(f[i] for f, i in zip(full, pick)) >= bound - 1e-12


# ---------------------------------------------------------------- the design
def test_the_design_is_the_oracle_under_a_binding_budget(panel, costs):
    df, T0 = panel
    [(want, chosen)] = _oracle(df, T0, costs, 6.0, 1, REGIONS)
    res = _budgeted(df, T0, costs, budget=6.0)
    assert res.globres.objective == pytest.approx(want, rel=1e-6)
    assert _treated(res) == chosen


def test_a_budget_two_cheap_markets_meet_moves_both_clusters(panel, costs, free):
    df, T0 = panel
    res = _budgeted(df, T0, costs, budget=2.0)
    [(want, chosen)] = _oracle(df, T0, costs, 2.0, 1, REGIONS)
    assert _treated(res) == chosen
    assert not (_treated(res) & _treated(free))
    assert res.globres.objective == pytest.approx(want, rel=1e-6)


def test_the_design_matches_the_mip(panel, costs):
    df, T0 = panel
    ex = _budgeted(df, T0, costs, budget=6.0)
    mip = _budgeted(df, T0, costs, budget=6.0, backend="mip")
    assert ex.globres.objective <= mip.globres.objective + 1e-6 * (1 + abs(mip.globres.objective))
    assert ex.globres.objective == pytest.approx(mip.globres.objective, rel=1e-4)


def test_a_slack_budget_changes_nothing(panel, costs, free):
    df, T0 = panel
    res = _budgeted(df, T0, costs, budget=100.0)
    assert _treated(res) == _treated(free)
    assert res.globres.objective == pytest.approx(free.globres.objective, rel=1e-12)


def test_short_lists_are_lengthened_until_the_bound_clears(panel, costs, monkeypatch):
    """Each cluster's best market costs 5, so lists of one design hold no
    combination within 6; the step lengthens them and still returns the
    oracle's design."""
    monkeypatch.setattr(exact_mod, "COMBINE_START", 1)
    df, T0 = panel
    [(want, chosen)] = _oracle(df, T0, costs, 6.0, 1, REGIONS)
    res = _budgeted(df, T0, costs, budget=6.0)
    assert _treated(res) == chosen
    assert res.globres.objective == pytest.approx(want, rel=1e-6)


def test_each_cluster_drops_sets_the_others_cheapest_leave_no_room_for():
    """Cluster 2's cheapest admissible market costs 3 of a budget of 4, so
    cluster 1 prices only its markets that cost at most 1 (two, not the three
    within the budget alone), and cluster 1's cheapest costs 1, so cluster 2
    prices only its markets that cost at most 3 (two, not five)."""
    df, T0 = _panel(J=10, seed=3, region_sizes=REGIONS)
    costs = [1.0, 1.0, 2.0, 5.0, 5.0, 3.0, 3.0, 4.0, 4.0, 4.0]
    res = _budgeted(df, T0, costs, budget=4.0)
    first, second = (res.clusters[k].search.counts for k in sorted(res.clusters))
    assert first.priced == 2
    assert second.priced == 2
    assert _spend(res, df, costs) <= 4.0


def test_every_pooled_design_is_the_oracles(panel, costs):
    df, T0 = panel
    want = _oracle(df, T0, costs, 6.0, 1, REGIONS, k=4)
    pool = exact_mod.solve_design_exact_pool(
        *_arrays(df, T0), top_K=4, m_eq=1,
        market_costs=np.asarray(costs), max_total_cost=6.0)
    assert [d["objective"] for d in pool] == pytest.approx([t for t, _ in want], rel=1e-6)
    for d in pool:
        assert float(np.asarray(costs) @ (d["z_opt"].sum(axis=1) > 0.5)) <= 6.0 + 1e-9


def _arrays(df, T0):
    labels, _ = _predictors(df, T0)
    wide = df.pivot(index="unit", columns="time", values="y").loc[labels]
    region = df.drop_duplicates("unit").set_index("unit").loc[labels, "region"]
    return wide.to_numpy(float), T0, region.to_numpy()


@settings(max_examples=20, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(costs=st.lists(st.sampled_from([0.0, 1.0, 2.0, 3.0, 5.0]), min_size=8, max_size=8),
       budget=st.sampled_from([1.0, 2.0, 3.0, 4.0, 6.0, 10.0]),
       seed=st.integers(0, 3), start=st.sampled_from([1, 2, 16]))
def test_the_design_is_the_oracle_over_the_domain(costs, budget, seed, start, monkeypatch):
    """Over costs, budgets, panels and starting list lengths: the design keeps
    to the budget and is the oracle's, or the fit reports that none exists."""
    monkeypatch.setattr(exact_mod, "COMBINE_START", start)
    df, T0 = _panel(J=8, seed=seed, region_sizes=(4, 4))
    want = _oracle(df, T0, costs, budget, 2, (4, 4))
    if not want:
        with pytest.raises(MlsynthEstimationError):
            _budgeted(df, T0, costs, budget, m=2)
        return
    res = _budgeted(df, T0, costs, budget, m=2)
    assert _spend(res, df, costs) <= budget + 1e-9
    assert res.globres.objective == pytest.approx(want[0][0], rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------- edges and failures
def test_a_budget_no_combination_meets_is_reported(panel, costs):
    df, T0 = panel
    with pytest.raises(MlsynthEstimationError,
                       match="cheapest treated markets each cluster admits cost 6"):
        _budgeted(df, T0, [3.0] * 10, budget=5.0)


def test_rules_the_cost_floor_ignores_can_leave_no_combination():
    """Each cluster's two cheapest markets border each other, so the floor
    (two markets at 1) is under what a cluster pays (1 + 3); every combination
    costs 8, over the budget of 6, though each cluster alone fits it."""
    df, T0 = _panel(J=10, seed=3, region_sizes=REGIONS)
    labels = [f"u{j}" for j in range(10)]
    A = pd.DataFrame(0.0, index=labels, columns=labels)
    for a, b in [("u0", "u1"), ("u5", "u6")]:
        A.loc[a, b] = A.loc[b, a] = 1.0
    with pytest.raises(MlsynthEstimationError, match="no admissible treated set per cluster"):
        _budgeted(df, T0, [1.0, 1.0, 3.0, 3.0, 3.0] * 2, budget=6.0, m=2,
                  adjacency=A, spillover_threshold=0.5)


def test_free_markets_never_bind(panel, free):
    df, T0 = panel
    res = _budgeted(df, T0, [0.0] * 10, budget=1.0)
    assert _treated(res) == _treated(free)


def test_lists_past_their_limit_are_refused(panel, costs, monkeypatch):
    monkeypatch.setattr(exact_mod, "COMBINE_START", 1)
    monkeypatch.setattr(exact_mod, "COMBINE_LIMIT", 1)
    df, T0 = panel
    with pytest.raises(MlsynthConfigError, match="backend='mip'"):
        _budgeted(df, T0, costs, budget=6.0)


def test_the_certificate_is_refused_across_clusters(panel, costs):
    df, T0 = panel
    with pytest.raises(MlsynthConfigError, match="certif"):
        _budgeted(df, T0, costs, budget=6.0, certify=True)


def test_the_configuration_refuses_the_certificate_across_clusters(panel, costs):
    """Refused when the configuration is built, before any search runs."""
    df, T0 = panel
    with pytest.raises(MlsynthConfigError, match="certif"):
        MAREXConfig(**_cfg(_with_costs(df, costs), T0, "exact", m_eq=1, cluster="region",
                           certify=True, market_cost_col="cost", max_total_cost=6.0))


def test_a_direct_search_refuses_the_certificate_across_clusters(panel, costs):
    df, T0 = panel
    with pytest.raises(MlsynthConfigError, match="certif"):
        exact_mod.solve_design_exact(*_arrays(df, T0), m_eq=1, certify=True,
                                     market_costs=np.asarray(costs), max_total_cost=6.0)


def test_one_cluster_keeps_its_certificate():
    df, T0 = _panel(J=9, seed=0)
    res = _fit(_with_costs(df, [1.0] * 9).assign(region=0), T0, "exact", m_eq=2,
               cluster="region", certify=True,
               market_cost_col="cost", max_total_cost=2.0)
    assert len(_treated(res)) == 2


def test_one_cluster_keeps_the_filter_alone(monkeypatch):
    """With one cluster the budget stays a rule on the treated set: the
    combination step is never entered."""
    def boom(*a, **k):
        raise AssertionError("the combination step ran for one cluster")
    monkeypatch.setattr(exact_mod, "k_best_affordable", boom)
    df, T0 = _panel(J=9, seed=0)
    res = _fit(_with_costs(df, [1.0] * 9), T0, "exact", m_eq=2,
               market_cost_col="cost", max_total_cost=2.0)
    assert len(_treated(res)) == 2
