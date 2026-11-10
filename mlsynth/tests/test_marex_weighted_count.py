"""A treated market is one the synthetic treated unit puts weight on.

Abadie and Zhao's design problem bounds the number of nonzero treated weights,
``m_min <= ||w||_0 <= m_max``. The exact backend fixes a treated set ``S`` and
fits simplex weights within it, and those weights can be exactly zero: the
fitted target sits on a face of the hull of ``S`` and one market of ``S`` is
off that face. Counting ``|S|`` in place of ``||w||_0`` then lets ``m_eq = m``
return a design that uses fewer than ``m`` markets, charges a total-cost budget
for a market the design ignores, and holds a market out of the control pool
for no fit it contributes.

So a treated set is admissible only when every market in it carries weight
above ``WEIGHT_FLOOR``, the threshold MAREX's results already use to say which
markets are treated. The search, the certified branch and bound and the
certificate all apply it; the MIQP holds it as ``w_j >= MIP_WEIGHT_FLOOR z_j``.

The oracle below enumerates every treated set of the requested sizes and keeps
the admissible ones. Under ``at most m`` the rule changes no optimum: a set
with an unweighted market fits its treated side exactly as well without it, and
leaves the control pool one market larger.

Levels: smoke, unit invariants against the oracle, edge, failure, and
hypothesis properties.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.marex_helpers.branch import branch_and_bound
from mlsynth.utils.marex_helpers.certify import certify_cluster
from mlsynth.utils.marex_helpers.diagnostics import SearchReport
from mlsynth.utils.marex_helpers.exact import (WEIGHT_FLOOR, ClusterDesign,
                                               search_cluster, side_fit)
from mlsynth.utils.syndes_helpers.exact import SubsetFilter
from mlsynth.tests.test_marex_exact_backend import _fit


def _cluster(seed, n=7, T=6, lift=4.0):
    """Markets as columns over the fit window and their mean as the target,
    the standard design's ``X-bar``. One market sits ``lift`` above the rest,
    which puts the target near a face of most treated sets' hulls."""
    rng = np.random.default_rng(seed)
    B = rng.normal(0.0, 1.0, (T, n))
    B[:, 0] += lift
    return B, B.mean(axis=1)


def _value(B, A, S):
    n = B.shape[1]
    g, w = side_fit(B, A, list(S))
    h, _ = side_fit(B, A, [j for j in range(n) if j not in S])
    return g + h, w


def _oracle(B, A, sizes, admissible=True, forced=()):
    """The best treated set over ``sizes``: every set when ``admissible`` is
    False, only the sets whose markets all carry weight when it is True."""
    n = B.shape[1]
    best = (np.inf, None)
    for m in sizes:
        for S in itertools.combinations(range(n), m):
            if not set(forced) <= set(S):
                continue
            value, w = _value(B, A, S)
            if admissible and not np.all(w > WEIGHT_FLOOR):
                continue
            if value < best[0]:
                best = (value, S)
    return best


# Seed 0, m = 3: over every set the best is (1, 3, 6), whose market 1 carries
# no weight; over the admissible sets it is (3, 5, 6).
B0, A0 = _cluster(0)
UNWEIGHTED, WEIGHTED = (1, 3, 6), (3, 5, 6)


# ----------------------------------------------------------------------
# smoke: the panel has the power the tests below rest on
# ----------------------------------------------------------------------


def test_the_unrestricted_best_set_carries_an_unweighted_market():
    value, S = _oracle(B0, A0, [3], admissible=False)
    assert S == UNWEIGHTED
    w = _value(B0, A0, S)[1]
    assert np.sum(w > WEIGHT_FLOOR) == 2


def test_the_search_returns_the_best_set_whose_markets_all_carry_weight():
    d = search_cluster(B0, A0, [3], filt=SubsetFilter())[0]
    assert d.treated == WEIGHTED == _oracle(B0, A0, [3])[1]
    assert np.all(d.w > WEIGHT_FLOOR)
    assert d.value == pytest.approx(_oracle(B0, A0, [3])[0], rel=1e-9)


# ----------------------------------------------------------------------
# unit: every path that fits a treated set applies the rule
# ----------------------------------------------------------------------


def test_the_report_counts_the_sets_refused_for_an_unweighted_market():
    rep = SearchReport()
    search_cluster(B0, A0, [3], filt=SubsetFilter(), report=rep)
    assert rep.counts["unweighted"] >= 1


def test_every_design_in_a_pool_carries_weight_on_every_market():
    designs = search_cluster(B0, A0, [3], filt=SubsetFilter(), top_k=6)
    assert len(designs) == 6
    for d in designs:
        assert np.all(d.w > WEIGHT_FLOOR), d.treated
    assert UNWEIGHTED not in {d.treated for d in designs}


def test_the_branch_and_bound_returns_the_admissible_best():
    found = branch_and_bound(B0, A0, [3], filt=SubsetFilter())
    d = found.designs[0]
    assert d.treated == WEIGHTED
    assert np.all(d.w > WEIGHT_FLOOR)


def test_the_certificate_does_not_hold_an_unweighted_set_against_the_design():
    n = B0.shape[1]
    control = tuple(j for j in range(n) if j not in WEIGHTED)
    g, w = side_fit(B0, A0, list(WEIGHTED))
    h, v = side_fit(B0, A0, list(control))
    d = ClusterDesign(value=g + h, treated=WEIGHTED, control=control, w=w, v=v)
    cert = certify_cluster(B0, A0, [3], SubsetFilter(), d)
    assert UNWEIGHTED not in cert.unresolved
    assert cert.unresolved == ()


def test_the_weakly_targeted_search_applies_the_rule_to_its_coupled_weights():
    for d in search_cluster(B0, A0, [3], filt=SubsetFilter(), beta=1.0, top_k=4):
        assert np.all(d.w > WEIGHT_FLOOR), d.treated


def test_at_most_m_the_rule_changes_no_optimum():
    sizes = [1, 2, 3]
    d = search_cluster(B0, A0, sizes, filt=SubsetFilter())[0]
    assert d.value == pytest.approx(_oracle(B0, A0, sizes, admissible=False)[0],
                                    rel=1e-9)


# ----------------------------------------------------------------------
# edge and failure
# ----------------------------------------------------------------------


def _one_sided(values=(1.0, 2.0, 3.0, 4.0, 10.0), T=3):
    """Every market above a target of zero: a pair's fit puts all its weight
    on the smaller market, so no pair carries weight on both."""
    B = np.outer(np.linspace(0.9, 1.1, T), values)
    return B, np.zeros(T)


def test_a_single_market_always_carries_weight():
    B, A = _one_sided()
    d = search_cluster(B, A, [1], filt=SubsetFilter())[0]
    assert d.w.tolist() == [1.0]


def test_no_set_whose_markets_all_carry_weight_is_reported():
    B, A = _one_sided()
    rep = SearchReport()
    with pytest.raises(MlsynthEstimationError, match="weight"):
        search_cluster(B, A, [2], filt=SubsetFilter(), report=rep)
    assert rep.counts["unweighted"] == 10


def test_the_branch_and_bound_reports_no_admissible_set():
    B, A = _one_sided()
    with pytest.raises(MlsynthEstimationError):
        branch_and_bound(B, A, [2], filt=SubsetFilter())


def test_a_forced_market_the_fit_gives_no_weight_is_reported():
    """Market 4 sits furthest from the target, so beside any other market of
    the one-sided panel it gets no weight; forcing it into a pair leaves no
    admissible design. Over two sides the same market can carry weight."""
    B, A = _one_sided()
    with pytest.raises(MlsynthEstimationError, match="weight"):
        search_cluster(B, A, [2], filt=SubsetFilter(forced_in=frozenset({4})))
    B2, A2 = _one_sided(values=(-1.0, 2.0, 3.0, 4.0, 10.0))
    d = search_cluster(B2, A2, [2], filt=SubsetFilter(forced_in=frozenset({4})))[0]
    assert 4 in d.treated and np.all(d.w > WEIGHT_FLOOR)


# ----------------------------------------------------------------------
# the estimator, on both backends
# ----------------------------------------------------------------------


def _frame(B, post=2, seed=0):
    """``B``'s markets as a long panel, ``post`` periods after the fit window."""
    T, n = B.shape
    rng = np.random.default_rng(seed)
    Y = np.vstack([B, B[-1] + rng.normal(0.0, 0.1, (post, n))]) + 20.0
    labels = [f"u{j}" for j in range(n)]
    return pd.DataFrame({"unit": np.repeat(labels, T + post),
                         "time": np.tile(np.arange(T + post), n),
                         "y": Y.T.reshape(-1)})


@pytest.mark.parametrize("backend", ["exact", "mip"])
def test_m_eq_treats_exactly_m_weighted_markets(backend):
    res = _fit(_frame(B0), B0.shape[0], backend, m_eq=3)
    w = np.asarray(res.globres.treated_weights_agg)
    assert int(np.sum(w > WEIGHT_FLOOR)) == 3
    assert len(res.assignment["treated"]) == 3


def test_a_budget_is_spent_only_on_weighted_markets():
    """With a budget near the cost of the four cheapest markets, the old count
    returned two weighted markets of four, the rest charged and unused."""
    rng = np.random.default_rng(37)
    T, n = 6, 12
    B = rng.normal(0.0, 1.0, (T, n))
    B[:, 0] += 4.0
    costs = rng.uniform(1.0, 10.0, n)
    df = _frame(B)
    df["cost"] = df["unit"].map({f"u{j}": float(c) for j, c in enumerate(costs)})
    budget = float(np.sort(costs)[:4].sum() * 1.15)
    res = _fit(df, T, "exact", m_eq=4, market_cost_col="cost",
               max_total_cost=budget)
    w = np.asarray(res.globres.treated_weights_agg)
    assert int(np.sum(w > WEIGHT_FLOOR)) == 4
    treated = [str(u) for u in res.assignment["treated"]]
    assert len(treated) == 4
    assert sum(float(costs[int(u[1:])]) for u in treated) <= budget + 1e-9


# ----------------------------------------------------------------------
# properties
# ----------------------------------------------------------------------


@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), n=st.integers(4, 7), m=st.integers(1, 3),
       lift=st.floats(0.0, 6.0))
@example(seed=8, n=7, m=3, lift=4.0)
@example(seed=31, n=5, m=2, lift=4.0)
def test_the_search_is_the_admissible_oracle(seed, n, m, lift):
    m = min(m, n - 1)
    B, A = _cluster(seed, n=n, T=5, lift=lift)
    value, S = _oracle(B, A, [m])
    if S is None:
        with pytest.raises(MlsynthEstimationError):
            search_cluster(B, A, [m], filt=SubsetFilter())
        return
    d = search_cluster(B, A, [m], filt=SubsetFilter())[0]
    assert np.all(d.w > WEIGHT_FLOOR)
    assert len(d.treated) == m
    assert d.value == pytest.approx(value, rel=1e-7, abs=1e-10)


@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), n=st.integers(4, 7), m=st.integers(1, 3),
       lift=st.floats(0.0, 6.0))
@example(seed=8, n=7, m=3, lift=4.0)
@example(seed=31, n=5, m=2, lift=4.0)
def test_at_most_m_the_optimum_is_the_unrestricted_one(seed, n, m, lift):
    m = min(m, n - 1)
    B, A = _cluster(seed, n=n, T=5, lift=lift)
    sizes = list(range(1, m + 1))
    d = search_cluster(B, A, sizes, filt=SubsetFilter())[0]
    assert np.all(d.w > WEIGHT_FLOOR)
    assert d.value == pytest.approx(_oracle(B, A, sizes, admissible=False)[0],
                                    rel=1e-7, abs=1e-10)


@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), n=st.integers(4, 7), m=st.integers(2, 3),
       lift=st.floats(0.0, 6.0))
@example(seed=8, n=7, m=3, lift=4.0)
@example(seed=31, n=5, m=2, lift=4.0)
def test_the_branch_and_bound_is_the_admissible_oracle(seed, n, m, lift):
    m = min(m, n - 1)
    B, A = _cluster(seed, n=n, T=5, lift=lift)
    value, S = _oracle(B, A, [m])
    if S is None:
        with pytest.raises(MlsynthEstimationError):
            branch_and_bound(B, A, [m], filt=SubsetFilter())
        return
    d = branch_and_bound(B, A, [m], filt=SubsetFilter()).designs[0]
    assert np.all(d.w > WEIGHT_FLOOR)
    assert d.value == pytest.approx(value, rel=1e-7, abs=1e-10)


# ----------------------------------------------------------------------
# the MIQP's cut loop, on a stubbed solve
# ----------------------------------------------------------------------


def _raw(w):
    """A one-cluster design dict with treated weights ``w`` on markets 0..2."""
    w = np.asarray(w, dtype=float)[:, None]
    z = (np.arange(len(w)) < 3).astype(float)[:, None]
    return {"z_opt": z, "w_opt": w}


def _stub(monkeypatch, outcomes):
    """Replace one MIQP solve with ``outcomes``, in order, recording the cuts
    each call was given."""
    from mlsynth.utils.marex_helpers import optimization as opt
    seen = []

    def once(**kwargs):
        seen.append([list(c) for c in kwargs["forbidden"]])
        out = outcomes[min(len(seen), len(outcomes)) - 1]
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(opt, "_solve_design_once", once)
    return opt, seen


def test_a_weighted_design_is_returned_with_no_cut(monkeypatch):
    good = _raw([0.5, 0.3, 0.2, 0.0])
    opt, seen = _stub(monkeypatch, [good])
    assert opt.solve_design(None, 1, None) is good
    assert seen == [[]]


def test_an_unweighted_set_is_cut_and_the_program_solved_again(monkeypatch):
    bad, good = _raw([0.6, 0.4, 0.0, 0.0]), _raw([0.5, 0.3, 0.2, 0.0])
    opt, seen = _stub(monkeypatch, [bad, good])
    assert opt.solve_design(None, 1, None, forbidden=[[(3, 0)]]) is good
    assert seen == [[[(3, 0)]], [[(3, 0)], [(0, 0), (1, 0), (2, 0)]]]


def test_a_program_left_infeasible_by_the_cuts_is_reported(monkeypatch):
    bad = _raw([0.6, 0.4, 0.0, 0.0])
    opt, _ = _stub(monkeypatch, [bad, MlsynthEstimationError("infeasible")])
    with pytest.raises(MlsynthEstimationError, match="every treated market carries"):
        opt.solve_design(None, 1, None)


def test_an_infeasible_program_without_cuts_keeps_its_own_message(monkeypatch):
    opt, _ = _stub(monkeypatch, [MlsynthEstimationError("the restrictions clash")])
    with pytest.raises(MlsynthEstimationError, match="the restrictions clash"):
        opt.solve_design(None, 1, None)


def test_the_cut_loop_stops_and_reports(monkeypatch):
    opt, seen = _stub(monkeypatch, [_raw([0.6, 0.4, 0.0, 0.0])])
    monkeypatch.setattr(opt, "MAX_WEIGHT_CUTS", 3)
    with pytest.raises(MlsynthEstimationError, match="unweighted market 3 times"):
        opt.solve_design(None, 1, None)
    assert len(seen) == 4


def test_every_cluster_is_checked(monkeypatch):
    """Cluster 0 is weighted; cluster 1's set has market 4 at zero weight, so
    the cut is that set in cluster 1."""
    z = np.zeros((6, 2)); w = np.zeros((6, 2))
    z[[0, 1], 0], w[[0, 1], 0] = 1.0, [0.5, 0.5]
    z[[3, 4], 1], w[[3, 4], 1] = 1.0, [1.0, 0.0]
    bad = {"z_opt": z, "w_opt": w}
    good = _raw([0.5, 0.3, 0.2, 0.0])
    opt, seen = _stub(monkeypatch, [bad, good])
    opt.solve_design(None, 1, None)
    assert seen[1] == [[(3, 1), (4, 1)]]
