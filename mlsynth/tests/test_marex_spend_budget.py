"""MAREX's total-spend budget: sum_j c_j z_j <= B.

A test costs money market by market (the media held back, the fees to run it),
so a design must fit a budget on the total cost of the markets it treats.
``market_cost_col`` names each market's cost and ``max_total_cost`` caps their
sum over the treated set. It is a rule on which markets are treated, like a
border conflict or a stratum quota, so the MIQP takes it as one linear row on
``z`` and the exact backend drops every treated set over budget before fitting
it, which leaves its bounds and its proof of optimality as they are.

The oracle below enumerates every admissible treated set and fits both
composites with Clarabel, importing nothing from the search.

Levels: smoke, unit invariants (against the oracle, on both backends), edge,
failure, and a hypothesis property.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import (MlsynthConfigError, MlsynthDataError,
                                MlsynthEstimationError)
from mlsynth.tests.test_marex_exact_backend import _fit, _panel, _treated
from mlsynth.tests.test_marex_exact_oracle import _predictors, _simplex_fit

M = 2


def _with_costs(df, costs):
    out = df.copy()
    labels = list(dict.fromkeys(out["unit"]))
    out["cost"] = out["unit"].map(dict(zip(labels, map(float, costs))))
    return out


def _budget_oracle(df, T0, costs, budget, m=M):
    """The standard design's minimum over treated sets within budget."""
    labels, X = _predictors(df, T0)
    A = X.mean(axis=0)
    best = (np.inf, None)
    for S in itertools.combinations(range(X.shape[0]), m):
        if sum(costs[i] for i in S) > budget + 1e-9:
            continue
        C = [i for i in range(X.shape[0]) if i not in S]
        f = _simplex_fit(X[list(S)].T, A) + _simplex_fit(X[C].T, A)
        if f < best[0]:
            best = (f, S)
    if best[1] is None:
        return np.inf, None
    return best[0], frozenset(labels[i] for i in best[1])


def _budgeted(df, T0, backend, costs, budget, **over):
    return _fit(_with_costs(df, costs), T0, backend, m_eq=M,
                market_cost_col="cost", max_total_cost=budget, **over)


def _cost_of(res, df, costs):
    labels = list(dict.fromkeys(df["unit"]))
    by = dict(zip(labels, costs))
    return sum(by[u] for u in _treated(res))


@pytest.fixture(scope="module")
def panel():
    return _panel(J=9, seed=0)


@pytest.fixture(scope="module")
def free(panel):
    df, T0 = panel
    return _fit(df, T0, "exact", m_eq=M)


@pytest.fixture(scope="module")
def costs(panel, free):
    """The free design's markets cost 5 each, every other market 1."""
    df, _ = panel
    labels = list(dict.fromkeys(df["unit"]))
    chosen = _treated(free)
    return [5.0 if u in chosen else 1.0 for u in labels]


# ---------------------------------------------------------------- smoke
def test_smoke_a_binding_budget_gives_a_design_within_it(panel, costs):
    df, T0 = panel
    res = _budgeted(df, T0, "exact", costs, budget=6.0)
    assert np.isfinite(res.globres.objective)
    assert len(_treated(res)) == M
    assert _cost_of(res, df, costs) <= 6.0


# ---------------------------------------------------------------- unit: against the oracle
@pytest.mark.parametrize("backend", ["exact", "mip"])
@pytest.mark.parametrize("budget", [2.0, 6.0, 10.0])
def test_the_design_is_the_best_within_budget(panel, costs, backend, budget):
    df, T0 = panel
    f, S = _budget_oracle(df, T0, costs, budget)
    res = _budgeted(df, T0, backend, costs, budget)
    assert _treated(res) == S
    assert res.globres.objective == pytest.approx(f, rel=1e-5, abs=1e-8)


def test_a_binding_budget_moves_the_design_and_costs_fit(panel, free, costs):
    df, T0 = panel
    res = _budgeted(df, T0, "exact", costs, budget=6.0)
    assert _treated(res) != _treated(free)
    assert res.globres.objective >= free.globres.objective - 1e-9


@pytest.mark.parametrize("backend", ["exact", "mip"])
def test_a_budget_every_design_meets_changes_nothing(panel, free, costs, backend):
    df, T0 = panel
    res = _budgeted(df, T0, backend, costs, budget=sum(costs))
    assert _treated(res) == _treated(free)
    assert res.globres.objective == pytest.approx(free.globres.objective, rel=1e-5, abs=1e-8)


def test_the_branch_and_bound_honours_the_budget(panel, costs, monkeypatch):
    """Above the candidate limit the search is a branch and bound."""
    from mlsynth.utils.marex_helpers import exact
    df, T0 = panel
    want = _budgeted(df, T0, "exact", costs, budget=6.0)
    monkeypatch.setattr(exact, "CANDIDATE_LIMIT", 0)
    got = _budgeted(df, T0, "exact", costs, budget=6.0)
    assert got.search.search == "branch_and_bound"
    assert _treated(got) == _treated(want)
    assert got.globres.objective == pytest.approx(want.globres.objective, rel=1e-9)


def test_the_certificate_holds_under_a_budget(panel, costs):
    df, T0 = panel
    res = _budgeted(df, T0, "exact", costs, budget=6.0, certify=True)
    f = res.globres.objective
    assert res.certificate.lower <= f + 1e-9 <= res.certificate.upper + 2e-9


def test_every_pooled_design_is_within_budget(panel, costs):
    df, T0 = panel
    res = _budgeted(df, T0, "exact", costs, budget=6.0, top_K=4)
    labels = list(dict.fromkeys(df["unit"]))
    by = dict(zip(labels, costs))
    assert res.pool
    for entry in res.pool:
        assert sum(by[str(u)] for u in entry["markets"]) <= 6.0


def test_each_market_is_charged_its_own_cost():
    """With twelve markets the panel orders u0, u1, u2, ..., u10, u11 and the
    cost column sorts u0, u1, u10, u11, u2, ...; only u10 and u11 are affordable,
    so the design must be exactly that pair."""
    df, T0 = _panel(J=12, seed=3)
    labels = list(dict.fromkeys(df["unit"]))
    costs = [1.0 if u in ("u10", "u11") else 9.0 for u in labels]
    res = _budgeted(df, T0, "exact", costs, budget=2.0)
    assert _treated(res) == frozenset({"u10", "u11"})


# ---------------------------------------------------------------- edge cases
def test_a_set_costing_exactly_the_budget_is_admitted(panel, costs):
    df, T0 = panel
    f, S = _budget_oracle(df, T0, costs, 6.0)
    exact_cost = sum(c for u, c in zip(dict.fromkeys(df["unit"]), costs) if u in S)
    res = _budgeted(df, T0, "exact", costs, budget=exact_cost)
    assert _treated(res) == S


def test_free_markets_never_bind(panel, free):
    df, T0 = panel
    res = _budgeted(df, T0, "exact", [0.0] * 9, budget=1.0)
    assert _treated(res) == _treated(free)


@pytest.mark.parametrize("backend", ["exact", "mip"])
def test_a_budget_no_set_meets_is_reported(panel, costs, backend):
    df, T0 = panel
    with pytest.raises(MlsynthEstimationError):
        _budgeted(df, T0, backend, [3.0] * 9, budget=5.0)


def test_the_branch_and_bound_reports_a_budget_no_set_meets(panel, monkeypatch):
    from mlsynth.utils.marex_helpers import exact
    df, T0 = panel
    monkeypatch.setattr(exact, "CANDIDATE_LIMIT", 0)
    with pytest.raises(MlsynthEstimationError, match="admissible treated set"):
        _budgeted(df, T0, "exact", [3.0] * 9, budget=5.0)


def test_a_forced_market_over_budget_is_reported(panel, costs, free):
    df, T0 = panel
    expensive = sorted(_treated(free))[0]
    with pytest.raises(MlsynthEstimationError):
        _budgeted(df, T0, "exact", costs, budget=5.5, to_be_treated=[expensive],
                  not_to_be_treated=None)


# ---------------------------------------------------------------- failures
def test_cost_without_a_budget_is_refused(panel, costs):
    df, T0 = panel
    with pytest.raises(MlsynthConfigError, match="max_total_cost"):
        _fit(_with_costs(df, costs), T0, "exact", m_eq=M, market_cost_col="cost")


def test_a_budget_without_costs_is_refused(panel):
    df, T0 = panel
    with pytest.raises(MlsynthConfigError, match="max_total_cost needs market_cost_col"):
        _fit(df, T0, "exact", m_eq=M, max_total_cost=5.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_budget_that_is_not_a_positive_number_is_refused(panel, costs, bad):
    df, T0 = panel
    with pytest.raises(MlsynthConfigError, match="max_total_cost"):
        _budgeted(df, T0, "exact", costs, budget=bad)


def test_an_unknown_cost_column_is_refused(panel):
    df, T0 = panel
    with pytest.raises(MlsynthConfigError, match="market_cost_col"):
        _fit(df, T0, "exact", m_eq=M, market_cost_col="nope", max_total_cost=5.0)


def test_a_cost_that_varies_within_a_market_is_refused(panel, costs):
    df, T0 = panel
    bad = _with_costs(df, costs)
    bad.loc[bad.index[0], "cost"] = 99.0
    with pytest.raises(MlsynthDataError, match="market_cost_col"):
        _fit(bad, T0, "exact", m_eq=M, market_cost_col="cost", max_total_cost=5.0)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_a_negative_or_missing_cost_is_refused(panel, costs, bad):
    df, T0 = panel
    c = list(costs)
    c[3] = bad
    with pytest.raises(MlsynthDataError, match="market_cost_col"):
        _budgeted(df, T0, "exact", c, budget=5.0)


def test_the_relaxed_program_refuses_a_budget(panel, costs):
    df, T0 = panel
    with pytest.raises(MlsynthConfigError, match="relaxed"):
        _budgeted(df, T0, "mip", costs, budget=6.0, relaxed=True)


def test_the_exact_backend_refuses_a_budget_across_clusters():
    df, T0 = _panel(J=12, seed=3, region_sizes=(6, 6))
    with pytest.raises(MlsynthConfigError, match="backend='mip'"):
        _fit(_with_costs(df, [1.0] * 12), T0, "exact", m_eq=1, cluster="region",
             market_cost_col="cost", max_total_cost=5.0)


def test_the_mip_takes_a_budget_across_clusters():
    """Region 2's only affordable market is the one it must treat, whatever the
    free design would pick there; the budget counts both regions' markets."""
    df, T0 = _panel(J=12, seed=3, region_sizes=(6, 6))
    labels = list(dict.fromkeys(df["unit"]))
    free = _fit(df, T0, "mip", m_eq=1, cluster="region")
    second = [u for u in labels[6:] if u not in _treated(free)][0]
    costs = [1.0] * 6 + [1.0 if u == second else 9.0 for u in labels[6:]]
    res = _fit(_with_costs(df, costs), T0, "mip", m_eq=1, cluster="region",
               market_cost_col="cost", max_total_cost=2.0)
    by = dict(zip(labels, costs))
    assert second in _treated(res)
    assert sum(by[u] for u in _treated(res)) <= 2.0


# ---------------------------------------------------------------- property
@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(costs=st.lists(st.sampled_from([0.0, 1.0, 2.0, 3.0, 5.0]), min_size=7, max_size=7),
       budget=st.sampled_from([1.0, 2.0, 3.0, 4.0, 6.0, 10.0]),
       seed=st.integers(0, 3))
def test_the_exact_design_is_the_oracle_within_budget(costs, budget, seed):
    """Over costs, budgets and panels: the design keeps to the budget and is the
    best treated set that does, or the fit reports that none exists."""
    df, T0 = _panel(J=7, seed=seed)
    f, S = _budget_oracle(df, T0, costs, budget)
    if S is None:
        with pytest.raises(MlsynthEstimationError):
            _budgeted(df, T0, "exact", costs, budget)
        return
    res = _budgeted(df, T0, "exact", costs, budget)
    assert _cost_of(res, df, costs) <= budget + 1e-9
    assert res.globres.objective == pytest.approx(f, rel=1e-5, abs=1e-8)
