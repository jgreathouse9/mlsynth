"""MAREX's exact backend under the weighted-average cost constraint (#953).

``costs`` and ``budget`` bound each cluster's treated weights,
``sum_j c_j w_j <= B_k``: the synthetic treated unit's average cost per
market. With the treated set fixed the constraint is one more linear
inequality in that set's fit, which stays a convex quadratic program. When the
set's unconstrained fit meets the budget it is the answer; otherwise the
inequality binds, and the fit is the simplex fit with a Lagrange multiplier on
the costs as a linear term, the multiplier found by a one-dimensional search.
A set whose cheapest market already costs more than the budget has no
admissible weights. The bound the search prunes with is unchanged: adding a
constraint only raises the minimum.

The oracle below solves each set's budgeted fit with Clarabel, importing
nothing from the search. Levels: smoke, unit (the fit, the design against the
oracle and against the MIQP), edge, failure, and hypothesis properties.
"""
from __future__ import annotations

import itertools

import cvxpy as cp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.tests.test_marex_exact_backend import _fit, _panel
from mlsynth.tests.test_marex_exact_oracle import _predictors
from mlsynth.utils.marex_helpers import exact as exact_mod
from mlsynth.utils.marex_helpers.exact import budgeted_side_fit, side_fit

CARRIED = 1e-8


def _qp(D, y, costs=None, budget=None):
    """``min ||y - D w||^2`` over the simplex, with ``costs'w <= budget`` when
    given, by Clarabel: ``(value, w)``, or ``None`` when infeasible."""
    w = cp.Variable(D.shape[1], nonneg=True)
    cons = [cp.sum(w) == 1]
    if costs is not None:
        cons.append(costs @ w <= budget)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(y - D @ w)), cons)
    try:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    except cp.error.SolverError:     # a budget near the cheapest cost leaves a sliver
        try:
            prob.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:
            prob.solve(solver=cp.SCS, eps=1e-10, max_iters=200_000)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    return float(prob.value), np.asarray(w.value, dtype=float)


def _oracle(X, groups, m, costs, budgets):
    """The standard design's minimum, each cluster's treated weights within its
    budget and weighting every treated market: ``inf`` when none qualifies."""
    total = 0.0
    for members, B_k in zip(groups, budgets):
        members = list(members)
        A = X[members].mean(axis=0)
        best = np.inf
        for S in itertools.combinations(range(len(members)), m):
            cols = [members[i] for i in S]
            if min(costs[cols]) > B_k:
                continue
            fit = _qp(X[cols].T, A, costs[cols], B_k)
            if fit is None or np.any(fit[1] <= CARRIED):
                continue
            C = [members[i] for i in range(len(members)) if i not in S]
            best = min(best, fit[0] + _qp(X[C].T, A)[0])
        if not np.isfinite(best):
            return np.inf
        total += best
    return total


COSTS = np.array([1.0, 4.0, 2.0, 8.0, 3.0, 5.0, 1.5, 6.0, 2.5])


# ---------------------------------------------------------------- the fit
def _cluster(seed=0, n=6, T=8):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(T, n)) + rng.normal(size=(1, n))
    return B, B.mean(axis=1)


def test_a_slack_budget_leaves_the_fit_alone():
    B, A = _cluster()
    cols = [0, 2, 4]
    plain = side_fit(B, A, cols)
    c = np.array([1.0, 2.0, 3.0])
    got = budgeted_side_fit(B, A, cols, costs=c, budget=float(c @ plain[1]) + 1.0)
    assert got[0] == pytest.approx(plain[0], rel=1e-12)
    np.testing.assert_allclose(got[1], plain[1], atol=1e-12)


def test_a_binding_budget_is_the_constrained_optimum():
    B, A = _cluster()
    cols = [0, 2, 4]
    c = np.array([1.0, 2.0, 3.0])
    plain = side_fit(B, A, cols)
    budget = 0.5 * (float(c @ plain[1]) + c.min())      # between the cheapest and the fit's
    value, w = budgeted_side_fit(B, A, cols, costs=c, budget=budget)
    ref_value, _ = _qp(B[:, cols], A, c, budget)
    assert value == pytest.approx(ref_value, rel=1e-7, abs=1e-10)
    assert float(c @ w) <= budget + 1e-9
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0)
    assert value >= plain[0] - 1e-12                      # a constraint only raises it


def test_a_budget_below_every_cost_has_no_fit():
    B, A = _cluster()
    assert budgeted_side_fit(B, A, [0, 2, 4], costs=np.array([2.0, 3.0, 4.0]), budget=1.5) is None


def test_a_budget_at_the_cheapest_cost_puts_all_weight_there():
    B, A = _cluster()
    value, w = budgeted_side_fit(B, A, [0, 2, 4], costs=np.array([2.0, 3.0, 4.0]), budget=2.0)
    np.testing.assert_allclose(w, [1.0, 0.0, 0.0], atol=1e-9)


def test_the_penalty_term_is_kept_in_the_value():
    B, A = _cluster()
    cols = [0, 2, 4]
    c, lin = np.array([1.0, 2.0, 3.0]), np.array([0.3, 0.1, 0.2])
    plain = side_fit(B, A, cols, lin)
    budget = 0.5 * (float(c @ plain[1]) + c.min())
    value, w = budgeted_side_fit(B, A, cols, lin, costs=c, budget=budget)
    r = A - B[:, cols] @ w
    assert value == pytest.approx(float(r @ r + lin @ w), rel=1e-12)


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10_000), k=st.integers(1, 5), frac=st.floats(0.0, 1.2))
def test_the_budgeted_fit_is_the_conic_optimum(seed, k, frac):
    """Over random sets, costs and budgets from below the cheapest market to
    above the unconstrained fit's cost: the same feasibility and value as a
    conic solve, and the budget held."""
    rng = np.random.default_rng(seed)
    B, A = _cluster(seed, n=6)
    cols = sorted(rng.choice(6, size=k, replace=False).tolist())
    c = rng.uniform(1.0, 10.0, k)
    lo, hi = c.min(), max(float(c @ side_fit(B, A, cols)[1]), c.min())
    budget = lo - 0.5 + frac * (hi - lo + 0.5)
    got = budgeted_side_fit(B, A, cols, costs=c, budget=budget)
    ref = _qp(B[:, cols], A, c, budget) if c.min() <= budget else None
    if ref is None:
        assert got is None
        return
    assert got is not None
    assert got[0] == pytest.approx(ref[0], rel=1e-6, abs=1e-9)
    assert float(c @ got[1]) <= budget + 1e-9


# ---------------------------------------------------------------- the design
def _budgets_for(df, T0, frac):
    """A budget between the cheapest treated average and the free design's."""
    free = _fit(df, T0, "exact", m_eq=3)
    w = np.asarray(free.globres.treated_weights_agg)
    return float(COSTS.min() + frac * (COSTS @ w - COSTS.min()))


def test_the_exact_design_is_the_oracle_under_a_binding_budget():
    df, T0 = _panel(seed=1)
    budget = _budgets_for(df, T0, 0.5)
    res = _fit(df, T0, "exact", m_eq=3, costs=COSTS.tolist(), budget=budget)
    _, X = _predictors(df, T0)
    assert res.globres.objective == pytest.approx(_oracle(X, [range(9)], 3, COSTS, [budget]), rel=1e-6)
    w = np.asarray(res.globres.treated_weights_agg)
    assert float(COSTS @ w) <= budget + 1e-8


def test_the_exact_design_matches_the_mip():
    df, T0 = _panel(seed=1)
    budget = _budgets_for(df, T0, 0.5)
    ex = _fit(df, T0, "exact", m_eq=3, costs=COSTS.tolist(), budget=budget)
    mip = _fit(df, T0, "mip", m_eq=3, costs=COSTS.tolist(), budget=budget)
    assert ex.globres.objective <= mip.globres.objective + 1e-6 * (1 + abs(mip.globres.objective))
    assert ex.globres.objective == pytest.approx(mip.globres.objective, rel=1e-4)


def test_a_slack_budget_changes_nothing():
    df, T0 = _panel(seed=1)
    free = _fit(df, T0, "exact", m_eq=3)
    res = _fit(df, T0, "exact", m_eq=3, costs=COSTS.tolist(), budget=1000)
    assert res.globres.objective == pytest.approx(free.globres.objective, rel=1e-12)


def _by_label(df, costs_in_panel_order):
    """``costs`` is positional in MAREX's unit order, which sorts the labels:
    with twelve units that is u0, u1, u10, u11, u2, ..., not the panel's order."""
    labels = list(dict.fromkeys(df["unit"]))
    cost_of = dict(zip(labels, costs_in_panel_order))
    return [float(cost_of[u]) for u in sorted(labels)]


def test_each_cluster_holds_its_own_budget():
    df, T0 = _panel(J=12, seed=3, region_sizes=(6, 6))
    costs = np.tile(COSTS[:6], 2)                     # in the panel's order
    budgets = {0: 2.5, 1: 3.5}
    res = _fit(df, T0, "exact", m_eq=2, cluster="region", costs=_by_label(df, costs),
               budget=budgets)
    _, X = _predictors(df, T0)
    want = _oracle(X, [range(6), range(6, 12)], 2, costs, [2.5, 3.5])
    assert res.globres.objective == pytest.approx(want, rel=1e-6)


@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 5_000), m=st.integers(1, 3), frac=st.floats(0.0, 1.0))
def test_the_exact_design_is_the_oracle_over_the_domain(seed, m, frac):
    df, T0 = _panel(J=7, seed=seed, n_good=2)
    costs = np.random.default_rng(seed).uniform(1.0, 9.0, 7)
    budget = float(costs.min() + frac * (costs.max() - costs.min()))
    _, X = _predictors(df, T0)
    want = _oracle(X, [range(7)], m, costs, [budget])
    fit = lambda: _fit(df, T0, "exact", m_eq=m, costs=costs.tolist(), budget=budget)
    if not np.isfinite(want):
        with pytest.raises(MlsynthEstimationError):
            fit()
        return
    assert fit().globres.objective == pytest.approx(want, rel=1e-6)


# ---------------------------------------------------------------- edge and failure
def test_a_budget_no_set_can_meet_is_reported():
    df, T0 = _panel(seed=1)
    with pytest.raises(MlsynthEstimationError):
        _fit(df, T0, "exact", m_eq=3, costs=COSTS.tolist(), budget=0.5)


def test_the_report_counts_the_sets_the_budget_refuses():
    df, T0 = _panel(seed=1)
    budget = _budgets_for(df, T0, 0.2)
    res = _fit(df, T0, "exact", m_eq=3, costs=COSTS.tolist(), budget=budget)
    assert res.search.counts.over_weight_budget > 0


@pytest.mark.parametrize("over, fragment", [
    (dict(design="weakly_targeted", beta=1.0), "weakly"),
    (dict(certify=True), "certif"),
])
def test_what_the_budgeted_fit_does_not_cover_is_refused(over, fragment):
    df, T0 = _panel(seed=1)
    with pytest.raises(MlsynthConfigError, match=fragment):
        _fit(df, T0, "exact", m_eq=3, costs=COSTS.tolist(), budget=3, **over)


def test_the_branch_and_bound_refuses_the_constraint(monkeypatch):
    monkeypatch.setattr(exact_mod, "CANDIDATE_LIMIT", 10)
    df, T0 = _panel(seed=1)
    with pytest.raises(MlsynthConfigError, match="candidate limit"):
        _fit(df, T0, "exact", m_eq=3, costs=COSTS.tolist(), budget=3)
