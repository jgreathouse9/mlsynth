"""SYNDES per-unit design by a branch and bound on units (#806).

Doudchenko et al. (2021, Appendix A.3): the per-unit problem is ``K`` one-way
problems, one per treated unit, each with every control as a donor. For a
treated set ``S``,

    f(S) = (1/K) sum_{i in S} phi_i(S^c),
    phi_i(P) = min over v on the simplex over P of (1/T)||y_i - Y_P v||^2 + lam ||v||^2.

``phi_i`` only rises as its pool shrinks. A node that fixes a set ``F`` treated
bounds every completion by ``F``'s fits on pools without ``F`` plus the
``K - |F|`` smallest fits of the undecided units on pools without ``F``, and the
bound equals ``f(S)`` at a leaf. The study behind the choice is
``benchmarks/studies/syndes_per_unit_exact/``.

The anchor is brute force: every treated set priced from fresh fits.
"""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES
from mlsynth.exceptions import (MlsynthConfigError, MlsynthDataError,
                                MlsynthEstimationError)
from mlsynth.utils.solvers.active_set import solve_simplex_qp
from mlsynth.utils.syndes_helpers.certificate import design_objective
from mlsynth.utils.syndes_helpers.optimization import (SYNDESSolveWarning,
                                                        estimate_lambda)
from mlsynth.utils.syndes_helpers.per_unit_exact import (
    UnitFits,
    solve_per_unit_exact,
    solve_per_unit_exact_pool,
)
from mlsynth.utils.syndes_helpers.restrictions import DesignRestrictions


def _panel(N=8, T=12, seed=0, rank=3, noise=0.3):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, rank)) @ rng.standard_normal((rank, N))
            + noise * rng.standard_normal((T, N)))


def _phi(Y, i, pool, lam):
    """``phi_i`` on ``pool`` from a fresh fit; infinite on an empty pool."""
    if not pool:
        return np.inf
    T = Y.shape[0]
    B = np.vstack([Y[:, pool] / np.sqrt(T), np.sqrt(lam) * np.eye(len(pool))])
    A = np.concatenate([Y[:, i] / np.sqrt(T), np.zeros(len(pool))])
    w = solve_simplex_qp(B, A)
    r = A - B @ w
    return float(r @ r)


def _brute_force(Y, K, lam, accepts=lambda S: True, excluded=None):
    """The optimum over every admissible treated set, from fresh fits."""
    excluded = excluded or {}
    N = Y.shape[1]
    best, best_set = np.inf, None
    for S in combinations(range(N), K):
        if not accepts(S):
            continue
        value = sum(_phi(Y, i, [j for j in range(N) if j not in S
                                and j not in excluded.get(i, ())], lam)
                    for i in S) / K
        if value < best:
            best, best_set = value, S
    return best_set, best


def _treated(design):
    return tuple(int(i) for i in design.selected_unit_indices)


def _df(Y, n_post=4):
    T, N = Y.shape
    return pd.DataFrame([{"unit": j, "time": t, "y": float(Y[t, j]),
                          "post": int(t >= T - n_post)}
                         for j in range(N) for t in range(T)])


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #


def test_a_panel_gets_a_certified_per_unit_design():
    Y = _panel()
    d = solve_per_unit_exact(Y, 3)
    assert d.mode == "per_unit" and len(_treated(d)) == 3
    assert np.isfinite(d.objective_value)
    out = d.solve_outcome
    assert out.status == "optimal" and out.solver_status == "exact"
    assert not out.stopped_at_limit and out.gap == 0.0
    assert d.raw_results["certified"] is True


# --------------------------------------------------------------------------- #
# the design is the optimum, and its weights attain it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("K", [1, 2, 3, 5])
def test_the_design_is_the_brute_force_optimum(seed, K):
    Y = _panel(seed=seed)
    lam = estimate_lambda(Y)
    S, value = _brute_force(Y, K, lam)
    d = solve_per_unit_exact(Y, K)
    assert _treated(d) == S
    assert d.objective_value == pytest.approx(value, rel=1e-9)


def test_the_objective_is_what_the_weights_attain():
    Y = _panel(seed=5)
    d = solve_per_unit_exact(Y, 3)
    assert design_objective(Y, "per_unit", d, d.lambda_value) == pytest.approx(
        d.objective_value, rel=1e-10)


def test_each_treated_unit_is_fitted_from_controls_on_the_simplex():
    Y = _panel(seed=6)
    d = solve_per_unit_exact(Y, 3)
    Q = np.asarray(d.treated_weights)
    S = list(_treated(d))
    controls = [j for j in range(Y.shape[1]) if j not in S]
    assert (Q >= 0).all()
    np.testing.assert_allclose(Q[S].sum(axis=1), 1.0, atol=1e-12)
    assert np.all(Q[:, S] == 0.0) and np.all(Q[controls] == 0.0)


def test_the_contrast_is_the_mean_of_the_unit_contrasts():
    Y = _panel(seed=7)
    d = solve_per_unit_exact(Y, 3)
    Q = np.asarray(d.treated_weights)
    np.testing.assert_allclose(d.contrast_weights,
                               (d.assignment - Q.sum(axis=0)) / 3, atol=1e-15)
    np.testing.assert_allclose(d.contrast_series, Y @ d.contrast_weights, atol=1e-12)
    assert d.pre_fit_rmse == pytest.approx(
        float(np.sqrt(np.mean(d.contrast_series ** 2))), rel=1e-12)


def test_the_default_penalty_is_the_mip_paths():
    Y = _panel(seed=8)
    assert solve_per_unit_exact(Y, 2).lambda_value == pytest.approx(
        estimate_lambda(Y), rel=1e-15)


def test_the_design_matches_scips_proven_optimum():
    pytest.importorskip("pyscipopt")
    from mlsynth.utils.syndes_helpers.optimization import solve_synthetic_design
    Y = _panel(N=8, seed=9)
    Y = Y / Y.std()                         # SCIP's tolerances are absolute (#814)
    mip = solve_synthetic_design(Y, 3, mode="per_unit", gap_limit=0.0,
                                 time_limit=None)
    d = solve_per_unit_exact(Y, 3)
    assert _treated(d) == tuple(int(i) for i in mip.selected_unit_indices)
    assert d.objective_value <= mip.objective_value * (1 + 1e-7)


@pytest.mark.parametrize("s", [1e-5, 1e5])
def test_a_panel_in_other_units_gets_the_same_design(s):
    """The active-set solver's tolerances are absolute; the fits run on a
    centred, unit-scale panel so they decide nothing. Hypothesis found a design
    1% worse than the optimum at ``s = 1e-5`` before they did."""
    Y = np.array([[0.31893615, 0.07530866, 0.51798845, -0.69944032, 0.57325405],
                  [0.96784802, 0.61540995, 1.98204415, -1.28630427, 0.15166373],
                  [-0.11831296, 0.36101934, -1.09201495, -1.07810411, 1.98241989]])
    ref = solve_per_unit_exact(Y, 3)
    got = solve_per_unit_exact(Y * s, 3)
    assert _treated(got) == _treated(ref)
    assert got.objective_value / s ** 2 == pytest.approx(ref.objective_value, rel=1e-9)


# --------------------------------------------------------------------------- #
# the fits
# --------------------------------------------------------------------------- #


def test_reused_fits_equal_fresh_ones():
    Y = _panel(N=7, seed=10)
    lam = estimate_lambda(Y)
    fits = UnitFits(Y, lam)
    for S in combinations(range(7), 3):
        removed = frozenset(S)
        for i in S:
            pool = [j for j in range(7) if j not in S]
            assert fits.phi(i, removed) == pytest.approx(_phi(Y, i, pool, lam), rel=1e-9)


def test_a_fit_that_does_not_converge_is_reported(monkeypatch):
    import mlsynth.utils.syndes_helpers.per_unit_exact as pue

    def stalled(B, A, **kw):
        w = np.full(B.shape[1], 1.0 / B.shape[1])
        return (w, {"converged": False, "pivots": 0}) if kw.get("return_info") else w
    monkeypatch.setattr(pue, "solve_simplex_qp", stalled)
    with pytest.raises(MlsynthEstimationError, match="converge"):
        solve_per_unit_exact(_panel(), 2)


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #


def test_one_treated_unit_is_the_best_fitted_unit():
    Y = _panel(seed=11)
    lam = estimate_lambda(Y)
    roots = [_phi(Y, i, [j for j in range(8) if j != i], lam) for i in range(8)]
    d = solve_per_unit_exact(Y, 1)
    assert _treated(d) == (int(np.argmin(roots)),)


def test_all_but_one_treated_puts_every_weight_on_the_last_control():
    Y = _panel(N=5, seed=12)
    d = solve_per_unit_exact(Y, 4)
    (c,) = [j for j in range(5) if j not in _treated(d)]
    Q = np.asarray(d.treated_weights)
    np.testing.assert_allclose(Q[list(_treated(d)), c], 1.0, atol=1e-12)


def test_two_units():
    Y = _panel(N=2, T=6, seed=13)
    lam = estimate_lambda(Y)
    S, value = _brute_force(Y, 1, lam)
    d = solve_per_unit_exact(Y, 1)
    assert _treated(d) == S and d.objective_value == pytest.approx(value, rel=1e-9)


def test_duplicate_units():
    Y = _panel(N=6, seed=14)
    Y[:, 5] = Y[:, 0]
    lam = estimate_lambda(Y)
    _, value = _brute_force(Y, 2, lam)
    assert solve_per_unit_exact(Y, 2).objective_value == pytest.approx(value, rel=1e-9)


def test_no_penalty():
    Y = _panel(N=6, T=20, seed=15)
    S, value = _brute_force(Y, 2, 0.0)
    d = solve_per_unit_exact(Y, 2, lam=0.0)
    assert d.objective_value == pytest.approx(value, rel=1e-8, abs=1e-12)


# --------------------------------------------------------------------------- #
# restrictions
# --------------------------------------------------------------------------- #


def test_donor_exclusions_shrink_the_treated_units_pools():
    Y = _panel(seed=16)
    lam = estimate_lambda(Y)
    pairs = [(i, j) for i in range(8) for j in range(8) if i != j and (i + j) % 3 == 0]
    excluded = {}
    for i, j in pairs:
        excluded.setdefault(i, set()).add(j)
    S, value = _brute_force(Y, 3, lam, excluded=excluded)
    d = solve_per_unit_exact(Y, 3, restrictions=DesignRestrictions(donor_exclusion=pairs))
    assert _treated(d) == S and d.objective_value == pytest.approx(value, rel=1e-9)
    Q = np.asarray(d.treated_weights)
    assert all(Q[i, j] == 0.0 for i, j in pairs)


def test_a_unit_whose_pool_is_empty_is_never_treated():
    Y = _panel(N=4, seed=17)
    pairs = [(0, 1), (0, 2), (0, 3)]
    d = solve_per_unit_exact(Y, 2, restrictions=DesignRestrictions(donor_exclusion=pairs))
    assert 0 not in _treated(d)


@pytest.mark.parametrize("rules", [
    dict(forced_in=[1]),
    dict(forbidden=[0, 2]),
    dict(conflict_pairs=[(0, 1), (1, 3), (2, 5)]),
    dict(strata=[((0, 1, 2, 3), 1, 1)]),
])
def test_treated_side_restrictions_hold_at_the_optimum(rules):
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter
    Y = _panel(seed=18)
    lam = estimate_lambda(Y)
    restrictions = DesignRestrictions(**rules)
    filt = SubsetFilter.build(restrictions, N=8)
    S, value = _brute_force(Y, 3, lam, accepts=filt.accepts)
    d = solve_per_unit_exact(Y, 3, restrictions=restrictions)
    assert _treated(d) == S and d.objective_value == pytest.approx(value, rel=1e-9)


def test_a_cost_budget_holds_at_the_optimum():
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter
    Y = _panel(seed=19)
    lam = estimate_lambda(Y)
    costs = np.arange(1.0, 9.0)
    filt = SubsetFilter.build(None, costs, 9.0, N=8)
    S, value = _brute_force(Y, 3, lam, accepts=filt.accepts)
    d = solve_per_unit_exact(Y, 3, costs=costs, budget=9.0)
    assert _treated(d) == S and costs[list(S)].sum() <= 9.0


def _two_best(Y):
    lam = estimate_lambda(Y)
    N = Y.shape[1]
    roots = [_phi(Y, i, [j for j in range(N) if j != i], lam) for i in range(N)]
    a, b = (int(u) for u in np.argsort(roots)[:2])
    return lam, a, b


def test_a_stratum_quota_prunes_the_branch_that_breaks_it():
    """The two best-fitted units share a stratum capped at one, so the search
    meets the cap while branching, not only at a leaf."""
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter
    Y = _panel(seed=26)
    lam, a, b = _two_best(Y)
    restrictions = DesignRestrictions(strata=[((a, b), None, 1)])
    S, value = _brute_force(Y, 3, lam,
                            accepts=SubsetFilter.build(restrictions, N=8).accepts)
    d = solve_per_unit_exact(Y, 3, restrictions=restrictions)
    assert _treated(d) == S and d.objective_value == pytest.approx(value, rel=1e-9)


def test_a_budget_prunes_the_branch_that_breaks_it():
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter
    Y = _panel(seed=27)
    lam, a, b = _two_best(Y)
    costs = np.ones(8)
    costs[[a, b]] = 5.0
    S, value = _brute_force(Y, 3, lam,
                            accepts=SubsetFilter.build(None, costs, 7.0, N=8).accepts)
    d = solve_per_unit_exact(Y, 3, costs=costs, budget=7.0)
    assert _treated(d) == S and costs[list(S)].sum() <= 7.0


@pytest.mark.parametrize("rules", [
    dict(forced_in=[0, 1, 2, 3]),                  # more forced than K
    dict(forbidden=[0, 1, 2, 3, 4, 5]),            # fewer treatable than K
])
def test_forced_and_forbidden_units_that_leave_no_design_are_reported(rules):
    with pytest.raises(MlsynthEstimationError, match="no feasible design"):
        solve_per_unit_exact(_panel(), 3, restrictions=DesignRestrictions(**rules))


def test_forcing_a_unit_with_an_emptied_pool_is_reported():
    """Unit 0 may use only donor 1, and both are forced in."""
    pairs = [(0, j) for j in range(2, 8)]
    with pytest.raises(MlsynthEstimationError, match="no feasible design"):
        solve_per_unit_exact(_panel(), 2, restrictions=DesignRestrictions(
            forced_in=[0, 1], donor_exclusion=pairs))


# --------------------------------------------------------------------------- #
# the pool
# --------------------------------------------------------------------------- #


def test_the_pool_is_the_best_designs_in_order():
    Y = _panel(N=6, seed=20)
    lam = estimate_lambda(Y)
    N = 6
    values = sorted(
        (sum(_phi(Y, i, [j for j in range(N) if j not in S], lam) for i in S) / 2, S)
        for S in combinations(range(N), 2))
    pool = solve_per_unit_exact_pool(Y, 2, top_K=4)
    assert [_treated(d) for d in pool] == [S for _, S in values[:4]]


def test_the_pool_reports_a_failure_that_is_not_exhaustion(monkeypatch):
    """Only an empty admissible region ends the pool. The error is caught and its
    message checked outside the ``except``, so ``conftest.py``'s optional-solver
    guard cannot reclassify it (#813)."""
    import mlsynth.utils.syndes_helpers.per_unit_exact as pue

    def stalled(B, A, **kw):
        w = np.full(B.shape[1], 1.0 / B.shape[1])
        return (w, {"converged": False, "pivots": 0}) if kw.get("return_info") else w
    monkeypatch.setattr(pue, "solve_simplex_qp", stalled)
    caught = None
    try:
        solve_per_unit_exact_pool(_panel(), 2, top_K=3)
    except MlsynthEstimationError as exc:
        caught = exc
    assert caught is not None and "converge" in str(caught)


@pytest.mark.parametrize("top_K", [0, -1, True, 2.0])
def test_the_pool_refuses_a_malformed_size(top_K):
    with pytest.raises(MlsynthConfigError, match="top_K"):
        solve_per_unit_exact_pool(_panel(), 2, top_K=top_K)


def test_the_pool_stops_when_the_region_is_exhausted():
    assert len(solve_per_unit_exact_pool(_panel(N=3), 1, top_K=5)) == 3


# --------------------------------------------------------------------------- #
# a search stopped at its node limit says so
# --------------------------------------------------------------------------- #


def test_a_search_stopped_at_its_node_limit_warns_and_reports_a_valid_bound():
    """Three nodes leave this panel's incumbent above the optimum, so the
    reported bound has to come from the open nodes, not from the incumbent."""
    Y = _panel(N=12, seed=22)
    lam = estimate_lambda(Y)
    _, optimum = _brute_force(Y, 5, lam)
    with pytest.warns(SYNDESSolveWarning, match="node limit"):
        d = solve_per_unit_exact(Y, 5, node_limit=3)
    out = d.solve_outcome
    assert out.status == "node_limit" and out.stopped_at_limit
    assert d.raw_results["certified"] is False
    assert d.objective_value > optimum * (1 + 1e-6)
    assert out.dual_bound <= optimum * (1 + 1e-12)
    assert out.gap == pytest.approx(
        (d.objective_value - out.dual_bound) / d.objective_value, rel=1e-12)


def test_a_node_limit_reached_before_any_design_raises():
    """The seed, the two best root fits, is ruled out by a conflict, and one
    node finds no leaf."""
    Y = _panel(seed=25)
    lam = estimate_lambda(Y)
    roots = [_phi(Y, i, [j for j in range(8) if j != i], lam) for i in range(8)]
    a, b = (int(u) for u in np.argsort(roots)[:2])
    with pytest.raises(MlsynthEstimationError, match="node limit"):
        solve_per_unit_exact(Y, 2, node_limit=1,
                             restrictions=DesignRestrictions(conflict_pairs=[(a, b)]))


def test_a_closed_search_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", SYNDESSolveWarning)
        solve_per_unit_exact(_panel(), 3)


# --------------------------------------------------------------------------- #
# failures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("K", [0, 8, -1])
def test_k_out_of_range_is_refused(K):
    with pytest.raises(MlsynthConfigError, match="K"):
        solve_per_unit_exact(_panel(), K)


def test_a_negative_penalty_is_refused():
    with pytest.raises(MlsynthConfigError, match="lam"):
        solve_per_unit_exact(_panel(), 2, lam=-1.0)


def test_a_one_dimensional_panel_is_refused():
    with pytest.raises(MlsynthDataError):
        solve_per_unit_exact(np.zeros(5), 1)


def test_a_restriction_set_with_no_design_is_reported():
    with pytest.raises(MlsynthEstimationError, match="no feasible design"):
        solve_per_unit_exact(_panel(), 3, restrictions=DesignRestrictions(
            forced_in=[0], conflict_pairs=[(0, j) for j in range(1, 8)]))


def test_costs_without_a_budget_are_refused():
    with pytest.raises(MlsynthConfigError):
        solve_per_unit_exact(_panel(), 2, costs=np.ones(8))


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #


def test_the_estimator_solves_per_unit_exactly_by_default():
    Y = _panel(N=8, T=16, seed=22)
    res = SYNDES({"df": _df(Y), "outcome": "y", "unitid": "unit", "time": "time",
                  "post_col": "post", "K": 3, "mode": "per_unit"}).fit()
    d = res.design
    assert d.solve_outcome.solver_status == "exact"
    lam = estimate_lambda(res.inputs.Y_pre)
    S, value = _brute_force(res.inputs.Y_pre, 3, lam)
    assert _treated(d) == S and d.objective_value == pytest.approx(value, rel=1e-9)


def test_the_estimator_keeps_the_mip_on_request():
    pytest.importorskip("pyscipopt")
    Y = _panel(N=6, T=14, seed=23)
    res = SYNDES({"df": _df(Y), "outcome": "y", "unitid": "unit", "time": "time",
                  "post_col": "post", "K": 2, "mode": "per_unit",
                  "backend": "mip"}).fit()
    assert res.design.solve_outcome.solver_status != "exact"


def test_one_way_still_refuses_the_exact_backend():
    Y = _panel(N=6, T=14, seed=24)
    with pytest.raises(MlsynthConfigError, match="exact"):
        SYNDES({"df": _df(Y), "outcome": "y", "unitid": "unit", "time": "time",
                "post_col": "post", "K": 2, "mode": "one_way_global",
                "backend": "exact"})
