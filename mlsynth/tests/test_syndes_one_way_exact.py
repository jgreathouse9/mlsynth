"""SYNDES one-way design by enumeration under a ridge bound (#807).

Doudchenko et al. (2021, Section 2 and Appendix A.2): one-way is two-way with the
treated weights fixed at ``1/K``. Naming the treated set ``S`` leaves one simplex
fit,

    f(S) = lam/K + min over c on the simplex over P of
           (1/T)||ybar_S - Y_P c||^2 + lam ||c||^2,

with ``P`` the controls a donor exclusion leaves. Dropping only the controls'
sign conditions gives the ridge bound ``d'(C'HC)^{-1}d``, ``C = [E_S, 1]``,
``d = [1/K; 0]``, ``H = (G + lam I)^{-1}``. The study behind the choice is
``benchmarks/studies/syndes_one_way_exact/``.

The anchor is brute force: every treated set priced from a fresh fit on the
panel as given.
"""

import warnings
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES
from mlsynth.exceptions import (MlsynthConfigError, MlsynthDataError,
                                MlsynthEstimationError)
from mlsynth.utils.solvers.active_set import solve_simplex_qp
from mlsynth.utils.syndes_helpers.certificate import design_objective
from mlsynth.utils.syndes_helpers.gram import (BoundEngine, TwoWayProblem,
                                               signs_from_subsets)
from mlsynth.utils.syndes_helpers.one_way_exact import (
    OneWayFits,
    ridge_bound,
    solve_one_way_exact,
    solve_one_way_exact_pool,
)
from mlsynth.utils.syndes_helpers.optimization import (SYNDESSolveWarning,
                                                        estimate_lambda)
from mlsynth.utils.syndes_helpers.restrictions import DesignRestrictions


def _panel(N=8, T=12, seed=0, rank=3, noise=0.3):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, rank)) @ rng.standard_normal((rank, N))
            + noise * rng.standard_normal((T, N)))


def _pool(N, S, excluded=None):
    """The controls left to ``S``: its complement, less every donor excluded
    for any of its units."""
    banned = set(S)
    for i in S:
        banned |= set((excluded or {}).get(i, ()))
    return [j for j in range(N) if j not in banned]


def _f(Y, S, lam, excluded=None):
    """``f(S)`` from a fresh fit on ``Y`` as given; infinite on an empty pool."""
    T, N = Y.shape
    K = len(S)
    pool = _pool(N, S, excluded)
    if not pool:
        return np.inf
    B = np.vstack([Y[:, pool] / np.sqrt(T), np.sqrt(lam) * np.eye(len(pool))])
    A = np.concatenate([Y[:, list(S)].mean(axis=1) / np.sqrt(T), np.zeros(len(pool))])
    c = solve_simplex_qp(B, A)
    r = A - B @ c
    return float(r @ r) + lam / K


def _brute_force(Y, K, lam, accepts=lambda S: True, excluded=None):
    """The optimum over every admissible treated set, from fresh fits."""
    best, best_set = np.inf, None
    for S in combinations(range(Y.shape[1]), K):
        if not accepts(S):
            continue
        value = _f(Y, S, lam, excluded)
        if value < best:
            best, best_set = value, S
    return best_set, best


def _engine(Y, lam):
    return BoundEngine.from_problem(TwoWayProblem.from_panel(Y, lam=lam))


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


def test_a_panel_gets_a_certified_one_way_design():
    Y = _panel()
    d = solve_one_way_exact(Y, 3)
    assert d.mode == "global_equal_weights" and len(_treated(d)) == 3
    assert np.isfinite(d.objective_value)
    out = d.solve_outcome
    assert out.status == "optimal" and out.solver_status == "exact"
    assert not out.stopped_at_limit and out.gap == 0.0
    assert out.dual_bound == d.objective_value
    assert d.raw_results["certified"] is True
    assert d.raw_results["n_designs"] == comb(8, 3)


# --------------------------------------------------------------------------- #
# the design is the optimum, and its weights attain it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("K", [1, 2, 3, 5, 7])
def test_the_design_is_the_brute_force_optimum(seed, K):
    Y = _panel(seed=seed)
    lam = estimate_lambda(Y)
    S, value = _brute_force(Y, K, lam)
    d = solve_one_way_exact(Y, K)
    assert _treated(d) == S
    assert d.objective_value == pytest.approx(value, rel=1e-9)


def test_the_objective_is_what_the_weights_attain():
    Y = _panel(seed=5)
    d = solve_one_way_exact(Y, 3)
    assert design_objective(Y, "global_equal_weights", d, d.lambda_value) == (
        pytest.approx(d.objective_value, rel=1e-10))


def test_the_treated_weights_are_equal_and_the_controls_on_the_simplex():
    Y = _panel(seed=6)
    d = solve_one_way_exact(Y, 3)
    S = list(_treated(d))
    a = np.asarray(d.treated_weights)
    b = np.asarray(d.control_weights)
    np.testing.assert_array_equal(a[S], 1.0 / 3)
    assert np.all(np.delete(a, S) == 0.0)
    assert (b >= 0).all() and np.all(b[S] == 0.0)
    assert b.sum() == pytest.approx(1.0, abs=1e-12)


def test_the_contrast_is_treated_less_control():
    Y = _panel(seed=7)
    d = solve_one_way_exact(Y, 3)
    np.testing.assert_allclose(d.contrast_weights,
                               np.asarray(d.treated_weights) - d.control_weights,
                               atol=1e-15)
    np.testing.assert_allclose(d.contrast_series, Y @ d.contrast_weights, atol=1e-12)
    assert d.pre_fit_rmse == pytest.approx(
        float(np.sqrt(np.mean(d.contrast_series ** 2))), rel=1e-12)
    assert d.w is None and d.q is None and d.z is None


def test_the_labels_follow_the_unit_index():
    from mlsynth.utils.fast_scm_helpers.structure import IndexSet
    Y = _panel(N=5, seed=32)
    labels = ["a", "b", "c", "d", "e"]
    d = solve_one_way_exact(Y, 2, unit_index=IndexSet.from_labels(labels))
    assert list(d.selected_unit_labels) == [labels[i] for i in _treated(d)]
    assert d.assignment_by_unit == {
        lab: int(i in _treated(d)) for i, lab in enumerate(labels)}


def test_the_default_penalty_is_the_mip_paths():
    Y = _panel(seed=8)
    assert solve_one_way_exact(Y, 2).lambda_value == pytest.approx(
        estimate_lambda(Y), rel=1e-15)


def test_the_reduction_is_the_mip_with_the_assignment_fixed():
    """``f(S)`` is the one-way program with ``D`` fixed, solved by cvxpy."""
    from mlsynth.utils.syndes_helpers.optimization import _refit
    Y = _panel(seed=9)
    d = solve_one_way_exact(Y, 3)
    _, value = _refit(Y, d.assignment, "global_equal_weights", d.lambda_value, None)
    assert d.objective_value == pytest.approx(value, rel=1e-6)


def test_the_design_matches_scips_proven_optimum():
    pytest.importorskip("pyscipopt")
    from mlsynth.utils.syndes_helpers.optimization import solve_synthetic_design
    Y = _panel(N=8, seed=10)
    Y = Y / Y.std()                         # SCIP's tolerances are absolute (#814)
    mip = solve_synthetic_design(Y, 3, mode="global_equal_weights", gap_limit=0.0,
                                 time_limit=None)
    d = solve_one_way_exact(Y, 3)
    assert _treated(d) == tuple(int(i) for i in mip.selected_unit_indices)
    assert d.objective_value <= mip.objective_value * (1 + 1e-7)


@pytest.mark.parametrize("s", [1e-6, 1e6])
def test_a_panel_in_other_units_gets_the_same_design(s):
    Y = _panel(seed=11)
    ref = solve_one_way_exact(Y, 3)
    got = solve_one_way_exact(Y * s, 3)
    assert _treated(got) == _treated(ref)
    assert got.objective_value / s ** 2 == pytest.approx(ref.objective_value, rel=1e-9)


@pytest.mark.parametrize("level", [1e6, 1e8])
def test_a_common_level_changes_nothing(level):
    """Uncentred, the fits stopped short of convergence on 222 of 3,328 sets at a
    level of 1e8. The tolerance is the stored data's rounding: entries of size
    1e8 carry an error of about 1e-8 of the panel's unit-scale variation."""
    Y = _panel(seed=12)
    ref = solve_one_way_exact(Y, 3)
    got = solve_one_way_exact(Y + level, 3, lam=ref.lambda_value)
    assert _treated(got) == _treated(ref)
    assert got.objective_value == pytest.approx(ref.objective_value, rel=1e-6)


# --------------------------------------------------------------------------- #
# the bound
# --------------------------------------------------------------------------- #


def test_the_ridge_bound_is_below_every_set_and_above_the_two_way_bound():
    Y = _panel(N=7, seed=13)
    lam = estimate_lambda(Y)
    engine = _engine(Y, lam)
    sets = np.array(list(combinations(range(7), 3)))
    ridge = ridge_bound(engine, sets)
    two_way = engine.bound(signs_from_subsets(7, sets)).lower
    values = np.array([_f(Y, S, lam) for S in sets])
    assert np.all(ridge <= values * (1 + 1e-10))
    assert np.all(ridge >= two_way * (1 - 1e-10))


def test_with_one_control_the_ridge_bound_is_the_value():
    """``u_S = 1/K`` and ``1'u = 0`` leave the lone control at ``-1``: the
    relaxation's only point is feasible, so it drops nothing."""
    Y = _panel(N=6, seed=14)
    lam = estimate_lambda(Y)
    sets = np.array(list(combinations(range(6), 5)))
    ridge = ridge_bound(_engine(Y, lam), sets)
    values = np.array([_f(Y, S, lam) for S in sets])
    np.testing.assert_allclose(ridge, values, rtol=1e-10)


def test_the_ridge_bound_of_one_set_is_its_row_in_a_batch():
    Y = _panel(N=7, seed=15)
    engine = _engine(Y, estimate_lambda(Y))
    sets = np.array(list(combinations(range(7), 2)))
    batch = ridge_bound(engine, sets)
    assert ridge_bound(engine, sets[4]) == pytest.approx(batch[4], rel=1e-14)


def test_the_search_prices_only_sets_whose_bound_is_below_the_optimum():
    Y = _panel(N=9, seed=16)
    lam = estimate_lambda(Y)
    d = solve_one_way_exact(Y, 3)
    sets = np.array(list(combinations(range(9), 3)))
    below = int(np.sum(ridge_bound(_engine(Y, lam), sets)
                       < d.objective_value * (1 + 1e-9)))
    assert 1 <= d.raw_results["n_priced"] <= 1 + below < len(sets)


def test_a_set_the_bound_favours_but_its_exclusions_spoil_is_priced_past():
    """Unit 0 is ``2 y_1 - y_2``, so units 0, 1 and 2 each sit on the line
    through the other two and their ridge bounds are near zero. Unit 1 is the
    midpoint of 0 and 2, but it may not use unit 0, and the bound does not see
    exclusions: the best-bounded set is unit 1, and its value is not the optimum.
    The search has to price on past it."""
    rng = np.random.default_rng(0)
    b, c, e, f = rng.standard_normal((4, 10))
    Y = np.column_stack([2 * b - c, b, c, e + 0.3 * rng.standard_normal(10), e, f])
    lam = 1e-3 * float(np.mean(np.var(Y, axis=0, ddof=1)))
    pairs = [(1, 0)]
    sets = np.array([[i] for i in range(6)])
    assert int(np.argmin(ridge_bound(_engine(Y, lam), sets))) == 1
    S, value = _brute_force(Y, 1, lam, excluded=_excluded(pairs))
    assert S != (1,) and _f(Y, (1,), lam, _excluded(pairs)) > 2 * value
    d = solve_one_way_exact(Y, 1, lam=lam,
                            restrictions=DesignRestrictions(donor_exclusion=pairs))
    assert _treated(d) == S and d.objective_value == pytest.approx(value, rel=1e-9)
    assert d.raw_results["certified"] is True and d.raw_results["n_priced"] > 1


def test_streaming_in_small_blocks_keeps_few_sets(monkeypatch):
    """Each block's best-bounded set is priced as it streams, so later blocks
    keep only the sets whose bound is below that incumbent."""
    import mlsynth.utils.syndes_helpers.one_way_exact as owe
    Y = _panel(N=9, seed=17)
    ref = solve_one_way_exact(Y, 3)
    monkeypatch.setattr(owe, "CHUNK", 5)
    got = solve_one_way_exact(Y, 3)
    assert _treated(got) == _treated(ref)
    assert got.objective_value == ref.objective_value
    assert got.raw_results["n_survivors"] < got.raw_results["n_designs"] // 4


# --------------------------------------------------------------------------- #
# the fits
# --------------------------------------------------------------------------- #


def test_a_priced_set_equals_a_fresh_fit():
    Y = _panel(N=7, seed=18)
    lam = estimate_lambda(Y)
    fits = OneWayFits(Y, lam)
    for S in combinations(range(7), 3):
        value, c = fits.price(S)
        assert value == pytest.approx(_f(Y, S, lam), rel=1e-9)
        assert c.shape == (7,) and np.all(c[list(S)] == 0.0)


def test_a_set_whose_pool_is_empty_is_priced_infinite():
    Y = _panel(N=4, seed=19)
    fits = OneWayFits(Y, estimate_lambda(Y), {0: frozenset({2, 3})})
    value, c = fits.price((0, 1))
    assert value == np.inf and c is None


def test_a_fit_that_does_not_converge_is_reported(monkeypatch):
    import mlsynth.utils.syndes_helpers.one_way_exact as owe

    def stalled(B, A, **kw):
        w = np.full(B.shape[1], 1.0 / B.shape[1])
        return (w, {"converged": False, "pivots": 0}) if kw.get("return_info") else w
    monkeypatch.setattr(owe, "solve_simplex_qp", stalled)
    with pytest.raises(MlsynthEstimationError, match="converge"):
        solve_one_way_exact(_panel(), 2)


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #


def test_all_but_one_treated_puts_every_weight_on_the_last_control():
    Y = _panel(N=5, seed=20)
    lam = estimate_lambda(Y)
    S, value = _brute_force(Y, 4, lam)
    d = solve_one_way_exact(Y, 4)
    (c,) = [j for j in range(5) if j not in _treated(d)]
    assert _treated(d) == S and d.objective_value == pytest.approx(value, rel=1e-9)
    assert d.control_weights[c] == pytest.approx(1.0, abs=1e-12)


def test_two_units():
    Y = _panel(N=2, T=6, seed=21)
    lam = estimate_lambda(Y)
    S, value = _brute_force(Y, 1, lam)
    d = solve_one_way_exact(Y, 1)
    assert _treated(d) == S and d.objective_value == pytest.approx(value, rel=1e-9)


def test_duplicate_units():
    Y = _panel(N=6, seed=22)
    Y[:, 5] = Y[:, 0]
    lam = estimate_lambda(Y)
    _, value = _brute_force(Y, 2, lam)
    assert solve_one_way_exact(Y, 2).objective_value == pytest.approx(value, rel=1e-9)


def test_no_penalty():
    Y = _panel(N=6, T=20, seed=23)
    S, value = _brute_force(Y, 2, 0.0)
    d = solve_one_way_exact(Y, 2, lam=0.0)
    assert d.objective_value == pytest.approx(value, rel=1e-8, abs=1e-12)


def _collinear(seed=37):
    """Unit 2 is the mean of units 0 and 1: with ``lam = 0`` the problem on
    contrasts is singular, and treating 0 and 1 against 2 fits exactly."""
    Y = _panel(N=8, T=14, seed=seed)
    Y[:, 2] = 0.5 * (Y[:, 0] + Y[:, 1])
    return Y


@pytest.mark.parametrize("Y", [_collinear(), _panel(N=8, T=3, seed=24)],
                         ids=["collinear", "short"])
def test_a_singular_problem_is_priced_in_full(Y):
    """No bound can be formed, so every admissible set is priced, and the
    design is still the optimum."""
    S, value = _brute_force(Y, 2, 0.0)
    d = solve_one_way_exact(Y, 2, lam=0.0)
    assert d.objective_value == pytest.approx(value, rel=1e-8, abs=1e-12)
    assert d.raw_results["certified"] is True
    assert d.raw_results["n_priced"] == d.raw_results["n_designs"] == comb(8, 2)


def test_the_estimators_one_way_default_handles_a_singular_problem():
    Y = _collinear()
    res = _fit(Y, K=2, lam=0.0)
    assert res.design.solve_outcome.solver_status == "exact"
    assert res.design.objective_value == pytest.approx(
        _brute_force(res.inputs.Y_pre, 2, 0.0)[1], abs=1e-12)


def test_a_singular_problem_past_the_pricing_limit_is_refused(monkeypatch):
    import mlsynth.utils.syndes_helpers.one_way_exact as owe
    monkeypatch.setattr(owe, "PRICE_ALL_LIMIT", 27)
    with pytest.raises(MlsynthEstimationError, match="singular to working precision"):
        solve_one_way_exact(_collinear(), 2, lam=0.0)


def test_a_singular_problem_past_the_candidate_limit_is_refused():
    with pytest.raises(MlsynthEstimationError, match="singular to working precision"):
        solve_one_way_exact(_collinear(), 2, lam=0.0, candidate_limit=5)


# --------------------------------------------------------------------------- #
# restrictions
# --------------------------------------------------------------------------- #


def _excluded(pairs):
    out = {}
    for i, j in pairs:
        out.setdefault(i, set()).add(j)
    return out


def test_donor_exclusions_shrink_the_control_pool():
    Y = _panel(seed=25)
    lam = estimate_lambda(Y)
    pairs = [(i, j) for i in range(8) for j in range(8) if i != j and (i + j) % 3 == 0]
    S, value = _brute_force(Y, 3, lam, excluded=_excluded(pairs))
    d = solve_one_way_exact(Y, 3, restrictions=DesignRestrictions(donor_exclusion=pairs))
    assert _treated(d) == S and d.objective_value == pytest.approx(value, rel=1e-9)
    assert all(d.control_weights[j] == 0.0 for i, j in pairs if i in S)


def test_a_set_whose_pool_is_emptied_is_never_chosen():
    Y = _panel(N=4, seed=26)
    pairs = [(0, 2), (0, 3)]
    d = solve_one_way_exact(Y, 2, restrictions=DesignRestrictions(donor_exclusion=pairs))
    assert _treated(d) != (0, 1)


@pytest.mark.parametrize("rules", [
    dict(forced_in=[1]),
    dict(forbidden=[0, 2]),
    dict(conflict_pairs=[(0, 1), (1, 3), (2, 5)]),
    dict(strata=[((0, 1, 2, 3), 1, 1)]),
])
def test_treated_side_restrictions_hold_at_the_optimum(rules):
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter
    Y = _panel(seed=27)
    lam = estimate_lambda(Y)
    restrictions = DesignRestrictions(**rules)
    filt = SubsetFilter.build(restrictions, N=8)
    S, value = _brute_force(Y, 3, lam, accepts=filt.accepts)
    d = solve_one_way_exact(Y, 3, restrictions=restrictions)
    assert _treated(d) == S and d.objective_value == pytest.approx(value, rel=1e-9)


def test_a_cost_budget_holds_at_the_optimum():
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter
    Y = _panel(seed=28)
    lam = estimate_lambda(Y)
    costs = np.arange(1.0, 9.0)
    S, value = _brute_force(Y, 3, lam,
                            accepts=SubsetFilter.build(None, costs, 9.0, N=8).accepts)
    d = solve_one_way_exact(Y, 3, costs=costs, budget=9.0)
    assert _treated(d) == S and costs[list(S)].sum() <= 9.0


@pytest.mark.parametrize("rules", [
    dict(forced_in=[0, 1, 2, 3]),                  # more forced than K
    dict(forbidden=[0, 1, 2, 3, 4, 5]),            # fewer treatable than K
    dict(forced_in=[0], conflict_pairs=[(0, j) for j in range(1, 8)]),
])
def test_restrictions_that_leave_no_design_are_reported(rules):
    with pytest.raises(MlsynthEstimationError, match="no feasible design"):
        solve_one_way_exact(_panel(), 3, restrictions=DesignRestrictions(**rules))


def test_donor_exclusions_that_empty_every_pool_are_reported():
    pairs = [(i, j) for i in range(4) for j in range(4) if i != j]
    with pytest.raises(MlsynthEstimationError, match="no feasible design"):
        solve_one_way_exact(_panel(N=4), 2,
                            restrictions=DesignRestrictions(donor_exclusion=pairs))


# --------------------------------------------------------------------------- #
# the pool
# --------------------------------------------------------------------------- #


def test_the_pool_is_the_best_designs_in_order():
    Y = _panel(N=6, seed=29)
    lam = estimate_lambda(Y)
    values = sorted((_f(Y, S, lam), S) for S in combinations(range(6), 2))
    pool = solve_one_way_exact_pool(Y, 2, top_K=4)
    assert [_treated(d) for d in pool] == [S for _, S in values[:4]]


def test_the_pool_reports_a_failure_that_is_not_exhaustion(monkeypatch):
    """Only an empty admissible region ends the pool. The error is caught and its
    message checked outside the ``except``, so ``conftest.py``'s optional-solver
    guard cannot reclassify it (#813)."""
    import mlsynth.utils.syndes_helpers.one_way_exact as owe

    def stalled(B, A, **kw):
        w = np.full(B.shape[1], 1.0 / B.shape[1])
        return (w, {"converged": False, "pivots": 0}) if kw.get("return_info") else w
    monkeypatch.setattr(owe, "solve_simplex_qp", stalled)
    caught = None
    try:
        solve_one_way_exact_pool(_panel(), 2, top_K=3)
    except MlsynthEstimationError as exc:
        caught = exc
    assert caught is not None and "converge" in str(caught)


@pytest.mark.parametrize("top_K", [0, -1, True, 2.0])
def test_the_pool_refuses_a_malformed_size(top_K):
    with pytest.raises(MlsynthConfigError, match="top_K"):
        solve_one_way_exact_pool(_panel(), 2, top_K=top_K)


def test_the_pool_stops_when_the_region_is_exhausted():
    assert len(solve_one_way_exact_pool(_panel(N=3), 1, top_K=5)) == 3


# --------------------------------------------------------------------------- #
# above the candidate limit
# --------------------------------------------------------------------------- #


def test_above_the_limit_the_swap_search_warns_and_reports_the_rayleigh_bound():
    Y = _panel(N=10, seed=30)
    lam = estimate_lambda(Y)
    _, optimum = _brute_force(Y, 4, lam)
    with pytest.warns(SYNDESSolveWarning, match="candidate limit"):
        d = solve_one_way_exact(Y, 4, candidate_limit=10)
    out = d.solve_outcome
    assert out.status == "candidate_limit" and out.stopped_at_limit
    assert d.raw_results["certified"] is False
    assert d.raw_results["n_designs"] == comb(10, 4)
    rayleigh = _engine(Y, lam).global_lower_bound(4)
    assert out.dual_bound == pytest.approx(rayleigh, rel=1e-12)
    assert out.dual_bound <= optimum <= d.objective_value * (1 + 1e-12)
    assert d.objective_value == pytest.approx(_f(Y, _treated(d), lam), rel=1e-9)
    assert out.gap == pytest.approx(
        (d.objective_value - out.dual_bound) / d.objective_value, rel=1e-12)


def test_above_the_limit_donor_exclusions_still_apply():
    Y = _panel(N=10, seed=31)
    lam = estimate_lambda(Y)
    pairs = [(i, j) for i in range(10) for j in range(10) if i != j and (i + j) % 4 == 0]
    with pytest.warns(SYNDESSolveWarning):
        d = solve_one_way_exact(Y, 4, candidate_limit=10,
                                restrictions=DesignRestrictions(donor_exclusion=pairs))
    S = _treated(d)
    assert d.objective_value == pytest.approx(_f(Y, S, lam, _excluded(pairs)), rel=1e-9)


def test_above_the_limit_treated_side_restrictions_are_refused():
    with pytest.raises(MlsynthConfigError, match="enumeration limit"):
        solve_one_way_exact(_panel(N=10), 4, candidate_limit=10,
                            restrictions=DesignRestrictions(forbidden=[0]))


def test_above_the_limit_no_admissible_candidate_is_reported():
    pairs = [(i, j) for i in range(6) for j in range(6) if i != j]
    with pytest.raises(MlsynthEstimationError, match="swap search"):
        solve_one_way_exact(_panel(N=6), 2, candidate_limit=1,
                            restrictions=DesignRestrictions(donor_exclusion=pairs))


def test_a_search_within_the_limit_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", SYNDESSolveWarning)
        solve_one_way_exact(_panel(), 3)


# --------------------------------------------------------------------------- #
# failures
# --------------------------------------------------------------------------- #


def test_costs_of_the_wrong_length_are_refused():
    with pytest.raises(MlsynthConfigError, match="length"):
        solve_one_way_exact(_panel(), 2, costs=np.ones(5), budget=3.0)


@pytest.mark.parametrize("K", [0, 8, -1, None, 2.0, True])
def test_k_out_of_range_is_refused(K):
    with pytest.raises(MlsynthConfigError, match="K"):
        solve_one_way_exact(_panel(), K)


def test_a_negative_penalty_is_refused():
    with pytest.raises(MlsynthConfigError, match="lam"):
        solve_one_way_exact(_panel(), 2, lam=-1.0)


def test_a_one_dimensional_panel_is_refused():
    with pytest.raises(MlsynthDataError):
        solve_one_way_exact(np.zeros(5), 1)


def test_costs_without_a_budget_are_refused():
    with pytest.raises(MlsynthConfigError):
        solve_one_way_exact(_panel(), 2, costs=np.ones(8))


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #


def _fit(Y, **over):
    cfg = {"df": _df(Y), "outcome": "y", "unitid": "unit", "time": "time",
           "post_col": "post", "K": 3, "mode": "one_way_global"}
    cfg.update(over)
    return SYNDES(cfg).fit()


def test_the_estimator_solves_one_way_exactly_by_default():
    Y = _panel(N=8, T=16, seed=33)
    res = _fit(Y)
    d = res.design
    assert d.mode == "one_way_global"
    assert d.solve_outcome.solver_status == "exact" and d.raw_results["certified"]
    lam = estimate_lambda(res.inputs.Y_pre)
    S, value = _brute_force(res.inputs.Y_pre, 3, lam)
    assert _treated(d) == S and d.objective_value == pytest.approx(value, rel=1e-9)


def test_the_estimator_passes_donor_rules_to_the_search():
    Y = _panel(N=8, T=16, seed=34)
    adjacency = pd.DataFrame(np.zeros((8, 8), dtype=int), index=range(8),
                             columns=range(8))
    for i, j in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        adjacency.loc[i, j] = adjacency.loc[j, i] = 1
    res = _fit(Y, donor_exclusion=adjacency)
    d = res.design
    assert d.solve_outcome.solver_status == "exact"
    pairs = [(i, j) for i in range(8) for j in range(8) if adjacency.loc[i, j]]
    lam = estimate_lambda(res.inputs.Y_pre)
    S, value = _brute_force(res.inputs.Y_pre, 3, lam, excluded=_excluded(pairs))
    assert _treated(d) == S and d.objective_value == pytest.approx(value, rel=1e-9)


def test_the_estimator_keeps_the_mip_on_request():
    pytest.importorskip("pyscipopt")
    res = _fit(_panel(N=6, T=14, seed=35), K=2, backend="mip")
    assert res.design.solve_outcome.solver_status != "exact"


def test_the_estimators_pool_is_the_searchs():
    Y = _panel(N=7, T=16, seed=36)
    res = _fit(Y, K=2, top_K=3, selection="in_sample")
    lam = estimate_lambda(res.inputs.Y_pre)
    ranked = sorted((_f(res.inputs.Y_pre, S, lam), S) for S in combinations(range(7), 2))
    assert tuple(res.design.selected_unit_indices) == ranked[0][1]
