"""TDD for the SYNDES solution pool (no-good-cut enumeration of near-optimal designs).

SYNDES's MIP returns a single MSE-optimal design, but managers want options. The
pool re-solves with a "no-good cut" each round -- forbidding the previously
chosen treated set (``sum_{i in S} D_i <= |S|-1``) -- so the next solve returns
the next-best *distinct* design. The result is the top-K designs ranked by MSE
(non-decreasing), the rank-1 being exactly the single-solve optimum.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.syndes_helpers.optimization import (
    solve_synthetic_design,
    solve_synthetic_design_pool,
)


def _Y(T=20, N=10, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, 3)); L = rng.standard_normal((3, N))
    return 100.0 + F @ L + rng.standard_normal((T, N)) * 0.5


class TestSolutionPool:
    def test_top1_matches_single_solve(self):
        Y = _Y()
        single = solve_synthetic_design(Y, K=3, mode="global_2way")
        pool = solve_synthetic_design_pool(Y, K=3, top_K=1, mode="global_2way")
        assert len(pool) == 1
        assert set(pool[0].selected_unit_indices) == set(single.selected_unit_indices)
        assert pool[0].objective_value == pytest.approx(single.objective_value, rel=1e-6)

    def test_pool_distinct_and_ranked(self):
        Y = _Y(seed=1)
        pool = solve_synthetic_design_pool(Y, K=3, top_K=4, mode="global_2way")
        assert len(pool) == 4
        sets = [frozenset(d.selected_unit_indices) for d in pool]
        assert len(set(sets)) == 4                                   # all distinct
        objs = [d.objective_value for d in pool]
        assert all(objs[i] <= objs[i + 1] + 1e-6 for i in range(len(objs) - 1))
        single = solve_synthetic_design(Y, K=3, mode="global_2way")
        assert sets[0] == frozenset(single.selected_unit_indices)   # rank-1 == optimum

    def test_pool_respects_budget(self):
        Y = _Y(seed=2)
        costs = np.arange(1.0, 11.0); budget = 12.0
        pool = solve_synthetic_design_pool(Y, K=3, top_K=3, mode="global_2way",
                                           costs=costs, budget=budget)
        assert len(pool) >= 1
        for d in pool:
            assert costs[d.selected_unit_indices].sum() <= budget + 1e-6

    def test_pool_early_stop_when_exhausted(self):
        Y = _Y(N=5, seed=3)                       # C(5,3) = 10 distinct treated sets
        pool = solve_synthetic_design_pool(Y, K=3, top_K=50, mode="global_2way")
        assert 1 <= len(pool) <= 10
        sets = [frozenset(d.selected_unit_indices) for d in pool]
        assert len(set(sets)) == len(pool)        # no duplicates ever returned

    def test_top_K_validated(self):
        with pytest.raises(Exception):
            solve_synthetic_design_pool(_Y(), K=3, top_K=0, mode="global_2way")


# ----------------------------------------------------------------------
# Estimator-level: SYNDES(top_K=...).fit() -> results.pool menu
# ----------------------------------------------------------------------

import pandas as pd  # noqa: E402

from mlsynth import SYNDES  # noqa: E402


def _panel(T=24, N=10, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, 3)); L = rng.standard_normal((3, N))
    Y = 100.0 + F @ L + rng.standard_normal((T, N)) * 0.5
    rows = []
    for i in range(N):
        for t in range(T):
            rows.append({"unit": f"u{i:02d}", "time": t, "y": float(Y[t, i]),
                         "post": int(t >= T - 4)})
    return pd.DataFrame(rows)


class TestEstimatorPool:
    def _cfg(self, **over):
        base = dict(df=_panel(), outcome="y", unitid="unit", time="time", K=3,
                    mode="two_way_global", post_col="post", run_inference=False)
        base.update(over)
        return base

    def test_top_K_default_no_pool(self):
        res = SYNDES(self._cfg()).fit()
        assert res.pool is None
        assert res.design.selected_unit_indices.size == 3

    def test_pool_menu_returned(self):
        res = SYNDES(self._cfg(top_K=4)).fit()
        assert res.pool is not None and len(res.pool) == 4
        # ranked by MSE (the objective), non-decreasing
        objs = [e["objective"] for e in res.pool]
        assert all(objs[i] <= objs[i + 1] + 1e-6 for i in range(len(objs) - 1))
        # rank-1 menu entry IS the returned optimal design
        assert sorted(res.pool[0]["markets"]) == sorted(res.design.selected_unit_labels.tolist())
        # distinct treated sets, each with the menu columns
        mkts = [tuple(sorted(e["markets"])) for e in res.pool]
        assert len(set(mkts)) == 4
        for e in res.pool:
            assert set(e) >= {"markets", "objective", "pre_fit_rmse", "mde_pct", "cost"}
            assert len(e["markets"]) == 3 and np.isfinite(e["mde_pct"])

    def test_pool_cost_column_with_costs(self):
        N = 10
        costs = list(np.arange(1.0, N + 1.0))
        res = SYNDES(self._cfg(top_K=3, costs=costs, budget=20.0)).fit()
        for e in res.pool:
            assert e["cost"] is not None and e["cost"] <= 20.0 + 1e-6
