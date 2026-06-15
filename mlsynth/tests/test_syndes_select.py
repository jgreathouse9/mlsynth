"""TDD for the SYNDES Pareto recommendation (utils/syndes_helpers/select.py).

Mirrors the LEXSCM recommender tests, but SYNDES selects by a GeoLift-style
composite score (weighted mean of fit/power dense ranks) rather than a
lexicographic validity gate, while still exposing the (fit, power) Pareto front.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.syndes_helpers.select import (
    SYNDESDesignMetrics,
    SYNDESRecommendation,
    _pareto_front,
    recommend_syndes,
)


def _entry(obj, mde, cost=None, rmse=None, markets=None):
    """A minimal pool-menu entry (only the fields the recommender reads)."""
    return {
        "markets": markets if markets is not None else [f"m{obj}"],
        "control_group": [],
        "objective": obj,
        "mde_pct": mde,
        "cost": cost,
        "pre_fit_rmse": rmse,
        "design": None,
    }


class TestPareto:
    def test_dominated_design_excluded(self):
        # D_dom is worse on both fit and power than the second entry -> off front.
        pool = [
            _entry(0.10, 2.0, markets=["a"]),   # D1
            _entry(0.20, 1.0, markets=["b"]),   # D2
            _entry(0.30, 0.5, markets=["c"]),   # D3
            _entry(0.25, 2.5, markets=["d"]),   # D4 dominated by D2 (lower both)
        ]
        front = _pareto_front(recommend_syndes(pool).shortlist)  # via metrics
        # build metrics directly for a clean unit test of the helper
        metrics = [
            SYNDESDesignMetrics("D1", ["a"], [], 0.10, 2.0, True),
            SYNDESDesignMetrics("D2", ["b"], [], 0.20, 1.0, True),
            SYNDESDesignMetrics("D3", ["c"], [], 0.30, 0.5, True),
            SYNDESDesignMetrics("D4", ["d"], [], 0.25, 2.5, True),
        ]
        ids = _pareto_front(metrics)
        assert "D4" not in ids
        assert {"D1", "D2", "D3"}.issubset(set(ids))

    def test_infeasible_mde_is_dominated(self):
        metrics = [
            SYNDESDesignMetrics("D1", [], [], 0.10, 1.0, True),
            SYNDESDesignMetrics("D2", [], [], 0.20, np.inf, False),
        ]
        # D2 (no power, worse fit) is dominated by D1.
        assert _pareto_front(metrics) == ["D1"]


class TestRecommend:
    def test_empty_pool(self):
        rec = recommend_syndes([])
        assert isinstance(rec, SYNDESRecommendation)
        assert rec.status == "EMPTY" and rec.winner is None
        assert rec.pareto_ids == [] and rec.table == []

    def test_score_prefers_power_by_default(self):
        # D1 best fit / worse power; D2 worse fit / best power. With power
        # weighted above fit, D2 must win the composite score.
        pool = [
            _entry(1.0, 2.0, markets=["fit"]),    # D1
            _entry(1.1, 1.0, markets=["pow"]),    # D2
        ]
        rec = recommend_syndes(pool)              # default 0.51 / 0.49
        assert rec.status == "OK"
        assert rec.winner.markets == ["pow"]

    def test_weights_can_flip_to_prefer_fit(self):
        pool = [
            _entry(1.0, 2.0, markets=["fit"]),
            _entry(1.1, 1.0, markets=["pow"]),
        ]
        rec = recommend_syndes(pool, power_weight=0.10, fit_weight=0.90)
        assert rec.winner.markets == ["fit"]

    def test_weights_are_normalised(self):
        rec = recommend_syndes([_entry(1.0, 1.0)], power_weight=2.0, fit_weight=2.0)
        assert rec.weights["power"] == pytest.approx(0.5)
        assert rec.weights["fit"] == pytest.approx(0.5)

    def test_cost_breaks_score_ties(self):
        # Identical fit and power -> identical ranks/score; cheaper design wins.
        pool = [
            _entry(1.0, 1.0, cost=10.0, markets=["pricey"]),
            _entry(1.0, 1.0, cost=2.0, markets=["cheap"]),
        ]
        rec = recommend_syndes(pool)
        assert rec.winner.markets == ["cheap"]

    def test_power_not_established_falls_back_to_fit(self):
        pool = [
            _entry(0.10, np.inf, markets=["best_fit"]),
            _entry(0.20, np.nan, markets=["other"]),
        ]
        rec = recommend_syndes(pool)
        assert rec.status == "POWER_NOT_ESTABLISHED"
        assert rec.winner.markets == ["best_fit"]

    def test_table_flags_one_winner_and_pareto(self):
        pool = [
            _entry(0.10, 2.0), _entry(0.20, 1.0),
            _entry(0.30, 0.5), _entry(0.25, 2.5),
        ]
        rec = recommend_syndes(pool)
        assert sum(r["winner"] for r in rec.table) == 1
        assert any(r["pareto"] for r in rec.table)
        assert all({"design_id", "score", "pareto", "winner"} <= set(r)
                   for r in rec.table)

    def test_shortlist_respects_max(self):
        pool = [_entry(float(i), float(10 - i)) for i in range(8)]
        rec = recommend_syndes(pool, max_shortlist=3)
        assert len(rec.shortlist) == 3

    def test_design_ids_numbered_by_fit(self):
        # D1 is the best-fitting design (lowest RMSE), regardless of input order.
        pool = [
            _entry(5.0, 1.0, rmse=0.9, markets=["worst_fit"]),
            _entry(1.0, 3.0, rmse=0.1, markets=["best_fit"]),
            _entry(3.0, 2.0, rmse=0.5, markets=["mid"]),
        ]
        rec = recommend_syndes(pool)
        by_id = {r["design_id"]: r for r in rec.table}
        assert by_id["D1"]["markets"] == ["best_fit"]
        assert by_id["D3"]["markets"] == ["worst_fit"]
        # fit RMSE increases monotonically with the design number
        rmses = [by_id[f"D{i}"]["fit_rmse"] for i in (1, 2, 3)]
        assert rmses == sorted(rmses)


# ----------------------------------------------------------------------
# Estimator integration: SYNDES(top_K=...).fit() -> results.recommendation
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


class TestEstimatorRecommendation:
    def _cfg(self, **over):
        base = dict(df=_panel(), outcome="y", unitid="unit", time="time", K=3,
                    mode="two_way_global", post_col="post", run_inference=False)
        base.update(over)
        return base

    def test_no_recommendation_without_pool(self):
        res = SYNDES(self._cfg()).fit()
        assert res.recommendation is None

    def test_recommendation_attached_with_pool(self):
        res = SYNDES(self._cfg(top_K=4)).fit()
        rec = res.recommendation
        assert isinstance(rec, SYNDESRecommendation)
        assert rec.status in {"OK", "POWER_NOT_ESTABLISHED"}
        assert rec.winner is not None
        # the winner is one of the pooled designs
        pool_markets = {tuple(sorted(e["markets"])) for e in res.pool}
        assert tuple(sorted(rec.winner.markets)) in pool_markets
        # default weighting prefers power over fit
        assert rec.weights["power"] > rec.weights["fit"]
        # exactly one winner flagged in the table
        assert sum(r["winner"] for r in rec.table) == 1

    def test_recommendation_weights_configurable(self):
        res = SYNDES(self._cfg(top_K=4, power_weight=0.9, fit_weight=0.1)).fit()
        assert res.recommendation.weights["power"] == pytest.approx(0.9)
