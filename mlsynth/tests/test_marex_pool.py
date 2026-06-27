"""TDD for the MAREX solution-pool + MDE power + recommendation (SYNDES parity).

``top_K=1`` (default) keeps the single-design behaviour. ``top_K>1`` enumerates a
menu of distinct designs via no-good cuts, scores each on a Newey-West MDE power
curve (the same power analysis SYNDES uses), and returns a composite
recommendation. Mirrors tests/test_syndes_pool.py in spirit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mlsynth import MAREX
from mlsynth.utils.marex_helpers.structures import MAREXResults


def _panel(J=8, T=16, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 1, (T, 2))
    lam = rng.normal(0, 1, (J, 2))
    Y = lam @ F.T + 0.2 * rng.standard_normal((J, T))
    return pd.DataFrame([{"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t])}
                         for j in range(J) for t in range(T)])


def _cfg(**over):
    base = dict(df=_panel(), outcome="y", unitid="unit", time="time", T0=12, m_eq=2)
    base.update(over)
    return base


# ---- Stage A: scaffolding (config knobs + default behaviour) ----------------

class TestPoolConfig:
    def test_default_top_K_returns_no_pool(self):
        res = MAREX(_cfg()).fit()
        assert isinstance(res, MAREXResults)
        assert res.pool is None
        assert res.recommendation is None

    def test_bad_top_K_rejected(self):
        with pytest.raises((ValidationError, Exception)):
            MAREX(_cfg(top_K=0))

    def test_power_weights_must_be_positive(self):
        with pytest.raises((ValidationError, Exception)):
            MAREX(_cfg(top_K=2, power_weight=0.0))


# ---- Stage B-E: the pool, MDE, and recommendation ---------------------------

class TestPool:
    def test_pool_returned_with_menu_fields(self):
        res = MAREX(_cfg(top_K=3)).fit()
        assert res.pool is not None
        assert 1 <= len(res.pool) <= 3
        entry = res.pool[0]
        for key in ("markets", "control_group", "objective", "mde_pct", "power_curve"):
            assert key in entry, f"pool entry missing '{key}'"
        assert isinstance(entry["markets"], list) and len(entry["markets"]) == 2

    def test_pool_designs_are_distinct(self):
        # no-good cuts must yield distinct treated sets
        res = MAREX(_cfg(top_K=3)).fit()
        sets = [tuple(sorted(e["markets"])) for e in res.pool]
        assert len(sets) == len(set(sets))

    def test_pool_mde_is_computed(self):
        res = MAREX(_cfg(top_K=3)).fit()
        # every entry has a numeric (possibly inf) MDE percent
        assert all(isinstance(e["mde_pct"], float) for e in res.pool)
        assert any(np.isfinite(e["mde_pct"]) for e in res.pool)

    def test_fit_metric_is_the_held_out_blank_rmse(self):
        # the ranking fit metric (objective) is the out-of-sample blank RMSE,
        # not the in-sample pre-period RMSE
        res = MAREX(_cfg(top_K=3)).fit()
        for e in res.pool:
            assert e["blank_rmse"] is not None
            assert e["objective"] == pytest.approx(e["blank_rmse"])


class TestRecommendation:
    def test_recommendation_present_and_consistent(self):
        res = MAREX(_cfg(top_K=3)).fit()
        rec = res.recommendation
        assert rec is not None
        assert isinstance(rec.winner, dict) and "markets" in rec.winner
        assert len(rec.shortlist) >= 1
        # the winner is one of the pooled designs
        pooled = {tuple(sorted(e["markets"])) for e in res.pool}
        assert tuple(sorted(rec.winner["markets"])) in pooled
        # weights are normalised to sum to one
        assert rec.weights["power"] + rec.weights["fit"] == pytest.approx(1.0, abs=1e-9)
