"""TDD for the cross-estimator design comparison (utils/design_compare.py).

GEOLIFT and SYNDES designs are scored on one shared fit-vs-power plane: fit is
the pre-period RMSE of the contrast, power is a simulated MDE at a common
horizon, both computed by the same harness so the comparison is apples-to-apples.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.design_compare import (
    DesignSpec,
    _pareto_mask,
    compare_pareto,
    from_geolift,
    from_syndes,
    plot_compare_pareto,
    simulated_mde,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# ----------------------------------------------------------------------
# simulated_mde
# ----------------------------------------------------------------------

class TestSimulatedMDE:
    def _grid(self):
        return np.arange(0.0, 5.0, 0.02)

    def test_noisier_contrast_has_larger_mde(self):
        rng = np.random.default_rng(0)
        quiet = rng.normal(0, 0.2, 60)
        loud = rng.normal(0, 1.0, 60)
        m_quiet = simulated_mde(quiet, horizon=5, effects_abs=self._grid())
        m_loud = simulated_mde(loud, horizon=5, effects_abs=self._grid())
        assert m_quiet < m_loud

    def test_longer_horizon_lowers_mde(self):
        rng = np.random.default_rng(1)
        g = rng.normal(0, 0.5, 80)
        short = simulated_mde(g, horizon=2, effects_abs=self._grid())
        long = simulated_mde(g, horizon=8, effects_abs=self._grid())
        assert long <= short

    def test_returns_inf_when_grid_too_small(self):
        rng = np.random.default_rng(2)
        g = rng.normal(0, 1.0, 60)
        assert simulated_mde(g, horizon=5, effects_abs=[0.0, 1e-6]) == float("inf")

    def test_returns_inf_when_too_few_pre_periods(self):
        assert simulated_mde(np.zeros(3), horizon=5,
                             effects_abs=[1.0]) == float("inf")


# ----------------------------------------------------------------------
# Pareto helper + compare_pareto
# ----------------------------------------------------------------------

class TestPareto:
    def test_pareto_mask_excludes_dominated(self):
        fit = np.array([0.1, 0.2, 0.3, 0.25])
        mde = np.array([2.0, 1.0, 0.5, 2.5])     # idx 3 dominated by idx 1
        mask = _pareto_mask(fit, mde)
        assert mask.tolist() == [True, True, True, False]

    def _panel(self, T=20, cols=("a", "b", "c", "d")):
        rng = np.random.default_rng(3)
        data = {c: 100 + rng.normal(0, 1, T) for c in cols}
        return pd.DataFrame(data)

    def test_compare_pareto_shape_and_flags(self):
        Ywide = self._panel()
        specs = [
            DesignSpec("SYNDES", "S1", {"a": 1.0, "b": -1.0}, ["a"]),
            DesignSpec("SYNDES", "S2", {"a": 1.0, "c": -1.0}, ["a"]),
            DesignSpec("GEOLIFT", "G1", {"b": 1.0, "d": -1.0}, ["b"]),
        ]
        out = compare_pareto(specs, Ywide, n_pre=15, horizon=5)
        assert set(out.columns) >= {"method", "label", "treated", "fit_rmse",
                                    "mde_pct", "pareto"}
        assert len(out) == 3
        assert set(out["method"]) == {"SYNDES", "GEOLIFT"}
        # each method has at least one Pareto-optimal design
        assert out.groupby("method")["pareto"].any().all()
        assert (out["fit_rmse"] >= 0).all()


# ----------------------------------------------------------------------
# Adapters + cross-method integration (small real fits)
# ----------------------------------------------------------------------

def _shared_panel(N=8, T=16, n_post=5, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2)); L = rng.uniform(0.3, 1.0, (N, 2))
    lvl = rng.uniform(8.0, 12.0, N)
    Y = lvl + F @ L.T + rng.normal(scale=0.3, size=(T, N))
    rows = [{"unit": f"u{j}", "time": t, "Y": float(Y[t, j]),
             "post": int(t >= T - n_post)}
            for j in range(N) for t in range(T)]
    return pd.DataFrame(rows), T, n_post


def _ywide(df):
    w = df.pivot(index="time", columns="unit", values="Y").sort_index()
    return w


class TestAdaptersIntegration:
    def test_from_syndes_specs(self):
        from mlsynth import SYNDES
        df, T, n_post = _shared_panel()
        res = SYNDES({"df": df, "outcome": "Y", "unitid": "unit", "time": "time",
                      "K": 2, "mode": "two_way_global", "post_col": "post",
                      "run_inference": False, "top_K": 4}).fit()
        specs = from_syndes(res)
        assert len(specs) == len(res.pool)
        assert all(s.method == "SYNDES" for s in specs)
        # contrast keys are real units; treated weights sum ~ +1
        for s in specs:
            assert set(s.contrast) <= set(f"u{j}" for j in range(8))
            assert sum(v for v in s.contrast.values() if v > 0) == pytest.approx(1.0, abs=1e-6)

    def test_from_geolift_specs(self):
        from mlsynth import GEOLIFT
        df, T, n_post = _shared_panel()
        res = GEOLIFT({"df": df, "outcome": "Y", "unitid": "unit", "time": "time",
                       "treatment_size": 2, "durations": [n_post],
                       "effect_sizes": [0.0, 0.1], "lookback_window": 1,
                       "post_col": "post", "how": "mean", "fixed_effects": True,
                       "alpha": 0.1, "ns": 150, "seed": 0,
                       "display_graphs": False}).fit()
        specs = from_geolift(res)
        assert len(specs) == len(res.search.candidates)
        assert all(s.method == "GEOLIFT" for s in specs)
        for s in specs:
            # treated side sums to +1, control side to ~ -1 -> net ~ 0
            assert sum(s.contrast.values()) == pytest.approx(0.0, abs=1e-6)

    def test_cross_method_compare_and_plot(self):
        from mlsynth import GEOLIFT, SYNDES
        df, T, n_post = _shared_panel()
        syn = SYNDES({"df": df, "outcome": "Y", "unitid": "unit", "time": "time",
                      "K": 2, "mode": "two_way_global", "post_col": "post",
                      "run_inference": False, "top_K": 4}).fit()
        gl = GEOLIFT({"df": df, "outcome": "Y", "unitid": "unit", "time": "time",
                      "treatment_size": 2, "durations": [n_post],
                      "effect_sizes": [0.0, 0.1], "lookback_window": 1,
                      "post_col": "post", "how": "mean", "fixed_effects": True,
                      "alpha": 0.1, "ns": 150, "seed": 0,
                      "display_graphs": False}).fit()
        specs = from_syndes(syn) + from_geolift(gl)
        Ywide = _ywide(df)
        out = compare_pareto(specs, Ywide, n_pre=T - n_post, horizon=5)
        assert set(out["method"]) == {"SYNDES", "GEOLIFT"}
        assert np.isfinite(out["fit_rmse"]).all()
        # at least one design somewhere reaches the power grid (finite MDE)
        assert np.isfinite(out["mde_pct"]).any()
        ax = plot_compare_pareto(out)
        assert ax is not None
