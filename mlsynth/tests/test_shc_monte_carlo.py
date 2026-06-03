"""Coverage tests for shc_helpers.monte_carlo.

Uses the smallest viable simulation config (1-2 reps) to keep runtime low.
"""

from __future__ import annotations

import numpy as np

from mlsynth.utils.shc_helpers import monte_carlo as mc_mod
from mlsynth.utils.shc_helpers.monte_carlo import _fit_one, monte_carlo_shc
from mlsynth.utils.shc_helpers.simulation import simulate_shc_panel

# Smallest valid simulation dimensions (T_o = m*(4h+1)).
_M, _H, _N = 6, 4, 4


def test_fit_one_returns_window_length():
    df, _ = simulate_shc_panel(m=_M, h=_H, n=_N, P=10.0, sigma=0.1, seed=0)
    cf = _fit_one(df, m=_M, use_augmented=False)
    assert cf.shape == (_M + _N,)
    assert np.all(np.isfinite(cf))


def test_monte_carlo_basic_structure():
    out = monte_carlo_shc(
        n_reps=2, m=_M, h=_H, n=_N, P=10.0, sigma=0.1,
        k_grid=(1, 3, _N), seed=0,
    )
    assert set(out) == {"mse_pre", "mse_post", "n_reps", "config"}
    assert out["n_reps"] == 2
    assert np.isfinite(out["mse_pre"])
    assert out["mse_pre"] >= 0.0
    # post errors keyed by k, all finite & non-negative
    assert set(out["mse_post"]) == {1, 3, _N}
    for v in out["mse_post"].values():
        assert np.isfinite(v) and v >= 0.0
    assert out["config"]["m"] == _M


def test_k_grid_filters_values_above_n():
    """k values exceeding n are dropped before fitting."""
    out = monte_carlo_shc(
        n_reps=1, m=_M, h=_H, n=_N, P=10.0, sigma=0.1,
        k_grid=(1, _N, _N + 100), seed=0,
    )
    assert (_N + 100) not in out["mse_post"]
    assert set(out["mse_post"]) == {1, _N}


def test_augmented_and_irregular_path():
    out = monte_carlo_shc(
        n_reps=1, m=_M, h=_H, n=_N, P=10.0, sigma=0.1,
        regular=False, use_augmented=True, k_grid=(1, _N), seed=5,
    )
    assert out["config"]["use_augmented"] is True
    assert out["config"]["regular"] is False
    assert out["n_reps"] == 1
    assert np.isfinite(out["mse_pre"])


def test_wrong_size_counterfactual_skips_all_reps(monkeypatch):
    """When every fit returns a mis-sized cf, all reps are skipped -> NaN aggregates."""
    monkeypatch.setattr(mc_mod, "_fit_one", lambda *a, **k: np.zeros(3))
    out = monte_carlo_shc(
        n_reps=2, m=_M, h=_H, n=_N, P=10.0, sigma=0.1,
        k_grid=(1, _N), seed=0,
    )
    assert out["n_reps"] == 0
    assert np.isnan(out["mse_pre"])
    assert all(np.isnan(v) for v in out["mse_post"].values())
