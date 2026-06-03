"""Coverage tests for shc_helpers.orchestration (solve_shc / summarize_effects)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.shc_helpers.orchestration import (
    _DEFAULT_BANDWIDTH_GRID,
    _lambda_grid,
    solve_shc,
    summarize_effects,
)
from mlsynth.utils.shc_helpers.setup import prepare_shc_inputs
from mlsynth.utils.shc_helpers.structures import SHCDesign


def _panel(T0: int = 8, n_post: int = 2, seed: int = 0) -> pd.DataFrame:
    T = T0 + n_post
    t = np.arange(1, T + 1)
    rng = np.random.RandomState(seed)
    y = np.linspace(1.0, 2.0, T) + 0.3 * np.sin(t / 2.0) + rng.normal(0.0, 0.02, T)
    return pd.DataFrame(
        {
            "unit": np.ones(T, dtype=int),
            "time": t,
            "y": y,
            "treated": (t > T0).astype(int),
        }
    )


def _inputs(T0=8, n_post=2, m=4, seed=0):
    return prepare_shc_inputs(_panel(T0, n_post, seed), "y", "treated", "unit", "time", m=m)


def test_lambda_grid_monotone_and_positive():
    rng = np.random.default_rng(0)
    L_pre = rng.normal(size=(5, 4))
    grid = _lambda_grid(L_pre, n_lambda=20)
    assert grid.shape == (20,)
    assert np.all(grid > 0)
    # geometric decreasing from lambda_max
    assert np.all(np.diff(grid) <= 0)


def test_solve_shc_plain_default_grid():
    inp = _inputs()
    design = solve_shc(inp)
    assert isinstance(design, SHCDesign)
    assert design.use_augmented is False
    assert design.best_lambda is None
    # weights simplex
    assert np.isclose(design.weights.sum(), 1.0)
    assert np.all(design.weights >= -1e-8)
    # bandwidth came from the default grid range
    assert _DEFAULT_BANDWIDTH_GRID.min() <= design.bandwidth <= _DEFAULT_BANDWIDTH_GRID.max()
    # latent_pre spans the pre-period
    assert design.latent_pre.shape == (inp.T0,)
    # counterfactual window spans m + n
    assert design.counterfactual_window.shape[0] == inp.m + inp.n
    # selected blocks and block_weights agree with nonzero weights
    nz = np.nonzero(design.weights)[0]
    assert design.selected_blocks == [int(i) for i in nz]
    assert len(design.block_weights) == len(nz)
    for i in nz:
        assert design.block_weights[f"block@{int(i)}"] == float(design.weights[i])


def test_solve_shc_custom_grid_used():
    inp = _inputs()
    design = solve_shc(inp, bandwidth_grid=[0.5, 1.0, 2.0])
    assert 0.5 <= design.bandwidth <= 2.0


def test_solve_shc_augmented_sets_lambda():
    inp = _inputs()
    design = solve_shc(inp, use_augmented=True, bandwidth_grid=[0.5, 1.0, 2.0])
    assert design.use_augmented is True
    assert design.best_lambda is not None
    assert np.isfinite(design.best_lambda)
    assert np.isclose(design.weights.sum(), 1.0)


def test_summarize_effects_structure():
    inp = _inputs()
    design = solve_shc(inp, bandwidth_grid=[0.5, 1.0, 2.0])
    att, att_pct, observed, cf, gap, window_time, fit = summarize_effects(inp, design)

    assert np.isfinite(att)
    assert np.isfinite(att_pct)
    assert observed.shape == cf.shape == gap.shape
    assert observed.shape[0] == inp.m + inp.n
    np.testing.assert_allclose(gap, observed - cf)
    np.testing.assert_allclose(cf, design.counterfactual_window)
    assert window_time.shape[0] == inp.m + inp.n
    assert set(fit) == {"rmse_pre", "rmse_post", "r_squared_pre"}


def test_summarize_effects_window_mismatch_raises():
    inp = _inputs()
    design = solve_shc(inp, bandwidth_grid=[0.5, 1.0, 2.0])
    # corrupt the counterfactual window length to trigger the guard
    bad = dataclasses.replace(
        design, counterfactual_window=design.counterfactual_window[:-1]
    )
    with pytest.raises(MlsynthEstimationError, match="lengths differ"):
        summarize_effects(inp, bad)
