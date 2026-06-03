"""Branch coverage for mlsynth.utils.syndes_helpers.plotter."""

import matplotlib

matplotlib.use("Agg")

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES
from mlsynth.exceptions import MlsynthPlottingError
from mlsynth.utils.syndes_helpers import plotter as P
from mlsynth.utils.syndes_helpers.plotter import (
    plot_syndes_design,
    plot_global_design,
    plot_per_unit_design,
    plot_relaxed_design,
    _stack_pre_post,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _panel(n_units=10, T=14, n_post=4, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((T, n_units)) * 0.4
    Y += np.linspace(0, 1, T)[:, None]
    Y += rng.standard_normal(n_units)
    rows = [{"unit": j, "time": t, "y": float(Y[t, j]), "post": int(t >= T - n_post)}
            for j in range(n_units) for t in range(T)]
    return pd.DataFrame(rows)


def _fit(mode, *, post_col="post", K=2):
    cfg = {"df": _panel(), "outcome": "y", "unitid": "unit", "time": "time",
           "K": K, "mode": mode, "run_inference": False}
    if post_col is not None:
        cfg["post_col"] = post_col
    return SYNDES(cfg).fit()


# --------------------------------------------------------------------------
# Dispatch via plot_syndes_design
# --------------------------------------------------------------------------

def test_dispatch_global():
    res = _fit("two_way_global")
    plot_syndes_design(res)


def test_dispatch_per_unit():
    res = _fit("per_unit")
    plot_syndes_design(res)


def test_dispatch_annealed():
    res = _fit("two_way_global_annealed")
    assert res.mode == "two_way_global_annealed"
    plot_syndes_design(res)


def test_dispatch_unknown_mode_raises():
    res = _fit("two_way_global")
    bad_design = dataclasses.replace(res.design, mode="not_a_mode")
    res2 = dataclasses.replace(res, design=bad_design)
    with pytest.raises(MlsynthPlottingError, match="Unknown SYNDES plot mode"):
        plot_syndes_design(res2)


# --------------------------------------------------------------------------
# plot_global_design
# --------------------------------------------------------------------------

def test_global_design_direct():
    res = _fit("two_way_global")
    plot_global_design(res)


def test_global_design_missing_weights_raises():
    res = _fit("two_way_global")
    bad = dataclasses.replace(res.design, treated_weights=None)
    res2 = dataclasses.replace(res, design=bad)
    with pytest.raises(MlsynthPlottingError, match="Missing treated/control"):
        plot_global_design(res2)


def test_global_design_missing_control_weights_raises():
    res = _fit("two_way_global")
    bad = dataclasses.replace(res.design, control_weights=None)
    res2 = dataclasses.replace(res, design=bad)
    with pytest.raises(MlsynthPlottingError, match="Missing treated/control"):
        plot_global_design(res2)


# --------------------------------------------------------------------------
# plot_per_unit_design
# --------------------------------------------------------------------------

def test_per_unit_design_multiple_units():
    res = _fit("per_unit", K=2)
    assert res.design.selected_unit_indices.size == 2
    plot_per_unit_design(res)


def test_per_unit_design_single_unit_wraps_axes():
    res = _fit("per_unit", K=1)
    assert res.design.selected_unit_indices.size == 1
    plot_per_unit_design(res)


def test_per_unit_design_missing_q_raises():
    res = _fit("per_unit")
    bad = dataclasses.replace(res.design, q=None)
    res2 = dataclasses.replace(res, design=bad)
    with pytest.raises(MlsynthPlottingError, match="requires q weights"):
        plot_per_unit_design(res2)


# --------------------------------------------------------------------------
# plot_relaxed_design
# --------------------------------------------------------------------------

def test_relaxed_design_direct():
    res = _fit("two_way_global_annealed")
    plot_relaxed_design(res)


def test_relaxed_design_no_inputs_raises():
    res = _fit("two_way_global_annealed")
    res2 = dataclasses.replace(res, inputs=None)
    with pytest.raises(MlsynthPlottingError, match="requires inputs"):
        plot_relaxed_design(res2)


# --------------------------------------------------------------------------
# _stack_pre_post: Y_post present vs None
# --------------------------------------------------------------------------

def test_stack_pre_post_with_post():
    res = _fit("two_way_global")
    full = _stack_pre_post(res)
    assert full.shape[0] == (res.inputs.Y_pre.shape[0] + res.inputs.Y_post.shape[0])


def test_stack_pre_post_design_only_no_post():
    # No post_col / T0 -> Y_post is None -> stack returns just Y_pre.
    res = SYNDES({"df": _panel(), "outcome": "y", "unitid": "unit",
                  "time": "time", "K": 2, "mode": "two_way_global"}).fit()
    assert res.inputs.Y_post is None
    full = _stack_pre_post(res)
    np.testing.assert_array_equal(full, res.inputs.Y_pre)
    # And the global plotter works on a design-only result too.
    plot_global_design(res)
