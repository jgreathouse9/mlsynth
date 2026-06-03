"""Coverage tests for mlsynth.utils.spcd_helpers.plotter."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import SPCD
from mlsynth.exceptions import MlsynthPlottingError
from mlsynth.utils.spcd_helpers.plotter import (
    plot_detectability,
    plot_mde_bars,
    plot_power_curves,
    plot_spcd_design,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _make_panel(n_units=20, T=60, T_post=10, L=3, sigma=0.5, seed=0):
    rng = np.random.default_rng(seed)
    gamma = rng.standard_normal((n_units, L))
    nu = rng.standard_normal((T, L))
    Y = nu @ gamma.T + sigma * rng.standard_normal((T, n_units))
    rows = []
    for i in range(n_units):
        for t in range(T):
            rows.append({"unitid": f"u{i:02d}", "time": t, "y": Y[t, i],
                         "post": int(t >= T - T_post)})
    return pd.DataFrame(rows)


_INF = dict(mde_n_sims=60, mde_n_trials=30, mde_horizon_grid=[2, 5, 8])


def _single_with_power():
    return SPCD({"df": _make_panel(), "outcome": "y", "unitid": "unitid",
                 "time": "time", "post_col": "post",
                 "enable_inference": True, "display_graph": False,
                 **_INF}).fit()


def _single_no_inference():
    return SPCD({"df": _make_panel(), "outcome": "y", "unitid": "unitid",
                 "time": "time", "post_col": "post",
                 "enable_inference": False, "display_graph": False}).fit()


def _multi_arm_with_power():
    df = _make_panel(n_units=24)
    code = df["unitid"].str[1:].astype(int)
    df["arm"] = np.where(code < 8, "A", np.where(code < 16, "B", "C"))
    return SPCD({"df": df, "arm": "arm", "outcome": "y", "unitid": "unitid",
                 "time": "time", "post_col": "post",
                 "enable_inference": True, "display_graph": False,
                 **_INF}).fit()


# --------------------------------------------------------------------------- #
# plot_mde_bars
# --------------------------------------------------------------------------- #
def test_mde_bars_single(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    fig = plot_mde_bars(_single_with_power())
    assert fig is not None


def test_mde_bars_multi_arm_with_pooled(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    fig = plot_mde_bars(_multi_arm_with_power())
    assert fig is not None


def test_mde_bars_with_provided_ax():
    fig, ax = plt.subplots()
    out = plot_mde_bars(_single_with_power(), ax=ax)
    assert out is fig


def test_mde_bars_no_mde_raises():
    with pytest.raises(MlsynthPlottingError):
        plot_mde_bars(_single_no_inference())


# --------------------------------------------------------------------------- #
# plot_power_curves
# --------------------------------------------------------------------------- #
def test_power_curves_single():
    fig = plot_power_curves(_single_with_power())
    assert fig is not None


def test_power_curves_multi_arm():
    fig = plot_power_curves(_multi_arm_with_power())
    assert fig is not None


def test_power_curves_provided_ax():
    fig, ax = plt.subplots()
    out = plot_power_curves(_single_with_power(), ax=ax)
    assert out is fig


def test_power_curves_none_raises():
    with pytest.raises(MlsynthPlottingError):
        plot_power_curves(_single_no_inference())


# --------------------------------------------------------------------------- #
# plot_detectability
# --------------------------------------------------------------------------- #
def test_detectability_single():
    fig = plot_detectability(_single_with_power())
    assert fig is not None


def test_detectability_multi_arm():
    fig = plot_detectability(_multi_arm_with_power())
    assert fig is not None


def test_detectability_provided_ax():
    fig, ax = plt.subplots()
    out = plot_detectability(_single_with_power(), ax=ax)
    assert out is fig


def test_detectability_none_raises():
    # No inference => no power => empty detectability => raise.
    with pytest.raises(MlsynthPlottingError):
        plot_detectability(_single_no_inference())


# --------------------------------------------------------------------------- #
# plot_spcd_design  (+ _stack_pre_post both branches)
# --------------------------------------------------------------------------- #
def test_spcd_design_with_post(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_spcd_design(_single_with_power())
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_spcd_design_no_inputs_raises():
    import dataclasses
    res = _single_no_inference()
    broken = dataclasses.replace(res, inputs=None)
    with pytest.raises(MlsynthPlottingError):
        plot_spcd_design(broken)


def test_spcd_design_no_post(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    df = _make_panel(T=50, T_post=0).drop(columns=["post"])
    res = SPCD({"df": df, "outcome": "y", "unitid": "unitid", "time": "time",
                "enable_inference": False, "display_graph": False}).fit()
    plot_spcd_design(res)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
