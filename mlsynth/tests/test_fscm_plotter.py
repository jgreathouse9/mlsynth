"""Branch coverage for mlsynth.utils.fscm_helpers.plotter.plot_fscm."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import FSCM
from mlsynth.utils.fscm_helpers.plotter import plot_fscm


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _panel(n_donors=10, T=30, T0=20, seed=0, effect=-5.0):
    rng = np.random.default_rng(seed)
    n_units = n_donors + 1
    K = 3
    factors = np.cumsum(rng.normal(size=(T, K)), axis=0)
    loadings = rng.uniform(0.2, 1.0, size=(n_units, K))
    base = factors @ loadings.T + rng.normal(scale=0.3, size=(T, n_units))
    rows = []
    states = ["treated"] + [f"d{j}" for j in range(n_donors)]
    for u, name in enumerate(states):
        for t in range(T):
            y = base[t, u]
            if name == "treated" and t >= T0:
                y += effect
            rows.append({"unit": name, "time": t, "y": y,
                         "treated": int(name == "treated" and t >= T0)})
    return pd.DataFrame(rows)


def _fit(forward_selection=True):
    return FSCM({"df": _panel(), "outcome": "y", "treat": "treated",
                 "unitid": "unit", "time": "time",
                 "forward_selection": forward_selection,
                 "display_graphs": False}).fit()


def test_two_panel_with_cv_curve():
    # forward_selection -> selection_path is not None -> two panels + CV curve.
    res = _fit(forward_selection=True)
    assert res.selection_path is not None
    plot_fscm(res, outcome="Outcome", time="Year")


def test_single_panel_no_path():
    # No forward selection -> selection_path None -> one panel only.
    res = _fit(forward_selection=False)
    assert res.selection_path is None
    plot_fscm(res, outcome="Outcome", time="Year")


def test_counterfactual_color_list():
    res = _fit(forward_selection=True)
    plot_fscm(res, outcome="Y", time="T",
              treated_color="navy", counterfactual_color=["green", "purple"])


def test_save_str(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = _fit(forward_selection=True)
    plot_fscm(res, outcome="Y", time="T", save="fscm_out.png")
    assert (tmp_path / "fscm_out.png").exists()


def test_save_true_default_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = _fit(forward_selection=False)
    plot_fscm(res, outcome="Y", time="T", save=True)
    assert (tmp_path / "fscm.png").exists()
