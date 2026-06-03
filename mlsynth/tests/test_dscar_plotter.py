"""Line-coverage tests for ``mlsynth.utils.dscar_helpers.plotter``.

Builds ``DSCARResults`` directly to exercise the two-panel trajectory/gap
plot and every title branch (SE present/absent, relative-effect NaN vs
finite) plus the ``save`` vs ``plt.show()`` paths.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

from mlsynth.utils.dscar_helpers.plotter import plot_dsc
from mlsynth.utils.dscar_helpers.structures import (
    DSCARFit,
    DSCARInputs,
    DSCARResults,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _make_results(*, se, att_relative, n_treated=1) -> DSCARResults:
    T, T0 = 12, 8
    rng = np.random.default_rng(1)
    Y_treated_mean = rng.standard_normal(T) + 5.0
    Y0_hat = Y_treated_mean - 0.5
    gap = Y_treated_mean - Y0_hat
    inputs = DSCARInputs(
        Y=rng.standard_normal((4, T)),
        Y_lag1=rng.standard_normal((4, T)),
        X=np.empty((4, T, 0)),
        var_names=(),
        y_name="pm25",
        treated_labels=tuple(f"t{i}" for i in range(n_treated)),
        donor_labels=("d0", "d1", "d2"),
        time_labels=np.arange(T),
        N=4, T=T, T0=T0, T1=T - T0, n_treated=n_treated,
    )
    fit = DSCARFit(
        weights=np.full((T, 3), 1.0 / 3.0),
        Y0_hat=Y0_hat,
        Y_treated_mean=Y_treated_mean,
        gap=gap,
        att=float(gap[T0:].mean()),
        att_relative=att_relative,
        se=se,
    )
    return DSCARResults(inputs=inputs, fit=fit)


def test_show_with_se_and_relative(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    res = _make_results(se=0.12, att_relative=0.05)
    plot_dsc(res)
    assert list(tmp_path.iterdir()) == []


def test_show_no_se_and_nan_relative(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    res = _make_results(se=None, att_relative=float("nan"), n_treated=3)
    plot_dsc(res)
    assert list(tmp_path.iterdir()) == []


def test_save(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    res = _make_results(se=0.3, att_relative=-0.1)
    out = tmp_path / "dsc.png"
    plot_dsc(res, save=str(out), treated_color="black",
             counterfactual_color="green")
    assert out.exists()
