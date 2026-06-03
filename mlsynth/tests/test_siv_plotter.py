"""Coverage tests for mlsynth.utils.siv_helpers.plotter.plot_event_study."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import SIV
from mlsynth.exceptions import MlsynthPlottingError
from mlsynth.utils.siv_helpers.plotter import plot_event_study


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _shift_share_panel(seed=0, J=14, T=12, T0=6):
    rng = np.random.default_rng(seed)
    Z = np.zeros((J, T))
    Z[:, T0:] = rng.normal(0, 1, (J, T - T0))
    R = np.zeros((J, T))
    R[:, T0:] = 0.5 * Z[:, T0:] + rng.normal(0, 0.1, (J, T - T0))
    theta = -0.2
    Y = theta * R + rng.normal(0, 0.05, (J, T))
    df = pd.DataFrame({
        "unit": np.repeat(np.arange(J), T),
        "time": np.tile(np.arange(T), J),
        "y": Y.reshape(-1),
        "r": R.reshape(-1),
        "z": Z.reshape(-1),
    })
    return df, T0


def _conformal_results():
    df, T0 = _shift_share_panel()
    return SIV({
        "df": df, "outcome": "y", "treat": "r", "instrument": "z",
        "unitid": "unit", "time": "time", "T0": T0,
        "inference_method": "conformal", "n_permutations": 200,
    }).fit()


def test_event_study_normal(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_event_study(_conformal_results())
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_event_study_label_fallback(monkeypatch, tmp_path):
    """Mismatched time-label length triggers the arange fallback branch."""
    monkeypatch.chdir(tmp_path)
    res = _conformal_results()
    coefs = res.inference.event_study_coefs
    short = coefs[:-2]  # length now differs from time labels
    new_inf = dataclasses.replace(res.inference, event_study_coefs=short)
    res2 = dataclasses.replace(res, inference=new_inf)
    plot_event_study(res2)


def test_event_study_empty_raises():
    res = _conformal_results()
    empty_inf = dataclasses.replace(
        res.inference, event_study_coefs=np.array([], dtype=float)
    )
    res2 = dataclasses.replace(res, inference=empty_inf)
    with pytest.raises(MlsynthPlottingError):
        plot_event_study(res2)
