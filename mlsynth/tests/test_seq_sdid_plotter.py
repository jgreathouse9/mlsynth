"""Coverage tests for mlsynth.utils.seq_sdid_helpers.plotter.plot_seq_sdid."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth.estimators.seq_sdid import SequentialSDID
from mlsynth.utils.seq_sdid_helpers.plotter import plot_seq_sdid


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _staggered_panel(seed=0, cohort_adoptions=(10, 15, 20, None),
                     n_per_cohort=20, T=30, true_effect=-5.0):
    rng = np.random.default_rng(seed)
    records = []
    unit_id = 0
    for adoption in cohort_adoptions:
        for _ in range(n_per_cohort):
            alpha_i = rng.standard_normal() * 5 + 50
            theta_i = rng.standard_normal() * 2
            for t in range(T):
                psi_t = np.cos(t / 5.0)
                beta_t = 0.5 * t
                mu = alpha_i + beta_t + theta_i * psi_t
                outcome = mu + rng.standard_normal() * 0.3
                treated = 0
                if adoption is not None and t >= adoption:
                    treated = 1
                    outcome += true_effect
                records.append({"unit": f"u_{unit_id}", "year": 2000 + t,
                                "y": outcome, "treated": treated})
            unit_id += 1
    return pd.DataFrame(records)


def _fit(n_bootstrap=20):
    return SequentialSDID({
        "df": _staggered_panel(), "outcome": "y", "treat": "treated",
        "unitid": "unit", "time": "year",
        "n_bootstrap": n_bootstrap, "eta": 1.0, "seed": 7,
        "display_graphs": False,
    }).fit()


def test_plot_no_save(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_seq_sdid(_fit(), title="my title", save=False)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_plot_save_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_seq_sdid(_fit(), save=True)
    assert (tmp_path / "seq_sdid_event_study.png").exists()


def test_plot_save_string(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "ssdid.png"
    plot_seq_sdid(_fit(), save=str(out))
    assert out.exists()


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    SequentialSDID({
        "df": _staggered_panel(), "outcome": "y", "treat": "treated",
        "unitid": "unit", "time": "year", "n_bootstrap": 0,
        "display_graphs": True,
    }).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
