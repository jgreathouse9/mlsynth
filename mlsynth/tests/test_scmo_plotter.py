"""Branch coverage for mlsynth.utils.scmo_helpers.plotter.plot_scmo."""

import matplotlib

matplotlib.use("Agg")

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth.estimators.scmo import SCMO
from mlsynth.utils.scmo_helpers import CONCATENATED, AVERAGED, MA
from mlsynth.utils.scmo_helpers.plotter import plot_scmo


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


SPEC = {"year": 1972, "vars": {"a": "y1", "b": "y2"}}


def _panel():
    rng = np.random.default_rng(0)
    N, T, T0, K = 8, 20, 15, 3
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    rows = []
    for i in range(N):
        load = rng.uniform(0.5, 1.5, size=2)
        pop = rng.uniform(5, 20)
        for t in range(T):
            base = 50.0 + f[t] @ load
            y = {f"y{k+1}": base * (1 + 0.1 * k) + rng.normal(scale=0.3)
                 for k in range(K)}
            treat = int(i == 0 and t >= T0)
            if treat:
                y["y1"] += 5.0
            rows.append({"unit": f"u{i}", "time": 1960 + t,
                         "Population levels": pop, "treat": treat, **y})
    return pd.DataFrame(rows)


def _fit():
    return SCMO({"df": _panel(), "outcome": "y1", "treat": "treat",
                 "unitid": "unit", "time": "time", "spec": SPEC,
                 "schemes": [CONCATENATED, AVERAGED, MA],
                 "display_graphs": False}).fit()


def test_display_multi_scheme():
    # default counterfactual_color is str; multiple fits; T0 < len(years)
    # so the intervention axvline is drawn; show() path.
    res = _fit()
    plot_scmo(res, outcome="Outcome", time="Year")


def test_counterfactual_color_list():
    res = _fit()
    plot_scmo(res, outcome="Y", time="T", treated_color="navy",
              counterfactual_color=["green", "purple", "orange"])


def test_save_str(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = _fit()
    plot_scmo(res, outcome="Y", time="T", save="scmo_out.png")
    assert (tmp_path / "scmo_out.png").exists()


def test_save_true_default_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = _fit()
    plot_scmo(res, outcome="Y", time="T", save=True)
    assert (tmp_path / "scmo_estimates.png").exists()


def test_t0_at_or_after_end_skips_axvline():
    # T0 >= len(years): the intervention axvline branch is skipped.
    res = _fit()
    big_t0 = dataclasses.replace(res.inputs, T0=len(res.inputs.time_index.labels))
    res2 = dataclasses.replace(res, inputs=big_t0)
    plot_scmo(res2, outcome="Y", time="T")


def test_plotting_failure_warns():
    # Force an attribute error inside the try-block to hit the except/warning
    # branch (results without a valid inputs object).
    class Broken:
        @property
        def inputs(self):
            raise RuntimeError("boom")
    with pytest.warns(UserWarning, match="SCMO plotting failed"):
        plot_scmo(Broken(), outcome="Y", time="T")
