"""Coverage tests for mlsynth.utils.ppscm_helpers.plotter.plot_ppscm."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import PPSCM
from mlsynth.utils.ppscm_helpers.plotter import plot_ppscm


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _staggered_panel(seed=0, adoption_offsets=(10, 15, 20), N_donors=8,
                     T=30, true_effect=-3.0, noise=0.4):
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((T, 2))
    loadings_donors = rng.standard_normal((N_donors, 2)) * 0.5
    loadings_treated = loadings_donors.mean(axis=0)
    records = []
    for j, T_j in enumerate(adoption_offsets):
        base_load = loadings_treated + 0.1 * rng.standard_normal(2)
        series = factors @ base_load + rng.standard_normal(T) * noise
        series[T_j:] += true_effect
        for t in range(T):
            records.append({"unit": f"treated_{j}", "year": 2000 + t,
                            "y": float(series[t]), "tr": int(t >= T_j)})
    for dd in range(N_donors):
        series = factors @ loadings_donors[dd] + rng.standard_normal(T) * noise
        for t in range(T):
            records.append({"unit": f"d_{dd}", "year": 2000 + t,
                            "y": float(series[t]), "tr": 0})
    return pd.DataFrame(records)


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="tr", unitid="unit", time="year",
                display_graphs=False, run_inference=True)
    base.update(kw)
    return base


def _fit():
    return PPSCM(_cfg(_staggered_panel())).fit()


def test_plot_no_save(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_ppscm(_fit(), save=False)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_plot_save_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_ppscm(_fit(), save=True, title="custom title")
    assert (tmp_path / "ppscm_event_study.png").exists()


def test_plot_save_string(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "ev.png"
    plot_ppscm(_fit(), save=str(out))
    assert out.exists()


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    PPSCM(_cfg(_staggered_panel(), display_graphs=True)).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


# --------------------------------------------------------------------------- #
# the legend has to name the interval that was actually computed
# --------------------------------------------------------------------------- #
# ``inference_method`` chooses between three intervals, and ``method=
# 'callaway_santanna'`` switches it to the influence function without the
# caller naming it. A legend that hardcodes one of the three reports whichever
# it was written for, which is the failure #459 and #463 bottomed out at --
# reached here through the plot instead of through the results object.
def _legend_labels(res, **kw):
    plot_ppscm(res, **kw)
    ax = plt.gcf().axes[0]
    legend = ax.get_legend()
    return [t.get_text() for t in legend.get_texts()], ax


def _panel_with_never_treated():
    """``method='callaway_santanna'`` needs a never-treated pool."""
    return _staggered_panel()


@pytest.mark.parametrize("over,expected,forbidden", [
    ({"inference_method": "jackknife"}, "jackknife", "influence-function"),
    ({"inference_method": "bootstrap", "n_boot": 25}, "bootstrap", "jackknife"),
    ({"method": "callaway_santanna"}, "influence-function", "jackknife"),
])
def test_the_legend_names_the_inference_that_ran(over, expected, forbidden):
    res = PPSCM(_cfg(_panel_with_never_treated(), **over)).fit()
    ran = res.design.conventions["inference_method"]
    labels, _ = _legend_labels(res)
    band = [t for t in labels if "CI" in t]
    assert band, f"expected a band label, got {labels}"
    assert expected in band[0], f"{ran!r} ran, legend said {band[0]!r}"
    assert forbidden not in band[0], f"{ran!r} ran, legend said {band[0]!r}"


def test_no_inference_claims_no_interval():
    """With inference off the band is degenerate -- ``ci`` is the point path
    twice over -- so drawing it and calling it a 95% CI reports an interval that
    was never computed."""
    res = PPSCM(_cfg(_panel_with_never_treated(), run_inference=False)).fit()
    assert res.design.conventions["inference_method"] is None
    labels, ax = _legend_labels(res)
    assert not [t for t in labels if "CI" in t], labels
    assert not ax.collections, "a degenerate band must not be drawn"
