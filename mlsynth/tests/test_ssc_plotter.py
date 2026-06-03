"""Branch coverage for mlsynth.utils.ssc_helpers.plotter.plot_ssc."""

import matplotlib

matplotlib.use("Agg")

import dataclasses

import matplotlib.pyplot as plt
import pytest

from mlsynth import SSC
from mlsynth.utils.ssc_helpers import simulate_ssc_panel
from mlsynth.utils.ssc_helpers.structures import SSCBand
from mlsynth.utils.ssc_helpers.plotter import plot_ssc


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _fit(inference=True, seed=1):
    df = simulate_ssc_panel(n_units=14, n_never=3, T0=40, S=5,
                            n_factors=2, base_effect=1.0, seed=seed)
    return SSC({"df": df, "outcome": "Y", "treat": "treated",
                "unitid": "unit", "time": "time",
                "inference": inference, "display_graphs": False}).fit()


def test_display_path_with_bands_and_att_band(monkeypatch):
    # show() branch: bands present, att_band present (prints p-value).
    res = _fit(inference=True)
    assert res.event_bands and res.att_band is not None
    plot_ssc(res)


def test_display_path_no_inference(monkeypatch):
    # att_band is None and event_bands empty: no band, title has no p-value.
    res = _fit(inference=False)
    assert res.att_band is None and not res.event_bands
    plot_ssc(res)


def test_save_str_with_extension(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    res = _fit(inference=True)
    out = tmp_path / "ssc_es.png"
    plot_ssc(res, save=str(out))
    assert out.exists()
    assert "Plot saved to" in capsys.readouterr().out


def test_save_str_without_extension(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = _fit(inference=True)
    plot_ssc(res, save="ssc_noext")
    assert (tmp_path / "ssc_noext.png").exists()


def test_save_true_default_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = _fit(inference=True)
    plot_ssc(res, save=True)
    assert (tmp_path / "SSC_event_study.png").exists()


def test_empty_event_att_returns_early():
    # es falsy -> early return, nothing drawn.
    res = _fit(inference=False)
    res2 = dataclasses.replace(res, event_att={})
    assert plot_ssc(res2) is None


def test_counterfactual_color_list_and_empty_list():
    res = _fit(inference=True)
    # List branch: takes first colour.
    plot_ssc(res, counterfactual_color=["green", "blue"])
    # Empty-list branch: falls back to "red".
    plot_ssc(res, counterfactual_color=[])


def test_att_band_present_but_no_event_bands():
    # Cover the att_band-not-None title branch while event_bands is empty
    # (fill_between skipped, p-value still appended).
    res = _fit(inference=True)
    band = SSCBand(label=None, point=res.att, lower=-0.1, upper=0.1,
                   p_value=0.04, n_cells=5)
    res2 = dataclasses.replace(res, event_bands={}, att_band=band)
    plot_ssc(res2)
