"""Tests for the cross-method counterfactual comparison helper.

Test-first per CLAUDE.md: smoke, invariants, edge cases, and failure modes for
``compare_counterfactuals`` / ``plot_counterfactual_comparison``. The helper
extracts each method's counterfactual (and its prediction interval, when present)
from either a standardized :class:`BaseEstimatorResults` or an explicit spec, so
one figure can overlay several methods and save the paper the per-method loop.
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlsynth.config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    TimeSeriesResults,
    WeightsResults,
)
from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.counterfactual_compare import (
    CounterfactualComparison,
    compare_counterfactuals,
    plot_counterfactual_comparison,
)


# --------------------------------------------------------------------------- #
# Fixtures: build real standardized results so the extraction path is exercised
# --------------------------------------------------------------------------- #
def _result(cf, *, time=None, observed=None, att=None, pre_rmse=None,
            band=None):
    """A minimal but real BaseEstimatorResults for the comparison helper."""
    ts = TimeSeriesResults(
        counterfactual_outcome=np.asarray(cf, dtype=float),
        observed_outcome=(None if observed is None
                          else np.asarray(observed, dtype=float)),
        time_periods=(None if time is None else np.asarray(time)),
    )
    inf = None
    if band is not None:
        inf = InferenceResults(details=band)
    return BaseEstimatorResults(
        effects=(None if att is None else EffectsResults(att=att)),
        fit_diagnostics=(None if pre_rmse is None
                         else FitDiagnosticsResults(rmse_pre=pre_rmse)),
        time_series=ts,
        inference=inf,
    )


@pytest.fixture
def two_results():
    time = [2000, 2001, 2002, 2003]
    a = _result([1.0, 2.0, 3.0, 4.0], time=time, observed=[1.1, 2.1, 2.5, 3.0],
                att=-0.7, pre_rmse=0.10,
                band={"periods": [2002, 2003],
                      "counterfactual_lower": [2.6, 3.5],
                      "counterfactual_upper": [3.4, 4.5]})
    b = _result([1.0, 2.1, 3.3, 4.6], time=time, observed=[1.1, 2.1, 2.5, 3.0],
                att=-1.05, pre_rmse=0.09)
    return {"alpha": a, "beta": b}


# --------------------------------------------------------------------------- #
# Smoke
# --------------------------------------------------------------------------- #
def test_smoke_returns_comparison(two_results):
    cmp = compare_counterfactuals(two_results)
    assert isinstance(cmp, CounterfactualComparison)
    assert list(cmp.summary.index) == ["alpha", "beta"]   # insertion order kept
    assert set(cmp.curves["method"]) == {"alpha", "beta"}
    assert np.isfinite(cmp.curves["counterfactual"]).all()


# --------------------------------------------------------------------------- #
# Invariants / extraction
# --------------------------------------------------------------------------- #
def test_summary_reads_stored_att_and_pre_rmse(two_results):
    cmp = compare_counterfactuals(two_results)
    assert cmp.summary.loc["alpha", "att"] == pytest.approx(-0.7)
    assert cmp.summary.loc["beta", "att"] == pytest.approx(-1.05)
    assert cmp.summary.loc["alpha", "pre_rmse"] == pytest.approx(0.10)


def test_curves_carry_band_only_where_present(two_results):
    cmp = compare_counterfactuals(two_results)
    a = cmp.curves[cmp.curves["method"] == "alpha"].set_index("period")
    # band declared for 2002, 2003 only -> NaN in the pre-band rows
    assert np.isnan(a.loc[2000, "lower"])
    assert a.loc[2002, "lower"] == pytest.approx(2.6)
    assert a.loc[2003, "upper"] == pytest.approx(4.5)
    # method with no inference -> all bounds NaN
    b = cmp.curves[cmp.curves["method"] == "beta"]
    assert b["lower"].isna().all() and b["upper"].isna().all()


def test_periods_come_from_result_time(two_results):
    cmp = compare_counterfactuals(two_results)
    a = cmp.curves[cmp.curves["method"] == "alpha"]
    assert list(a["period"]) == [2000, 2001, 2002, 2003]


def test_observed_inferred_from_first_result(two_results):
    cmp = compare_counterfactuals(two_results)
    assert cmp.observed is not None
    assert list(cmp.observed.index) == [2000, 2001, 2002, 2003]
    assert cmp.observed.loc[2000] == pytest.approx(1.1)


# --------------------------------------------------------------------------- #
# fit_window: windowed RMSE of observed vs counterfactual, off the same object
# --------------------------------------------------------------------------- #
def test_no_window_means_no_window_rmse_column(two_results):
    cmp = compare_counterfactuals(two_results)
    assert "window_rmse" not in cmp.summary.columns


def test_window_rmse_matches_manual(two_results):
    # observed = [1.1, 2.1, 2.5, 3.0]; alpha cf = [1,2,3,4] at 2000..2003
    cmp = compare_counterfactuals(two_results, fit_window=(2000, 2001))
    obs, cf = np.array([1.1, 2.1]), np.array([1.0, 2.0])
    expect = float(np.sqrt(np.mean((obs - cf) ** 2)))
    assert cmp.summary.loc["alpha", "window_rmse"] == pytest.approx(expect)
    # full-pre stored rmse stays available alongside the windowed one
    assert "pre_rmse" in cmp.summary.columns


def test_window_outside_support_is_nan(two_results):
    cmp = compare_counterfactuals(two_results, fit_window=(1800, 1801))
    assert np.isnan(cmp.summary.loc["alpha", "window_rmse"])


def test_window_without_observed_raises():
    # plain-array specs carry no observed and none is supplied
    with pytest.raises(MlsynthConfigError):
        compare_counterfactuals({"m": [1.0, 2.0, 3.0]}, time=[1, 2, 3],
                                fit_window=(1, 2))


def test_window_uses_explicit_observed():
    cmp = compare_counterfactuals(
        {"m": [1.0, 2.0, 3.0]}, time=[1, 2, 3],
        observed=[1.0, 2.0, 4.0], fit_window=(3, 3))
    # only period 3 in window: |4 - 3| = 1
    assert cmp.summary.loc["m", "window_rmse"] == pytest.approx(1.0)


def test_window_must_be_pair():
    with pytest.raises(MlsynthConfigError):
        compare_counterfactuals({"m": [1.0, 2.0]}, time=[1, 2],
                                observed=[1.0, 2.0], fit_window=(1, 2, 3))


def test_explicit_observed_overrides(two_results):
    obs = pd.Series([9.0, 9.0, 9.0, 9.0], index=[2000, 2001, 2002, 2003])
    cmp = compare_counterfactuals(two_results, observed=obs)
    assert cmp.observed.loc[2001] == pytest.approx(9.0)


# --------------------------------------------------------------------------- #
# Flexible specs (the spillover path: explicit arrays, no standardized result)
# --------------------------------------------------------------------------- #
def test_accepts_plain_array_spec():
    cmp = compare_counterfactuals({"m": [1.0, 2.0, 3.0]}, time=[1, 2, 3])
    row = cmp.curves
    assert list(row["counterfactual"]) == [1.0, 2.0, 3.0]
    assert row["lower"].isna().all()
    assert np.isnan(cmp.summary.loc["m", "att"])


def test_accepts_dict_spec_with_bounds_and_att():
    spec = {"counterfactual": [1.0, 2.0, 3.0, 4.0],
            "lower": [0.5, 1.5, 2.5, 3.5],
            "upper": [1.5, 2.5, 3.5, 4.5],
            "att": -2.0, "time": [1, 2, 3, 4]}
    cmp = compare_counterfactuals({"m": spec})
    assert cmp.summary.loc["m", "att"] == pytest.approx(-2.0)
    assert cmp.curves["lower"].notna().all()


def test_dict_spec_band_aligned_by_periods():
    # bounds shorter than the curve, aligned by their declared periods
    spec = {"counterfactual": [1.0, 2.0, 3.0, 4.0], "time": [1, 2, 3, 4],
            "periods": [3, 4], "lower": [2.5, 3.5], "upper": [3.5, 4.5]}
    cmp = compare_counterfactuals({"m": spec})
    g = cmp.curves.set_index("period")
    assert np.isnan(g.loc[1, "lower"])
    assert g.loc[3, "lower"] == pytest.approx(2.5)


def test_single_method_allowed():
    cmp = compare_counterfactuals({"only": [1.0, 2.0]}, time=[1, 2])
    assert list(cmp.summary.index) == ["only"]


# --------------------------------------------------------------------------- #
# Failure modes (translated errors, surfaced not swallowed)
# --------------------------------------------------------------------------- #
def test_empty_methods_raises():
    with pytest.raises(MlsynthConfigError):
        compare_counterfactuals({})


def test_one_sided_band_raises():
    spec = {"counterfactual": [1.0, 2.0], "lower": [0.5, 1.5], "time": [1, 2]}
    with pytest.raises(MlsynthConfigError):
        compare_counterfactuals({"m": spec})


def test_length_mismatch_raises():
    spec = {"counterfactual": [1.0, 2.0, 3.0], "time": [1, 2]}
    with pytest.raises(MlsynthEstimationError):
        compare_counterfactuals({"m": spec})


def test_missing_counterfactual_in_result_raises():
    bad = _result([1.0, 2.0], time=[1, 2])
    bad.time_series.counterfactual_outcome = None
    with pytest.raises(MlsynthEstimationError):
        compare_counterfactuals({"m": bad})


def test_unalignable_band_raises():
    # band length neither equal to the curve nor accompanied by periods
    spec = {"counterfactual": [1.0, 2.0, 3.0, 4.0], "time": [1, 2, 3, 4],
            "lower": [2.5, 3.5], "upper": [3.5, 4.5]}
    with pytest.raises(MlsynthEstimationError):
        compare_counterfactuals({"m": spec})


def test_methods_must_be_mapping():
    with pytest.raises(MlsynthConfigError):
        compare_counterfactuals([1.0, 2.0])           # not a {label: spec} map


def test_dict_spec_without_counterfactual_raises():
    with pytest.raises(MlsynthConfigError):
        compare_counterfactuals({"m": {"att": -1.0, "time": [1, 2]}})


def test_empty_counterfactual_raises():
    with pytest.raises(MlsynthEstimationError):
        compare_counterfactuals({"m": []}, time=[])


def test_band_bounds_length_disagree_raises():
    spec = {"counterfactual": [1.0, 2.0], "time": [1, 2],
            "lower": [0.5], "upper": [1.5, 2.5]}
    with pytest.raises(MlsynthConfigError):
        compare_counterfactuals({"m": spec})


def test_band_periods_length_mismatch_raises():
    spec = {"counterfactual": [1.0, 2.0, 3.0], "time": [1, 2, 3],
            "periods": [2, 3], "lower": [1.5], "upper": [2.5]}
    with pytest.raises(MlsynthEstimationError):
        compare_counterfactuals({"m": spec})


def test_band_period_outside_axis_raises():
    spec = {"counterfactual": [1.0, 2.0, 3.0], "time": [1, 2, 3],
            "periods": [9], "lower": [1.5], "upper": [2.5]}
    with pytest.raises(MlsynthEstimationError):
        compare_counterfactuals({"m": spec})


def test_observed_array_length_mismatch_raises():
    with pytest.raises(MlsynthEstimationError):
        compare_counterfactuals({"m": [1.0, 2.0, 3.0]}, time=[1, 2, 3],
                                observed=[9.0, 9.0])


# --------------------------------------------------------------------------- #
# Fallbacks
# --------------------------------------------------------------------------- #
def test_time_defaults_to_index_when_absent():
    cmp = compare_counterfactuals({"m": [5.0, 6.0, 7.0]})
    assert list(cmp.curves["period"]) == [0, 1, 2]


def test_observed_bare_array_paired_with_time():
    cmp = compare_counterfactuals({"m": [1.0, 2.0]}, time=[1, 2],
                                  observed=[8.0, 9.0])
    assert cmp.observed.loc[2] == pytest.approx(9.0)


def test_inference_details_not_a_mapping_yields_no_band():
    r = _result([1.0, 2.0], time=[1, 2], band="not-a-dict")
    cmp = compare_counterfactuals({"m": r})
    assert cmp.curves["lower"].isna().all()


def test_plot_without_axis_creates_and_returns(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    cmp = compare_counterfactuals({"only": [1.0, 2.0]}, time=[1, 2])
    ax = plot_counterfactual_comparison(cmp)
    assert ax is not None
    plt.close("all")


# --------------------------------------------------------------------------- #
# Plotting (smoke: returns an axis, draws the expected artists)
# --------------------------------------------------------------------------- #
def test_plot_returns_axis_and_draws_lines(two_results):
    cmp = compare_counterfactuals(two_results)
    _, ax = plt.subplots()
    out = plot_counterfactual_comparison(cmp, ax=ax)
    assert out is ax
    # observed + two counterfactual lines
    assert len(ax.get_lines()) >= 3
    plt.close("all")


def test_plot_method_delegates(two_results):
    cmp = compare_counterfactuals(two_results)
    _, ax = plt.subplots()
    assert cmp.plot(ax=ax) is ax
    plt.close("all")


def test_plot_dodge_and_colors(two_results):
    cmp = compare_counterfactuals(two_results)
    _, ax = plt.subplots()
    plot_counterfactual_comparison(
        cmp, ax=ax, dodge=0.5,
        colors={"alpha": "C0", "beta": "C3"},
        styles={"alpha": "--", "beta": "-."})
    # error-bar containers exist because alpha carries a band
    assert len(ax.containers) >= 1
    plt.close("all")


def test_plot_band_fill_draws_shaded_region(two_results):
    from matplotlib.collections import PolyCollection
    cmp = compare_counterfactuals(two_results)
    _, ax = plt.subplots()
    plot_counterfactual_comparison(cmp, ax=ax, band="fill")
    # a shaded region (fill_between -> PolyCollection) is drawn, not error bars
    assert any(isinstance(c, PolyCollection) for c in ax.collections)
    assert len(ax.containers) == 0
    plt.close("all")


def test_plot_band_errorbar_is_default(two_results):
    from matplotlib.collections import PolyCollection
    cmp = compare_counterfactuals(two_results)
    _, ax = plt.subplots()
    plot_counterfactual_comparison(cmp, ax=ax)            # default == errorbar
    assert len(ax.containers) >= 1
    assert not any(isinstance(c, PolyCollection) for c in ax.collections)
    plt.close("all")


def test_plot_band_invalid_raises(two_results):
    cmp = compare_counterfactuals(two_results)
    _, ax = plt.subplots()
    with pytest.raises(ValueError):
        plot_counterfactual_comparison(cmp, ax=ax, band="bogus")
    plt.close("all")



# --------------------------------------------------------------------------- #
# Donor weights: extraction frame + plot_weights (so the paper drops the bar
# chart code the same way it dropped the per-method overlay loop)
# --------------------------------------------------------------------------- #
def _result_w(cf, weights, **kw):
    res = _result(cf, **kw)
    res.weights = WeightsResults(donor_weights=weights)
    return res


@pytest.fixture
def two_weighted():
    t = [2000, 2001, 2002]
    a = _result_w([1.0, 2.0, 3.0], {"d1": 0.6, "d2": 0.4, "d3": 0.0005},
                  time=t, att=-0.7)
    b = _result_w([1.0, 2.1, 3.2], {"d2": 0.5, "d4": 0.5},
                  time=t, att=-0.9)
    return {"alpha": a, "beta": b}


def test_weights_frame_is_tidy(two_weighted):
    cmp = compare_counterfactuals(two_weighted)
    w = cmp.weights
    assert set(w.columns) >= {"method", "donor", "weight"}
    a = w[w["method"] == "alpha"].set_index("donor")["weight"]
    assert a.loc["d1"] == pytest.approx(0.6)
    assert set(w[w["method"] == "beta"]["donor"]) == {"d2", "d4"}


def test_weights_absent_method_contributes_no_rows():
    cmp = compare_counterfactuals(
        {"has": _result_w([1.0, 2.0], {"d1": 1.0}, time=[0, 1]),
         "none": [1.0, 2.0]})            # bare array -> no weights
    assert set(cmp.weights["method"]) == {"has"}


def test_weights_from_mapping_spec():
    cmp = compare_counterfactuals(
        {"m": {"counterfactual": [1.0, 2.0], "time": [0, 1],
               "weights": {"d1": 0.7, "d2": 0.3}}})
    w = cmp.weights.set_index("donor")["weight"]
    assert w.loc["d1"] == pytest.approx(0.7)


def test_plot_weights_smoke(two_weighted):
    cmp = compare_counterfactuals(two_weighted)
    fig, ax = plt.subplots()
    out = cmp.plot_weights(ax=ax, colors={"alpha": "C0", "beta": "C3"})
    assert out is ax
    assert len(ax.patches) > 0          # bars drawn
    plt.close("all")


def test_plot_weights_threshold_drops_negligible(two_weighted):
    cmp = compare_counterfactuals(two_weighted)
    fig, ax = plt.subplots()
    cmp.plot_weights(ax=ax, threshold=1e-3)
    labels = {t.get_text() for t in ax.get_xticklabels()}
    assert "d3" not in labels           # 0.0005 < threshold
    assert {"d1", "d2", "d4"} <= labels
    plt.close("all")


def test_plot_weights_label_callable(two_weighted):
    cmp = compare_counterfactuals(two_weighted)
    fig, ax = plt.subplots()
    cmp.plot_weights(ax=ax, label=lambda d: d.upper())
    labels = {t.get_text() for t in ax.get_xticklabels()}
    assert "D1" in labels
    plt.close("all")


def test_plot_weights_max_donors_keeps_top(two_weighted):
    cmp = compare_counterfactuals(two_weighted)
    fig, ax = plt.subplots()
    cmp.plot_weights(ax=ax, max_donors=2)
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert len(labels) == 2
    plt.close("all")


def test_plot_weights_no_weights_raises():
    cmp = compare_counterfactuals({"m": [1.0, 2.0, 3.0]}, time=[0, 1, 2])
    with pytest.raises(MlsynthConfigError):
        cmp.plot_weights()


def test_plot_weights_without_axis_creates_and_returns(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    cmp = compare_counterfactuals(
        {"m": {"counterfactual": [1.0, 2.0], "time": [0, 1],
               "weights": {"d1": 0.6, "d2": 0.4}}})
    ax = cmp.plot_weights()
    assert ax is not None
    plt.close("all")
