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
