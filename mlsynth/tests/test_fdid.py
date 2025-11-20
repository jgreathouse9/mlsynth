import pytest
import numpy as np
import pandas as pd
from typing import Any, Dict
from mlsynth.estimators.fdid import FDIDOutput
from mlsynth.estimators.fdid import FDID, FDIDConfig, FDIDOutput
from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError
from mlsynth.utils.resultutils import DID_org
# Import the Pydantic models that DID_org returns
from mlsynth.config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)

# Public and private functions to test
from mlsynth.utils.selector_helpers import (
    _choose_optimal_subset,
    _compute_did_all,
    _did_from_mean,
    _compute_fdid_result,
    _precompute_treated_stats,
    _record_verbose_step,
    _select_best_donor,
    _update_synthetic_control,
    _r2_batch
)

from mlsynth.utils.estutils import fast_DID_selector


@pytest.fixture
def sample_fdid_data() -> pd.DataFrame:
    """Create a small balanced panel dataset for FDID testing."""
    data = {
        "unit": ["T"] * 4 + ["C1"] * 4 + ["C2"] * 4,
        "time": [1, 2, 3, 4] * 3,
        "y": [10, 12, 20, 22, 8, 9, 10, 11, 9, 10, 11, 12],
        "treated_indicator": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    return pd.DataFrame(data)


def test_fdid_creation(sample_fdid_data: pd.DataFrame):
    config = FDIDConfig(
        df=sample_fdid_data,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    estimator = FDID(config=config)
    assert isinstance(estimator, FDID)


def test_fdid_fit_smoke(sample_fdid_data: pd.DataFrame):
    """Smoke test for FDID fit method."""
    config = FDIDConfig(
        df=sample_fdid_data,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    estimator = FDID(config=config)
    results = estimator.fit()

    # FDID.fit now returns FDIDOutput
    assert isinstance(results, FDIDOutput)
    assert "FDID" in results.results
    assert "DID" in results.results


def test_fdid_fit_insufficient_periods(sample_fdid_data: pd.DataFrame):
    """Test FDID fit with insufficient pre or post periods."""
    df = sample_fdid_data.copy()
    # Make pre-period only 1 row for treated
    df.loc[(df["unit"] == "T") & (df["time"] == 1), "treated_indicator"] = 0
    df.loc[(df["unit"] == "T") & (df["time"] == 2), "treated_indicator"] = 1

    config = FDIDConfig(
        df=df,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    estimator = FDID(config=config)

    # Expect MlsynthEstimationError due to all-NaN slice in selector
    with pytest.raises(MlsynthEstimationError, match="Insufficient pre-periods for estimation."):
        estimator.fit()


def test_fdid_fit_insufficient_donors(sample_fdid_data: pd.DataFrame):
    """Test FDID fit with too few or no donor units."""
    # 1. No donor units
    df_no_donors = sample_fdid_data[sample_fdid_data["unit"] == "T"].copy()
    config_no_donors = FDIDConfig(
        df=df_no_donors,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    estimator_no_donors = FDID(config=config_no_donors)
    with pytest.raises(MlsynthDataError, match="No donor units found"):
        estimator_no_donors.fit()

    # 2. Only one donor unit
    df_one_donor = sample_fdid_data[sample_fdid_data["unit"].isin(["T", "C1"])].copy()
    config_one_donor = FDIDConfig(
        df=df_one_donor,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    estimator_one_donor = FDID(config=config_one_donor)
    results = estimator_one_donor.fit()
    assert isinstance(results, FDIDOutput)
    assert "FDID" in results.results
    assert "DID" in results.results


def test_fdid_fit_nan_in_outcome(sample_fdid_data: pd.DataFrame):
    """Test FDID fit with NaN values in the outcome variable."""
    df_nan = sample_fdid_data.copy()
    df_nan.loc[(df_nan["unit"] == "T") & (df_nan["time"] == 1), "y"] = np.nan

    config = FDIDConfig(
        df=df_nan,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    estimator = FDID(config=config)

    with pytest.raises(MlsynthEstimationError, match="All-NaN slice encountered"):
        estimator.fit()


def test_fdid_results_structure_and_types(sample_fdid_data: pd.DataFrame):
    """Check that FDID.fit returns correct structure and types."""
    config = FDIDConfig(
        df=sample_fdid_data,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    estimator = FDID(config=config)
    results = estimator.fit()

    assert isinstance(results, FDIDOutput)
    for method_name, method_result in results.results.items():
        # Each result should have effects, time_series, weights, and inference
        assert hasattr(method_result, "effects")
        assert hasattr(method_result, "time_series")
        assert hasattr(method_result, "weights")
        assert hasattr(method_result, "inference")


def test_fdid_plotting_runs(sample_fdid_data):
    config = FDIDConfig(
        df=sample_fdid_data,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=True,
        save=False
    )
    estimator = FDID(config=config)
    results = estimator.fit()
    assert isinstance(results, FDIDOutput)

# -----------------------------
# Test _precompute_treated_stats
# -----------------------------
def test_precompute_treated_stats_basic():
    arr = np.array([1, 2, 3])
    y_c, ss_tot = _precompute_treated_stats(arr)
    assert np.allclose(y_c, arr - arr.mean())
    assert ss_tot > 0

def test_precompute_treated_stats_zeros():
    arr = np.zeros(5)
    y_c, ss_tot = _precompute_treated_stats(arr)
    assert ss_tot == 1e-12  # clipped
    assert np.allclose(y_c, np.zeros_like(arr))

def test_precompute_treated_stats_single():
    arr = np.array([42])
    y_c, ss_tot = _precompute_treated_stats(arr)
    assert y_c == 0
    assert ss_tot == 1e-12


# -----------------------------
# Test _r2_batch
# -----------------------------
def test_r2_batch_basic():
    y_c = np.array([1.0, -1.0])
    ss_tot = np.sum(y_c ** 2)
    X_pre = np.array([[1.0, 2.0], [-1.0, -2.0]])
    r2 = _r2_batch(y_c, ss_tot, X_pre)
    assert r2.shape[0] == 2
    assert np.all(r2 <= 1.0)


def test_r2_batch_zero_variance():
    y_c = np.array([1.0, -1.0])
    ss_tot = np.sum(y_c ** 2)
    X_pre = np.ones((2, 3))
    r2 = _r2_batch(y_c, ss_tot, X_pre)
    assert np.all(np.isfinite(r2))


# -----------------------------
# Test _did_from_mean
# -----------------------------
def test_did_from_mean_basic():
    treated = np.array([1, 2, 3, 4])
    control = np.array([0.5, 0.5, 2, 2])
    T0 = 2
    res = _did_from_mean(treated, control, T0)
    assert "Effects" in res
    assert "Fit" in res
    assert "Vectors" in res
    assert "Observed" in res["Vectors"]
    assert np.allclose(res["Vectors"]["Counterfactual"], np.round(res["Vectors"]["Counterfactual"], 3))


def test_did_from_mean_single_period_post():
    treated = np.array([1, 2])
    control = np.array([0.5, 0.5])
    T0 = 1
    res = _did_from_mean(treated, control, T0)
    assert not np.isnan(res["Effects"]["ATT"])


def test_did_from_mean_all_zero_control():
    treated = np.array([1, 2, 3])
    control = np.zeros(3)
    T0 = 2
    res = _did_from_mean(treated, control, T0)
    assert np.all(np.isfinite([res["Effects"]["ATT"], res["Fit"]["R-Squared"]]))


# -----------------------------
# Test _compute_did_all
# -----------------------------
def test_compute_did_all_basic():
    treated = np.array([1, 2, 3, 4])  # T = 4
    controls = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ])  # T x N (4 x 2)
    T0 = 2

    # mean across donors
    mean_all = controls.mean(axis=1)
    res = _compute_did_all(treated, controls, T0)

    assert np.isclose(res['Effects']['ATT'], 2.0)  # expected ATT for this simple example


# -----------------------------
# Test _select_best_donor & _update_synthetic_control
# -----------------------------
def test_select_best_donor_single_remaining():
    X_pre = np.array([[1, 2], [3, 4]])
    current_mean = np.array([1, 3])
    y_c = np.array([0.5, -0.5])
    ss_tot = np.sum(y_c ** 2)
    idx, r2, r2_all = _select_best_donor(X_pre, current_mean, 1, [1], y_c, ss_tot)
    assert idx == 1
    assert r2 == r2_all[0]


def test_update_synthetic_control_basic():
    current_mean = np.array([1.0, 2.0])
    control_outcomes = np.array([[2.0, 3.0], [4.0, 5.0]])
    updated = _update_synthetic_control(current_mean, control_outcomes, 0, 1)
    assert updated.shape == current_mean.shape


# -----------------------------
# Test _record_verbose_step
# -----------------------------
def test_record_verbose_step_appends():
    intermediary = []
    donor_names = ["A", "B"]
    _record_verbose_step(intermediary, 0, 0, 0.9, np.array([0.5, 0.6]), [0], donor_names, np.array([1, 2]), 1)
    assert len(intermediary) == 1
    assert intermediary[0]["iteration"] == 1
    assert "selected_name" in intermediary[0]


# -----------------------------
# Test _choose_optimal_subset
# -----------------------------
def test_choose_optimal_subset_basic():
    selected = [0, 1, 2]
    R2_path = [0.1, 0.5, 0.3]

    idxs, path = _choose_optimal_subset(selected, R2_path)

    # up to argmax of R²
    assert idxs == [0, 1]
    assert path == R2_path[:len(idxs)]


def test_choose_optimal_subset_empty():
    # should handle empty inputs gracefully
    idxs, path = _choose_optimal_subset([], [])
    assert idxs == []
    assert path == []


# -----------------------------
# Test _compute_fdid_result
# -----------------------------
def test_compute_fdid_result_basic():
    treated = np.array([1, 2, 3, 4])  # T = 4
    controls = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ])  # T x N
    optimal_idxs = [0, 1]
    T0 = 2
    R2_path = [0.5, 0.6]
    donor_names = ["C1", "C2"]

    res = _compute_fdid_result(treated, controls, optimal_idxs, T0, R2_path, donor_names)

    assert 'ATT' in res['Effects']
    assert res['Vectors']['Counterfactual'].shape == treated.shape



def test_fast_DID_selector_basic():
    # Small deterministic dataset
    treated = np.array([1, 2, 3, 4])  # T=4
    controls = np.array([
        [1, 0, 1],   # T x N = 4 x 3
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    T0 = 2
    donor_names = ["C1", "C2", "C3"]

    result = fast_DID_selector(treated, controls, T0, donor_names, verbose=True)

    # Check keys
    assert "DID" in result
    assert "FDID" in result

    did_res = result["DID"]
    fdid_res = result["FDID"]

    # --- DID sanity checks ---
    # Pre/post split
    treated_pre = treated[:T0]
    treated_post = treated[T0:]
    ctrl_pre = controls[:T0].mean(axis=1)
    ctrl_post = controls[T0:].mean(axis=1)

    expected_att = (treated_post.mean() - treated_pre.mean()) - (ctrl_post.mean() - ctrl_pre.mean())
    assert np.isclose(did_res["Effects"]["ATT"], expected_att)

    # --- FDID sanity checks ---
    # Ensure optimal subset indices are valid
    optimal_idxs = fdid_res["selected_controls"]
    assert all(0 <= idx < controls.shape[1] for idx in optimal_idxs)

    # Ensure selected_names match selected_controls
    selected_names = fdid_res["selected_names"]
    assert selected_names == [donor_names[i] for i in optimal_idxs]

    # Check that counterfactual is numeric and same length as treated
    counterfactual = fdid_res["Vectors"]["Counterfactual"]
    assert counterfactual.shape == treated.shape
    assert np.issubdtype(counterfactual.dtype, np.floating)

    # Gap vector sanity
    gap = fdid_res["Vectors"]["Gap"]
    assert gap.shape[0] == treated.shape[0]
    assert gap.shape[1] == 2  # diff + time index adjustment
















def test_did_org_basic():
    """Test DID_org with a minimal, valid input dictionary."""
    result_dict = {
        "Effects": {"ATT": 5, "Percent ATT": 0.2},
        "Fit": {"R-Squared": 0.9, "T0 RMSE": 1.5},
        "Vectors": {
            "Observed": np.array([10, 12, 14]),
            "Counterfactual": np.array([9, 11, 13]),
            "Gap": np.array([[1], [1], [1]])
        },
        "Inference": {"P-Value": 0.05, "95% CI": (4, 6), "SE": 0.5},
        "selected_names": ["C1", "C2"]
    }
    preppeddict = {"time_labels": [1, 2, 3]}

    res = DID_org(result_dict, preppeddict, method_name="FDID")

    # Type checks
    assert isinstance(res, BaseEstimatorResults)
    assert isinstance(res.effects, EffectsResults)
    assert isinstance(res.fit_diagnostics, FitDiagnosticsResults)
    assert isinstance(res.time_series, TimeSeriesResults)
    assert isinstance(res.weights, WeightsResults)
    assert isinstance(res.inference, InferenceResults)
    assert isinstance(res.method_details, MethodDetailsResults)

    # Effects
    assert res.effects.att == 5
    assert res.effects.att_percent == 0.2

    # Time series
    np.testing.assert_array_equal(res.time_series.observed_outcome, [10, 12, 14])
    np.testing.assert_array_equal(res.time_series.counterfactual_outcome, [9, 11, 13])
    np.testing.assert_array_equal(res.time_series.estimated_gap, [1, 1, 1])
    np.testing.assert_array_equal(res.time_series.time_periods, [1, 2, 3])

    # Weights
    assert res.weights.donor_weights == {"C1": 0.5, "C2": 0.5}
    assert res.weights.summary_stats["num_selected"] == 2
    assert np.isclose(res.weights.summary_stats["total_weight"], 1.0)

    # Inference
    assert res.inference.p_value == 0.05
    assert res.inference.ci_lower == 4
    assert res.inference.ci_upper == 6
    assert res.inference.standard_error == 0.5

    # Method details
    assert res.method_details.method_name == "FDID"
    assert res.method_details.parameters_used["selected_names"] == ["C1", "C2"]


def test_did_org_no_selected_names():
    """Test DID_org when no donor units are selected."""
    result_dict = {
        "Effects": {"ATT": 2, "Percent ATT": 0.1},
        "Fit": {"R-Squared": 0.8, "T0 RMSE": 0.5},
        "Vectors": {
            "Observed": np.array([1, 2]),
            "Counterfactual": np.array([1, 2]),
            "Gap": np.array([[0], [0]])
        },
        "Inference": {"P-Value": 0.2, "95% CI": (1, 3), "SE": 0.1},
        "selected_names": []
    }
    preppeddict = {"time_labels": [1, 2]}

    res = DID_org(result_dict, preppeddict)

    assert res.weights.donor_weights is None
    assert res.weights.summary_stats is None


def test_did_org_extra_keys():
    """Test DID_org when Effects and Inference dictionaries have extra keys."""
    result_dict = {
        "Effects": {"ATT": 1, "Percent ATT": 0.05, "Other": 99},
        "Fit": {"R-Squared": 1, "T0 RMSE": 0.1},
        "Vectors": {
            "Observed": np.array([1]),
            "Counterfactual": np.array([0]),
            "Gap": np.array([[1]])
        },
        "Inference": {"P-Value": 0.05, "95% CI": (0, 1), "SE": 0.01, "Extra": "foo"},
        "selected_names": ["C1"]
    }
    preppeddict = {"time_labels": [1]}

    res = DID_org(result_dict, preppeddict)

    assert res.effects.additional_effects == {"Other": 99}
    assert res.inference.details == {"Extra": "foo"}


def test_did_org_gap_none():
    """Test DID_org when the Gap array is None."""
    result_dict = {
        "Effects": {"ATT": 0, "Percent ATT": 0},
        "Fit": {"R-Squared": 0, "T0 RMSE": 0},
        "Vectors": {
            "Observed": np.array([0]),
            "Counterfactual": np.array([0]),
            "Gap": None
        },
        "Inference": {},
        "selected_names": ["C1"]
    }
    preppeddict = {"time_labels": [1]}

    res = DID_org(result_dict, preppeddict)
    assert res.time_series.estimated_gap is None
