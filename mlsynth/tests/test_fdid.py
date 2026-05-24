import pytest
import numpy as np
import pandas as pd

from mlsynth.estimators.fdid import FDID
from mlsynth.config_models import FDIDConfig
from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError

from mlsynth.utils.fdid_helpers import (
    FDIDInputs,
    FDIDMethodFit,
    FDIDResults,
    assemble_fdid_results,
    did_from_mean,
    forward_did_select,
    prepare_fdid_inputs,
)
from mlsynth.utils.fdid_helpers.estimation import (
    _choose_optimal_subset,
    _compute_fdid_result,
    _r2_batch,
    _record_verbose_step,
    _select_best_donor,
    _update_synthetic_control,
)
from mlsynth.utils.fdid_helpers.inference import did_inference


@pytest.fixture
def sample_fdid_data() -> pd.DataFrame:
    """Create a small balanced panel dataset for FDID testing."""
    data = {
        "unit": ["T"] * 4 + ["C1"] * 4 + ["C2"] * 4,
        "time": [1, 2, 3, 4] * 3,
        "y": [10, 12, 20, 22, 8, 9, 10, 11, 9, 10, 11, 12],
        "treated_indicator": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    return pd.DataFrame(data)


# -----------------------------
# Estimator-level behaviour
# -----------------------------
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
    config = FDIDConfig(
        df=sample_fdid_data,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    results = FDID(config=config).fit()

    assert isinstance(results, FDIDResults)
    assert isinstance(results.fdid, FDIDMethodFit)
    assert isinstance(results.did, FDIDMethodFit)
    assert set(results.methods.keys()) == {"FDID", "DID"}
    # Convenience aliases delegate to the FDID (primary) fit.
    assert results.att == results.fdid.att
    assert results.counterfactual.shape == results.inputs.y.shape


def test_fdid_fit_insufficient_periods(sample_fdid_data: pd.DataFrame):
    df = sample_fdid_data.copy()
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
    with pytest.raises(MlsynthEstimationError, match="Insufficient pre-periods for estimation."):
        FDID(config=config).fit()


def test_fdid_fit_insufficient_donors(sample_fdid_data: pd.DataFrame):
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
    with pytest.raises(MlsynthDataError, match="No donor units found"):
        FDID(config=config_no_donors).fit()

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
    results = FDID(config=config_one_donor).fit()
    assert isinstance(results, FDIDResults)
    assert set(results.methods.keys()) == {"FDID", "DID"}


def test_fdid_fit_nan_in_outcome(sample_fdid_data: pd.DataFrame):
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
    with pytest.raises(MlsynthEstimationError, match="All-NaN slice encountered"):
        FDID(config=config).fit()


def test_fdid_results_structure_and_types(sample_fdid_data: pd.DataFrame):
    config = FDIDConfig(
        df=sample_fdid_data,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    results = FDID(config=config).fit()

    assert isinstance(results, FDIDResults)
    for fit in results.methods.values():
        assert isinstance(fit, FDIDMethodFit)
        assert fit.counterfactual.shape == results.inputs.y.shape
        assert fit.gap.shape == results.inputs.y.shape
        assert isinstance(fit.donor_weights, dict)
        assert len(fit.ci) == 2
    # DID uses every donor; FDID uses a (possibly smaller) selected subset.
    assert len(results.did.selected_names) == results.inputs.n_donors
    assert set(results.fdid.selected_names).issubset(set(results.inputs.donor_names))


def test_fdid_plotting_runs(sample_fdid_data):
    config = FDIDConfig(
        df=sample_fdid_data,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=True,
        save=False,
    )
    results = FDID(config=config).fit()
    assert isinstance(results, FDIDResults)


def test_fdid_aggregators(sample_fdid_data):
    config = FDIDConfig(
        df=sample_fdid_data,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    results = FDID(config=config).fit()
    assert set(results.att_by_method().keys()) == {"FDID", "DID"}
    assert set(results.se_by_method().keys()) == {"FDID", "DID"}
    assert set(results.ci_by_method().keys()) == {"FDID", "DID"}
    assert results.att_by_method()["FDID"] == results.fdid.att


# -----------------------------
# setup.prepare_fdid_inputs
# -----------------------------
def test_prepare_fdid_inputs(sample_fdid_data):
    inputs = prepare_fdid_inputs(
        sample_fdid_data, outcome="y", treat="treated_indicator",
        unitid="unit", time="time",
    )
    assert isinstance(inputs, FDIDInputs)
    assert inputs.pre_periods == 2
    assert inputs.post_periods == 2
    assert inputs.T == 4
    assert inputs.n_donors == 2


# -----------------------------
# inference.did_inference
# -----------------------------
def test_did_inference_basic():
    se, ci, pval, satt = did_inference(2.0, np.array([0.1, -0.1, 0.2]), 3, 2)
    assert se > 0
    assert ci[0] < 2.0 < ci[1]
    assert 0.0 <= pval <= 1.0
    assert np.isfinite(satt)


def test_did_inference_degenerate():
    se, ci, pval, satt = did_inference(2.0, np.array([0.0, 0.0]), 0, 0)
    assert np.isnan(se)
    assert np.isnan(ci[0]) and np.isnan(ci[1])


# -----------------------------
# estimation._r2_batch
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
# estimation.did_from_mean
# -----------------------------
def test_did_from_mean_basic():
    treated = np.array([1, 2, 3, 4])
    control = np.array([0.5, 0.5, 2, 2])
    res = did_from_mean(treated, control, 2)
    assert "Effects" in res and "Fit" in res and "Vectors" in res
    assert "Observed" in res["Vectors"]
    assert np.allclose(
        res["Vectors"]["Counterfactual"], np.round(res["Vectors"]["Counterfactual"], 3)
    )


def test_did_from_mean_single_period_post():
    treated = np.array([1, 2])
    control = np.array([0.5, 0.5])
    res = did_from_mean(treated, control, 1)
    assert not np.isnan(res["Effects"]["ATT"])


def test_did_from_mean_all_zero_control():
    treated = np.array([1, 2, 3])
    control = np.zeros(3)
    res = did_from_mean(treated, control, 2)
    assert np.all(np.isfinite([res["Effects"]["ATT"], res["Fit"]["R-Squared"]]))


# -----------------------------
# estimation forward-selection helpers
# -----------------------------
def test_select_best_donor_single_remaining():
    X_pre = np.array([[1, 2], [3, 4]])
    current_mean = np.array([1, 3])
    y_c = np.array([0.5, -0.5])
    ss_tot = np.sum(y_c ** 2)
    idx, r2, r2_all = _select_best_donor(X_pre, current_mean, 1, np.array([1]), y_c, ss_tot)
    assert idx == 1
    assert r2 == r2_all[0]


def test_update_synthetic_control_basic():
    current_mean = np.array([1.0, 2.0])
    control_outcomes = np.array([[2.0, 3.0], [4.0, 5.0]])
    updated = _update_synthetic_control(current_mean, control_outcomes, 0, 1)
    assert updated.shape == current_mean.shape


def test_record_verbose_step_appends():
    intermediary = []
    donor_names = ["A", "B"]
    _record_verbose_step(intermediary, 0, 0, 0.9, np.array([0.5, 0.6]), [0], donor_names, np.array([1, 2]), 1)
    assert len(intermediary) == 1
    assert intermediary[0]["iteration"] == 1
    assert "selected_name" in intermediary[0]


def test_choose_optimal_subset_basic():
    selected = [0, 1, 2]
    R2_path = [0.1, 0.5, 0.3]
    idxs, path = _choose_optimal_subset(selected, R2_path)
    assert idxs == [0, 1]
    assert path == R2_path[: len(idxs)]


def test_choose_optimal_subset_empty():
    idxs, path = _choose_optimal_subset([], [])
    assert idxs == []
    assert path == []


def test_compute_fdid_result_basic():
    treated = np.array([1, 2, 3, 4])
    controls = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    res = _compute_fdid_result(treated, controls, [0, 1], 2, [0.5, 0.6], ["C1", "C2"])
    assert "ATT" in res["Effects"]
    assert res["Vectors"]["Counterfactual"].shape == treated.shape


# -----------------------------
# estimation.forward_did_select
# -----------------------------
def test_forward_did_select_basic():
    treated = np.array([1, 2, 3, 4])
    controls = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]])
    T0 = 2
    donor_names = ["C1", "C2", "C3"]

    result = forward_did_select(treated, controls, T0, donor_names, verbose=True)
    assert "DID" in result and "FDID" in result

    treated_pre, treated_post = treated[:T0], treated[T0:]
    ctrl_pre = controls[:T0].mean(axis=1)
    ctrl_post = controls[T0:].mean(axis=1)
    expected_att = (treated_post.mean() - treated_pre.mean()) - (ctrl_post.mean() - ctrl_pre.mean())
    assert np.isclose(result["DID"]["Effects"]["ATT"], round(expected_att, 4))

    fdid_res = result["FDID"]
    optimal_idxs = fdid_res["selected_controls"]
    assert all(0 <= idx < controls.shape[1] for idx in optimal_idxs)
    assert fdid_res["selected_names"] == [donor_names[i] for i in optimal_idxs]
    counterfactual = fdid_res["Vectors"]["Counterfactual"]
    assert counterfactual.shape == treated.shape
    assert fdid_res["Vectors"]["Gap"].shape == (treated.shape[0], 2)


def test_forward_did_select_name_length_mismatch():
    treated = np.array([1, 2, 3, 4])
    controls = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    with pytest.raises(ValueError, match="donor_names length"):
        forward_did_select(treated, controls, 2, ["only_one"], verbose=False)


# -----------------------------
# results_assembly.assemble_fdid_results
# -----------------------------
def test_assemble_fdid_results(sample_fdid_data):
    inputs = prepare_fdid_inputs(
        sample_fdid_data, outcome="y", treat="treated_indicator",
        unitid="unit", time="time",
    )
    selector_output = forward_did_select(
        inputs.y, inputs.donor_matrix, inputs.pre_periods,
        donor_names=list(inputs.donor_names), verbose=True,
    )
    results = assemble_fdid_results(selector_output, inputs)
    assert isinstance(results, FDIDResults)
    assert results.fdid.name == "FDID"
    assert results.did.name == "DID"
    # Equal weights over the selected donors sum to 1.
    assert np.isclose(sum(results.fdid.donor_weights.values()), 1.0)
