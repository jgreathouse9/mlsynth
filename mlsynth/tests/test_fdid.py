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
    _record_verbose_step,
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


# -----------------------------------------------------------------------------
# forward_did_select: equivalence to Li's definitional algorithm
#
# Independent brute-force oracle transcribed cell-for-cell from the author's
# Fun_FDID.R (recompute rowMeans over the growing selected set for every
# candidate at every step; which.max = first-tie argmax; full N-step path; the
# optimal donor count is which.max of the R^2 path). Any optimized rewrite of
# forward_did_select must reproduce this exactly.
# -----------------------------------------------------------------------------
def _naive_forward_did(treated, X, T0):
    """O(N^3) reference forward-selected DID (mirrors Fun_FDID.R)."""
    T, N = X.shape
    yp = treated[:T0]
    Xp = X[:T0]
    ss_tot = np.sum((yp - yp.mean()) ** 2)

    def r2_of(cols):
        mean_pre = Xp[:, cols].mean(axis=1)
        beta = (yp - mean_pre).mean()          # DID intercept
        yhat = beta + mean_pre
        return 1.0 - np.sum((yp - yhat) ** 2) / ss_tot

    r2s = np.array([r2_of([j]) for j in range(N)])
    select = [int(np.argmax(r2s))]
    R2_path = [float(r2s[select[0]])]
    for _ in range(1, N):
        left = [j for j in range(N) if j not in select]
        cand_r2 = np.array([r2_of(select + [j]) for j in left])
        best = left[int(np.argmax(cand_r2))]   # first-tie argmax
        select.append(best)
        R2_path.append(float(cand_r2.max()))
    num_c = int(np.argmax(R2_path))             # optimal donor count
    opt = select[: num_c + 1]

    mean_full = X[:, opt].mean(axis=1)
    tp, tpost = treated[:T0], treated[T0:]
    cp, cpost = mean_full[:T0], mean_full[T0:]
    att = (tpost.mean() - tp.mean()) - (cpost.mean() - cp.mean())
    return opt, np.array(R2_path[: num_c + 1]), float(att)


@pytest.mark.parametrize("seed,N,T0,T1", [
    (0, 6, 8, 3), (1, 20, 10, 4), (2, 50, 12, 5), (3, 15, 6, 2), (4, 80, 14, 3),
])
def test_forward_did_matches_naive_reference(seed, N, T0, T1):
    """Selections, R^2 path, and ATT match Li's definitional algorithm exactly."""
    rng = np.random.default_rng(seed)
    T = T0 + T1
    controls = rng.standard_normal((T, N)) * 2.0 + rng.standard_normal(N)
    treated = controls[:, :3].mean(axis=1) + 0.3 * rng.standard_normal(T) + 1.0
    names = [f"c{j}" for j in range(N)]

    got = forward_did_select(treated, controls, T0, names)["FDID"]
    opt, r2_path, att = _naive_forward_did(treated, controls, T0)

    assert got["selected_controls"] == opt                       # exact order
    assert np.allclose(got["R2_at_each_step"], r2_path, atol=1e-9)
    assert np.isclose(got["Effects"]["ATT"], round(att, 4), atol=1e-4)


def test_forward_did_zero_variance_donor_matches_naive():
    """A constant (zero-variance) donor must not crash and must match the oracle."""
    rng = np.random.default_rng(7)
    T0, T = 8, 11
    controls = rng.standard_normal((T, 10)) * 1.5
    controls[:, 4] = 3.0                                          # constant donor
    treated = controls[:, :2].mean(axis=1) + 0.2 * rng.standard_normal(T)
    names = [f"c{j}" for j in range(10)]
    got = forward_did_select(treated, controls, T0, names)["FDID"]
    opt, r2_path, att = _naive_forward_did(treated, controls, T0)
    assert np.all(np.isfinite(got["R2_at_each_step"]))
    assert got["selected_controls"] == opt
    assert np.isclose(got["Effects"]["ATT"], round(att, 4), atol=1e-4)


def test_forward_did_duplicate_donor_estimate_invariant():
    """Exact-duplicate donors are a measure-zero tie: the *estimate* is invariant.

    When two donor columns are identical, adding the second leaves the donor
    average -- and therefore the R^2 and the ATT -- exactly unchanged, so which
    of the tied prefixes the R^2-argmax keeps is ambiguous at the level of
    floating-point summation order. The contract for this degenerate input is
    the estimate (ATT, peak R^2, counterfactual), not the exact selected set:
    the fit is identical whether or not the redundant duplicate is retained.
    """
    rng = np.random.default_rng(11)
    T0, T = 6, 9
    base = rng.standard_normal((T, 5))
    controls = np.column_stack([base, base[:, 1]])               # col 5 duplicates col 1
    treated = base[:, 1] + 0.1 * rng.standard_normal(T)
    names = [f"c{j}" for j in range(controls.shape[1])]
    got = forward_did_select(treated, controls, T0, names)["FDID"]
    opt, r2_path, att = _naive_forward_did(treated, controls, T0)

    assert np.isclose(got["Effects"]["ATT"], round(att, 4), atol=1e-9)     # estimate
    assert np.isclose(max(got["R2_at_each_step"]), max(r2_path), atol=1e-9)
    assert set(opt).issubset(set(got["selected_controls"]))               # only dups added
    # the retained donors' average equals the oracle's (redundant duplicates)
    assert np.allclose(controls[:, got["selected_controls"]].mean(axis=1),
                       controls[:, opt].mean(axis=1))


def test_forward_did_single_donor():
    """N = 1 selects the lone donor."""
    treated = np.array([1.0, 2.0, 3.0, 5.0])
    controls = np.array([[1.0], [2.0], [3.0], [4.0]])
    got = forward_did_select(treated, controls, 3, ["c0"])["FDID"]
    assert got["selected_controls"] == [0]
    assert np.all(np.isfinite(got["R2_at_each_step"]))


def test_forward_did_constant_treated_pre_period():
    """Zero-variance treated pre-period hits the ss_tot guard without dividing by 0."""
    treated = np.array([5.0, 5.0, 5.0, 8.0])           # constant over the 3 pre-periods
    rng = np.random.default_rng(3)
    controls = rng.standard_normal((4, 4))
    got = forward_did_select(treated, controls, 3, [f"c{j}" for j in range(4)])["FDID"]
    assert np.all(np.isfinite(got["R2_at_each_step"]))
    assert np.isfinite(got["Effects"]["ATT"])


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
