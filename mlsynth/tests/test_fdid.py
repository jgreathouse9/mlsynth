import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm

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
from mlsynth.utils.effectutils import standardized_att


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
# inference.did_inference -- the standardised ATT
#
# Four sources agree on what this quantity is, which is why these tests assert
# an equality and not a range.
#
#   1. Proposition 2.1: sqrt(T2) * (ATT_hat - ATT) / sqrt(omega_1 + omega_2)
#      is asymptotically standard normal.
#   2. Li's replication package (MKSC 2022.0212, FDID_Matlab.m line 45):
#          ATT_std_FDID = sqrt(t2) * ATT_FDID / std_Omega_hat_FDID
#      annotated "it is N(0,1) under H0, ATT=0", with std_Omega_hat_FDID =
#      sqrt(Omega_1 + Omega_2). Her CI on line 49 is ATT +/- 1.96 *
#      std_Omega_hat_FDID / sqrt(t2), so her standard error is the one this
#      module returns and her statistic is the estimate over it.
#   3. effectutils.standardized_att, which computes the same quantity for the
#      library at large.
#   4. This module's own p-value, which is the two-sided normal tail of
#      att / se and so already encodes the statistic.
#
# All four reduce to att / se. Source 4 is what test_satt_agrees_with_its_own
# _p_value rests on: it needs no reference at all, because a result that
# reports a statistic and a p-value which disagree is inconsistent with itself
# whatever the right scaling turns out to be.
# -----------------------------
_SATT_SHAPES = [(10, 5), (20, 10), (30, 12), (50, 25), (100, 40)]


def _mean_zero_residuals(n: int, seed: int) -> np.ndarray:
    """Pre-period residuals of a fitted DiD: mean zero by construction."""
    e = np.random.default_rng(seed).normal(0.0, 1.0, n)
    return e - e.mean()


@pytest.mark.parametrize("method", ["analytic", "hac"])
@pytest.mark.parametrize(("pre_periods", "post_periods"), _SATT_SHAPES)
def test_satt_agrees_with_its_own_p_value(method, pre_periods, post_periods):
    """The p-value is the two-sided normal tail of the statistic reported beside it."""
    residuals = _mean_zero_residuals(pre_periods, seed=pre_periods)
    _se, _ci, p_value, satt = did_inference(
        0.35, residuals, pre_periods, post_periods, method=method
    )
    assert p_value == pytest.approx(2.0 * (1.0 - norm.cdf(abs(satt))), abs=1e-12)


@pytest.mark.parametrize("method", ["analytic", "hac"])
@pytest.mark.parametrize(("pre_periods", "post_periods"), _SATT_SHAPES)
def test_satt_is_the_estimate_over_its_standard_error(method, pre_periods, post_periods):
    """SATT is att / se -- the scaling Li's own code and the CI both use."""
    residuals = _mean_zero_residuals(pre_periods, seed=pre_periods + 1)
    att = 2.5
    se, _ci, _p, satt = did_inference(
        att, residuals, pre_periods, post_periods, method=method
    )
    assert satt == pytest.approx(att / se, rel=1e-12)


@pytest.mark.parametrize(("pre_periods", "post_periods"), _SATT_SHAPES)
def test_satt_matches_proposition_2_1_written_out(pre_periods, post_periods):
    """sqrt(T2) * ATT / sqrt(omega_1 + omega_2), the paper's and the MATLAB's form."""
    residuals = _mean_zero_residuals(pre_periods, seed=pre_periods + 2)
    att = 2.5
    omega2 = float(np.mean(residuals ** 2))
    omega1 = (post_periods / pre_periods) * omega2
    expected = np.sqrt(post_periods) * att / np.sqrt(omega1 + omega2)

    _se, _ci, _p, satt = did_inference(att, residuals, pre_periods, post_periods)
    assert satt == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(("pre_periods", "post_periods"), _SATT_SHAPES)
def test_satt_matches_the_library_wide_primitive(pre_periods, post_periods):
    """effectutils.standardized_att computes this statistic for every estimator.

    FDID fits an intercept so the pre-period gap has mean zero and the post-period
    gap has mean att, which is the input that primitive expects.
    """
    residuals = _mean_zero_residuals(pre_periods, seed=pre_periods + 3)
    att = 2.5
    _se, _ci, _p, satt = did_inference(att, residuals, pre_periods, post_periods)
    assert satt == pytest.approx(
        standardized_att(residuals, np.full(post_periods, att)), rel=1e-10
    )


@pytest.mark.parametrize("method", ["analytic", "hac"])
def test_satt_is_invariant_to_the_outcome_scale(method):
    """Rescaling the outcome rescales att and the residuals together, so the
    statistic -- a ratio of two quantities in the outcome's units -- must not move."""
    residuals = _mean_zero_residuals(40, seed=7)
    _se, _ci, _p, satt = did_inference(2.5, residuals, 40, 16, method=method)
    _se_s, _ci_s, _p_s, satt_scaled = did_inference(
        2.5 * 1000.0, residuals * 1000.0, 40, 16, method=method
    )
    assert satt_scaled == pytest.approx(satt, rel=1e-10)


@pytest.fixture
def borderline_fdid_panel() -> pd.DataFrame:
    """A panel whose effect sits near the significance boundary.

    The four-period ``sample_fdid_data`` fixture cannot test this: it has two
    pre-periods and a near-exact fit, so the statistic is enormous and both
    p-values saturate at zero whatever the scaling. The assertion would pass
    against the defect. Here the ATT is 0.2898 against a standard error of
    0.2344, so the correct statistic is 1.236 and the two readings of it are
    0.216 and 0.000092 -- opposite conclusions about the same fit, which is
    what gives the assertion power.
    """
    rng = np.random.default_rng(42)
    pre_periods, post_periods, n_donors, noise, effect = 24, 10, 3, 0.7, 0.5
    T = pre_periods + post_periods
    common = 10.0 + 0.3 * np.arange(T, dtype=float)

    rows = []
    for j in range(n_donors):
        y = common + rng.normal(0.0, noise, T) + 0.5 * j
        rows += [{"unit": f"C{j}", "time": i + 1, "y": y[i], "d": 0}
                 for i in range(T)]
    y = common + 2.0 + rng.normal(0.0, noise, T)
    y[pre_periods:] += effect
    rows += [{"unit": "T", "time": i + 1, "y": y[i],
              "d": int(i >= pre_periods)} for i in range(T)]
    return pd.DataFrame(rows)


def test_satt_on_the_public_surface_is_att_over_its_standard_error(
    borderline_fdid_panel,
):
    """The reported SATT, att and att_se are one statement, not three."""
    fit = FDID(
        FDIDConfig(
            df=borderline_fdid_panel, outcome="y", treat="d",
            unitid="unit", time="time", display_graphs=False,
        )
    ).fit().fdid
    # Nothing is rounded on the way out any more, so the identity is exact and
    # the tolerance is floating point. The defect this pins scaled SATT by
    # sqrt(post_periods), a factor of 3.16 on this panel.
    assert fit.satt == pytest.approx(fit.att / fit.att_se, rel=1e-12)


def test_satt_reported_by_the_estimator_agrees_with_its_p_value(
    borderline_fdid_panel,
):
    """The same invariant on the public surface, not only inside the helper."""
    fit = FDID(
        FDIDConfig(
            df=borderline_fdid_panel, outcome="y", treat="d",
            unitid="unit", time="time", display_graphs=False,
        )
    ).fit().fdid
    assert fit.p_value == pytest.approx(
        2.0 * (1.0 - norm.cdf(abs(fit.satt))), abs=5e-3
    )


def test_satt_is_nan_when_the_panel_has_no_post_period():
    """The degenerate panel returns nan, not a scaled zero."""
    _se, _ci, _p, satt = did_inference(2.0, np.array([0.0, 0.0]), 0, 0)
    assert np.isnan(satt)


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
    assert np.isclose(result["DID"]["Effects"]["ATT"], expected_att, atol=1e-12)

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
    assert np.isclose(got["Effects"]["ATT"], att, atol=1e-12)


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
    assert np.isclose(got["Effects"]["ATT"], att, atol=1e-12)


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

    assert np.isclose(got["Effects"]["ATT"], att, atol=1e-9)               # estimate
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


# --------------------------------------------------------------------------- #
# FDID reports what it computed
#
# ``did_from_mean`` used to round its whole return dict on the way out -- the
# counterfactual and the observed series to 3 decimals, the ATT, R^2, RMSE,
# standard error, interval and intercept to 4 -- and ``results_assembly`` read
# the rounded vectors straight into the typed result, so the quantization
# reached ``FDIDMethodFit`` and the standardized ``time_series`` contract with it.
#
# On a proportion-scale outcome that is not cosmetic. With the outcome in
# [0.048, 0.235] and an effect of -0.000166, a 4-decimal quantum reported it as
# -0.0002: a fifth of the number. The rounding was also not the author's. Li's
# released ``Fun_FDID.R`` returns raw doubles and its readme prints
# ``ATT_FDID = 0.02540494``, to eight significant figures.
# --------------------------------------------------------------------------- #

def _proportion_panel():
    """A panel whose outcome is a small proportion and whose effect is tiny.

    This is the regime the quantization destroyed: a 4-decimal quantum is a
    fifth of an effect this size.
    """
    rng = np.random.default_rng(3)
    T, T0, N = 40, 30, 5
    common = 0.12 + 0.03 * rng.standard_normal(T)
    # Both noise scales have to be small against the effect. The DID ATT differences
    # the treated series against the donor average, so the common part cancels but
    # each side's idiosyncratic part does not: at a donor noise of 0.01 the ATT
    # carries a noise floor near 0.0016, ten times the planted effect, and the
    # quantum stops being a large share of it -- which would leave the power
    # assertion below with nothing to detect.
    donors = common[:, None] + 1e-6 * rng.standard_normal((T, N))
    treated = common + 0.004 + 1e-6 * rng.standard_normal(T)
    treated[T0:] += 1.7e-4                       # the effect, below the 4dp quantum
    Y = np.column_stack([treated, donors])
    units = np.repeat(np.arange(N + 1), T)
    times = np.tile(np.arange(T), N + 1)
    return pd.DataFrame({
        "unit": units, "time": times, "y": Y.T.ravel(),
        "d": ((units == 0) & (times >= T0)).astype(int),
    }), Y, T0


def _exact_did(Y, T0):
    """Li's closed form: ``beta = mean(y1 - x1)``, ``yhat = beta + x``."""
    y, X = Y[:, 0], Y[:, 1:]
    xb = X.mean(axis=1)
    beta = float((y[:T0] - xb[:T0]).mean())
    cf = beta + xb
    att = float((y[T0:] - cf[T0:]).mean())
    r2 = 1.0 - float(((y[:T0] - cf[:T0]) ** 2).mean()) / float(
        ((y[:T0] - y[:T0].mean()) ** 2).mean())
    rmse = float(np.sqrt(((y[:T0] - cf[:T0]) ** 2).mean()))
    return cf, att, r2, rmse


def _fit_proportion_panel():
    df, Y, T0 = _proportion_panel()
    res = FDID(FDIDConfig(df=df, outcome="y", treat="d", unitid="unit",
                          time="time", display_graphs=False)).fit()
    return res, Y, T0


class TestFDIDDoesNotQuantizeItsOutput:
    def test_did_counterfactual_matches_the_closed_form(self):
        res, Y, T0 = _fit_proportion_panel()
        cf, _, _, _ = _exact_did(Y, T0)
        got = np.asarray(res.did.counterfactual, dtype=float)
        assert np.max(np.abs(got - cf)) < 1e-12

    def test_did_counterfactual_is_not_a_grid_of_3dp_multiples(self):
        """The signature of the defect: every value an exact 3-decimal multiple."""
        res, _, _ = _fit_proportion_panel()
        got = np.asarray(res.did.counterfactual, dtype=float)
        assert not np.allclose(got, np.round(got, 3), atol=1e-12)

    def test_a_tiny_effect_is_not_snapped_to_the_quantum(self):
        res, Y, T0 = _fit_proportion_panel()
        _, att, _, _ = _exact_did(Y, T0)
        assert abs(res.did.att - att) < 1e-12
        # and the loss the quantum would have caused is real, so the test has power
        assert abs(round(att, 4) - att) / abs(att) > 0.05

    def test_r_squared_and_rmse_keep_their_digits(self):
        res, Y, T0 = _fit_proportion_panel()
        _, _, r2, rmse = _exact_did(Y, T0)
        assert abs(res.did.r_squared - r2) < 1e-12
        assert abs(res.did.pre_rmse - rmse) < 1e-12

    def test_gap_is_the_observed_series_minus_the_counterfactual(self):
        res, _, _ = _fit_proportion_panel()
        obs = np.asarray(res.time_series.observed_outcome, dtype=float)
        cf = np.asarray(res.time_series.counterfactual_outcome, dtype=float)
        np.testing.assert_allclose(
            np.asarray(res.time_series.estimated_gap, dtype=float), obs - cf,
            atol=1e-15)

    def test_the_standardized_contract_carries_full_precision(self):
        res, Y, T0 = _fit_proportion_panel()
        cf, _, _, _ = _exact_did(Y, T0)
        ts = np.asarray(res.time_series.counterfactual_outcome, dtype=float)
        # FDID is the reported variant, so time_series is the forward fit; what
        # is asserted is that it is not quantized, whichever variant it is.
        assert not np.allclose(ts, np.round(ts, 3), atol=1e-12)

    def test_fdid_variant_is_unquantized_too(self):
        res, _, _ = _fit_proportion_panel()
        got = np.asarray(res.fdid.counterfactual, dtype=float)
        assert not np.allclose(got, np.round(got, 3), atol=1e-12)
        assert abs(res.fdid.att - round(res.fdid.att, 4)) > 0.0

    def test_did_from_mean_returns_raw_vectors(self):
        treated = np.array([0.1234567, 0.2345678, 0.3456789, 0.4567891])
        control = np.array([0.0512345, 0.0523456, 0.2034567, 0.2045678])
        res = did_from_mean(treated, control, 2)
        cf = np.asarray(res["Vectors"]["Counterfactual"], dtype=float)
        assert not np.allclose(cf, np.round(cf, 3), atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(res["Vectors"]["Observed"], dtype=float), treated,
            atol=0.0)

    def test_satt_is_exactly_att_over_att_se(self):
        """With nothing rounded the identity is exact, not approximate.

        The comment this replaces allowed a percent of error because SATT was
        rounded to 3 decimals and the pair to 4.
        """
        res, _, _ = _fit_proportion_panel()
        f = res.fdid
        assert f.satt == pytest.approx(f.att / f.att_se, rel=1e-12)


# --------------------------------------------------------------------------- #
# ADID: Li and Van den Bulte (2022), the slope-adjusted variant
# --------------------------------------------------------------------------- #

def _adid_panel(T=60, T0=40, N=6, slope=2.5, intercept=7.0, tau=0.0, seed=3):
    """A panel where the treated unit is a known affine map of the donor mean.

    DID cannot fit this unless ``slope`` is 1, which is what makes the design
    separate the two methods instead of letting them coincide.
    """
    rng = np.random.default_rng(seed)
    donors = 10.0 + np.cumsum(rng.standard_normal((T, N)), axis=0)
    treated = intercept + slope * donors.mean(axis=1) + 0.01 * rng.standard_normal(T)
    treated[T0:] += tau
    Y = np.column_stack([treated, donors])
    units = np.repeat(np.arange(N + 1), T)
    times = np.tile(np.arange(T), N + 1)
    df = pd.DataFrame({
        "unit": units, "time": times, "y": Y.T.ravel(),
        "d": ((units == 0) & (times >= T0)).astype(int),
    })
    return df, treated, donors, T0


class TestADIDEstimation:
    def test_counterfactual_is_the_fitted_affine_map(self):
        from mlsynth.utils.fdid_helpers import adid_from_mean

        _, y, donors, T0 = _adid_panel()
        mean_ctrl = donors.mean(axis=1)
        raw = adid_from_mean(y, mean_ctrl, T0)
        d1 = raw["Inference"]["Intercept"]
        d2 = raw["Inference"]["Slope"]
        np.testing.assert_allclose(
            raw["Vectors"]["Counterfactual"], d1 + d2 * mean_ctrl, atol=1e-12)

    def test_it_recovers_a_planted_slope_and_intercept(self):
        """Equation 2.4 is a projection, so on data generated from it the
        coefficients come back."""
        from mlsynth.utils.fdid_helpers import adid_from_mean

        _, y, donors, T0 = _adid_panel(slope=2.5, intercept=7.0, tau=0.0)
        raw = adid_from_mean(y, donors.mean(axis=1), T0)
        assert abs(raw["Inference"]["Slope"] - 2.5) < 5e-3   # about 1.6 standard errors
        assert abs(raw["Inference"]["Intercept"] - 7.0) < 5e-2
        assert abs(raw["Effects"]["ATT"]) < 5e-2

    def test_it_recovers_a_planted_effect_where_did_cannot(self):
        from mlsynth.utils.fdid_helpers import adid_from_mean, did_from_mean

        _, y, donors, T0 = _adid_panel(slope=2.5, tau=4.0)
        mean_ctrl = donors.mean(axis=1)
        adid = adid_from_mean(y, mean_ctrl, T0)["Effects"]["ATT"]
        did = did_from_mean(y, mean_ctrl, T0)["Effects"]["ATT"]
        assert abs(adid - 4.0) < 0.1
        assert abs(did - 4.0) > 1.0          # the slope-one restriction bites

    def test_it_is_did_when_the_slope_is_one(self):
        """The two estimators are nested, so on slope-one data they agree."""
        from mlsynth.utils.fdid_helpers import adid_from_mean, did_from_mean

        _, y, donors, T0 = _adid_panel(slope=1.0, intercept=3.0, tau=2.0)
        mean_ctrl = donors.mean(axis=1)
        adid = adid_from_mean(y, mean_ctrl, T0)
        did = did_from_mean(y, mean_ctrl, T0)
        assert abs(adid["Inference"]["Slope"] - 1.0) < 1e-2
        assert abs(adid["Effects"]["ATT"] - did["Effects"]["ATT"]) < 0.05

    def test_it_fits_at_least_as_well_in_sample_as_did(self):
        """OLS on a superset of DID's restriction cannot fit worse."""
        from mlsynth.utils.fdid_helpers import adid_from_mean, did_from_mean

        for slope in (1.0, 2.5):
            _, y, donors, T0 = _adid_panel(slope=slope)
            m = donors.mean(axis=1)
            assert (adid_from_mean(y, m, T0)["Fit"]["T0 RMSE"]
                    <= did_from_mean(y, m, T0)["Fit"]["T0 RMSE"] + 1e-9)

    def test_the_variance_is_appendix_a1(self):
        """Sigma = (T2/T1) sigma^2 eta' Psi^-1 eta + sigma^2, their .m lines 57-66."""
        from mlsynth.utils.fdid_helpers import adid_from_mean

        _, y, donors, T0 = _adid_panel(T=60, T0=40)
        m = donors.mean(axis=1)
        T, T2 = len(y), len(y) - T0
        raw = adid_from_mean(y, m, T0)

        X = np.column_stack([np.ones(T), m])
        delta = np.linalg.solve(X[:T0].T @ X[:T0], X[:T0].T @ y[:T0])
        resid = y[:T0] - X[:T0] @ delta
        sigma2 = float(np.mean(resid ** 2))
        eta = X[T0:].mean(axis=0)
        psi = X[:T0].T @ X[:T0] / T0
        omega = (T2 / T0) * sigma2 * eta @ np.linalg.solve(psi, eta) + sigma2
        assert abs(raw["Inference"]["SE"] - np.sqrt(omega / T2)) < 1e-12
        assert abs(raw["Effects"]["SATT"]
                   - np.sqrt(T2) * raw["Effects"]["ATT"] / np.sqrt(omega)) < 1e-10

    def test_hac_moves_the_standard_error_on_a_correlated_residual(self):
        from mlsynth.utils.fdid_helpers import adid_from_mean

        rng = np.random.default_rng(5)
        T, T0, N = 80, 60, 5
        donors = 10.0 + np.cumsum(rng.standard_normal((T, N)), axis=0)
        e = np.zeros(T)
        for t in range(1, T):                       # AR(1), heavily persistent
            e[t] = 0.85 * e[t - 1] + rng.standard_normal()
        y = 2.0 + 1.5 * donors.mean(axis=1) + e
        m = donors.mean(axis=1)
        a = adid_from_mean(y, m, T0, inference="analytic")["Inference"]["SE"]
        h = adid_from_mean(y, m, T0, inference="hac")["Inference"]["SE"]
        assert h != a and np.isfinite(h) and h > 0

    def test_hac_at_lag_zero_is_their_heteroskedasticity_robust_branch(self):
        """Appendix A.1's third form, which is not the analytic one.

        The paper gives ``V = T1^-1 sum e^2 x x'`` for uncorrelated but
        heteroskedastic errors, where the authors' script uses ``sigma^2 Psi``.
        Both are theirs, they coincide only under homoskedasticity, and at lag
        zero the HAC double sum collapses to the first.
        """
        from mlsynth.utils.fdid_helpers import adid_from_mean

        _, y, donors, T0 = _adid_panel()
        m = donors.mean(axis=1)
        T, T2 = len(y), len(y) - T0
        X = np.column_stack([np.ones(T), m])
        delta = np.linalg.solve(X[:T0].T @ X[:T0], X[:T0].T @ y[:T0])
        e = y[:T0] - X[:T0] @ delta
        psi = X[:T0].T @ X[:T0] / T0
        b = np.linalg.solve(psi, X[T0:].mean(axis=0))
        V = (X[:T0] * (e ** 2)[:, None]).T @ X[:T0] / T0
        omega = (T2 / T0) * float(b @ V @ b) + float(np.mean(e ** 2))

        h = adid_from_mean(y, m, T0, inference="hac", lrvar_lag=0)
        assert abs(h["Inference"]["SE"] - np.sqrt(omega / T2)) < 1e-12

        a = adid_from_mean(y, m, T0, inference="analytic")["Inference"]["SE"]
        assert h["Inference"]["SE"] != a

    def test_a_constant_donor_average_is_refused(self):
        """The design is then rank one and the slope is not identified."""
        from mlsynth.utils.fdid_helpers import adid_from_mean

        y = np.arange(30.0)
        with pytest.raises(MlsynthEstimationError, match="[Ss]lope"):
            adid_from_mean(y, np.full(30, 4.0), 20)

    def test_too_few_pre_periods_is_refused(self):
        from mlsynth.utils.fdid_helpers import adid_from_mean

        y = np.arange(10.0)
        with pytest.raises(MlsynthEstimationError, match="pre-treatment"):
            adid_from_mean(y, np.arange(10.0) * 0.5, 2)


class TestADIDInferenceGuards:
    """The three refusals of ``adid_inference``, none of which the estimator
    can reach: ``adid_from_mean`` fixes the method and the lag and refuses a
    short pre-period before calling it."""

    @staticmethod
    def _args():
        rng = np.random.default_rng(1)
        X = np.column_stack([np.ones(20), rng.standard_normal(20)])
        return dict(att=1.0, pre_residuals=rng.standard_normal(20),
                    pre_design=X, post_design_mean=X.mean(axis=0),
                    pre_periods=20, post_periods=5)

    def test_an_unknown_method(self):
        from mlsynth.utils.fdid_helpers import adid_inference

        with pytest.raises(ValueError, match="method must be one of"):
            adid_inference(**self._args(), method="bootstrap")

    def test_a_negative_lag(self):
        from mlsynth.utils.fdid_helpers import adid_inference

        with pytest.raises(ValueError, match="non-negative"):
            adid_inference(**self._args(), method="hac", lrvar_lag=-1)

    def test_an_empty_window_returns_nan_instead_of_raising(self):
        """A caller can hand it a panel with nothing after the treatment, and
        an undefined interval is the answer, not an exception."""
        from mlsynth.utils.fdid_helpers import adid_inference

        args = self._args()
        args["post_periods"] = 0
        se, ci, p, satt = adid_inference(**args)
        assert all(np.isnan(v) for v in (se, p, satt)) and all(np.isnan(c) for c in ci)


class TestADIDOnTheResult:
    @staticmethod
    def _fit(**kw):
        df, _, _, _ = _adid_panel()
        cfg = dict(df=df, outcome="y", treat="d", unitid="unit", time="time",
                   display_graphs=False)
        cfg.update(kw)
        return FDID(FDIDConfig(**cfg)).fit()

    def test_adid_is_a_third_fit(self):
        res = self._fit()
        assert isinstance(res.adid, FDIDMethodFit)
        assert res.adid.name == "ADID"
        assert set(res.methods) == {"FDID", "DID", "ADID"}
        assert set(res.att_by_method()) == {"FDID", "DID", "ADID"}

    def test_only_adid_reports_a_slope(self):
        """FDID and DID hold it at one by construction, so they do not fit one."""
        res = self._fit()
        assert res.adid.slope is not None
        assert res.fdid.slope is None and res.did.slope is None

    def test_adid_uses_every_donor(self):
        """Their Equation 2.4 averages the whole control group, and the forward
        selection is a different paper."""
        res = self._fit()
        assert len(res.adid.selected_names) == res.inputs.n_donors
        assert res.adid.donor_weights == res.did.donor_weights

    def test_selecting_adid_routes_the_aliases(self):
        res = self._fit()
        object.__setattr__(res, "selected_variant", "ADID")
        assert res.att == res.adid.att
        np.testing.assert_allclose(res.counterfactual, res.adid.counterfactual)

    def test_the_gap_is_observed_minus_the_adid_counterfactual(self):
        res = self._fit()
        np.testing.assert_allclose(
            res.adid.gap,
            np.asarray(res.inputs.y, dtype=float) - res.adid.counterfactual,
            atol=1e-12)
