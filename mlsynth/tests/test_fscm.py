"""Tests for the migrated Forward-Selected SCM (Cerulli 2024) estimator."""

import numpy as np
import pandas as pd
import pytest

from mlsynth import FSCM
from mlsynth.config_models import FSCMConfig
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.fscm_helpers import (
    FSCMResults,
    FSCMSelectionPath,
    prepare_fscm_inputs,
    run_fscm,
)


def _factor_panel(n_donors=12, T=30, T0=20, seed=0, effect=-5.0):
    """Synthetic factor-model panel with one treated unit (unit 0)."""
    rng = np.random.default_rng(seed)
    n_units = n_donors + 1
    K = 3
    factors = np.cumsum(rng.normal(size=(T, K)), axis=0)
    loadings = rng.uniform(0.2, 1.0, size=(n_units, K))
    base = factors @ loadings.T + rng.normal(scale=0.3, size=(T, n_units))

    rows = []
    states = ["treated"] + [f"d{j}" for j in range(n_donors)]
    for u, name in enumerate(states):
        for t in range(T):
            y = base[t, u]
            treated_flag = int(name == "treated" and t >= T0)
            if treated_flag:
                y += effect
            rows.append({
                "unit": name, "time": t, "y": y,
                "x1": base[t, u] * 0.5 + rng.normal(scale=0.1),
                "treated": treated_flag,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def panel():
    return _factor_panel()


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="treated", unitid="unit",
                time="time", display_graphs=False)
    base.update(kw)
    return base


def test_fit_returns_results(panel):
    res = FSCM(_cfg(panel)).fit()
    assert isinstance(res, FSCMResults)
    assert isinstance(res.selection_path, FSCMSelectionPath)
    assert res.counterfactual.shape == (panel["time"].nunique(),)
    assert res.gap.shape == res.counterfactual.shape


def test_weights_form_simplex(panel):
    res = FSCM(_cfg(panel)).fit()
    assert np.all(res.weights_vector >= -1e-9)
    assert res.weights_vector.sum() == pytest.approx(1.0, abs=1e-6)
    # Full donor-weight dict zeros out the unselected donors.
    nonzero = [k for k, v in res.donor_weights.items() if v > 0]
    assert set(nonzero).issubset(set(res.selected_donors))


def test_optimal_size_consistency(panel):
    res = FSCM(_cfg(panel)).fit()
    path = res.selection_path
    assert res.n_selected == path.optimal_size
    assert path.optimal_size == int(np.argmin(path.test_rmspe)) + 1
    assert res.selected_donors == path.order[: res.n_selected]
    assert len(path.train_rmspe) == len(path.test_rmspe) == len(path.sizes)


def test_recovers_effect_sign(panel):
    res = FSCM(_cfg(panel)).fit()
    # Effect was -5; estimate should be clearly negative and roughly on target.
    assert res.att < -2.0
    assert res.diagnostics["pre_rmse"] < 1.0


def test_selects_fewer_than_full_pool(panel):
    # Forward selection should not always keep every donor.
    res = FSCM(_cfg(panel)).fit()
    assert res.n_selected <= res.diagnostics["n_donors_available"]


def test_max_donors_cap(panel):
    res = FSCM(_cfg(panel, max_donors=3)).fit()
    assert len(res.selection_path.sizes) == 3
    assert res.n_selected <= 3


def test_covariates_path(panel):
    res = FSCM(_cfg(panel, covariates=["x1"])).fit()
    assert isinstance(res, FSCMResults)
    assert res.metadata["covariates"] == ["x1"]
    assert res.weights_vector.sum() == pytest.approx(1.0, abs=1e-6)


def test_match_periods_path(panel):
    res = FSCM(_cfg(panel, match_periods=[5, 10, 15])).fit()
    assert isinstance(res, FSCMResults)
    assert res.metadata["match_periods"] == [5, 10, 15]
    assert res.weights_vector.sum() == pytest.approx(1.0, abs=1e-6)


def test_match_periods_with_covariates(panel):
    res = FSCM(_cfg(panel, covariates=["x1"], match_periods=[5, 10, 15])).fit()
    assert res.metadata["covariates"] == ["x1"]
    assert res.metadata["match_periods"] == [5, 10, 15]


def test_match_period_post_treatment_raises(panel):
    # t=25 is post-treatment (T0=20); must be rejected.
    with pytest.raises(MlsynthDataError):
        FSCM(_cfg(panel, match_periods=[25])).fit()


def test_match_period_missing_raises(panel):
    with pytest.raises(MlsynthDataError):
        FSCM(_cfg(panel, match_periods=[9999])).fit()


def test_rolling_origin_metadata(panel):
    res = FSCM(_cfg(panel, cv_split=0.6)).fit()
    T0 = 20
    assert res.metadata["cv_method"] == "rolling_origin"
    assert res.metadata["matching_mode"] == "trajectory"
    # First origin is round(T0 * cv_split); origins run to the end of the pre-period.
    assert res.metadata["cv_origins"][0] == round(T0 * 0.6)
    assert res.metadata["cv_origins"][-1] == T0 - 1
    assert res.diagnostics["n_cv_origins"] == len(res.metadata["cv_origins"])


def test_covariate_windows_and_global_v(panel):
    res = FSCM(_cfg(panel, covariates=["x1"], covariate_windows={"x1": (2, 12)},
                    match_periods=[5, 10])).fit()
    assert res.metadata["matching_mode"] == "predictor"
    # Global V is reported over covariates + lagged-outcome predictors.
    assert set(res.metadata["V_weights"]) == {"x1", "y[5]", "y[10]"}
    assert res.weights_vector.sum() == pytest.approx(1.0, abs=1e-6)


def test_bad_covariate_window_raises(panel):
    # Window selecting no periods must be rejected.
    with pytest.raises(MlsynthDataError):
        FSCM(_cfg(panel, covariates=["x1"], covariate_windows={"x1": (9000, 9001)})).fit()


def test_no_forward_selection_trajectory(panel):
    res = FSCM(_cfg(panel, forward_selection=False)).fit()
    assert res.metadata["forward_selection"] is False
    assert res.selection_path is None
    assert res.metadata["solver"] == "simplex_lstsq"
    assert res.weights_vector.sum() == pytest.approx(1.0, abs=1e-6)
    # Reported donors are the weight-bearing ones from the full solve.
    assert all(w > 0 for w in res.weights_vector)


def test_no_forward_selection_predictor_mode(panel):
    res = FSCM(_cfg(panel, forward_selection=False, covariates=["x1"],
                    match_periods=[5, 10])).fit()
    assert res.metadata["forward_selection"] is False
    assert res.metadata["solver"] == "bilevel"
    assert res.selection_path is None
    assert "V_weights" in res.metadata
    assert res.weights_vector.sum() == pytest.approx(1.0, abs=1e-6)


def test_fs_true_false_both_run_with_and_without_covariates(panel):
    # The DoD: all four combinations produce a finite ATT and valid weights.
    for fs in (True, False):
        for cov in ({}, {"covariates": ["x1"]}):
            res = FSCM(_cfg(panel, forward_selection=fs, **cov)).fit()
            assert np.isfinite(res.att)
            assert res.weights_vector.sum() == pytest.approx(1.0, abs=1e-6)
            assert res.counterfactual.shape == (panel["time"].nunique(),)


def test_prepare_inputs_shapes(panel):
    inputs = prepare_fscm_inputs(panel, unitid="unit", time="time",
                                 outcome="y", treat="treated")
    assert inputs.y.shape == (30,)
    assert inputs.Y.shape == (30, 12)
    assert inputs.T0 == 20
    assert inputs.treated_label == "treated"
    assert not inputs.has_covariates


def test_run_fscm_directly(panel):
    inputs = prepare_fscm_inputs(panel, unitid="unit", time="time",
                                 outcome="y", treat="treated")
    res = run_fscm(inputs, max_donors=5)
    assert isinstance(res, FSCMResults)
    assert 1 <= res.n_selected <= 5


def test_no_treated_unit_raises(panel):
    df = panel.copy()
    df["treated"] = 0
    with pytest.raises(MlsynthDataError):
        FSCM(_cfg(df)).fit()


def test_too_few_pre_periods_raises():
    df = _factor_panel(n_donors=5, T=6, T0=2)
    with pytest.raises(MlsynthDataError):
        FSCM(_cfg(df)).fit()


def test_config_rejects_bad_cv_split(panel):
    with pytest.raises(Exception):
        FSCMConfig(df=panel, outcome="y", treat="treated", unitid="unit",
                   time="time", cv_split=1.5)
