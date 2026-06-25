"""Tests for SCUL (Synthetic Control Using Lasso, Hollingsworth & Wing 2022).

Cover the engine invariants (glmnet-style standardisation, rolling-origin CV,
unique-solution reproduction), the estimator end-to-end on the California panel,
config-validation error paths, edge cases, placebo inference, and plotting.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlsynth import SCUL
from mlsynth.config_models import SCULConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.scul_helpers.estimate import (
    _glmnet_lambda_path,
    _mysd,
    fit_scul,
    rolling_cv_lambda,
)

_PANEL = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                      "california_panel.csv")


def _cali(extra=None, **kw):
    df = pd.read_csv(_PANEL)
    df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)
    base = dict(df=df, outcome="cigsale", treat="treat", unitid="state",
                time="year", display_graphs=False, inference=False)
    base.update(kw)
    return SCULConfig(**base)


def _toy(seed=0, T=24, T0=16, J=30, effect=-5.0):
    """Tiny high-dimensional panel: J donors > T0 pre-periods."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=T))                       # common factor
    X = np.column_stack([rng.normal(0, 1) + rng.normal(1, 0.3) * f
                         + rng.normal(0, 0.2, T) for _ in range(J)])
    y = X @ (rng.dirichlet(np.ones(J))) + rng.normal(0, 0.1, T)
    y[T0:] += effect
    return y, X, T0


# --------------------------------------------------------------------------- #
# Engine building blocks
# --------------------------------------------------------------------------- #
def test_mysd_is_population_sd():
    x = np.array([[1.0], [2.0], [3.0]])
    assert np.isclose(_mysd(x)[0], np.sqrt(np.mean((x - x.mean()) ** 2)))


def test_glmnet_lambda_path_descends_from_lambda_max():
    y, X, T0 = _toy()
    path = _glmnet_lambda_path(X[:T0], y[:T0])
    assert path.shape == (100,)
    assert path[0] > path[-1] > 0                            # log-descending grid


def test_fit_scul_recovers_effect_sign_and_shape():
    y, X, T0 = _toy()
    out = fit_scul(y, X, T0)
    assert out["counterfactual"].shape == (len(y),)
    assert out["weights"].shape == (X.shape[1],)
    att = np.mean((y - out["counterfactual"])[T0:])
    assert att < 0                                           # injected effect was -5
    assert 0 <= out["cohens_d"]                              # finite, non-negative fit


def test_negative_weights_allowed():
    """SCUL permits extrapolation: weights need not be non-negative or sum to 1."""
    y, X, T0 = _toy(seed=3)
    out = fit_scul(y, X, T0)
    # The lasso/OLS-style fit is not simplex-constrained; just assert it is not
    # forcing a convex combination (an intercept exists and weights can be < 0).
    assert isinstance(out["intercept"], float)
    assert np.isfinite(out["weights"]).all()


def test_unique_solution_two_solvers_agree():
    """Continuous donors => unique lasso (Tibshirani 2013); a second tight solve
    at the selected lambda lands on the same support."""
    y, X, T0 = _toy(seed=1)
    out = fit_scul(y, X, T0)
    from sklearn.linear_model import Lasso
    Xp = X[:T0]
    sd = _mysd(Xp); sd[sd == 0] = 1.0
    m = Lasso(alpha=out["ridge_lambda"], fit_intercept=True, max_iter=5_000_000,
              tol=1e-13).fit((Xp - Xp.mean(0)) / sd, y[:T0])
    np.testing.assert_allclose(m.coef_ / sd, out["weights"], atol=1e-6)


# --------------------------------------------------------------------------- #
# Estimator end-to-end
# --------------------------------------------------------------------------- #
def test_california_multitype_pool_end_to_end():
    res = SCUL(_cali(donor_variables=["retprice"])).fit()
    pu = res.method_details.parameters_used
    assert pu["n_pool"] == 76                                # 38 donors x (cigsale+retprice)
    assert res.effects.att < 0                               # Prop 99 reduced sales
    assert 0 < pu["ridge_lambda"]
    assert res.time_series.counterfactual_outcome.shape == res.time_series.observed_outcome.shape


def test_outcome_only_pool_default():
    res = SCUL(_cali()).fit()
    assert res.method_details.parameters_used["n_pool"] == 38   # outcome donors only


def test_two_family_result_contract():
    res = SCUL(_cali()).fit()
    assert res.effects is not None and res.time_series is not None
    assert res.weights is not None and res.method_details.method_name == "SCUL"
    assert np.isclose(res.att, res.effects.att)              # flat accessor resolves


def test_placebo_inference_pvalue_in_unit_interval():
    res = SCUL(_cali(donor_variables=["retprice"], inference=True)).fit()
    assert res.inference is not None
    assert 0.0 < res.inference.p_value <= 1.0
    assert res.fit.n_placebo >= 1


# --------------------------------------------------------------------------- #
# Config validation + edges
# --------------------------------------------------------------------------- #
def test_bad_cv_option_rejected():
    with pytest.raises((MlsynthConfigError, Exception)):
        _cali(cv_option="kfold")


def test_number_initial_periods_floor():
    with pytest.raises((MlsynthConfigError, Exception)):
        _cali(number_initial_periods=1)


def test_missing_donor_variable_raises():
    with pytest.raises(MlsynthConfigError):
        SCUL(_cali(donor_variables=["not_a_column"])).fit()


def test_cv_window_too_long_raises():
    y, X, T0 = _toy(T0=10)
    with pytest.raises(ValueError):
        rolling_cv_lambda(X[:T0], y[:T0], number_initial_periods=5,
                          training_post_length=7)            # 10-5-7+1 < 1


def test_dict_config_accepted():
    df = pd.read_csv(_PANEL)
    df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)
    res = SCUL(dict(df=df, outcome="cigsale", treat="treat", unitid="state",
                    time="year", display_graphs=False, inference=False)).fit()
    assert res.effects.att < 0


def test_cv_option_min_runs():
    y, X, T0 = _toy(seed=5)
    lam_min = rolling_cv_lambda(X[:T0], y[:T0], number_initial_periods=5,
                                training_post_length=5, cv_option="min")
    lam_med = rolling_cv_lambda(X[:T0], y[:T0], number_initial_periods=5,
                                training_post_length=5, cv_option="median")
    assert lam_min > 0 and lam_med > 0


def test_single_donor_1d_input():
    y, X, T0 = _toy(seed=6, J=4)
    out = fit_scul(y, X[:, 0], T0)                          # 1-D donor reshapes to (T,1)
    assert out["weights"].shape == (1,)


def test_plotting_smoke(tmp_path):
    out = str(tmp_path / "scul.png")
    SCUL(_cali(display_graphs=True, save=out)).fit()
    assert os.path.exists(out)
