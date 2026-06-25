"""Unit tests for the proximal_helpers estimation and inference functions."""

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.proximal_helpers import (
    bartlett,
    hac,
    estimate_pi,
    estimate_pi_surrogate,
    estimate_pi_surrogate_post,
    estimate_spsc,
)


# --- bartlett / hac ---

def test_bartlett_weights():
    assert bartlett(0, 5) == 1.0
    assert bartlett(3, 5) == pytest.approx(0.5)
    assert bartlett(6, 5) == 0.0


def test_hac_shape_and_symmetry():
    rng = np.random.default_rng(0)
    U = rng.standard_normal((50, 3))
    omega = hac(U, 4)
    assert omega.shape == (3, 3)
    np.testing.assert_allclose(omega, omega.T, atol=1e-12)


def test_hac_zero_lag_is_gram_over_n():
    rng = np.random.default_rng(1)
    U = rng.standard_normal((40, 2))
    np.testing.assert_allclose(hac(U, 0), (U.T @ U) / U.shape[0])


# --- estimate_pi ---

def _pi_inputs(seed=0, T=30, T0=20, n=2):
    rng = np.random.default_rng(seed)
    W_pre = rng.random((T0, n))
    Z0_pre = W_pre @ rng.random((n, n)) + 0.01 * rng.random((T0, n))  # non-singular Z0'W
    W = np.vstack((W_pre, rng.random((T - T0, n))))
    Z0 = np.vstack((Z0_pre, rng.random((T - T0, n))))
    Y = rng.random(T)
    return Y, W, Z0, T0, T


def test_estimate_pi_smoke():
    Y, W, Z0, T0, T = _pi_inputs()
    cf, alpha, se = estimate_pi(Y, W, Z0, T0, T - T0, T, 3)
    assert cf.shape == (T,) and np.all(np.isfinite(cf))
    assert alpha.shape == (W.shape[1],) and np.all(np.isfinite(alpha))
    assert isinstance(se, float)


def test_estimate_pi_with_covariates():
    Y, W, Z0, T0, T = _pi_inputs(seed=2, T=30, T0=12)
    Cw = np.random.default_rng(3).random((T, 1))
    Cy = np.random.default_rng(4).random((T, 1))
    cf, alpha, se = estimate_pi(
        Y, W, Z0, T0, T - T0, T, 3,
        common_aux_covariates_1=Cw, common_aux_covariates_2=Cy,
    )
    # alpha is returned for the original donors only.
    assert alpha.shape == (W.shape[1],)


def test_estimate_pi_dimension_mismatch():
    Y, W, Z0, T0, T = _pi_inputs()
    with pytest.raises(MlsynthConfigError):
        estimate_pi(
            Y, W, Z0[:, :1], T0, T - T0, T, 3,
            common_aux_covariates_1=np.ones((T, 1)),
            common_aux_covariates_2=np.ones((T, 2)),
        )


# --- estimate_pi_surrogate / _post ---

def _surrogate_inputs(seed=5, T=30, T0=20, n=2, k=2):
    rng = np.random.default_rng(seed)
    W_pre = rng.random((T0, n))
    Z0_pre = W_pre @ rng.random((n, n)) + 0.01 * rng.random((T0, n))
    W = np.vstack((W_pre, rng.random((T - T0, n))))
    Z0 = np.vstack((Z0_pre, rng.random((T - T0, n))))
    X_post = rng.random((T - T0, k))
    Z1_post = X_post @ rng.random((k, k)) + 0.01 * rng.random((T - T0, k))
    X = np.vstack((rng.random((T0, k)), X_post))
    Z1 = np.vstack((rng.random((T0, k)), Z1_post))
    Y = rng.random(T)
    return Y, W, Z0, Z1, X, T0, T


def test_estimate_pi_surrogate_smoke():
    Y, W, Z0, Z1, X, T0, T = _surrogate_inputs()
    tau, taut, alpha, se = estimate_pi_surrogate(Y, W, Z0, Z1, X, T0, T - T0, T, 3)
    assert np.isfinite(tau)
    assert taut.shape == (T,)
    assert alpha.shape == (W.shape[1],)
    assert isinstance(se, float)


def test_estimate_pi_surrogate_post_smoke():
    Y, W, Z0, Z1, X, T0, T = _surrogate_inputs()
    tau, taut, params_W, se = estimate_pi_surrogate_post(Y, W, Z0, Z1, X, T0, T - T0, 3)
    assert np.isfinite(tau)
    assert taut.shape == (T,)
    assert params_W.shape == (W.shape[1],)
    assert isinstance(se, float)


def test_estimate_pi_surrogate_dimension_mismatch():
    Y, W, Z0, Z1, X, T0, T = _surrogate_inputs()
    with pytest.raises(MlsynthConfigError):
        estimate_pi_surrogate(Y, W, Z0[:, :1], Z1, X, T0, T - T0, T, 3)


# --- SPSC (single proxy synthetic control) ---

def _spsc_panel(seed=0, T=120, T0=80, N=6):
    """One factor-model panel: donors proxy the treated potential outcome."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=T))                 # common factor
    loads = rng.uniform(0.5, 1.5, size=N)
    W = np.outer(f, loads) + rng.normal(scale=0.3, size=(T, N))
    y = f * loads.mean() + rng.normal(scale=0.3, size=T)
    y[T0:] += 2.0                                      # +2 effect post
    return y, W, T0


def test_estimate_spsc_smoke_nodt():
    y, W, T0 = _spsc_panel()
    cf, gamma, att, se, trend, lam, _path, _pse = estimate_spsc(y, W, T0, detrend=False)
    assert cf.shape == (len(y),)
    assert gamma.shape == (W.shape[1],)
    assert np.isfinite(att)
    assert isinstance(se, float)
    assert np.allclose(trend, 0.0)


def test_estimate_spsc_smoke_dt():
    y, W, T0 = _spsc_panel(seed=1)
    cf, gamma, att, se, trend, lam, _path, _pse = estimate_spsc(y, W, T0, detrend=True, spline_df=5)
    assert cf.shape == (len(y),)
    assert np.isfinite(att)
    assert trend.shape == (len(y),)


def test_estimate_spsc_deterministic():
    """The CV ridge solve must be deterministic across repeated calls."""
    y, W, T0 = _spsc_panel(seed=2)
    a1 = estimate_spsc(y, W, T0, detrend=True)
    a2 = estimate_spsc(y, W, T0, detrend=True)
    assert a1[2] == a2[2] and a1[3] == a2[3] and a1[5] == a2[5]


def test_estimate_spsc_fixed_lambda():
    y, W, T0 = _spsc_panel(seed=3)
    cf, gamma, att, se, trend, lam, _path, _pse = estimate_spsc(y, W, T0, detrend=False, ridge_lambda=-1.0)
    assert lam == -1.0


# --- DR + PIPW (doubly robust proximal; Qiu et al. 2024) ---

from mlsynth.utils.proximal_helpers import estimate_dr, estimate_pipw


def _dr_panel(seed=0, T=600, nU=3, effect=3.0):
    """The DR paper's DGP: AR(1) latent factors with an exponential shift at T0."""
    rng = np.random.default_rng(seed)
    T0 = T // 2
    from scipy.stats import norm
    U = np.empty((T, nU)); U[0] = rng.normal(size=nU)
    for t in range(1, T):
        U[t] = 0.1 * U[t - 1] + 0.9 * rng.normal(size=nU)
    U = norm.cdf(U)
    U[:T0] = -np.log1p(-U[:T0]) / 1.0
    U[T0:] = -np.log1p(-U[T0:]) / 2.0
    Ymean = effect * (np.arange(1, T + 1) > T0) + 2 * U.sum(1)
    Y = rng.uniform(Ymean - 1, Ymean + 1)
    W = np.column_stack([rng.uniform(2 * U[:, j] - 1, 2 * U[:, j] + 1) for j in range(nU)])
    Z = np.column_stack([rng.uniform(2 * U[:, j] - 1, 2 * U[:, j] + 1) for j in range(nU)])
    return Y, W, Z, T0


def test_estimate_dr_recovers_effect():
    Y, W, Z, T0 = _dr_panel(seed=1, T=800, effect=3.0)
    cf, alpha, beta, att, se = estimate_dr(Y, W, Z, T0, hac_bandwidth=4)
    assert cf.shape == (len(Y),)
    assert alpha.shape == (W.shape[1] + 1,)        # intercept + donors
    assert beta.shape == (Z.shape[1] + 1,)
    assert abs(att - 3.0) < 0.6                    # doubly-robust ATT near truth
    assert np.isfinite(se) and se > 0


def test_estimate_pipw_recovers_effect():
    Y, W, Z, T0 = _dr_panel(seed=2, T=800, effect=3.0)
    beta, att, se = estimate_pipw(Y, W, Z, T0, hac_bandwidth=4)
    assert beta.shape == (Z.shape[1] + 1,)
    assert abs(att - 3.0) < 0.6
    assert np.isfinite(se) and se > 0


def test_dr_pipw_deterministic():
    Y, W, Z, T0 = _dr_panel(seed=3)
    a = estimate_dr(Y, W, Z, T0, hac_bandwidth=4)
    b = estimate_dr(Y, W, Z, T0, hac_bandwidth=4)
    assert a[3] == b[3] and a[4] == b[4]


def _dr_panel_nonlinear(seed=0, T=1200, nU=2, effect=2.0):
    """Same DGP but Y(0) is NONLINEAR in U, so the linear outcome bridge is misspecified."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed); T0 = T // 2
    U = np.empty((T, nU)); U[0] = rng.normal(size=nU)
    for t in range(1, T): U[t] = 0.1 * U[t-1] + 0.9 * rng.normal(size=nU)
    U = norm.cdf(U); U[:T0] = -np.log1p(-U[:T0]); U[T0:] = -np.log1p(-U[T0:]) / 2.0
    Ymean = effect * (np.arange(1, T+1) > T0) + 2 * (U ** 2).sum(1)   # nonlinear
    Y = rng.uniform(Ymean - 1, Ymean + 1)
    W = np.column_stack([rng.uniform(2*U[:,j]-1, 2*U[:,j]+1) for j in range(nU)])
    Z = np.column_stack([rng.uniform(2*U[:,j]-1, 2*U[:,j]+1) for j in range(nU)])
    return Y, W, Z, T0


def test_dr_double_robustness_under_misspecified_outcome():
    """DR stays consistent when the outcome bridge is misspecified (correct q rescues it)."""
    bw = 4
    dr_vals, obr_vals = [], []
    for s in range(8):
        Y, W, Z, T0 = _dr_panel_nonlinear(seed=300 + s)
        dr_vals.append(estimate_dr(Y, W, Z, T0, bw)[3])
        cf = estimate_pi(Y, W, Z, T0, len(Y) - T0, len(Y), bw)[0]
        obr_vals.append((Y[T0:] - cf[T0:]).mean())
    dr_mean, obr_mean = np.mean(dr_vals), np.mean(obr_vals)
    assert abs(dr_mean - 2.0) < 0.5            # doubly robust: still ~2
    assert abs(obr_mean - 2.0) > 1.0           # outcome bridge alone is badly biased
