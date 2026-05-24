"""Unit tests for the proximal_helpers estimation and inference functions."""

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.proximal_helpers.inference import bartlett, hac
from mlsynth.utils.proximal_helpers.estimation import (
    estimate_pi,
    estimate_pi_surrogate,
    estimate_pi_surrogate_post,
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
