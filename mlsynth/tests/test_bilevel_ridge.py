"""Tests for the ridge-augmentation layer of the bilevel SCM engine (ASCM).

Covers the standalone ridge primitives, the ``BilevelSCM(augment="ridge")``
capability, and a regression against the ``augsynth`` R package on its canonical
Kansas tax-cut example (Average ATT -0.0401, plain SCM -0.0294).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from mlsynth.utils.bilevel import (
    BilevelSCM,
    best_lambda,
    generate_lambdas,
    ridge_augment_weights,
    solve_ridge,
)

cp = pytest.importorskip("cvxpy")  # ridge base is an exact simplex QP

_KANSAS = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "kansas_taxcut.csv"
)


def _toy(seed: int = 0, T0: int = 20, J: int = 6):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(T0 + 5, 2))
    loads = rng.uniform(0.2, 1.0, size=(J, 2))
    Y0 = factor @ loads.T + 0.05 * rng.normal(size=(T0 + 5, J))
    w = np.zeros(J); w[: 3] = [0.5, 0.3, 0.2]
    y = Y0 @ w + 0.05 * rng.normal(size=T0 + 5)
    return y[:T0], Y0[:T0], y, Y0


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def test_generate_lambdas_grid():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 8))
    lambdas = generate_lambdas(X, lambda_min_ratio=1e-8, n_lambda=20)
    assert lambdas.shape == (21,)                      # augsynth: n_lambda+1
    assert np.all(np.diff(lambdas) < 0)                # strictly descending
    lmax = np.linalg.svd(X, compute_uv=False)[0] ** 2
    assert lambdas[0] == pytest.approx(lmax)
    assert lambdas[-1] == pytest.approx(lmax * 1e-8)


def test_solve_ridge_zero_lambda_interpolates():
    # With lambda -> 0 and a wide donor matrix the ridge correction drives the
    # pre-period imbalance to ~0 (exact interpolation of the centered target).
    rng = np.random.default_rng(1)
    B = rng.normal(size=(10, 15))           # m < J: underdetermined
    A = rng.normal(size=10)
    W = np.full(15, 1 / 15)
    W_aug = W + solve_ridge(A, B, W, lambda_=1e-10)
    assert np.linalg.norm(A - B @ W_aug) < 1e-4


def test_best_lambda_1se_is_larger_than_min():
    lambdas = np.array([10.0, 3.0, 1.0, 0.3, 0.1])
    mean = np.array([5.0, 2.0, 1.05, 1.0, 1.2])      # min at lambda=0.3
    se = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
    lam_min = best_lambda(lambdas, mean, se, min_1se=False)
    lam_1se = best_lambda(lambdas, mean, se, min_1se=True)
    assert lam_min == 0.3                              # outright CV minimum
    # 1-SE picks the largest lambda within min_error+se (1.3): lambda=1.0
    assert lam_1se == 1.0 and lam_1se > lam_min


# --------------------------------------------------------------------------- #
# ridge_augment_weights
# --------------------------------------------------------------------------- #
def test_ridge_augment_base_is_simplex():
    y_pre, Y0_pre, _, _ = _toy()
    ra = ridge_augment_weights(y_pre, Y0_pre)
    assert ra.W_base.min() >= -1e-8
    assert ra.W_base.sum() == pytest.approx(1.0, abs=1e-6)
    # augmentation moves off the simplex but the correction is additive
    assert np.allclose(ra.W, ra.W_base + ra.W_ridge)


def test_ridge_augment_fixed_lambda_skips_cv():
    y_pre, Y0_pre, _, _ = _toy()
    ra = ridge_augment_weights(y_pre, Y0_pre, lambda_=1.0)
    assert ra.lambda_ == 1.0
    assert ra.cv is None
    ra_cv = ridge_augment_weights(y_pre, Y0_pre)
    assert ra_cv.cv is not None
    assert set(ra_cv.cv) == {"lambdas", "errors_mean", "errors_se"}


def test_ridge_augment_improves_pre_fit_over_simplex():
    # The whole point of ASCM: the augmented weights close residual pre-period
    # imbalance the simplex SCM leaves behind.
    y_pre, Y0_pre, _, _ = _toy(seed=3, T0=25, J=8)
    ra = ridge_augment_weights(y_pre, Y0_pre, lambda_=1e-3)
    mu = Y0_pre.mean(axis=1)
    base_imb = np.sum((y_pre - mu - (Y0_pre - mu[:, None]) @ ra.W_base) ** 2)
    aug_imb = np.sum((y_pre - mu - (Y0_pre - mu[:, None]) @ ra.W) ** 2)
    assert aug_imb < base_imb


# --------------------------------------------------------------------------- #
# Engine capability
# --------------------------------------------------------------------------- #
def test_engine_augment_ridge_flags():
    y_pre, Y0_pre, _, _ = _toy()
    r = BilevelSCM("outcome-only", augment="ridge").fit(y_pre, Y0_pre)
    assert r.augment == "ridge"
    assert r.is_simplex is False
    assert r.lambda_ is not None
    assert r.W_base is not None and r.W_base.sum() == pytest.approx(1.0, abs=1e-6)
    assert "cv" in r.augment_diagnostics


def test_engine_no_augment_is_simplex():
    y_pre, Y0_pre, _, _ = _toy()
    r = BilevelSCM("outcome-only").fit(y_pre, Y0_pre)
    assert r.augment is None
    assert r.is_simplex is True
    assert r.W_base is None and r.lambda_ is None


def test_engine_bad_augment_raises():
    with pytest.raises(ValueError, match="augment"):
        BilevelSCM("outcome-only", augment="lasso")


def test_engine_ridge_fixed_lambda():
    y_pre, Y0_pre, _, _ = _toy()
    r = BilevelSCM("outcome-only", augment="ridge", ridge_lambda=2.0).fit(y_pre, Y0_pre)
    assert r.lambda_ == 2.0
    assert "cv" not in r.augment_diagnostics      # fixed lambda -> no CV curve


# --------------------------------------------------------------------------- #
# Regression: augsynth R Kansas tax-cut (single_augsynth, progfunc="Ridge")
# --------------------------------------------------------------------------- #
def test_augsynth_kansas_replication():
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(_KANSAS)
    piv = df.pivot(index="fips", columns="year_qtr", values="lngdpcapita").sort_index()
    times = np.array(sorted(df["year_qtr"].unique()))
    pre = int((times < 2012.25).sum())
    y = piv.loc[20.0].to_numpy()
    Y0 = piv.drop(index=20.0).to_numpy().T          # (T, J)
    y_pre, Y0_pre = y[:pre], Y0[:pre]

    r = BilevelSCM("outcome-only", augment="ridge").fit(y_pre, Y0_pre)
    att = float(np.mean((y - r.counterfactual(Y0))[pre:]))
    att_base = float(np.mean((y - Y0 @ r.W_base)[pre:]))

    # augsynth vignette: plain SCM Average ATT -0.0294; Ridge ASCM -0.0401.
    assert att_base == pytest.approx(-0.0294, abs=5e-4)
    assert att == pytest.approx(-0.0401, abs=5e-4)
    assert r.lambda_ == pytest.approx(0.0787, abs=5e-3)
