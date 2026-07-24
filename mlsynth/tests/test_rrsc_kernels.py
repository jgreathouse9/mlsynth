"""Unit tests for the RRSC numerical kernels (factor analysis + robust
regression + LTS). Value-for-value fidelity to the reference R
(``fa.em`` / ``factanal`` / ``robustbase::ltsReg``) on the empirical panels is
pinned in the golden benchmark; here we lock the kernels' mathematical
invariants and the transcribed robustbase constants, self-contained.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.rrsc_helpers.factor import (
    fa_em, factanal_mle, fa_pc, select_nfactors,
)
from mlsynth.utils.rrsc_helpers.robust import (
    robust_factor_estimation, lts_fit, h_alpha_n, _lts_cnp2, _raw_scale,
)


def _factor_panel(n_obs, p, r, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n_obs, r))
    Lam = rng.normal(size=(p, r))
    X = F @ Lam.T + noise * rng.normal(size=(n_obs, p))
    return X, Lam


# -------------------------------------------------------------------- factor
def test_fa_em_recovers_low_rank_subspace():
    X, Lam = _factor_panel(400, 8, 2, noise=0.2)
    G, Sig = fa_em(X, 2)
    assert G.shape == (8, 2) and Sig.shape == (8,)
    assert np.all(Sig > 0) and np.all(np.isfinite(G))
    # estimated loadings span (approximately) the true loading subspace
    Q, _ = np.linalg.qr(G)
    resid = Lam - Q @ (Q.T @ Lam)
    assert np.linalg.norm(resid) / np.linalg.norm(Lam) < 0.15


def test_fa_pc_reconstruction_shapes():
    X, _ = _factor_panel(50, 10, 3)
    G, Sig = fa_pc(X, 3)
    assert G.shape == (10, 3) and Sig.shape == (10,) and np.all(Sig >= 0)


def test_fa_em_wide_panel_obs_le_vars():
    # obs <= vars -- the regime the large-N (Beijing) factor step actually uses
    X, _ = _factor_panel(20, 30, 3, noise=0.2)
    G, Sig = fa_em(X, 3)
    assert G.shape == (30, 3) and np.all(Sig > 0) and np.all(np.isfinite(G))


def test_factanal_rejects_bad_rank():
    X, _ = _factor_panel(80, 4, 2)
    with pytest.raises(MlsynthConfigError):
        factanal_mle(X, 4)                 # q must be < p
    with pytest.raises(MlsynthDataError):
        factanal_mle(X.ravel(), 2)         # not 2D


def test_select_nfactors_tiny_panel_raises():
    with pytest.raises(MlsynthConfigError):
        select_nfactors(np.ones((1, 3)), k_max=5)   # single unit -> no factor selectable


def test_fa_em_rejects_bad_rank():
    X, _ = _factor_panel(30, 4, 2)
    with pytest.raises(MlsynthConfigError):
        fa_em(X, 4)                       # r must be < p
    with pytest.raises(MlsynthDataError):
        fa_em(X.ravel(), 2)               # not 2D


def test_factanal_uniquenesses_and_reconstruction():
    X, _ = _factor_panel(300, 6, 2, noise=0.5)
    load, Psi = factanal_mle(X, 2)
    assert load.shape == (6, 2)
    assert np.all(Psi > 0) and np.all(Psi <= 1.0 + 1e-8)
    # correlation-scale loadings (matching R factanal $loadings) reconstruct the
    # correlation matrix: R ~ Lambda Lambda' + diag(Psi)
    Rc = np.corrcoef(X, rowvar=False)
    approx = load @ load.T + np.diag(Psi)
    assert np.abs(approx - Rc).max() < 0.1


def test_factanal_rejects_constant_series():
    X, _ = _factor_panel(100, 5, 2)
    X[:, 0] = 3.0                          # constant column -> NaN correlations
    with pytest.raises(MlsynthDataError):
        factanal_mle(X, 2)


def test_select_nfactors_recovers_true_rank():
    # N units x T0 periods with a clean 3-factor structure
    rng = np.random.default_rng(1)
    r = 3
    F = rng.normal(size=(60, r)); Lam = rng.normal(size=(40, r))
    Y = (Lam @ F.T) + 0.1 * rng.normal(size=(40, 60))
    assert select_nfactors(Y, k_max=8) == r


# -------------------------------------------------------------------- robust
def test_robust_factor_estimation_matches_gls_without_outliers():
    rng = np.random.default_rng(2)
    Lam = rng.normal(size=(20, 2)); f_true = np.array([1.5, -0.7])
    sigma = np.ones(20)
    Yt = Lam @ f_true                      # no noise, no outliers
    f = robust_factor_estimation(Yt, Lam, sigma)
    assert np.allclose(f, f_true, atol=1e-6)


def test_robust_factor_estimation_downweights_outlier():
    rng = np.random.default_rng(3)
    Lam = rng.normal(size=(30, 2)); f_true = np.array([2.0, 1.0])
    sigma = np.ones(30)
    Yt = Lam @ f_true + 0.01 * rng.normal(size=30)
    Yt[0] += 500.0                         # one gross interference outlier
    f_rob = robust_factor_estimation(Yt, Lam, sigma)
    f_ols = np.linalg.lstsq(Lam, Yt, rcond=None)[0]
    # Huber is far closer to the truth than OLS, which the outlier drags away
    assert np.linalg.norm(f_rob - f_true) < np.linalg.norm(f_ols - f_true)
    assert np.linalg.norm(f_rob - f_true) < 0.2


# ----------------------------------------------------------------------- LTS
def test_lts_cnp2_matches_robustbase_constants():
    # transcribed Pison-Van Aelst-Willems (2002) values (robustbase LTScnp2)
    assert _lts_cnp2(2, 9, 0.5) == pytest.approx(1.80358, abs=1e-4)
    assert _lts_cnp2(1, 40, 0.5) == pytest.approx(1.08096, abs=1e-4)


def test_h_alpha_n():
    assert h_alpha_n(0.5, 9, 2) == 6
    assert h_alpha_n(0.5, 40, 1) == 21


def test_lts_fit_ignores_outliers_and_recovers_line():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(20, 2)); beta = np.array([1.0, -2.0])
    y = X @ beta + 0.01 * rng.normal(size=20)
    y[[0, 1, 2]] += np.array([50.0, -40.0, 60.0])   # 3 outliers (< 50% of 20)
    coef, raw_coef, w = lts_fit(X, y, alpha=0.5)
    assert np.allclose(raw_coef, beta, atol=0.1)     # raw LTS locks onto the clean majority
    assert not w[[0, 1, 2]].any()                    # the 3 outliers are down-weighted out


def test_lts_fit_clean_data_keeps_all_and_equals_ols():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(15, 2)); y = X @ np.array([0.5, 0.5]) + 0.05 * rng.normal(size=15)
    coef, _raw, w = lts_fit(X, y, alpha=0.5)
    assert w.all()                                   # robustbase raw scale keeps clean units
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(coef, ols, atol=1e-9)         # reweighted == full OLS


def test_raw_scale_formula():
    # base * consistency * cnp2, positive and finite
    resid = np.array([0.1, -0.2, 0.15, -0.05, 0.3, -0.25, 10.0, -8.0, 9.0])
    s = _raw_scale(resid, h=6, n=9, p=2, alpha=0.5)
    assert s > 0 and np.isfinite(s)


def test_lts_fit_rejects_bad_shape():
    with pytest.raises(MlsynthConfigError):
        lts_fit(np.ones((2, 3)), np.ones(2))         # n <= p
