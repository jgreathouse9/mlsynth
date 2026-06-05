"""Tests for SPILLSYNTH method='sar' (Sakaguchi & Tagawa 2026 spatial SCM)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import SPILLSYNTH
from mlsynth.config_models import SPILLSYNTHConfig
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.spillsynth_helpers.sar.sampler import row_normalize


def _rook(nr, nc):
    N = nr * nc
    W = np.zeros((N, N))
    idx = lambda r, c: r * nc + c
    for r in range(nr):
        for c in range(nc):
            i = idx(r, c)
            if r > 0: W[i, idx(r - 1, c)] = 1
            if r < nr - 1: W[i, idx(r + 1, c)] = 1
            if c > 0: W[i, idx(r, c - 1)] = 1
            if c < nc - 1: W[i, idx(r, c + 1)] = 1
    return W


def _sar_panel(rho, nr=4, nc=4, T=30, T0=20, sigma2=0.1, seed=0):
    """Generate one spatial-spillover panel (the paper's simulation DGP)."""
    rng = np.random.default_rng(seed)
    N = nr * nc
    Wn = row_normalize(_rook(nr, nc))
    w = np.zeros(N); w[:4] = 1.0; wn = w / w.sum()
    alpha = np.zeros(N)
    alpha[0] = 0.5; alpha[1] = -0.2; alpha[2:4] = 0.4
    alpha[4:min(10, N)] = 0.1 / 6
    IN = np.eye(N)
    Ainv = np.linalg.inv(IN - rho * Wn - rho * np.outer(wn, alpha))
    Apost = np.linalg.inv(IN - rho * Wn)
    err = rng.normal(0, np.sqrt(sigma2), (T, N))
    Yc0 = (Ainv @ err.T).T
    Y00 = Yc0 @ alpha
    tau = rng.normal(1.0, 1.0, T - T0)
    Y0 = Y00.copy(); Y0[T0:] += tau
    Yc = Yc0.copy()
    for t in range(T0, T):
        Yc[t] = Apost @ (rho * wn * Y0[t] + err[t])
    labels = ["T"] + [f"c{i}" for i in range(N)]
    Ypanel = np.vstack([Y0[None, :], Yc.T])
    rows = []
    for ui, lab in enumerate(labels):
        for t in range(T):
            rows.append({"unit": lab, "time": t, "y": Ypanel[ui, t],
                         "d": int(lab == "T" and t >= T0)})
    df = pd.DataFrame(rows)
    Wdf = pd.DataFrame(Wn, index=labels[1:], columns=labels[1:])
    wser = pd.Series(wn, index=labels[1:])
    return df, Wdf, wser, float(tau.mean())


def _fit(df, Wdf, wser, **kw):
    cfg = dict(df=df, outcome="y", treat="d", unitid="unit", time="time",
               method="sar", spatial_W=Wdf, spatial_w=wser, p_factors=0,
               mcmc_iter=3000, mcmc_burn=1000, step_rho=0.05, mcmc_seed=1,
               display_graphs=False)
    cfg.update(kw)
    return SPILLSYNTH(cfg).fit()


# --------------------------------------------------------------------------
def test_recovers_rho_and_att_under_spillover():
    df, Wdf, wser, true_att = _sar_panel(rho=0.6, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _fit(df, Wdf, wser)
    assert res.method == "sar"
    # rho recovered near truth
    assert res.sar.rho_hat == pytest.approx(0.6, abs=0.12)
    # SAR ATT closer to the truth than the spillover-biased SCM ATT
    assert abs(res.att - true_att) < abs(res.att_scm - true_att)
    # accessors route and have the right shapes
    assert res.gap.shape == res.counterfactual.shape
    assert len(res.spillover_effects) == 16
    assert res.sar.ate_ci[0] <= res.att <= res.sar.ate_ci[1]


def test_rho_zero_reduces_to_scm():
    df, Wdf, wser, _ = _sar_panel(rho=0.0, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _fit(df, Wdf, wser)
    # no spatial dependence -> rho near 0 and SAR ~ SCM
    assert abs(res.sar.rho_hat) < 0.15
    assert res.att == pytest.approx(res.att_scm, abs=0.25)


def test_covariates_run_and_factor_block():
    df, Wdf, wser, _ = _sar_panel(rho=0.5, seed=1)
    rng = np.random.default_rng(2)
    df = df.copy()
    df["x1"] = rng.normal(size=len(df))            # a spurious covariate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _fit(df, Wdf, wser, covariates=["x1"], p_factors=1,
                   mcmc_iter=2000, mcmc_burn=600)
    assert res.sar.beta_hat is not None and res.sar.beta_hat.shape == (1,)
    assert np.isfinite(res.sar.rho_hat)


def test_W_as_plain_array_in_label_order():
    df, Wdf, wser, _ = _sar_panel(rho=0.4, seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _fit(df, Wdf.to_numpy(), wser.to_numpy())   # bare arrays
    assert np.isfinite(res.sar.rho_hat)


# --------------------------------------------------------------------------
def test_missing_spatial_weights_raises():
    df, Wdf, wser, _ = _sar_panel(rho=0.3, seed=0)
    with pytest.raises(MlsynthConfigError):
        SPILLSYNTHConfig(df=df, outcome="y", treat="d", unitid="unit",
                         time="time", method="sar", spatial_w=wser)


def test_burn_exceeds_iter_raises():
    df, Wdf, wser, _ = _sar_panel(rho=0.3, seed=0)
    with pytest.raises(MlsynthConfigError):
        SPILLSYNTHConfig(df=df, outcome="y", treat="d", unitid="unit",
                         time="time", method="sar", spatial_W=Wdf,
                         spatial_w=wser, mcmc_iter=1000, mcmc_burn=1000)


def test_plot_runs(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    df, Wdf, wser, _ = _sar_panel(rho=0.5, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _fit(df, Wdf, wser, display_graphs=True,
                   save="/tmp/_sar_test_plot.png")
    assert res.method == "sar"
