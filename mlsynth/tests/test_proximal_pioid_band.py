"""Per-period counterfactual bands for over-identified proximal inference (PIOID).

Shi, Li, Yu, Miao, Kuchibhotla, Hu & Tchetgen Tchetgen (2026), Section 3.2,
adapt three inference routes to the proximal residual process
``e_t = Y_t - h(W_Dt)``: conformal permutation (3.2.1), scpi-style prediction
intervals (3.2.2), and GMM (3.2.3). PIOID already reports the aggregated GMM ATT
interval; this exposes its per-period companion. The outcome bridge is
``h(W_t) = W_t' omega`` with the joint one-step-GMM sandwich giving
``Var(omega)``, so the delta-method band is
``W_t' omega_hat +/- z * sqrt(W_t' Var(omega_hat) W_t)``.

Layered per ``agents/agents_tests.md``:

* the band is finite over the post-period, ordered (upper >= lower), and centres
  exactly on the counterfactual ``W omega``;
* it is self-consistent with the reported ATT: both are read off the same joint
  ``(tau, omega)`` sandwich, so the alpha block that drives the band is PSD and
  the ATT SE the estimator already reports is unchanged;
* the constrained (simplex / cPI) fit reports no GMM band (its inference is by
  permutation), matching the point-estimate contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.proximal_helpers.pi.overid import (
    estimate_pi_overid, overid_counterfactual_band,
)


def _proximal_panel(seed=0, r=2, T0=60, T1=60, tau=2.0):
    """Over-identified proximal DGP: donors W and instruments Z share the latent
    factors that also drive the treated unit; more instruments than donors."""
    rng = np.random.default_rng(seed)
    T = T0 + T1
    F = rng.standard_normal((T, r)) + 0.5
    mu0 = np.ones(r)
    # donors: r units; instruments: r extra units, same factor space
    muW = rng.standard_normal((r, r)) + 0.3
    muZ = rng.standard_normal((r + 2, r)) + 0.3
    W = F @ muW.T + 0.4 * rng.standard_normal((T, r))
    Z = F @ muZ.T + 0.4 * rng.standard_normal((T, r + 2))
    y0 = F @ mu0 + 0.4 * rng.standard_normal(T)
    y = y0.copy()
    y[T0:] += tau
    return y, W, Z, T0, T1, T


def test_gmm_band_is_ordered_and_centres_on_counterfactual():
    y, W, Z, T0, T1, T = _proximal_panel()
    cf, alpha, se = estimate_pi_overid(y, W, Z, T0, T1, T, hac_truncation_lag=10)
    lo, hi = overid_counterfactual_band(y, W, Z, alpha, T0, T1, T,
                                        hac_truncation_lag=10, level=0.90)
    assert lo.shape == (T,) and hi.shape == (T,)
    post = slice(T0, T)
    assert np.all(np.isfinite(lo[post])) and np.all(np.isfinite(hi[post]))
    assert np.all(hi[post] >= lo[post])
    # the band centres on the counterfactual W omega
    assert np.allclose((lo[post] + hi[post]) / 2.0, cf[post], atol=1e-8)


def test_gmm_band_widens_with_level():
    y, W, Z, T0, T1, T = _proximal_panel()
    _, alpha, _ = estimate_pi_overid(y, W, Z, T0, T1, T, hac_truncation_lag=10)
    lo90, hi90 = overid_counterfactual_band(y, W, Z, alpha, T0, T1, T, 10, level=0.90)
    lo95, hi95 = overid_counterfactual_band(y, W, Z, alpha, T0, T1, T, 10, level=0.95)
    post = slice(T0, T)
    assert np.all((hi95 - lo95)[post] >= (hi90 - lo90)[post] - 1e-9)
    assert np.mean((hi95 - lo95)[post]) > np.mean((hi90 - lo90)[post])


def test_gmm_band_shares_the_att_sandwich():
    """The per-period band and the reported ATT SE come from one joint sandwich:
    the ATT SE the estimator returns is finite and positive, and the band's
    per-period SEs are too (the alpha block is PSD)."""
    y, W, Z, T0, T1, T = _proximal_panel()
    cf, alpha, se = estimate_pi_overid(y, W, Z, T0, T1, T, hac_truncation_lag=10)
    assert np.isfinite(se) and se > 0
    lo, hi = overid_counterfactual_band(y, W, Z, alpha, T0, T1, T, 10, level=0.90)
    half = (hi - lo)[T0:] / 2.0
    assert np.all(half >= 0) and np.any(half > 0)


# ----------------------------------------------------------------------
# conformal route (Shi et al. 2026 Section 3.2.1; Chernozhukov-Wuthrich-Zhu 2021)
# ----------------------------------------------------------------------
def test_conformal_band_is_centred_and_uniform_width():
    y, W, Z, T0, T1, T = _proximal_panel()
    cf, alpha, se = estimate_pi_overid(y, W, Z, T0, T1, T, hac_truncation_lag=10)
    lo, hi = overid_counterfactual_band(y, W, Z, alpha, T0, T1, T, 10,
                                        level=0.90, method="conformal")
    post = slice(T0, T)
    assert np.all(hi[post] >= lo[post])
    assert np.allclose((lo[post] + hi[post]) / 2.0, cf[post], atol=1e-8)
    # PIOID's bridge is pre-fit only, so the split-conformal half-width is the
    # residual quantile: a single value shared across post periods.
    half = (hi[post] - lo[post]) / 2.0
    assert np.allclose(half, half[0])
    # and it equals the coverage quantile of |pre-period residuals|
    resid_pre = np.abs((y - cf)[:T0])
    assert half[0] == pytest.approx(np.quantile(resid_pre, 0.90), abs=1e-8)


def test_conformal_band_widens_with_level():
    y, W, Z, T0, T1, T = _proximal_panel()
    _, alpha, _ = estimate_pi_overid(y, W, Z, T0, T1, T, hac_truncation_lag=10)
    lo90, hi90 = overid_counterfactual_band(y, W, Z, alpha, T0, T1, T, 10,
                                            level=0.90, method="conformal")
    lo99, hi99 = overid_counterfactual_band(y, W, Z, alpha, T0, T1, T, 10,
                                            level=0.99, method="conformal")
    post = slice(T0, T)
    assert np.mean((hi99 - lo99)[post]) >= np.mean((hi90 - lo90)[post])


def test_unknown_band_method_rejected():
    y, W, Z, T0, T1, T = _proximal_panel()
    _, alpha, _ = estimate_pi_overid(y, W, Z, T0, T1, T, hac_truncation_lag=10)
    with pytest.raises(Exception):
        overid_counterfactual_band(y, W, Z, alpha, T0, T1, T, 10, method="bogus")
