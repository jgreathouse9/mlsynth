"""PIOID over-identification (Hansen J) test: size and power on the authors' DGP.

Validates the Hansen (1982) J-test of the PIOID over-identifying restrictions
against the authors' own linear interactive-fixed-effects simulation
(``shixu0830/SyntheticControl``, ``Simulation/myfunctions_LinearSetting.R``,
``generate.U`` / ``run.one``) behind

    Shi, X., Li, K. T., Yu, M., Miao, W., Kuchibhotla, A. K., Hu, M., &
    Tchetgen Tchetgen, E. J. (2026). "Theory for Identification and Inference
    with Synthetic Controls: A Proximal Causal Inference Framework." JASA.

The DGP: ``m`` nonstationary log-trend factors ``lambda_t = log(1:t) +
N(0, sd)``; each control loads a single factor (identity blocks) and the treated
unit loads all-ones, so the donor weight ``omega = 1`` reproduces the treated
counterfactual and the effect ``true.beta = 2`` is recovered. The authors' own
runs are just-identified (``n.Z = n.W``), so this case keeps their factor,
loading and error structure and adds extra proxies (further noisy measurements
of the same factors -- valid, over-identifying instruments) to exercise the
excess restrictions; the power arm contaminates one proxy with the treated
error, an exclusion violation.

The J-test statistic ``J = T0 * mbar' S^{-1} mbar`` is asymptotically
``chi^2_{d-p}`` (``d`` proxies, ``p`` donors) under the null that every
instrument is a valid proximal control. Reproduced facts (``m = 3`` factors,
``d = 6`` proxies, ``p = 3`` donors so ``df = 3``, ``t0 = 100``, moment
bandwidth 0 -- the classical i.i.d. setting):

  =====================  ===========  ==================
  Quantity               J-test       reference
  =====================  ===========  ==================
  size, valid proxies    ~0.05        nominal 0.05
  power, invalid proxy   ~0.75        >> size
  df                     3            d - p = 6 - 3
  just-identified J      nan          undefined (df 0)
  =====================  ===========  ==================

Path B (the authors' DGP): the case asserts the geometry -- near-nominal size
under valid proxies and clear power against an exclusion-violating proxy -- not
exact rejection cells. The moment bandwidth is 0, which is exactly calibrated
for the paper's classical i.i.d.-error setting; a wide bandwidth over-smooths
the test toward conservatism (that sensitivity is documented on the estimator).
"""
from __future__ import annotations

import numpy as np

_M_FACTORS = 3
_N_PROXY = 6
_T0 = 100
_TRUE_BETA = 2.0
_HAC_LAG = 0        # classical i.i.d. setting: matched moment bandwidth
M = 400             # Monte Carlo replications per arm


def _draw(rng, n_proxy=_N_PROXY, bad=0.0):
    """One draw of the authors' linear IFEM setting (log-trend factors)."""
    m, t0 = _M_FACTORS, _T0
    t = 2 * t0
    lam = np.column_stack(
        [np.log(np.arange(1, t + 1)) + rng.normal(0, 1.0, t) for _ in range(m)])
    eps_Y = rng.normal(0, 1.0, t)
    Y = lam.sum(axis=1) + eps_Y + _TRUE_BETA * (np.arange(t) >= t0)
    W = np.column_stack([lam[:, j] + rng.normal(0, 1.0, t) for j in range(m)])
    Z = np.column_stack(
        [lam[:, i % m] + rng.normal(0, 1.0, t) + (bad * eps_Y if i == 0 else 0.0)
         for i in range(n_proxy)])
    return Y, W, Z, t0, t


def _reject_and_att(base_seed: int, bad: float):
    """Rejection rate at 5% and mean recovered ATT over M draws."""
    from mlsynth.utils.proximal_helpers.pi.overid import (
        estimate_pi_overid, overid_j_test)

    rng = np.random.default_rng(base_seed)
    pvals = np.empty(M)
    atts = np.empty(M)
    df_seen = -1
    for i in range(M):
        Y, W, Z, T0, T = _draw(rng, bad=bad)
        _j, df_seen, pvals[i] = overid_j_test(Y, W, Z, T0, _HAC_LAG)
        cf, _a, _se = estimate_pi_overid(Y, W, Z, T0, T - T0, T, _HAC_LAG)
        atts[i] = np.mean((Y - cf)[T0:])
    return float(np.mean(pvals < 0.05)), float(np.mean(atts)), int(df_seen)


def run() -> dict:
    from mlsynth.utils.proximal_helpers.pi.overid import overid_j_test

    size, att_valid, df = _reject_and_att(4321, bad=0.0)
    power, _att_bad, _df = _reject_and_att(8765, bad=1.5)

    # Just-identified (d == p): the test is undefined and returns nan / df 0.
    rng = np.random.default_rng(111)
    Yj, Wj, Zj, T0j, _T = _draw(rng, n_proxy=_M_FACTORS)   # d == p == m
    j_ji, df_ji, p_ji = overid_j_test(Yj, Wj, Zj, T0j, _HAC_LAG)

    return {
        "size_valid_proxies": size,
        "power_invalid_proxy": power,
        "recovered_att": att_valid,
        "df": float(df),
        "just_identified_is_nan": float(np.isnan(j_ji) and df_ji == 0),
    }


# Deterministic (seeds 4321 / 8765, M=400, m=3 factors, d=6 proxies, p=3 donors,
# t0=100, moment bandwidth 0). The J-test is well-calibrated on the authors' own
# linear IFEM DGP -- near-nominal size under valid over-identified proxies and
# clear power against a proxy that violates the exclusion restriction -- and the
# estimator recovers true.beta=2. df = d - p = 3; the just-identified case is
# undefined (nan, df 0).
EXPECTED = {
    "size_valid_proxies": (0.03, 0.05),        # controlled, not over-rejecting (nominal ~0.05)
    "power_invalid_proxy": (0.75, 0.22),       # clearly rejects the bad proxy
    "recovered_att": (2.0, 0.15),              # estimator recovers true.beta
    "df": (3.0, 0.0),                          # d - p = 6 - 3
    "just_identified_is_nan": (1.0, 0.0),
}
