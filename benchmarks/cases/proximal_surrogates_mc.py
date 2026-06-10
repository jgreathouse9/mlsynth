"""Proximal-with-surrogates Path-B: the Section 4.1 robustness Monte Carlo.

Validates mlsynth's PI / PIS / PIPost against the data-generating process the
authors ship in their reference repo (``freshtaste/proximal`` ``dgp.py``),
accompanying

    Liu, J., Tchetgen Tchetgen, E. J., & Varjao, C. (2024). "Proximal Causal
    Inference for Synthetic Control with Surrogates." AISTATS.

The DGP
(:func:`mlsynth.utils.proximal_helpers.simulation.simulate_proximal_surrogates`)
puts a **trending** ``log t`` latent factor behind the treated unit and the
donor block, with the donor outcomes ``W`` and an independent proxy block
``Z0`` as two error-prone copies of that factor. This is the regime the paper
highlights: classical SC fits an error-laden donor regression and extrapolates
the trend with growing bias, whereas the proximal estimators -- which
instrument ``W`` with ``Z0`` -- recover the true ``ATT = 1`` with near-nominal
coverage. The surrogate estimator **PIS** additionally borrows the post-period
surrogates ``X`` and attains the lowest MSE.

  ========  ============  ===========  ============
  Estimator mean ATT      MSE          95% coverage
  ========  ============  ===========  ============
  PI        ~1.00         ~0.09        ~0.95
  PIS       ~1.00         ~0.05 (low)  ~0.96
  PIPost    ~1.03         ~0.12        ~0.95
  SC        ~1.32 (bias)  ~0.19        --
  ========  ============  ===========  ============

Path B (scenario 3, the authors' own DGP): the case asserts the geometry --
the proximal trio recovers the truth with near-nominal coverage, PIS attains
the lowest MSE of the three, and all three beat the biased classical-SC
baseline -- not exact Monte Carlo cells. The estimators are driven at the
array level (closed-form GMM) for speed.
"""
from __future__ import annotations

import warnings

import numpy as np

from mlsynth.utils.proximal_helpers.simulation import simulate_proximal_surrogates

T = 200
M = 120          # Monte Carlo replications
_TRUE = 1.0
_LAG = 0         # the reference drives the HAC at lag 0 for this design


def _sc_baseline(y: np.ndarray, W: np.ndarray, T0: int) -> float:
    """Non-negative pre-period SC fit -- the errors-in-variables baseline."""
    from scipy.optimize import nnls
    w, _ = nnls(W[:T0], y[:T0])
    cf = W @ w
    return float(np.mean(y[T0:] - cf[T0:]))


def run() -> dict:
    from mlsynth.utils.proximal_helpers.pi.estimation import estimate_pi
    from mlsynth.utils.proximal_helpers.pis.estimation import estimate_pi_surrogate
    from mlsynth.utils.proximal_helpers.pipost.estimation import estimate_pi_surrogate_post

    rng = np.random.default_rng(5500)
    pi = np.empty(M); pis = np.empty(M); pipost = np.empty(M); sc = np.empty(M)
    pi_cov = np.empty(M); pis_cov = np.empty(M); pipost_cov = np.empty(M)
    for i in range(M):
        s = simulate_proximal_surrogates(T=T, rng=rng)
        T0 = s.T0; npost = T - T0
        W, Z0 = s.donor_outcomes, s.donor_proxies
        X, Z1 = s.surrogate_outcomes, s.surrogate_proxies
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cf, _, se = estimate_pi(s.y, W, Z0, T0, npost, T, _LAG)
            tau_s, _, _, se_s = estimate_pi_surrogate(s.y, W, Z0, Z1, X, T0, npost, T, _LAG)
            tau_p, _, _, se_p = estimate_pi_surrogate_post(s.y, W, Z0, Z1, X, T0, npost, _LAG)
        pi_i = float(np.mean(s.y[T0:] - cf[T0:]))
        pi[i], pis[i], pipost[i] = pi_i, tau_s, tau_p
        pi_cov[i] = abs(pi_i - s.true_att) <= 1.96 * se
        pis_cov[i] = abs(tau_s - s.true_att) <= 1.96 * se_s
        pipost_cov[i] = abs(tau_p - s.true_att) <= 1.96 * se_p
        sc[i] = _sc_baseline(s.y, W, T0)

    def mse(a):
        return float(np.mean((a - _TRUE) ** 2))

    return {
        "pi_bias": float(pi.mean()) - _TRUE,
        "pis_bias": float(pis.mean()) - _TRUE,
        "pipost_bias": float(pipost.mean()) - _TRUE,
        "pi_coverage": float(pi_cov.mean()),
        "pis_coverage": float(pis_cov.mean()),
        "pipost_coverage": float(pipost_cov.mean()),
        "pis_mse": mse(pis),
        "sc_bias": float(sc.mean()) - _TRUE,
        "proximal_beats_sc_mse": float(max(mse(pi), mse(pis), mse(pipost)) < mse(sc)),
        "pis_lowest_mse": float(mse(pis) <= min(mse(pi), mse(pipost))),
    }


# Deterministic (seeded). Tolerances absorb the Monte Carlo noise at M=120. The
# reproduced facts (Liu-Tchetgen Tchetgen-Varjao Sec 4.1): under a trending
# latent factor the proximal trio recovers the true ATT=1 with near-nominal
# coverage, PIS attains the lowest MSE, and all three beat the biased classical
# SC baseline.
EXPECTED = {
    "pi_bias": (0.0, 0.08),
    "pis_bias": (0.0, 0.08),
    "pipost_bias": (0.0, 0.10),
    "pi_coverage": (0.95, 0.10),
    "pis_coverage": (0.96, 0.10),
    "pipost_coverage": (0.95, 0.10),
    "pis_mse": (0.05, 0.05),
    "sc_bias": (0.32, 0.30),               # classical SC: biased by the trend
    "proximal_beats_sc_mse": (1.0, 0.0),
    "pis_lowest_mse": (1.0, 0.0),
}
