"""Calibrated simulation study from Powell (2022), Section 5 / Table 1.

Reproduces Models 1-4 of the paper's Monte Carlo, the data-generating
process for which is

.. math::

   Y_{it} = \\beta_i d_{it} + 5 \\sum_{k=1}^{2} \\lambda_t^{(k)} \\mu_i^{(k)}
            + \\epsilon_{it},

with the piecewise factor paths :math:`\\lambda_t^{(1)}, \\lambda_t^{(2)}`
defined below, :math:`\\mu_i^{(k)} \\sim U(0, 1)`,
:math:`\\epsilon_{it} \\sim N(0, \\tfrac14)`, and unit-specific effects
:math:`\\beta_i = \\sum_k \\mu_i^{(k)} - \\tfrac1n \\sum_i \\sum_k \\mu_i^{(k)}`
(so the true average effect is exactly zero each draw). The treatment
:math:`d_{it}` is continuous and a function of the same factors, so it is
correlated with the interactive fixed effects -- which is why two-way
fixed effects is badly biased and CTSC is not.

Model variations:

======  ========================================  ====  ====
Model   treatment construction                    n     T
======  ========================================  ====  ====
1       d = factors + U(0, 10)                    10    50
2       d = factors + U(0, 10)                    30    50
3       d = factors + U(0, 5) * (beta_i + 3)      10    50
4       d = factors + U(0, 10)                    10    20
======  ========================================  ====  ====

The headline calibration target (paper Table 1): CTSC mean bias
:math:`\\approx 0` with small MAD/RMSE, while the two-way fixed-effects
estimator has mean bias :math:`\\approx 0.85` (Models 1-2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .estimate import fit_ctsc

_MODEL_DIMS = {1: (10, 50), 2: (30, 50), 3: (10, 50), 4: (10, 20)}


def _lambda1(t: int) -> float:
    if t < 20:
        return 0.1 * t
    if t < 30:
        return 2.0
    return 2.0 + (t - 30) / 30.0


def _lambda2(t: int) -> float:
    if t <= 5:
        return 2.0
    if t < 20:
        return -4.0
    if t < 40:
        return 6.0
    return 1.0


def generate_model(model: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float]:
    """Draw one panel from Powell (2022) Model ``model`` in ``{1, 2, 3, 4}``.

    Returns ``(Y, D, true_ae)`` where ``Y`` is ``(n, T)``, ``D`` is
    ``(n, T, 1)``, and ``true_ae`` is the true average effect (0).
    """
    if model not in _MODEL_DIMS:
        raise ValueError(f"model must be in {{1,2,3,4}}, got {model}.")
    n, T = _MODEL_DIMS[model]
    mu = rng.uniform(0.0, 1.0, (n, 2))
    beta = mu.sum(axis=1) - mu.sum(axis=1).mean()         # centered -> true AE = 0
    L1 = np.array([_lambda1(t) for t in range(1, T + 1)])
    L2 = np.array([_lambda2(t) for t in range(1, T + 1)])
    factors = np.outer(mu[:, 0], L1) + np.outer(mu[:, 1], L2)   # (n, T)
    eps = rng.normal(0.0, 0.5, (n, T))
    if model in (1, 2, 4):
        d = factors + rng.uniform(0.0, 10.0, (n, T))
    else:  # model 3: within-unit variation scaled by (beta_i + 3)
        d = factors + rng.uniform(0.0, 5.0, (n, T)) * (beta + 3.0)[:, None]
    Y = beta[:, None] * d + 5.0 * factors + eps
    return Y, d[:, :, None], 0.0


def twoway_fe_effect(Y: np.ndarray, D: np.ndarray) -> float:
    """Two-way (unit + time) fixed-effects slope on the treatment -- the
    biased baseline the paper compares against.
    """
    d = D[:, :, 0]
    Yw = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()
    Dw = d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()
    num = float(Dw.ravel() @ Yw.ravel())
    den = float(Dw.ravel() @ Dw.ravel())
    return num / den if abs(den) > 1e-12 else np.nan


@dataclass(frozen=True)
class SimulationSummary:
    """Monte-Carlo summary for one model (mean bias, MAD, RMSE)."""

    model: int
    n_sims: int
    ctsc_mean_bias: float
    ctsc_mad: float
    ctsc_rmse: float
    fe_mean_bias: float
    fe_mad: float
    fe_rmse: float


def run_simulation(
    model: int, n_sims: int = 100, *, seed: int = 0,
) -> SimulationSummary:
    """Run ``n_sims`` Monte-Carlo draws of ``model`` and summarise CTSC vs FE.

    The true average effect is zero, so bias equals the mean estimate.
    """
    rng = np.random.default_rng(seed)
    ctsc = np.empty(n_sims)
    fe = np.empty(n_sims)
    for s in range(n_sims):
        Y, D, _ = generate_model(model, rng)
        ctsc[s] = fit_ctsc(Y, D)["average_effect"][0]
        fe[s] = twoway_fe_effect(Y, D)
    return SimulationSummary(
        model=model,
        n_sims=n_sims,
        ctsc_mean_bias=float(ctsc.mean()),
        ctsc_mad=float(np.median(np.abs(ctsc))),
        ctsc_rmse=float(np.sqrt((ctsc ** 2).mean())),
        fe_mean_bias=float(fe.mean()),
        fe_mad=float(np.median(np.abs(fe))),
        fe_rmse=float(np.sqrt((fe ** 2).mean())),
    )
