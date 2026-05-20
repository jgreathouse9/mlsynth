"""Pre-experiment power analysis for SPCD.

Adapts the Monte Carlo MDE machinery from
``mlsynth.utils.fast_scm_helpers.power_helpers`` to operate directly on
the SPCD holdout residuals ``r_B``. The MDE is the smallest treatment
effect tau such that the mean-absolute-effect statistic rejects the
null with probability at least ``power_target`` at significance
``alpha``.

Because SPCD's post-period estimator is a fixed linear functional
``tau_hat_t = sum_i contrast_weights[i] * Y[i, t]``, the noise
distribution of ``tau_hat_t`` under H0 is fully characterized by the
distribution of ``r_B`` -- no further model-fitting is needed at the
power stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SPCDPowerAnalysis:
    """Container for MDE and detectability outputs.

    Attributes
    ----------
    mde_tau : float
        Smallest absolute effect detectable at ``power_target``.
        ``np.nan`` when no point on the grid reaches the target.
    mde_pct : float
        ``mde_tau`` expressed as a percentage of the holdout baseline.
    baseline : float
        Baseline level used for the percentage scaling
        (mean of the synthetic-treated trajectory over the holdout
        window).
    critical_stat : float
        ``(1 - alpha)``-quantile of the null distribution of the test
        statistic at horizon ``n_post``.
    feasible : bool
        ``True`` if a non-``nan`` MDE was identified.
    n_post : int
        Post-treatment horizon for which the MDE was computed.
    alpha : float
        Significance level used.
    power_target : float
        Power target used.
    detectability : dict[int, float] or None
        Optional ``horizon -> MDE`` mapping if a horizon grid was
        supplied; ``None`` otherwise.
    """

    mde_tau: float
    mde_pct: float
    baseline: float
    critical_stat: float
    feasible: bool
    n_post: int
    alpha: float
    power_target: float
    detectability: Optional[Dict[int, float]] = None


def _null_distribution(
    residuals: np.ndarray, n_post: int, n_sims: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample the null distribution of mean(|sample|) at horizon n_post.

    Pads the residual pool with Gaussian draws so that resampling of
    size ``n_post`` is always feasible even when ``n_post > len(residuals)``.
    """

    sigma = float(np.std(residuals)) + 1e-12
    pool = np.concatenate([residuals, rng.normal(0.0, sigma, max(n_post, 1))])
    stats = np.empty(n_sims)
    for i in range(n_sims):
        idx = rng.choice(len(pool), size=n_post, replace=False)
        stats[i] = float(np.mean(np.abs(pool[idx])))
    stats.sort()
    return stats


def compute_mde(
    residuals_B: np.ndarray,
    baseline: float,
    n_post: int,
    alpha: float = 0.05,
    power_target: float = 0.8,
    tau_grid: Optional[np.ndarray] = None,
    n_sims: int = 5000,
    n_trials: int = 400,
    seed: int = 1400,
) -> SPCDPowerAnalysis:
    """Compute the minimum detectable effect via Monte Carlo.

    Procedure follows ``power_helpers._analytical_mde`` in
    ``fast_scm_helpers``:

    1. Build the null distribution of the test statistic at horizon
       ``n_post`` by resampling ``residuals_B`` (padded with Gaussian
       draws to handle horizon overflow).
    2. Compute the critical value
       ``c_alpha = quantile(null_stats, 1 - alpha)``.
    3. For each candidate ``tau`` on the grid, draw ``n_trials``
       post-period vectors of the form ``tau + Gaussian noise`` and
       record the fraction of trials whose statistic exceeds
       ``c_alpha``. The smallest ``tau`` with empirical power at or
       above ``power_target`` is the MDE.

    Parameters
    ----------
    residuals_B : np.ndarray
        Out-of-sample residuals on the holdout window.
    baseline : float
        Baseline level used to express the MDE as a percentage.
        Typically the mean of the synthetic-treated trajectory over the
        holdout window.
    n_post : int
        Post-treatment horizon for the MDE.
    alpha, power_target : float
        Significance and power.
    tau_grid : np.ndarray, optional
        Grid of candidate effect sizes (in absolute units). Defaults
        to ``linspace(0, 5 * std(residuals_B), 60)``.
    n_sims, n_trials, seed : int
        Monte Carlo control.

    Returns
    -------
    SPCDPowerAnalysis
    """

    residuals_B = np.asarray(residuals_B, dtype=float).ravel()
    rng = np.random.default_rng(seed)

    sigma = float(np.std(residuals_B)) + 1e-12

    if tau_grid is None:
        tau_grid = np.linspace(1e-6, 5.0 * sigma, 60)

    null_stats = _null_distribution(residuals_B, n_post, n_sims, rng)
    c_alpha = float(np.quantile(null_stats, 1.0 - alpha))

    safe_baseline = abs(baseline) if abs(baseline) > 1e-12 else 1.0

    mde_tau = float("nan")
    feasible = False
    for tau in tau_grid:
        hits = 0
        for _ in range(n_trials):
            noise = rng.normal(0.0, sigma, n_post)
            stat = float(np.mean(np.abs(tau + noise)))
            if stat >= c_alpha:
                hits += 1
        if hits / n_trials >= power_target:
            mde_tau = float(tau)
            feasible = True
            break

    mde_pct = (mde_tau / safe_baseline * 100.0) if feasible else float("nan")

    return SPCDPowerAnalysis(
        mde_tau=mde_tau,
        mde_pct=mde_pct,
        baseline=float(baseline),
        critical_stat=c_alpha,
        feasible=feasible,
        n_post=int(n_post),
        alpha=float(alpha),
        power_target=float(power_target),
        detectability=None,
    )


def compute_detectability_curve(
    residuals_B: np.ndarray,
    baseline: float,
    horizon_grid: List[int],
    alpha: float = 0.05,
    power_target: float = 0.8,
    n_sims: int = 5000,
    n_trials: int = 400,
    seed: int = 1400,
) -> Dict[int, float]:
    """Compute MDE as a function of post-treatment horizon length.

    Useful for answering "how long does the experiment need to run to
    detect a target effect?" before committing to a treatment window.

    Parameters
    ----------
    residuals_B : np.ndarray
        Out-of-sample holdout residuals.
    baseline : float
        Baseline level for percentage scaling.
    horizon_grid : list of int
        Horizons (number of post periods) at which to evaluate the MDE.
    alpha, power_target, n_sims, n_trials, seed : see :func:`compute_mde`.

    Returns
    -------
    dict
        Mapping ``horizon -> MDE in percent``. Infeasible entries are
        recorded as ``nan``.
    """

    curve: Dict[int, float] = {}
    for h in horizon_grid:
        result = compute_mde(
            residuals_B=residuals_B,
            baseline=baseline,
            n_post=int(h),
            alpha=alpha,
            power_target=power_target,
            n_sims=n_sims,
            n_trials=n_trials,
            seed=seed,
        )
        curve[int(h)] = result.mde_pct
    return curve
