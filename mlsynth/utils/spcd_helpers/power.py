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

from dataclasses import dataclass, field, replace
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


def compute_pooled_average_mde(
    residuals_by_arm: Dict[object, np.ndarray],
    baselines_by_arm: Dict[object, float],
    sizes_by_arm: Dict[object, int],
    n_post: int,
    weights: str = "size",
    alpha: float = 0.05,
    power_target: float = 0.8,
    n_sims: int = 5000,
    n_trials: int = 400,
    seed: int = 1400,
    horizon_grid: Optional[List[int]] = None,
) -> SPCDPowerAnalysis:
    """MDE for the (size- or equal-weighted) **average** effect across arms.

    Forms the pooled contrast series ``g_t = sum_a w_a * r_B^(a)_t`` on
    **time-aligned** holdout residuals, then runs the ordinary
    single-series :func:`compute_mde` on ``g``. Because the arms share
    calendar time, summing the aligned series makes the cross-arm
    correlation enter through ``Var(g) = w' Sigma w`` automatically -- so
    one must pool the aligned *series*, never resample arms
    independently, or positive correlation is dropped and the MDE comes
    out over-optimistic.

    This answers *"what average effect across arms can we detect?"* and is
    the most powerful pooled question (averaging cancels idiosyncratic
    noise). Its estimand is the weighted-average effect, so heterogeneous,
    **opposite-signed** arm effects can cancel and be missed; use the
    per-arm analyses when individual-arm detection matters.

    Parameters
    ----------
    residuals_by_arm : dict
        ``{arm_label: r_B}`` out-of-sample holdout residual series. The
        series are truncated to the common (minimum) length and aligned
        from the start, which is correct when arms share the calendar
        time index.
    baselines_by_arm : dict
        ``{arm_label: baseline}`` per-arm holdout baselines; pooled by the
        same weights for the percentage scaling.
    sizes_by_arm : dict
        ``{arm_label: n_units}`` used for ``weights="size"``.
    n_post : int
        Post-treatment horizon for the MDE.
    weights : {"size", "equal"}
        ``"size"`` weights each arm by its unit count (the
        population-average effect); ``"equal"`` weights arms equally.
    horizon_grid : list of int, optional
        If given, also computes the pooled-average MDE at each
        post-treatment horizon and attaches it as ``detectability``
        (``{horizon -> MDE percent}``). This answers *"how long must the
        study run to detect a target average effect?"*.
    alpha, power_target, n_sims, n_trials, seed : see :func:`compute_mde`.

    Returns
    -------
    SPCDPowerAnalysis
        The MDE of the pooled average-effect contrast (with a
        ``detectability`` curve when ``horizon_grid`` is supplied).
    """

    labels = sorted(residuals_by_arm, key=str)
    if len(labels) < 2:
        raise ValueError("Pooled average MDE requires at least two arms.")

    n_B = min(np.asarray(residuals_by_arm[l]).ravel().size for l in labels)
    if n_B < 1:
        raise ValueError("Holdout residual series are empty; cannot pool.")
    R = np.column_stack(
        [np.asarray(residuals_by_arm[l], dtype=float).ravel()[:n_B] for l in labels]
    )

    if weights == "equal":
        w = np.full(len(labels), 1.0 / len(labels))
    elif weights == "size":
        sizes = np.array([float(sizes_by_arm[l]) for l in labels])
        w = sizes / sizes.sum()
    else:
        raise ValueError(
            f"Unknown pooled weights mode '{weights}'. Use 'size' or 'equal'."
        )

    pooled_residuals = R @ w
    pooled_baseline = float(
        sum(w[i] * float(baselines_by_arm[labels[i]]) for i in range(len(labels)))
    )

    result = compute_mde(
        residuals_B=pooled_residuals,
        baseline=pooled_baseline,
        n_post=int(n_post),
        alpha=alpha,
        power_target=power_target,
        n_sims=n_sims,
        n_trials=n_trials,
        seed=seed,
    )

    if horizon_grid:
        curve = compute_detectability_curve(
            residuals_B=pooled_residuals,
            baseline=pooled_baseline,
            horizon_grid=list(horizon_grid),
            alpha=alpha,
            power_target=power_target,
            n_sims=n_sims,
            n_trials=n_trials,
            seed=seed,
        )
        result = replace(result, detectability=curve)

    return result


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
