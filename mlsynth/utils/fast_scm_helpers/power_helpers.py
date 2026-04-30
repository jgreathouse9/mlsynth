from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Literal
from math import comb

from .structure import SEDCandidate


# =========================================================
# TYPES
# =========================================================
PostImputation = Literal["mean", "max", "double_max"]
TestStatistic = Literal["mean_abs", "mean", "rms"]


# =========================================================
# TEST STATISTIC
# =========================================================
def test_statistic(e: np.ndarray, method: TestStatistic = "mean_abs") -> float:
    """
    Compute test statistic used in permutation inference.

    Parameters
    ----------
    e : np.ndarray
        Vector of residuals or treatment effects.
    method : {"mean_abs", "mean", "rms"}
        Statistic to compute.

    Returns
    -------
    float
        Scalar test statistic.
    """
    if method == "mean_abs":
        return float(np.mean(np.abs(e)))
    elif method == "mean":
        return float(np.mean(e))
    elif method == "rms":
        return float(np.sqrt(np.mean(e ** 2)))
    else:
        raise ValueError(f"Unknown statistic method: '{method}'")


# =========================================================
# NULL DISTRIBUTION
# =========================================================
def compute_null_distribution(
    full_series: np.ndarray,
    n_post: int,
    statistic: TestStatistic = "mean_abs",
    n_sims: int = 10000,
    seed: Optional[int] = 1400,
) -> np.ndarray:
    """
    Estimate null distribution via Monte Carlo permutation.

    Parameters
    ----------
    full_series : np.ndarray
        Residuals under null (pre + imputed post).
    n_post : int
        Size of pseudo post-treatment window.
    statistic : TestStatistic
        Test statistic to compute.
    n_sims : int
        Number of simulations.
    seed : int, optional
        RNG seed.

    Returns
    -------
    np.ndarray
        Sorted null statistics.
    """
    rng = np.random.default_rng(seed)
    n_total = len(full_series)

    stats = np.empty(n_sims)

    for i in range(n_sims):
        idx = rng.choice(n_total, size=n_post, replace=False)
        stats[i] = test_statistic(full_series[idx], method=statistic)

    stats.sort()
    return stats


def critical_value_from_null(null_stats: np.ndarray, alpha: float) -> float:
    """
    Compute (1 - alpha) quantile from null distribution.
    """
    return float(np.quantile(null_stats, 1 - alpha))


# =========================================================
# NOISE IMPUTATION
# =========================================================
def impute_noise_level(
    residuals_B: np.ndarray,
    method: PostImputation = "mean"
) -> float:
    """
    Estimate post-treatment noise level from pre-treatment residuals.
    """
    abs_r = np.abs(residuals_B)

    if method == "mean":
        return float(np.mean(abs_r))
    elif method == "max":
        return float(np.max(abs_r))
    elif method == "double_max":
        return float(2.0 * np.max(abs_r))
    else:
        raise ValueError(f"Unknown method: '{method}'")


# =========================================================
# MDE CORE
# =========================================================
def _analytical_mde(
    residuals_B: np.ndarray,
    synth_treated: np.ndarray,
    n_post: int,
    alpha: float = 0.05,
    noise_level: Optional[float] = None,
    statistic: TestStatistic = "mean_abs",
    n_sims: int = 10000,
    seed: Optional[int] = 1400,
) -> Dict:
    """
    Compute Minimum Detectable Effect (MDE).

    Returns
    -------
    dict with keys:
        mde_tau, mde_pct, baseline, critical_stat, feasible, p_value_lb
    """
    if noise_level is None:
        noise_level = impute_noise_level(residuals_B)

    n_B = len(residuals_B)

    baseline = float(np.mean(synth_treated[-n_post:])) if n_post > 0 else 1.0
    baseline = max(baseline, 1e-8)

    # Feasibility check
    n_perms_exact = comb(n_B + n_post, n_post)
    p_value_lb = 1.0 / n_perms_exact

    if p_value_lb >= alpha:
        return {
            "mde_tau": np.nan,
            "mde_pct": np.nan,
            "baseline": baseline,
            "critical_stat": np.nan,
            "n_perms_exact": n_perms_exact,
            "n_sims": n_sims,
            "p_value_lb": p_value_lb,
            "feasible": False,
        }

    # Build null
    post_null = np.full(n_post, noise_level)
    full_series = np.concatenate([residuals_B, post_null])

    null_stats = compute_null_distribution(
        full_series,
        n_post,
        statistic=statistic,
        n_sims=n_sims,
        seed=seed,
    )

    c_star = critical_value_from_null(null_stats, alpha)

    mde_tau = max(0.0, (c_star - noise_level) / baseline)

    return {
        "mde_tau": float(mde_tau),
        "mde_pct": float(100.0 * mde_tau),
        "baseline": baseline,
        "critical_stat": float(c_star),
        "n_perms_exact": n_perms_exact,
        "n_sims": n_sims,
        "p_value_lb": p_value_lb,
        "feasible": True,
    }


# =========================================================
# DETECTABILITY CURVE
# =========================================================
def compute_detectability_curve(
    candidate: SEDCandidate,
    n_post_grid: List[int],
    alpha: float = 0.05,
    n_sims: int = 10000,
    post_imputation: PostImputation = "mean",
    statistic: TestStatistic = "mean_abs",
    seed: Optional[int] = 1400,
) -> Dict:
    """
    Compute MDE curve over multiple horizons.
    """
    residuals_B = np.asarray(candidate.predictions.residuals_B)
    synth_treated = np.asarray(candidate.predictions.synthetic_treated)

    noise_level = impute_noise_level(residuals_B, post_imputation)

    curve: Dict[int, float] = {}
    details: Dict[int, Dict] = {}

    for n_post in n_post_grid:
        res = _analytical_mde(
            residuals_B,
            synth_treated,
            n_post,
            alpha,
            noise_level,
            statistic,
            n_sims,
            seed,
        )
        curve[n_post] = res["mde_tau"]
        details[n_post] = res

    return {
        "curve": curve,
        "details": details,
        "noise_level": noise_level,
    }


# =========================================================
# BATCH
# =========================================================
def run_mde_analysis(
    candidates: List[SEDCandidate],
    n_post_grid: Optional[List[int]] = None,
    **kwargs,
) -> List[SEDCandidate]:
    """
    Attach MDE results to candidates.
    """
    if n_post_grid is None:
        n_post_grid = list(range(2, 9))

    for c in candidates:
        c.mde_results = compute_detectability_curve(
            candidate=c,
            n_post_grid=n_post_grid,
            **kwargs,
        )

    return candidates


# =========================================================
# SUMMARY
# =========================================================
def mde_summary_table(candidates: List[SEDCandidate]) -> pd.DataFrame:
    """
    Build summary table of NMSE and MDEs.
    """
    rows = []

    for c in candidates:
        r = c.mde_results or {}
        d = r.get("details", {})

        row = {
            "tuple_id": getattr(c.identification, "tuple_id", "unknown"),
            "nmse_B": getattr(c.losses, "nmse_B", np.nan),
            "noise_level": r.get("noise_level", np.nan),
        }

        for w in range(2, 9):
            row[f"mde_{w}w"] = d.get(w, {}).get("mde_pct", np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("nmse_B").reset_index(drop=True)


# =========================================================
# SELECTION
# =========================================================
def select_best_tuple(
    candidates: List[SEDCandidate],
    relative_delta: float = 1.5,
    target: str = "early",
    max_shortlist: int = 5,
) -> Tuple[SEDCandidate, pd.DataFrame]:
    """
    Validity-first selection: filter by fit, then optimize power.
    """
    df = mde_summary_table(candidates)

    if df.empty:
        raise ValueError("No candidates")

    best_nmse = df["nmse_B"].min()
    threshold = best_nmse * relative_delta

    df = df[df["nmse_B"] <= threshold].copy()

    if df.empty:
        raise RuntimeError("No candidates pass fit threshold")

    if target == "early":
        cols = [f"mde_{w}w" for w in (2, 3, 4)]
        df["mde_score"] = df[cols].mean(axis=1)
    elif target.startswith("mde_"):
        df["mde_score"] = df[target]
    else:
        raise ValueError("Unknown target")

    df = df.sort_values("mde_score").reset_index(drop=True)

    df["recommended"] = False
    df.loc[0, "recommended"] = True

    winner_id = df.iloc[0]["tuple_id"]

    winner = next(
        c for c in candidates
        if getattr(c.identification, "tuple_id", None) == winner_id
    )

    return winner, df.head(max_shortlist)
