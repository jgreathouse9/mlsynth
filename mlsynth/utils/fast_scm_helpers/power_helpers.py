from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from .structure import SEDCandidate


# =========================================================
# TEST STATISTIC
# =========================================================
def test_statistic(x: np.ndarray) -> float:
    """
    Mean absolute treatment effect statistic.

    This is the core detection statistic used throughout the
    permutation test and power analysis.

    Parameters
    ----------
    x : np.ndarray
        Treatment effect series.

    Returns
    -------
    float
        Mean absolute effect.
    """
    return float(np.mean(np.abs(x)))


# =========================================================
# NULL DISTRIBUTION
# =========================================================
def compute_null_distribution(full_series, n_post, n_sims=5000, seed=1400):
    """
    Monte Carlo approximation of the null distribution of the test statistic.

    Parameters
    ----------
    full_series : array-like
        Residual pool (pre + pseudo-post under null).
    n_post : int
        Length of pseudo post-treatment window.
    n_sims : int
        Number of permutations.
    seed : int

    Returns
    -------
    np.ndarray
        Sorted null statistics.
    """
    rng = np.random.default_rng(seed)
    full_series = np.asarray(full_series)

    stats = np.zeros(n_sims)

    for i in range(n_sims):
        idx = rng.choice(len(full_series), size=n_post, replace=False)
        stats[i] = np.mean(np.abs(full_series[idx]))

    return np.sort(stats)


def critical_value_from_null(null_stats: np.ndarray, alpha: float) -> float:
    """
    Compute critical value from null distribution.

    Parameters
    ----------
    null_stats : np.ndarray
        Sorted null distribution of test statistics.
    alpha : float
        Significance level.

    Returns
    -------
    float
        (1 - alpha) quantile of the null distribution.
    """
    return float(np.quantile(null_stats, 1 - alpha))


# =========================================================
# MDE CORE
# =========================================================
def _analytical_mde(
    residuals_B,
    synth_treated,
    n_post,
    alpha=0.05,
    n_sims=5000,
    seed=1400
):
    """
    Compute minimum detectable effect (MDE) in relative percentage terms.

    The MDE is defined as the smallest treatment effect tau such that:

        P(T(τ) > c_alpha) ≥ 0.8

    where:
        - T is the test statistic (mean absolute effect)
        - c_alpha is the (1 - alpha) critical value of the null distribution

    Parameters
    ----------
    residuals_B : np.ndarray
        Pre-treatment residuals (baseline noise structure).
    synth_treated : np.ndarray
        Synthetic treated unit series.
    n_post : int
        Post-treatment horizon length.
    alpha : float, default=0.05
        Significance level.
    n_sims : int, default=5000
        Monte Carlo draws for null distribution.
    seed : int, default=1400
        Random seed.

    Returns
    -------
    dict
        Dictionary containing:
        - mde_tau : float
            Minimum detectable effect (absolute scale)
        - mde_pct : float
            Percentage-scale MDE
        - baseline : float
            Baseline level of outcome
        - critical_stat : float
            Critical value of test statistic
        - feasible : bool
            Whether MDE was identified within grid
    """
    rng = np.random.default_rng(seed)

    residuals_B = np.asarray(residuals_B)
    synth_treated = np.asarray(synth_treated)

    baseline = np.mean(synth_treated[-n_post:])
    baseline = max(baseline, 1e-8)

    rel_B = residuals_B / baseline

    full_null = np.concatenate([
        rel_B,
        rng.normal(0, np.std(rel_B) + 1e-8, n_post)
    ])

    null_stats = compute_null_distribution(full_null, n_post, n_sims, seed)
    c_star = critical_value_from_null(null_stats, alpha)

    tau_grid = np.linspace(0.001, 0.10, 60)

    for tau in tau_grid:

        hits = 0

        for _ in range(400):
            noise = rng.normal(0, np.std(rel_B) + 1e-8, n_post)
            post = tau + noise

            if np.mean(np.abs(post)) >= c_star:
                hits += 1

        power = hits / 400

        if power >= 0.8:
            return {
                "mde_tau": float(tau),
                "mde_pct": float(100 * tau),
                "baseline": baseline,
                "critical_stat": float(c_star),
                "feasible": True
            }

    return {
        "mde_tau": np.nan,
        "mde_pct": np.nan,
        "baseline": baseline,
        "critical_stat": float(c_star),
        "feasible": False
    }


# =========================================================
# DETECTABILITY CURVE
# =========================================================
def compute_detectability_curve(
    candidate,
    n_post_grid,
    alpha=0.05,
    n_sims=5000,
    seed=1400
):
    """
    Compute detectability curve mapping horizon length to MDE.

    This function evaluates how statistical power evolves with the length
    of the post-treatment window.

    Parameters
    ----------
    candidate : SEDCandidate
        Candidate experimental design.
    n_post_grid : list of int
        Grid of post-treatment horizons.
    alpha : float, default=0.05
        Significance level.
    n_sims : int, default=5000
        Monte Carlo simulations.
    seed : int, default=1400
        Random seed.

    Returns
    -------
    dict
        Dictionary containing:
        - curve : dict
            Mapping horizon → MDE (%)
        - details : dict
            Full MDE metadata per horizon
        - n_post_10pct : int or nan
            First horizon achieving ≤ 10% MDE
        - n_post_5pct : int or nan
            First horizon achieving ≤ 5% MDE
    """
    residuals_B = np.asarray(candidate.predictions.residuals_B)
    synth_treated = np.asarray(candidate.predictions.synthetic_treated)

    curve = {}
    details = {}

    for n_post in n_post_grid:

        res = _analytical_mde(
            residuals_B,
            synth_treated,
            n_post,
            alpha,
            n_sims,
            seed
        )

        curve[n_post] = res["mde_pct"]
        details[n_post] = res

    return {
        "curve": curve,
        "details": details,
        "n_post_10pct": next((k for k, v in curve.items() if v <= 10), np.nan),
        "n_post_5pct": next((k for k, v in curve.items() if v <= 5), np.nan),
    }


# =========================================================
# BATCH ANALYSIS
# =========================================================
def run_mde_analysis(
    candidates: List[SEDCandidate],
    n_post_grid=None,
    alpha=0.05,
    n_sims=5000,
    seed=1400
):
    """
    Compute detectability curves for a set of candidate designs.

    This function mutates each candidate in-place by attaching MDE results.

    Parameters
    ----------
    candidates : list of SEDCandidate
        Candidate designs.
    n_post_grid : list[int], optional
        Grid of horizons. If None, defaults to range(2, 9).
    alpha : float, default=0.05
        Significance level.
    n_sims : int, default=5000
        Monte Carlo simulations.
    seed : int, default=1400
        Random seed.

    Returns
    -------
    list of SEDCandidate
        Updated candidates with `.mde_results` populated.
    """
    if n_post_grid is None:
        n_post_grid = list(range(2, 9))

    for c in candidates:
        c.mde_results = compute_detectability_curve(
            c,
            n_post_grid,
            alpha,
            n_sims,
            seed
        )

    return candidates


# =========================================================
# SUMMARY TABLE
# =========================================================
def mde_summary_table(candidates):
    """
    Flatten candidate detectability results into a structured DataFrame.

    Parameters
    ----------
    candidates : list of SEDCandidate
        Evaluated candidates with MDE results.

    Returns
    -------
    pd.DataFrame
        Summary table with:
        - tuple_id
        - nmse_B
        - mde_{w}w columns for w in [2,...,8]
    """
    rows = []

    for c in candidates:

        r = c.mde_results or {}
        d = r.get("details", {})

        row = {
            "tuple_id": getattr(c.identification, "tuple_id", "unknown"),
            "nmse_B": getattr(c.losses, "nmse_B", np.nan),
        }

        for w in range(2, 9):
            row[f"mde_{w}w"] = d.get(w, {}).get("mde_pct", np.nan)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("nmse_B").reset_index(drop=True)


# =========================================================
# PARETO DOMINANCE
# =========================================================
def _dominates(a_nmse, a_mde, b_nmse, b_mde):
    """
    Pareto dominance comparison.

    A dominates B if it is no worse in both objectives and strictly better
    in at least one.

    Parameters
    ----------
    a_nmse : float
        NMSE of candidate A.
    a_mde : float
        MDE of candidate A.
    b_nmse : float
        NMSE of candidate B.
    b_mde : float
        MDE of candidate B.

    Returns
    -------
    bool
        True if A dominates B.
    """
    return (
        (a_nmse <= b_nmse and a_mde <= b_mde)
        and (a_nmse < b_nmse or a_mde < b_mde)
    )


# =========================================================
# FINAL DESIGN SELECTION (PARETO)
# =========================================================


def select_best_tuple(
    candidates,
    mde_horizon="late",
    n_post_aggregation=(2, 3, 4),
    max_shortlist=5
):
    """
    Select optimal experimental design via Pareto frontier.

    The selection balances:
        (1) Pre-treatment fit (NMSE_B)
        (2) Detectability (MDE)

    The function constructs a Pareto frontier and applies a scalar
    tie-breaking score.

    Parameters
    ----------
    candidates : list of SEDCandidate
        Candidate designs with MDE results.
    mde_horizon : {"early_mean", "early_min", "late"}, default="late"
        Defines how MDE is aggregated across horizons.
    n_post_aggregation : tuple of int, default=(2,3,4)
        Horizons used for early aggregation rules.
    max_shortlist : int, default=5
        Number of top candidates returned for inspection.

    Returns
    -------
    winner : SEDCandidate
        Selected optimal design.
    shortlist : pd.DataFrame
        Ranked Pareto subset of candidates.
    """
    df = mde_summary_table(candidates).copy()

    if df.empty:
        raise ValueError("No candidates")

    # -----------------------------
    # DEFINE MDE OBJECTIVE
    # -----------------------------
    if mde_horizon == "early_mean":
        cols = [f"mde_{w}w" for w in n_post_aggregation]
        df["mde_score"] = df[cols].mean(axis=1)

    elif mde_horizon == "early_min":
        cols = [f"mde_{w}w" for w in n_post_aggregation]
        df["mde_score"] = df[cols].min(axis=1)

    elif mde_horizon == "late":
        df["mde_score"] = df["mde_8w"]

    else:
        raise ValueError("Unknown mde_horizon")

    df = df.dropna(subset=["mde_score", "nmse_B"]).reset_index(drop=True)

    # -----------------------------
    # PARETO FRONTIER
    # -----------------------------
    pareto = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue

            if _dominates(
                df.loc[j, "nmse_B"],
                df.loc[j, "mde_score"],
                df.loc[i, "nmse_B"],
                df.loc[i, "mde_score"]
            ):
                pareto[i] = False
                break

    pareto_df = df[pareto].copy()

    # -----------------------------
    # TIE-BREAK
    # -----------------------------
    pareto_df["score"] = (
        pareto_df["nmse_B"] / pareto_df["nmse_B"].max()
        + pareto_df["mde_score"] / pareto_df["mde_score"].max()
    )

    pareto_df = pareto_df.sort_values("score").reset_index(drop=True)
    pareto_df["recommended"] = False
    pareto_df.loc[0, "recommended"] = True

    winner_id = pareto_df.iloc[0]["tuple_id"]

    winner = next(
        c for c in candidates
        if getattr(c.identification, "tuple_id", None) == winner_id
    )


    return winner, pareto_df.head(max_shortlist)
