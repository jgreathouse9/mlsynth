from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Literal
from math import comb
from .structure import SEDCandidate


# =========================================================
# TEST STATISTIC
# =========================================================
def test_statistic(x: np.ndarray) -> float:
    """
    Compute the test statistic used in the permutation procedure.

    Parameters
    ----------
    x : np.ndarray
        Vector of treatment effect residuals.

    Returns
    -------
    float
        Mean absolute value of the input vector.
    """
    return float(np.mean(np.abs(x)))


# =========================================================
# NULL DISTRIBUTION
# =========================================================
def compute_null_distribution(full_series, n_post, n_sims=5000, seed=1400):
    """
    Estimate the null distribution of the test statistic via permutation.

    Parameters
    ----------
    full_series : array-like
        Combined residual series (pre + pseudo-post under null).
    n_post : int
        Number of observations sampled as pseudo post-treatment period.
    n_sims : int
        Number of Monte Carlo permutations.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Sorted simulated test statistics under the null hypothesis.
    """
    rng = np.random.default_rng(seed)
    full_series = np.asarray(full_series)

    null_stats = np.zeros(n_sims)

    for i in range(n_sims):
        idx = rng.choice(len(full_series), size=n_post, replace=False)
        null_stats[i] = np.mean(np.abs(full_series[idx]))

    return np.sort(null_stats)


def critical_value_from_null(null_stats, alpha):
    """
    Compute critical value from null distribution.

    Parameters
    ----------
    null_stats : np.ndarray
        Sorted null test statistics.
    alpha : float
        Significance level.

    Returns
    -------
    float
        (1 - alpha) quantile of null distribution.
    """
    return float(np.quantile(null_stats, 1 - alpha))


# =========================================================
# NOISE
# =========================================================
def impute_noise_level(residuals_B, method="mean"):
    """
    Estimate relative noise level from pre-treatment residuals.

    Parameters
    ----------
    residuals_B : array-like
        Pre-treatment residuals.
    method : {"mean", "max", "double_max"}
        Aggregation rule for noise estimation.

    Returns
    -------
    float
        Relative noise level (scale-free, percentage space compatible).
    """
    r = np.asarray(residuals_B)

    scale = np.mean(np.abs(r)) + 1e-8
    r = r / scale

    if method == "mean":
        return float(np.mean(np.abs(r)))
    if method == "max":
        return float(np.max(np.abs(r)))
    if method == "double_max":
        return float(2 * np.max(np.abs(r)))

    raise ValueError()


# =========================================================
# MDE CORE (FIXED)
# =========================================================
def _analytical_mde(residuals_B,
                    synth_treated,
                    n_post,
                    alpha=0.05,
                    n_sims=5000,
                    seed=1400):
    """
    Compute Minimum Detectable Effect (MDE) in percentage space.

    This function simulates power across effect sizes and returns the
    smallest detectable percentage effect at 80% power.

    Parameters
    ----------
    residuals_B : array-like
        Pre-treatment residuals.
    synth_treated : array-like
        Synthetic treated outcome series.
    n_post : int
        Post-treatment window length.
    alpha : float
        Significance level.
    n_sims : int
        Number of Monte Carlo simulations for null.
    seed : int
        Random seed.

    Returns
    -------
    dict
        MDE result including percentage effect size and feasibility flag.
    """
    rng = np.random.default_rng(seed)

    residuals_B = np.asarray(residuals_B)
    synth_treated = np.asarray(synth_treated)

    baseline = np.mean(synth_treated[-n_post:])
    baseline = max(baseline, 1e-8)

    rel_B = residuals_B / baseline

    # --- FIX: null must include same structure as test statistic space
    full_null = np.concatenate([
        rel_B,
        rng.normal(0, np.std(rel_B) + 1e-8, n_post)
    ])

    null_stats = compute_null_distribution(full_null, n_post, n_sims, seed)
    c_star = critical_value_from_null(null_stats, alpha)

    tau_grid = np.linspace(0.001, 0.10, 60)

    for tau in tau_grid:

        hits = 0

        for _ in range(300):
            noise = rng.normal(0, np.std(rel_B) + 1e-8, n_post)

            post = tau + noise  # percent effect space

            stat = np.mean(np.abs(post))

            if stat >= c_star:
                hits += 1

        power = hits / 300

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
def compute_detectability_curve(candidate,
                                n_post_grid,
                                alpha=0.05,
                                n_sims=5000,
                                seed=1400):
    """
    Compute MDE curve across different post-treatment horizons.

    Parameters
    ----------
    candidate : SEDCandidate
        Synthetic experiment candidate.
    n_post_grid : list[int]
        List of post-treatment lengths to evaluate.
    alpha : float
        Significance level.
    n_sims : int
        Number of Monte Carlo simulations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Curve of MDE (%) by horizon and summary thresholds.
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
# BATCH
# =========================================================
def run_mde_analysis(candidates,
                     n_post_grid=None,
                     alpha=0.05,
                     n_sims=5000,
                     seed=1400):
    """
    Run MDE analysis over multiple candidates.

    Parameters
    ----------
    candidates : list[SEDCandidate]
        Candidate synthetic designs.
    n_post_grid : list[int]
        Post-treatment horizons.
    alpha : float
        Significance level.
    n_sims : int
        Monte Carlo simulations.
    seed : int
        Random seed.

    Returns
    -------
    list[SEDCandidate]
        Candidates with attached MDE results.
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
# SUMMARY
# =========================================================
def mde_summary_table(candidates):
    """
    Build summary table of NMSE and MDE curves.

    Parameters
    ----------
    candidates : list[SEDCandidate]

    Returns
    -------
    pd.DataFrame
        Summary table sorted by pre-treatment fit quality.
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



def _dominates(a_nmse, a_mde, b_nmse, b_mde):
    return (a_nmse <= b_nmse and a_mde <= b_mde) and (a_nmse < b_nmse or a_mde < b_mde)

def select_best_tuple(
    candidates,
    n_post_aggregation=(2, 3, 4),
    max_shortlist=5,
    mde_mode="early_min"
):
    df = mde_summary_table(candidates).copy()

    if df.empty:
        raise ValueError("No candidates")

    # -------------------------------------------------
    # MDE LOSS DEFINITION
    # -------------------------------------------------
    if mde_mode == "early_mean":
        cols = [f"mde_{w}w" for w in n_post_aggregation]
        df["mde_score"] = df[cols].mean(axis=1)

    elif mde_mode == "early_min":
        cols = [f"mde_{w}w" for w in n_post_aggregation]
        df["mde_score"] = df[cols].min(axis=1)

    elif mde_mode == "late":
        # default: most stable / realistic detectability horizon
        df["mde_score"] = df["mde_8w"]

    else:
        raise ValueError("Unknown mde_mode")

    df = df.dropna(subset=["mde_score", "nmse_B"]).reset_index(drop=True)

    # -------------------------------------------------
    # PARETO FRONTIER
    # -------------------------------------------------
    def dominates(a_nmse, a_mde, b_nmse, b_mde):
        return (
            a_nmse <= b_nmse and a_mde <= b_mde
        ) and (
            a_nmse < b_nmse or a_mde < b_mde
        )

    pareto_mask = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue

            if dominates(
                df.loc[j, "nmse_B"],
                df.loc[j, "mde_score"],
                df.loc[i, "nmse_B"],
                df.loc[i, "mde_score"]
            ):
                pareto_mask[i] = False
                break

    pareto_df = df[pareto_mask].copy()

    # -------------------------------------------------
    # TRADE-OFF SCORING INSIDE PARETO SET
    # -------------------------------------------------
    # normalize so both axes are comparable
    pareto_df["score"] = (
        pareto_df["nmse_B"] / (pareto_df["nmse_B"].max() + 1e-8)
        + pareto_df["mde_score"] / (pareto_df["mde_score"].max() + 1e-8)
    )

    pareto_df = pareto_df.sort_values("score").reset_index(drop=True)

    pareto_df["recommended"] = False
    pareto_df.loc[0, "recommended"] = True

    winner_id = pareto_df.iloc[0]["tuple_id"]

    winner = next(
        c for c in candidates
        if getattr(c.identification, "tuple_id", None) == winner_id
    )


    return winner, pareto_df.head(max_shortlist)        synthetic control placebo tests and is invariant to the sign of the effect.
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
# NULL DISTRIBUTION (Monte Carlo)
# =========================================================
def compute_null_distribution(
        full_series: np.ndarray,
        n_post: int,
        n_sims: int = 100000,
        seed: Optional[int] = 1400
) -> np.ndarray:
    """
        Estimate the null distribution of the test statistic using Monte Carlo permutations.

        Under the sharp null of no treatment effect, we randomly re-assign which periods
        are labeled as "post-treatment" and compute the test statistic.

        Parameters
        ----------
        full_series : np.ndarray
            Combined series of residuals under the null (blank period + imputed post noise).
        n_post : int
            Number of periods assigned to the pseudo post-treatment window.
        n_sims : int, default=100000
            Number of Monte Carlo simulations.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Sorted array of simulated test statistics under the null.
    """
    rng = np.random.default_rng(seed)
    n_total = len(full_series)
    null_stats = np.zeros(n_sims)

    for i in range(n_sims):
        idx = rng.choice(n_total, size=n_post, replace=False)
        null_stats[i] = test_statistic(full_series[idx])

    null_stats.sort()
    return null_stats


def critical_value_from_null(null_stats: np.ndarray, alpha: float) -> float:
    """
        Compute the critical value from the null distribution for a one-sided test.

        Parameters
        ----------
        null_stats : np.ndarray
            Sorted array of test statistics under the null hypothesis.
        alpha : float
            Significance level (e.g., 0.05).

        Returns
        -------
        float
            The (1 - alpha) quantile of the null distribution.
    """
    return float(np.quantile(null_stats, 1 - alpha))


# =========================================================
# NOISE IMPUTATION
# =========================================================
PostImputation = Literal["mean", "max", "double_max"]


def impute_noise_level(residuals_B: np.ndarray, method: PostImputation = "mean") -> float:
    """
        Estimate the magnitude of post-treatment noise from residuals in the blank period.

        This is a key step in prospective power analysis, as we have not yet observed
        real post-treatment outcomes.

        Parameters
        ----------
        residuals_B : np.ndarray
            Residuals (effects) from the blank/validation pre-treatment period.
        method : {"mean", "max", "double_max"}, default="mean"
            Method for imputing noise level:
            - "mean": Average absolute residual (standard choice).
            - "max": Most extreme observed residual (conservative).
            - "double_max": Twice the maximum residual (very conservative).

        Returns
        -------
        float
            Estimated noise level to be used for post-treatment periods under the null.
    """
    abs_resid = np.abs(residuals_B)
    if method == "mean":
        return float(np.mean(abs_resid))
    elif method == "max":
        return float(np.max(abs_resid))
    elif method == "double_max":
        return float(2.0 * np.max(abs_resid))
    else:
        raise ValueError(f"Unknown post_imputation method: '{method}'")


# =========================================================
# ANALYTICAL MDE
# =========================================================
def _analytical_mde(
        residuals_B: np.ndarray,
        synth_treated: np.ndarray,
        n_post: int,
        alpha: float = 0.05,
        noise_level: float = None,
        statistic: str = "mean_abs",
        n_sims: int = 100000,
        seed: Optional[int] = 1400
) -> Dict:
    """
        Compute the Minimum Detectable Effect (MDE) for a fixed post-treatment length.

        This function inverts the permutation test to find the smallest constant
        treatment effect tau that would be statistically detectable at level alpha.

        Parameters
        ----------
        residuals_B : np.ndarray
            Residuals from the blank/validation period.
        synth_treated : np.ndarray
            Synthetic treated unit time series.
        n_post : int
            Length of the hypothetical post-treatment period.
        alpha : float, default=0.05
            Significance level.
        noise_level : float, optional
            Pre-computed noise level. If None, it will be imputed.
        statistic : str, default="mean_abs"
            Test statistic to use.
        n_sims : int, default=100000
            Number of Monte Carlo simulations for the null distribution.
        seed : int, optional
            Random seed.

        Returns
        -------
        dict
            Dictionary containing:
            - mde_tau : Minimum detectable effect (in original units)
            - mde_pct : MDE as percentage of baseline
            - baseline : Mean level of synthetic treated in post period
            - critical_stat : Critical value from null distribution
            - feasible : Whether inference is possible at given alpha
            - p_value_lb : Theoretical lower bound on p-value
    """
    if noise_level is None:
        noise_level = impute_noise_level(residuals_B, "mean")

    n_B = len(residuals_B)
    baseline = float(np.mean(synth_treated[-n_post:])) if n_post > 0 else 1.0
    baseline = max(baseline, 1e-8)  # avoid division by zero

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
            "feasible": False
        }

    # Build null series
    post_null = np.full(n_post, noise_level)
    full_series_null = np.concatenate([residuals_B, post_null])

    # Monte Carlo null distribution
    null_stats = compute_null_distribution(full_series_null, n_post, n_sims, seed)
    c_star = critical_value_from_null(null_stats, alpha)

    # Solve for MDE
    mde_tau = max(0.0, (c_star - noise_level) / baseline)

    return {
        "mde_tau": float(mde_tau),
        "mde_pct": float(100.0 * mde_tau),
        "baseline": baseline,
        "critical_stat": float(c_star),
        "n_perms_exact": n_perms_exact,
        "n_sims": n_sims,
        "p_value_lb": p_value_lb,
        "feasible": True
    }


# =========================================================
# DETECTABILITY CURVE
# =========================================================
def compute_detectability_curve(
        candidate: SEDCandidate,
        n_post_grid: List[int],
        alpha: float = 0.05,
        n_sims: int = 100000,
        post_imputation: PostImputation = "mean",
        statistic: str = "mean_abs",
        seed: Optional[int] = 1400
) -> Dict:
    """
        Compute the detectability curve (MDE as a function of post-treatment duration).

        For each value in `n_post_grid`, this function calculates the smallest treatment
        effect that could be detected with statistical significance α.

        Parameters
        ----------
        candidate : SEDCandidate
            A single evaluated synthetic experimental design candidate.
        n_post_grid : list of int
            List of post-treatment lengths to evaluate (e.g., [2, 3, 4, 6, 8]).
        alpha : float, default=0.05
            Significance level.
        n_sims : int, default=100000
            Number of Monte Carlo simulations per MDE calculation.
        post_imputation : {"mean", "max", "double_max"}, default="mean"
            Method for imputing post-treatment noise from blank period residuals.
        statistic : str, default="mean_abs"
            Test statistic used in the permutation test.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict
            Contains:
            - "curve": dict mapping n_post to mde_tau
            - "details": detailed results per horizon
            - "n_post_10pct": smallest n_post with MDE ≤ 10%
            - "n_post_5pct": smallest n_post with MDE ≤ 5%
            - "noise_level": imputed noise level used
    """
    residuals_B = np.asarray(candidate.predictions.residuals_B)
    synth_treated = np.asarray(candidate.predictions.synthetic_treated)

    noise_level = impute_noise_level(residuals_B, post_imputation)

    curve: Dict[int, float] = {}
    details: Dict[int, Dict] = {}

    for n_post in n_post_grid:
        res = _analytical_mde(
            residuals_B=residuals_B,
            synth_treated=synth_treated,
            n_post=n_post,
            alpha=alpha,
            noise_level=noise_level,
            statistic=statistic,
            n_sims=n_sims,
            seed=seed
        )
        curve[n_post] = res["mde_tau"]
        details[n_post] = res

    # Find smallest n_post achieving certain MDE thresholds

    return {
        "curve": curve,
        "details": details,
        "noise_level": noise_level,
        "post_imputation": post_imputation
    }


# =========================================================
# BATCH PROCESSING
# =========================================================
def run_mde_analysis(
        candidates: List[SEDCandidate],
        n_post_grid: Optional[List[int]] = None,
        alpha: float = 0.05,
        n_sims: int = 10000,
        post_imputation: PostImputation = "mean",
        statistic: str = "mean_abs",
        seed: Optional[int] = 1400
) -> List[SEDCandidate]:
    """
        Run power analysis (MDE computation) on a list of candidate designs.

        Mutates each candidate by attaching `.mde_results`.

        Parameters
        ----------
        candidates : list of SEDCandidate
            List of candidates after synthetic control weights have been computed.
        n_post_grid : list of int, optional
            Post-treatment horizons to evaluate. Default is [2, 3, 4, 5, 6, 7, 8].
        alpha, n_sims, post_imputation, statistic, seed
            Passed to `compute_detectability_curve`.

        Returns
        -------
        list of SEDCandidate
            The same list with `.mde_results` populated on each candidate.
    """
    if n_post_grid is None:
        n_post_grid = list(range(2, 9))  # 2 to 8 post periods

    for cand in candidates:
        cand.mde_results = compute_detectability_curve(
            candidate=cand,
            n_post_grid=n_post_grid,
            alpha=alpha,
            n_sims=n_sims,
            post_imputation=post_imputation,
            statistic=statistic,
            seed=seed
        )

    return candidates


# =========================================================
# SELECTION & SUMMARY
# =========================================================
def mde_summary_table(candidates: List[SEDCandidate]) -> pd.DataFrame:
    """
        Create a clean summary table of blank-period fit and Minimum Detectable Effects
        across all candidate synthetic experimental designs.

        This table is useful for comparing designs and feeding into the final selection step.

        Parameters
        ----------
        candidates : list of SEDCandidate
            List of candidates after power analysis (`run_mde_analysis`) has been run.
            Each candidate must have `.mde_results` and `.losses.nmse_B`.

        Returns
        -------
        pd.DataFrame
            Summary table with columns:
            - tuple_id: Identifier of the treated unit subset
            - nmse_B: Normalized MSE in the blank/validation period (lower = better fit)
            - noise_level: Imputed post-treatment noise level
            - mde_2w, mde_3w, ..., mde_8w: MDE percentages for each post-treatment horizon
            - (sorted by nmse_B ascending)

        Notes
        -----
        This table is primarily used as input to `select_best_tuple()`.
        Lower `nmse_B` indicates better pre-treatment validity (less risk of overfitting).
    """
    rows = []
    for cand in candidates:
        r = cand.mde_results or {}
        details = r.get("details", {})
        row = {
            "tuple_id": getattr(cand.identification, "tuple_id", "unknown"),
            "nmse_B": getattr(cand.losses, "nmse_B", np.nan),
            "noise_level": r.get("noise_level", np.nan),
        }
        for week in range(2, 9):
            key = f"mde_{week}w"
            row[key] = details.get(week, {}).get("mde_pct", np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("nmse_B").reset_index(drop=True) if not df.empty else df


def select_best_tuple(
        candidates: List[SEDCandidate],
        delta: float = 0.015,
        relative_delta: Optional[float] = 1.5,
        target_mde_horizon: str = "early_mde_avg",
        return_shortlist: bool = True,
        max_shortlist: int = 5
) -> Tuple[SEDCandidate, pd.DataFrame]:
    """
        Select the best synthetic experimental design using a lexicographic (hierarchical) rule.

        This implements a "validity-first, then power" philosophy that is highly recommended
        for marketing science and policy experiments:

        1. Filter candidates to those with sufficiently good blank-period fit (nmse_B).
        2. Among the qualifying candidates, select the one with the best statistical power
           (lowest Minimum Detectable Effect).

        Parameters
        ----------
        candidates : list of SEDCandidate
            List of candidates after `run_mde_analysis()` has been called.
        delta : float, default=0.015
            Absolute tolerance on nmse_B (used only if `relative_delta` is None).
        relative_delta : float, default=1.5
            Relative tolerance multiplier. Allows candidates whose nmse_B is up to
            `relative_delta` times the best observed nmse_B.
            Recommended default (1.5) means "up to 50% worse fit than the best".
        target_mde_horizon : str, default="early_mde_avg"
            Which MDE column to optimize for. Common choices:
            - "early_mde_avg": Average MDE over weeks 2-4 (recommended for marketing)
            - "mde_4w", "mde_6w", "mde_8w": Specific horizons
        return_shortlist : bool, default=True
            Whether to return a shortlist of top qualifying candidates.
        max_shortlist : int, default=5
            Maximum number of candidates to include in the shortlist.

        Returns
        -------
        winner : SEDCandidate
            The recommended best candidate (with full inference results).
        shortlist : pd.DataFrame
            DataFrame containing the top qualifying candidates (with fit and MDE columns).
            Includes a 'recommended' column marking the winner.

        Notes
        -----
        This selection rule prioritizes **statistical validity** (good pre-treatment fit
        in the blank period) before optimizing for **statistical power**.

        Using `relative_delta=1.5` is a practical and defensible default in marketing
        lift testing: it keeps only designs that are reasonably well-matched while still
        allowing some flexibility to choose the most powerful option.
    """
    df = mde_summary_table(candidates).copy()

    if df.empty:
        raise ValueError("No candidates to select from")

    best_nmse = df['nmse_B'].min()

    if relative_delta is not None:
        threshold = best_nmse * relative_delta
        threshold_desc = f"{relative_delta:.1f}x best fit"
    else:
        threshold = best_nmse + delta
        threshold_desc = f"best + {delta:.4f}"

    passing = df[df['nmse_B'] <= threshold].copy()

    if passing.empty:
        print(f"Warning: No candidates met fit threshold. Using best fit only.")
        passing = df.iloc[[0]].copy()

    # Compute early MDE average if needed
    if target_mde_horizon == "early_mde_avg":
        early_cols = [f"mde_{w}w" for w in range(2, 5) if f"mde_{w}w" in passing.columns]
        passing['early_mde_avg'] = passing[early_cols].mean(axis=1)

    # Sort by power (lower MDE = better)
    passing = passing.sort_values(target_mde_horizon).reset_index(drop=True)

    # Mark winner
    passing['recommended'] = False
    passing.loc[0, 'recommended'] = True

    winner_tuple_id = passing.iloc[0]['tuple_id']
    winner = next(c for c in candidates if getattr(c.identification, "tuple_id", None) == winner_tuple_id)

    print(f"Recommended tuple: {winner_tuple_id}")
    print(f"   Blank-period NMSE_B : {passing.iloc[0]['nmse_B']:.4f} "
          f"(threshold = {threshold:.4f} using {threshold_desc})")
    print(f"   Early MDE           : {passing.iloc[0][target_mde_horizon]:.2f}%")

    return winner, passing.head(max_shortlist) if return_shortlist else passing.head(1)
