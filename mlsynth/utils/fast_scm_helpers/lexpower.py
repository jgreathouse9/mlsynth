"""Power / minimum-detectable-effect (MDE) analysis for LEXSCM designs.

Rebuilt to be a *consistent* nonparametric permutation analysis, replacing the
earlier version whose null was quasi-empirical while its alternative was a pure
Gaussian shift (an internally inconsistent power calc), and whose effect was
normalised by ``mean(synth_treated[-n_post:])`` floored at 1e-8 (which exploded
on zero-mean outcomes).

Design
------
* **One residual model for null and alternative.** Both are built by
  moving-block resampling the *placebo* residuals (the held-out blank-period
  gaps treated - control, which are pure noise under H0), so serial dependence
  is preserved -- matching Abadie & Zhao's time-series-robust inference.
* **MDE in standard-deviation units** (effect size), the scale used by
  Vives-i-Bastida ("detect effects > 0.1 s.d."), plus the absolute outcome-unit
  value.  No division by a fragile level, so zero-mean outcomes are fine.
* **Adaptive effect grid**: the search extends until ``power_target`` is reached
  (or a cap), so a detectable-but-large effect never returns NaN.
* **Multi-series placebo pooling**: pass the candidate's blank gaps *and* the
  donor-unit placebo gaps; the null resamples within a randomly chosen series,
  giving the paper's cross-unit placebo distribution.

Test statistic: ``S(e) = mean(|e_t|)`` over the post window (matches the
permutation test used for inference).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union
import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


# ----------------------------------------------------------------------
# Moving-block resampling
# ----------------------------------------------------------------------

def _as_series_list(noise_pool) -> List[np.ndarray]:
    if isinstance(noise_pool, (list, tuple)) and len(noise_pool) and np.ndim(noise_pool[0]) >= 1:
        return [np.asarray(s, float).ravel() for s in noise_pool if len(np.asarray(s).ravel()) > 0]
    arr = np.asarray(noise_pool, float).ravel()
    return [arr] if arr.size else []


def _auto_block_len(series_list, n_post) -> int:
    L = int(np.median([len(s) for s in series_list]))
    return int(max(1, min(n_post, round(L ** (1.0 / 3.0)))))


def block_resample_windows(series_list, n_post, n_draws, block_len, rng):
    """Return an (n_draws, n_post) array of moving-block resampled windows.

    Each draw: pick a random placebo series, then tile moving blocks (with
    wraparound) until the window has length ``n_post``.  Preserves within-series
    autocorrelation and pools across series.
    """
    out = np.empty((n_draws, n_post))
    n_series = len(series_list)
    for d in range(n_draws):
        s = series_list[rng.integers(n_series)]
        Ls = len(s)
        pieces = []
        filled = 0
        while filled < n_post:
            start = int(rng.integers(Ls))
            blk = np.take(s, np.arange(start, start + block_len), mode="wrap")
            pieces.append(blk)
            filled += block_len
        out[d] = np.concatenate(pieces)[:n_post]
    return out


# ----------------------------------------------------------------------
# MDE at a single horizon
# ----------------------------------------------------------------------

def compute_mde(
    noise_pool,
    n_post: int,
    *,
    alpha: float = 0.05,
    power_target: float = 0.8,
    block_len: Optional[int] = None,
    n_null: int = 4000,
    n_power: int = 2000,
    max_sd: float = 8.0,
    n_grid: int = 64,
    random_state: int = 0,
) -> Dict:
    """Minimum detectable (constant) effect at horizon ``n_post``.

    Parameters
    ----------
    noise_pool : 1D array, or list of 1D arrays
        Placebo residual series (held-out blank-period gaps; optionally also the
        donor-unit placebo gaps).  Outcome units.
    n_post : int
        Post-treatment window length.
    alpha, power_target : float
        Test level and target power (e.g. 0.05 and 0.80).
    block_len : int, optional
        Moving-block length; default ~ L**(1/3).
    max_sd : float
        Largest effect (in residual-SD units) the adaptive grid will probe.

    Returns
    -------
    dict
        Dictionary with::

            mde_sd      : MDE in residual-standard-deviation units (np.inf if not
                          reached within ``max_sd``).
            mde_abs     : MDE in outcome units (= mde_sd * sigma).
            sigma       : residual SD scale.
            c_alpha     : critical value of the placebo null statistic.
            power_at_mde: achieved power at the reported MDE.
            feasible    : whether power_target was reached within max_sd.
            curve       : list of (effect_sd, power) probed.
    """
    rng = np.random.default_rng(random_state)
    series = _as_series_list(noise_pool)
    if not series:
        return {"mde_sd": np.inf, "mde_abs": np.inf, "sigma": np.nan,
                "c_alpha": np.nan, "power_at_mde": 0.0, "feasible": False, "curve": []}

    allvals = np.concatenate(series)
    sigma = float(np.std(allvals, ddof=1)) if allvals.size > 1 else float(np.std(allvals))
    sigma = max(sigma, 1e-12)
    if block_len is None:
        block_len = _auto_block_len(series, n_post)

    # placebo null -> critical value
    null_win = block_resample_windows(series, n_post, n_null, block_len, rng)
    null_stat = np.mean(np.abs(null_win), axis=1)
    c_alpha = float(np.quantile(null_stat, 1.0 - alpha))

    # power(tau) using the SAME resampled residual structure + a constant shift
    def power_of(tau_abs):
        win = block_resample_windows(series, n_post, n_power, block_len, rng)
        stat = np.mean(np.abs(win + tau_abs), axis=1)
        return float(np.mean(stat >= c_alpha))

    grid_sd = np.linspace(0.0, max_sd, n_grid)
    curve = []
    mde_sd, power_at = np.inf, 0.0
    prev = (0.0, power_of(0.0))
    curve.append((0.0, prev[1]))
    for g in grid_sd[1:]:
        p = power_of(g * sigma)
        curve.append((float(g), p))
        if p >= power_target:
            # linear interpolation between prev and current for a finer MDE
            g0, p0 = prev
            mde_sd = g0 + (power_target - p0) * (g - g0) / (p - p0 + 1e-12) if p > p0 else g
            power_at = p
            break
        prev = (g, p)

    feasible = np.isfinite(mde_sd)
    return {
        "mde_sd": float(mde_sd),
        "mde_abs": float(mde_sd * sigma) if feasible else np.inf,
        "sigma": sigma,
        "c_alpha": c_alpha,
        "power_at_mde": power_at,
        "feasible": bool(feasible),
        "block_len": int(block_len),
        "curve": curve,
    }


# ----------------------------------------------------------------------
# Detectability curve across horizons
# ----------------------------------------------------------------------

def detectability_curve(noise_pool, n_post_grid, **kw) -> Dict:
    """MDE at each horizon in ``n_post_grid``; returns {'details', 'curve_sd'}."""
    details, curve_sd = {}, {}
    for w in n_post_grid:
        r = compute_mde(noise_pool, w, **kw)
        details[w] = r
        curve_sd[w] = r["mde_sd"]
    feas = [w for w, r in details.items() if r["feasible"]]
    return {
        "details": details,
        "curve_sd": curve_sd,
        "min_horizon_mde_le_0p1sd": next((w for w in sorted(feas)
                                          if details[w]["mde_sd"] <= 0.1), None),
    }
