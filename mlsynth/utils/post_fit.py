"""Standardized post-fit diagnostics for synthetic control designs and the
matching power-analysis surface that consumes them.

After any MAREX-family estimator (LEXSCM, MAREX, SYNDES, PANGEO, ...) solves
its design problem, downstream consumers (the SAGE dashboard, paper-style
reports, comparison tables) all need the same numbers:

  - the post-treatment ATT, total effect, percentage lift, per-period gap;
  - pre / blank / post root-mean-squared-error of the synthetic gap;
  - inference scalars (p-value, CI bounds) when computed;
  - covariate-balance standardized mean differences (SMDs) when covariates
    were used in the design.

This module exposes one frozen dataclass (:class:`SyntheticControlPostFit`)
and three free functions:

  - :func:`compute_smd`             -- standalone, panel-independent SMD
                                        from any (cov_matrix, treated_w, control_w);
  - :func:`compute_post_fit`        -- the full diagnostic bundle from
                                        trajectories + boundaries + (optional)
                                        covariate matrix + (optional) inference;
  - :func:`compute_post_fit_marex`  -- adapter that builds the bundle from a
                                        ``MAREXResults`` + ``MAREXPanel`` pair.

The free-function entry points are deliberately small and reusable, so the
LEXSCM / SYNDES / PANGEO equivalents can be added one-at-a-time without
touching this module: they just compose the same primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyntheticControlPostFit:
    """Standardized post-fit diagnostics for a single synthetic control design.

    Field semantics are estimator-agnostic; every MAREX-family adapter
    populates the same shape. Any field that isn't naturally computable for
    the producing estimator is left ``None``.
    """

    # -------- Trajectories (always populated)
    treated_series: np.ndarray             # (T,)  synthetic treated
    control_series: np.ndarray             # (T,)  synthetic control
    gap_series: np.ndarray                 # (T,)  treated - control

    n_fit: int                             # estimation window length
    n_blank: int                           # holdout window length (0 if none)
    n_post: int                            # post-treatment window length

    # -------- Effects (None when no post window)
    ate: Optional[float] = None
    total_effect: Optional[float] = None
    ate_percent: Optional[float] = None
    ate_per_period: Optional[np.ndarray] = None       # (n_post,)
    cumulative_effect: Optional[np.ndarray] = None    # cumsum on post window

    # -------- Inference (None when not computed)
    p_value: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    inference_method: Optional[str] = None

    # -------- Fit quality (whichever are computable)
    rmse_fit: Optional[float] = None
    rmse_blank: Optional[float] = None
    rmse_post: Optional[float] = None

    # -------- Covariate balance (None when no covariates supplied)
    # Three pairwise comparisons, each with a per-covariate signed dict plus
    # two summary scalars (max abs SMD + Σ SMD²). The treated-vs-control pair
    # is the internal-validity check; the two vs-population pairs are the
    # external-representativeness checks Abadie & Zhou's design objective
    # explicitly targets (||X̄ − Σ wⱼ Xⱼ||² + ||X̄ − Σ vⱼ Xⱼ||²).
    covariate_names: Tuple[str, ...] = ()
    # synthetic_treated vs synthetic_control
    covariate_smd: Optional[Dict[str, float]] = None
    covariate_smd_abs_max: Optional[float] = None
    covariate_smd_squared_sum: Optional[float] = None
    # synthetic_treated vs population
    covariate_smd_treated_vs_pop: Optional[Dict[str, float]] = None
    covariate_smd_treated_vs_pop_abs_max: Optional[float] = None
    covariate_smd_treated_vs_pop_squared_sum: Optional[float] = None
    # synthetic_control vs population
    covariate_smd_control_vs_pop: Optional[Dict[str, float]] = None
    covariate_smd_control_vs_pop_abs_max: Optional[float] = None
    covariate_smd_control_vs_pop_squared_sum: Optional[float] = None

    # -------- Power analysis (None unless compute_power_analysis was run)
    power: Optional["PowerAnalysis"] = None


@dataclass(frozen=True)
class MDEPoint:
    """Minimum detectable effect at a single post-treatment horizon."""

    post_periods: int                       # horizon length used
    mde_absolute: float                     # MDE on the outcome scale
    mde_pct: float                          # MDE as % of baseline
    se: float                               # implied standard error of mean(post gap)
    power_at_observed: Optional[float] = None  # power to detect the design's observed ATT


@dataclass(frozen=True)
class PowerAnalysis:
    """Standardized power-analysis output attached to ``SyntheticControlPostFit``.

    Built from the placebo / blank-period gap variance and an analytical
    Gaussian approximation, with AR(1) variance inflation to handle serial
    correlation in the gap residuals. The intent matches the per-estimator
    power modules already in the library (PangeoPower, SPCDPowerAnalysis,
    SYNDESPower) but consumes the same ``SyntheticControlPostFit`` shape so
    every covariate-aware SCM-family estimator gets the surface for free.

    Attributes
    ----------
    headline : MDEPoint
        MDE for the actual ``n_post`` horizon of the realised design.
    curve : list of MDEPoint
        MDE / power values across the requested ``post_grid`` horizons (so
        callers can read a detectability curve).
    alpha : float
        Two-sided significance level assumed.
    power_target : float
        Target power the MDEs are computed at (default 0.80).
    sigma_placebo : float
        Standard deviation of the placebo gap series used as the noise scale.
    serial_correlation : float
        Lag-1 (AR(1)) autocorrelation of the placebo gap residuals used to
        inflate the variance for serial dependence.
    baseline : float
        Mean of the control trajectory on the post window (denominator for
        ``mde_pct``). NaN when no post window exists.
    method : str
        ``"analytical_ar1"`` for the closed-form Gaussian + AR(1) MDE used
        here. Reserved for future ``"monte_carlo"`` extensions.
    """

    headline: MDEPoint
    curve: Tuple[MDEPoint, ...]
    alpha: float
    power_target: float
    sigma_placebo: float
    serial_correlation: float
    baseline: float
    method: str = "analytical_ar1"

    def mde_by_horizon(self) -> Dict[int, float]:
        """``{post_periods: mde_pct}`` for quick lookup."""
        return {pt.post_periods: pt.mde_pct for pt in self.curve}


# ---------------------------------------------------------------------------
# Free function: standalone SMD (Option (i) from the conversation)
# ---------------------------------------------------------------------------

def compute_smd(
    cov_matrix: np.ndarray,
    treated_weights: np.ndarray,
    control_weights: np.ndarray,
    *,
    cov_names: Optional[Sequence[str]] = None,
    cov_scales: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Standardized mean differences between weighted treated and control means.

    Parameters
    ----------
    cov_matrix : ndarray, shape (N, M)
        Per-unit covariate values; rows align to ``treated_weights`` and
        ``control_weights``.
    treated_weights, control_weights : ndarray, shape (N,)
        Non-negative weights with disjoint supports. They are renormalised
        to sum to 1 internally (so callers may pass raw sums-to-K weights).
    cov_names : sequence of str, optional
        Names for the M covariates. Defaults to ``("cov_0", "cov_1", ...)``.
    cov_scales : ndarray, shape (M,), optional
        Pre-computed per-covariate standardization scales (cross-unit std).
        Defaults to the std of ``cov_matrix`` columns. Passing the value
        already cached by ``build_covariate_matrix`` is the right move.

    Returns
    -------
    dict with keys ``smd`` (the per-covariate dict), ``smd_abs_max``,
    and ``smd_squared_sum``. Returns empty / NaN summaries if either weight
    vector is all-zero.
    """
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    N, M = cov_matrix.shape
    tw = _normalize_weights(treated_weights, N)
    cw = _normalize_weights(control_weights, N)
    if tw is None or cw is None:
        return dict(smd={}, smd_abs_max=float("nan"), smd_squared_sum=float("nan"))

    if cov_scales is None:
        scales = cov_matrix.std(axis=0, ddof=1)
    else:
        scales = np.asarray(cov_scales, dtype=float).flatten()
    scales = np.where(scales > 1e-12, scales, 1.0)

    if cov_names is None:
        cov_names = tuple(f"cov_{m}" for m in range(M))
    cov_names = tuple(cov_names)

    treated_mean = tw @ cov_matrix
    control_mean = cw @ cov_matrix
    smd_vec = (treated_mean - control_mean) / scales
    smd = {n: float(v) for n, v in zip(cov_names, smd_vec)}
    return dict(
        smd=smd,
        smd_abs_max=float(np.max(np.abs(smd_vec))) if smd_vec.size else float("nan"),
        smd_squared_sum=float(smd_vec @ smd_vec),
    )


# ---------------------------------------------------------------------------
# Free function: the full diagnostic bundle (Option (ii))
# ---------------------------------------------------------------------------

def compute_post_fit(
    treated_series: np.ndarray,
    control_series: np.ndarray,
    *,
    n_fit: int,
    n_blank: int = 0,
    n_post: Optional[int] = None,
    # Covariate balance (cov_matrix + at least one of treated_weights /
    # control_weights triggers the balance block; population_weights
    # defaults to uniform if not supplied)
    cov_matrix: Optional[np.ndarray] = None,
    cov_names: Optional[Sequence[str]] = None,
    cov_scales: Optional[np.ndarray] = None,
    treated_weights: Optional[np.ndarray] = None,
    control_weights: Optional[np.ndarray] = None,
    population_weights: Optional[np.ndarray] = None,
    # Inference
    inference: Optional[Any] = None,
    n_treated_units: Optional[int] = None,
) -> SyntheticControlPostFit:
    """Compute a :class:`SyntheticControlPostFit` from trajectories + boundaries.

    The trajectories ``treated_series`` and ``control_series`` are the estimator's
    own synthetic constructs (Σⱼ wⱼ Yⱼ and Σⱼ vⱼ Yⱼ in Abadie-Zhou notation).
    ``n_post`` defaults to ``len(treated_series) - n_fit - n_blank``.

    Covariate balance fields are populated when ``cov_matrix`` + ``treated_weights``
    + ``control_weights`` are all supplied (the natural inputs for any MAREX-family
    design). The :func:`compute_smd` helper does the work, so the SMD numbers
    are exactly consistent with a standalone call to :func:`compute_smd`.

    Inference scalars are pulled from the estimator's inference object via
    :func:`_extract_inference`, which knows about the four common shapes
    (LEXSCM ``Inference``, MAREX ``MAREXInference``, SYNDES ``SYNDESInference``,
    or a plain dict). All inference fields are optional.
    """
    treated_series = np.asarray(treated_series, dtype=float).flatten()
    control_series = np.asarray(control_series, dtype=float).flatten()
    T = min(treated_series.size, control_series.size)
    treated_series = treated_series[:T]
    control_series = control_series[:T]
    gap = treated_series - control_series

    if n_post is None:
        n_post = max(0, T - n_fit - n_blank)
    blank_end = n_fit + n_blank
    post_end = blank_end + n_post

    gap_fit = gap[:n_fit]
    gap_blank = gap[n_fit:blank_end]
    gap_post = gap[blank_end:post_end]

    # ---- Effects
    ate = total = pct = None
    per_period = cumulative = None
    if gap_post.size:
        ate = float(np.mean(gap_post))
        total = float(np.sum(gap_post))
        per_period = gap_post.copy()
        cumulative = np.cumsum(gap_post).astype(float)
        baseline = float(np.mean(control_series[blank_end:post_end]))
        if abs(baseline) > 1e-12:
            pct = ate / baseline * 100.0

    # ---- Fit quality
    rmse_fit = float(np.sqrt(np.mean(gap_fit ** 2))) if gap_fit.size else None
    rmse_blank = float(np.sqrt(np.mean(gap_blank ** 2))) if gap_blank.size else None
    rmse_post = float(np.sqrt(np.mean(gap_post ** 2))) if gap_post.size else None

    # ---- Inference scalars
    inf = _extract_inference(inference)

    # ---- Covariate balance (three pairwise comparisons)
    cov_names_t: Tuple[str, ...] = ()
    smd_tc: Optional[Dict[str, float]] = None
    smd_tc_abs_max: Optional[float] = None
    smd_tc_sq_sum: Optional[float] = None
    smd_tp: Optional[Dict[str, float]] = None
    smd_tp_abs_max: Optional[float] = None
    smd_tp_sq_sum: Optional[float] = None
    smd_cp: Optional[Dict[str, float]] = None
    smd_cp_abs_max: Optional[float] = None
    smd_cp_sq_sum: Optional[float] = None
    if cov_matrix is not None and (treated_weights is not None
                                    or control_weights is not None):
        cov_arr = np.asarray(cov_matrix, dtype=float)
        N_units = cov_arr.shape[0]

        # Pre-compute population weights once (uniform when not given) so all
        # three comparisons share the same X̄ reference.
        if population_weights is None:
            pop_w = np.ones(N_units) / N_units
        else:
            pop_w_arr = np.asarray(population_weights, dtype=float).flatten()
            s = pop_w_arr.sum()
            pop_w = pop_w_arr / s if s > 0 else np.ones(N_units) / N_units

        # Pre-compute scales once so all three SMD calls share them.
        if cov_scales is None:
            shared_scales = cov_arr.std(axis=0, ddof=1)
        else:
            shared_scales = np.asarray(cov_scales, dtype=float).flatten()

        if treated_weights is not None and control_weights is not None:
            tc = compute_smd(cov_arr, treated_weights, control_weights,
                              cov_names=cov_names, cov_scales=shared_scales)
            smd_tc = tc["smd"]
            smd_tc_abs_max = tc["smd_abs_max"]
            smd_tc_sq_sum = tc["smd_squared_sum"]
            cov_names_t = tuple(smd_tc.keys())

        if treated_weights is not None:
            tp = compute_smd(cov_arr, treated_weights, pop_w,
                              cov_names=cov_names, cov_scales=shared_scales)
            smd_tp = tp["smd"]
            smd_tp_abs_max = tp["smd_abs_max"]
            smd_tp_sq_sum = tp["smd_squared_sum"]
            if not cov_names_t:
                cov_names_t = tuple(smd_tp.keys())

        if control_weights is not None:
            cp = compute_smd(cov_arr, control_weights, pop_w,
                              cov_names=cov_names, cov_scales=shared_scales)
            smd_cp = cp["smd"]
            smd_cp_abs_max = cp["smd_abs_max"]
            smd_cp_sq_sum = cp["smd_squared_sum"]
            if not cov_names_t:
                cov_names_t = tuple(smd_cp.keys())

    return SyntheticControlPostFit(
        treated_series=treated_series,
        control_series=control_series,
        gap_series=gap,
        n_fit=int(n_fit), n_blank=int(n_blank), n_post=int(n_post),
        ate=ate, total_effect=total, ate_percent=pct,
        ate_per_period=per_period, cumulative_effect=cumulative,
        p_value=inf.get("p_value"),
        ci_lower=inf.get("ci_lower"), ci_upper=inf.get("ci_upper"),
        inference_method=inf.get("method"),
        rmse_fit=rmse_fit, rmse_blank=rmse_blank, rmse_post=rmse_post,
        covariate_names=cov_names_t,
        covariate_smd=smd_tc,
        covariate_smd_abs_max=smd_tc_abs_max,
        covariate_smd_squared_sum=smd_tc_sq_sum,
        covariate_smd_treated_vs_pop=smd_tp,
        covariate_smd_treated_vs_pop_abs_max=smd_tp_abs_max,
        covariate_smd_treated_vs_pop_squared_sum=smd_tp_sq_sum,
        covariate_smd_control_vs_pop=smd_cp,
        covariate_smd_control_vs_pop_abs_max=smd_cp_abs_max,
        covariate_smd_control_vs_pop_squared_sum=smd_cp_sq_sum,
    )


# ---------------------------------------------------------------------------
# Free function: MAREX adapter (Option (iii) — attached as raw.post_fit by fit())
# ---------------------------------------------------------------------------

def compute_post_fit_marex(raw, panel, *, cov_scales: Optional[np.ndarray] = None,
                            ) -> SyntheticControlPostFit:
    """Adapt a ``MAREXResults`` + ``MAREXPanel`` pair into a ``SyntheticControlPostFit``.

    Pulls the aggregate synthetic-treated / synthetic-control trajectories from
    ``raw.globres``, the (T0, blank_periods) split from ``panel.T0`` and
    ``panel.blank_periods``, the inference object from ``raw.globres.inference``,
    and the covariate matrix from ``panel.covariates`` (when present).
    """
    glob = raw.globres
    T = glob.synthetic_treated.size
    n_fit = panel.T0 - panel.blank_periods
    n_blank = panel.blank_periods
    n_post = T - panel.T0
    cov_matrix = panel.covariates
    cov_names = None  # MAREXPanel doesn't carry names; user knows them from config
    tw = glob.treated_weights_agg
    cw = glob.control_weights_agg

    return compute_post_fit(
        treated_series=glob.synthetic_treated,
        control_series=glob.synthetic_control,
        n_fit=n_fit, n_blank=n_blank, n_post=n_post,
        cov_matrix=cov_matrix, cov_names=cov_names,
        cov_scales=cov_scales,
        treated_weights=tw, control_weights=cw,
        inference=getattr(glob, "inference", None),
        n_treated_units=int((np.asarray(tw) > 1e-8).sum()),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_weights(w, N: int) -> Optional[np.ndarray]:
    w = np.asarray(w, dtype=float).flatten()
    if w.size != N:
        raise ValueError(f"weight vector length {w.size} != N={N}")
    s = float(w.sum())
    if s <= 0:
        return None
    return w / s


def _extract_inference(inference: Any) -> Dict[str, Optional[float]]:
    """Normalise CI / p-value across estimator-specific shapes."""
    out: Dict[str, Optional[float]] = {
        "p_value": None, "ci_lower": None, "ci_upper": None, "method": None,
    }
    if inference is None:
        return out

    if isinstance(inference, dict):
        for k in out:
            if k in inference:
                out[k] = inference[k]
        return out

    cls_name = type(inference).__name__

    if cls_name == "MAREXInference":
        out["p_value"] = _safe_float(getattr(inference, "global_p_value", None))
        out["method"] = "marex_permutation"
        ci_band = getattr(inference, "ci", None)
        if ci_band is not None:
            ci = np.asarray(ci_band, dtype=float)
            if ci.ndim == 2 and ci.shape[1] == 2:
                with np.errstate(all="ignore"):
                    lo = np.nanmean(ci[:, 0])
                    hi = np.nanmean(ci[:, 1])
                if np.isfinite(lo) and np.isfinite(hi):
                    out["ci_lower"], out["ci_upper"] = float(lo), float(hi)
        return out

    if cls_name == "SYNDESInference":
        atet = _safe_float(getattr(inference, "atet", None))
        out["p_value"] = _safe_float(getattr(inference, "p_value", None))
        out["method"] = "syndes_permutation"
        null_stats = getattr(inference, "null_stats", None)
        alpha = float(getattr(inference, "alpha", 0.05) or 0.05)
        if atet is not None and null_stats is not None:
            ns = np.asarray(null_stats, dtype=float)
            if ns.size:
                lo_q, hi_q = np.quantile(ns, [alpha / 2.0, 1.0 - alpha / 2.0])
                out["ci_lower"] = float(atet - hi_q)
                out["ci_upper"] = float(atet - lo_q)
        return out

    if cls_name == "Inference":  # LEXSCM
        out["p_value"] = _safe_float(getattr(inference, "p_value", None))
        out["ci_lower"] = _safe_float(getattr(inference, "ci_lower", None))
        out["ci_upper"] = _safe_float(getattr(inference, "ci_upper", None))
        out["method"] = "lexscm_conformal"
        return out

    for k in out:
        if hasattr(inference, k):
            out[k] = _safe_float(getattr(inference, k))
    return out


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# ---------------------------------------------------------------------------
# Power analysis (analytical, Gaussian + AR(1) variance inflation)
# ---------------------------------------------------------------------------

def _ar1_rho(residuals: np.ndarray) -> float:
    """Lag-1 autocorrelation of a residual series, clipped to ``(-0.99, 0.99)``."""
    r = np.asarray(residuals, dtype=float).flatten()
    if r.size < 2:
        return 0.0
    num = float(r[:-1] @ r[1:])
    den = float(r @ r)
    if den <= 1e-12:
        return 0.0
    return float(np.clip(num / den, -0.99, 0.99))


def _variance_inflation(n: int, rho: float) -> float:
    """``Var(mean of n serially-correlated periods) / sigma^2``.

    For an AR(1) process with lag-1 correlation ``rho`` this is
    ``(1 + 2 * sum_{k=1}^{n-1} (1 - k/n) rho^k) / n`` (``= 1/n`` when
    ``rho = 0``). Matches the same formula PANGEO uses.
    """
    if n <= 0:
        return 0.0
    if n == 1 or abs(rho) < 1e-12:
        return 1.0 / n
    k = np.arange(1, n)
    s = float(np.sum((1.0 - k / n) * rho ** k))
    return (1.0 + 2.0 * s) / n


def compute_power_analysis(
    post_fit: SyntheticControlPostFit,
    *,
    alpha: float = 0.05,
    power_target: float = 0.80,
    post_grid: Optional[Sequence[int]] = None,
) -> PowerAnalysis:
    """Analytical MDE + power curve for a design's :class:`SyntheticControlPostFit`.

    Uses the placebo / blank-period gap residuals (or the pre-period gap when
    no blank window was carved out) to estimate the noise standard deviation
    ``sigma_placebo`` and the AR(1) autocorrelation ``rho``, then computes
    the minimum detectable effect for each horizon ``T`` in ``post_grid`` via
    the Gaussian formula

        MDE(T) = (z_{1-alpha/2} + z_{power}) * sigma_placebo * sqrt(VIF(T, rho)),

    where ``VIF(T, rho) = Var(mean of T AR(1) periods) / sigma_placebo^2``.
    The headline MDE uses ``T = post_fit.n_post`` (the realised post window).

    Parameters
    ----------
    post_fit : SyntheticControlPostFit
        The standardized post-fit from any MAREX-family estimator.
    alpha : float, default 0.05
        Two-sided significance level.
    power_target : float, default 0.80
        Target power for the MDE.
    post_grid : sequence of int, optional
        Post-treatment horizons at which to compute MDE. Defaults to a small
        geometric grid centered on ``post_fit.n_post`` so users see the
        detectability tradeoff vs. running the experiment longer.

    Returns
    -------
    PowerAnalysis
        Headline MDE + a curve over the requested horizons.
    """
    z_a = float(norm.ppf(1.0 - alpha / 2.0))
    z_p = float(norm.ppf(power_target))
    factor = z_a + z_p

    # Power analysis is a DESIGN quantity and must never depend on post-period
    # data being present. The "placebo" window -- periods on which the weights
    # were NOT learned -- is the blank/holdout window when present, else the
    # fit/pre window. BOTH the noise scale and the percentage baseline are taken
    # from this same placebo window, so the MDE is well-defined and identical
    # whether or not a post period exists (in design-only mode the blank periods
    # play the role of the post period). Both windows are zero-mean under H0 so
    # std() is the right noise estimator.
    if post_fit.n_blank > 0:
        placebo_slice = slice(post_fit.n_fit, post_fit.n_fit + post_fit.n_blank)
    else:
        placebo_slice = slice(0, post_fit.n_fit)
    placebo = np.asarray(post_fit.gap_series[placebo_slice], dtype=float)
    if placebo.size < 2 or not np.isfinite(placebo).all():
        sigma_placebo = float("nan")
        rho = 0.0
    else:
        sigma_placebo = float(placebo.std(ddof=1))
        rho = _ar1_rho(placebo)

    # Baseline for percentage scaling: mean of the synthetic control over the
    # SAME placebo window (an untreated outcome level, computed without any
    # post-period data).
    baseline_vals = np.asarray(post_fit.control_series[placebo_slice], dtype=float)
    baseline = (float(baseline_vals.mean())
                if baseline_vals.size and np.isfinite(baseline_vals).all()
                else float("nan"))

    # Default horizon grid: 1, 2, ..., max(n_post, 12), tagged with the
    # realised horizon if it isn't already in the grid.
    if post_grid is None:
        n_post = max(post_fit.n_post, 1)
        grid = sorted({int(h) for h in [1, 2, 4, 6, 8, 12, n_post] if h >= 1})
    else:
        grid = sorted({int(h) for h in post_grid if int(h) >= 1})

    def _point(T: int) -> MDEPoint:
        if not np.isfinite(sigma_placebo) or sigma_placebo <= 0:
            return MDEPoint(T, float("nan"), float("nan"), float("nan"))
        se = sigma_placebo * float(np.sqrt(_variance_inflation(T, rho)))
        mde_abs = factor * se
        mde_pct = (mde_abs / baseline * 100.0
                   if np.isfinite(baseline) and abs(baseline) > 1e-12
                   else float("nan"))
        # Power to detect the observed ATT, if there is one.
        power_at = None
        if post_fit.ate is not None and se > 0:
            z = abs(post_fit.ate) / se
            power_at = float(norm.cdf(z - z_a) + norm.cdf(-z - z_a))
        return MDEPoint(T, mde_abs, mde_pct, se, power_at_observed=power_at)

    headline_T = max(post_fit.n_post, 1)
    headline = _point(headline_T)
    curve = tuple(_point(T) for T in grid)

    return PowerAnalysis(
        headline=headline, curve=curve,
        alpha=float(alpha), power_target=float(power_target),
        sigma_placebo=sigma_placebo, serial_correlation=rho,
        baseline=baseline, method="analytical_ar1",
    )


# ---------------------------------------------------------------------------
# Standardized-result adapter
# ---------------------------------------------------------------------------

def to_effect_result(
    pf: "SyntheticControlPostFit",
    *,
    time_periods: Optional[np.ndarray] = None,
    intervention_time: Optional[Any] = None,
    method_name: Optional[str] = None,
    donor_weights: Optional[Dict[str, float]] = None,
) -> Any:
    """Convert a :class:`SyntheticControlPostFit` into a standardized ``EffectResult``.

    The single, family-wide adapter from the rich post-fit bundle to the
    contract's ``EffectResult`` view, so every MAREX-family estimator (LEXSCM,
    MAREX, SYNDES, PANGEO) gets ``report`` for free instead of hand-copying
    fields. The realized effect's standard scalars populate the standard
    sub-models; everything the contract has no slot for (per-period effects,
    cumulative effect, covariate SMDs, and the full ``post_fit`` object itself)
    is carried in ``additional_outputs`` so it remains discoverable.
    """
    from ..config_models import (
        EffectResult, EffectsResults, FitDiagnosticsResults, InferenceResults,
        MethodDetailsResults, TimeSeriesResults, WeightsResults,
    )

    treated = np.asarray(pf.treated_series, dtype=float)
    control = np.asarray(pf.control_series, dtype=float)
    gap = np.asarray(pf.gap_series, dtype=float)
    tp = np.asarray(time_periods) if time_periods is not None else None
    if tp is not None and tp.shape[0] != treated.shape[0]:
        tp = None

    extras: Dict[str, Any] = {"post_fit": pf}
    if pf.ate_per_period is not None:
        extras["ate_per_period"] = pf.ate_per_period
    if pf.cumulative_effect is not None:
        extras["cumulative_effect"] = pf.cumulative_effect
    for name in ("covariate_smd", "covariate_smd_treated_vs_pop",
                 "covariate_smd_control_vs_pop"):
        val = getattr(pf, name, None)
        if val is not None:
            extras[name] = val

    blank = ({"rmse_blank": pf.rmse_blank} if pf.rmse_blank is not None else None)
    return EffectResult(
        effects=EffectsResults(att=pf.ate, att_percent=pf.ate_percent),
        time_series=TimeSeriesResults(
            observed_outcome=treated, counterfactual_outcome=control,
            estimated_gap=gap, time_periods=tp, intervention_time=intervention_time,
        ),
        fit_diagnostics=FitDiagnosticsResults(
            rmse_pre=pf.rmse_fit, rmse_post=pf.rmse_post, additional_metrics=blank),
        inference=InferenceResults(
            p_value=pf.p_value, ci_lower=pf.ci_lower, ci_upper=pf.ci_upper,
            method=pf.inference_method),
        weights=(WeightsResults(donor_weights=donor_weights)
                 if donor_weights is not None else None),
        method_details=(MethodDetailsResults(method_name=method_name)
                        if method_name else None),
        additional_outputs=extras,
    )
