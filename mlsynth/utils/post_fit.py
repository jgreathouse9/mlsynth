"""Standardized post-fit diagnostics for synthetic control designs.

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
