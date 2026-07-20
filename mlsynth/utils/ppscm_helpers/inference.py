"""Delete-one jackknife inference for PPSCM (Ben-Michael et al. 2022).

The paper's jackknife drops each unit ``i`` (treated *or* control), refits the
full staggered estimator on the remaining ``n - 1`` units (holding ``nu``
fixed), and forms

    se^2 = (n - 1) / n * sum_i (theta_i - mean_i theta_i)^2

separately for the overall ATT and each relative-time horizon. Wald intervals
are built from these SEs around the full-sample point estimates.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from scipy.stats import norm

from .engine import run_multisynth, predict_tau


def per_unit_intervals(
    M: np.ndarray, tau_rel: np.ndarray, *, alpha: float,
    time_dependence: str = "iid",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-unit CFPT/SCPI prediction intervals for each unit's effect path.

    The pooled bootstrap / jackknife measures variability *across* units and so
    cannot give one treated unit its own interval. This builds a per-unit band from
    that unit's own fit and reuses mlsynth's out-of-sample interval engine (the same
    CFPT/SCPI machinery MSQRT uses), so PPSCM's per-unit bands are methodologically
    consistent with MSQRT's.

    For unit ``k`` the bands come from its post-period effect path
    ``tau_rel[k, :]`` (the CFPT ``effects``) and its pre-period residuals
    ``M[:, k]`` (the CFPT ``pre_residuals``): the residual moments set the
    sub-Gaussian scale of the counterfactual prediction error, which correctly
    accounts for the in-sample fit -- unlike a naive permutation over the
    QP-optimised pre-residuals, which are not exchangeable with the post gaps and
    over-reject. The engine is called per unit (one column at a time), so units
    with different post horizons (ragged ``NaN``) are handled by trimming.

    A single engine call returns the full CFPT family, so both the time-averaged
    band (``TAUS``) and the per-period pointwise bands (``TSUS``) come out of the
    same computation: the pointwise bands are the ``TAUS`` band's per-horizon
    counterpart and are wider (``TAUS`` shrinks by ``sqrt(L)`` under
    ``time_dependence="iid"``; a single period does not).

    Parameters
    ----------
    M : numpy.ndarray
        Pre-period residual columns, shape ``(d, J)`` (a 1-D array is a single
        unit). ``NaN`` entries are dropped per unit.
    tau_rel : numpy.ndarray
        Post-period relative-time effect paths, shape ``(J, H)`` (a 1-D array is a
        single unit). ``NaN`` (past a unit's horizon) is dropped per unit.
    alpha : float
        Total miscoverage level; the interval is ``100 * (1 - alpha)`` percent.
        Keyword-only.
    time_dependence : {"iid", "general"}, default "iid"
        Time-averaging bound passed through to the CFPT engine (it affects only the
        time-averaged band, never the per-period bands). Keyword-only.

    Returns
    -------
    tuple of numpy.ndarray
        ``(ci_lower, ci_upper, p_value, tau_lower, tau_upper)``. The first three
        have shape ``(J,)``: the per-unit band bounds on the time-averaged ATT and a
        band-implied two-sided p-value (the house convention
        ``2 * (alpha/2) ** ((point/half_width) ** 2)``, clamped to ``[0, 1]``). The
        last two have shape ``(J, H)`` -- the per-unit, per-period band bounds,
        aligned with ``tau_rel`` (``NaN`` where ``tau_rel`` is ``NaN``). A unit with
        no usable residuals yields ``NaN`` throughout.
    """
    from ..scpi_helpers import out_of_sample_intervals

    M = np.asarray(M, dtype=float)
    tau_rel = np.asarray(tau_rel, dtype=float)
    if M.ndim == 1:
        M = M[:, None]
    if tau_rel.ndim == 1:
        tau_rel = tau_rel[None, :]
    J, H = tau_rel.shape
    lo = np.full(J, np.nan)
    hi = np.full(J, np.nan)
    pval = np.full(J, np.nan)
    tsu_lo = np.full((J, H), np.nan)
    tsu_hi = np.full((J, H), np.nan)

    for k in range(J):
        finite = np.isfinite(tau_rel[k, :])
        pre = M[:, k][np.isfinite(M[:, k])]
        post = tau_rel[k, finite]
        if pre.size == 0 or post.size == 0:  # pragma: no cover - guarded upstream
            continue
        res = out_of_sample_intervals(
            effects=post[:, None], pre_residuals=pre[:, None],
            unit_names=[k], period_labels=list(range(post.size)),
            alpha=alpha, time_dependence=time_dependence,
        )
        band = res.taus[k]
        lo[k], hi[k] = float(band.lower), float(band.upper)
        half = 0.5 * (band.upper - band.lower)
        if half > 0:
            pval[k] = float(min(1.0, 2.0 * (alpha / 2.0) ** ((band.point / half) ** 2)))
        else:  # pragma: no cover - degenerate zero-width band
            pval[k] = 1.0
        # Per-period (TSUS) bands -- already computed in the same call. Place each
        # back at the horizon it came from so the output aligns with ``tau_rel``.
        for p, col in enumerate(np.flatnonzero(finite)):
            pband = res.tsus[(k, p)]
            tsu_lo[k, col] = float(pband.lower)
            tsu_hi[k, col] = float(pband.upper)
    return lo, hi, pval, tsu_lo, tsu_hi


# Mammen (1993) two-point wild-bootstrap multipliers (mean 0, variance 1) --
# augsynth's default ``rwild_b``.
_PHI = np.sqrt(5.0)
_WILD_VALUES = np.array([-(_PHI - 1.0) / 2.0, (_PHI + 1.0) / 2.0])
_WILD_PROBS = np.array([(_PHI + 1.0) / (2.0 * _PHI), (_PHI - 1.0) / (2.0 * _PHI)])


def bootstrap_inference(
    fit: dict, *, alpha: float, n_boot: int, seed: int,
    per_time_full: np.ndarray, att_full: float,
):
    """augsynth's default Mammen wild/multiplier bootstrap (``weighted_bootstrap_multi``).

    Reweights the *single* fit by per-unit multipliers ``Z`` (no refit): for each
    draw, ``predict_tau(bs_weight=Z) - (sum(Z)/n_treated) * point_estimate``; the
    bootstrap SE is the root-mean-square of the centered draws. Returns
    ``(att, se, ci, per_time_se, per_time_ci)`` matching ``jackknife_inference``.
    """
    res = fit["res"]
    groups, adopt_of, members = fit["groups"], fit["adopt_of"], fit["members"]
    donors, W, n1 = fit["donors"], fit["weights"], fit["n1"]
    n, H = fit["n"], fit["n_leads"]
    n_treated = float(np.sum(n1))
    rng = np.random.default_rng(seed)

    att_b = np.empty(n_boot)
    pt_b = np.full((n_boot, H), np.nan)
    for b in range(n_boot):
        Z = rng.choice(_WILD_VALUES, size=n, p=_WILD_PROBS)
        _, pt, a = predict_tau(res, groups, adopt_of, members, donors, W, n1, H, n,
                               bs_weight=Z)
        shift = Z.sum() / n_treated
        att_b[b] = a - shift * att_full
        pt_b[b] = pt - shift * per_time_full

    se = float(np.sqrt(np.mean((att_b - att_b.mean()) ** 2)))
    per_time_se = np.sqrt(np.nanmean((pt_b - np.nanmean(pt_b, axis=0)) ** 2, axis=0))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    ci = (att_full - z * se, att_full + z * se)
    per_time_ci = np.column_stack([per_time_full - z * per_time_se,
                                   per_time_full + z * per_time_se])
    return float(att_full), se, ci, per_time_se, per_time_ci


def jackknife_inference(
    Xy: np.ndarray, trt: np.ndarray, d: int, n_leads: int, n_lags: int,
    *, fixedeff: bool, time_cohort: bool, nu_used: float, lam: float,
    solver: Any, alpha: float, per_time_full: np.ndarray, att_full: float,
) -> Tuple[float, float, Tuple[float, float], np.ndarray, np.ndarray]:
    """Return ``(att, se, ci, per_time_se, per_time_ci)``."""
    n = Xy.shape[0]
    H = n_leads
    att_loo = np.full(n, np.nan)
    pt_loo = np.full((n, H), np.nan)

    for i in range(n):
        keep = np.ones(n, dtype=bool); keep[i] = False
        trt_i = trt[keep]
        if not np.isfinite(trt_i).any() or np.isfinite(trt_i).all():
            continue                                  # need >=1 treated and >=1 control
        try:
            fit_i = run_multisynth(
                Xy[keep], trt_i, d, n_leads, n_lags,
                fixedeff=fixedeff, time_cohort=time_cohort,
                nu=nu_used, lam=lam, solver=solver,
            )
        except Exception:
            continue
        att_loo[i] = fit_i["att"]
        pt = fit_i["per_time"]
        pt_loo[i, : len(pt)] = pt

    def _se(col: np.ndarray) -> float:
        x = col[~np.isnan(col)]
        m = x.size
        if m < 2:
            return float("nan")
        return float(np.sqrt((m - 1) / m * np.sum((x - x.mean()) ** 2)))

    se = _se(att_loo)
    per_time_se = np.array([_se(pt_loo[:, h]) for h in range(H)])
    z = float(norm.ppf(1.0 - alpha / 2.0))
    ci = (att_full - z * se, att_full + z * se)
    per_time_ci = np.column_stack([per_time_full - z * per_time_se,
                                   per_time_full + z * per_time_se])
    return float(att_full), se, ci, per_time_se, per_time_ci
