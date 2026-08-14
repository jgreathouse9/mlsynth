"""Delete-one jackknife inference for PPSCM (Ben-Michael et al. 2022).

The paper's jackknife drops each unit ``i`` (treated *or* control), refits the
full staggered estimator on the remaining ``n - 1`` units (holding ``nu``
fixed), and forms

    se^2 = (n - 1) / n * sum_i (theta_i - mean_i theta_i)^2

separately for the overall ATT and each relative-time horizon. Wald intervals
are built from these SEs around the full-sample point estimates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import norm

from ...exceptions import MlsynthConfigError, MlsynthDataError

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


def rolling_pooled_block_sums(
    Xy: np.ndarray, trt: np.ndarray, d: int, n_leads: int, n_lags: int,
    *, fixedeff: bool, time_cohort: bool, nu_used: float, lam: float,
    solver: Any, horizon: int, min_train_frac: float = 0.3,
) -> List[np.ndarray]:
    """Per-unit cumulative out-of-sample errors, one pooled solve per origin.

    Slides an origin across the pre-period and, at each one, pretends every treated
    unit adopted there: a single partially-pooled fit on the data before the origin
    yields *every* unit's weights at once, and each unit's summed effect over the next
    ``horizon`` periods is one conformity score for it. So a pass costs one solve per
    origin, not one per unit per origin.

    Origins step by ``horizon``, so the windows do not overlap and the scores stay
    exchangeable, and each fit sees only data strictly before the window it scores.

    Returns
    -------
    list of numpy.ndarray
        One array of finite scores per treated cohort, in ``groups`` order.
    """
    from ..conformal import MIN_TRAIN_PERIODS

    trt = np.asarray(trt, dtype=float)
    adopted = np.isfinite(trt)
    if not adopted.any():
        raise MlsynthDataError(
            "cumulative conformal inference needs at least one treated unit; "
            "no unit has a finite adoption time."
        )
    horizon = int(horizon)
    if (isinstance(min_train_frac, bool)
            or not isinstance(min_train_frac, (int, float, np.floating))
            or not 0.0 < float(min_train_frac) < 1.0):
        raise MlsynthConfigError(
            "min_train_frac must be a number in the open interval (0, 1); got "
            f"{min_train_frac!r}."
        )
    earliest = int(np.min(trt[adopted]))
    start = max(MIN_TRAIN_PERIODS, int(earliest * float(min_train_frac)))

    scores: Dict[int, List[float]] = {}
    n_groups = 0
    for origin in range(start, earliest - horizon + 1, horizon):
        trt_o = trt.copy()
        trt_o[adopted] = origin
        try:
            fo = run_multisynth(
                Xy, trt_o, origin, horizon, origin,
                fixedeff=fixedeff, time_cohort=time_cohort,
                nu=nu_used, lam=lam, solver=solver,
            )
        except Exception:  # pragma: no cover - a degenerate origin is skipped
            continue
        n_groups = max(n_groups, len(fo["groups"]))
        for k in range(len(fo["groups"])):
            path = np.asarray(fo["tau_rel"][k], dtype=float)[:horizon]
            if path.size == horizon and np.isfinite(path).all():
                scores.setdefault(k, []).append(float(path.sum()))
    return [np.asarray(scores.get(k, []), dtype=float) for k in range(n_groups)]


def cumulative_conformal_per_unit(
    Xy: np.ndarray, trt: np.ndarray, d: int, n_leads: int, n_lags: int,
    *, fixedeff: bool, time_cohort: bool, nu_used: float, lam: float,
    solver: Any, alpha: float, horizon: int, min_train_frac: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-unit conformal band for each treated unit's cumulative effect.

    The point estimates come from the full fit's ``tau_rel``; the calibration comes
    from :func:`rolling_pooled_block_sums` and
    :func:`mlsynth.utils.conformal.cumulative_conformal_interval` -- the same
    combiner VanillaSC uses, so the order statistic has one definition.

    Returns
    -------
    tuple of numpy.ndarray
        ``(point, lower, upper, n_scores)``, each of shape ``(J,)`` in ``groups``
        order. A unit with too few calibration windows for the requested level gets
        an infinite band rather than a narrow one that does not cover.
    """
    from ..conformal import cumulative_conformal_interval

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating)):
        raise MlsynthConfigError(f"alpha must be a number in (0, 1); got {alpha!r}.")
    if not 0.0 < float(alpha) < 1.0:
        raise MlsynthConfigError(
            f"alpha must lie in the open interval (0, 1); got {alpha!r}."
        )
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or int(horizon) < 1:
        raise MlsynthConfigError(f"horizon must be a positive integer; got {horizon!r}.")
    if (isinstance(min_train_frac, bool)
            or not isinstance(min_train_frac, (int, float, np.floating))
            or not 0.0 < float(min_train_frac) < 1.0):
        raise MlsynthConfigError(
            "min_train_frac must be a number in the open interval (0, 1); got "
            f"{min_train_frac!r}."
        )
    horizon = int(horizon)
    if horizon > int(n_leads):
        raise MlsynthDataError(
            f"horizon ({horizon}) exceeds the {int(n_leads)} post-period lead(s) "
            "estimated; there is no full window to accumulate."
        )

    if not np.isfinite(np.asarray(trt, dtype=float)).any():
        raise MlsynthDataError(
            "cumulative conformal inference needs at least one treated unit; "
            "no unit has a finite adoption time."
        )

    full = run_multisynth(
        Xy, trt, d, n_leads, n_lags, fixedeff=fixedeff, time_cohort=time_cohort,
        nu=nu_used, lam=lam, solver=solver,
    )
    n_units = len(full["groups"])
    point = np.array(
        [float(np.sum(np.asarray(full["tau_rel"][k], dtype=float)[:horizon]))
         for k in range(n_units)], dtype=float,
    )

    per_unit_scores = rolling_pooled_block_sums(
        Xy, trt, d, n_leads, n_lags, fixedeff=fixedeff, time_cohort=time_cohort,
        nu_used=nu_used, lam=lam, solver=solver, horizon=horizon,
        min_train_frac=min_train_frac,
    )

    lower = np.full(n_units, np.nan)
    upper = np.full(n_units, np.nan)
    n_scores = np.zeros(n_units, dtype=int)
    for k in range(n_units):
        s = per_unit_scores[k] if k < len(per_unit_scores) else np.asarray([])
        n_scores[k] = int(s.size)
        if s.size == 0:
            lower[k], upper[k] = -np.inf, np.inf
            continue
        band = cumulative_conformal_interval(
            point[k], s, alpha=float(alpha), horizon=horizon)
        lower[k], upper[k] = band.lower, band.upper
    return point, lower, upper, n_scores
