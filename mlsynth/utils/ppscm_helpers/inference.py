"""Delete-one jackknife inference for PPSCM (Ben-Michael et al. 2022).

The paper's jackknife drops each unit ``i`` (treated *or* control), refits the
full staggered estimator on the remaining ``n - 1`` units (holding ``nu``
fixed), and forms

    se^2 = (n - 1) / n * sum_i (theta_i - mean_i theta_i)^2

separately for the overall ATT and each relative-time horizon. Wald intervals
are built from these SEs around the full-sample point estimates.

Every function here that refits the panel -- the jackknife, and the conformal
band's rolling origins -- takes the fit's
:class:`~mlsynth.utils.ppscm_helpers.engine.Conventions` and passes it on. A
replicate has to be a refit of the estimator that produced the point estimate,
and #467 is what happens when it is not: the replicates ran augsynth's donor
weighting, baseline and donor pool while the estimate ran the caller's, giving
a standard error that was finite, plausible, and for a different estimator.

Two guards follow from the same incident. ``run_multisynth`` refuses a
non-finite ``nu``, since a uniform-weight fit poses no program and reports
``nu_used`` as ``NaN``; and a jackknife that ends with fewer than two usable
replicates raises instead of returning ``NaN``, because a missing standard
error read as a degenerate panel for a whole review.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from ...exceptions import (
    MlsynthConfigError, MlsynthDataError, MlsynthEstimationError)

from .engine import Conventions, run_multisynth, predict_tau


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
    per_time_full: np.ndarray, att_full: float, return_paths: bool = False,
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
    if return_paths:
        return float(att_full), se, ci, per_time_se, per_time_ci, pt_b
    return float(att_full), se, ci, per_time_se, per_time_ci


def cumulative_supt_band(
    per_time_full: np.ndarray,
    replicate_paths: np.ndarray,
    *,
    alpha: float,
    jackknife: bool = True,
    n_sims: int = 200_000,
    seed: Optional[int] = 0,
    method: str = "jackknife",
):
    """Simultaneous band for the running total, from the replicate paths.

    An interval for a cumulative effect is not the running total of the
    per-period intervals. Adding endpoints treats the period errors as moving in
    lockstep, so the width grows with the number of periods; rescaling one
    period's interval assumes the opposite. Here the replicate paths are
    accumulated first and the standard error taken after, so whatever
    correlation the errors have is the correlation the band inherits.

    The band is simultaneous over horizons
    (:func:`mlsynth.utils.supt.supt_critical_value`), because a cumulative path
    is read as a path.

    Parameters
    ----------
    per_time_full : np.ndarray, shape (H,)
        The per-horizon effect path from the full fit.
    replicate_paths : np.ndarray, shape (n_replicates, H)
        One per-horizon path per replicate. Rows containing ``NaN`` are dropped,
        so a leave-one-out refit that failed is absent instead of counted as zero.
    alpha : float
        The band is simultaneous at ``1 - alpha``.
    jackknife : bool, default True
        Apply the delete-one inflation to the standard error. True for
        leave-one-out replicates, which differ from the full estimate by
        ``O(1/m)``; False for bootstrap draws, already on the estimator's scale.
    n_sims, seed
        Tabulation of the sup-t critical value.
    method : str
        Which ensemble produced the paths, recorded on the result -- a jackknife
        band and a bootstrap band are not interchangeable numbers.

    Returns
    -------
    PPSCMCumulativeBand
    """
    from ..supt import cumulative_from_paths, jackknife_se, supt_critical_value
    from .structures import PPSCMCumulativeBand

    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating))
            or not 0.0 < float(alpha) < 1.0):
        raise MlsynthConfigError(
            f"alpha must be a number in the open interval (0, 1); got {alpha!r}."
        )
    pt = np.asarray(per_time_full, dtype=float).ravel()
    R = np.asarray(replicate_paths, dtype=float)
    if R.ndim != 2 or R.shape[1] != pt.size:
        raise MlsynthDataError(
            f"replicate_paths must be (n_replicates, {pt.size}); got shape {R.shape}."
        )
    R = R[np.isfinite(R).all(axis=1)]
    if R.shape[0] < 2:
        raise MlsynthDataError(
            f"need at least 2 complete replicate paths to form a band; got {R.shape[0]}."
        )

    cum = cumulative_from_paths(R)
    se = jackknife_se(cum, jackknife=jackknife)
    q = supt_critical_value(cum, alpha=float(alpha), n_sims=n_sims, seed=seed)
    point = np.cumsum(pt)
    return PPSCMCumulativeBand(
        horizons=np.arange(1, pt.size + 1),
        point=point, lower=point - q * se, upper=point + q * se, se=se,
        critical_value=float(q), alpha=float(alpha),
        n_replicates=int(R.shape[0]), method=str(method),
    )


def jackknife_inference(
    Xy: np.ndarray, trt: np.ndarray, d: int, n_leads: int, n_lags: int,
    *, fixedeff: bool, time_cohort: bool, nu_used: float, lam: float,
    solver: Any, alpha: float, per_time_full: np.ndarray, att_full: float,
    conventions: Conventions = Conventions(), return_paths: bool = False,
) -> Tuple[float, float, Tuple[float, float], np.ndarray, np.ndarray]:
    """Return ``(att, se, ci, per_time_se, per_time_ci)``.

    With ``return_paths`` the leave-one-out per-horizon paths are appended. They
    are computed either way; keeping them lets a caller build a cumulative band
    without refitting the whole jackknife a second time.
    """
    n = Xy.shape[0]
    H = n_leads
    att_loo = np.full(n, np.nan)
    pt_loo = np.full((n, H), np.nan)
    # ``nu_used`` is NaN when no quadratic program was posed (uniform donor
    # weights), and that branch ignores the pooling level entirely. Passing the
    # NaN on would trip ``run_multisynth``'s finiteness guard on every replicate.
    nu_refit = float(nu_used) if np.isfinite(nu_used) else None

    # Which replicates removed a treated unit, as opposed to a control. The two
    # are different quantities: deleting a control moves the synthetic
    # counterfactual a little, deleting a treated unit removes one of the few
    # draws the effect is averaged over, and only the second is the sampling
    # variability of the pooled estimand. This is recorded from the loop and not
    # derived from ``trt`` afterwards, because a treated refit can also raise and
    # be skipped -- which rows are present is not which units are treated.
    treated_loo = np.zeros(n, dtype=bool)
    was_treated = np.isfinite(trt)

    for i in range(n):
        keep = np.ones(n, dtype=bool); keep[i] = False
        trt_i = trt[keep]
        if not np.isfinite(trt_i).any() or np.isfinite(trt_i).all():
            continue                                  # need >=1 treated and >=1 control
        try:
            fit_i = run_multisynth(
                Xy[keep], trt_i, d, n_leads, n_lags,
                fixedeff=fixedeff, time_cohort=time_cohort,
                nu=nu_refit, lam=lam, solver=solver, conventions=conventions,
            )
        except Exception:
            continue
        att_loo[i] = fit_i["att"]
        treated_loo[i] = bool(was_treated[i])
        pt = fit_i["per_time"]
        pt_loo[i, : len(pt)] = pt

    def _se(col: np.ndarray) -> float:
        x = col[~np.isnan(col)]
        m = x.size
        if m < 2:
            return float("nan")
        return float(np.sqrt((m - 1) / m * np.sum((x - x.mean()) ** 2)))

    usable = int(np.isfinite(att_loo).sum())
    if usable < 2:
        raise MlsynthEstimationError(
            f"the delete-one jackknife finished with {usable} usable replicate(s) "
            f"out of {n} units; at least 2 are needed for a standard error. Every "
            "leave-one-out refit raised, so there is no inference to report -- "
            "returning NaN here is what hid #467.")
    n_treated_loo = int(treated_loo.sum())
    if n_treated_loo == 0:
        n_treated = int(was_treated.sum())
        raise MlsynthEstimationError(
            f"the delete-one jackknife admitted {usable} replicate(s), and none of "
            f"them removed a treated unit; the panel has "
            f"{n_treated} treated unit{'' if n_treated == 1 else 's'}. Every "
            "replicate is a control deletion, so the spread being measured is "
            "donor substitution and not the sampling variability of the effect -- "
            "the treated unit's own outcomes cannot move the standard error at "
            "all. With a single treated unit its deletion leaves no treated unit "
            "and is skipped, which is why counting usable replicates does not "
            "catch this. Use an inference method that does not rest on deleting "
            "treated units, or report the point estimate without an interval.")
    se = _se(att_loo)
    per_time_se = np.array([_se(pt_loo[:, h]) for h in range(H)])
    z = float(norm.ppf(1.0 - alpha / 2.0))
    ci = (att_full - z * se, att_full + z * se)
    per_time_ci = np.column_stack([per_time_full - z * per_time_se,
                                   per_time_full + z * per_time_se])
    if return_paths:
        return float(att_full), se, ci, per_time_se, per_time_ci, pt_loo
    return float(att_full), se, ci, per_time_se, per_time_ci


def rolling_pooled_block_sums(
    Xy: np.ndarray, trt: np.ndarray, d: int, n_leads: int, n_lags: int,
    *, fixedeff: bool, time_cohort: bool, nu_used: float, lam: float,
    solver: Any, horizon: int, min_train_frac: float = 0.3,
    conventions: Conventions = Conventions(),
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
                nu=(float(nu_used) if np.isfinite(nu_used) else None),
                lam=lam, solver=solver, conventions=conventions,
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
    conventions: Conventions = Conventions(),
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
        nu=(float(nu_used) if np.isfinite(nu_used) else None),
        lam=lam, solver=solver, conventions=conventions,
    )
    n_units = len(full["groups"])
    point = np.array(
        [float(np.sum(np.asarray(full["tau_rel"][k], dtype=float)[:horizon]))
         for k in range(n_units)], dtype=float,
    )

    per_unit_scores = rolling_pooled_block_sums(
        Xy, trt, d, n_leads, n_lags, fixedeff=fixedeff, time_cohort=time_cohort,
        nu_used=nu_used, lam=lam, solver=solver, horizon=horizon,
        min_train_frac=min_train_frac, conventions=conventions,
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


def cwz_cumulative_per_unit(
    Xy: np.ndarray, trt: np.ndarray, d: int, n_leads: int, n_lags: int,
    *, fixedeff: bool, time_cohort: bool, nu_used: float, lam: float,
    solver: Any, alpha: float, horizon: int, n_nulls: int = 25,
    grid_scale: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-unit cumulative band by inverting a moving-block conformal test.

    The counterpart of :func:`cumulative_conformal_per_unit`, calibrated against
    the ``T`` cyclic shifts of the residual path instead of a disjoint split of
    the pre-period. The split version's window count is roughly ``0.7 * T0 / L``
    and a finite ``1 - alpha`` band needs ``ceil((m+1)(1-alpha)) <= m``, so at the
    90 percent level it needs nine windows -- a floor of about ``12.8 * L``
    pre-periods before a band exists at all, and every feasible design then sits
    in the regime where the rank never trims and the half-width is simply the
    largest score. The cyclic reference set does not depend on the horizon, so
    neither the floor nor that regime applies.

    The price is a shape assumption. Test inversion needs a null to subtract, so
    the null here is a constant per-period effect and the reported band is
    ``horizon`` times the accepted range of that effect. An effect that ramps is
    not in the null family, and the honest outcome then is an empty accepted set,
    which :func:`~mlsynth.utils.conformal.confidence_set_bounds` returns as
    ``(nan, nan)``.

    Two construction details decide the answer, both established by measurement;
    the full account is in
    :func:`~mlsynth.utils.conformal.moving_block_pvalue`.

    The statistic is ``mean_abs``, the reference implementation's. The absolute
    block sum is the intuitive choice for a running total and is invalid here:
    the quadratic program leaves end-of-window residuals sign-coherent, so block
    sums ramp toward the end of the window while magnitudes stay flat, and the
    trailing block always occupies the most inflated position. It is not a
    compromise -- displacing a mean-zero block by ``delta`` raises the mean of its
    absolute values, so the test has power against precisely the constant shift
    being inverted.

    The null refit balances every period. Under the null the adjusted series is
    an untreated series, so all of it is fitting data, and leaving a period out
    would put the trailing block partly outside the fit its reference blocks come
    from. Balancing the whole window requires an explicit ``nu``: the automatic
    rule sits exactly on its boundary when one unit is treated, and cvxpy refuses
    the program. That is why ``nu_used`` is a parameter, not a default.

    The grid is an approximation and it errs in one direction. The band is the
    range of accepted candidates, so a coarse grid samples fewer of them and
    reports a band that is too narrow, converging upward as ``n_nulls`` rises --
    measured on a two-unit panel, a width of 3.14 at five candidates against 5.86
    at thirty-one. Coarse is anti-conservative here, not conservative, which is
    the opposite of the usual intuition about discretisation.

    An accepted set that reaches an end of the grid is bounded by ``grid_scale``
    and not by the data, and that end is returned as infinite. Reporting the
    endpoint instead would understate the band silently, since it looks like any
    other number.

    Returns
    -------
    tuple of numpy.ndarray
        ``(point, lower, upper, p_zero)``, each of shape ``(J,)`` in ``groups``
        order. ``lower`` and ``upper`` are on the cumulative scale and may be
        infinite; ``p_zero`` is the permutation p-value of the no-effect null.
    """
    from ..conformal import confidence_set_bounds, moving_block_pvalue

    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating))
            or not 0.0 < float(alpha) < 1.0):
        raise MlsynthConfigError(
            f"alpha must be a number in the open interval (0, 1); got {alpha!r}.")
    if (isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer))
            or int(horizon) < 1):
        raise MlsynthConfigError(
            f"horizon must be a positive integer; got {horizon!r}.")
    if (isinstance(n_nulls, bool) or not isinstance(n_nulls, (int, np.integer))
            or int(n_nulls) < 3):
        raise MlsynthConfigError(
            f"n_nulls must be an integer of at least 3; got {n_nulls!r}. A "
            "two-point grid cannot express an interval -- it can only return its "
            "own endpoints, which would read as a band and be an artifact of the "
            "grid.")
    horizon, n_nulls = int(horizon), int(n_nulls)
    if horizon > int(n_leads):
        raise MlsynthDataError(
            f"horizon ({horizon}) exceeds the {int(n_leads)} post-period lead(s) "
            "estimated; there is no full window to accumulate.")

    Xy = np.asarray(Xy, dtype=float)
    trt = np.asarray(trt, dtype=float)
    adopted = np.isfinite(trt)
    if not adopted.any():
        raise MlsynthDataError(
            "cumulative conformal inference needs at least one treated unit; no "
            "unit has a finite adoption time.")

    full = run_multisynth(Xy, trt, d, n_leads, n_lags, fixedeff=fixedeff,
                          time_cohort=time_cohort, nu=nu_used, lam=lam,
                          solver=solver)
    tau = np.asarray(full["tau_rel"], dtype=float)
    groups = full["groups"]
    t0 = int(np.min(trt[adopted]))
    T = Xy.shape[1]

    # Under the null every period is fitting data, so the refit balances all T.
    trt_null = np.full(trt.shape, np.inf)
    trt_null[adopted] = T

    def _resid(fit, k):
        g = fit["groups"][k]
        res = np.asarray(fit["res"][fit["adopt_of"][g]], dtype=float)
        row = fit["members"][g][0]
        return res[row] - np.asarray(fit["weights"][g], dtype=float) @ res

    def _pvalue(theta, row, k):
        adj = Xy.copy()
        adj[row, t0:t0 + horizon] -= theta
        f = run_multisynth(adj, trt_null, T, 1, T, fixedeff=fixedeff,
                           time_cohort=time_cohort, nu=nu_used, lam=lam,
                           solver=solver)
        return moving_block_pvalue(_resid(f, k)[:t0 + horizon], block=horizon,
                                   statistic="mean_abs")

    J = len(groups)
    point = np.empty(J)
    lower = np.empty(J)
    upper = np.empty(J)
    p_zero = np.empty(J)
    for k, g in enumerate(groups):
        row = int(full["members"][g][0])
        total = float(np.nansum(tau[k, :horizon]))
        point[k] = total
        per_period = total / horizon
        pre = _resid(full, k)[:t0]
        half = max(float(grid_scale) * float(np.std(pre, ddof=1)), 1e-9)
        grid = np.linspace(per_period - half, per_period + half, n_nulls)
        pv = np.array([_pvalue(theta, row, k) for theta in grid], dtype=float)
        lo, hi = confidence_set_bounds(grid, pv, float(alpha))
        # An accepted set reaching an end of the grid is bounded by the grid, not
        # by the data: the test would have gone on accepting had it been asked.
        # Reporting the endpoint would understate the band by an amount set by
        # ``grid_scale``, and silently, since it looks like any other number.
        if np.isfinite(lo) and lo <= grid[0]:
            lo = -np.inf
        if np.isfinite(hi) and hi >= grid[-1]:
            hi = np.inf
        lower[k], upper[k] = lo * horizon, hi * horizon
        p_zero[k] = _pvalue(0.0, row, k)
    return point, lower, upper, p_zero
