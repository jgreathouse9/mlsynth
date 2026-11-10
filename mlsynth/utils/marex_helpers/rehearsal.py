"""MAREX power by rehearsal: run the planned test where nothing happened (#947).

The analytic MDEs (the pool's Newey-West formula and ``post_fit.power``) price a
normal test of a mean from an in-sample noise scale. On simulated panels with a
persistent regional factor both promised a test about twice as sensitive as the
cumulative band MAREX reports: a 0.68% and a 0.95% MDE for a 15-period test whose
realized MDE was 1.55%.

Before launch nothing is treated, so the test can be run on the panel itself:

1. Draw ``n_placebos`` random subsets of the markets.
2. On each, solve MAREX's design with the launch moved ``horizon`` periods
   earlier. Run the configured cumulative band on the ``horizon`` pseudo-post
   periods, which are the last pre-launch periods.

Every placebo's estimate is an error, since nothing happened. A relative effect
``e`` on the test markets moves the composite's total by ``e`` times the
composite's pseudo-post total, and both cumulative bands move with an additive
change in the post period. So one band per placebo prices every effect size: the
placebo rejects at ``e`` when ``0`` is outside its band shifted by ``e``.

On the same panels, placebos drawing 85% of the markets predicted a 1.80% MDE.
At each panel's predicted MDE the realized test rejected 84% of the time, against
a target of 80%. The rehearsal does not rank panels by difficulty: it is a
planning number for a design setting, not a forecast of one launch's precision.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError
from .structures import MAREXPowerRehearsal
from .windows import MIN_FIT_PERIODS, default_blank_periods

DIRECTIONS = ("drop", "lift")


def _bands(lower, upper):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise MlsynthDataError(
            f"lower and upper must have the same length; got {lower.shape} and "
            f"{upper.shape}.")
    if lower.size == 0:
        raise MlsynthDataError("A rehearsal needs at least one placebo band.")
    if np.isnan(lower).any() or np.isnan(upper).any():
        raise MlsynthDataError("A placebo band has a NaN end.")
    if np.any(lower > upper):
        raise MlsynthDataError("A placebo band has its lower end above its upper end.")
    return lower, upper


def rehearsal_power(lower, upper, effect: float) -> float:
    """Share of placebo bands that exclude zero once shifted by ``effect``.

    ``lower`` and ``upper`` are the placebos' bands as shares of their
    composites' totals; ``effect`` is relative and signed (``-0.02`` is a 2%
    drop). An unbounded end never rejects.
    """
    lower, upper = _bands(lower, upper)
    e = float(effect)
    return float(np.mean((upper + e < 0) | (lower + e > 0)))


def rehearsal_mde(lower, upper, target: float, direction: str) -> float:
    """The smallest effect, as a positive share, past which power stays at ``target``.

    For a drop of size ``m`` a placebo rejects when ``m > upper`` or
    ``m < lower``; for a lift, when ``m > -lower`` or ``m < -upper``. Power is
    constant between those ends, so it is read once per interval. The MDE is the
    right end of the last interval where power falls short; ``0`` when none
    does, ``inf`` when the target is never reached.
    """
    lower, upper = _bands(lower, upper)
    if not 0.0 < float(target) < 1.0:
        raise MlsynthDataError(
            f"The power target must be strictly between 0 and 1; got {target}.")
    if direction not in DIRECTIONS:
        raise MlsynthDataError(f"direction must be 'drop' or 'lift'; got {direction!r}.")
    above, below = (upper, lower) if direction == "drop" else (-lower, -upper)
    ends = np.concatenate([above, below])
    points = np.unique(np.concatenate([[0.0], ends[np.isfinite(ends) & (ends > 0)]]))
    probes = np.append((points[:-1] + points[1:]) / 2.0, points[-1] + 1.0)
    power = np.array([np.mean((m > above) | (m < below)) for m in probes])
    short = np.flatnonzero(power < float(target))
    if short.size == 0:
        return 0.0
    if short[-1] == probes.size - 1:
        return float("inf")
    return float(points[short[-1] + 1])


def rehearse_power(
    Y_pre,
    *,
    horizon: int,
    n_treated: int,
    n_placebos: int,
    share: float,
    seed: int,
    alpha: float,
    power_target: float,
    cumulative_method: str,
    blank_periods: int = 0,
    covariates: Optional[np.ndarray] = None,
    solve_kwargs: Optional[Dict[str, Any]] = None,
) -> MAREXPowerRehearsal:
    """Rehearse a ``horizon``-period MAREX test on the pre-launch panel.

    Parameters
    ----------
    Y_pre : np.ndarray
        Outcomes before the real launch, shape ``(N, T0)``. Nothing later is read.
    horizon : int
        Periods the planned test runs.
    n_treated : int
        Test markets in each placebo design: the real design's count.
    n_placebos, share, seed : int, float, int
        How many placebos, the share of the markets each draws, and the seed.
    alpha, power_target : float
        The test's level and the power the MDEs are read at.
    cumulative_method : str
        The cumulative band to rehearse: the one the real test reports.
    blank_periods : int
        The real design's blank window, reused at the pseudo-launch; ``0``
        takes MAREX's default for the shorter pre-period.
    covariates : np.ndarray, optional
        Time-invariant covariates, shape ``(N, R)``, subset with the markets.
    solve_kwargs : dict, optional
        The design's own settings (design, penalties, backend, ...), passed to
        :func:`~mlsynth.utils.marex_helpers.orchestration.solve_marex`.

    Raises
    ------
    MlsynthConfigError
        When a placebo would draw every market or too few to hold a design, or
        the pseudo-launch leaves too few periods to fit and hold out.
    MlsynthDataError
        When a placebo's composite totals zero or less over the pseudo-post
        periods, where a relative effect has no meaning.
    """
    from .orchestration import solve_marex

    Y_pre = np.asarray(Y_pre, dtype=float)
    N, T0 = Y_pre.shape
    k = int(round(float(share) * N))
    if k >= N:
        raise MlsynthConfigError(
            f"power_share={share} draws all {N} markets for every placebo, so "
            f"every placebo is the same design; lower it.")
    if k < n_treated + 2:
        raise MlsynthConfigError(
            f"power_share={share} draws {k} markets per placebo; a design of "
            f"{n_treated} test markets needs at least {n_treated + 2}, two of "
            f"them controls.")
    T0p = T0 - int(horizon)
    if blank_periods:
        blank = int(blank_periods)
        if T0p - blank < MIN_FIT_PERIODS:
            raise MlsynthConfigError(
                f"power_horizon={horizon} moves the pseudo-launch to period {T0p}, "
                f"which leaves {T0p - blank} fitting periods before the "
                f"{blank} blank ones; MAREX needs at least {MIN_FIT_PERIODS}.")
    else:
        try:
            blank = default_blank_periods(Y_pre[:, :max(T0p, 1)], alpha)
        except MlsynthConfigError as exc:
            raise MlsynthConfigError(
                f"power_horizon={horizon} moves the pseudo-launch to period "
                f"{T0p}, too early to fit and hold out a blank window: {exc}") from exc

    rng = np.random.default_rng(int(seed))
    kwargs = dict(solve_kwargs or {})
    lower, upper, point = [], [], []
    for _ in range(int(n_placebos)):
        sub = np.sort(rng.choice(N, k, replace=False))
        res = solve_marex(
            Y_full=Y_pre[sub], T0=T0p, clusters=np.zeros(k, dtype=int),
            blank_periods=blank, m_eq=int(n_treated), inference=True,
            alpha=alpha, T_post=int(horizon), cumulative_band=True,
            cumulative_method=cumulative_method,
            covariates=None if covariates is None else np.asarray(covariates)[sub],
            **kwargs)
        g = res.globres
        band = g.inference.cumulative_band
        total = float(np.sum(np.asarray(g.synthetic_treated)[T0p:T0p + int(horizon)]))
        if not total > 0:
            raise MlsynthDataError(
                f"A placebo's test composite totals {total} over the "
                f"{horizon} pseudo-post periods; a relative effect needs a "
                f"positive total, so rehearse on a positive outcome such as sales.")
        lower.append(band.lower / total)
        upper.append(band.upper / total)
        point.append(band.point / total)
    lower, upper = np.array(lower), np.array(upper)
    return MAREXPowerRehearsal(
        horizon=int(horizon), n_placebos=int(n_placebos), n_units=k,
        share=float(share), seed=int(seed), alpha=float(alpha),
        power_target=float(power_target), cumulative_method=str(cumulative_method),
        lower=lower, upper=upper, estimates=np.array(point),
        mde_drop=rehearsal_mde(lower, upper, power_target, "drop"),
        mde_lift=rehearsal_mde(lower, upper, power_target, "lift"))
