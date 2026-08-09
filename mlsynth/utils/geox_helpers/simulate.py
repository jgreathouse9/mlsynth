"""One pseudo-experiment for GEOX market-selection scoring.

A single backtest: fit SDID on the backtest's pre-period, take the
placebo standard error once, then sweep the effect sizes. Because neither SDID
weight program reads the treated post block (see
:mod:`~mlsynth.utils.geox_helpers.engine`), the sweep is exact arithmetic on
the ATT and the placebo sigma is shared across it -- so the whole grid of effect
sizes costs one fit and one placebo run.

The row schema matches what
:func:`~mlsynth.utils.geox_helpers.aggregate.compute_power` consumes, one row
per effect size.
"""

from typing import List, Optional

import numpy as np

from ...exceptions import MlsynthConfigError
from .engines import resolve_engine
from .windows import backtest_pre_periods, backtest_treatment_window


def inject_effect(treated, start: int, end: int, es: float) -> np.ndarray:
    """Scale the block ``[start, end]`` of ``treated`` by ``(1 + es)``.

    The multiplicative injection GeoLift uses (``Y[D == 1] *= 1 + es``), so an
    effect size reads as a percentage lift on the treated markets' own volume.
    Returns a new array; the input is never mutated.

    Raises
    ------
    MlsynthConfigError
        If ``[start, end]`` is not a valid in-bounds, non-empty window.
    """
    t = np.asarray(treated, dtype=float).ravel()
    n = t.shape[0]
    if not (0 <= start <= end < n):
        raise MlsynthConfigError(
            f"invalid post window [{start}, {end}] for a series of length {n}.")
    out = t.copy()
    out[start:end + 1] = out[start:end + 1] * (1.0 + float(es))
    return out


def simulate_backtest(
    treated, donors, n_periods: int, duration: int, sim: int, effect_sizes,
    *, n_draws: int = 200, n_tr: int = 1, seed: int = 0,
    cpic: Optional[float] = None, treated_total: Optional[np.ndarray] = None,
    analytic: bool = True, engine: str = "sdid",
    engine_kwargs: Optional[dict] = None, alpha: float = 0.1,
) -> List[dict]:
    """Simulate one backtest across a grid of effect sizes.

    Parameters
    ----------
    treated : array-like, shape (n_periods,)
        Aggregated treated series over the full panel.
    donors : array-like, shape (n_periods, J)
        Donor pool over the full panel.
    n_periods, duration, sim : int
        Panel length, pseudo-treatment duration, and backtest index.
    effect_sizes : iterable of float
        Effect sizes to sweep.
    n_draws : int
        Placebo draws behind the standard error.
    n_tr : int
        Number of treated markets in the candidate.
    cpic : float, optional
        Cost per incremental conversion. When given, each row reports the
        required ``cpic * effect_size * summed-treated-volume`` investment.
    treated_total : np.ndarray, optional
        Summed treated series for the investment volume, which stays a total
        even when the fit runs on the per-market mean.
    analytic : bool, default True
        Use the closed-form ``tau(es) = tau(0) + es * mean(y_post)``. Setting
        this to ``False`` re-injects and recomputes for each effect size, which
        gives the same answer and exists so the tests can prove the shortcut.

    Returns
    -------
    list of dict
        One row per effect size, carrying ``sim``, ``duration``,
        ``effect_size``, ``p_value``, ``placebo_mean_effect`` (the SDID ATT),
        ``detected_lift`` (ATT over the counterfactual post mean),
        ``scaled_l2``, ``pre_rmspe`` and ``investment``.

    Raises
    ------
    MlsynthConfigError
        If the backtest runs off the start of the panel, or ``treated`` /
        ``donors`` do not have ``n_periods`` rows.
    """
    n_pre = backtest_pre_periods(n_periods, duration, sim)
    start, end = backtest_treatment_window(n_periods, duration, sim)

    treated_arr = np.asarray(treated, dtype=float).ravel()
    donors_arr = np.asarray(donors, dtype=float)
    if treated_arr.shape[0] != n_periods or donors_arr.shape[0] != n_periods:
        raise MlsynthConfigError(
            f"treated and donors must both have n_periods={n_periods} rows; got "
            f"{treated_arr.shape[0]} and {donors_arr.shape[0]}.")

    eng = resolve_engine(engine)
    ekw = dict(engine_kwargs or {})
    fit = eng.fit_once(treated_arr, donors_arr, n_pre, start, end, n_tr, **ekw)
    cf_post_mean = float(np.mean(fit.counterfactual[start:end + 1]))

    # The engine owns the effect grid: what can be hoisted out of it (a placebo
    # draw) and what cannot (a conformal permutation) differs by procedure.
    swept = eng.sweep_p_values(fit, treated_arr, donors_arr, n_pre, start, end,
                               list(effect_sizes), n_draws=n_draws, n_tr=n_tr,
                               seed=seed, analytic=analytic, alpha=alpha,
                               **ekw)

    total_arr = (np.asarray(treated_total, dtype=float).ravel()
                 if treated_total is not None else treated_arr)
    window_volume = float(np.sum(total_arr[start:end + 1]))

    rows: List[dict] = []
    for es, tau, p_value in zip(effect_sizes, swept["tau"], swept["p_value"]):
        rows.append({
            "sim": sim,
            "duration": duration,
            "effect_size": float(es),
            "p_value": p_value,
            "placebo_mean_effect": tau,
            "detected_lift": (tau / cf_post_mean if cf_post_mean != 0.0
                              else float("nan")),
            "scaled_l2": fit.scaled_l2,
            "pre_rmspe": fit.pre_rmspe,
            "pre_rmspe_lambda": float(fit.extras.get("pre_rmspe_lambda",
                                                     float("nan"))),
            "boundary_up": swept.get("boundary_up", float("nan")),
            "boundary_down": swept.get("boundary_down", float("nan")),
            "investment": (cpic * float(es) * window_volume
                           if cpic is not None else float("nan")),
        })
    return rows
