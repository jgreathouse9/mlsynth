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


# --- the Recast geo-experiment simulation study ---------------------------
#
# The data-generating process of getrecast/geolift-simulation-study, ported
# from ``src/R/generate_panels.R``. That study runs 1000 Monte Carlo iterations
# per cell across four stress scenarios and publishes per-tool bias, coverage
# and false-positive rates, with GeoLift among the tools and configured as
# augmented SC (ridge) with block conformal inference -- which is GEOX's
# augsynth engine. Their published GeoLift row is therefore an external
# referent for this engine's inference, which is why it is ported here instead
# of a DGP being invented.
#
# Scenario overrides, from their `scenarios` list:
#     A1 textbook          defaults
#     A2 outlier           treated geo's baseline inflated 5x after selection
#     A3 small pool        9 controls instead of 20
#     A4 short pre-period  45 total days, 30 pre

_DOW_PROFILE = (-1.0, -0.5, 0.0, 0.2, 0.8, 1.0, 0.5)   # Mon..Sun, day 1 = Mon

RECAST_SCENARIOS = {
    "A1": {},
    "A2": {"outlier": True},
    "A3": {"n_control": 9},
    "A4": {"total_days": 45, "pre_days": 30},
}


def simulate_recast_panel(scenario: str, effect_pct: float, seed: int, *,
                          n_control: int = 20, total_days: int = 105,
                          pre_days: int = 90, baseline_mean: float = 4000.0,
                          baseline_spread: float = 0.6,
                          trend_slope: float = 0.001,
                          seasonality_amplitude: float = 0.10,
                          autocorrelation: float = 0.30,
                          noise_level: float = 0.20,
                          outlier_multiplier: float = 5.0):
    """One panel from the Recast study's DGP.

    ``Y_cf[i,t] = baseline_i * trend_t * season_t *
    exp(noise_level * scale_i * ar_noise[i,t])`` with
    ``scale_i = sqrt(noise_baseline_i / mean(noise_baselines))``, and the
    treated geo's post window multiplied by ``1 + effect_pct``. Every geo shares
    the trend and the weekly profile; only the baseline level and the noise draw
    differ, which is the study's stated cross-geo structure (and its stated
    caveat -- donors move in lockstep up to noise).

    ``exp`` keeps the panel strictly positive whatever the noise draw, so a
    multiplicative lift is always well defined.

    The treated geo is the one closest to the median baseline, chosen before any
    scenario modification; A2's inflation is applied to that geo afterwards and
    deliberately does not feed the noise scaling, matching their
    ``noise_baselines`` argument.

    Returns
    -------
    (wide, treated, cf)
        ``wide`` -- ``(total_days, n_geos)`` observed outcomes, columns
        ``City 1 ...``; ``treated`` -- the treated column's name; ``cf`` -- the
        treated geo's untreated counterfactual, so the true ATT is exact and
        not approximated.
    """
    import pandas as pd

    over = dict(RECAST_SCENARIOS.get(scenario, {}))
    n_control = over.get("n_control", n_control)
    total_days = over.get("total_days", total_days)
    pre_days = over.get("pre_days", pre_days)
    outlier = over.get("outlier", False)

    n_geos = 1 + n_control
    rng = np.random.default_rng(seed)

    # Log-normal baselines with the mean pinned to `baseline_mean`, sorted.
    meanlog = np.log(baseline_mean) - baseline_spread ** 2 / 2.0
    baselines = np.sort(rng.lognormal(meanlog, baseline_spread, size=n_geos))
    names = [f"City {i + 1}" for i in range(n_geos)]

    # Treated = closest to the median baseline, before any inflation.
    t_idx = int(np.argmin(np.abs(baselines - np.median(baselines))))
    noise_baselines = baselines.copy()          # A2 must not amplify the noise
    if outlier:
        baselines = baselines.copy()
        baselines[t_idx] *= outlier_multiplier

    t_seq = np.arange(1, total_days + 1)
    trend = 1.0 + trend_slope * t_seq
    season = 1.0 + seasonality_amplitude * np.asarray(
        [_DOW_PROFILE[(t - 1) % 7] for t in t_seq])

    mean_baseline = float(noise_baselines.mean())
    Y_cf = np.empty((total_days, n_geos))
    for i in range(n_geos):
        innov = rng.normal(size=total_days)
        noise = np.empty(total_days)
        noise[0] = innov[0]
        for t in range(1, total_days):
            noise[t] = autocorrelation * noise[t - 1] + innov[t]
        scale_i = np.sqrt(noise_baselines[i] / mean_baseline)
        Y_cf[:, i] = (baselines[i] * trend * season
                      * np.exp(noise_level * scale_i * noise))

    Y = Y_cf.copy()
    Y[pre_days:, t_idx] = Y_cf[pre_days:, t_idx] * (1.0 + effect_pct)

    wide = pd.DataFrame(Y, columns=names,
                        index=pd.RangeIndex(1, total_days + 1, name="date"))
    return wide, names[t_idx], Y_cf[:, t_idx], pre_days
