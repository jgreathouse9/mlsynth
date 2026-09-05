"""Aggregation of the simulation cube for GEOX market selection.

Pure array/groupby reductions on the long p-value cube from
:func:`run_simulations`, faithful to ``GeoLiftMarketSelection``:

1. :func:`compute_power` -- collapse the backtest dimension: power = detection
   rate, plus the backtest-averaged metrics.
2. :func:`compute_mde` -- the minimum detectable effect per (candidate,
   duration), with GeoLift's signed positive/negative selection.
3. :func:`compute_accuracy` -- the same collapse read for error instead of
   detection: how far the design's estimate lands from the truth.

(The composite rank is built on top of the first two.)
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthEstimationError

_POWER_COLUMNS = [
    "candidate", "duration", "effect_size",
    "power", "placebo_mean_effect", "detected_lift", "scaled_l2", "pre_rmspe",
    "pre_rmspe_lambda", "investment",
]

_ACCURACY_COLUMNS = [
    "candidate", "duration", "bias", "error_sd", "rmse", "sigma_mean",
    "calibration_ratio",
]


def compute_power(cube: pd.DataFrame, *, alpha: float = 0.1) -> pd.DataFrame:
    """Collapse the backtest (``sim``) dimension into power + averaged metrics.

    Power is the detection rate ``mean(p_value < alpha)`` over the backtest
    backtests; the other quantities are averaged over the same backtests
    (``scaled_l2`` / ``pre_rmspe`` are constant across ``sim`` only if the panel
    is, so they are averaged for generality).

    Parameters
    ----------
    cube : pd.DataFrame
        Long simulation table from :func:`run_simulations`.
    alpha : float, default 0.1
        Significance level for the detection test.

    Returns
    -------
    pd.DataFrame
        One row per (candidate, duration, effect_size) with ``power``,
        ``placebo_mean_effect``, ``scaled_l2``, ``pre_rmspe``.
    """
    if cube.empty:
        return pd.DataFrame(columns=_POWER_COLUMNS)
    tmp = cube.assign(_detected=(cube["p_value"] < alpha).astype(float))
    has_investment = "investment" in cube.columns
    aggs = dict(
        power=("_detected", "mean"),
        placebo_mean_effect=("placebo_mean_effect", "mean"),
        detected_lift=("detected_lift", "mean"),
        scaled_l2=("scaled_l2", "mean"),
        pre_rmspe=("pre_rmspe", "mean"),
        pre_rmspe_lambda=("pre_rmspe_lambda", "mean"),
    )
    if has_investment:
        # investment = cpic * es * volume is constant across backtests.
        aggs["investment"] = ("investment", "mean")
    # sort=False: candidate keys are frozensets (unorderable).
    return tmp.groupby(
        ["candidate", "duration", "effect_size"], as_index=False, sort=False, observed=True
    ).agg(**aggs)


# The accuracy statistics propagate a nan instead of skipping it, which is
# where they part company with the pandas reductions used elsewhere in this
# module. A backtest whose fit failed leaves a nan error, and skipping it would
# report the accuracy of the backtests that happened to work.
def _mean(values: pd.Series) -> float:
    return float(np.mean(values.to_numpy(dtype=float)))


def _sd(values: pd.Series) -> float:
    return float(np.std(values.to_numpy(dtype=float), ddof=0))


def _rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values.to_numpy(dtype=float)))))


def compute_accuracy(cube: pd.DataFrame) -> pd.DataFrame:
    """Collapse the backtest dimension into the design's estimation error.

    The MDE says the smallest effect a design can detect. It says nothing about
    how far the estimate lands from the truth once an effect is there, and the
    two come apart when the estimator is biased under the design -- for
    synthetic control, a treated region outside the donor hull.

    Each backtest carries one error, ``estimation_error``: the ATT with nothing
    injected. Injecting a lift ``e`` moves the truth and the estimate by the
    same ``e * mean(y_post)``, so that value is the estimate minus the truth at
    every effect size, and the error is a property of the backtest, not of the
    grid. Rows are reduced to one per (candidate, duration, backtest) before
    aggregating, so the numbers below are over backtests however long the grid.

    Over those backtests::

        bias     = mean(error)
        error_sd = std(error)                  # population, so that
        rmse     = sqrt(mean(error ** 2))      # rmse^2 = bias^2 + error_sd^2

    ``calibration_ratio = rmse / sigma_mean`` sets the error against the scale
    the design's own inference tests it at. The two are not the same
    measurement: the placebo standard error is drawn across donor markets
    reassigned as pseudo-treated, while the error varies across backtest
    windows for one fixed region. So the ratio has no value it should sit at,
    and a small one is the ordinary case -- a region's own ATT is usually far
    more stable across windows than the placebo pool is across markets. What it
    catches is the other end: a ratio above one says the design's error exceeds
    what its null admits, so the p-values, and the MDE built from them, are
    anti-conservative. It is ``nan`` where the engine's procedure has no
    standard error, as conformal inference does not.

    ``error_sd`` is a lower bound on across-experiment variability, because
    consecutive backtests shift their window by one period and so overlap
    heavily. On a short backtest set ``rmse`` is dominated by ``bias``.

    A ``nan`` error from a backtest whose fit failed propagates into all three
    statistics. Dropping it would report the accuracy of the backtests that
    happened to work.

    Parameters
    ----------
    cube : pd.DataFrame
        Long simulation table from :func:`run_simulations`.

    Returns
    -------
    pd.DataFrame
        One row per (candidate, duration) with ``bias``, ``error_sd``,
        ``rmse``, ``sigma_mean`` and ``calibration_ratio``.

    Raises
    ------
    MlsynthEstimationError
        If the cube carries no ``estimation_error`` column, which means the
        engine did not report it.
    """
    if cube.empty:
        return pd.DataFrame(columns=_ACCURACY_COLUMNS)
    if "estimation_error" not in cube.columns:
        raise MlsynthEstimationError(
            "the simulation cube carries no estimation_error column, so the "
            "design's accuracy cannot be measured; the engine's sweep must "
            "return tau0.")

    # One row per backtest: the error is constant across the effect grid, so
    # aggregating the long cube directly would weight nothing differently but
    # would state the reduction less plainly.
    per_backtest = cube.drop_duplicates(
        subset=["candidate", "duration", "sim"], keep="first")
    if "placebo_sigma" not in per_backtest.columns:
        # An engine with no standard error reports none; the ratio is then
        # absent, which is a different thing from the error being unmeasured.
        per_backtest = per_backtest.assign(placebo_sigma=np.nan)

    grouped = per_backtest.groupby(
        ["candidate", "duration"], as_index=False, sort=False, observed=True
    ).agg(
        bias=("estimation_error", _mean),
        # ddof=0 keeps rmse^2 = bias^2 + error_sd^2 exact, and gives a single
        # backtest a spread of zero instead of nan.
        error_sd=("estimation_error", _sd),
        rmse=("estimation_error", _rmse),
        sigma_mean=("placebo_sigma", _mean),
    )
    grouped["calibration_ratio"] = np.where(
        grouped["sigma_mean"].to_numpy(dtype=float) > 0,
        grouped["rmse"] / grouped["sigma_mean"],
        np.nan,
    )
    return grouped[_ACCURACY_COLUMNS]


def compute_mde(power_table: pd.DataFrame, *, power_threshold: float = 0.8) -> pd.DataFrame:
    """Minimum detectable effect per (candidate, duration).

    Faithful to GeoLift: among effect sizes whose power exceeds
    ``power_threshold``, take the smallest-magnitude detectable positive and
    negative effects and keep the smaller magnitude (ties -> positive). If
    nothing is detectable, the MDE is ``nan``.

    Parameters
    ----------
    power_table : pd.DataFrame
        Output of :func:`compute_power` (needs ``effect_size`` and ``power``).
    power_threshold : float, default 0.8
        Power a candidate must exceed to "detect" an effect.

    Returns
    -------
    pd.DataFrame
        One row per (candidate, duration) with the signed ``mde``.
    """
    if power_table.empty:
        return pd.DataFrame(columns=["candidate", "duration", "mde"])

    es_min = float(power_table["effect_size"].min())
    es_max = float(power_table["effect_size"].max())

    rows: List[dict] = []
    for (candidate, duration), group in power_table.groupby(
        ["candidate", "duration"], sort=False, observed=True
    ):
        detectable = group.loc[group["power"] > power_threshold, "effect_size"].to_numpy()
        if detectable.size == 0:
            mde = float("nan")
        else:
            negatives = detectable[detectable < 0]
            positives = detectable[detectable > 0]
            # Sentinels mirror GeoLift's min(effect_size)-1 / max(effect_size)+1.
            negative_mde = float(negatives.max()) if negatives.size else (es_min - 1.0)
            positive_mde = float(positives.min()) if positives.size else (es_max + 1.0)
            if positive_mde > abs(negative_mde) and negative_mde != 0:
                mde = negative_mde
            else:
                mde = positive_mde
        rows.append({"candidate": candidate, "duration": duration, "mde": mde})
    return pd.DataFrame(rows)


def compute_exact_mde(cube: pd.DataFrame, *,
                      power_threshold: float = 0.8) -> pd.DataFrame:
    """The effect a design actually detects, per (candidate, duration).

    ``compute_mde`` reports the smallest simulated effect whose power clears the
    threshold, so its resolution is the effect grid's step: a design that truly
    detects 0.1014 reports 0.150 on a 0.05 grid, because 0.1014 misses the 0.10
    point. This reports the crossing itself.

    The per-backtest boundaries come from the engine (closed form where the
    p-value is analytic in the effect). Power above the threshold means at least
    ``k = floor(power_threshold * n_sims) + 1`` backtests detect, so the design's
    boundary is the k-th smallest of them -- an order statistic, which is why
    this works on the pre-aggregation cube instead of riding through
    :func:`compute_power`, which averages.

    Both directions are kept. The backtest's own placebo effect is generally
    nonzero, so the detection interval sits off centre and a design can need
    less of a push downward than upward.

    Returns ``nan`` for a design whose boundaries are not finite, which is what
    an engine reports when its p-value is not analytic in the effect.
    """
    if cube.empty or "boundary_up" not in cube.columns:
        return pd.DataFrame(columns=["candidate", "duration",
                                     "mde_exact_up", "mde_exact_down"])

    rows: List[dict] = []
    per_sim = cube.drop_duplicates(subset=["candidate", "duration", "sim"])
    for (candidate, duration), group in per_sim.groupby(
        ["candidate", "duration"], sort=False, observed=True
    ):
        ups = group["boundary_up"].to_numpy(dtype=float)
        downs = group["boundary_down"].to_numpy(dtype=float)
        n_sims = ups.shape[0]
        k = int(np.floor(power_threshold * n_sims)) + 1
        up = down = float("nan")
        finite_up = np.sort(ups[np.isfinite(ups)])
        finite_down = np.sort(downs[np.isfinite(downs)])[::-1]
        if finite_up.size >= k:
            up = float(finite_up[k - 1])
        if finite_down.size >= k:
            down = float(finite_down[k - 1])
        rows.append({"candidate": candidate, "duration": duration,
                     "mde_exact_up": up, "mde_exact_down": down})
    return pd.DataFrame(rows)


_RANK_COLUMNS = [
    "candidate", "duration", "mde", "power", "detected_lift", "abs_lift_in_zero",
    "scaled_l2", "pre_rmspe", "pre_rmspe_lambda", "investment",
    "rank_mde", "rank_pvalue", "rank_abszero", "rank",
]


def compute_rank(power_table: pd.DataFrame, *, power_threshold: float = 0.8,
                 budget: Optional[float] = None) -> pd.DataFrame:
    """Rank candidate designs, haircut-faithful to ``GeoLiftMarketSelection``.

    Builds the per-(candidate, duration) MDE row, then the GeoLift composite
    rank: the mean of three ``dense_rank`` components -- ``|mde|``, ``power`` (at
    the MDE; ascending, as in GeoLift), and ``abs_lift_in_zero`` (the recovery
    error ``|AvgDetectedLift - mde|`` at the MDE). ``scaled_l2`` / ``pre_rmspe``
    are carried for reporting but are **not** ranked. Candidates with no
    detectable effect (``mde`` NaN) are dropped. Lower ``rank`` = better.

    Parameters
    ----------
    power_table : pd.DataFrame
        Output of :func:`compute_power`.
    power_threshold : float, default 0.8
        Forwarded to :func:`compute_mde`.

    Returns
    -------
    pd.DataFrame
        One row per ranked (candidate, duration), sorted by ``rank``.
    """
    if power_table.empty:
        return pd.DataFrame(columns=_RANK_COLUMNS)

    # CPIC budget gate (GeoLift: filter abs(budget) > abs(Investment)). Drop the
    # over-budget effect-size rows *before* the MDE, so a candidate whose
    # cheapest detectable effect still busts the budget falls out entirely.
    if budget is not None and "investment" in power_table.columns:
        power_table = power_table[
            power_table["investment"].abs() < abs(budget)
        ].copy()
        if power_table.empty:
            return pd.DataFrame(columns=_RANK_COLUMNS)

    mde_table = compute_mde(power_table, power_threshold=power_threshold)
    merged = mde_table.merge(power_table, on=["candidate", "duration"])
    # Keep only each group's MDE row (drops NaN-MDE groups: NaN != any effect).
    at_mde = merged[merged["effect_size"] == merged["mde"]].copy()
    if at_mde.empty:
        return pd.DataFrame(columns=_RANK_COLUMNS)
    if "investment" not in at_mde.columns:          # cpic not supplied
        at_mde["investment"] = float("nan")

    at_mde["abs_lift_in_zero"] = (at_mde["detected_lift"] - at_mde["mde"]).abs().round(3)

    # GeoLift dense ranks (ascending; lower value -> rank 1). Note rank_pvalue
    # ranks power ascending, i.e. lower power-at-MDE ranks better.
    at_mde["rank_mde"] = at_mde["mde"].abs().rank(method="dense")
    at_mde["rank_pvalue"] = at_mde["power"].rank(method="dense")
    at_mde["rank_abszero"] = at_mde["abs_lift_in_zero"].rank(method="dense")
    mean_rank = at_mde[["rank_mde", "rank_pvalue", "rank_abszero"]].mean(axis=1)
    at_mde["rank"] = mean_rank.rank(method="min")

    return at_mde[_RANK_COLUMNS].sort_values("rank").reset_index(drop=True)
