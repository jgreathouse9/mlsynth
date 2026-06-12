"""Aggregation of the simulation cube for GeoLift market selection.

Pure array/groupby reductions on the long p-value cube from
:func:`run_simulations`, faithful to ``GeoLiftMarketSelection``:

1. :func:`compute_power` -- collapse the lookback dimension: power = detection
   rate, plus the lookback-averaged metrics.
2. :func:`compute_mde` -- the minimum detectable effect per (candidate,
   duration), with GeoLift's signed positive/negative selection.

(The composite rank is built on top of these.)
"""

from typing import List

import numpy as np
import pandas as pd

_POWER_COLUMNS = [
    "candidate", "duration", "effect_size",
    "power", "placebo_mean_effect", "detected_lift", "scaled_l2", "pre_rmspe",
]


def compute_power(cube: pd.DataFrame, *, alpha: float = 0.1) -> pd.DataFrame:
    """Collapse the lookback (``sim``) dimension into power + averaged metrics.

    Power is the detection rate ``mean(p_value < alpha)`` over the lookback
    placements; the other quantities are averaged over the same placements
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
    # sort=False: candidate keys are frozensets (unorderable).
    return tmp.groupby(
        ["candidate", "duration", "effect_size"], as_index=False, sort=False
    ).agg(
        power=("_detected", "mean"),
        placebo_mean_effect=("placebo_mean_effect", "mean"),
        detected_lift=("detected_lift", "mean"),
        scaled_l2=("scaled_l2", "mean"),
        pre_rmspe=("pre_rmspe", "mean"),
    )


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
        ["candidate", "duration"], sort=False
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


_RANK_COLUMNS = [
    "candidate", "duration", "mde", "power", "detected_lift", "abs_lift_in_zero",
    "scaled_l2", "pre_rmspe",
    "rank_mde", "rank_pvalue", "rank_abszero", "rank",
]


def compute_rank(power_table: pd.DataFrame, *, power_threshold: float = 0.8) -> pd.DataFrame:
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

    mde_table = compute_mde(power_table, power_threshold=power_threshold)
    merged = mde_table.merge(power_table, on=["candidate", "duration"])
    # Keep only each group's MDE row (drops NaN-MDE groups: NaN != any effect).
    at_mde = merged[merged["effect_size"] == merged["mde"]].copy()
    if at_mde.empty:
        return pd.DataFrame(columns=_RANK_COLUMNS)

    at_mde["abs_lift_in_zero"] = (at_mde["detected_lift"] - at_mde["mde"]).abs().round(3)

    # GeoLift dense ranks (ascending; lower value -> rank 1). Note rank_pvalue
    # ranks power ascending, i.e. lower power-at-MDE ranks better.
    at_mde["rank_mde"] = at_mde["mde"].abs().rank(method="dense")
    at_mde["rank_pvalue"] = at_mde["power"].rank(method="dense")
    at_mde["rank_abszero"] = at_mde["abs_lift_in_zero"].rank(method="dense")
    mean_rank = at_mde[["rank_mde", "rank_pvalue", "rank_abszero"]].mean(axis=1)
    at_mde["rank"] = mean_rank.rank(method="min")

    return at_mde[_RANK_COLUMNS].sort_values("rank").reset_index(drop=True)
