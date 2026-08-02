"""Ingestion for FSC: a long panel of (unit, time, argument, value) rows.

Each argument slice is a perfectly ordinary scalar panel, so each goes through
:func:`mlsynth.utils.datautils.dataprep` and the slices are stacked along the
argument axis. That is the same construction :mod:`mlsynth.utils.compsc_helpers`
uses for its share columns, and it means the treated/donor split, the period
ordering and the pre-period count are the library's rather than this module's.

What this module adds on top is the grid contract: every ``(unit, time)`` cell
must be observed at the same argument values, in the same order. A panel that
fails that is not a panel of objects and there is nothing sensible to embed.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import FSCInputs


def prepare_fsc_inputs(df: pd.DataFrame, outcome: str, treat: str, unitid: str,
                       time: str, argument: str) -> FSCInputs:
    """Pivot a long panel of objects into an ``(N, T, M)`` cube.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per (unit, time, argument).
    outcome, treat, unitid, time, argument : str
        Column names. ``treat`` is constant within a ``(unit, time)`` cell.

    Returns
    -------
    FSCInputs

    Raises
    ------
    MlsynthDataError
        A column is missing or non-numeric, the grid is ragged, values are not
        finite, or the panel has no treated unit, no donors, or no pre-periods.
    """
    missing = [c for c in (outcome, treat, unitid, time, argument)
               if c not in df.columns]
    if missing:
        raise MlsynthDataError(
            f"column(s) {missing} not in the panel; available: "
            f"{list(df.columns)}. Note FSC needs an argument column naming "
            "the point at which each outcome value is observed.")

    for column in (outcome, argument):
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise MlsynthDataError(
                f"column {column!r} is not numeric; FSC needs a numeric "
                "outcome observed on a numeric argument grid.")

    grid = np.sort(pd.unique(df[argument].to_numpy()))
    if grid.size < 2:
        raise MlsynthDataError(
            f"the argument column {argument!r} takes {grid.size} distinct "
            "value(s); an FSC outcome is an object sampled at several points, "
            "so at least two are needed. With a single value per cell the "
            "outcome is a scalar -- use VanillaSC.")

    counts = df.groupby([unitid, time], sort=False)[argument].agg(
        ["count", "nunique"])
    bad = counts[(counts["count"] != grid.size)
                 | (counts["nunique"] != grid.size)]
    if not bad.empty:
        cell = bad.index[0]
        raise MlsynthDataError(
            f"the argument grid is ragged: {len(bad)} (unit, time) cell(s) are "
            f"not observed at all {grid.size} argument values, e.g. {cell} "
            f"with {int(bad.iloc[0]['nunique'])} distinct value(s). FSC needs "
            "every cell on a common grid.")

    slices: List[np.ndarray] = []
    pre_periods = 0
    unit_names: List = []
    time_labels: List = []
    for value in grid:
        prepped = dataprep(
            df[df[argument] == value].drop(columns=[argument]),
            unit_id_column_name=unitid, time_period_column_name=time,
            outcome_column_name=outcome, treatment_indicator_column_name=treat,
        )
        treated = np.asarray(prepped["y"], dtype=float).ravel()
        donors = np.asarray(prepped["donor_matrix"], dtype=float)
        slices.append(np.vstack([treated, donors.T]))
        pre_periods = int(prepped["pre_periods"])
        unit_names = [prepped["treated_unit_name"], *prepped["donor_names"]]
        time_labels = list(prepped.get("time_labels", range(treated.size)))

    values = np.stack(slices, axis=-1)                       # (N, T, M)
    if not np.all(np.isfinite(values)):
        raise MlsynthDataError(
            "the panel carries NaN or inf in the outcome; FSC needs complete "
            "objects -- every cell observed at every argument value.")
    if values.shape[0] < 2:
        raise MlsynthDataError(
            "no donor units: FSC needs at least one untreated unit to form "
            "the counterfactual object.")
    if pre_periods < 1:
        raise MlsynthDataError(
            "no pre-treatment periods: treatment is in force from the first "
            "period. FSC matches on the pre-treatment objects, so at least "
            "one pre-period is required.")

    return FSCInputs(values=values, grid=grid, unit_names=unit_names,
                     time_labels=time_labels, pre_periods=pre_periods,
                     outcome=outcome, argument=argument)
