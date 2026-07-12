"""Grouped-microdata preparation for the SCD estimator.

Converts a long-format micro-panel (one row per individual observation, with
an optional survey-weight column) into :class:`SCDInputs`: survey-weighted
group means and weighted cell totals for the point estimator, plus the raw
per-individual ``(group, time, outcome, weight)`` arrays for the
influence-function variance.

Treatment is applied at the unit-time level via the ``treat`` column, exactly
as in classical SCM: the treated unit is the unique unit with ``treat == 1``
for some period. Its first treated period defines ``Tstar``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import SCDInputs


def prepare_scd_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    weight_col: Optional[str] = None,
) -> SCDInputs:
    """Pivot a long-format grouped micro-panel into :class:`SCDInputs`.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel, one row per individual observation, with columns
        ``unitid``, ``time``, ``outcome``, ``treat`` (unit-time level), and
        optionally ``weight_col``.
    outcome, treat, unitid, time : str
        Column names.
    weight_col : str, optional
        Individual survey-weight column. Unweighted (weight 1) if ``None``.
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(f"Required column {col!r} missing from input DataFrame.")
    if weight_col is not None and weight_col not in df.columns:
        raise MlsynthDataError(f"Weight column {weight_col!r} missing from input DataFrame.")
    if df[outcome].isna().any():
        raise MlsynthDataError("Outcome column contains NaN values.")

    # Treated unit: identified from the treat column at the unit-time level.
    treat_by_cell = df.groupby([unitid, time])[treat].max().reset_index()
    treated_cells = treat_by_cell[treat_by_cell[treat] == 1]
    if treated_cells.empty:
        raise MlsynthDataError(
            f"No treated (unit, time) cell found -- need at least one row with {treat}=1."
        )
    treated_units = treated_cells[unitid].unique()
    if treated_units.size != 1:
        raise MlsynthDataError(
            f"SCD requires exactly one treated unit; found {treated_units.tolist()}."
        )
    treated_name = treated_units[0]

    time_labels = np.array(sorted(df[time].unique()))
    Ttot = int(time_labels.size)

    treated_sorted = (
        treat_by_cell[treat_by_cell[unitid] == treated_name].sort_values(time).reset_index(drop=True)
    )
    treat_series = treated_sorted[treat].to_numpy()
    if treat_series[0] == 1:
        raise MlsynthDataError("Treated unit has no pre-period (treat=1 at the earliest time).")
    # The treated unit is selected from cells with treat==1, so treat_series
    # always contains a 1 (index >= 1, since index 0 is checked above); hence
    # T0 = argmax >= 1 and Ttot - T0 >= 1. The two guards below are defensive.
    if not treat_series.any():  # pragma: no cover - treated unit always has a 1
        raise MlsynthDataError("Treated unit never receives treatment (treat is 0 at every t).")
    T0 = int(np.argmax(treat_series == 1))
    Tstar = T0 + 1
    if T0 < 1:  # pragma: no cover - index 0 is checked, so argmax >= 1
        raise MlsynthDataError("SCD needs at least one pre-treatment period.")
    if Ttot - T0 < 1:  # pragma: no cover - T0 <= Ttot-1, so a post-period exists
        raise MlsynthDataError("SCD needs at least one post-treatment period.")

    donor_names = [u for u in sorted(df[unitid].unique()) if u != treated_name]
    if not donor_names:
        raise MlsynthDataError("SCD needs at least one donor unit.")
    groups = [treated_name] + donor_names
    gid = {u: i for i, u in enumerate(groups)}
    K = len(donor_names)

    # Per-individual arrays.
    G = df[unitid].map(gid).to_numpy()
    t_idx = pd.Index(time_labels).get_indexer(df[time].to_numpy()) + 1  # 1-based period
    Y = df[outcome].to_numpy(dtype=float)
    wgt = (np.ones(len(df)) if weight_col is None else df[weight_col].to_numpy(dtype=float))
    if np.any(wgt < 0) or np.isnan(wgt).any():
        raise MlsynthDataError("Survey weights must be non-negative and non-NaN.")

    # Survey-weighted group means and weighted cell totals ((K+1) x Ttot).
    den = np.zeros((K + 1, Ttot))
    wsum = np.zeros((K + 1, Ttot))
    np.add.at(den, (G, t_idx - 1), wgt)
    np.add.at(wsum, (G, t_idx - 1), wgt * Y)
    if np.any(den == 0):
        empty = np.argwhere(den == 0)
        raise MlsynthDataError(
            f"Empty (group, period) cell(s); first: group {empty[0, 0]}, period "
            f"{time_labels[empty[0, 1]]!r}. SCD needs every cell populated."
        )
    group_means = wsum / den

    return SCDInputs(
        treated_name=treated_name,
        donor_names=donor_names,
        group_means=group_means,
        n_jt=den,
        G=G,
        t=t_idx,
        Y=Y,
        weight=wgt,
        time_labels=time_labels,
        T0=T0,
        Tstar=Tstar,
        Ttot=Ttot,
    )
