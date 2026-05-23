"""Panel ingestion for the CTSC estimator.

Unlike a binary-treatment estimator, CTSC takes one or more
**treatment / explanatory** columns (continuous or discrete) that vary
over units and time, and pivots them into a ``(n, T, K)`` array
alongside the ``(n, T)`` outcome.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from .structures import CTSCInputs


def prepare_ctsc_inputs(
    df: pd.DataFrame,
    outcome: str,
    treatment_vars: Sequence[str],
    unitid: str,
    time: str,
    population_col: Optional[str] = None,
) -> CTSCInputs:
    """Pivot a long panel into :class:`CTSCInputs`.

    Parameters
    ----------
    df : pd.DataFrame
        Balanced long panel; one row per ``(unit, time)``.
    outcome : str
        Outcome column.
    treatment_vars : sequence of str
        The ``K >= 1`` treatment / explanatory columns (continuous or
        discrete). CTSC estimates a marginal effect for each.
    unitid, time : str
        Unit-id and time column names.
    population_col : str, optional
        Time-invariant per-unit weight column for the average effect
        (e.g. population). Defaults to uniform weights.
    """
    treatment_vars = list(treatment_vars)
    if len(treatment_vars) == 0:
        raise MlsynthConfigError("CTSC requires at least one treatment variable.")
    for col in (outcome, unitid, time, *treatment_vars):
        if col not in df.columns:
            raise MlsynthDataError(f"Required column {col!r} missing.")
    if df[outcome].isna().any():
        raise MlsynthDataError("Outcome column contains NaN values.")

    time_labels = np.array(sorted(df[time].unique()))
    T = int(time_labels.size)
    unit_names = sorted(df[unitid].unique())
    n = len(unit_names)
    if n < 3:
        raise MlsynthDataError(
            "CTSC needs at least 3 units (a synthetic control is built for "
            "every unit from the others)."
        )

    y_wide = df.pivot(index=unitid, columns=time, values=outcome)
    if y_wide.isna().any().any():
        raise MlsynthDataError("Panel is unbalanced (missing unit-time cells).")
    Y = y_wide.loc[unit_names, time_labels].to_numpy(dtype=float)

    K = len(treatment_vars)
    D = np.empty((n, T, K), dtype=float)
    for k, col in enumerate(treatment_vars):
        w = df.pivot(index=unitid, columns=time, values=col)
        if w.isna().any().any():
            raise MlsynthDataError(f"Treatment variable {col!r} has missing cells.")
        D[:, :, k] = w.loc[unit_names, time_labels].to_numpy(dtype=float)

    if population_col is not None:
        if population_col not in df.columns:
            raise MlsynthDataError(f"Population column {population_col!r} missing.")
        pop = df.groupby(unitid)[population_col].first().loc[unit_names].to_numpy(float)
        if (pop < 0).any() or pop.sum() <= 0:
            raise MlsynthDataError("Population weights must be non-negative and sum > 0.")
        population_weights = pop / pop.sum()
    else:
        population_weights = np.full(n, 1.0 / n)

    return CTSCInputs(
        Y=Y,
        D=D,
        unit_names=list(unit_names),
        time_labels=time_labels,
        treatment_names=treatment_vars,
        population_weights=population_weights,
    )
