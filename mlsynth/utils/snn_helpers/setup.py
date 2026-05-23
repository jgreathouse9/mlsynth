"""Panel ingestion for the SNN estimator.

Pivots a long panel into ``(N, T)`` outcome and treatment matrices. SNN
treats the treated post-treatment cells as missing and imputes them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import SNNInputs


def prepare_snn_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> SNNInputs:
    """Pivot a long panel into :class:`SNNInputs`.

    A treated unit has ``treat == 1`` from a common adoption period
    onward; SNN imputes those cells' untreated potential outcomes.
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(f"Required column {col!r} missing.")
    if df[outcome].isna().any():
        raise MlsynthDataError("Outcome column contains NaN values.")

    time_labels = np.array(sorted(df[time].unique()))
    T = int(time_labels.size)
    unit_names = sorted(df[unitid].unique())
    N = len(unit_names)
    if N < 3:
        raise MlsynthDataError("SNN needs at least 3 units.")

    y_wide = df.pivot(index=unitid, columns=time, values=outcome)
    d_wide = df.pivot(index=unitid, columns=time, values=treat)
    if y_wide.isna().any().any() or d_wide.isna().any().any():
        raise MlsynthDataError("Panel is unbalanced (missing unit-time cells).")
    Y = y_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    D = d_wide.loc[unit_names, time_labels].to_numpy(dtype=float)

    any_treated_at_t = D.max(axis=0)
    if any_treated_at_t[0] == 1:
        raise MlsynthDataError("Some unit is treated at the earliest period.")
    if not any_treated_at_t.any():
        raise MlsynthDataError("No treated unit-periods found (all treat == 0).")
    T0 = int(np.argmax(any_treated_at_t == 1))
    if T0 < 1:
        raise MlsynthDataError("SNN requires at least one pre-period.")
    if T - T0 < 1:
        raise MlsynthDataError("SNN requires at least one post-period.")

    treated_idx = np.where(D.max(axis=1) == 1)[0]
    if treated_idx.size == 0:
        raise MlsynthDataError("No ever-treated units found.")
    n_control = N - treated_idx.size
    if n_control < 2:
        raise MlsynthDataError("SNN needs at least 2 control (donor) units.")

    return SNNInputs(
        Y=Y, D=D, treated_idx=treated_idx, T0=T0,
        unit_names=list(unit_names), time_labels=time_labels,
    )
