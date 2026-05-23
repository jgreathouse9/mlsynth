"""Panel ingestion for the MC-NNM estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import MCNNMInputs


def prepare_mcnnm_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> MCNNMInputs:
    """Pivot a long panel into :class:`MCNNMInputs`.

    Observed entries (``mask == 1``) are the control units and the treated
    units' pre-treatment periods; the treated post-treatment cells are the
    missing entries MC-NNM imputes.
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
        raise MlsynthDataError("MC-NNM needs at least 3 units.")

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
        raise MlsynthDataError("MC-NNM requires at least one pre-period.")
    if T - T0 < 1:
        raise MlsynthDataError("MC-NNM requires at least one post-period.")

    treated_idx = np.where(D.max(axis=1) == 1)[0]
    if treated_idx.size == 0:
        raise MlsynthDataError("No ever-treated units found.")
    if N - treated_idx.size < 2:
        raise MlsynthDataError("MC-NNM needs at least 2 control units.")

    mask = (D == 0).astype(float)   # observed where untreated

    return MCNNMInputs(
        Y=Y, mask=mask, D=D, treated_idx=treated_idx, T0=T0,
        unit_names=list(unit_names), time_labels=time_labels,
    )
