"""Panel ingestion for the MSQRT estimator (block, multiple treated units)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import MSQRTInputs


def prepare_msqrt_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    weight_col: Optional[str] = None,
) -> MSQRTInputs:
    """Pivot a long panel into the stacked ``Y = X Theta + E`` block matrices.

    MSQRT (Shen, Song & Abadie 2025) assumes a **block** design: every
    ever-treated unit adopts at the same period. Treated units form the columns
    of ``Y``; never-treated units form the columns of ``X``.
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(f"Required column {col!r} missing.")
    if df[outcome].isna().any():
        raise MlsynthDataError("Outcome column contains NaN values.")

    time_labels = np.array(sorted(df[time].unique()))
    unit_names = sorted(df[unitid].unique())

    y_wide = df.pivot(index=unitid, columns=time, values=outcome)
    d_wide = df.pivot(index=unitid, columns=time, values=treat)
    if y_wide.isna().any().any() or d_wide.isna().any().any():
        raise MlsynthDataError("Panel is unbalanced (missing unit-time cells).")
    Y = y_wide.loc[unit_names, time_labels].to_numpy(dtype=float)   # (N, T)
    D = d_wide.loc[unit_names, time_labels].to_numpy(dtype=float)

    any_treated_at_t = D.max(axis=0)
    if any_treated_at_t[0] == 1:
        raise MlsynthDataError("Some unit is treated at the earliest period.")
    if not any_treated_at_t.any():
        raise MlsynthDataError("No treated unit-periods found (all treat == 0).")
    T0 = int(np.argmax(any_treated_at_t == 1))
    T = time_labels.size
    if T0 < 2:
        raise MlsynthDataError("MSQRT needs at least 2 pre-treatment periods.")
    if T - T0 < 1:
        raise MlsynthDataError("MSQRT needs at least 1 post-treatment period.")

    treated_idx = np.where(D.max(axis=1) == 1)[0]
    control_idx = np.where(D.max(axis=1) == 0)[0]
    if treated_idx.size == 0:
        raise MlsynthDataError("No ever-treated units found.")
    if control_idx.size < 2:
        raise MlsynthDataError("MSQRT needs at least 2 never-treated donor units.")

    # Block-design check: all treated units share one adoption period.
    first_treat = np.array([int(np.argmax(D[i] == 1)) for i in treated_idx])
    if not np.all(first_treat == T0):
        raise MlsynthDataError(
            "MSQRT assumes a block design (all treated units adopt at the same "
            "period). Staggered adoption detected -- use SDID, SequentialSDID, "
            "PPSCM, or MCNNM instead."
        )

    treated_names = [unit_names[i] for i in treated_idx]
    control_names = [unit_names[i] for i in control_idx]

    # Optional per-treated-unit size weights for the weighted aggregate ATT.
    # The column is taken to be unit-constant (first value per unit).
    unit_weights = None
    if weight_col is not None:
        if weight_col not in df.columns:
            raise MlsynthDataError(f"weight_col {weight_col!r} missing from df.")
        wmap = (df.drop_duplicates(subset=[unitid])
                  .set_index(unitid)[weight_col])
        unit_weights = {}
        for u in treated_names:
            val = wmap.get(u)
            if val is None or pd.isna(val):
                raise MlsynthDataError(
                    f"weight_col {weight_col!r} has no value for treated unit {u!r}.")
            fv = float(val)
            if fv < 0:
                raise MlsynthDataError(f"weight_col {weight_col!r} is negative for {u!r}.")
            unit_weights[str(u)] = fv
        if sum(unit_weights.values()) <= 0:
            raise MlsynthDataError(
                f"weight_col {weight_col!r} sums to zero across treated units.")

    Y_pre = Y[np.ix_(treated_idx, np.arange(T0))].T        # (T0, m)
    Y_post = Y[np.ix_(treated_idx, np.arange(T0, T))].T    # (T_post, m)
    X_pre = Y[np.ix_(control_idx, np.arange(T0))].T        # (T0, n)
    X_post = Y[np.ix_(control_idx, np.arange(T0, T))].T    # (T_post, n)

    return MSQRTInputs(
        Y_pre=Y_pre, Y_post=Y_post, X_pre=X_pre, X_post=X_post,
        treated_names=treated_names, control_names=control_names,
        time_labels=time_labels, T0=int(T0), unit_weights=unit_weights,
    )
