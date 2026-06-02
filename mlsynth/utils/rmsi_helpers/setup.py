"""Panel + side-information ingestion for the RMSI estimator."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import RMSIInputs


def prepare_rmsi_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    unit_covariates: Optional[List[str]] = None,
    time_covariates: Optional[List[str]] = None,
) -> RMSIInputs:
    """Pivot a long panel into block matrices and extract side information.

    RMSI (Agarwal, Choi & Yuan 2026) assumes a **block** design: every
    ever-treated unit adopts at the same period ``T0``. ``unit_covariates`` are
    columns (approximately) constant within a unit -> the row feature matrix
    ``X`` (one row per unit, time-averaged); ``time_covariates`` are columns
    constant within a period -> the column feature matrix ``Z`` (one row per
    period, unit-averaged). Either may be empty, in which case the corresponding
    projection captures only the mean.
    """
    unit_covariates = list(unit_covariates or [])
    time_covariates = list(time_covariates or [])
    required = {outcome, treat, unitid, time, *unit_covariates, *time_covariates}
    missing = required - set(df.columns)
    if missing:
        raise MlsynthDataError(f"Missing columns: {', '.join(sorted(missing))}.")
    if df[outcome].isna().any():
        raise MlsynthDataError("Outcome column contains NaN values.")

    time_labels = np.array(sorted(df[time].unique()))
    T = int(time_labels.size)
    unit_names = sorted(df[unitid].unique())
    N = len(unit_names)
    if N < 3:
        raise MlsynthDataError("RMSI needs at least 3 units.")

    y_wide = df.pivot(index=unitid, columns=time, values=outcome)
    d_wide = df.pivot(index=unitid, columns=time, values=treat)
    if y_wide.isna().any().any() or d_wide.isna().any().any():
        raise MlsynthDataError("Panel is unbalanced (missing unit-time cells).")
    Y = y_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    D = (d_wide.loc[unit_names, time_labels].to_numpy() != 0).astype(int)

    any_treated_at_t = D.max(axis=0)
    if any_treated_at_t[0] == 1:
        raise MlsynthDataError("Some unit is treated at the earliest period.")
    if not any_treated_at_t.any():
        raise MlsynthDataError("No treated unit-periods found (all treat == 0).")
    T0 = int(np.argmax(any_treated_at_t == 1))
    if T0 < 2:
        raise MlsynthDataError("RMSI needs at least 2 pre-treatment periods.")
    if T - T0 < 1:
        raise MlsynthDataError("RMSI needs at least 1 post-treatment period.")

    treated_idx = np.where(D.max(axis=1) == 1)[0]
    control_idx = np.where(D.max(axis=1) == 0)[0]
    if treated_idx.size == 0:
        raise MlsynthDataError("No ever-treated units found.")
    if control_idx.size < 2:
        raise MlsynthDataError("RMSI needs at least 2 never-treated control units.")

    # Block-design check: all treated units share one adoption period.
    first_treat = np.array([int(np.argmax(D[i] == 1)) for i in treated_idx])
    if not np.all(first_treat == T0):
        raise MlsynthDataError(
            "RMSI assumes a block design (all treated units adopt at the same "
            "period). Staggered adoption detected -- use SDID, SequentialSDID, "
            "PPSCM, SSC, or MCNNM instead."
        )

    # Row features X (one row per unit): unit-level mean of each unit covariate.
    if unit_covariates:
        X = np.column_stack([
            df.groupby(unitid)[c].mean().loc[unit_names].to_numpy(dtype=float)
            for c in unit_covariates])
    else:
        X = np.zeros((N, 0))
    # Column features Z (one row per period): period-level mean of each time cov.
    if time_covariates:
        Z = np.column_stack([
            df.groupby(time)[c].mean().loc[time_labels].to_numpy(dtype=float)
            for c in time_covariates])
    else:
        Z = np.zeros((T, 0))

    return RMSIInputs(
        Y=Y, D=D, X=X, Z=Z, T0=T0, treated_idx=treated_idx,
        control_idx=control_idx, unit_names=list(unit_names),
        time_labels=time_labels, unit_covariates=unit_covariates,
        time_covariates=time_covariates,
    )
