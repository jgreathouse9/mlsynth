"""Panel ingestion for the SSC estimator (staggered adoption)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import SSCInputs


def prepare_ssc_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> SSCInputs:
    """Pivot a long panel into :class:`SSCInputs`.

    SSC (Cao, Lu & Wu 2026) targets **staggered** adoption with a long
    pre-period: ``T0`` is the number of clean periods before *any* unit is
    treated, and all units -- including not-yet-treated ones -- serve as donors.
    The treatment indicator must be absorbing (once 1, stays 1).
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(f"Required column {col!r} missing.")
    if df[outcome].isna().any():
        raise MlsynthDataError("Outcome column contains NaN values.")

    time_labels = np.array(sorted(df[time].unique()))
    T1 = int(time_labels.size)
    unit_names = sorted(df[unitid].unique())
    N = len(unit_names)
    if N < 3:
        raise MlsynthDataError("SSC needs at least 3 units (each unit is "
                               "predicted from the others).")

    y_wide = df.pivot(index=unitid, columns=time, values=outcome)
    d_wide = df.pivot(index=unitid, columns=time, values=treat)
    if y_wide.isna().any().any() or d_wide.isna().any().any():
        raise MlsynthDataError("Panel is unbalanced (missing unit-time cells).")
    Y = y_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    D = (d_wide.loc[unit_names, time_labels].to_numpy() != 0).astype(int)

    any_treated_at_t = D.max(axis=0)
    if any_treated_at_t[0] == 1:
        raise MlsynthDataError("Some unit is treated at the earliest period; "
                               "SSC needs clean pre-treatment periods.")
    if not any_treated_at_t.any():
        raise MlsynthDataError("No treated unit-periods found (all treat == 0).")
    T0 = int(np.argmax(any_treated_at_t == 1))   # first period with any treatment
    S = T1 - T0
    if T0 < 2:
        raise MlsynthDataError("SSC needs at least 2 clean pre-treatment periods.")
    if S < 1:
        raise MlsynthDataError("SSC needs at least 1 post-treatment period.")
    # Note: end-of-sample inference additionally needs T0 > S (so at least one
    # pre-treatment placebo window exists). When T0 <= S the point estimates are
    # still computed but the bands are NaN (matching the reference).

    # Absorbing-treatment check + per-unit adoption time.
    adoption = np.full(N, -1, dtype=int)
    for i in range(N):
        treated_t = np.where(D[i] == 1)[0]
        if treated_t.size:
            first = int(treated_t[0])
            adoption[i] = first
            if not D[i, first:].all():
                raise MlsynthDataError(
                    f"Unit {unit_names[i]!r} has non-absorbing treatment "
                    "(turns off after turning on); SSC assumes once treated, "
                    "always treated."
                )
    treated_idx = np.where(adoption >= 0)[0]
    if treated_idx.size == 0:
        raise MlsynthDataError("No ever-treated units found.")

    return SSCInputs(
        Y=Y, D=D, T0=T0, unit_names=list(unit_names), time_labels=time_labels,
        treated_idx=treated_idx, adoption=adoption,
    )
