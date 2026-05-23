"""Panel ingestion for the ISCM estimator.

Pivots a long-format panel into the dense ``(N, T)`` outcome and
treatment matrices ISCM operates on. ISCM builds a synthetic control for
*every* unit, so -- unlike single-treated SCM -- there is no treated /
donor split at this stage; all units are retained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import ISCMInputs


def prepare_iscm_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> ISCMInputs:
    """Pivot a long panel into :class:`ISCMInputs`.

    Parameters
    ----------
    df : pd.DataFrame
        Balanced long panel; one row per ``(unit, time)``.
    outcome, treat, unitid, time : str
        Column names. A treated unit has ``treat == 1`` from a common
        adoption period onward; ISCM assumes a single adoption date (no
        staggered timing).
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
        raise MlsynthDataError(
            "ISCM needs at least 3 units (a synthetic control is built for "
            "every unit from the others)."
        )

    y_wide = df.pivot(index=unitid, columns=time, values=outcome)
    d_wide = df.pivot(index=unitid, columns=time, values=treat)
    if y_wide.isna().any().any() or d_wide.isna().any().any():
        raise MlsynthDataError("Panel is unbalanced (missing unit-time cells).")

    Y = y_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    D = d_wide.loc[unit_names, time_labels].to_numpy(dtype=float)

    # Adoption period: first time any unit is treated.
    any_treated_at_t = D.max(axis=0)
    if any_treated_at_t[0] == 1:
        raise MlsynthDataError("Some unit is treated at the earliest period.")
    if not any_treated_at_t.any():
        raise MlsynthDataError("No treated unit-periods found (all treat == 0).")
    T0 = int(np.argmax(any_treated_at_t == 1))
    if T0 < 2:
        raise MlsynthDataError(
            "ISCM requires at least 2 pre-treatment periods (and assumes a "
            "large pre-period for its asymptotics)."
        )
    if T - T0 < 1:
        raise MlsynthDataError("ISCM requires at least one post-period.")

    treated_idx = np.where(D[:, T0:].max(axis=1) == 1)[0]
    if treated_idx.size == 0:
        raise MlsynthDataError("No ever-treated units found.")

    return ISCMInputs(
        Y=Y,
        D=D,
        T0=T0,
        unit_names=list(unit_names),
        time_labels=time_labels,
        treated_idx=treated_idx,
    )
