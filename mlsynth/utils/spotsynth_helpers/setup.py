"""Panel ingestion for the SPOTSYNTH estimator (single treated unit)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import SpotSynthInputs


def prepare_spotsynth_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> SpotSynthInputs:
    """Pivot a long panel into a treated series and a balanced donor matrix.

    SPOTSYNTH targets the classic single-treated-unit synthetic-control design:
    one unit adopts the intervention at a common time ``T0`` and every other unit
    is a candidate donor to be screened for spillover contamination.
    """
    required = {outcome, treat, unitid, time}
    missing = required - set(df.columns)
    if missing:
        raise MlsynthDataError(f"Missing columns: {', '.join(sorted(missing))}.")
    if df[outcome].isna().any():
        raise MlsynthDataError("Outcome column contains NaN values.")

    time_labels = np.array(sorted(df[time].unique()))
    unit_names = sorted(df[unitid].unique())
    if len(unit_names) < 3:
        raise MlsynthDataError("SPOTSYNTH needs a treated unit and at least 2 donors.")

    y_wide = df.pivot(index=unitid, columns=time, values=outcome)
    d_wide = df.pivot(index=unitid, columns=time, values=treat)
    if y_wide.isna().any().any() or d_wide.isna().any().any():
        raise MlsynthDataError("Panel is unbalanced (missing unit-time cells).")
    Y = y_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    Dmat = (d_wide.loc[unit_names, time_labels].to_numpy() != 0).astype(int)

    treated_mask = Dmat.max(axis=1) == 1
    treated_rows = np.where(treated_mask)[0]
    if treated_rows.size == 0:
        raise MlsynthDataError("No treated unit found (all treat == 0).")
    if treated_rows.size > 1:
        raise MlsynthDataError(
            "SPOTSYNTH expects a single treated unit. Found "
            f"{treated_rows.size}; restrict the panel or aggregate first."
        )
    ti = int(treated_rows[0])

    any_treated_at_t = Dmat.max(axis=0)
    if any_treated_at_t[0] == 1:
        raise MlsynthDataError("Treated unit is treated at the earliest period.")
    T0 = int(np.argmax(any_treated_at_t == 1))
    if T0 < 3:
        raise MlsynthDataError("SPOTSYNTH needs at least 3 pre-intervention periods.")
    if Y.shape[1] - T0 < 1:
        raise MlsynthDataError("SPOTSYNTH needs at least 1 post-intervention period.")

    donor_rows = [i for i in range(len(unit_names)) if i != ti]
    if len(donor_rows) < 2:
        raise MlsynthDataError("SPOTSYNTH needs at least 2 donors.")

    y = Y[ti]
    D = Y[donor_rows].T  # (T, n_donors)
    donor_names = [unit_names[i] for i in donor_rows]
    return SpotSynthInputs(
        y=y, D=D, T0=T0, donor_names=donor_names,
        treated_name=unit_names[ti], time_labels=time_labels,
    )
