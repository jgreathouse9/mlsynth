"""Long-DataFrame -> NumPy boundary for BFSC (wraps ``dataprep``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import BFSCInputs


def prepare_bfsc_inputs(
    df: pd.DataFrame, outcome: str, unitid: str, time: str, treat: str,
) -> BFSCInputs:
    """Pivot a long panel into BFSC's ``(J, T)`` matrix (treated first)."""
    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "BFSC supports a single treated unit; the panel appears to "
            "contain multiple treatment cohorts."
        )
    y_target = np.asarray(prepared["y"], dtype=float)          # (T,)
    donors = np.asarray(prepared["donor_matrix"], dtype=float)  # (T, N)
    if donors.ndim != 2 or donors.shape[1] < 1:  # pragma: no cover - dataprep guards
        raise MlsynthDataError("BFSC requires a 2D donor matrix with >= 1 donor.")
    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    if T0 < 2:
        raise MlsynthDataError(f"BFSC needs at least 2 pre-treatment periods; got {T0}.")
    if T - T0 < 1:  # pragma: no cover - a treated period forces >= 1 post
        raise MlsynthDataError("BFSC needs at least 1 post-treatment period.")
    Y = np.vstack([y_target[None, :], donors.T])               # (J, T), treated row 0
    if not np.isfinite(Y).all():
        raise MlsynthDataError("BFSC requires a balanced panel with no missing outcomes.")
    return BFSCInputs(
        Y=Y, y_target=y_target, T0=T0, T=T, J=Y.shape[0],
        treated_unit_name=str(prepared.get("treated_unit_name", "treated")),
        donor_names=list(prepared.get("donor_names", range(donors.shape[1]))),
        time_labels=np.asarray(prepared["time_labels"]),
    )
