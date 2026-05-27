"""Data preparation for HSC.

Wraps ``datautils.dataprep`` to pivot the long panel into the treated
outcome vector and donor matrix HSC consumes, and validates that the
pre-treatment window is long enough for the difference operator and the
rolling-origin cross-validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import HSCInputs


def prepare_hsc_inputs(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    treat: str,
    q: int,
    n_splits: int = 3,
) -> HSCInputs:
    """Pivot panel data into the ``(T,)`` / ``(T, N)`` layout HSC consumes.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel data.
    outcome, unitid, time, treat : str
        Column names for outcome, units, time, and the binary treatment
        indicator (1 for the treated unit in post-treatment periods).
    q : int
        Smoothness order (1 or 2); determines the minimum pre-period length.
    n_splits : int
        Rolling-origin CV folds; used only to validate the pre-period is long
        enough to cross-validate.

    Returns
    -------
    HSCInputs
    """

    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "HSC supports a single treated unit; the panel appears to contain "
            "multiple treatment cohorts."
        )

    y_target = np.asarray(prepared["y"], dtype=float).reshape(-1)
    donor_matrix = np.asarray(prepared["donor_matrix"], dtype=float)
    if donor_matrix.ndim != 2:
        raise MlsynthDataError(
            f"HSC requires a 2D donor matrix; got shape {donor_matrix.shape}."
        )

    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    n_donors = donor_matrix.shape[1]

    if n_donors < 2:
        raise MlsynthDataError("HSC requires at least two donor units.")
    if T0 <= q + 2:
        raise MlsynthDataError(
            f"HSC needs more than q + 2 = {q + 2} pre-treatment periods to form "
            f"the difference operator and cross-validate; got T0={T0}."
        )
    if T0 < n_splits + 2:
        raise MlsynthDataError(
            f"HSC needs at least n_splits + 2 = {n_splits + 2} pre-treatment "
            f"periods for rolling-origin CV; got T0={T0}."
        )

    return HSCInputs(
        y_target=y_target,
        donor_matrix=donor_matrix,
        T=T,
        T0=T0,
        treated_unit_name=prepared["treated_unit_name"],
        donor_names=list(prepared["donor_names"]),
        time_labels=np.asarray(prepared["time_labels"]),
        q=q,
    )
