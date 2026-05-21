"""Data preparation for SBC.

Wraps ``datautils.dataprep`` and validates that the pre-treatment window
is long enough to accommodate the Hamilton-filter lag structure
``h + p`` periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import SBCInputs


def prepare_sbc_inputs(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    treat: str,
    h: int,
    p: int,
) -> SBCInputs:
    """Pivot panel data into the (T, N) wide layout SBC consumes.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel data.
    outcome, unitid, time, treat : str
        Column names identifying the outcome, units, time periods, and the
        binary treatment indicator.
    h, p : int
        Hamilton filter horizon and lag count. Validated to fit within the
        pre-treatment window.

    Returns
    -------
    SBCInputs
    """

    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "SBC currently supports a single treated unit; the input panel "
            "appears to contain multiple treatment cohorts."
        )

    y_target = np.asarray(prepared["y"], dtype=float)
    donor_matrix_tn = np.asarray(prepared["donor_matrix"], dtype=float)
    if donor_matrix_tn.ndim != 2:
        raise MlsynthDataError(
            "SBC requires a 2D donor matrix; got shape "
            f"{donor_matrix_tn.shape}."
        )

    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    n_donors = donor_matrix_tn.shape[1]

    if n_donors < 1:
        raise MlsynthDataError("SBC requires at least one donor unit.")

    min_pre = h + p
    if T0 < min_pre:
        raise MlsynthDataError(
            f"SBC needs at least h + p = {min_pre} pre-treatment periods to "
            f"fit the Hamilton filter (h={h}, p={p}); got T0={T0}."
        )

    Y_full = np.column_stack(
        [y_target.reshape(T, 1), donor_matrix_tn]
    )

    return SBCInputs(
        Y_full=Y_full,
        T=T,
        T0=T0,
        N=Y_full.shape[1],
        treated_unit_name=prepared["treated_unit_name"],
        donor_names=list(prepared["donor_names"]),
        time_labels=np.asarray(prepared["time_labels"]),
        Ywide=prepared["Ywide"],
        y_target=y_target,
    )
