"""Data preparation for MVBBSC.

Wraps :func:`mlsynth.utils.datautils.dataprep`. The sampler standardizes the
treated and donor series internally by their pre-period moments, so the donor
blocks are passed through here on their original scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import MVBBSCInputs


def prepare_mvbbsc_inputs(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    treat: str,
) -> MVBBSCInputs:
    """Pivot panel data into the arrays the MVBBSC sampler consumes.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel data.
    outcome, unitid, time, treat : str
        Column names identifying the outcome, units, time periods, and the
        binary treatment indicator.

    Returns
    -------
    MVBBSCInputs
    """

    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "MVBBSC currently supports a single treated unit; the input "
            "panel appears to contain multiple treatment cohorts."
        )

    y_target = np.asarray(prepared["y"], dtype=float)
    donor_matrix_tn = np.asarray(prepared["donor_matrix"], dtype=float)
    if donor_matrix_tn.ndim != 2:  # pragma: no cover - dataprep guarantees 2D
        raise MlsynthDataError(
            "MVBBSC requires a 2D donor matrix; got shape "
            f"{donor_matrix_tn.shape}."
        )

    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    n_donors = donor_matrix_tn.shape[1]

    if n_donors < 1:  # pragma: no cover - dataprep guarantees >= 1 donor
        raise MlsynthDataError("MVBBSC requires at least one donor unit.")
    if T0 < 2:
        raise MlsynthDataError(
            "MVBBSC requires at least two pre-treatment periods."
        )

    return MVBBSCInputs(
        y_target=y_target,
        X_all=donor_matrix_tn,
        T0=T0,
        T=T,
        N=n_donors,
        treated_unit_name=prepared["treated_unit_name"],
        donor_names=list(prepared["donor_names"]),
        time_labels=np.asarray(prepared["time_labels"]),
    )
