"""Data preparation for BLTVP.

Wraps :func:`mlsynth.utils.datautils.dataprep`. The donor blocks pass through
on their original scale: the coefficients carry the level, and the optional
intercept is added inside the sampler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import BLTVPInputs


def prepare_bltvp_inputs(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    treat: str,
) -> BLTVPInputs:
    """Pivot panel data into the arrays the BLTVP sampler consumes.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel data.
    outcome, unitid, time, treat : str
        Column names identifying the outcome, units, time periods, and the
        binary treatment indicator.

    Returns
    -------
    BLTVPInputs
    """

    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "BLTVP supports a single treated unit; the input panel appears "
            "to contain multiple treatment cohorts."
        )

    y_target = np.asarray(prepared["y"], dtype=float).ravel()
    donor_matrix = np.asarray(prepared["donor_matrix"], dtype=float)
    if donor_matrix.ndim != 2:  # pragma: no cover - dataprep guarantees 2D
        raise MlsynthDataError(
            f"BLTVP requires a 2D donor matrix; got shape {donor_matrix.shape}."
        )

    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    n_donors = donor_matrix.shape[1]

    if n_donors < 1:  # pragma: no cover - dataprep guarantees >= 1 donor
        raise MlsynthDataError("BLTVP requires at least one donor unit.")
    if T0 < 2:
        raise MlsynthDataError(
            f"BLTVP requires at least two pre-treatment periods; got {T0}. "
            "A random-walk coefficient needs at least one transition to be "
            "informed by the data."
        )

    return BLTVPInputs(
        y_pre=y_target[:T0],
        X_pre=donor_matrix[:T0, :],
        X_all=donor_matrix,
        y_target=y_target,
        T0=T0,
        T=T,
        N=n_donors,
        treated_unit_name=prepared["treated_unit_name"],
        donor_names=list(prepared["donor_names"]),
        time_labels=np.asarray(prepared["time_labels"]),
    )
