"""Data preparation for BSCM.

Wraps :func:`mlsynth.utils.datautils.dataprep`. BSCM fits with an explicit
intercept, so -- unlike BVSS -- the donor blocks are passed through on their
original scale (no demeaning).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import BSCMInputs


def prepare_bscm_inputs(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    treat: str,
) -> BSCMInputs:
    """Pivot panel data into the arrays the BSCM sampler consumes.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel data.
    outcome, unitid, time, treat : str
        Column names identifying the outcome, units, time periods, and the
        binary treatment indicator.

    Returns
    -------
    BSCMInputs
    """

    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "BSCM currently supports a single treated unit; the input "
            "panel appears to contain multiple treatment cohorts."
        )

    y_target = np.asarray(prepared["y"], dtype=float)
    donor_matrix_tn = np.asarray(prepared["donor_matrix"], dtype=float)
    if donor_matrix_tn.ndim != 2:  # pragma: no cover - dataprep guarantees 2D
        raise MlsynthDataError(
            "BSCM requires a 2D donor matrix; got shape "
            f"{donor_matrix_tn.shape}."
        )

    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    n_donors = donor_matrix_tn.shape[1]

    if n_donors < 1:  # pragma: no cover - dataprep guarantees >= 1 donor
        raise MlsynthDataError("BSCM requires at least one donor unit.")
    if T0 < 2:
        raise MlsynthDataError(
            "BSCM requires at least two pre-treatment periods."
        )

    return BSCMInputs(
        y_pre=y_target[:T0],
        X_pre=donor_matrix_tn[:T0, :],
        X_all=donor_matrix_tn,
        y_target=y_target,
        T0=T0,
        T=T,
        N=n_donors,
        treated_unit_name=prepared["treated_unit_name"],
        donor_names=list(prepared["donor_names"]),
        time_labels=np.asarray(prepared["time_labels"]),
    )
