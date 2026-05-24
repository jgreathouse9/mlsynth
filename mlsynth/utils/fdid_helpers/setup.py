"""Data preparation for the Forward Difference-in-Differences estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError, MlsynthEstimationError
from ..datautils import balance, dataprep
from .structures import FDIDInputs


def prepare_fdid_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    verbose: bool = True,
) -> FDIDInputs:
    """Balance the panel, pivot it, and package it into :class:`FDIDInputs`.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel with outcome, treatment, unit, and time columns.
    outcome, treat, unitid, time : str
        Column names identifying the outcome, treatment indicator, unit,
        and time period.
    verbose : bool, default True
        Whether the forward-selection path should be recorded step by step.

    Returns
    -------
    FDIDInputs
        Preprocessed panel ready for forward selection.

    Raises
    ------
    MlsynthDataError
        If panel balancing or data preparation fails (e.g. no donor units).
    MlsynthEstimationError
        If fewer than two pre-treatment periods are available.
    """
    try:
        balance(df, unitid, time)
    except Exception as e:  # noqa: BLE001 - re-wrap as repository error
        raise MlsynthDataError(f"Error balancing panel data: {str(e)}") from e

    try:
        prepped = dataprep(df, unitid, time, outcome, treat)
    except MlsynthDataError:
        raise
    except Exception as e:  # noqa: BLE001
        raise MlsynthDataError(f"Error preparing data matrices: {str(e)}") from e

    pre_periods = prepped.get("pre_periods")
    if pre_periods is None or pre_periods < 2:
        raise MlsynthEstimationError("Insufficient pre-periods for estimation.")

    return FDIDInputs(
        y=np.asarray(prepped["y"], dtype=float),
        donor_matrix=np.asarray(prepped["donor_matrix"], dtype=float),
        pre_periods=int(pre_periods),
        post_periods=int(prepped["post_periods"]),
        T=int(prepped["total_periods"]),
        donor_names=list(prepped["donor_names"]),
        time_labels=np.asarray(prepped["time_labels"]),
        treated_unit_name=prepped["treated_unit_name"],
        verbose=verbose,
        prepped=prepped,
    )
