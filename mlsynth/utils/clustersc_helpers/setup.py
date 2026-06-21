"""Data preparation helpers for CLUSTERSC."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import CLUSTERSCInputs


def prepare_clustersc_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> CLUSTERSCInputs:
    """Pivot the long panel into the CLUSTERSC layout.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel.
    outcome, treat, unitid, time : str
        Column names.

    Returns
    -------
    CLUSTERSCInputs
        Preprocessed panel.

    Raises
    ------
    MlsynthDataError
        On unbalanced panel, missing values, multiple treated cohorts,
        or zero pre/post-treatment periods.
    """

    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "CLUSTERSC supports a single treated unit; the panel "
            "contains multiple treated cohorts."
        )

    y = np.asarray(prepared["y"], dtype=float).flatten()
    Y0 = np.asarray(prepared["donor_matrix"], dtype=float)
    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    donor_names = np.asarray(prepared["donor_names"])
    treated_name = prepared["treated_unit_name"]
    time_labels = np.asarray(prepared.get("time_labels", np.arange(T)))

    if np.isnan(y).any() or np.isnan(Y0).any():
        raise MlsynthDataError(
            "CLUSTERSC does not support missing data; please impute "
            "or drop missing values."
        )
    if T0 < 2:
        raise MlsynthDataError(
            f"CLUSTERSC requires at least 2 pre-treatment periods; got {T0}."
        )
    if T - T0 < 1:
        raise MlsynthDataError(
            "CLUSTERSC requires at least 1 post-treatment period."
        )
    if Y0.shape[1] < 1:
        raise MlsynthDataError("CLUSTERSC requires at least one donor unit.")

    return CLUSTERSCInputs(
        treated_outcome=y,
        donor_outcomes=Y0,
        donor_names=donor_names,
        treated_unit_name=treated_name,
        T=T,
        T0=T0,
        time_labels=time_labels,
        outcome_name=outcome,
        time_name=time,
    )
