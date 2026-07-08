"""Data preparation for the CFM estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import CFMInputs


def prepare_cfm_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> CFMInputs:
    """Pivot the panel and assemble CFM inputs via :func:`dataprep`.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel with one row per ``(unit, time)``.
    outcome, treat, unitid, time : str
        Column names.

    Returns
    -------
    CFMInputs
        Preprocessed panel for a single treated unit.

    Raises
    ------
    MlsynthDataError
        If the panel has multiple treated cohorts, missing data, too few
        pre-treatment periods, or no control units.
    """

    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "CFM supports a single treated unit; the panel contains "
            "multiple treated cohorts."
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
            "CFM does not support missing data; impute or drop missing "
            "values before calling fit()."
        )
    if T0 < 2:
        raise MlsynthDataError(
            f"CFM requires at least 2 pre-treatment periods; got {T0}."
        )
    if T - T0 < 1:  # pragma: no cover - dataprep always yields >=1 post period for a treated unit
        raise MlsynthDataError(
            f"CFM requires at least 1 post-treatment period; got {T - T0}."
        )
    if Y0.shape[1] < 1:  # pragma: no cover - dataprep always yields >=1 donor for a valid panel
        raise MlsynthDataError("CFM requires at least one control unit.")

    return CFMInputs(
        treated_outcome=y,
        control_outcomes=Y0,
        donor_names=donor_names,
        treated_unit_name=treated_name,
        T=T,
        T0=T0,
        time_labels=time_labels,
    )
