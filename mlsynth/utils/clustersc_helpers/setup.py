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

    The panel is read with ``dataprep(..., missing_outcomes="keep")``: a NaN
    outcome in a donor's row stays NaN, for the PCR path's forecast algorithm
    (Agarwal, Amjad, Shah and Shen 2018, Section 3.3; see
    :mod:`.pcr.missing`). The estimator refuses it on every other branch.
    The treated unit's cells are refused here: the forecast regresses on the
    treated unit's complete series (the target row :math:`X_L` of step 2,
    taken fully observed in Appendix C), and the paper defines no missing
    target cell.

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
        On unbalanced panel, a missing treated cell, an infinite value,
        multiple treated cohorts, or zero pre/post-treatment periods.
    """

    prepared = dataprep(df, unitid, time, outcome, treat, missing_outcomes="keep")
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

    gaps = np.flatnonzero(np.isnan(y))
    if gaps.size:
        periods = ", ".join(str(time_labels[i]) for i in gaps)
        raise MlsynthDataError(
            f"CLUSTERSC: the treated unit '{treated_name}' has {gaps.size} "
            f"missing outcome cell(s), at period(s) {periods}. The forecast "
            "algorithm of Agarwal, Amjad, Shah and Shen (2018) regresses on the "
            "treated unit's complete series and defines no missing treated "
            "cell; only donor cells may be missing."
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
