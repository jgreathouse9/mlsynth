"""Data preparation for the FMA estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import dataprep
from .structures import FMAInputs


def prepare_fma_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    preprocessing: str = "demean",
    stationarity: str = "nonstationary",
) -> FMAInputs:
    """Pivot the panel and assemble FMA inputs.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel with one row per ``(unit, time)``.
    outcome, treat, unitid, time : str
        Column names.
    preprocessing : {"demean", "standardize"}
        Preprocessing applied to the control panel before factor
        extraction. Demeaning (default) follows Bai (2003); standardizing
        is preferred when the control series have very heterogeneous
        variances.
    stationarity : {"stationary", "nonstationary"}
        Selects the factor-selection criterion downstream: ``"stationary"``
        uses the modified Bai-Ng (MBN) criterion (Li & Sonnier 2023, Web
        Appendix D), ``"nonstationary"`` uses Bai (2004) IPC1.

    Returns
    -------
    FMAInputs
        Preprocessed panel.

    Raises
    ------
    MlsynthDataError, MlsynthConfigError
    """

    if preprocessing not in {"demean", "standardize"}:
        raise MlsynthConfigError(
            f"preprocessing must be 'demean' or 'standardize'; got {preprocessing!r}."
        )
    if stationarity not in {"stationary", "nonstationary"}:
        raise MlsynthConfigError(
            f"stationarity must be 'stationary' or 'nonstationary'; got {stationarity!r}."
        )

    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "FMA supports a single treated unit; the panel contains "
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
            "FMA does not currently support missing data; please impute "
            "or drop missing values before calling fit()."
        )
    if T0 < 2:
        raise MlsynthDataError(
            f"FMA requires at least 2 pre-treatment periods; got {T0}."
        )
    if Y0.shape[1] < 1:
        raise MlsynthDataError(
            "FMA requires at least one control unit."
        )

    return FMAInputs(
        treated_outcome=y,
        control_outcomes=Y0,
        donor_names=donor_names,
        treated_unit_name=treated_name,
        T=T,
        T0=T0,
        time_labels=time_labels,
        preprocessing=preprocessing,
        stationarity=stationarity,
    )
