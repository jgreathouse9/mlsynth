"""Data preparation helpers for the NSC estimator.

Builds the :class:`NSCInputs` container by pivoting a long balanced
panel and assembling the pre-period matching matrix ``Z_0``. By
default the matching variables are the pre-period outcomes (the
paper's empirical convention); additional covariates can be stacked
via the ``covariates`` argument.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import dataprep
from .structures import NSCInputs


def prepare_nsc_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    covariates: Optional[List[str]] = None,
) -> NSCInputs:
    """Pivot the panel and assemble the NSC matching matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel with one row per ``(unit, time)``.
    outcome, treat, unitid, time : str
        Column names.
    covariates : list of str, optional
        Optional column names to use as additional matching variables
        (each collapsed to its per-unit pre-treatment mean before
        being stacked alongside the pre-period outcomes in ``Z_0``).

    Returns
    -------
    NSCInputs
        Preprocessed panel.

    Raises
    ------
    MlsynthDataError
        If the panel is unbalanced or has missing entries.
    MlsynthConfigError
        If ``covariates`` references unknown columns.
    """

    if covariates is not None:
        unknown = [c for c in covariates if c not in df.columns]
        if unknown:
            raise MlsynthConfigError(
                f"covariates references unknown columns: {unknown}"
            )

    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "NSC supports a single treated unit; the panel contains "
            "multiple treated cohorts."
        )

    y = np.asarray(prepared["y"], dtype=float).flatten()
    Y0 = np.asarray(prepared["donor_matrix"], dtype=float)
    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    donor_names = np.asarray(prepared["donor_names"])
    treated_name = prepared["treated_unit_name"]
    time_labels = np.asarray(prepared.get("time_labels", np.arange(T)))

    if Y0.shape[0] != T or y.shape[0] != T:
        raise MlsynthDataError(
            "NSC requires aligned (T,) outcome arrays."
        )

    # Matching variables: pretreatment outcomes, optionally stacked
    # with per-unit pretreatment-mean covariates.
    y_pre_treated = y[:T0]
    Y0_pre = Y0[:T0, :]

    cov_blocks_treated: list[np.ndarray] = []
    cov_blocks_donors: list[np.ndarray] = []
    if covariates:
        Ywide = prepared["Ywide"]
        units_in_pivot = list(Ywide.columns)
        # Compute pre-period means of each covariate per unit.
        pre_mask = np.arange(T) < T0
        for col in covariates:
            wide = df.pivot(index=time, columns=unitid, values=col)
            wide = wide.reindex(index=Ywide.index, columns=units_in_pivot)
            if wide.isna().any().any():
                raise MlsynthDataError(
                    f"Covariate '{col}' has missing values after pivoting."
                )
            means = wide.iloc[pre_mask].mean(axis=0).to_numpy(dtype=float)
            treated_idx = units_in_pivot.index(treated_name)
            cov_blocks_treated.append(np.asarray([means[treated_idx]]))
            donor_mask = np.asarray(
                [u != treated_name for u in units_in_pivot]
            )
            cov_blocks_donors.append(means[donor_mask])

    if cov_blocks_treated:
        Z1 = np.concatenate([y_pre_treated] + cov_blocks_treated)
        cov_stack = np.column_stack(cov_blocks_donors)
        Z0 = np.column_stack([Y0_pre.T, cov_stack])
    else:
        Z1 = y_pre_treated.copy()
        Z0 = Y0_pre.T.copy()        # shape (J, T0)

    return NSCInputs(
        treated_outcome=y,
        donor_outcomes=Y0,
        matching_matrix=Z0,
        treated_matching_vector=Z1,
        donor_names=donor_names,
        treated_unit_name=treated_name,
        T=T,
        T0=T0,
        time_labels=time_labels,
    )
