"""Panel ingestion for the ATEL estimator.

Outcomes come through :func:`mlsynth.utils.datautils.dataprep`, which supplies
the treated series, the donor matrix, the sorted period labels and the count of
pre-treatment periods. The covariates are pivoted separately and aligned to that
same unit and period order, so the cube and the outcome panel cannot disagree
about which row is which unit.

``dataprep`` aggregates its own ``covariates`` argument to pre-period means,
which is the wrong shape here: the sieve needs each covariate at every
unit-period, since that is what makes the loading vary with time.

Two requirements go beyond what ``dataprep`` enforces, and each is checked with
its own message: the covariates are fully observed, because a missing cell has
no basis value; and the pre-period is at least twice the factor count, because
the local linear design carries a level and a slope per factor.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from .structures import ATELInputs

__all__ = ["prepare_atel_inputs"]


def _wide(df: pd.DataFrame, column: str, unit_col: str, time_col: str) -> pd.DataFrame:
    """Pivot ``column`` to ``unit x time``, rejecting any unobserved cell."""
    wide = df.pivot(index=unit_col, columns=time_col, values=column)
    if wide.isna().to_numpy().any():
        raise MlsynthDataError(
            f"Covariate {column!r} is missing for some unit-period cells; ATEL "
            "needs every covariate observed to evaluate the sieve basis."
        )
    return wide


def _aligned(wide: pd.DataFrame, units: Sequence, times: Sequence) -> np.ndarray:
    """Put a pivoted frame in the outcome panel's row and column order."""
    return wide.reindex(index=list(units), columns=list(times)).to_numpy(dtype=float)


def prepare_atel_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    covariates: List[str],
    n_factors: int,
) -> ATELInputs:
    """Assemble the outcome panel and covariate cube ATEL runs on.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel, one row per ``(unit, period)``.
    outcome, treat, unitid, time : str
        Column names.
    covariates : list of str
        Time-varying covariate columns, at least one.
    n_factors : int
        Number of factors, used only to check the pre-period is long enough.

    Returns
    -------
    ATELInputs

    Raises
    ------
    MlsynthDataError
        If a named column is absent, the panel is unbalanced, a covariate has a
        missing cell, no covariate is given, or the pre-period is shorter than
        twice the factor count.
    """
    covariates = list(covariates)
    if not covariates:
        raise MlsynthDataError(
            "ATEL builds its diversified weights from covariates, so at least "
            "one covariate column is required."
        )
    missing = [c for c in [outcome, treat, unitid, time, *covariates]
               if c not in df.columns]
    if missing:
        raise MlsynthDataError(f"Required column(s) {missing} not found in the panel.")

    balance(df, unitid, time)
    prep = dataprep(df, unitid, time, outcome, treat)

    treated_name = prep["treated_unit_name"]
    donor_names = list(prep["donor_names"])
    units = [treated_name, *donor_names]
    times = list(prep["time_labels"])

    outcomes = np.vstack(
        [np.asarray(prep["y"], dtype=float).ravel(),
         np.asarray(prep["donor_matrix"], dtype=float).T]
    )
    n_pre = int(prep["pre_periods"])

    if n_pre < 2 * int(n_factors):
        raise MlsynthDataError(
            f"ATEL's local linear design carries a level and a slope for each "
            f"of {n_factors} factors, so it needs at least {2 * int(n_factors)} "
            f"pre-treatment periods; this panel has {n_pre}."
        )

    cube = np.stack(
        [_aligned(_wide(df, c, unitid, time), units, times) for c in covariates],
        axis=2,
    )
    if not np.isfinite(cube).all():
        raise MlsynthDataError("The covariate cube contains non-finite values.")

    return ATELInputs(
        outcomes=outcomes,
        covariates=cube,
        n_pre=n_pre,
        unit_labels=np.asarray(units, dtype=object),
        time_labels=np.asarray(times),
        covariate_names=tuple(covariates),
    )
