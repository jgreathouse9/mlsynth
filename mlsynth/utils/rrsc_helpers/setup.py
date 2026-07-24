"""Data preparation for the RRSC estimator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import dataprep
from .structures import RRSCInputs


def prepare_rrsc_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    regime: str = "auto",
) -> RRSCInputs:
    """Pivot the panel and assemble RRSC inputs (full ``N x T`` panel with the
    treated unit in row 0), resolving the asymptotic regime.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel, one row per ``(unit, time)``.
    outcome, treat, unitid, time : str
        Column names.
    regime : {"auto", "fixed_n", "large_n"}
        Regime; ``"auto"`` picks ``"fixed_n"`` when ``T0 >= 5 N`` else
        ``"large_n"``.

    Returns
    -------
    RRSCInputs

    Raises
    ------
    MlsynthConfigError, MlsynthDataError
    """
    if regime not in {"auto", "fixed_n", "large_n"}:
        raise MlsynthConfigError(
            f"regime must be 'auto', 'fixed_n' or 'large_n'; got {regime!r}."
        )

    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "RRSC supports a single treated unit; the panel contains multiple "
            "treated cohorts."
        )

    y = np.asarray(prepared["y"], dtype=float).ravel()
    Y0 = np.asarray(prepared["donor_matrix"], dtype=float)          # T x N0
    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    donor_names = np.asarray(prepared["donor_names"])
    treated_name = prepared["treated_unit_name"]
    time_labels = np.asarray(prepared.get("time_labels", np.arange(T)))

    if np.isnan(y).any() or np.isnan(Y0).any():
        raise MlsynthDataError(
            "RRSC does not support missing data; impute or drop missing values "
            "before calling fit()."
        )
    if T0 < 2:
        raise MlsynthDataError(f"RRSC requires at least 2 pre-treatment periods; got {T0}.")
    if T - T0 < 1:                                                # pragma: no cover
        raise MlsynthDataError("RRSC requires at least 1 post-treatment period.")
    if Y0.shape[1] < 2:
        raise MlsynthDataError(
            "RRSC requires at least 2 control units to estimate the factor model."
        )

    Y = np.vstack([y[None, :], Y0.T])                              # N x T, treated row 0
    N = Y.shape[0]
    X = np.concatenate([np.zeros(T0, dtype=int), np.ones(T - T0, dtype=int)])
    unit_names = np.concatenate([np.array([treated_name], dtype=object),
                                 donor_names.astype(object)])

    resolved = regime
    if resolved == "auto":
        resolved = "fixed_n" if T0 >= 5 * N else "large_n"

    return RRSCInputs(
        Y=Y, X=X, unit_names=unit_names, treated_name=treated_name,
        T=T, T0=T0, time_labels=time_labels, regime=resolved,
    )
