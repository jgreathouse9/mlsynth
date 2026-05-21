"""Panel and predictor preparation for SparseSC.

The estimator needs two parallel data structures:

  * A balanced long-format outcome panel with a single treated unit and
    a donor pool, as in canonical SCM.
  * A unit-by-predictor table giving one numeric value per (unit,
    predictor) pair. Predictor names must be supplied explicitly so
    that the V-weights returned later can be mapped back to them.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from .structures import SparseSCInputs


def prepare_sparse_sc_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    predictors_df: pd.DataFrame,
    predictors_unitid: str,
    predictor_cols: Optional[Sequence[str]] = None,
    T0_train: Optional[int] = None,
    standardize: bool = True,
) -> SparseSCInputs:
    """Build SparseSC inputs from an outcome panel and a predictor table.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format balanced panel: one row per ``(unit, time)`` with
        the outcome and a binary treatment indicator.
    outcome, treat, unitid, time : str
        Column names in ``df``.
    predictors_df : pd.DataFrame
        Unit-level predictor table: one row per unit, with a column
        identifying the unit and one column per predictor. Units must
        cover every unit appearing in ``df``.
    predictors_unitid : str
        Column in ``predictors_df`` matching ``unitid`` in ``df``.
    predictor_cols : Sequence[str], optional
        Subset of columns in ``predictors_df`` to use as predictors.
        Defaults to every column except ``predictors_unitid``.
    T0_train : int, optional
        End of the training block within the pre-period (exclusive).
        Defaults to ``floor(T0_total * 0.75)`` -- i.e., a 75/25 split.
    standardize : bool
        Standardize each predictor by its sample standard deviation
        across all units (treated + donors). Default ``True``.
    """

    balance(df, unitid, time)
    prep = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prep:
        raise MlsynthDataError(
            "SparseSC currently supports a single treated unit; the panel "
            "appears to contain multiple treatment cohorts."
        )

    Y1 = np.asarray(prep["y"], dtype=float)
    Y0 = np.asarray(prep["donor_matrix"], dtype=float)
    T = int(prep["total_periods"])
    T0_total = int(prep["pre_periods"])
    treated_unit_name = prep["treated_unit_name"]
    donor_names = list(prep["donor_names"])
    time_labels = np.asarray(prep["time_labels"])

    if T0_total < 4:
        raise MlsynthDataError(
            "SparseSC requires at least 4 pre-treatment periods so the "
            "train/validation split has at least 2 + 2 observations."
        )

    if T0_train is None:
        T0_train = max(2, int(T0_total * 0.75))
    if not (1 < T0_train < T0_total):
        raise MlsynthDataError(
            f"T0_train must lie strictly between 1 and T0_total = {T0_total}; "
            f"got T0_train = {T0_train}."
        )

    # Predictor table sanity checks.
    if predictors_unitid not in predictors_df.columns:
        raise MlsynthDataError(
            f"predictors_unitid '{predictors_unitid}' not found in predictors_df."
        )
    if predictor_cols is None:
        predictor_cols = [c for c in predictors_df.columns
                          if c != predictors_unitid]
    predictor_cols = list(predictor_cols)
    missing = [c for c in predictor_cols if c not in predictors_df.columns]
    if missing:
        raise MlsynthDataError(
            f"predictors_df missing columns: {missing}"
        )

    # Index predictors by unit and align with the treated / donor order.
    pred_indexed = (
        predictors_df.drop_duplicates(subset=[predictors_unitid])
        .set_index(predictors_unitid)
    )
    required_units = [treated_unit_name] + donor_names
    missing_units = [u for u in required_units if u not in pred_indexed.index]
    if missing_units:
        raise MlsynthDataError(
            f"predictors_df missing rows for these units: "
            f"{missing_units[:5]} (and possibly more)."
        )

    X_treated = pred_indexed.loc[treated_unit_name, predictor_cols].to_numpy(dtype=float)
    X_donors = pred_indexed.loc[donor_names, predictor_cols].to_numpy(dtype=float).T
    # X1 shape (P,), X0 shape (P, N)

    if X_treated.ndim != 1 or X_donors.ndim != 2:
        raise MlsynthDataError(
            "Unexpected predictor matrix shape after alignment."
        )

    if standardize:
        big = np.column_stack([X_donors, X_treated.reshape(-1, 1)])
        sd = big.std(axis=1, ddof=1)
        sd = np.where(sd == 0, 1.0, sd)
        X_donors = X_donors / sd[:, None]
        X_treated = X_treated / sd

    return SparseSCInputs(
        Y0=Y0, Y1=Y1,
        X0=X_donors, X1=X_treated,
        T=T, T0_total=T0_total, T0_train=T0_train,
        treated_unit_name=treated_unit_name,
        donor_names=donor_names,
        predictor_names=predictor_cols,
        time_labels=time_labels,
        Ywide=prep["Ywide"],
        outcome=outcome,
    )
