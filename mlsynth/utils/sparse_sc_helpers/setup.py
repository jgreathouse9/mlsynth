"""Panel and predictor preparation for SparseSC.

The estimator takes a single long-format ``df`` (one row per
``(unit, time)``) and constructs both the outcome panel and the
unit-by-predictor matrix internally. Predictors come from two
sources:

* ``covariates`` -- columns in ``df`` whose per-unit pre-treatment
  mean becomes one predictor row.
* ``outcome_lag_periods`` -- specific pre-treatment time labels whose
  outcome values become additional predictor rows (the canonical
  Abadie, Diamond & Hainmueller (2010) lagged-outcome predictors).

The first predictor (first entry of ``covariates`` if any, otherwise
the first outcome lag) is the *anchor* whose V-weight is fixed at 1.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from .structures import SparseSCInputs


def _per_unit_pre_means(
    df: pd.DataFrame,
    column: str,
    unitid: str,
    time: str,
    pre_time_labels: np.ndarray,
    unit_order: Sequence,
) -> np.ndarray:
    """Return per-unit pre-treatment mean of ``column`` in ``unit_order``."""
    pre_mask = df[time].isin(pre_time_labels)
    means = (
        df.loc[pre_mask].groupby(unitid)[column].mean()
    )
    # Reindex to unit_order; missing units (none expected for a balanced
    # panel after `balance()`) come back NaN.
    aligned = means.reindex(unit_order)
    if aligned.isna().any():
        missing = aligned.index[aligned.isna()].tolist()
        raise MlsynthDataError(
            f"Covariate '{column}' has no pre-treatment observations "
            f"for these units: {missing[:5]}"
        )
    return aligned.to_numpy(dtype=float)


def _outcome_at_period(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    period_label: Any,
    unit_order: Sequence,
) -> np.ndarray:
    """Return ``outcome`` at ``period_label`` for every unit in order."""
    row = df.loc[df[time] == period_label, [unitid, outcome]]
    if row.empty:
        raise MlsynthDataError(
            f"outcome_lag_periods entry {period_label!r} not found in "
            f"the '{time}' column."
        )
    aligned = (
        row.drop_duplicates(subset=[unitid])
        .set_index(unitid)[outcome]
        .reindex(unit_order)
    )
    if aligned.isna().any():
        missing = aligned.index[aligned.isna()].tolist()
        raise MlsynthDataError(
            f"Outcome at period {period_label!r} missing for these units: "
            f"{missing[:5]}"
        )
    return aligned.to_numpy(dtype=float)


def prepare_sparse_sc_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    covariates: Optional[Sequence[str]] = None,
    outcome_lag_periods: Optional[Sequence[Any]] = None,
    T0_train: Optional[int] = None,
    standardize: bool = True,
) -> SparseSCInputs:
    """Build SparseSC inputs from a single long-format panel.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format balanced panel: one row per ``(unit, time)`` with
        the outcome, a binary treatment indicator, and any covariates.
    outcome, treat, unitid, time : str
        Column names in ``df``.
    covariates : Sequence[str], optional
        Columns in ``df`` whose per-unit pre-treatment mean becomes a
        predictor row. The first covariate is the anchor (V-weight
        pinned to 1).
    outcome_lag_periods : Sequence, optional
        Specific pre-treatment time labels whose outcome values become
        additional predictor rows. Appended after ``covariates``.
    T0_train : int, optional
        End of the training block within the pre-period (exclusive).
        Defaults to ``floor(T0_total * 0.75)`` -- a 75/25 split.
    standardize : bool
        Standardize each predictor row by its sample standard deviation
        across all units. Default ``True``.
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

    covariates = list(covariates) if covariates else []
    outcome_lag_periods = list(outcome_lag_periods) if outcome_lag_periods else []
    if not covariates and not outcome_lag_periods:
        raise MlsynthDataError(
            "SparseSC needs at least one predictor: provide ``covariates`` "
            "and/or ``outcome_lag_periods``."
        )

    missing_cov = [c for c in covariates if c not in df.columns]
    if missing_cov:
        raise MlsynthDataError(
            f"covariate columns missing from df: {missing_cov}"
        )

    pre_time_labels = time_labels[:T0_total]
    bad_lags = [p for p in outcome_lag_periods if p not in pre_time_labels]
    if bad_lags:
        raise MlsynthDataError(
            f"outcome_lag_periods must lie in the pre-treatment window; "
            f"these are not pre-period: {bad_lags}"
        )

    unit_order = [treated_unit_name] + donor_names

    predictor_rows = []
    predictor_names = []
    for cov in covariates:
        predictor_rows.append(
            _per_unit_pre_means(df, cov, unitid, time,
                                pre_time_labels, unit_order)
        )
        predictor_names.append(cov)
    for lag_period in outcome_lag_periods:
        predictor_rows.append(
            _outcome_at_period(df, outcome, unitid, time,
                               lag_period, unit_order)
        )
        predictor_names.append(f"{outcome}@{lag_period}")

    big = np.vstack(predictor_rows)            # shape (P, N+1)
    X_treated = big[:, 0].astype(float)        # (P,)
    X_donors = big[:, 1:].astype(float)        # (P, N)

    if standardize:
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
        predictor_names=predictor_names,
        time_labels=time_labels,
        Ywide=prep["Ywide"],
        outcome=outcome,
    )
