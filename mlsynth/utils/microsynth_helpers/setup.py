"""Long-DataFrame ingestion for MicroSynth.

Converts a long-format panel (one row per ``(user, time)``) into the
matrices the dual solver needs:

* ``X_T``, ``X_C`` -- treated and control covariate matrices, one
  row per user.
* ``Y_T``, ``Y_C`` -- post-treatment outcome matrices, one row per
  user and one column per post-treatment period.

Conventions:

* A "treated user" is any unit that has ``treat = 1`` for at least
  one period (the actual-exposure indicator).
* A "control user" has ``treat = 0`` for every period.
* The cohort time ``T0`` is the first period where any user has
  ``treat = 1``. Users with treatment onsets at different times
  (staggered adoption) are rejected -- MicroSynth assumes a single
  cohort.
* Covariates listed in ``covariates`` must be time-invariant per
  user (a single value per ``user_id``). Time-varying features
  should be collapsed by the caller, or passed via
  ``outcome_lag_periods`` if they're pre-treatment outcomes.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance
from .structures import MicroSynthInputs


def _validate_time_invariant(
    df: pd.DataFrame,
    column: str,
    unitid: str,
) -> None:
    """Raise if ``column`` varies within any unit."""
    n_unique = df.groupby(unitid)[column].nunique(dropna=False)
    bad = n_unique[n_unique > 1]
    if not bad.empty:
        examples = bad.index[:5].tolist()
        raise MlsynthDataError(
            f"Covariate '{column}' is not time-invariant: varies "
            f"within {len(bad)} units (e.g. {examples}). MicroSynth "
            "expects unit-level covariates; collapse time-varying "
            "features before passing, or move pre-period outcomes to "
            "``outcome_lag_periods``."
        )


def _infer_cohort_time(
    df: pd.DataFrame,
    treat: str,
    time: str,
    unitid: str,
) -> Any:
    """Infer the single cohort time from the treatment indicator."""
    treated_rows = df[df[treat] == 1]
    if treated_rows.empty:
        raise MlsynthDataError(
            "No treated rows found (no unit has ``treat = 1``). "
            "MicroSynth requires at least one treated user."
        )
    first_treat_time = treated_rows.groupby(unitid)[time].min()
    distinct_cohorts = first_treat_time.unique()
    if len(distinct_cohorts) > 1:
        raise MlsynthDataError(
            f"MicroSynth requires a single cohort time but the panel "
            f"has {len(distinct_cohorts)} distinct treatment-onset "
            f"times: {sorted(distinct_cohorts)[:5]}. Use a "
            f"staggered-adoption estimator (PPSCM, SequentialSDID) "
            f"instead."
        )
    return distinct_cohorts[0]


def prepare_microsynth_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    covariates: Sequence[str],
    outcome_lag_periods: Optional[Sequence[Any]] = None,
    standardize: bool = True,
    match_outcomes: Optional[Sequence[str]] = None,
) -> MicroSynthInputs:
    """Build MicroSynth inputs from a long-format panel.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel: one row per ``(user, time)``.
    outcome, treat, unitid, time : str
        Column names.
    covariates : Sequence[str]
        Columns in ``df`` to use as balancing covariates. Each must
        be time-invariant per user.
    outcome_lag_periods : Sequence, optional
        Specific pre-treatment time labels whose outcome values
        become additional balancing constraints.
    standardize : bool
        Z-score covariates across all users before fitting.
    match_outcomes : Sequence[str], optional
        Outcome columns whose pre-period values (at
        ``outcome_lag_periods``) are balanced jointly. Defaults to the
        primary ``outcome`` alone (current single-outcome behaviour).

    Returns
    -------
    MicroSynthInputs
    """
    if not covariates and not outcome_lag_periods:
        raise MlsynthDataError(
            "MicroSynth needs at least one balancing constraint: "
            "provide ``covariates`` and/or ``outcome_lag_periods``."
        )

    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(f"Required column '{col}' not in df.")
    for col in covariates:
        if col not in df.columns:
            raise MlsynthDataError(f"Covariate '{col}' not in df.")

    balance(df, unitid, time)

    cohort_time = _infer_cohort_time(df, treat, time, unitid)

    # Identify treated vs control users by whether they ever had treat=1.
    ever_treated = df.groupby(unitid)[treat].max()
    treated_units = ever_treated.index[ever_treated == 1].tolist()
    control_units = ever_treated.index[ever_treated == 0].tolist()
    if not treated_units:
        raise MlsynthDataError("No treated users found.")
    if not control_units:
        raise MlsynthDataError("No control users found (all units treated).")

    # Validate time-invariance of covariates.
    for col in covariates:
        _validate_time_invariant(df, col, unitid)

    # Get one row per unit for time-invariant covariates.
    per_unit = df.drop_duplicates(subset=[unitid]).set_index(unitid)

    cov_T = per_unit.loc[treated_units, list(covariates)].to_numpy(dtype=float)
    cov_C = per_unit.loc[control_units, list(covariates)].to_numpy(dtype=float)
    cov_names = list(covariates)

    # Optional outcome-lag predictors.
    outcome_lag_periods = list(outcome_lag_periods) if outcome_lag_periods else []
    time_labels_all = df[time].unique()
    pre_time_labels = time_labels_all[time_labels_all < cohort_time]
    bad_lags = [p for p in outcome_lag_periods if p not in pre_time_labels]
    if bad_lags:
        raise MlsynthDataError(
            f"outcome_lag_periods must lie in the pre-treatment window "
            f"(before cohort time {cohort_time}); these are not: "
            f"{bad_lags}"
        )
    # Which outcomes' pre-period values to balance (microsynth multi-outcome
    # match.out). Defaults to the primary outcome only.
    lag_outcomes = list(match_outcomes) if match_outcomes else [outcome]
    for oc in lag_outcomes:
        if oc not in df.columns:
            raise MlsynthDataError(f"match_outcomes column '{oc}' not in df.")
    lag_T_cols = []
    lag_C_cols = []
    for oc in lag_outcomes:
        for lag in outcome_lag_periods:
            row = df.loc[df[time] == lag, [unitid, oc]]
            if row.empty:
                raise MlsynthDataError(
                    f"outcome_lag_periods entry {lag!r} not found in '{time}'."
                )
            lag_series = (
                row.drop_duplicates(subset=[unitid])
                .set_index(unitid)[oc]
            )
            try:
                lag_T_cols.append(
                    lag_series.reindex(treated_units).to_numpy(dtype=float)
                )
                lag_C_cols.append(
                    lag_series.reindex(control_units).to_numpy(dtype=float)
                )
            except Exception as exc:
                raise MlsynthDataError(
                    f"Failed to align outcome {oc!r} at lag {lag!r}: {exc}"
                ) from exc
            cov_names.append(f"{oc}@{lag}")

    # Keep the raw (un-standardized) covariate and lag blocks separate; the
    # panel-method QP balances treated *totals* and needs raw values, and it
    # treats covariates (hard equality) and lagged outcomes (soft LS) apart.
    cov_T_raw, cov_C_raw = cov_T.copy(), cov_C.copy()
    if lag_T_cols:
        lag_T_raw = np.column_stack(lag_T_cols)
        lag_C_raw = np.column_stack(lag_C_cols)
        X_T = np.column_stack([cov_T, lag_T_raw])
        X_C = np.column_stack([cov_C, lag_C_raw])
    else:
        lag_T_raw = np.empty((cov_T.shape[0], 0))
        lag_C_raw = np.empty((cov_C.shape[0], 0))
        X_T, X_C = cov_T, cov_C

    if np.isnan(X_T).any() or np.isnan(X_C).any():
        raise MlsynthDataError(
            "Covariate matrix has NaN entries after assembly. "
            "Drop or impute missing values before calling MicroSynth."
        )

    # Pool standardization across treated + control.
    cov_sd: Optional[np.ndarray] = None
    if standardize:
        big = np.vstack([X_T, X_C])
        cov_sd = big.std(axis=0, ddof=1)
        cov_sd = np.where(cov_sd == 0, 1.0, cov_sd)
        X_T = X_T / cov_sd
        X_C = X_C / cov_sd

    # Post-treatment outcomes per user.
    post_df = df[df[time] >= cohort_time]
    post_times = np.sort(post_df[time].unique())
    Y_T = (
        post_df[post_df[unitid].isin(treated_units)]
        .pivot(index=unitid, columns=time, values=outcome)
        .reindex(index=treated_units, columns=post_times)
        .to_numpy(dtype=float)
    )
    Y_C = (
        post_df[post_df[unitid].isin(control_units)]
        .pivot(index=unitid, columns=time, values=outcome)
        .reindex(index=control_units, columns=post_times)
        .to_numpy(dtype=float)
    )
    if np.isnan(Y_T).any() or np.isnan(Y_C).any():
        raise MlsynthDataError(
            "Post-treatment outcome matrix has NaN entries. The panel "
            "is unbalanced over the post-treatment window."
        )
    if Y_T.shape[1] == 1:
        Y_T = Y_T[:, 0]
        Y_C = Y_C[:, 0]

    return MicroSynthInputs(
        X_T=X_T,
        X_C=X_C,
        Y_T=Y_T,
        Y_C=Y_C,
        treated_unit_names=treated_units,
        control_unit_names=control_units,
        covariate_names=cov_names,
        cohort_time=cohort_time,
        covariate_sd=cov_sd,
        outcome=outcome,
        cov_T_raw=cov_T_raw,
        cov_C_raw=cov_C_raw,
        lag_T_raw=lag_T_raw,
        lag_C_raw=lag_C_raw,
    )
