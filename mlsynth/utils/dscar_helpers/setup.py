"""Panel reshape for the Dynamic Synthetic Control estimator.

DSC operates on a long-format panel with a treated indicator, an
optional lagged-outcome column (``pm25_lag1`` in the paper's air-
pollution example), and one or more time-varying exogenous covariates.

This module:

1. Validates the columns the user named in :class:`DSCARConfig`.
2. Pivots to wide format ``(N, T)`` with treated units first.
3. Constructs the one-period-lag outcome cube ``Y_lag1`` -- using the
   user-provided lag column for ``t = 1`` and ``Y[:, t - 1]`` for later
   periods.
4. Stacks the exogenous covariates into an ``(N, T, p)`` cube.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import DSCARInputs


def prepare_dsc_inputs(
    df: pd.DataFrame,
    *,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    exog_covariates: Optional[Sequence[str]] = None,
    lagged_outcome: Optional[str] = None,
) -> DSCARInputs:
    """Pivot a long-format panel into the inputs DSC consumes.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel with one row per unit-time.
    outcome, treat, unitid, time : str
        Column names. ``treat`` should be ``1`` at every row where the
        unit is part of the directly-treated group (regardless of pre /
        post timing); the per-row pre/post split is inferred from the
        first period at which any ``treat == 1`` row appears.
    exog_covariates : sequence of str, optional
        Time-varying exogenous covariate columns. When ``None`` the
        DSC matching uses only the lagged outcome.
    lagged_outcome : str, optional
        Column carrying the externally-supplied ``Y_{t-1}`` at the
        first sample period. When ``None``, the lag for ``t = 1`` is
        ``NaN`` for every unit and the corresponding matching
        constraint is dropped at ``t = 1``.

    Returns
    -------
    DSCARInputs
    """
    needed = [outcome, treat, unitid, time]
    if exog_covariates:
        needed.extend(exog_covariates)
    if lagged_outcome:
        needed.append(lagged_outcome)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise MlsynthDataError(
            f"DSC: required column(s) not in df: {missing}."
        )

    df = df.loc[:, list(dict.fromkeys(needed))].copy()
    if df.empty:
        raise MlsynthDataError("DSC: panel is empty.")

    # Determine the treated and donor unit sets.
    treated_set = set(df.loc[df[treat] != 0, unitid].unique())
    if not treated_set:
        raise MlsynthDataError("DSC: no treated rows found (treat == 0 everywhere).")
    donor_set = set(df[unitid].unique()) - treated_set
    if not donor_set:
        raise MlsynthDataError(
            "DSC: no donor units found (every unit appears as treated)."
        )

    treated_labels = tuple(sorted(treated_set, key=str))
    donor_labels = tuple(sorted(donor_set, key=str))
    unit_order = [*treated_labels, *donor_labels]
    n_treated = len(treated_labels)

    # Wide pivot in fixed unit order.
    Y_wide = (
        df.pivot(index=time, columns=unitid, values=outcome)
          .reindex(columns=unit_order)
          .sort_index()
    )
    # Preserve NaNs as a missingness mask -- the R reference drops
    # control units that have any NaN at the current period (per-
    # period complete-case match). The pipeline applies that mask
    # itself, so the outcome panel stays raw here.
    time_labels = Y_wide.index.to_numpy()
    T = Y_wide.shape[0]
    Y = Y_wide.to_numpy(dtype=float).T          # (N, T) ordered

    # Determine T0 from the first time period at which any treated row has
    # treat != 0. The treat indicator is allowed to be 1 on every row of
    # the treated units (the per-unit pre/post split is then read off the
    # time index).
    treat_wide = (
        df.pivot(index=time, columns=unitid, values=treat)
          .reindex(columns=unit_order)
          .sort_index()
          .to_numpy(dtype=float).T              # (N, T)
    )
    # Each treated unit must be marked at the same first post-treatment
    # period (we don't yet support staggered DSC).
    first_on_per_treated = []
    for i in range(n_treated):
        on = np.where(treat_wide[i] > 0)[0]
        if on.size == 0:
            raise MlsynthDataError(
                f"DSC: treated unit {treated_labels[i]!r} never has treat != 0."
            )
        first_on_per_treated.append(int(on.min()))
    if len(set(first_on_per_treated)) > 1:
        raise MlsynthDataError(
            "DSC: treated units must share a common intervention time. "
            f"Got per-unit first-on indices: "
            f"{dict(zip(treated_labels, first_on_per_treated))}."
        )
    T0 = first_on_per_treated[0]
    if T0 < 2:
        raise MlsynthDataError(
            f"DSC: need T0 >= 2 pre-periods; got T0 = {T0}."
        )
    T1 = T - T0
    if T1 < 1:
        raise MlsynthDataError("DSC: no post-treatment periods.")

    # Lagged-outcome cube: Y_lag1[:, t] = Y[:, t-1] for t >= 1; t == 0
    # carries the user-supplied initial lag (or NaN if missing).
    Y_lag1 = np.full((Y.shape[0], T), np.nan)
    Y_lag1[:, 1:] = Y[:, :-1]
    if lagged_outcome is not None:
        first_lag_wide = (
            df[df[time] == np.sort(time_labels)[0]]
              .set_index(unitid)
              [lagged_outcome]
              .reindex(unit_order)
              .to_numpy(dtype=float)
        )
        Y_lag1[:, 0] = first_lag_wide

    # Covariate cube (N, T, p).
    exog_cols = tuple(exog_covariates or ())
    p = len(exog_cols)
    if p > 0:
        X = np.empty((Y.shape[0], T, p), dtype=float)
        for k, col in enumerate(exog_cols):
            X_df = (
                df.pivot(index=time, columns=unitid, values=col)
                  .reindex(columns=unit_order)
                  .sort_index()
            )
            X[:, :, k] = X_df.to_numpy(dtype=float).T
    else:
        X = np.zeros((Y.shape[0], T, 0), dtype=float)

    return DSCARInputs(
        Y=Y,
        Y_lag1=Y_lag1,
        X=X,
        var_names=exog_cols,
        y_name=outcome,
        treated_labels=treated_labels,
        donor_labels=donor_labels,
        time_labels=time_labels,
        N=Y.shape[0],
        T=T,
        T0=T0,
        T1=T1,
        n_treated=n_treated,
    )
