"""Panel ingestion for the Dynamic Synthetic Control estimator.

DSC operates on a long-format panel with a treated indicator, an
optional lagged-outcome column (``pm25_lag1`` in the paper's air-
pollution example), and one or more time-varying exogenous covariates.

This module:

1. Validates the columns the user named in :class:`DSCARConfig`.
2. Reads the outcome and the treatment through
   :func:`mlsynth.utils.datautils.dataprep` with
   ``missing_outcomes="keep"``, and each covariate and the lag column
   through their own ``dataprep`` call on the same key scan, into ``(N, T)``
   matrices with the treated units first.
3. Constructs the one-period-lag outcome cube ``Y_lag1`` -- using the
   user-provided lag column for ``t = 1`` and ``Y[:, t - 1]`` for later
   periods.
4. Stacks the exogenous covariates into an ``(N, T, p)`` cube.

A missing value stays ``NaN``: the pipeline matches each period on the donors
with complete records (:func:`.pipeline._build_period_targets`) and averages a
treated unit's missing covariate or lag through. A missing outcome of a treated
unit at or after the intervention is refused
(:func:`_refuse_missing_treated_post`).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from ..prepared import SinglePanel
from .structures import DSCARInputs


def _refuse_missing_treated_post(Y: np.ndarray, n_treated: int, T0: int,
                                 treated_labels, time_labels, outcome: str) -> None:
    """Raise if a treated unit's outcome is missing at or after the intervention.

    DSCAR's treated mean at a period averages the treated outcomes, and the ATT
    is the post-period mean of the gap (Zheng and Chen 2024, Section 2). The
    paper's application has no missing treated outcome, and the per-period
    complete-case rule this port documents selects donors; neither says
    whether a treated unit missing its outcome after the intervention leaves
    that period's treated mean or makes it undefined, so the cell is refused.
    The first such cell is the first in time order, then unit order.
    """
    bad = np.isnan(Y[:n_treated, T0:])
    if not bad.any():
        return
    t, i = np.argwhere(bad.T)[0]
    raise MlsynthDataError(
        f"Missing outcome for a treated unit after the intervention: {int(bad.sum())} treated "
        f"post-treatment cell(s) of '{outcome}' are NaN, the first at treated unit "
        f"'{treated_labels[i]}', time {time_labels[T0 + t]}. DSCAR's treated mean averages "
        "the treated outcomes at each period, and DSCAR (Zheng and Chen 2024) does not define "
        "it when one of them is missing. Drop or fill that unit-period before fitting; a "
        "missing value anywhere else is handled by the per-period complete-case match."
    )


def _unit_rows(prep, unit_order) -> np.ndarray:
    """``prep.Ywide`` as an ``(N, T)`` float matrix with rows in ``unit_order``.

    The chain is the one the old pivot ended with (reindex the columns, sort
    the index, transpose), so the matrix has the old memory layout on every
    pandas version: column-major on pandas 3, row-major on pandas 2. The index
    is already sorted. The per-period weights read donor outcomes through
    strided dot products, and a change of layout moves the orange-alert ATT
    by about 1e-6.
    """
    return prep.Ywide.reindex(columns=unit_order).sort_index().to_numpy(dtype=float).T


def _series(df, unitid, time, column, treat, keys, unit_order, what):
    """One column of the panel as an ``(N, T)`` matrix in ``unit_order``, NaN kept."""
    try:
        prep = dataprep(df, unitid, time, column, treat, keys=keys,
                        missing_outcomes="keep")
    except MlsynthDataError as exc:
        raise MlsynthDataError(f"{what} '{column}' cannot be read: {exc}") from exc
    return _unit_rows(prep, unit_order)


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
    """Read a long-format panel into the inputs DSC consumes.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel with one row per unit-time.
    outcome, treat, unitid, time : str
        Column names. ``treat`` is ``1`` at the treated units' rows from
        the intervention period on and ``0`` elsewhere; every treated unit
        shares the intervention period.
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

    Raises
    ------
    MlsynthDataError
        A named column is absent or the panel is empty; ``balance`` and
        ``dataprep`` refuse an absent or duplicate unit-period, an infinite
        outcome, covariate or lag value, a missing or non-binary treatment
        and a treatment that switches off; the treated units do not share
        an intervention period, fewer than two periods precede it, or no
        donor is left; a treated unit's outcome is missing at or after the
        intervention.
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

    if df.empty:
        raise MlsynthDataError("DSC: panel is empty.")

    # balance() refuses a duplicate or absent unit-period; its key scan is
    # shared by every dataprep call below.
    keys = balance(df, unitid, time)
    prep = dataprep(df, unitid, time, outcome, treat, keys=keys,
                    missing_outcomes="keep")
    if isinstance(prep, SinglePanel):
        starts = {prep.treated_unit_name: int(prep.pre_periods)}
    else:
        starts = {unit: int(cohort.pre_periods)
                  for cohort in prep.cohorts.values() for unit in cohort.treated_units}

    # Treated units first, then donors, each sorted by their string form.
    columns = prep.Ywide.columns.to_numpy()
    treated_labels = tuple(sorted((u for u in columns if u in starts), key=str))
    donor_labels = tuple(sorted((u for u in columns if u not in starts), key=str))
    if not donor_labels:
        raise MlsynthDataError(
            "DSC: no donor units found (every unit appears as treated)."
        )
    unit_order = [*treated_labels, *donor_labels]
    n_treated = len(treated_labels)

    # Each treated unit must be marked at the same first post-treatment
    # period (we don't yet support staggered DSC).
    if len(set(starts.values())) > 1:
        raise MlsynthDataError(
            "DSC: treated units must share a common intervention time. "
            f"Got per-unit first-on indices: "
            f"{ {u: starts[u] for u in treated_labels} }."
        )
    time_labels = np.asarray(prep.time_labels)
    T = int(time_labels.size)
    Y = _unit_rows(prep, unit_order)                             # (N, T) ordered
    T0 = starts[treated_labels[0]]
    if T0 < 2:
        raise MlsynthDataError(
            f"DSC: need T0 >= 2 pre-periods; got T0 = {T0}."
        )
    T1 = T - T0
    _refuse_missing_treated_post(Y, n_treated, T0, treated_labels, time_labels, outcome)

    # Lagged-outcome cube: Y_lag1[:, t] = Y[:, t-1] for t >= 1; t == 0
    # carries the user-supplied initial lag (or NaN if missing).
    Y_lag1 = np.full((Y.shape[0], T), np.nan)
    Y_lag1[:, 1:] = Y[:, :-1]
    if lagged_outcome is not None:
        Y_lag1[:, 0] = _series(df, unitid, time, lagged_outcome, treat, keys,
                               unit_order, "Lagged outcome")[:, 0]

    # Covariate cube (N, T, p).
    exog_cols = tuple(exog_covariates or ())
    p = len(exog_cols)
    if p > 0:
        X = np.empty((Y.shape[0], T, p), dtype=float)
        for k, col in enumerate(exog_cols):
            X[:, :, k] = _series(df, unitid, time, col, treat, keys, unit_order,
                                 "Covariate")
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
