"""Ingestion for STACKEDSC: cohorts, donors, and the base-period indexing.

Ingestion goes through :func:`mlsynth.utils.datautils.dataprep`, which already
returns a staggered panel split by adoption time -- exactly the grouping this
estimator needs, since the base period ``T0i`` is a property of the cohort
rather than of the unit.

That last point is what makes the design batchable. ``normalize`` indexes the
treated unit *and every donor* to 100 at ``T0i``, so donors take one normalised
form per cohort, not one per treated unit. Six cohorts on the Walmart panel
means six donor blocks for 566 treated units.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError

_EPS = 1e-12


class Cohort:
    """One adoption time: a shared donor block and the units that face it."""

    __slots__ = ("adopt", "base_period", "units", "donors", "Y", "D",
                 "X1", "X0", "pred_names", "t0_index")

    def __init__(self, adopt, base_period, units, donors, Y, D, t0_index,
                 X1=None, X0=None, pred_names=None):
        self.adopt = adopt
        self.base_period = base_period
        self.units = units          # treated ids, in column order of Y
        self.donors = donors        # donor ids, in column order of D
        self.Y = Y                  # (T, n_units)
        self.D = D                  # (T, n_donors)
        self.t0_index = t0_index    # row index of base_period
        self.X1 = X1                # (K, n_units) or None
        self.X0 = X0                # (K, n_donors) or None
        self.pred_names = pred_names or []


def adoption_times(df: pd.DataFrame, unitid: str, time: str,
                   treat: str) -> Dict[Any, Any]:
    """First treated period per unit; absent for never-treated units."""
    treated_rows = df[df[treat] == 1]
    if treated_rows.empty:
        raise MlsynthDataError(
            f"no rows have {treat} == 1, so there is nothing to estimate.")
    return treated_rows.groupby(unitid, observed=True)[time].min().to_dict()


def aggregation_weights(df: pd.DataFrame, unitid: str, column: Optional[str],
                        units: List[Any]) -> np.ndarray:
    """Per-unit gamma_i, normalised to sum to one over the treated units.

    The weight is a property of the unit, so a column that varies within one is
    rejected rather than resolved by taking the first value -- there would be
    no principled choice among the values it takes.
    """
    if column is None:
        return np.full(len(units), 1.0 / max(len(units), 1))
    if column not in df.columns:
        raise MlsynthDataError(
            f"agg_weights column {column!r} is not in the DataFrame.")
    sub = df[df[unitid].isin(units)]
    spread = sub.groupby(unitid, observed=True)[column].nunique(dropna=False)
    varying = spread[spread > 1].index.tolist()
    if varying:
        raise MlsynthDataError(
            f"agg_weights column {column!r} varies within unit(s) "
            f"{varying[:5]}; an aggregation weight must be constant per unit.")
    w = sub.groupby(unitid, observed=True)[column].first().reindex(units).to_numpy(float)
    if not np.isfinite(w).all():
        raise MlsynthDataError(
            f"agg_weights column {column!r} has missing values for some "
            f"treated units.")
    if (w < 0).any():
        raise MlsynthDataError(
            f"agg_weights column {column!r} has negative values.")
    total = w.sum()
    if total <= _EPS:
        raise MlsynthDataError(
            f"agg_weights column {column!r} sums to zero over treated units.")
    return w / total


def _covariate_block(df, unitid, time, units, covariates, windows,
                     base_period):
    """(K, n_units) predictor means, averaged over each covariate's window."""
    rows = []
    for c in covariates:
        lo, hi = (windows or {}).get(c, (None, base_period))
        sub = df[df[unitid].isin(units)]
        if lo is not None:
            sub = sub[sub[time] >= lo]
        if hi is not None:
            sub = sub[sub[time] <= hi]
        m = sub.groupby(unitid, observed=True)[c].mean().reindex(units)
        if m.isna().any():
            missing = m[m.isna()].index.tolist()[:5]
            raise MlsynthDataError(
                f"covariate {c!r} has no observations in its window for "
                f"unit(s) {missing}.")
        rows.append(m.to_numpy(float))
    return np.vstack(rows)


def build_cohorts(df: pd.DataFrame, unitid: str, time: str, outcome: str,
                  treat: str, *, normalize: bool = True,
                  covariates: Optional[List[str]] = None,
                  covariate_windows=None) -> Tuple[List[Cohort], List[Any]]:
    """Split the panel into adoption cohorts with their donor blocks.

    Returns the cohorts and the never-treated donor ids.
    """
    from ..datautils import dataprep  # local: keeps import cost off the module

    adopt = adoption_times(df, unitid, time, treat)
    donors = sorted(u for u in df[unitid].unique() if u not in adopt)
    if not donors:
        raise MlsynthDataError(
            "no never-treated units: a stacked synthetic control needs a donor "
            "pool that is never treated over the sample window.")

    prep = dataprep(df, unitid, time, outcome, treat)
    periods = np.asarray(prep["time_labels"])
    wide = df.pivot(index=time, columns=unitid, values=outcome).reindex(
        periods)

    # dataprep returns a `cohorts` mapping for a staggered panel, but flattens
    # to top-level keys when there is exactly one treated unit. Normalise the
    # two shapes here so the rest of the estimator sees one thing.
    specs = prep.get("cohorts")
    if specs is None:
        only = prep["treated_unit_name"]
        specs = {adopt[only]: {"treated_units": [only],
                               "pre_periods": prep["pre_periods"]}}

    cohorts: List[Cohort] = []
    for g, spec in sorted(specs.items()):
        units = list(spec["treated_units"])
        t0 = int(spec["pre_periods"]) - 1          # index of T0i
        if t0 < 0:
            raise MlsynthDataError(
                f"cohort adopting at {g} has no pre-treatment period.")
        base_period = periods[t0]
        Y = wide[units].to_numpy(float)
        D = wide[donors].to_numpy(float)
        if normalize:
            by, bd = Y[t0].copy(), D[t0].copy()
            if (np.abs(by) < _EPS).any() or (np.abs(bd) < _EPS).any():
                raise MlsynthDataError(
                    f"cohort adopting at {g} has a zero outcome at its base "
                    f"period {base_period}, so it cannot be indexed to it; "
                    f"pass normalize=False or shift the outcome.")
            Y, D = 100.0 * Y / by, 100.0 * D / bd
        X1 = X0 = None
        names: List[str] = []
        if covariates:
            X1 = _covariate_block(df, unitid, time, units, covariates,
                                  covariate_windows, base_period)
            X0 = _covariate_block(df, unitid, time, donors, covariates,
                                  covariate_windows, base_period)
            names = list(covariates)
        cohorts.append(Cohort(g, base_period, units, donors, Y, D, t0,
                              X1, X0, names))
    return cohorts, donors


def event_window(cohorts: List[Cohort], n_lags: Optional[int],
                 n_leads: Optional[int]) -> np.ndarray:
    """The event-time grid every cohort can supply, unless overridden.

    Defaulting to the common window is the reference implementation's
    ``balanced``: reporting a horizon only some cohorts reach would make the
    average change composition along the x-axis.
    """
    # e = row - t0_index - 1, so the base period sits at e = -1 and the first
    # treated period at e = 0. Both ends take the *tightest* cohort, since a
    # horizon only some cohorts reach would change the average's composition
    # along the x-axis.
    lo = max(-c.t0_index - 1 for c in cohorts)
    hi = min(c.Y.shape[0] - c.t0_index - 2 for c in cohorts)
    if n_lags is not None:
        lo = max(lo, -int(n_lags))
    if n_leads is not None:
        hi = min(hi, int(n_leads) - 1)
    if hi < 0:
        raise MlsynthDataError(
            "no post-treatment period is observed for every cohort.")
    return np.arange(lo, hi + 1)
