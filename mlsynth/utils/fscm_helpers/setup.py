"""Long-DataFrame -> NumPy boundary for FSCM (the only pandas touchpoint)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import _wide_pivot, balance, dataprep
from ..fast_scm_helpers.structure import IndexSet
from ..prepared import SinglePanel
from .structures import FSCMInputs


def derive_treatment(df: pd.DataFrame, unitid: str, time: str, treat: str) -> Tuple[Any, Any]:
    """Read the single treated unit and its first treated period from ``treat``."""
    treated_rows = df[df[treat] == 1]
    if treated_rows.empty:
        raise MlsynthDataError(f"No treated rows (treat == 1) found in '{treat}'.")
    treated_units = pd.unique(treated_rows[unitid])
    if len(treated_units) != 1:
        raise MlsynthDataError(
            f"FSCM expects exactly one treated unit; found {len(treated_units)}."
        )
    treated_unit = treated_units[0]
    intervention_time = treated_rows.loc[treated_rows[unitid] == treated_unit, time].min()
    return treated_unit, intervention_time


def prepare_fscm_inputs(
    df: pd.DataFrame,
    *,
    unitid: str,
    time: str,
    outcome: str,
    treat: str,
    covariates: Optional[List[str]] = None,
    covariate_windows: Optional[Dict[str, Tuple[Any, Any]]] = None,
    match_periods: Optional[List[Any]] = None,
) -> FSCMInputs:
    """Build :class:`FSCMInputs` from a long panel through ``balance`` and ``dataprep``.

    ``dataprep`` supplies the outcome matrix, the time labels, the treated unit
    and the period its treatment starts. ``balance`` and ``dataprep`` refuse a
    duplicate or absent unit-period, a missing or non-finite outcome anywhere in
    the panel, a non-binary treatment, a treatment that switches off after it
    starts, a panel with no treated unit and one with no donor. FSCM's own
    requirements stay on top: exactly one treated unit (Cerulli 2024 studies a
    single treated unit) and four pre-treatment periods to cross-validate on.

    Covariate predictors are each averaged over an aggregation window (Abadie's
    specification): ``covariate_windows`` maps a covariate to an inclusive
    ``(start, end)`` label range; covariates not listed are averaged over the
    full pre-treatment period. A covariate's gaps are filled forward, then
    backward, within each unit before it is averaged, so each covariate is read
    with ``datautils``' pivot on the key scan ``balance`` took, not through
    ``dataprep``'s covariate block, which neither fills gaps nor takes windows.
    ``match_periods`` are specific pre-treatment periods whose outcome value is
    matched directly.
    """
    keys = balance(df, unitid, time)
    prep = dataprep(df, unitid, time, outcome, treat, keys=keys)
    if not isinstance(prep, SinglePanel):
        n_treated = sum(len(c.treated_units) for c in prep.cohorts.values())
        raise MlsynthDataError(
            f"FSCM expects exactly one treated unit; found {n_treated}."
        )
    treated_unit = prep.treated_unit_name
    windows = dict(covariate_windows or {})

    times = np.asarray(prep.time_labels)
    time_index = IndexSet.from_labels(times)

    wide = prep.Ywide
    donors = list(prep.donor_names)
    unit_index = IndexSet.from_labels(donors)

    y = wide[treated_unit].to_numpy(dtype=float)
    Y = wide[donors].to_numpy(dtype=float)
    T0 = int(prep.pre_periods)
    intervention_time = prep.time_labels[T0]
    if T0 < 4:
        raise MlsynthDataError(
            "FSCM needs at least four pre-treatment periods to cross-validate."
        )

    pre_mask = times < intervention_time
    cov_treated = np.empty(0)
    cov_donors = np.empty((len(donors), 0))
    cov_names: List[str] = list(covariates or [])
    if cov_names:
        treated_vals, donor_vals = [], []
        for c in cov_names:
            cwide = _wide_pivot(df, index=time, columns=unitid, values=c, keys=keys)
            cwide = cwide.ffill().bfill()
            if cwide[[treated_unit, *donors]].isna().any().any():
                raise MlsynthDataError(
                    f"Covariate '{c}' is entirely missing for some unit; cannot match."
                )
            if c in windows:
                start, end = windows[c]
                mask = (times >= start) & (times <= end)
                if not mask.any():
                    raise MlsynthDataError(
                        f"Covariate window {windows[c]!r} for '{c}' selects no periods."
                    )
            else:
                mask = pre_mask
            treated_vals.append(float(cwide.loc[mask, treated_unit].mean()))
            donor_vals.append(cwide.loc[mask, donors].mean().to_numpy(dtype=float))
        cov_treated = np.asarray(treated_vals)               # (P,)
        cov_donors = np.column_stack(donor_vals)             # (N, P)

    match_idx = np.empty(0, dtype=int)
    match_labels: List[Any] = []
    if match_periods:
        resolved = []
        for p in match_periods:
            if p not in time_index.label_to_idx:
                raise MlsynthDataError(f"Match period {p!r} not found in '{time}'.")
            ti = int(time_index.label_to_idx[p])
            if ti >= T0:
                raise MlsynthDataError(
                    f"Match period {p!r} is not pre-treatment (must precede the "
                    "intervention)."
                )
            resolved.append(ti)
        match_idx = np.array(sorted(set(resolved)), dtype=int)
        match_labels = list(match_periods)

    return FSCMInputs(
        unit_index=unit_index,
        time_index=time_index,
        y=y,
        Y=Y,
        T0=T0,
        treated_label=treated_unit,
        cov_treated=cov_treated,
        cov_donors=cov_donors,
        covariate_names=cov_names,
        match_idx=match_idx,
        match_periods=match_labels,
        metadata={"outcome": outcome, "intervention_time": intervention_time},
    )
