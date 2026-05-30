"""Pivot a long panel into the matrices the MASC estimator consumes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import MASCInputs


def prepare_masc_inputs(
    df: pd.DataFrame,
    *,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    covariates: Optional[Sequence[str]] = None,
    covariate_windows: Optional[Dict[str, Tuple[Any, Any]]] = None,
) -> MASCInputs:
    """Pivot ``df`` into MASC's ``(Y_treated, Y_donors, treatment_period)``.

    Single-treated-unit only for v1. The treated unit is the panel
    unit with any ``treat == 1`` row; the donor pool is every other
    never-treated unit. ``treatment_period`` is the 1-indexed position
    of the first treated period (matches the R reference's
    ``treatment`` argument).
    """
    treated_rows = df.loc[df[treat] == 1, [unitid, time]]
    if treated_rows.empty:
        raise MlsynthDataError(
            f"MASC requires at least one row with {treat}=1; none found."
        )
    treated_units = treated_rows[unitid].unique()
    if treated_units.size != 1:
        raise MlsynthDataError(
            "MASC currently supports a single treated unit; found "
            f"{treated_units.size}."
        )
    treated_label = treated_units[0]
    intervention_time = treated_rows[time].min()

    all_units = list(df[unitid].unique())
    never_treated = [
        u for u in all_units
        if u != treated_label
        and df.loc[df[unitid] == u, treat].sum() == 0
    ]
    if len(never_treated) == 0:
        raise MlsynthDataError("MASC: donor pool is empty.")

    Y_wide = (
        df.pivot(index=time, columns=unitid, values=outcome)
        .sort_index()
    )
    time_index = Y_wide.index.to_numpy()

    Y_treated = Y_wide[treated_label].to_numpy(dtype=float)
    Y_donors = Y_wide[never_treated].to_numpy(dtype=float)

    if not (np.isfinite(Y_treated).all() and np.isfinite(Y_donors).all()):
        raise MlsynthDataError(
            "MASC requires a balanced panel with no missing outcomes."
        )

    # Convert intervention_time to 1-indexed treatment_period.
    treatment_period = int(np.argmax(time_index >= intervention_time)) + 1
    T = int(time_index.size)
    T0 = treatment_period - 1
    T1 = T - T0
    if T0 < 2:
        raise MlsynthDataError(
            f"MASC needs at least 2 pre-treatment periods; got T0={T0}."
        )
    if T1 < 1:
        raise MlsynthDataError(
            f"MASC needs at least 1 post-treatment period; got T1={T1}."
        )

    # Covariate panels (optional). Stored as ``(T, P)`` for the treated unit
    # and ``(T, J, P)`` for donors so CV folds can aggregate within their own
    # pre-window. ``covariate_windows`` (optional) overrides the aggregation
    # range per covariate -- defaults to "1..fold_treatment - 1" inside
    # downstream callers.
    cov_treated_panel: Optional[np.ndarray] = None
    cov_donors_panel: Optional[np.ndarray] = None
    if covariates:
        treated_stacks: List[np.ndarray] = []
        donor_stacks: List[np.ndarray] = []
        for c in covariates:
            if c not in df.columns:
                raise MlsynthDataError(f"Covariate {c!r} not found in df.columns.")
            cwide = (
                df.pivot(index=time, columns=unitid, values=c)
                .reindex(time_index)
                .ffill().bfill()
            )
            if cwide[[treated_label, *never_treated]].isna().any().any():
                raise MlsynthDataError(
                    f"Covariate {c!r} is missing for some unit-period after "
                    "forward/back fill; cannot aggregate."
                )
            treated_stacks.append(
                cwide[treated_label].to_numpy(dtype=float)
            )
            donor_stacks.append(
                cwide[list(never_treated)].to_numpy(dtype=float)
            )
        cov_treated_panel = np.column_stack(treated_stacks)         # (T, P)
        cov_donors_panel = np.stack(donor_stacks, axis=-1)          # (T, J, P)

    return MASCInputs(
        Y_treated=Y_treated,
        Y_donors=Y_donors,
        treated_label=treated_label,
        donor_labels=tuple(never_treated),
        time_index=time_index,
        intervention_time=intervention_time,
        treatment_period=treatment_period,
        T=T,
        T0=T0,
        T1=T1,
        J=Y_donors.shape[1],
        cov_treated_panel=cov_treated_panel,
        cov_donors_panel=cov_donors_panel,
        covariate_names=tuple(covariates or ()),
        covariate_windows=dict(covariate_windows or {}),
    )
