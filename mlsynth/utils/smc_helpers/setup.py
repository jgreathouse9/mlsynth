"""Long-DataFrame -> NumPy boundary for SMC (the only pandas touchpoint)."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import SMCInputs


def prepare_smc_inputs(
    df: pd.DataFrame,
    *,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    covariates: Optional[Sequence[str]] = None,
) -> SMCInputs:
    """Pivot ``df`` into SMC's ``(Y_treated, Y_donors, T0)`` matrices.

    Single treated unit only: the treated unit is the one with any ``treat == 1``
    row, the donor pool is every never-treated unit. Requires a balanced panel
    with a complete outcome column. Optional ``covariates`` are averaged over the
    pre-treatment window (one matching row each, standardized later).
    """
    treated_rows = df.loc[df[treat] == 1, [unitid, time]]
    if treated_rows.empty:
        raise MlsynthDataError(
            f"SMC requires at least one row with {treat}=1; none found."
        )
    treated_units = treated_rows[unitid].unique()
    if treated_units.size != 1:
        raise MlsynthDataError(
            f"SMC supports a single treated unit; found {treated_units.size}."
        )
    treated_label = treated_units[0]
    intervention_time = treated_rows[time].min()

    all_units = list(df[unitid].unique())
    donors = [
        u for u in all_units
        if u != treated_label and df.loc[df[unitid] == u, treat].sum() == 0
    ]
    if len(donors) == 0:
        raise MlsynthDataError("SMC: donor pool is empty.")

    Y_wide = df.pivot(index=time, columns=unitid, values=outcome).sort_index()
    time_index = Y_wide.index.to_numpy()
    Y_treated = Y_wide[treated_label].to_numpy(dtype=float)
    Y_donors = Y_wide[donors].to_numpy(dtype=float)
    if not (np.isfinite(Y_treated).all() and np.isfinite(Y_donors).all()):
        raise MlsynthDataError(
            "SMC requires a balanced panel with no missing outcomes."
        )

    treatment_period = int(np.argmax(time_index >= intervention_time)) + 1
    T = int(time_index.size)
    T0 = treatment_period - 1
    T1 = T - T0
    if T0 < 2:
        raise MlsynthDataError(
            f"SMC needs at least 2 pre-treatment periods; got T0={T0}."
        )
    if T1 < 1:  # pragma: no cover - unreachable: a treated period forces T1 >= 1
        raise MlsynthDataError(
            f"SMC needs at least 1 post-treatment period; got T1={T1}."
        )

    cov_treated: Optional[np.ndarray] = None
    cov_donors: Optional[np.ndarray] = None
    cov_names: List[Any] = list(covariates or [])
    if cov_names:
        pre_mask = time_index < intervention_time
        treated_vals, donor_vals = [], []
        for c in cov_names:
            cwide = df.pivot(index=time, columns=unitid, values=c).sort_index()
            cwide = cwide.reindex(index=time_index).ffill().bfill()
            block = cwide[[treated_label, *donors]]
            if block.isna().any().any():
                raise MlsynthDataError(
                    f"Covariate '{c}' is entirely missing for some unit; cannot match."
                )
            treated_vals.append(float(cwide.loc[pre_mask, treated_label].mean()))
            donor_vals.append(cwide.loc[pre_mask, donors].mean().to_numpy(dtype=float))
        cov_treated = np.asarray(treated_vals, dtype=float)       # (P,)
        cov_donors = np.vstack(donor_vals)                        # (P, J)

    return SMCInputs(
        Y_treated=Y_treated,
        Y_donors=Y_donors,
        treated_label=treated_label,
        donor_labels=tuple(donors),
        time_index=time_index,
        intervention_time=intervention_time,
        treatment_period=treatment_period,
        T=T, T0=T0, T1=T1, J=len(donors),
        cov_treated=cov_treated,
        cov_donors=cov_donors,
        covariate_names=tuple(cov_names),
    )
