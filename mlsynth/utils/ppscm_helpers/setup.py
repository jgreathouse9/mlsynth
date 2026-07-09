"""Long-DataFrame -> NumPy boundary for PPSCM (staggered adoption).

Mirrors augsynth::format_data_stag: derive each unit's first treated period,
split the panel at the *last* adoption time into pre (``X``) and post (``y``),
and index adoption by position in the sorted time vector (Inf for never-treated).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import PPSCMInputs


def _adopted(a) -> bool:
    """True if ``a`` is an actual adoption time, False for the never-treated
    sentinel ``np.inf``.

    Adoption times are whatever the ``time`` column holds -- an integer/float
    year or a ``pd.Timestamp`` -- so we cannot use ``np.isfinite`` (it rejects
    datetimes). Only the never-treated units carry the float ``np.inf`` marker.
    """
    return not (isinstance(a, float) and np.isinf(a))


def prepare_ppscm_inputs(df: pd.DataFrame, *, outcome: str, treat: str,
                         unitid: str, time: str,
                         covariates: Optional[List[str]] = None) -> PPSCMInputs:
    t_vec = np.sort(pd.unique(df[time]))
    units = np.sort(pd.unique(df[unitid]))

    if covariates:
        missing = [c for c in covariates if c not in df.columns]
        if missing:
            raise MlsynthDataError(
                f"Covariate column(s) not in DataFrame: {missing}."
            )

    adopt = {}
    for u, g in df.groupby(unitid):
        yrs = g.loc[g[treat] == 1, time]
        adopt[u] = yrs.min() if len(yrs) else np.inf
    finite = [a for a in adopt.values() if _adopted(a)]
    if not finite:
        raise MlsynthDataError("PPSCM requires at least one treated unit.")
    if len(finite) == len(units):
        raise MlsynthDataError("PPSCM requires at least one never-treated (control) unit.")

    trt = np.array([
        (int(np.where(t_vec == adopt[u])[0][0]) if _adopted(adopt[u]) else np.inf)
        for u in units
    ], dtype=float)

    t_int = max(finite)                                  # last adoption time
    pre_times = t_vec[t_vec < t_int]
    if len(pre_times) < 2:
        raise MlsynthDataError("PPSCM needs at least two pre-treatment periods.")

    wide = df.pivot(index=unitid, columns=time, values=outcome).reindex(index=units)
    Xy = wide.to_numpy(dtype=float)
    if np.isnan(Xy[:, : len(pre_times)]).any():
        raise MlsynthDataError(
            "PPSCM requires a complete outcome panel over the analysis window; "
            "restrict to years with measurements for every unit."
        )

    Z = None
    cov_names = None
    if covariates:
        # augsynth cov_agg default: mean over periods before the FIRST adoption.
        first_adopt = min(finite)
        pre_first = df[df[time] < first_adopt]
        agg = (pre_first.groupby(unitid)[list(covariates)].mean()
               .reindex(index=units))
        if agg.isna().any().any():
            bad = agg.index[agg.isna().any(axis=1)].tolist()
            raise MlsynthDataError(
                "PPSCM covariates have missing values for unit(s) "
                f"{bad}; drop them or supply complete covariates."
            )
        Z = agg.to_numpy(dtype=float)
        cov_names = tuple(covariates)

    return PPSCMInputs(
        Xy=Xy, trt=trt, n_pre=int(len(pre_times)),
        time_labels=t_vec, units=units, outcome=outcome,
        intervention_time=t_int, Z=Z, cov_names=cov_names,
    )
