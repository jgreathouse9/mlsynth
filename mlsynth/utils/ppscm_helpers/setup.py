"""Long-DataFrame -> NumPy boundary for PPSCM (staggered adoption).

Mirrors augsynth::format_data_stag: derive each unit's first treated period,
split the panel at the *last* adoption time into pre (``X``) and post (``y``),
and index adoption by position in the sorted time vector (Inf for never-treated).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import PPSCMInputs


def prepare_ppscm_inputs(df: pd.DataFrame, *, outcome: str, treat: str,
                         unitid: str, time: str) -> PPSCMInputs:
    t_vec = np.sort(pd.unique(df[time]))
    units = np.sort(pd.unique(df[unitid]))

    adopt = {}
    for u, g in df.groupby(unitid):
        yrs = g.loc[g[treat] == 1, time]
        adopt[u] = yrs.min() if len(yrs) else np.inf
    finite = [a for a in adopt.values() if np.isfinite(a)]
    if not finite:
        raise MlsynthDataError("PPSCM requires at least one treated unit.")
    if len(finite) == len(units):
        raise MlsynthDataError("PPSCM requires at least one never-treated (control) unit.")

    trt = np.array([
        (int(np.where(t_vec == adopt[u])[0][0]) if np.isfinite(adopt[u]) else np.inf)
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

    return PPSCMInputs(
        Xy=Xy, trt=trt, n_pre=int(len(pre_times)),
        time_labels=t_vec, units=units, outcome=outcome,
        intervention_time=t_int,
    )
