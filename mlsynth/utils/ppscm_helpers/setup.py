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
from ..datautils import balance, dataprep
from ..prepared import SinglePanel
from .structures import PPSCMInputs


def _adoption_index(prep) -> dict:
    """Each treated unit's first treated period, as a position in the time axis.

    ``dataprep`` has already refused a treatment that switches off, so a unit
    is treated from that position to the end of the panel.
    """
    if isinstance(prep, SinglePanel):
        return {prep.treated_unit_name: int(prep.pre_periods)}
    return {unit: int(cohort.pre_periods)
            for cohort in prep.cohorts.values() for unit in cohort.treated_units}


def prepare_ppscm_inputs(df: pd.DataFrame, *, outcome: str, treat: str,
                         unitid: str, time: str,
                         covariates: Optional[List[str]] = None) -> PPSCMInputs:
    """Build :class:`PPSCMInputs` from a long panel through ``balance`` and ``dataprep``.

    ``balance`` and ``dataprep`` refuse a duplicate or absent unit-period, a
    missing or non-finite outcome anywhere in the panel, a non-binary
    treatment, and a treatment that switches off after it starts. The adoption
    bookkeeping then follows augsynth's ``format_data_stag`` (Ben-Michael,
    Feller and Rothstein, JRSS-B 84 (2022), section 2): each unit's adoption is
    its first treated period, indexed by position in the sorted time vector
    (``inf`` for never-treated units), and the panel is split at the last
    adoption into the pre block and the rest.
    """
    if covariates:
        missing = [c for c in covariates if c not in df.columns]
        if missing:
            raise MlsynthDataError(
                f"Covariate column(s) not in DataFrame: {missing}."
            )

    keys = balance(df, unitid, time)
    prep = dataprep(df, unitid, time, outcome, treat, keys=keys)
    t_vec = np.asarray(prep.time_labels)
    units = np.sort(np.asarray(prep.Ywide.columns))

    starts = _adoption_index(prep)
    if len(starts) == len(units):
        raise MlsynthDataError("PPSCM requires at least one never-treated (control) unit.")
    trt = np.array([float(starts[u]) if u in starts else np.inf for u in units],
                   dtype=float)

    last = max(starts.values())
    t_int = t_vec[last]                                  # last adoption time
    if last < 2:
        raise MlsynthDataError("PPSCM needs at least two pre-treatment periods.")

    Xy = np.ascontiguousarray(prep.Ywide[units].to_numpy(dtype=float).T)
    finite = [t_vec[i] for i in starts.values()]

    Z = None
    cov_names = None
    if covariates:
        # augsynth cov_agg default: mean over periods before the FIRST adoption.
        first_adopt = min(finite)
        pre_first = df[df[time] < first_adopt]
        agg = (pre_first.groupby(unitid, observed=True)[list(covariates)].mean()
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
        Xy=Xy, trt=trt, n_pre=int(last),
        time_labels=t_vec, units=units, outcome=outcome,
        intervention_time=t_int, Z=Z, cov_names=cov_names,
    )
