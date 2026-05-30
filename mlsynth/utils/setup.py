"""Long-DataFrame → NumPy boundary for MUSC (the only pandas touchpoint)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..fast_scm_helpers.structure import IndexSet
from .structures import MUSCInputs


def prepare_musc_inputs(
    df: pd.DataFrame,
    *,
    unitid: str,
    time: str,
    outcome: str,
    treated_unit: Any,
    intervention_time: Any,
) -> MUSCInputs:
    """Pivot the long panel to NumPy and build the ``MUSCInputs``.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel (one row per unit-period).
    unitid, time, outcome : str
        Column names for the unit id, period index, and outcome.
    treated_unit : Any
        Label of the treated unit.
    intervention_time : Any
        First treated period; pre-period is ``time < intervention_time``.

    Returns
    -------
    MUSCInputs
        Pure-NumPy container for the MUSC engine.
    """
    units = list(pd.unique(df[unitid]))
    if treated_unit not in units:
        raise MlsynthDataError(
            f"treated_unit {treated_unit!r} not found in column '{unitid}'."
        )
    unit_index = IndexSet.from_labels(units)
    treated_idx = int(unit_index.get_index([treated_unit])[0])
    donor_idx = np.asarray(
        [i for i in range(len(units)) if i != treated_idx], dtype=int
    )
    if donor_idx.size < 3:
        raise MlsynthDataError(
            "MUSC requires at least 3 donor units (N - 1 >= 3); got "
            f"{donor_idx.size}."
        )

    times = list(pd.unique(df[time]))
    time_index = IndexSet.from_labels(times)

    wide = df.pivot(index=time, columns=unitid, values=outcome)
    wide = wide.reindex(index=times, columns=units)
    if wide.isna().any().any():
        bad = wide.isna().sum().sum()
        raise MlsynthDataError(
            f"Outcome panel has {int(bad)} missing entries after pivoting; "
            "MUSC requires a strongly balanced panel."
        )
    Y = wide.to_numpy(dtype=float).T                            # (N, T)

    if intervention_time not in times:
        raise MlsynthDataError(
            f"intervention_time {intervention_time!r} not in time column."
        )
    T0 = int(time_index.get_index([intervention_time])[0])
    if T0 < 2:
        raise MlsynthDataError(
            f"MUSC requires at least 2 pre-treatment periods; got T0={T0}."
        )
    if T0 >= Y.shape[1]:
        raise MlsynthDataError(
            "intervention_time must leave at least one post-treatment period."
        )

    return MUSCInputs(
        unit_index=unit_index,
        time_index=time_index,
        treated_idx=int(treated_idx),
        donor_idx=donor_idx,
        Y=Y,
        T0=T0,
        metadata={
            "outcome": outcome,
            "treated_unit": treated_unit,
            "intervention_time": intervention_time,
        },
    )
