"""Long-DataFrame -> NumPy boundary for SCMO (the only pandas touchpoint)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..fast_scm_helpers.structure import IndexSet
from .matrix_builder import build_matching_matrix
from .structures import SCMOInputs


def prepare_scmo_inputs(
    df: pd.DataFrame,
    *,
    unitid: str,
    time: str,
    outcome: str,
    spec: Dict[str, Any],
    treated_unit: Any,
    intervention_time: Any,
) -> SCMOInputs:
    """Pivot the panel to NumPy, build ``IndexSet``\\ s and the matching matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel (one row per unit-period).
    unitid, time, outcome : str
        Column names for unit id, time, and the primary outcome.
    spec : dict
        Matching specification consumed by
        :func:`mlsynth.utils.scmo_helpers.matrix_builder.build_matching_matrix`.
    treated_unit : Any
        Label of the treated unit.
    intervention_time : Any
        First treated period; pre-period is ``time < intervention_time``.

    Returns
    -------
    SCMOInputs
        Pure-NumPy container for the estimation engine.
    """
    units = list(pd.unique(df[unitid]))
    if treated_unit not in units:
        raise MlsynthDataError(f"treated_unit {treated_unit!r} not found in '{unitid}'.")
    unit_index = IndexSet.from_labels(units)

    times = np.sort(pd.unique(df[time]))
    time_index = IndexSet.from_labels(times)

    Ywide = df.pivot(index=unitid, columns=time, values=outcome).reindex(unit_index.labels)[times]
    if Ywide.isna().any().any():
        raise MlsynthDataError("SCMO requires a complete outcome panel after pivoting.")
    Y = Ywide.to_numpy(dtype=float)

    T0 = int(np.sum(times < intervention_time))
    if T0 < 1:
        raise MlsynthDataError("No pre-treatment periods (check intervention_time).")

    Z, predictor_labels, col_period = build_matching_matrix(
        df, unitid=unitid, time=time, spec=spec, unit_index=unit_index
    )

    treated_idx = int(unit_index.get_index([treated_unit])[0])
    donor_idx = np.array([i for i in range(len(unit_index)) if i != treated_idx], dtype=int)

    return SCMOInputs(
        unit_index=unit_index,
        time_index=time_index,
        treated_idx=treated_idx,
        donor_idx=donor_idx,
        Y=Y,
        T0=T0,
        Z=Z,
        predictor_labels=predictor_labels,
        col_period=col_period,
        metadata={"spec": spec, "outcome": outcome, "intervention_time": intervention_time},
    )
