"""Long-DataFrame -> NumPy boundary for RESCM (the only pandas touchpoint)."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..fast_scm_helpers.structure import IndexSet
from .structures import RESCMInputs


def derive_treatment(df: pd.DataFrame, unitid: str, time: str, treat: str) -> Tuple[Any, Any]:
    """Read the single treated unit and its first treated period from ``treat``."""
    treated_rows = df[df[treat] == 1]
    if treated_rows.empty:
        raise MlsynthDataError(f"No treated rows (treat == 1) found in '{treat}'.")
    treated_units = pd.unique(treated_rows[unitid])
    if len(treated_units) != 1:
        raise MlsynthDataError(
            f"RESCM expects exactly one treated unit; found {len(treated_units)}."
        )
    treated_unit = treated_units[0]
    intervention_time = treated_rows.loc[treated_rows[unitid] == treated_unit, time].min()
    return treated_unit, intervention_time


def prepare_rescm_inputs(
    df: pd.DataFrame,
    *,
    unitid: str,
    time: str,
    outcome: str,
    treat: str,
) -> RESCMInputs:
    """Pivot the panel to NumPy, build ``IndexSet``es, split pre/post.

    Returns
    -------
    RESCMInputs
        Pure-NumPy container: treated vector ``y`` (T,), donor matrix ``X``
        (T, N), ``T0``, and unit/time :class:`IndexSet`es.
    """
    treated_unit, intervention_time = derive_treatment(df, unitid, time, treat)

    times = np.sort(pd.unique(df[time]))
    time_index = IndexSet.from_labels(times)

    wide = df.pivot(index=time, columns=unitid, values=outcome).reindex(times)
    if wide.isna().any().any():
        raise MlsynthDataError("RESCM requires a complete outcome panel after pivoting.")

    donors = [u for u in wide.columns if u != treated_unit]
    unit_index = IndexSet.from_labels(donors)

    y = wide[treated_unit].to_numpy(dtype=float)
    X = wide[donors].to_numpy(dtype=float)
    T0 = int(np.sum(times < intervention_time))
    if T0 < 2:
        raise MlsynthDataError("RESCM needs at least two pre-treatment periods.")

    return RESCMInputs(
        unit_index=unit_index,
        time_index=time_index,
        y=y,
        X=X,
        T0=T0,
        treated_label=treated_unit,
        metadata={"outcome": outcome, "intervention_time": intervention_time},
    )
