"""Micro-panel data preparation for the Distributional Synthetic Control estimator.

DSC operates at a *finer granularity* than the rest of mlsynth: each
``(unit, time)`` cell carries multiple individual observations rather
than a single aggregated outcome. This module converts a long-format
DataFrame (one row per individual observation) into the
:class:`DSCInputs` container that the orchestrator expects.

Treatment is still applied at the unit-time level, exactly as in
classical SCM. The treated unit is identified from the ``treat`` column:
the unique unit-id with ``treat == 1`` for some ``t``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import DSCInputs


def prepare_dsc_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> DSCInputs:
    """Pivot a long-format micro-panel into :class:`DSCInputs`.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel where each row is one individual observation.
        Must contain columns ``unitid``, ``time``, ``outcome``, and
        ``treat`` (the treatment indicator, set at the unit-time level;
        identical for all individual rows within a given cell).
    outcome : str
        Column with the individual-level outcome value.
    treat : str
        Column with the 0/1 treatment indicator.
    unitid : str
        Column with the unit identifier.
    time : str
        Column with the time identifier.
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(
                f"Required column {col!r} missing from input DataFrame."
            )

    if df[outcome].isna().any():
        raise MlsynthDataError("Outcome column contains NaN values.")

    treat_by_cell = (
        df.groupby([unitid, time])[treat].first().reset_index()
    )
    treated_cells = treat_by_cell[treat_by_cell[treat] == 1]
    if treated_cells.empty:
        raise MlsynthDataError(
            "No treated (unit, time) cell found -- need at least one row "
            f"with {treat}=1."
        )
    treated_units = treated_cells[unitid].unique()
    if treated_units.size != 1:
        raise MlsynthDataError(
            f"DSC requires exactly one treated unit; found {treated_units}."
        )
    treated_unit_name = treated_units[0]

    time_labels = np.array(sorted(df[time].unique()))
    T = int(time_labels.size)

    # T0 = number of consecutive pre-treatment periods for the
    # treated unit: largest t* such that treat==0 for all t <= t*.
    treated_cells_sorted = (
        treat_by_cell[treat_by_cell[unitid] == treated_unit_name]
        .sort_values(time)
        .reset_index(drop=True)
    )
    treat_series = treated_cells_sorted[treat].to_numpy()
    if treat_series[0] == 1:
        raise MlsynthDataError(
            "Treated unit has no pre-period (treat=1 at the earliest time)."
        )
    T0 = int(np.argmax(treat_series == 1))
    if T0 == 0 and treat_series[0] == 0 and not treat_series.any():
        raise MlsynthDataError(
            "Treated unit never receives treatment (treat is 0 at every t)."
        )
    if T - T0 < 1:
        raise MlsynthDataError("DSC needs at least one post-treatment period.")
    if T0 < 1:
        raise MlsynthDataError("DSC needs at least one pre-treatment period.")

    donor_units = [u for u in sorted(df[unitid].unique()) if u != treated_unit_name]
    unit_names = [treated_unit_name] + donor_units

    cell_samples: Dict[Tuple[Any, Any], np.ndarray] = {}
    for (u, t), grp in df.groupby([unitid, time]):
        arr = grp[outcome].to_numpy(dtype=float)
        if arr.size == 0:
            raise MlsynthDataError(
                f"Empty (unit, time) cell at ({u!r}, {t!r})."
            )
        cell_samples[(u, t)] = arr

    expected_cells = len(unit_names) * T
    if len(cell_samples) != expected_cells:
        missing = []
        for u in unit_names:
            for t in time_labels:
                if (u, t) not in cell_samples:
                    missing.append((u, t))
        raise MlsynthDataError(
            f"Panel is not balanced; missing {len(missing)} cell(s); "
            f"first 5: {missing[:5]}"
        )

    return DSCInputs(
        cell_samples=cell_samples,
        unit_names=unit_names,
        time_labels=time_labels,
        T=T,
        T0=T0,
        treated_unit_name=treated_unit_name,
    )
