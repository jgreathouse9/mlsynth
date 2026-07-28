"""Data ingestion for ESC: long panel -> :class:`ESCInputs` via ``dataprep``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ...utils.datautils import dataprep
from .structures import ESCInputs


def prepare_esc_inputs(df: pd.DataFrame, *, unitid: str, time: str, outcome: str,
                       treat: str) -> ESCInputs:
    """Build the NumPy-only :class:`ESCInputs` for a single-treated-unit panel."""
    prep = dataprep(
        df=df, unit_id_column_name=unitid, time_period_column_name=time,
        outcome_column_name=outcome, treatment_indicator_column_name=treat,
    )
    if "cohorts" in prep:
        raise MlsynthDataError(
            "ESC handles a single treated unit; the panel has multiple treated "
            "cohorts (staggered adoption)."
        )
    if "y" not in prep or "donor_matrix" not in prep:  # pragma: no cover - dataprep always returns these for a single treated unit
        raise MlsynthDataError("ESC could not prepare the data (no single-treated structure).")

    y = np.asarray(prep["y"], dtype=float).ravel()
    Y0 = np.asarray(prep["donor_matrix"], dtype=float)
    T0 = int(prep["pre_periods"])
    if Y0.ndim != 2 or Y0.shape[1] < 1:  # pragma: no cover - dataprep raises on an empty donor pool upstream
        raise MlsynthDataError("ESC needs at least one donor unit.")
    if T0 < 2:
        raise MlsynthDataError(
            f"ESC needs at least two pre-treatment periods; got T0={T0}."
        )
    if T0 >= y.shape[0]:  # pragma: no cover - a valid treated unit always leaves >=1 post period
        raise MlsynthDataError("ESC found no post-treatment periods.")

    return ESCInputs(
        treated_name=prep.get("treated_unit_name", "treated"),
        donor_names=[str(x) for x in prep["donor_names"]],
        y=y, donor_matrix=Y0, T0=T0,
        time_labels=np.asarray(prep["time_labels"]),
        metadata={"outcome": outcome},
    )
