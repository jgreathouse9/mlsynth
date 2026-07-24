"""Data boundary for DROSC: long DataFrame -> :class:`DROSCInputs` via
:func:`mlsynth.utils.datautils.dataprep`."""
from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import DROSCInputs


def prepare_drosc_inputs(config) -> DROSCInputs:
    """Ingest ``config.df`` through ``dataprep`` and return the canonical DROSC
    inputs. Raises :class:`MlsynthDataError` for a multi-treated / staggered panel
    or an incomplete outcome matrix."""
    prep = dataprep(
        df=config.df,
        unit_id_column_name=config.unitid,
        time_period_column_name=config.time,
        outcome_column_name=config.outcome,
        treatment_indicator_column_name=config.treat,
    )
    if "cohorts" in prep or "y" not in prep or "donor_matrix" not in prep:
        raise MlsynthDataError(
            "DROSC supports a single treated unit; dataprep returned a "
            "staggered/multi-treated layout."
        )
    y = np.asarray(prep["y"], dtype=float).ravel()
    donor_matrix = np.asarray(prep["donor_matrix"], dtype=float)
    pre = int(prep["pre_periods"])
    if donor_matrix.shape[1] < 1:
        raise MlsynthDataError("DROSC requires at least one donor unit.")
    if pre < 2:
        raise MlsynthDataError(
            "DROSC requires at least two pre-treatment periods to form the "
            "pre-treatment moment band."
        )
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(donor_matrix)):
        raise MlsynthDataError("DROSC requires a complete (finite) outcome panel.")
    return DROSCInputs(
        y=y,
        donor_matrix=donor_matrix,
        pre_periods=pre,
        time_labels=np.asarray(prep["time_labels"]),
        donor_names=list(prep["donor_names"]),
        treated_name=prep.get("treated_unit_name"),
    )
