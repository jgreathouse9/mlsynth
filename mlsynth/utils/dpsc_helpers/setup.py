"""Long-panel ingestion for DPSC (via :func:`mlsynth.utils.datautils.dataprep`)."""
from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import DPSCInputs


def prepare_dpsc_inputs(config) -> DPSCInputs:
    """Build :class:`DPSCInputs` from the config's long DataFrame.

    Uses the canonical ``dataprep`` contract: the single treated unit's outcome
    path ``y``, the donor matrix ``Y0`` (one column per donor), and the
    pre-period count ``T0``.
    """
    prep = dataprep(config.df, config.unitid, config.time, config.outcome, config.treat)
    if "cohorts" in prep:
        raise MlsynthDataError(
            "DPSC supports a single treated unit (one adoption date); the panel "
            "has multiple treated cohorts.")

    y_treated = np.asarray(prep["y"], dtype=float).flatten()
    donor_matrix = np.asarray(prep["donor_matrix"], dtype=float)
    T0 = int(prep["pre_periods"])
    # Defensive guards: dataprep already guarantees a pre-period, at least one
    # donor, and a balanced (finite) panel for a single-treated design.
    if T0 < 1:  # pragma: no cover - dataprep guarantees a pre-period
        raise MlsynthDataError("DPSC requires at least one pre-treatment period.")
    if donor_matrix.shape[1] < 1:  # pragma: no cover - dataprep guarantees donors
        raise MlsynthDataError("DPSC requires at least one donor unit.")
    if not np.all(np.isfinite(donor_matrix)) or not np.all(np.isfinite(y_treated)):  # pragma: no cover - dataprep balances the panel
        raise MlsynthDataError("DPSC: the outcome/donor matrix contains non-finite values.")

    return DPSCInputs(
        y_treated=y_treated,
        donor_matrix=donor_matrix,
        T0=T0,
        time_labels=np.asarray(prep["time_labels"]),
        treated_name=prep["treated_unit_name"],
        donor_names=tuple(prep["donor_names"]),
    )
