"""Panel ingestion for DTWSC -- a thin wrapper over :func:`dataprep`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import DTWSCInputs


def prepare_dtwsc_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> DTWSCInputs:
    """Pivot a long panel into :class:`DTWSCInputs` via :func:`dataprep`.

    DTWSC is a single-treated-unit, block-treatment method: it learns one warp
    per donor against one treated path.
    """
    prepped = dataprep(
        df=df,
        unit_id_column_name=unitid,
        time_period_column_name=time,
        outcome_column_name=outcome,
        treatment_indicator_column_name=treat,
    )
    if "y" not in prepped or "donor_matrix" not in prepped:
        raise MlsynthDataError(
            "DTWSC requires a single treated unit with a common treatment "
            "date; the panel resolved to staggered cohorts instead."
        )

    y = np.asarray(prepped["y"], dtype=float).ravel()
    donors = np.asarray(prepped["donor_matrix"], dtype=float)
    if donors.ndim == 1:  # pragma: no cover - dataprep returns (T, J) even for
        # a single donor; this normalises defensively for a future caller.
        donors = donors[:, None]
    n_pre = int(prepped["pre_periods"])

    if donors.shape[1] < 1:  # pragma: no cover - dataprep raises on an empty
        # donor pool before returning, unless allow_no_donors is set, which
        # DTWSC never does; kept so the requirement is stated at this seam.
        raise MlsynthDataError("DTWSC needs at least one donor unit.")
    if n_pre < 3:
        raise MlsynthDataError(
            f"DTWSC needs at least 3 pre-treatment periods to learn a warp; "
            f"the panel has {n_pre}."
        )
    if int(prepped["post_periods"]) < 1:  # pragma: no cover - a panel with no
        # treated period fails dataprep's own treatment-detection first; kept
        # so the requirement is visible where DTWSC depends on it.
        raise MlsynthDataError("DTWSC needs at least one post-treatment period.")
    if not np.isfinite(y).all() or not np.isfinite(donors).all():
        raise MlsynthDataError("DTWSC: outcome contains NaN or infinite values.")

    return DTWSCInputs(
        y=y,
        donor_matrix=donors,
        donor_names=list(prepped["donor_names"]),
        time_labels=np.asarray(prepped["time_labels"]),
        n_pre=n_pre,
        treated_name=prepped.get("treated_unit_name"),
    )
