"""Data preparation for the CSCM estimator."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ...exceptions import MlsynthDataError, MlsynthEstimationError
from ..datautils import balance, dataprep


def prepare_cscm_inputs(df, outcome: str, treat: str, unitid: str, time: str
                        ) -> Dict[str, Any]:
    """Balance and pivot the panel into CSCM's single-treated inputs.

    Returns ``y`` (treated outcome), ``Y0`` (T x J donors), ``donor_names``,
    ``pre_periods`` (T0), ``post_periods`` (T1), and ``time_labels``.
    """
    try:
        balance(df, unitid, time)
    except Exception as e:  # noqa: BLE001 - defensive re-wrap of balancing errors
        raise MlsynthDataError(f"Error balancing panel data: {e}") from e  # pragma: no cover

    try:
        prep = dataprep(df, unitid, time, outcome, treat)
    except MlsynthDataError:
        raise
    except Exception as e:  # noqa: BLE001 - defensive re-wrap of dataprep errors
        raise MlsynthDataError(f"Error preparing data matrices: {e}") from e  # pragma: no cover

    T0 = prep.get("pre_periods")
    T1 = prep.get("post_periods")
    if T0 is None or T0 < 2:
        raise MlsynthEstimationError("CSCM needs at least 2 pre-treatment periods.")
    if T1 is None or T1 < 1:  # pragma: no cover - dataprep requires a treated post-period
        raise MlsynthEstimationError("CSCM needs at least 1 post-treatment period.")

    Y0 = np.asarray(prep["donor_matrix"], dtype=float)
    if Y0.shape[1] < 1:  # pragma: no cover - dataprep requires >=1 donor
        raise MlsynthEstimationError("CSCM needs at least one donor unit.")

    return {
        "y": np.asarray(prep["y"], dtype=float).ravel(),
        "Y0": Y0,
        "donor_names": list(prep["donor_names"]),
        "pre_periods": int(T0),
        "post_periods": int(T1),
        "time_labels": prep.get("time_labels"),
    }
