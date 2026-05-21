"""Data preparation for BVS-SS.

Wraps :func:`mlsynth.utils.datautils.dataprep` and demeans the
pre-treatment block before passing it to the Gibbs sampler. The
post-treatment donor block is demeaned using the *pre-treatment* column
means so that out-of-sample counterfactual paths are computed in the
same centered coordinate system as the sampler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import BVSSInputs


def prepare_bvss_inputs(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    treat: str,
) -> BVSSInputs:
    """Pivot panel data and demean to match the BVS-SS sampler's contract.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel data.
    outcome, unitid, time, treat : str
        Column names identifying the outcome, units, time periods, and the
        binary treatment indicator.

    Returns
    -------
    BVSSInputs
    """

    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "BVSS currently supports a single treated unit; the input "
            "panel appears to contain multiple treatment cohorts."
        )

    y_target = np.asarray(prepared["y"], dtype=float)
    donor_matrix_tn = np.asarray(prepared["donor_matrix"], dtype=float)
    if donor_matrix_tn.ndim != 2:
        raise MlsynthDataError(
            "BVSS requires a 2D donor matrix; got shape "
            f"{donor_matrix_tn.shape}."
        )

    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    n_donors = donor_matrix_tn.shape[1]

    if n_donors < 1:
        raise MlsynthDataError("BVSS requires at least one donor unit.")
    if T0 < 2:
        raise MlsynthDataError(
            "BVSS requires at least two pre-treatment periods."
        )

    X_pre = donor_matrix_tn[:T0, :]
    Y_pre = y_target[:T0]

    mean_X = X_pre.mean(axis=0)
    mean_Y = float(Y_pre.mean())
    X_pre_demean = X_pre - mean_X
    Y_pre_demean = Y_pre - mean_Y

    if T0 < T:
        X_post = donor_matrix_tn[T0:, :]
        X_post_demean = X_post - mean_X
    else:
        X_post_demean = None

    Gram = X_pre_demean.T @ X_pre_demean

    return BVSSInputs(
        Y_pre_demean=Y_pre_demean,
        X_pre_demean=X_pre_demean,
        X_post_demean=X_post_demean,
        Gram=Gram,
        mean_Y=mean_Y,
        mean_X=mean_X,
        T0=T0,
        T=T,
        N=n_donors,
        treated_unit_name=prepared["treated_unit_name"],
        donor_names=list(prepared["donor_names"]),
        time_labels=np.asarray(prepared["time_labels"]),
        y_target=y_target,
    )
