"""Data preparation for the Synthetic Historical Control (SHC) estimator.

Wraps ``datautils.dataprep`` (in single-series mode, since SHC needs no
cross-sectional donors), builds a time :class:`IndexSet`, and validates
that the pre-treatment window is long enough to form at least one
historical block: ``T0 > m + n - 1`` (Section 2.2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from ..helperutils import IndexSet
from .structures import SHCInputs


def prepare_shc_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    m: int,
) -> SHCInputs:
    """Pivot a single-unit panel into :class:`SHCInputs`.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel for one treated unit (donors optional and
        ignored).
    outcome, treat, unitid, time : str
        Column names.
    m : int
        Pre-intervention block length. Must satisfy ``T0 > m + n - 1`` so
        at least one historical block exists.

    Returns
    -------
    SHCInputs
    """
    prepared = dataprep(df, unitid, time, outcome, treat, allow_no_donors=True)

    T0 = prepared.get("pre_periods")
    n_post = prepared.get("post_periods")
    if T0 is None or n_post is None:
        raise MlsynthDataError(
            "dataprep did not return pre/post period counts; ensure the "
            "treated unit and treatment timing are identified."
        )
    T0 = int(T0)
    n_post = int(n_post)

    if m <= 0:
        raise MlsynthDataError(f"SHC requires m > 0; got m={m}.")
    if n_post < 1:
        raise MlsynthDataError("SHC needs at least one post-treatment period.")

    N = T0 - m - n_post + 1
    if N <= 0:
        raise MlsynthDataError(
            f"Insufficient pre-treatment data for the donor pool: "
            f"T0 - m - n + 1 = {N} <= 0 (T0={T0}, m={m}, n={n_post}). "
            f"Reduce m or use a longer pre-period."
        )

    y = np.asarray(prepared["y"], dtype=float).ravel()
    time_labels = np.asarray(prepared["time_labels"])

    return SHCInputs(
        time_index=IndexSet.from_labels(time_labels),
        y=y,
        T0=T0,
        m=int(m),
        treated_label=prepared.get("treated_unit_name"),
        metadata={
            "Ywide": prepared.get("Ywide"),
            "n_historical_blocks": int(N),
        },
    )
