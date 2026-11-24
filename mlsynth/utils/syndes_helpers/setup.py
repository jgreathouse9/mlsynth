"""Data preparation helpers for SYNDES."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import geoex_dataprep
from ..fast_scm_helpers.structure import IndexSet
from .structures import SYNDESInputs


def prepare_syndes_inputs(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    T0: Optional[int] = None,
    post_col: Optional[str] = None,
) -> SYNDESInputs:
    """Build the pre/post outcome matrices for SYNDES through ``geoex_dataprep``.

    SYNDES chooses which units to treat, so the panel has no treatment column
    and :func:`~mlsynth.utils.datautils.dataprep` cannot split it.
    :func:`~mlsynth.utils.datautils.geoex_dataprep` is the design-side sibling:
    it returns the balanced ``time x unit`` outcome matrix and refuses a
    duplicate or absent unit-period and a missing or non-finite outcome. With
    ``post_col`` it also refuses a column that is undefined for some period,
    that marks every period as post-treatment, or whose post periods are not a
    single block at the end of the panel; the permutation inference and the
    plotter stack ``Y_pre`` over ``Y_post`` and read the result in time order.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel data.
    outcome, unitid, time : str
        Column names identifying the outcome, units, and time periods.
    T0 : Optional[int]
        Number of pre-treatment periods when ``post_col`` is not supplied.
    post_col : Optional[str]
        Optional 0/1 or boolean column identifying post-treatment periods.

    Returns
    -------
    SYNDESInputs
        Wide pre/post matrices and label metadata.
    """

    panel = geoex_dataprep(df, unitid, time, outcome)
    Ywide = panel["Ywide"]
    time_labels = np.asarray(Ywide.index.to_list())
    unit_labels = np.asarray(Ywide.columns.to_list())
    Y_full = panel["Y"]
    n_periods = Y_full.shape[0]

    if post_col is not None:
        # geoex_dataprep validates post_col and keeps the pre-period slice; its
        # length is where the post block starts in the full matrix.
        T0 = int(geoex_dataprep(df, unitid, time, outcome, post_col=post_col)["pre_periods"])
    else:
        if T0 is None:
            T0 = n_periods
        if T0 <= 0 or T0 > n_periods:
            raise MlsynthConfigError(
                f"T0 must be between 1 and the number of periods ({n_periods})."
            )

    Y_pre = Y_full[:T0, :]
    Y_post = Y_full[T0:, :] if T0 < n_periods else None
    pre_labels = time_labels[:T0]
    post_labels = time_labels[T0:] if T0 < n_periods else None

    if Y_pre.shape[0] < 2:
        raise MlsynthDataError("SYNDES requires at least two pre-treatment periods.")
    if Y_pre.shape[1] < 2:
        raise MlsynthDataError("SYNDES requires at least two units.")

    return SYNDESInputs(
        Y_pre=Y_pre,
        Y_post=Y_post,
        unit_index=IndexSet.from_labels(unit_labels),
        time_index=IndexSet.from_labels(time_labels),
        pre_time_index=IndexSet.from_labels(pre_labels),
        post_time_index=IndexSet.from_labels(post_labels) if post_labels is not None else None,
        outcome=outcome)
