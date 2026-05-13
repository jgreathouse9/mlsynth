"""Data preparation helpers for SCDI."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..fast_scm_helpers.structure import IndexSet
from .structures import SCDIInputs


def prepare_scdi_inputs(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    T0: Optional[int] = None,
    post_col: Optional[str] = None,
) -> SCDIInputs:
    """Pivot long panel data and split it into pre/post matrices for SCDI.

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
    SCDIInputs
        Wide pre/post matrices and label metadata.
    """

    if post_col is not None and post_col not in df.columns:
        raise MlsynthDataError(f"post_col '{post_col}' is not present in df.")

    ordered_df = df.sort_values([time, unitid]).copy()
    Ywide = ordered_df.pivot(index=time, columns=unitid, values=outcome).sort_index()

    if Ywide.isna().any().any():
        raise MlsynthDataError(
            "SCDI requires a complete balanced outcome matrix after pivoting."
        )

    time_labels = np.asarray(Ywide.index.to_list())
    unit_labels = np.asarray(Ywide.columns.to_list())
    Y_full = Ywide.to_numpy(dtype=float)

    if post_col is not None:
        post_by_time = (
            ordered_df[[time, post_col]]
            .drop_duplicates(subset=[time])
            .set_index(time)
            .reindex(time_labels)[post_col]
        )
        if post_by_time.isna().any():
            raise MlsynthDataError(
                "post_col must be defined for every time period in the panel."
            )
        post_mask = post_by_time.astype(bool).to_numpy()
        if post_mask.all():
            raise MlsynthConfigError("post_col marks every period as post-treatment.")
        pre_mask = ~post_mask
        Y_pre = Y_full[pre_mask, :]
        Y_post = Y_full[post_mask, :] if post_mask.any() else None
        pre_labels = time_labels[pre_mask]
        post_labels = time_labels[post_mask] if post_mask.any() else None
    else:
        if T0 is None:
            T0 = Y_full.shape[0]
        if T0 <= 0 or T0 > Y_full.shape[0]:
            raise MlsynthConfigError(
                f"T0 must be between 1 and the number of periods ({Y_full.shape[0]})."
            )
        Y_pre = Y_full[:T0, :]
        Y_post = Y_full[T0:, :] if T0 < Y_full.shape[0] else None
        pre_labels = time_labels[:T0]
        post_labels = time_labels[T0:] if T0 < Y_full.shape[0] else None

    if Y_pre.shape[0] < 2:
        raise MlsynthDataError("SCDI requires at least two pre-treatment periods.")
    if Y_pre.shape[1] < 2:
        raise MlsynthDataError("SCDI requires at least two units.")

    return SCDIInputs(
        Y_pre=Y_pre,
        Y_post=Y_post,
        unit_index=IndexSet.from_labels(unit_labels),
        time_index=IndexSet.from_labels(time_labels),
        pre_time_index=IndexSet.from_labels(pre_labels),
        post_time_index=IndexSet.from_labels(post_labels) if post_labels is not None else None,
        outcome=outcome)
