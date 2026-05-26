"""Data preparation helpers for SPCD.

Pivots long panel data into the wide ``(T, N)`` matrix consumed by the
SPCD iteration. The orientation here is the inverse of the paper's
``Y in R^{N x T}`` (Algorithm 1, page 7), so the iteration matrix from
Eq. (2),

    M = Y Y^T + alpha I + lambda 1 1^T,

is implemented as ``Y_pre.T @ Y_pre + alpha I + lambda 1 1^T`` further
downstream in ``formulation.py``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..fast_scm_helpers.structure import IndexSet
from .structures import SPCDInputs


def prepare_spcd_inputs(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    T0: Optional[int] = None,
    post_col: Optional[str] = None,
    covariates: Optional[List[str]] = None,
) -> SPCDInputs:
    """Pivot long panel data and split it into pre/post matrices for SPCD.

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
    covariates : Optional[list of str]
        Optional covariate columns to balance on alongside the outcomes.
        Each unit's per-covariate **pre-period mean** is taken and the
        resulting ``(N, P)`` matrix is z-scored across units. Time-invariant
        covariates collapse to their constant value.

    Returns
    -------
    SPCDInputs
        Wide pre/post matrices and label metadata.
    """

    if post_col is not None and post_col not in df.columns:
        raise MlsynthDataError(f"post_col '{post_col}' is not present in df.")

    ordered_df = df.sort_values([time, unitid]).copy()
    Ywide = ordered_df.pivot(index=time, columns=unitid, values=outcome).sort_index()

    if Ywide.isna().any().any():
        raise MlsynthDataError(
            "SPCD requires a complete balanced outcome matrix after pivoting."
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
        raise MlsynthDataError("SPCD requires at least two pre-treatment periods.")
    if Y_pre.shape[1] < 2:
        raise MlsynthDataError("SPCD requires at least two units.")

    cov_matrix = None
    cov_names = None
    if covariates:
        cov_names = list(covariates)
        pre_label_set = set(np.asarray(pre_labels).tolist())
        columns = []
        for c in cov_names:
            if c not in ordered_df.columns:
                raise MlsynthDataError(f"covariate '{c}' is not present in df.")
            cwide = ordered_df.pivot(index=time, columns=unitid, values=c).sort_index()
            pre_rows = cwide.index.isin(pre_label_set)
            per_unit = cwide.loc[pre_rows, :].mean(axis=0)  # per-unit pre-period mean
            col = per_unit.reindex(unit_labels).to_numpy(dtype=float)
            if np.isnan(col).any():
                raise MlsynthDataError(
                    f"covariate '{c}' has missing values for some units in the "
                    f"pre-treatment period."
                )
            columns.append(col)
        X = np.column_stack(columns)  # (N, P)
        # z-score each covariate across units so they share a common scale.
        X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-12)
        cov_matrix = X

    return SPCDInputs(
        Y_pre=Y_pre,
        Y_post=Y_post,
        unit_index=IndexSet.from_labels(unit_labels),
        time_index=IndexSet.from_labels(time_labels),
        pre_time_index=IndexSet.from_labels(pre_labels),
        post_time_index=IndexSet.from_labels(post_labels) if post_labels is not None else None,
        outcome=outcome,
        covariates=cov_matrix,
        covariate_names=cov_names,
    )
