"""Data preparation helpers for SPCD.

Builds the wide ``(T, N)`` outcome matrix consumed by the SPCD iteration
through :func:`~mlsynth.utils.datautils.geoex_dataprep`. The orientation here
is the inverse of the paper's ``Y in R^{N x T}`` (Algorithm 1, page 7), so the
iteration matrix from Eq. (2),

    M = Y Y^T + alpha I + lambda 1 1^T,

is implemented as ``Y_pre.T @ Y_pre + alpha I + lambda 1 1^T`` further
downstream in ``formulation.py``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import balance, build_covariate_matrix, geoex_dataprep
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
    """Build the pre/post outcome matrices for SPCD through ``geoex_dataprep``.

    SPCD chooses which units to treat, so the panel has no treatment column
    and :func:`~mlsynth.utils.datautils.dataprep` cannot split it.
    :func:`~mlsynth.utils.datautils.geoex_dataprep` is the design-side sibling:
    it returns the balanced ``time x unit`` outcome matrix and refuses a
    duplicate or absent unit-period and a missing or non-finite outcome. With
    ``post_col`` it also refuses a column that is undefined for some period,
    that marks every period as post-treatment, or whose post periods are not a
    single block at the end of the panel; the plotter and the synthetic paths
    stack ``Y_pre`` over ``Y_post`` and read the result in time order.

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
        Each unit's per-covariate pre-period mean is taken
        (:func:`~mlsynth.utils.datautils.build_covariate_matrix`) and the
        resulting ``(N, P)`` matrix is z-scored across units. Time-invariant
        covariates collapse to their constant value.

    Returns
    -------
    SPCDInputs
        Wide pre/post matrices and label metadata.
    """

    if post_col is not None and post_col not in df.columns:
        raise MlsynthDataError(f"post_col '{post_col}' is not present in df.")

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
        raise MlsynthDataError("SPCD requires at least two pre-treatment periods.")
    if Y_pre.shape[1] < 2:
        raise MlsynthDataError("SPCD requires at least two units.")

    cov_matrix = None
    cov_names = None
    if covariates:
        cov_names = list(covariates)
        for c in cov_names:
            if c not in df.columns:
                raise MlsynthDataError(f"covariate '{c}' is not present in df.")
        # Each unit's mean over the first T0 periods, in the outcome matrix's
        # unit order. SPCD scales across units with ddof=0 below, so
        # build_covariate_matrix's own standardisation (ddof=1) is off.
        X, _, _, _ = build_covariate_matrix(
            df, unitid, time, cov_names, T0, list(unit_labels),
            normalize=False, keys=balance(df, unitid, time),
        )
        # build_covariate_matrix returns the transpose of its (P, N) rows. The
        # column means below are summed in memory order, so the (N, P) block is
        # laid out row-major, as the column_stack this replaced was.
        X = np.ascontiguousarray(X)
        for j, c in enumerate(cov_names):
            if np.isnan(X[:, j]).any():
                raise MlsynthDataError(
                    f"covariate '{c}' has missing values for some units in the "
                    f"pre-treatment period."
                )
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
