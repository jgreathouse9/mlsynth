"""Data preparation helpers for the SIV estimator.

The Synthetic IV procedure of Gulek and Vives-i-Bastida (2024)
requires a balanced ``(unit, time)`` panel with three series per
cell: the outcome ``Y``, the treatment intensity ``R``, and the
instrument ``Z``. The estimator targets shift-share-style designs
where ``R`` and ``Z`` are zero in the pre-period (e.g., the Syrian
refugee example), but the pipeline also handles cases where one or
both have pre-treatment variation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..fast_scm_helpers.structure import IndexSet


def prepare_siv_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    instrument: str,
    unitid: str,
    time: str,
    T0: Optional[int] = None,
    post_col: Optional[str] = None,
    T0_train: Optional[int] = None,
) -> SIVInputs:
    """Pivot a long balanced panel into the ``(J, T)`` SIV layout.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel.
    outcome, treat, instrument, unitid, time : str
        Column names.
    T0 : Optional[int]
        Number of pre-treatment periods. If ``None``, ``post_col``
        must be supplied.
    post_col : Optional[str]
        Optional 0/1 column identifying post-treatment periods. Used
        only if ``T0`` is None.
    T0_train : Optional[int]
        Optional end of the training block inside the pre-period
        (exclusive). The remaining pre-periods become the "blank"
        block used by the conformal inference and the ensemble CV.
        Defaults to ``floor(0.75 * T0)``.

    Raises
    ------
    MlsynthDataError
        If the panel is not balanced or has missing entries in the
        required columns.
    MlsynthConfigError
        If T0 / post_col are missing or inconsistent.
    """

    required = [outcome, treat, instrument, unitid, time]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MlsynthDataError(
            f"Missing required columns: {missing}"
        )

    ordered = df.sort_values([unitid, time]).copy()
    Y_wide = ordered.pivot(index=time, columns=unitid, values=outcome).sort_index()
    R_wide = ordered.pivot(index=time, columns=unitid, values=treat).sort_index()
    Z_wide = ordered.pivot(index=time, columns=unitid, values=instrument).sort_index()

    for name, wide in [("outcome", Y_wide), ("treat", R_wide), ("instrument", Z_wide)]:
        if wide.isna().any().any():
            raise MlsynthDataError(
                f"SIV requires a complete balanced panel; missing values "
                f"in {name} column after pivoting."
            )

    Y = Y_wide.to_numpy(dtype=float).T   # (J, T)
    R = R_wide.to_numpy(dtype=float).T
    Z = Z_wide.to_numpy(dtype=float).T
    J, T = Y.shape

    if J < 3:
        raise MlsynthDataError(
            f"SIV requires at least 3 units; panel has {J}."
        )

    # Resolve T0 from either argument.
    if post_col is not None:
        if post_col not in df.columns:
            raise MlsynthConfigError(f"post_col '{post_col}' is not in df.")
        post_by_time = (
            ordered[[time, post_col]]
            .drop_duplicates(subset=[time])
            .set_index(time)
            .reindex(Y_wide.index)[post_col]
            .astype(bool)
            .to_numpy()
        )
        if post_by_time.all() or not post_by_time.any():
            raise MlsynthConfigError(
                "post_col must mark both pre and post periods."
            )
        T0_resolved = int(np.argmax(post_by_time))
    elif T0 is not None:
        if not (1 <= T0 < T):
            raise MlsynthConfigError(
                f"T0 must be in [1, {T - 1}]; got {T0}."
            )
        T0_resolved = int(T0)
    else:
        raise MlsynthConfigError(
            "Either T0 or post_col must be supplied to SIVConfig."
        )

    if T0_train is None:
        T0_train_resolved: Optional[int] = max(2, int(0.75 * T0_resolved))
    else:
        if not (1 < T0_train < T0_resolved):
            raise MlsynthConfigError(
                f"T0_train must lie strictly inside (1, T0={T0_resolved}); "
                f"got {T0_train}."
            )
        T0_train_resolved = int(T0_train)

    has_pre_treatment = bool(np.any(R[:, :T0_resolved] != 0))
    has_pre_instrument = bool(np.any(Z[:, :T0_resolved] != 0))

    return SIVInputs(
        Y=Y,
        R=R,
        Z=Z,
        unit_index=IndexSet.from_labels(Y_wide.columns.to_list()),
        time_index=IndexSet.from_labels(Y_wide.index.to_list()),
        T0=T0_resolved,
        T0_train=T0_train_resolved,
        has_pre_treatment=has_pre_treatment,
        has_pre_instrument=has_pre_instrument,
    )


def build_design_matrix(
    inputs: SIVInputs,
    series: str = "default",
) -> np.ndarray:
    """Construct the (J, p) pre-period predictor matrix used by SC.

    For each unit ``i``, the SC weights solve
    ``min_w ||D_i - D_{-i}' w||_2^2`` with ``D_i`` the ``i``-th row of
    the returned matrix. ``series`` controls which pre-period series
    enter the design:

    * ``"default"`` — stack whichever of ``[Y_pre; R_pre; Z_pre]`` have
      non-zero variation in the pre-period. This is the paper's
      "Step 1" design matrix.
    * ``"outcome_only"`` — outcome lags only (``Y_pre``). Useful for
      the projected variant after the instrument-space projection has
      replaced ``Y`` with its instrument-space projection.
    """

    T0 = inputs.T0
    blocks: list[np.ndarray] = [inputs.Y[:, :T0]]

    if series == "default":
        if inputs.has_pre_treatment:
            blocks.append(inputs.R[:, :T0])
        if inputs.has_pre_instrument:
            blocks.append(inputs.Z[:, :T0])
    elif series != "outcome_only":
        raise MlsynthConfigError(
            f"Unknown design-matrix series mode {series!r}; expected "
            f"'default' or 'outcome_only'."
        )

    return np.concatenate(blocks, axis=1)
