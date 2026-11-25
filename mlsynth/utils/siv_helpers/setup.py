"""Data preparation helpers for the SIV estimator.

The Synthetic IV procedure of Gulek and Vives-i-Bastida (2024)
requires a balanced ``(unit, time)`` panel with three series per
cell: the outcome ``Y``, the treatment intensity ``R``, and the
instrument ``Z``. The estimator targets shift-share-style designs
where ``R`` and ``Z`` are zero in the pre-period (e.g., the Syrian
refugee example), but the pipeline also handles cases where one or
both have pre-treatment variation.

``R`` and ``Z`` are continuous, which
:func:`~mlsynth.utils.datautils.dataprep` refuses as a non-binary treatment,
so the panel is read through :func:`~mlsynth.utils.datautils.geoex_dataprep`,
the treatment-agnostic reader: once for each of the three columns, each read
as that call's outcome.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import geoex_dataprep
from ..fast_scm_helpers.structure import IndexSet
from .structures import SIVInputs


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
    """Read a long balanced panel into the ``(J, T)`` SIV layout.

    ``geoex_dataprep`` refuses a duplicate or absent unit-period, a row with
    a missing unit or period, and a missing or non-finite outcome. ``R`` and
    ``Z`` are read the same way, and a missing or non-finite value of either
    is refused under the column's name. SIV's own checks (three units, ``T0``
    or ``post_col``, ``T0_train``) stay on top.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel. It is not modified.
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
        If the panel is not balanced, has a duplicate unit-period or a row
        without a unit or period, or has a missing or non-finite value in
        the outcome, treatment or instrument column.
    MlsynthConfigError
        If T0 / post_col are missing or inconsistent.
    """

    required = [outcome, treat, instrument, unitid, time]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MlsynthDataError(
            f"Missing required columns: {missing}"
        )

    # geoex_dataprep runs balance() first, so a duplicate or absent
    # unit-period and a missing key are refused before the matrix is built,
    # and a missing or non-finite outcome after it.
    panel = geoex_dataprep(df, unitid, time, outcome)
    Y_wide = panel["Ywide"]                              # (T, J), both axes sorted
    # Unit by period and C-ordered, the layout the old setup gave on pandas
    # 2.3 (CI): its sort_index() copy of the pivot was column-major there.
    # On pandas 3 that copy is row-major, so the old layout depended on the
    # version; this one does not.
    Y = np.ascontiguousarray(panel["Y"].T)               # (J, T)
    matrices = {}
    for role, col in (("treatment", treat), ("instrument", instrument)):
        # The same panel, so the same sorted axes as the outcome's matrix.
        try:
            matrices[role] = np.ascontiguousarray(geoex_dataprep(df, unitid, time, col)["Y"].T)
        except MlsynthDataError as exc:
            raise MlsynthDataError(
                f"SIV's {role} column {col!r} is incomplete; SIV needs it "
                f"observed and finite in every unit-period. {exc}"
            ) from exc
    R = matrices["treatment"]
    Z = matrices["instrument"]
    J, T = Y.shape

    if J < 3:
        raise MlsynthDataError(
            f"SIV requires at least 3 units; panel has {J}."
        )

    # Resolve T0 from either argument.
    if post_col is not None:
        if post_col not in df.columns:
            raise MlsynthConfigError(f"post_col '{post_col}' is not in df.")
        # Read per period from the first unit in sorted order, as before.
        post_by_time = (
            df.sort_values([unitid, time])[[time, post_col]]
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
