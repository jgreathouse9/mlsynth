"""Panel preparation and spillover-structure construction for SPILLSYNTH.

Two responsibilities:

1. :func:`prepare_spillsynth_inputs` turns the long-format DataFrame into
   the ``(N, T)`` panel matrix that the Cao-Dowd estimator consumes,
   with row 0 the treated unit, rows ``1 .. p`` the user-declared
   affected units, and the rest the clean controls.
2. :func:`build_A_example3` constructs the spillover-structure matrix
   ``A`` of Example 3 from Cao & Dowd (2023, Section 2.2): a basis
   vector for the treated unit and one basis vector per affected
   control unit (i.e.\ each affected unit gets its own free spillover
   coefficient).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import SpillSynthInputs


def _resolve_treated_label(df: pd.DataFrame, unitid: str, treat: str) -> Any:
    """Identify the unique treated unit in the long-format panel."""
    treated_rows = df.loc[df[treat] != 0, unitid].unique()
    if len(treated_rows) == 0:
        raise MlsynthDataError(
            "SPILLSYNTH: no treated rows found (every row has treat == 0)."
        )
    if len(treated_rows) > 1:
        raise MlsynthDataError(
            "SPILLSYNTH currently supports a single treated unit. "
            f"Found {len(treated_rows)}: {list(treated_rows)}."
        )
    return treated_rows[0]


def _resolve_intervention_time(
    df: pd.DataFrame, unitid: str, time: str, treat: str, treated_label: Any
) -> Any:
    """First time period at which the treated unit has treat != 0."""
    treated_df = df.loc[df[unitid] == treated_label, [time, treat]]
    on = treated_df.loc[treated_df[treat] != 0, time]
    if on.empty:
        raise MlsynthDataError(
            f"Treated unit {treated_label!r} never has treat != 0."
        )
    return on.min()


def build_A_example3(N: int, p: int) -> np.ndarray:
    """Spillover-structure matrix from Cao-Dowd Example 3.

    Returns an ``(N, 1 + p)`` matrix whose columns are unit basis
    vectors for the treated unit (row 0) and each of the ``p``
    potentially-affected control units (rows 1..p). All other rows are
    zero, so clean controls have zero spillover.

    Parameters
    ----------
    N : int
        Total number of units.
    p : int
        Number of potentially-affected control units.

    Returns
    -------
    np.ndarray
        Shape ``(N, 1 + p)``.
    """
    if p < 0 or N < 1 + p:
        raise MlsynthDataError(
            f"SPILLSYNTH: cannot build A with N={N}, p={p}."
        )
    A = np.zeros((N, 1 + p))
    A[0, 0] = 1.0
    for k in range(p):
        A[1 + k, 1 + k] = 1.0
    return A


def prepare_spillsynth_inputs(
    df: pd.DataFrame,
    *,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    affected_units: Optional[Sequence[Any]] = None,
) -> SpillSynthInputs:
    """Reshape a long-format panel into the inputs SPILLSYNTH expects.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel.
    outcome, treat, unitid, time : str
        Column names.
    affected_units : sequence, optional
        Labels of control units (must be in ``df[unitid]``) that the
        researcher believes are potentially exposed to spillover from
        the treated unit. The treated unit itself must NOT appear in
        this list. If ``None`` (the default), no affected units are
        declared and the estimator reduces to vanilla demeaned SCM.

    Returns
    -------
    SpillSynthInputs
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(
                f"SPILLSYNTH: required column {col!r} not in df.columns."
            )

    df = df[[unitid, time, outcome, treat]].copy()
    df = df.dropna(subset=[outcome])
    if df.empty:
        raise MlsynthDataError("SPILLSYNTH: panel is empty after dropping NaN outcomes.")

    treated_label = _resolve_treated_label(df, unitid, treat)
    intervention = _resolve_intervention_time(df, unitid, time, treat, treated_label)

    affected = tuple(affected_units or ())
    if treated_label in affected:
        raise MlsynthDataError(
            f"SPILLSYNTH: the treated unit {treated_label!r} cannot also "
            "appear in affected_units."
        )
    if len(set(affected)) != len(affected):
        raise MlsynthDataError("SPILLSYNTH: affected_units contains duplicates.")
    units_in_df = set(df[unitid].unique())
    missing = [u for u in affected if u not in units_in_df]
    if missing:
        raise MlsynthDataError(
            f"SPILLSYNTH: affected_units {missing} not present in panel."
        )

    # Order: treated first, affected next, clean controls after (sorted).
    all_controls = sorted([u for u in units_in_df if u != treated_label],
                          key=lambda x: str(x))
    clean = tuple(u for u in all_controls if u not in affected)
    units = [treated_label, *affected, *clean]

    # Pivot to wide (T x N), then transpose to (N, T).
    wide = (df.pivot(index=time, columns=unitid, values=outcome)
              .reindex(columns=units)
              .sort_index())
    if wide.isna().any().any():
        bad_cells = wide.isna().sum().sum()
        raise MlsynthDataError(
            f"SPILLSYNTH: panel has {int(bad_cells)} missing cells after pivot."
        )

    time_labels = wide.index.to_numpy()
    pre_mask = time_labels < intervention
    if not pre_mask.any():
        raise MlsynthDataError(
            "SPILLSYNTH: no pre-treatment periods (intervention at first period)."
        )
    if pre_mask.all():
        raise MlsynthDataError(
            "SPILLSYNTH: no post-treatment periods."
        )

    Y = wide.to_numpy(dtype=float).T                # (N, T)
    N = Y.shape[0]
    T = Y.shape[1]
    T0 = int(pre_mask.sum())
    T1 = T - T0
    p = len(affected)
    Y_pre = Y[:, :T0]
    Y_post = Y[:, T0:]

    if T0 < 2:
        raise MlsynthDataError(
            f"SPILLSYNTH: need T0 >= 2 pre-periods to estimate SCM weights "
            f"(got T0={T0})."
        )

    A = build_A_example3(N, p)
    return SpillSynthInputs(
        Y=Y, Y_pre=Y_pre, Y_post=Y_post, A=A,
        treated_label=treated_label,
        affected_labels=affected,
        clean_labels=clean,
        time_labels=time_labels,
        pre_time=time_labels[:T0],
        post_time=time_labels[T0:],
        N=N, T=T, T0=T0, T1=T1, p=p,
    )
