"""Input preparation for MAREX: long panel -> design-ready arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MAREXPanel:
    """Prepared MAREX inputs."""

    Y_full: pd.DataFrame          # units x time, indexed by unit label
    clusters: np.ndarray          # cluster label per unit, shape (N,)
    T0: int
    blank_periods: int
    covariates: Optional[np.ndarray] = None   # time-invariant predictors, (N, R)


def prepare_marex_panel(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    cluster: Optional[str],
    T0: Optional[int],
    inference: bool,
    blank_periods: int,
    T_post: Optional[int],
    covariates: Optional[List[str]] = None,
) -> MAREXPanel:
    """Pivot the long panel to ``units x time`` and resolve T0 / blank periods.

    ``T0`` defaults to ``T - 1``. When ``inference`` is on, ``blank_periods``
    defaults to the requested post-window length (``T_post``) or ``T - T0``.
    ``covariates`` (time-invariant columns) are extracted as an ``(N, R)`` matrix
    aligned to the unit order and matched on alongside the pre-period outcomes.
    """
    unit_labels = df[unitid].unique()
    if cluster is not None:
        clusters = (df.drop_duplicates(subset=[unitid]).set_index(unitid)[cluster]
                    .reindex(unit_labels).to_numpy())
    else:
        clusters = np.zeros(len(unit_labels), dtype=int)

    cov = None
    if covariates:
        per_unit = df.drop_duplicates(subset=[unitid]).set_index(unitid)
        cov = per_unit.reindex(unit_labels)[covariates].to_numpy(dtype=float)

    Y_full = df.pivot(index=unitid, columns=time, values=outcome).reindex(unit_labels)
    T_total = df[time].nunique()
    T0_eff = T0 if T0 is not None else T_total - 1

    if inference:
        blanks = blank_periods if blank_periods else (T_post if T_post else T_total - T0_eff)
        if blanks < 0 or blanks >= T0_eff:
            raise ValueError(
                f"blank_periods must be 0 <= blank_periods < T0 (T0={T0_eff}, got {blanks})"
            )
    else:
        blanks = blank_periods

    return MAREXPanel(Y_full=Y_full, clusters=clusters, T0=T0_eff,
                      blank_periods=blanks, covariates=cov)
