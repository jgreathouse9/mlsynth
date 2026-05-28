"""Input preparation for MAREX: long panel -> design-ready arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..datautils import build_covariate_matrix


@dataclass(frozen=True)
class MAREXPanel:
    """Prepared MAREX inputs."""

    Y_full: pd.DataFrame          # units x time, indexed by unit label
    clusters: np.ndarray          # cluster label per unit, shape (N,)
    T0: int
    blank_periods: int
    covariates: Optional[np.ndarray] = None   # time-invariant predictors, (N, R)
    covariate_names: Tuple[str, ...] = ()     # column names aligned to ``covariates``


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
    """Pivot the long panel to ``units x time`` and forward the resolved
    ``T0`` / ``blank_periods``.

    The MAREX config validator is the single source of truth for resolving
    ``T0`` (from either an explicit scalar or a ``post_col`` 0/1 column) and
    the default 30%-of-pre-tail blank window — by the time this helper runs
    both are concrete integers. ``covariates`` columns are aggregated to a
    per-unit pre-period mean via
    :func:`mlsynth.utils.datautils.build_covariate_matrix` and returned as an
    ``(N, R)`` matrix aligned to the unit order. The matrix is left
    un-normalised here so MAREX's existing ``standardize=True`` flag (applied
    to the combined ``[Y_fit; covariate_weight * Z]`` predictor matrix in
    :mod:`marex_helpers.optimization`) keeps its previous behaviour.
    """
    unit_labels = df[unitid].unique()
    if cluster is not None:
        clusters = (df.drop_duplicates(subset=[unitid]).set_index(unitid)[cluster]
                    .reindex(unit_labels).to_numpy())
    else:
        clusters = np.zeros(len(unit_labels), dtype=int)

    Y_full = df.pivot(index=unitid, columns=time, values=outcome).reindex(unit_labels)
    T_total = df[time].nunique()
    T0_eff = T0 if T0 is not None else T_total - 1

    cov: Optional[np.ndarray] = None
    if covariates:
        # Pre-period mean aggregation via the shared dataprep helper; passing
        # ``normalize=False`` preserves the raw scale so the downstream
        # ``covariate_weight`` * ``standardize`` pipeline keeps its semantics.
        cov, _names, _means, _scales = build_covariate_matrix(
            df=df,
            unit_id_column_name=unitid,
            time_period_column_name=time,
            covariates=covariates,
            pre_periods=T0_eff,
            unit_order=list(unit_labels),
            aggregation="pre_mean",
            normalize=False,
        )

    blanks = int(blank_periods or 0)
    if blanks < 0 or blanks >= T0_eff:
        raise ValueError(
            f"blank_periods must be 0 <= blank_periods < T0 (T0={T0_eff}, got {blanks})"
        )

    return MAREXPanel(Y_full=Y_full, clusters=clusters, T0=T0_eff,
                      blank_periods=blanks, covariates=cov,
                      covariate_names=tuple(covariates) if covariates else ())
