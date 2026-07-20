"""Panel ingestion for MEDSC: outcome via ``dataprep`` plus the mediator panel.

Builds the two donor pools (total / direct), aligns the mediator to the same
units and periods as the outcome, and assembles the optional covariate blocks.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep


def _resolve_pool(
    requested: Optional[List[Any]], donor_names: List[Any], label: str
) -> List[Any]:
    """Validate a requested donor pool against the available donors."""
    if requested is None:
        return list(donor_names)
    donor_set = set(donor_names)
    unknown = [u for u in requested if u not in donor_set]
    if unknown:
        raise MlsynthDataError(
            f"{label} contains unit(s) {unknown} that are not donors "
            f"(non-treated units) in df.")
    # Preserve the caller's order, dropping duplicates.
    seen: set = set()
    pool = [u for u in requested if not (u in seen or seen.add(u))]
    return pool


def _covariate_block(
    df: pd.DataFrame, unitid: str, time: str, covariates: List[str],
    pre_labels: np.ndarray, units: List[Any], treated_name: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-treatment covariate means: donor block ``(L, J)`` and treated ``(L,)``.

    Units missing a covariate value get the pool mean for that covariate (a
    benign fill that leaves fully-observed panels unchanged).
    """
    pre = df[df[time].isin(list(pre_labels))]
    means = pre.groupby(unitid)[covariates].mean()
    donor = means.reindex(units)
    donor = donor.fillna(donor.mean())
    treated = means.reindex([treated_name])
    treated = treated.fillna(donor.mean())
    return donor.to_numpy(dtype=float).T, treated.to_numpy(dtype=float).ravel()


def prepare_medsc_inputs(
    df: pd.DataFrame,
    outcome: str,
    mediator: str,
    treat: str,
    unitid: str,
    time: str,
    total_donors: Optional[List[Any]],
    direct_donors: Optional[List[Any]],
    covariates: Optional[List[str]],
):
    """Assemble :class:`MEDSCInputs` from a long panel.

    Parameters mirror the estimator config. Returns a populated
    :class:`~mlsynth.utils.medsc_helpers.structures.MEDSCInputs`.
    """
    from .structures import MEDSCInputs

    prepped = dataprep(df, unitid, time, outcome, treat)
    treated_name = prepped["treated_unit_name"]
    Ywide = prepped["Ywide"]
    donor_names = list(prepped["donor_names"])
    T0 = int(prepped["pre_periods"])
    time_labels = np.asarray(Ywide.index)
    T = int(len(time_labels))

    # Mediator panel, aligned to the same periods and unit columns as Ywide.
    Mwide = df.pivot(index=time, columns=unitid, values=mediator)
    Mwide = Mwide.reindex(index=Ywide.index, columns=Ywide.columns)
    if Mwide[treated_name].isna().any():
        raise MlsynthDataError(
            f"mediator '{mediator}' has missing values for the treated unit "
            f"'{treated_name}'.")

    total_pool = _resolve_pool(total_donors, donor_names, "total_donors")
    direct_pool = _resolve_pool(
        direct_donors if direct_donors is not None else total_donors,
        donor_names, "direct_donors")

    treated_outcome = Ywide[treated_name].to_numpy(dtype=float)
    treated_mediator = Mwide[treated_name].to_numpy(dtype=float)
    Y_total = Ywide[total_pool].to_numpy(dtype=float)
    Y_direct = Ywide[direct_pool].to_numpy(dtype=float)
    M_direct = Mwide[direct_pool].to_numpy(dtype=float)

    for name, arr in (("treated outcome", treated_outcome),
                      ("total-pool donor outcomes", Y_total),
                      ("direct-pool donor outcomes", Y_direct),
                      ("direct-pool donor mediators", M_direct)):
        if not np.all(np.isfinite(arr)):
            raise MlsynthDataError(
                f"{name} contain missing or non-finite values; MEDSC needs a "
                f"balanced outcome/mediator panel over the selected pools.")

    treated_cov: Optional[np.ndarray] = None
    total_cov: Optional[np.ndarray] = None
    direct_cov: Optional[np.ndarray] = None
    cov_names: Tuple[Any, ...] = tuple()
    if covariates:
        pre_labels = time_labels[:T0]
        total_cov, treated_cov = _covariate_block(
            df, unitid, time, covariates, pre_labels, total_pool, treated_name)
        direct_cov, _ = _covariate_block(
            df, unitid, time, covariates, pre_labels, direct_pool, treated_name)
        cov_names = tuple(covariates)

    return MEDSCInputs(
        treated_outcome=treated_outcome,
        treated_mediator=treated_mediator,
        total_donor_outcomes=Y_total,
        direct_donor_outcomes=Y_direct,
        direct_donor_mediators=M_direct,
        total_donor_names=tuple(total_pool),
        direct_donor_names=tuple(direct_pool),
        treated_covariates=treated_cov,
        total_covariates=total_cov,
        direct_covariates=direct_cov,
        covariate_names=cov_names,
        time_labels=time_labels,
        T=T,
        T0=T0,
        treated_name=treated_name,
    )
