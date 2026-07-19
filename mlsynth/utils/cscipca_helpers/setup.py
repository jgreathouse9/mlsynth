"""Data preparation for the CSCIPCA estimator.

Rides :func:`dataprep` for the canonical single-treated outcome contract, then
adds the piece ``dataprep``'s covariate path cannot: the full time-varying
covariate cube ``X_it`` of shape ``(N, T, L)``. (``dataprep(covariates=...)``
aggregates each covariate to a per-unit pre-period mean -- the Abadie predictor
block -- which discards exactly the time variation CSC-IPCA's instrumented
loadings ``Lambda_it = X_it Gamma`` are built on.)
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import CSCIPCAInputs


def _covariate_cube(
    df: pd.DataFrame,
    unit_order: List,
    covariates: List[str],
    unitid: str,
    time: str,
    time_labels: np.ndarray,
) -> np.ndarray:
    """Pivot the covariate columns into an ``(N, T, L)`` cube in ``unit_order``."""
    layers = []
    for c in covariates:
        wide = df.pivot(index=unitid, columns=time, values=c)
        wide = wide.reindex(index=unit_order, columns=list(time_labels))
        layers.append(wide.to_numpy(dtype=float))
    return np.stack(layers, axis=-1)          # (N, T, L)


def prepare_cscipca_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    covariates: List[str],
    n_factors: int,
) -> CSCIPCAInputs:
    """Pivot the panel and assemble CSC-IPCA inputs via :func:`dataprep`.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel with one row per ``(unit, time)``.
    outcome, treat, unitid, time : str
        Column names.
    covariates : list of str
        Time-varying covariate columns instrumenting the loadings.
    n_factors : int
        Number of latent factors ``K`` (needed to check identification).

    Returns
    -------
    CSCIPCAInputs
        Preprocessed panel for a single treated unit.

    Raises
    ------
    MlsynthDataError
        If the panel has multiple treated cohorts, missing data, a missing
        covariate column, too few pre-treatment periods to identify the
        treated mapping, or no control units.
    """
    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise MlsynthDataError(
            f"covariate column(s) not found in df: {missing}."
        )

    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "CSCIPCA supports a single treated unit; the panel contains "
            "multiple treated cohorts."
        )

    y = np.asarray(prepared["y"], dtype=float).flatten()
    Y0 = np.asarray(prepared["donor_matrix"], dtype=float)          # (T, N_co)
    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    donor_names = np.asarray(prepared["donor_names"])
    treated_name = prepared["treated_unit_name"]
    time_labels = np.asarray(prepared.get("time_labels", np.arange(T)))

    L, K = len(covariates), int(n_factors)

    if np.isnan(y).any() or np.isnan(Y0).any():
        raise MlsynthDataError(
            "CSCIPCA does not support missing outcomes; impute or drop "
            "missing values before calling fit()."
        )
    if Y0.shape[1] < 1:  # pragma: no cover - dataprep always yields >=1 donor for a valid panel
        raise MlsynthDataError("CSCIPCA requires at least one control unit.")
    if T - T0 < 1:  # pragma: no cover - dataprep always yields >=1 post period for a treated unit
        raise MlsynthDataError(
            f"CSCIPCA requires at least 1 post-treatment period; got {T - T0}."
        )
    # Treated mapping Gamma (L x K) is solved from the treated unit's T0
    # pre-periods: the LK normal equations need T0 >= L*K to be identified.
    if T0 < L * K:
        raise MlsynthDataError(
            f"CSCIPCA needs at least L*K = {L * K} pre-treatment periods to "
            f"identify the treated mapping (L={L} covariates, K={K} factors); "
            f"got T0={T0}. Reduce n_factors/covariates or use a longer "
            "pre-period."
        )

    control_cov = _covariate_cube(
        df, list(donor_names), covariates, unitid, time, time_labels)   # (N_co, T, L)
    treated_cov = _covariate_cube(
        df, [treated_name], covariates, unitid, time, time_labels)[0]   # (T, L)

    if np.isnan(control_cov).any() or np.isnan(treated_cov).any():
        raise MlsynthDataError(
            "CSCIPCA does not support missing covariates; impute or drop "
            "missing values before calling fit()."
        )

    return CSCIPCAInputs(
        treated_outcome=y,
        control_outcomes=Y0,
        treated_covariates=treated_cov,
        control_covariates=control_cov,
        covariate_names=tuple(covariates),
        donor_names=donor_names,
        treated_unit_name=treated_name,
        T=T,
        T0=T0,
        time_labels=time_labels,
    )
