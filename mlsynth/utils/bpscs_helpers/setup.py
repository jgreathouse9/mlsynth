"""Long-DataFrame -> NumPy boundary for BPSCS (wraps ``dataprep``).

Reproduces the reference pre-processing (Fernandez-Morales et al. 2026): outcomes
standardized on the pre-period, baseline covariates z-scored across units, and a
per-donor utility score blending covariate similarity with spatial distance to
the treated unit. The utility feeds the shrinkage prior; for ``ds2`` it also sets
the inclusion radius.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import dataprep
from .structures import BPSCSInputs


def _unit_feature_matrix(
    df: pd.DataFrame, unitid: str, columns: List[str], order: List,
    what: str,
) -> np.ndarray:
    """Baseline (time-invariant) per-unit feature matrix, rows ordered by ``order``."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise MlsynthConfigError(f"BPSCS {what} column(s) not in the panel: {missing}")
    first = df.groupby(unitid)[columns].first()
    try:
        mat = first.reindex(index=order).to_numpy(dtype=float)
    except Exception as exc:  # pragma: no cover - defensive
        raise MlsynthDataError(f"could not align BPSCS {what}: {exc}") from exc
    if not np.isfinite(mat).all():
        raise MlsynthDataError(f"BPSCS {what} must be fully observed and finite.")
    return mat


def prepare_bpscs_inputs(
    df: pd.DataFrame, outcome: str, unitid: str, time: str, treat: str,
    covariates: List[str], coords: List[str], kappa_d: float,
    inclusion_quantile: float, prior: str,
) -> BPSCSInputs:
    """Pivot a long panel into BPSCS's standardized inputs (treated column 0)."""
    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:  # pragma: no cover - dataprep guards single-cohort
        raise MlsynthDataError(
            "BPSCS supports a single treated unit; the panel appears to contain "
            "multiple treatment cohorts."
        )
    y_target = np.asarray(prepared["y"], dtype=float)             # (T,)
    donors = np.asarray(prepared["donor_matrix"], dtype=float)     # (T, N)
    if donors.ndim != 2 or donors.shape[1] < 1:  # pragma: no cover - dataprep guards
        raise MlsynthDataError("BPSCS requires a 2D donor matrix with >= 1 donor.")
    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    if T0 < 2:  # pragma: no cover - degenerate single-pre-period panel
        raise MlsynthDataError(f"BPSCS needs at least 2 pre-treatment periods; got {T0}.")
    if T - T0 < 1:  # pragma: no cover - a treated period forces >= 1 post
        raise MlsynthDataError("BPSCS needs at least 1 post-treatment period.")

    Y = np.column_stack([y_target, donors])                       # (T, D) treated col 0
    if not np.isfinite(Y).all():  # pragma: no cover - dataprep guards balance
        raise MlsynthDataError("BPSCS requires a balanced panel with no missing outcomes.")

    treated_name = str(prepared.get("treated_unit_name", "treated"))
    donor_names = list(prepared.get("donor_names", range(donors.shape[1])))
    time_labels = np.asarray(prepared["time_labels"])
    order = [treated_name, *donor_names]

    # standardize each series on the pre-period (reference dataprep)
    pre = slice(0, T0)
    col_mean = Y[pre].mean(axis=0)
    col_sd = Y[pre].std(axis=0, ddof=1)
    if np.any(col_sd <= 0):
        raise MlsynthDataError("BPSCS: a series is constant over the pre-period.")
    Y_std = (Y - col_mean) / col_sd
    mean_pre_treated = float(col_mean[0])
    stdv0 = float(col_sd[0])

    # baseline covariates (z-scored across units) and coordinates
    X_cov = _unit_feature_matrix(df, unitid, covariates, order, "covariate")
    cov_mean = X_cov.mean(axis=0)
    cov_sd = X_cov.std(axis=0, ddof=1)
    cov_sd = np.where(cov_sd > 0, cov_sd, 1.0)
    X_cov = (X_cov - cov_mean) / cov_sd
    P_coords = _unit_feature_matrix(df, unitid, coords, order, "coordinate")

    # utility = kappa * covariate-similarity + (1 - kappa) * normalized distance
    dist = np.linalg.norm(P_coords - P_coords[0], axis=1)          # to treated (index 0)
    S = float(dist.max())
    d_P = dist / S if S > 0 else np.zeros_like(dist)
    cov_dissim = np.linalg.norm(X_cov - X_cov[0], axis=1)
    d_X = 1.0 / (1.0 + cov_dissim)
    u_C = kappa_d * d_X + (1.0 - kappa_d) * d_P                    # (D,) treated at 0

    rho = float(np.quantile(u_C, inclusion_quantile))
    donor_d = u_C[1:].astype(float).copy()                        # (J,)
    if prior == "dhs":
        donor_d[donor_d <= 0] = 1e-10                             # positive prior scale

    return BPSCSInputs(
        Y_std=Y_std, y_target=y_target, mean_pre_treated=mean_pre_treated,
        stdv0=stdv0, X_cov=X_cov, donor_d=donor_d, rho=rho, kappa_d=float(kappa_d),
        T0=T0, T=T, D=Y.shape[1], prior=prior, treated_unit_name=treated_name,
        donor_names=donor_names, time_labels=time_labels,
    )
