"""Panel ingestion for the PANGEO design estimator.

Pivots a historical (pre-treatment) long panel into a wide
``units x time`` outcome matrix and records each unit's treatment-arm
eligibility from a single categorical ``arm`` column (values ``A``, ``B``,
... ). The design is run independently within each arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError


@dataclass(frozen=True)
class PangeoInputs:
    """Preprocessed pre-treatment panel for PANGEO.

    Attributes
    ----------
    Y : np.ndarray
        Pre-period outcomes, shape ``(N, T)``; rows = units in
        ``unit_names`` order.
    unit_names : list
        Length-``N`` unit identifiers.
    time_labels : np.ndarray
        Length-``T`` time labels.
    arm_of : dict
        ``{unit_name: arm_label}``.
    arm_units : dict
        ``{arm_label: np.ndarray of row indices}`` (the arm's geo pool).
    covariates : np.ndarray or None
        Baseline covariate matrix, shape ``(N, M)`` aligned with
        ``unit_names`` rows (``None`` if no covariates requested).
    covariate_names : list
        Length-``M`` covariate column names (empty if none).
    covariate_scales : np.ndarray or None
        Length-``M`` cross-unit standard deviations used to standardize the
        covariate imbalance (``None`` if no covariates).
    """

    Y: np.ndarray
    unit_names: List[Any]
    time_labels: np.ndarray
    arm_of: Dict[Any, Any]
    arm_units: Dict[Any, np.ndarray]
    covariates: Optional[np.ndarray] = None
    covariate_names: List[str] = field(default_factory=list)
    covariate_scales: Optional[np.ndarray] = None


def prepare_pangeo_inputs(
    df: pd.DataFrame,
    outcome: str,
    arm: str,
    unitid: str,
    time: str,
    min_units_per_arm: int = 2,
    covariates: Optional[List[str]] = None,
    standardize_covariates: bool = True,
) -> PangeoInputs:
    """Pivot a historical panel into :class:`PangeoInputs`.

    Parameters
    ----------
    df : pd.DataFrame
        Balanced pre-treatment long panel; one row per ``(unit, time)``.
    outcome : str
        Historical outcome column (e.g. sales).
    arm : str
        Single categorical column naming each geo's eligible treatment arm
        (e.g. values ``A``/``B``/``C``). Units are designed within their arm.
    unitid, time : str
        Unit-id and time column names.
    min_units_per_arm : int
        Minimum geos required per arm to form at least one supergeo pair.
    covariates : list of str, optional
        Baseline covariate columns to balance across supergeo halves. Each
        unit's covariate value is its mean over the panel (so a column that
        varies over time is reduced to a per-unit baseline level).
    standardize_covariates : bool
        Divide each covariate's imbalance by its cross-unit std (default).
        With ``False`` the raw scale is used (``scales = 1``).
    """
    for col in (outcome, arm, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(f"Required column {col!r} missing.")
    if df[outcome].isna().any():
        raise MlsynthDataError("Outcome column contains NaN values.")
    for col in covariates or []:
        if col not in df.columns:
            raise MlsynthDataError(f"Covariate column {col!r} missing.")

    time_labels = np.array(sorted(df[time].unique()))
    unit_names = sorted(df[unitid].unique())
    if len(unit_names) < 2:
        raise MlsynthDataError("PANGEO needs at least 2 units.")

    wide = df.pivot(index=unitid, columns=time, values=outcome)
    if wide.isna().any().any():
        raise MlsynthDataError("Panel is unbalanced (missing unit-time cells).")
    Y = wide.loc[unit_names, time_labels].to_numpy(dtype=float)

    # Arm membership is time-invariant per unit.
    arm_by_unit = df.groupby(unitid)[arm].agg(lambda s: s.iloc[0])
    if df.groupby(unitid)[arm].nunique().max() > 1:
        raise MlsynthDataError("The arm column varies within a unit over time.")
    arm_of = {u: arm_by_unit.loc[u] for u in unit_names}

    arm_units: Dict[Any, np.ndarray] = {}
    for i, u in enumerate(unit_names):
        arm_units.setdefault(arm_of[u], []).append(i)
    arm_units = {a: np.array(idx, dtype=int) for a, idx in arm_units.items()}

    small = [a for a, idx in arm_units.items() if idx.size < min_units_per_arm]
    if small:
        raise MlsynthConfigError(
            f"Arm(s) {small} have fewer than {min_units_per_arm} units; "
            "cannot form a supergeo pair."
        )

    cov = None
    cov_names: List[str] = []
    cov_scales = None
    if covariates:
        if df[covariates].isna().any().any():
            raise MlsynthDataError("Covariate column(s) contain NaN values.")
        cov_names = list(covariates)
        baseline = df.groupby(unitid)[cov_names].mean()
        cov = baseline.loc[unit_names].to_numpy(dtype=float)  # (N, M)
        if standardize_covariates:
            scales = cov.std(axis=0, ddof=0)
            cov_scales = np.where(scales > 1e-12, scales, 1.0)
        else:
            cov_scales = np.ones(cov.shape[1])

    return PangeoInputs(
        Y=Y, unit_names=list(unit_names), time_labels=time_labels,
        arm_of=arm_of, arm_units=arm_units,
        covariates=cov, covariate_names=cov_names, covariate_scales=cov_scales,
    )
