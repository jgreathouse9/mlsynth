"""Factor extraction + selection for FMA.

Wraps the existing :func:`mlsynth.utils.denoiseutils.nbpiid` helper
to apply the appropriate Li & Sonnier (2023) selection criterion:

* ``stationarity="stationary"`` -> modified Bai-Ng (MBN) -- nbpiid
  criterion code 10 with ``N_co + max(70 - N_co, 0)`` /
  ``T + max(70 - T, 0)`` adjustments (Web Appendix D.1).
* ``stationarity="nonstationary"`` -> Bai (2004) IPC1 with the
  log-log adjustment -- nbpiid criterion code 11.

A user may also bypass the data-driven selection by passing
``n_factors`` directly to :func:`extract_factors`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthEstimationError
from ..denoiseutils import nbpiid


_STATIONARITY_TO_CRITERION = {
    "stationary": 10,
    "nonstationary": 11,
}


def _mbn_adjustments(N_co: int, T: int) -> Tuple[int, int]:
    """Web Appendix D.1: m_N = max(0, 70 - N_co), m_T = max(0, 70 - T)."""
    return max(0, 70 - int(N_co)), max(0, 70 - int(T))


def extract_factors(
    control_panel: np.ndarray,
    stationarity: str = "nonstationary",
    preprocessing: str = "demean",
    n_factors: Optional[int] = None,
    max_factors: int = 10,
) -> Tuple[int, np.ndarray, np.ndarray, str]:
    """Estimate the factor structure from the control panel.

    Parameters
    ----------
    control_panel : np.ndarray
        ``(T, N_co)`` matrix of control-unit outcomes.
    stationarity : {"stationary", "nonstationary"}
        Drives the factor-selection criterion (MBN vs IPC1).
    preprocessing : {"demean", "standardize"}
        Mapped to ``nbpiid``'s preprocessing_method_code: 1 for demean,
        2 for standardize.
    n_factors : int, optional
        Override the data-driven selection. ``None`` triggers MBN/IPC1.
    max_factors : int
        Upper bound passed to the selection routine; ignored when
        ``n_factors`` is supplied.

    Returns
    -------
    n_factors_selected : int
        Final number of factors used.
    common_component : np.ndarray
        ``(T, N_co)`` reconstruction of the control panel.
    factors : np.ndarray
        ``(T, r)`` matrix of estimated factors.
    source : str
        ``"MBN"``, ``"IPC1"``, or ``"user"``.
    """

    if stationarity not in _STATIONARITY_TO_CRITERION:
        raise MlsynthConfigError(
            f"stationarity must be 'stationary' or 'nonstationary'; "
            f"got {stationarity!r}."
        )
    if preprocessing not in {"demean", "standardize"}:
        raise MlsynthConfigError(
            f"preprocessing must be 'demean' or 'standardize'; "
            f"got {preprocessing!r}."
        )

    T, N_co = control_panel.shape
    preprocessing_code = 1 if preprocessing == "demean" else 2
    upper_bound = int(min(max_factors, T, N_co))
    if upper_bound < 1:
        raise MlsynthEstimationError(
            f"Control panel too small to extract factors: T={T}, "
            f"N_co={N_co}."
        )

    if n_factors is not None:
        if not (1 <= n_factors <= upper_bound):
            raise MlsynthConfigError(
                f"n_factors must lie in [1, {upper_bound}]; got {n_factors}."
            )
        # PCA at the user's rank: preprocess the panel exactly as nbpiid
        # does (demean column-by-column, optionally standardise) and
        # take the top-r left singular vectors as the factors.
        X = control_panel.astype(float).copy()
        if preprocessing_code >= 1:
            X = X - X.mean(axis=0, keepdims=True)
        if preprocessing_code == 2:
            sd = X.std(axis=0, ddof=0)
            sd = np.where(sd > 0, sd, 1.0)
            X = X / sd
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        r_user = int(n_factors)
        F_user = U[:, :r_user] * np.sqrt(T)
        lambdas = np.linalg.lstsq(F_user, control_panel, rcond=None)[0]
        common_user = F_user @ lambdas
        return r_user, common_user, F_user, "user"

    criterion_code = _STATIONARITY_TO_CRITERION[stationarity]
    if criterion_code == 10:
        m_N, m_T = _mbn_adjustments(N_co, T)
        source = "MBN"
    else:
        m_N, m_T = None, None
        source = "IPC1"

    n_sel, common, F = nbpiid(
        input_panel_data=control_panel,
        max_factors_to_test=upper_bound,
        criterion_selector_code=criterion_code,
        preprocessing_method_code=preprocessing_code,
        N_series_adjustment=m_N,
        T_obs_adjustment=m_T,
    )
    return int(n_sel), common, F, source
