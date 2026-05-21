"""Data preparation and EM initialization helpers for TASC.

``datautils.dataprep`` returns the wide outcome matrix in the
``(T, N)`` orientation (rows = time, columns = unit). The TASC paper works
with ``Y in R^{N x T}`` (rows = unit, columns = time), so the wrappers in
this module take care of the transpose once and only once.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError, MlsynthEstimationError
from ..datautils import dataprep
from .structures import TASCInputs, TASCParameters


def prepare_tasc_inputs(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    treat: str,
) -> TASCInputs:
    """Run ``dataprep`` and reshape the result into the paper's N x T layout.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel data.
    outcome, unitid, time, treat : str
        Column names identifying the outcome, units, time periods, and the
        binary treatment indicator.

    Returns
    -------
    TASCInputs
        Pre / post matrices in ``(N, T)`` orientation along with metadata.
    """

    prepared = dataprep(df, unitid, time, outcome, treat)

    if "cohorts" in prepared:
        raise MlsynthDataError(
            "TASC currently supports a single treated unit; the input panel "
            "appears to contain multiple treatment cohorts."
        )

    y_target = np.asarray(prepared["y"], dtype=float)
    donor_matrix_tn = np.asarray(prepared["donor_matrix"], dtype=float)
    if donor_matrix_tn.ndim != 2:
        raise MlsynthDataError(
            "TASC requires a 2D donor matrix; got shape "
            f"{donor_matrix_tn.shape}."
        )

    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    n_donors = donor_matrix_tn.shape[1]

    if n_donors < 1:
        raise MlsynthDataError("TASC requires at least one donor unit.")
    if T0 < 2:
        raise MlsynthDataError(
            "TASC requires at least two pre-treatment periods to fit the "
            "state-space model."
        )

    Y_full = np.vstack([y_target.reshape(1, T), donor_matrix_tn.T])
    Y_pre = Y_full[:, :T0]
    if T0 < T:
        Y_post_donors = Y_full[1:, T0:]
    else:
        Y_post_donors = None

    return TASCInputs(
        Y_full=Y_full,
        Y_pre=Y_pre,
        Y_post_donors=Y_post_donors,
        T0=T0,
        T=T,
        N=Y_full.shape[0],
        treated_unit_name=prepared["treated_unit_name"],
        donor_names=list(prepared["donor_names"]),
        time_labels=np.asarray(prepared["time_labels"]),
        pre_periods=T0,
        post_periods=int(prepared["post_periods"]),
        Ywide=prepared["Ywide"],
        y_target=y_target,
    )


def initialize_parameters(
    Y_pre: np.ndarray,
    d: int,
    seed: Optional[int] = None,
) -> TASCParameters:
    """Spectral initialization for the EM parameters.

    Performs a thin SVD of ``Y_pre`` (``N x T0``) and uses the top-``d``
    left singular vectors as the initial observation matrix ``H``. The
    initial latent trajectory is then the corresponding right singular
    vectors scaled by the singular values, from which a simple AR(1) least
    squares fit gives ``A``. ``Q``, ``R``, ``P0`` are seeded from the
    associated residual variances.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment outcome matrix of shape ``(N, T0)``.
    d : int
        Hidden state dimension.
    seed : int or None
        Reserved for tie-breaking; not currently consumed but retained for
        API symmetry with other estimators.

    Returns
    -------
    TASCParameters
        Initial ``theta_0``.
    """

    del seed  # spectral init is deterministic up to SVD sign conventions

    N, T0 = Y_pre.shape
    if d > min(N, T0):
        raise MlsynthEstimationError(
            f"Hidden state dimension d={d} exceeds min(N, T0)=min({N}, {T0})."
        )

    U, s, Vt = np.linalg.svd(Y_pre, full_matrices=False)
    U_d = U[:, :d]
    s_d = s[:d]
    V_d = Vt[:d, :].T  # (T0, d)

    H = U_d * s_d  # (N, d); columns scaled so that x_t lives on the V_d rows
    X = V_d  # (T0, d) latent trajectory estimate

    # AR(1) least squares fit for A on the spectral trajectory.
    if T0 >= 2:
        X_prev = X[:-1, :]
        X_next = X[1:, :]
        gram = X_prev.T @ X_prev
        ridge = 1e-8 * np.eye(d)
        A = np.linalg.solve(gram + ridge, X_prev.T @ X_next).T
        state_resid = X_next - X_prev @ A.T
        q_diag = np.maximum(state_resid.var(axis=0, ddof=1) if T0 > 2 else
                            np.full(d, 1e-2), 1e-6)
    else:  # pragma: no cover - guarded above by T0 >= 2
        A = np.eye(d)
        q_diag = np.full(d, 1e-2)

    Q = np.diag(q_diag)

    obs_resid = Y_pre - H @ X.T
    r_diag = np.maximum(obs_resid.var(axis=1, ddof=1) if T0 > 1 else
                        np.full(N, 1e-2), 1e-6)
    R = np.diag(r_diag)

    m0 = X[0, :].copy()
    P0 = Q.copy()

    return TASCParameters(A=A, H=H, Q=Q, R=R, m0=m0, P0=P0)
