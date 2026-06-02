"""Orchestration for the RMSI estimator (Agarwal, Choi & Yuan 2026)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .core import algorithm3
from .structures import RMSIInputs, RMSIResults


def run_rmsi(inputs: RMSIInputs, *, J: int = 2, rank: Optional[int] = None,
             C2: float = 1.0, C3: float = 1.0, C4: float = 1.0) -> RMSIResults:
    """Run RMSI (Algorithm 3) and assemble :class:`RMSIResults`.

    Parameters
    ----------
    inputs : RMSIInputs
    J : int
        Polynomial sieve order (default 2, matching the paper).
    rank : int, optional
        Factor rank; estimated by the eigenvalue-ratio method if omitted.
    C2, C3, C4 : float
        Penalty constants for the soft-thresholded components.
    """
    Y, D, T0 = inputs.Y, inputs.D, inputs.T0
    N, T = inputs.N, inputs.T

    M_hat, used_rank = algorithm3(
        Y, inputs.X, inputs.Z, control_idx=inputs.control_idx, T0=T0,
        J=J, rank=rank, C2=C2, C3=C3, C4=C4,
    )

    treated = D > 0
    effects = np.full((N, T), np.nan)
    effects[treated] = Y[treated] - M_hat[treated]
    att = float(np.nanmean(effects[treated])) if treated.any() else float("nan")

    att_by_period = {}
    for t in range(T0, T):
        col = treated[:, t]
        if col.any():
            att_by_period[inputs.time_labels[t]] = float(np.mean(effects[col, t]))

    tr = inputs.treated_idx
    treated_mean = Y[tr].mean(axis=0)
    synthetic_mean = M_hat[tr].mean(axis=0)

    metadata = {
        "N": N, "T": T, "T0": T0, "rank": used_rank,
        "n_treated": int(tr.size), "n_control": int(inputs.control_idx.size),
        "d_unit_cov": int(inputs.X.shape[1]), "d_time_cov": int(inputs.Z.shape[1]),
        "sieve_order": int(J), "estimator": "RMSI",
    }
    return RMSIResults(
        inputs=inputs, att=att, counterfactual=M_hat, effects=effects,
        att_by_period=att_by_period, treated_mean=treated_mean,
        synthetic_mean=synthetic_mean, rank=used_rank, metadata=metadata,
    )
