"""Cross-validation tuners for the RPCA-SC pipeline.

Bayani (2021) takes the Candes-Li-Ma-Wright (2011) optimal-recovery
value :math:`\\lambda = 1/\\sqrt{\\max(N, T)}` for the PCP sparsity
penalty, and selects the HQF factorisation rank by an explained-
variance threshold. Both defaults are conservative for the *L/S
decomposition identifiability* problem but can be poor for the
*counterfactual prediction* problem (e.g. on the California
Proposition 99 panel they leave PCP under-regularised by ~2x and
PCR-SC beats RPCA-SC on pre-period RMSE by a factor of ~2).

This module adds leave-one-time-period-out cross-validation for the
PCP :math:`\\lambda` and HQF :math:`r` knobs. For each candidate
value:

1. Run the RPCA decomposition on the *full* pre-period donor matrix
   (the donor data is observed at every period -- only the treated
   unit's pre-period is "held out").
2. For each pre-period :math:`t \\in [0, T_0)`, refit the NNLS weights
   on the other :math:`T_0 - 1` periods and predict the held-out
   :math:`y_t`. Aggregate the squared errors.
3. Pick the value with the lowest mean held-out MSE.

This evaluates exactly the quantity the user cares about
(out-of-sample prediction accuracy of the synthetic control)
without touching the donor matrix (so the RPCA decomposition is
not over-fit to the held-out treated period).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ....exceptions import MlsynthEstimationError
from ...bilevel.nnls import nnls_select

# scipy's nnls where it is a fixed, fast release (>= 1.15), else the in-house
# solver (scipy 1.12-1.14 regressed -- raises on the iteration cap).
nnls = nnls_select()
from .hqf import hqf_decompose
from .pcp import pcp_decompose


@dataclass(frozen=True)
class CVResult:
    """Output of one CV sweep."""

    grid: np.ndarray            # candidate values
    cv_mse: np.ndarray          # mean held-out MSE per candidate
    best: float                 # selected value
    best_idx: int


def _loo_nnls_mse(L: np.ndarray, y_pre: np.ndarray) -> float:
    """Mean leave-one-time-period-out NNLS prediction MSE.

    Parameters
    ----------
    L : np.ndarray
        Denoised pre-period donor matrix, shape ``(T0, J)`` (columns
        are donors). This is the rank-r RPCA reconstruction projected
        to the pre-period.
    y_pre : np.ndarray
        Treated unit's pre-period outcomes, shape ``(T0,)``.
    """
    T0 = L.shape[0]
    if T0 < 3:
        # LOO needs at least 2 fitting rows + 1 holdout.
        raise MlsynthEstimationError(
            f"RPCA CV needs T0 >= 3; got T0={T0}."
        )
    sq_err = np.zeros(T0)
    for t in range(T0):
        mask = np.ones(T0, dtype=bool)
        mask[t] = False
        beta, _ = nnls(L[mask, :], y_pre[mask])
        sq_err[t] = (y_pre[t] - L[t, :] @ beta) ** 2
    return float(np.mean(sq_err))


def cv_pcp_lambda(
    donor_pre: np.ndarray,
    treated_pre: np.ndarray,
    grid: Optional[Sequence[float]] = None,
    *,
    base_lambda: Optional[float] = None,
    multipliers: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0),
    pcp_mu: Optional[float] = None,
    pcp_max_iter: int = 1000,
    pcp_tol: float = 1e-9,
) -> CVResult:
    """Leave-one-time-out CV for PCP's sparsity penalty :math:`\\lambda`.

    Parameters
    ----------
    donor_pre : np.ndarray
        Pre-period donor matrix, shape ``(T0, J)`` (columns are donors).
    treated_pre : np.ndarray
        Treated unit's pre-period outcomes, shape ``(T0,)``.
    grid : sequence of float, optional
        Explicit grid of lambda values. If ``None``, the grid is
        built as ``multipliers * base_lambda`` where ``base_lambda``
        defaults to Candes' value :math:`1/\\sqrt{\\max(J, T_0)}`.
    base_lambda, multipliers
        Used only when ``grid`` is not supplied.
    pcp_mu, pcp_max_iter, pcp_tol
        Forwarded to :func:`mlsynth.utils.clustersc_helpers.rpca.pcp.pcp_decompose`.
    """
    T0, J = donor_pre.shape
    if grid is None:
        if base_lambda is None:
            base_lambda = 1.0 / np.sqrt(max(J, T0))
        grid_arr = np.array([base_lambda * m for m in multipliers], dtype=float)
    else:
        grid_arr = np.asarray(list(grid), dtype=float)

    # Donor matrix as (donors-as-rows) for the PCP solver.
    donor_pre_T = donor_pre.T  # (J, T0)
    cv_mse = np.zeros(grid_arr.size)
    for i, lam in enumerate(grid_arr):
        result = pcp_decompose(
            Y=donor_pre_T, lam=float(lam),
            mu=pcp_mu, max_iter=pcp_max_iter, tol=pcp_tol,
        )
        L_pre = result.low_rank.T  # back to (T0, J) shape
        cv_mse[i] = _loo_nnls_mse(L_pre, treated_pre)

    best_idx = int(np.argmin(cv_mse))
    return CVResult(
        grid=grid_arr, cv_mse=cv_mse,
        best=float(grid_arr[best_idx]), best_idx=best_idx,
    )


def cv_hqf_rank(
    donor_pre: np.ndarray,
    treated_pre: np.ndarray,
    grid: Optional[Sequence[int]] = None,
    *,
    max_rank: Optional[int] = None,
    hqf_lambda: Optional[float] = None,
    hqf_ip: float = 1.0,
    hqf_max_iter: int = 1000,
    random_state: int = 0,
) -> CVResult:
    """Leave-one-time-out CV for HQF's factorisation rank :math:`r`.

    Parameters
    ----------
    donor_pre : np.ndarray
        Pre-period donor matrix, shape ``(T0, J)``.
    treated_pre : np.ndarray
        Treated unit's pre-period outcomes, shape ``(T0,)``.
    grid : sequence of int, optional
        Integer ranks to try. Defaults to ``range(1, max_rank+1)``
        with ``max_rank = min(J, T0-1)``.
    max_rank
        Upper bound for the default grid.
    hqf_lambda, hqf_ip, hqf_max_iter, random_state
        Forwarded to :func:`mlsynth.utils.clustersc_helpers.rpca.hqf.hqf_decompose`.
    """
    T0, J = donor_pre.shape
    cap = max(1, min(max_rank if max_rank is not None else min(J, T0 - 1),
                     min(J, T0 - 1)))
    if grid is None:
        grid_arr = np.arange(1, cap + 1, dtype=int)
    else:
        grid_arr = np.asarray(list(grid), dtype=int)

    donor_pre_T = donor_pre.T
    cv_mse = np.zeros(grid_arr.size)
    for i, r in enumerate(grid_arr):
        result = hqf_decompose(
            Y=donor_pre_T, rank=int(r),
            lam=hqf_lambda, ip=hqf_ip,
            max_iter=hqf_max_iter, random_state=random_state,
        )
        L_pre = result.low_rank.T
        cv_mse[i] = _loo_nnls_mse(L_pre, treated_pre)

    best_idx = int(np.argmin(cv_mse))
    return CVResult(
        grid=grid_arr, cv_mse=cv_mse,
        best=int(grid_arr[best_idx]), best_idx=best_idx,
    )
