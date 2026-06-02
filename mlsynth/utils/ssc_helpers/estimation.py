"""SSC effect estimation and Andrews end-of-sample inference.

Given the synthetic-control weights ``(a_hat, B_hat)`` fitted on the clean
pre-period, this module (i) stacks the treated unit-period cells into the
selector tensor ``A_s``, (ii) solves the GLS estimator for the full vector of
individual effects ``tau`` (Cao, Lu & Wu 2026, eq. 2.4), (iii) aggregates to any
linear target ``gamma = L tau`` (event-time / overall ATT), and (iv) calibrates
an end-of-sample stability band (Andrews 2003) from pre-treatment residual
windows.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .structures import SSCBand


def build_treatment_structure(D: np.ndarray, T0: int) -> Tuple[np.ndarray, np.ndarray]:
    """Index the treated post-period cells and build the selector tensor.

    Parameters
    ----------
    D : np.ndarray, shape (N, T1)
        Absorbing treatment indicators.
    T0 : int
        Number of clean pre-treatment periods.

    Returns
    -------
    index : np.ndarray, shape (K, 3)
        Rows ``[post_period s (1-based), unit_index, event_time e (0-based)]``.
    A : np.ndarray, shape (N, K, S)
        ``A[i, k, s-1] = 1`` iff treated cell ``k`` is unit ``i`` at post
        period ``s``.
    """
    N, T1 = D.shape
    S = T1 - T0
    D_S = D[:, T0:]
    K = int(D_S.sum())
    index = np.zeros((K, 3), dtype=int)
    A = np.zeros((N, K, S))
    k = 0
    for s in range(1, S + 1):
        for i in range(N):
            if D_S[i, s - 1] == 1:
                e = int(D_S[i, :s].sum()) - 1          # event time, 0-based
                index[k] = (s, i, e)
                A[i, k, s - 1] = 1.0
                k += 1
    return index, A


def estimate_tau(Y: np.ndarray, T0: int, A: np.ndarray,
                 a_hat: np.ndarray, B_hat: np.ndarray):
    """Solve the GLS estimator (eq. 2.4) for the individual-effect vector.

    Parameters
    ----------
    Y : np.ndarray, shape (N, T1)
        Full outcome panel.
    T0 : int
        Clean pre-period count.
    A : np.ndarray, shape (N, K, S)
        Selector tensor from :func:`build_treatment_structure`.
    a_hat : np.ndarray, shape (N,)
        Synthetic-control intercepts.
    B_hat : np.ndarray, shape (N, N)
        Synthetic-control weight matrix.

    Returns
    -------
    tau : np.ndarray, shape (K,)
        Estimated individual treatment effects.
    gram : np.ndarray, shape (K, K)
        ``sum_s A_s' M A_s`` (the design Gram; invertible under Assumption 2.1).
    residuals : np.ndarray, shape (N, T0)
        Pre-treatment prediction errors ``Y_T - (a + B Y_T)``.
    """
    N = Y.shape[0]
    S = A.shape[2]
    ImB = np.eye(N) - B_hat
    M = ImB.T @ ImB
    Y_S = Y[:, T0:]

    gram = sum(A[:, :, s].T @ M @ A[:, :, s] for s in range(S))
    rhs = sum(A[:, :, s].T @ ImB.T @ (ImB @ Y_S[:, s] - a_hat) for s in range(S))
    tau = _solve(gram, rhs)

    Y_T = Y[:, :T0]
    residuals = Y_T - (a_hat[:, None] + B_hat @ Y_T)
    return tau, gram, residuals


def placebo_windows(gram: np.ndarray, A: np.ndarray, B_hat: np.ndarray,
                    residuals: np.ndarray, T0: int) -> np.ndarray:
    """Pre-treatment "placebo effect" estimates for end-of-sample inference.

    Slides a length-``S`` window across the pre-treatment residuals and, for
    each, applies the same GLS map used for ``tau`` -- yielding ``T0 - S``
    draws of the estimator under the null of no effect (Andrews 2003).

    Returns
    -------
    V : np.ndarray, shape (K, T0 - S)
        Placebo individual-effect vectors (columns).
    """
    N = residuals.shape[0]
    S = A.shape[2]
    ImB = np.eye(N) - B_hat
    n_windows = max(0, T0 - S)        # empty when T0 <= S (no placebo windows)
    V = np.zeros((gram.shape[0], n_windows))
    if n_windows == 0:
        return V
    # The Gram is fixed across windows: factor (pseudo-invert) it once.
    gram_pinv = np.linalg.pinv(gram)
    ImB_T = ImB.T
    AtImB = [A[:, :, s].T @ ImB_T for s in range(S)]   # reused each window
    for w, t in enumerate(range(1, n_windows + 1)):
        rhs = sum(AtImB[s] @ residuals[:, t + s] for s in range(S))
        V[:, w] = gram_pinv @ rhs
    return V


def aggregate(L: np.ndarray, tau: np.ndarray, V: np.ndarray,
              alpha: float, label, n_cells: int) -> SSCBand:
    """Aggregate ``tau`` by ``L`` and attach the end-of-sample band + p-value.

    The placebo draws ``L V`` are (asymptotically) mean-zero replicates of the
    estimator's error ``L\tau_hat - L\tau``, so inverting gives the band
    ``[point - q_{1-alpha/2}, point - q_{alpha/2}]`` (Cao, Lu & Wu 2026; the
    reference implementation's ``att - ub`` / ``att - lb``). The two-sided
    p-value for ``H0: L tau = 0`` is the share of placebo draws at least as
    large in magnitude as the point estimate.
    """
    point = float(np.ravel(np.asarray(L) @ tau)[0])
    if V is None or V.shape[1] == 0:
        # No pre-treatment placebo windows (T0 <= S): point estimate only.
        return SSCBand(label=label, point=point, lower=float("nan"),
                       upper=float("nan"), p_value=float("nan"),
                       n_cells=int(n_cells))
    null = np.asarray(L @ V, dtype=float).ravel()
    # "hazen" plotting positions ((i-0.5)/n) match MATLAB's `quantile`, the
    # reference implementation's convention.
    q_lo = float(np.quantile(null, alpha / 2.0, method="hazen"))
    q_hi = float(np.quantile(null, 1.0 - alpha / 2.0, method="hazen"))
    lo = point - q_hi
    hi = point - q_lo
    p = float(np.mean(np.abs(null) >= abs(point)))
    return SSCBand(label=label, point=point, lower=lo, upper=hi,
                   p_value=p, n_cells=int(n_cells))


def event_time_maps(index: np.ndarray) -> Dict[int, np.ndarray]:
    """For each event time ``e``, the averaging row ``L_e`` (1/n_e on its cells)."""
    K = index.shape[0]
    maps: Dict[int, np.ndarray] = {}
    for e in sorted(set(index[:, 2].tolist())):
        mask = (index[:, 2] == e).astype(float)
        n_e = mask.sum()
        maps[int(e)] = mask / n_e
    return maps


def _solve(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``gram x = rhs``, falling back to least squares if near-singular."""
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]
