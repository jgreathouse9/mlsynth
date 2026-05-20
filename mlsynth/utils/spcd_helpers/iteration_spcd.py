"""SPCD iteration step (generalized power method).

Implements the **SPCD update box** from Algorithm 1 (page 7) and the
identical update from Algorithm 2 (page 13) of the paper:

    Eq. (4) / Eq. (7):
        y^{t+1} = sgn[ (M^{-1} + beta I) y^t ],

where ``M = Y Y^T + alpha I + lambda 1 1^T`` (Eq. 2) and ``beta`` is a
pre-defined step parameter.

Theoretical justification: Algorithm 3 of Appendix 3.2 (page 20) gives
the abstract meta-version of this iteration on a generic Hermitian
perturbed rank-1 matrix. Theorem 3 (page 9, informal) gives global
convergence under Assumptions 1-2.

Reference
---------
Lu, Y., Li, J., Ying, L., & Blanchet, J. (2022).
"Synthetic Principal Component Design: Fast Covariate Balancing with
Synthetic Controls." arXiv:2211.15241v1.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def spcd_step(M_inv: np.ndarray, y: np.ndarray, beta: float) -> np.ndarray:
    """One iteration of the SPCD update box, Eq. (4)/(7).

    Computes::

        y^{t+1} = sgn[ (M^{-1} + beta I) y^t ]

    Parameters
    ----------
    M_inv : np.ndarray
        N x N inverse of the iteration matrix from Eq. (2).
    y : np.ndarray
        Length-N sign vector at iteration ``t``.
    beta : float
        Iteration step parameter from Eq. (4)/(7).

    Returns
    -------
    y_new : np.ndarray
        Length-N sign vector at iteration ``t + 1``.
    """

    z = M_inv @ y + beta * y
    y_new = np.sign(z)
    y_new[y_new == 0] = 1.0
    return y_new.astype(float)


def run_spcd_iteration(
    M_inv: np.ndarray,
    y0: np.ndarray,
    beta: float,
    max_iter: int,
) -> Tuple[np.ndarray, int, bool]:
    """Run the SPCD ``while Converged do`` loop using Eq. (4)/(7).

    Implements the SPCD branch of the outer loop in Algorithm 1
    (page 7) and Algorithm 2 (page 13). The loop terminates as soon as
    the sign vector stops changing between successive iterations.

    Parameters
    ----------
    M_inv : np.ndarray
        N x N inverse of the iteration matrix from Eq. (2).
    y0 : np.ndarray
        Initial sign vector from Spectral Initialization.
    beta : float
        Iteration step parameter from Eq. (4)/(7).
    max_iter : int
        Maximum number of iterations to perform before terminating.

    Returns
    -------
    y_final : np.ndarray
        Final sign vector ``y* in {-1, +1}^N``.
    n_iter : int
        Number of iterations actually performed.
    converged : bool
        ``True`` if the sign vector stabilized before ``max_iter``.
    """

    y = y0.astype(float).copy()
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        y_new = spcd_step(M_inv, y, beta)
        if np.array_equal(y_new, y):
            converged = True
            y = y_new
            break
        y = y_new

    return y, n_iter, converged
