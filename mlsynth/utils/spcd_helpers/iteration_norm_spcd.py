"""NormSPCD iteration step (normalized generalized power method).

Implements the **NormSPCD update box** from Algorithm 1 (page 7) and
the identical update from Algorithm 2 (page 13) of the paper:

    Eq. (5) / Eq. (8):
        y^{t+1} = sgn[ (M^{-1} + beta I) (y^t / d) ],

where::

    M = Y Y^T + alpha I + lambda 1 1^T          (Eq. 2)
    d = sqrt(diag(M^{-1}))                      (definition below Eq. 5)

and ``/`` denotes element-wise division.

Theoretical justification: Algorithm 4 of Appendix 3.2 (page 20) gives
the abstract meta-version of this iteration. In the same appendix the
paper interprets NormSPCD as a Riemannian gradient descent under a
diagonal preconditioning given by the inverse-covariance diagonal
(``Appendix 3``, page 8 and Appendix 3.2). Theorem 3 (page 9) gives
linear-rate global convergence under Assumptions 1-2 with ``epsilon > 0``.

Reference
---------
Lu, Y., Li, J., Ying, L., & Blanchet, J. (2022).
"Synthetic Principal Component Design: Fast Covariate Balancing with
Synthetic Controls." arXiv:2211.15241v1.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def norm_spcd_step(
    M_inv: np.ndarray,
    d: np.ndarray,
    y: np.ndarray,
    beta: float,
) -> np.ndarray:
    """One iteration of the NormSPCD update box, Eq. (5)/(8).

    Computes::

        y^{t+1} = sgn[ (M^{-1} + beta I) (y^t / d) ]

    Parameters
    ----------
    M_inv : np.ndarray
        N x N inverse of the iteration matrix from Eq. (2).
    d : np.ndarray
        Length-N normalization vector ``sqrt(diag(M^{-1}))``. See the
        definition immediately below Eq. (5), page 7.
    y : np.ndarray
        Length-N sign vector at iteration ``t``.
    beta : float
        Iteration step parameter from Eq. (5)/(8).

    Returns
    -------
    y_new : np.ndarray
        Length-N sign vector at iteration ``t + 1``.
    """

    y_scaled = y / d
    z = M_inv @ y_scaled + beta * y_scaled
    y_new = np.sign(z)
    y_new[y_new == 0] = 1.0
    return y_new.astype(float)


def run_norm_spcd_iteration(
    M_inv: np.ndarray,
    y0: np.ndarray,
    beta: float,
    max_iter: int,
) -> Tuple[np.ndarray, int, bool]:
    """Run the NormSPCD ``while Converged do`` loop using Eq. (5)/(8).

    Implements the NormSPCD branch of the outer loop in Algorithm 1
    (page 7) and Algorithm 2 (page 13). The loop terminates as soon as
    the sign vector stops changing between successive iterations.

    Parameters
    ----------
    M_inv : np.ndarray
        N x N inverse of the iteration matrix from Eq. (2).
    y0 : np.ndarray
        Initial sign vector from Spectral Initialization.
    beta : float
        Iteration step parameter from Eq. (5)/(8).
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

    diag_M_inv = np.diag(M_inv).astype(float)
    diag_M_inv = np.where(diag_M_inv > 0, diag_M_inv, np.finfo(float).eps)
    d = np.sqrt(diag_M_inv)

    y = y0.astype(float).copy()
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        y_new = norm_spcd_step(M_inv, d, y, beta)
        if np.array_equal(y_new, y):
            converged = True
            y = y_new
            break
        y = y_new

    return y, n_iter, converged
