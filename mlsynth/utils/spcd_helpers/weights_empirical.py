"""Closed-form empirical weights (Algorithm 2 of the paper).

Implements Eq. (9) of the paper (page 13), the closed-form weight
construction used by "Algorithm 2: Empirical Implementation of SPCD":

    w = 2 (Y Y^T + alpha I + lambda 1 1^T)^{-1} y*
        / || (Y Y^T + alpha I + lambda 1 1^T)^{-1} y* ||_1

The paper notes that this expression is *exactly* the optimal weight
under the optimality conditions of Eq. (6) when the global solution
sign vector ``y*`` is known (see the remark "From [10] we know that
once the optimal design profile y* is obtained, then w is the optimal
design weight" on page 9). In practice, since Algorithm 1's iteration
recovers the correct *signs* (Theorem 1) but not necessarily the exact
optimum of the convex QP (Eq. 6), this closed-form gives a fast
near-optimal approximation. The paper states: "In all the experiment
in this paper, we use this simplified implementation" (page 9, top).

Reference
---------
Lu, Y., Li, J., Ying, L., & Blanchet, J. (2022).
"Synthetic Principal Component Design: Fast Covariate Balancing with
Synthetic Controls." arXiv:2211.15241v1.
"""

from __future__ import annotations

import numpy as np


def empirical_weights(M_inv: np.ndarray, y_star: np.ndarray) -> np.ndarray:
    """Compute the closed-form SPCD weights from Eq. (9).

    Implements::

        w = 2 * M^{-1} y* / || M^{-1} y* ||_1

    where ``M`` is the iteration matrix from Eq. (2) and ``y*`` is the
    converged sign vector from Algorithm 1's iteration.

    Parameters
    ----------
    M_inv : np.ndarray
        N x N inverse of the iteration matrix from Eq. (2).
    y_star : np.ndarray
        Length-N final sign vector ``y* in {-1, +1}^N`` from the SPCD or
        NormSPCD iteration.

    Returns
    -------
    w : np.ndarray
        Length-N signed weight vector. Per the optimality condition
        noted on page 13 ("The optimality condition ensures
        sgn(w) = gamma"), the signs of ``w`` should agree with
        ``y_star`` whenever the iteration has converged to a fixed point
        of the closed-form map.
    """

    u = M_inv @ y_star
    denom = np.sum(np.abs(u))
    if denom == 0:
        return np.zeros_like(u)
    return 2.0 * u / denom
