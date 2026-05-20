"""Spectral initialization for the SPCD iteration.

Implements the "Spectral Initialization" step of Algorithm 1 (page 7)
and Algorithm 2 (page 13) of the SPCD paper:

    y^0 = sgn(v),

where ``v`` is the smallest eigenvector of the iteration matrix

    M = Y Y^T + alpha I + lambda 1 1^T          (Eq. 2)

The theoretical justification appears in Appendix 3.2.1 ("The Spectral
Initialization", page 20). Under Assumption 2 (realizability) and the
generative process of Section 3.2, the spectral estimator's sign vector
agrees with the global optimum's sign vector up to a bounded fraction of
entries (Lemma 4, page 20).

Reference
---------
Lu, Y., Li, J., Ying, L., & Blanchet, J. (2022).
"Synthetic Principal Component Design: Fast Covariate Balancing with
Synthetic Controls." arXiv:2211.15241v1.
"""

from __future__ import annotations

import numpy as np


def spectral_initialization(M: np.ndarray) -> np.ndarray:
    """Compute the spectral-initialization sign vector for SPCD.

    Implements the "Spectral Initialization" step from Algorithm 1
    (page 7) and Algorithm 2 (page 13) of the paper:

        y^0 = sgn(v),    v = smallest eigenvector of M.

    Parameters
    ----------
    M : np.ndarray
        N x N symmetric positive-definite iteration matrix from Eq. (2).

    Returns
    -------
    y0 : np.ndarray
        Length-N sign vector with entries in ``{-1, +1}``.

    Notes
    -----
    ``np.linalg.eigh`` returns eigenvalues in ascending order, so column
    zero of the eigenvector matrix is the eigenvector associated with
    the smallest eigenvalue. Any zero entries in ``sgn(v)`` are mapped
    to ``+1`` so that ``y^0`` is strictly in ``{-1, +1}^N``.
    """

    eigvals, eigvecs = np.linalg.eigh(M)
    v = eigvecs[:, 0]

    y0 = np.sign(v)
    y0[y0 == 0] = 1.0
    return y0.astype(float)
