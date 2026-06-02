"""The inclusive-SCM cross-weight system (Di Stefano & Mellace 2024, eq. 6).

When affected units stay in each other's donor pools, the observed
synthetic-control gap for affected unit ``i`` mixes its own effect with the
effects on the other affected units it borrows from:

    gap_i = theta_i - sum_{k != i} w_i[k] * theta_k ,

where ``w_i[k]`` is the weight affected unit ``k`` receives in unit ``i``'s
synthetic control. Stacking the ``m`` affected units gives ``Omega theta =
gap`` with ``Omega[i, i] = 1`` and ``Omega[i, k] = -w_i[k]``; inverting
``Omega`` de-contaminates the post-period gaps into the true effects.
"""

from __future__ import annotations

import numpy as np


def build_omega(cross: np.ndarray) -> np.ndarray:
    """Assemble the ``m x m`` system matrix from the cross-weight matrix.

    Parameters
    ----------
    cross : np.ndarray
        ``(m, m)`` matrix whose ``[i, k]`` entry (``k != i``) is the weight
        affected unit ``k`` receives in affected unit ``i``'s synthetic
        control. The diagonal is ignored.

    Returns
    -------
    np.ndarray
        ``Omega`` with unit diagonal and ``-cross`` off the diagonal.
    """
    m = cross.shape[0]
    omega = np.eye(m)
    off = ~np.eye(m, dtype=bool)
    omega[off] = -cross[off]
    return omega


def solve_inclusive(omega: np.ndarray, gaps_affected_post: np.ndarray) -> np.ndarray:
    """De-contaminate the post-treatment gaps via ``Omega^{-1}``.

    Parameters
    ----------
    omega : np.ndarray
        ``(m, m)`` system matrix from :func:`build_omega`.
    gaps_affected_post : np.ndarray
        ``(m, T1)`` raw post-period SC gaps for the affected set (treated
        unit first).

    Returns
    -------
    np.ndarray
        ``(m, T1)`` de-contaminated effects; row 0 is the treated unit.
    """
    return np.linalg.solve(omega, gaps_affected_post)
