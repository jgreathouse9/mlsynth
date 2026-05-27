"""Data-generating processes for the SI simulation studies.

Self-contained reimplementations of the two low-rank DGPs in Agarwal, Shah &
Shen (2026), so the consistency (Section 5.1) and inference (Section 5.2)
studies can be replicated without the authors' external code:

* :func:`generate_low_rank_matrix` -- the inference-study DGP: post-period
  time-intervention factors are projected onto the *pre-period* factor span, so
  the target's signal is recoverable from the donor pool (used to measure CI
  coverage).
* :func:`generate_low_rank_matrices` -- the consistency-study DGP: returns an
  in-span (``A8 holds``) and an out-of-span (``A8 fails``) post-period, to show
  SI-PCR is consistent only when the rank condition holds.

In both, the target unit's loading is forced into the convex/linear span of the
donor loadings (Assumption 4).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _target_in_span(N: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """Unit factors ``(N, r)`` with the last (target) row in the donors' span."""
    V_core = rng.standard_normal((N - 1, r))
    coeffs = rng.standard_normal(N - 1)
    coeffs /= np.linalg.norm(coeffs)
    V_target = coeffs @ V_core
    return np.vstack([V_core, V_target])


def generate_low_rank_matrix(
    N: int, T0: int, T1: int, r: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Inference-study DGP (paper Section 5.2).

    Builds an ``(T0 + T1, N)`` expected-outcome matrix whose post-period
    time-intervention factors lie in the pre-period factor span (so the target
    is recoverable). The last column is the target unit.

    Returns
    -------
    np.ndarray
        Expected outcomes ``A``, shape ``(T0 + T1, N)``.
    """
    rng = rng or np.random.default_rng()
    V = _target_in_span(N, r, rng)
    U_core = rng.standard_normal((T0, r))
    P = np.linalg.pinv(U_core) @ U_core
    U_post = rng.uniform(0, 1, size=(T1, r)) @ P
    U = np.vstack([U_core, U_post])
    return U @ V.T


def generate_low_rank_matrices(
    N: int, T0: int, T1: int, r: int, r_pre: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Consistency-study DGP (paper Section 5.1).

    Returns two expected-outcome matrices that share the pre-period but differ
    post-period: ``A_in`` projects the post-period factors onto the pre-period
    span (rank condition holds, ``A8 holds``) and ``A_out`` onto its orthogonal
    complement (rank condition fails, ``A8 fails``).

    Returns
    -------
    (A_in, A_out) : tuple of np.ndarray
        Each shape ``(T0 + T1, N)``.
    """
    rng = rng or np.random.default_rng()
    V = _target_in_span(N, r, rng)
    A = rng.standard_normal((T0, r_pre))
    B = rng.standard_normal((r, r_pre))
    U_core = A @ B.T
    P = np.linalg.pinv(U_core) @ U_core
    P_perp = np.eye(r) - P
    U_post = rng.uniform(0, 1, size=(T1, r))
    U_in = np.vstack([U_core, U_post @ P])
    U_out = np.vstack([U_core, U_post @ P_perp])
    return U_in @ V.T, U_out @ V.T
