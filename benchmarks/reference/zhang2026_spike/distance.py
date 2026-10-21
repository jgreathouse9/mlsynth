"""Pairwise pseudo-distances on a block of pre-treatment outcomes.

Both variants reduce the same vector. With the Gram matrix G = A A' / p,

    (1/p) sum_t (Y_k1t - Y_k2t)(Y_it - Y_jt) = v_k1 - v_k2,
    v_k := G[k, i] - G[k, j]

so the paper's max over PAIRS (k1, k2) -- which reads as O(n^4 p) -- is the
RANGE of v, an O(n) reduction, and Feng's max over a single unit is max |v|:

    zhang_range   d_ij = max_k v_k - min_k v_k      (Zhang 2026 eq. 3.4)
    feng_maxabs   d_ij = max_k |v_k|                (Feng 2024; mlsynth LPCA)

Total cost is O(n^2 p + n^3) for either, the shape already in
``mlsynth.utils.lpca_helpers.core.pseudo_max_distance``.

The range is exactly invariant to an additive common time trend: adding lambda_t
to every unit leaves both cross-sectional differences untouched. Feng's version
contracts against a LEVEL x_l, which carries the trend, so it is not.
"""

from __future__ import annotations

import numpy as np

METRICS = ("zhang_range", "feng_maxabs")


def pseudo_distance(block: np.ndarray, metric: str = "zhang_range") -> np.ndarray:
    """Symmetric ``(n, n)`` pseudo-distances, zero diagonal.

    Parameters
    ----------
    block : (n, p) float
        Pre-treatment outcomes, units by periods.
    metric : {"zhang_range", "feng_maxabs"}

    Notes
    -----
    ``k1, k2 != i, j`` is enforced by masking row ``i`` and the diagonal of the
    working matrix, which are exactly the excluded ``k``. Masking is done in
    place with +/-inf, so the reduction costs no copies.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    A = np.asarray(block, dtype=float)
    n = A.shape[0]
    gram = A @ A.T / A.shape[1]
    out = np.empty((n, n))

    for i in range(n):
        # M[k, j] = G[k, j] - G[k, i] = -v_k(j); negation preserves both the
        # range and the max absolute value, so no sign fix is needed.
        M = gram - gram[i][:, None]
        if metric == "feng_maxabs":
            np.abs(M, out=M)
            M[i, :] = -np.inf
            np.fill_diagonal(M, -np.inf)
            out[i] = M.max(axis=0)
        else:
            M[i, :] = -np.inf
            np.fill_diagonal(M, -np.inf)
            high = M.max(axis=0)
            M[i, :] = np.inf
            np.fill_diagonal(M, np.inf)
            out[i] = high - M.min(axis=0)

    np.fill_diagonal(out, 0.0)
    return out
