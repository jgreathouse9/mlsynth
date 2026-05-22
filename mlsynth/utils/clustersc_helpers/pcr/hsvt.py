"""HSVT (Hard Singular Value Thresholding) primitives for PCR-SC.

Implements rank selection and the rank-:math:`r` truncation step
:math:`\\widetilde{M} = HSVT_r(X)` from Rho, Tang, Bergam, Cummings &
Misra (2025), *ClusterSC: Advancing Synthetic Control with Donor
Selection*, Algorithm 2.

Three rank-selection modes are exposed:

* ``"cumvar"`` -- smallest :math:`r` whose cumulative spectral energy
  reaches a user threshold (paper's empirical default of 95%, Section 6.1).
* ``"fixed"`` -- caller supplies the explicit rank :math:`r`.
* ``"usvt"`` -- Universal Singular Value Thresholding (Chatterjee 2015;
  Donoho & Gavish 2014). Preserved for back-compat with the legacy
  Amjad-Shah-Shen 2018 path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ....exceptions import MlsynthConfigError, MlsynthEstimationError


def select_rank(
    X: np.ndarray,
    method: str = "cumvar",
    cumvar_threshold: float = 0.95,
    r: Optional[int] = None,
) -> int:
    """Pick a truncation rank for ``X`` under the chosen rule.

    Parameters
    ----------
    X : np.ndarray
        Matrix to decompose, shape ``(m, n)``.
    method : {"cumvar", "fixed", "usvt"}
        Rank-selection rule.
    cumvar_threshold : float
        Cumulative-variance target in ``(0, 1]`` for ``method="cumvar"``.
    r : int, optional
        Explicit rank for ``method="fixed"``.

    Returns
    -------
    int
        Selected rank :math:`r \\in [1, \\min(m, n)]`.
    """
    if X.ndim != 2:
        raise MlsynthEstimationError("HSVT input must be a 2D matrix.")
    m, n = X.shape
    rank_cap = max(1, min(m, n))

    if method == "fixed":
        if r is None or r < 1:
            raise MlsynthConfigError(
                "rank_method='fixed' requires a positive integer `rank`."
            )
        return int(min(r, rank_cap))

    if method == "cumvar":
        if not (0.0 < cumvar_threshold <= 1.0):
            raise MlsynthConfigError(
                "cumvar_threshold must lie in (0, 1]."
            )
        s = np.linalg.svd(X, compute_uv=False)
        energy = s ** 2
        total = float(energy.sum())
        if total <= 0.0:
            return 1
        cum = np.cumsum(energy) / total
        # smallest r with cum[r-1] >= threshold
        idx = int(np.searchsorted(cum, cumvar_threshold) + 1)
        return max(1, min(idx, rank_cap))

    if method == "usvt":
        # Universal SVT threshold (Chatterjee 2015; Donoho-Gavish 2014):
        # keep singular values exceeding 2.858 * median when the matrix
        # is square-ish, with the well-known (m/n)-dependent constant.
        s = np.linalg.svd(X, compute_uv=False)
        beta = min(m, n) / max(m, n)
        omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.43 + 1.82 * beta
        threshold = omega * np.median(s)
        r_usvt = int(np.sum(s > threshold))
        return max(1, min(r_usvt, rank_cap))

    raise MlsynthConfigError(
        f"Unknown rank_method {method!r}; expected 'cumvar', 'fixed', or 'usvt'."
    )


def hsvt(
    X: np.ndarray,
    rank: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply hard rank-``rank`` truncation to ``X``.

    Computes :math:`\\widetilde{M} = \\sum_{i=1}^{r} \\sigma_i u_i v_i^\\top`
    (Algorithm 2 step 2 of Rho et al. 2025).

    Returns
    -------
    M_hat : np.ndarray
        Rank-``rank`` reconstruction, shape ``(m, n)``.
    U_r : np.ndarray
        Truncated left singular vectors, shape ``(m, rank)``.
    s_r : np.ndarray
        Truncated singular values, shape ``(rank,)``.
    Vt_r : np.ndarray
        Truncated right singular vectors (transposed), shape ``(rank, n)``.
    """
    if X.ndim != 2:
        raise MlsynthEstimationError("HSVT input must be a 2D matrix.")
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = max(1, min(int(rank), s.size))
    U_r = U[:, :r]
    s_r = s[:r]
    Vt_r = Vt[:r, :]
    M_hat = (U_r * s_r) @ Vt_r
    return M_hat, U_r, s_r, Vt_r
