"""SI-PCR estimation math (Agarwal, Shah & Shen 2026).

Two estimators are implemented on top of the shared HSVT primitives in
:mod:`mlsynth.utils.clustersc_helpers.pcr.hsvt`:

* :func:`si_pcr_weights` -- the plain SI-PCR weights (paper eq. 10): regress the
  target's pre-period control outcomes onto the rank-``k`` denoised donor pool.
* :func:`bias_corrected_fit` -- the bias-corrected SI-PCR estimator (Section
  4.3): restrict to a rank-complete donor subset ``Omega`` (``|Omega| = k``),
  fit weights by the pseudo-inverse (eq. 12), and return the noise-variance
  estimate (eq. 14) and weight norm needed for the asymptotic-normality CI
  (eq. 13).

All routines take the *pre-treatment* donor matrix (under control) and the
target's pre-treatment outcomes; the post-period prediction (applying the
weights to donor outcomes under the intervention) lives in the orchestrator.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.linalg import qr

from ..clustersc_helpers.pcr.hsvt import hsvt, select_rank


def donoho_rank(s: np.ndarray, ratio: float) -> int:
    """Gavish-Donoho (2014) optimal hard threshold, as applied by Agarwal-Shah-Shen.

    The authors evaluate the :math:`\\omega(\\beta)` approximation at
    ``ratio = m / n`` -- the donor pre-matrix's rows-over-columns
    (:math:`T_0 / N_d`) -- rather than the canonical ``min/max`` aspect ratio.
    Reproduced here verbatim so SI matches the paper's reported ranks; see
    ``rank_method="donoho"``.
    """
    omega = 0.56 * ratio ** 3 - 0.95 * ratio ** 2 + 1.43 + 1.82 * ratio
    t = omega * np.median(s)
    return max(int(np.sum(s > t)), 1)


def si_pcr_weights(
    donor_pre: np.ndarray,
    target_pre: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Plain SI-PCR donor weights over the full pool (paper eq. 10).

    ``w_hat = (sum_{l<=k} (1/s_l) v_l u_l^T) y_pre,i``, i.e. regress the target's
    pre-period outcomes onto the top-``rank`` principal subspace of the donor
    pre-matrix.

    Parameters
    ----------
    donor_pre : np.ndarray
        Donor pre-treatment (control) outcomes, shape ``(T0, Nd)``.
    target_pre : np.ndarray
        Target pre-treatment (control) outcomes, shape ``(T0,)``.
    rank : int
        Spectral truncation rank ``k``.

    Returns
    -------
    np.ndarray
        Donor weight vector, shape ``(Nd,)``.
    """
    target_pre = np.ravel(target_pre)
    _, U_r, s_r, Vt_r = hsvt(donor_pre, rank)
    # w = V_k diag(1/s) U_k^T y
    return Vt_r.T @ ((U_r.T @ target_pre) / s_r)


def select_omega(donor_pre: np.ndarray, rank: int) -> List[int]:
    """Pick a rank-complete donor subset ``Omega`` (``|Omega| = k``).

    Selects ``rank`` columns of the rank-``k`` denoised donor matrix that are
    linearly independent (full column rank ``k``) via column-pivoted QR -- the
    pivots are the most independent columns, which is the structured model
    selection the bias-corrected estimator relies on (Section 4.3).

    Parameters
    ----------
    donor_pre : np.ndarray
        Donor pre-treatment outcomes, shape ``(T0, Nd)``.
    rank : int
        Number of donors to retain (``k``).

    Returns
    -------
    list of int
        Column indices of the selected donors (length ``min(rank, Nd)``).
    """
    M_hat, _, _, _ = hsvt(donor_pre, rank)
    k = max(1, min(int(rank), M_hat.shape[1]))
    # column-pivoted QR: the first k pivots are a rank-complete column subset
    _, _, piv = qr(M_hat, mode="economic", pivoting=True)
    return sorted(int(i) for i in piv[:k])


def bias_corrected_fit(
    donor_pre: np.ndarray,
    target_pre: np.ndarray,
    rank: int,
) -> Tuple[List[int], np.ndarray, float]:
    """Bias-corrected SI-PCR fit on a rank-complete subset (Section 4.3).

    Restricts to a rank-complete donor subset ``Omega`` and fits weights by the
    pseudo-inverse of the denoised pre-matrix (eq. 12), then estimates the noise
    variance from the target's residual against the rank-``k`` donor subspace
    (eq. 14).

    Parameters
    ----------
    donor_pre : np.ndarray
        Donor pre-treatment (control) outcomes, shape ``(T0, Nd)``.
    target_pre : np.ndarray
        Target pre-treatment (control) outcomes, shape ``(T0,)``.
    rank : int
        Spectral rank ``k`` (also ``|Omega|``).

    Returns
    -------
    omega : list of int
        Indices of the active donor subset.
    w_omega : np.ndarray
        Bias-corrected weights on ``omega``, shape ``(|Omega|,)``.
    sigma_hat : float
        Estimated noise standard deviation (square root of eq. 14).
    """
    target_pre = np.ravel(target_pre)
    M_hat, U_r, _, _ = hsvt(donor_pre, rank)
    k = U_r.shape[1]
    omega = select_omega(donor_pre, k)

    # eq. 12: w_hat(i, d, Omega) = (Y^k_{pre, Omega})^+ y_pre,i
    Yk_omega = M_hat[:, omega]
    w_omega = np.linalg.pinv(Yk_omega) @ target_pre

    # eq. 14: sigma^2 = ||(I - U_k U_k^T) y_pre,i||^2 / (T0 - k)
    resid = target_pre - U_r @ (U_r.T @ target_pre)
    T0 = donor_pre.shape[0]
    denom = max(T0 - k, 1)
    sigma2 = float(np.linalg.norm(resid) ** 2) / denom
    return omega, w_omega, float(np.sqrt(max(sigma2, 0.0)))


def resolve_rank(
    donor_pre: np.ndarray,
    rank_method: str,
    rank: int = None,
    cumvar_threshold: float = 0.95,
) -> int:
    """Resolve the spectral rank ``k`` for a donor pre-matrix.

    ``"donoho"`` (the SI default) reproduces Agarwal-Shah-Shen's exact rank rule
    (:func:`donoho_rank` with ``ratio = T0 / Nd``). The remaining modes delegate
    to :func:`mlsynth.utils.clustersc_helpers.pcr.hsvt.select_rank` so SI shares
    ClusterSC's HSVT machinery (``"usvt"`` is the same threshold evaluated at the
    canonical ``min/max`` aspect ratio, ``"cumvar"`` / ``"fixed"`` as in HSVT).
    """
    if rank_method == "donoho":
        m, n = donor_pre.shape
        s = np.linalg.svd(donor_pre, compute_uv=False)
        return max(1, min(donoho_rank(s, ratio=m / n), min(m, n)))
    return select_rank(
        donor_pre,
        method=rank_method,
        cumvar_threshold=cumvar_threshold,
        r=rank,
    )


def variance_estimation(
    U_k: np.ndarray,
    V_k: np.ndarray,
    target_pre: np.ndarray,
    donor_post: np.ndarray,
) -> Tuple[float, float, float]:
    """Noise-standard-deviation estimates (Agarwal-Shah-Shen ``inference.py``).

    Returns ``(double, units, time_iv)``:

    * ``units`` -- the main-text estimator (eq. 14): residual of the target's
      pre-period against the donor left-singular subspace, over ``T0 - k``.
    * ``time_iv`` -- the donor post-period residual against the right-singular
      subspace, over ``T1 (Nd - k)``.
    * ``double`` -- the degrees-of-freedom-weighted combination of the two
      (the estimator the paper's code uses for its intervals).

    Parameters
    ----------
    U_k : np.ndarray
        Left singular vectors of the rank-``k`` donor pre-matrix, shape
        ``(T0, k)``.
    V_k : np.ndarray
        Right singular vectors, shape ``(Nd, k)``.
    target_pre : np.ndarray
        Target pre-period outcomes, shape ``(T0,)``.
    donor_post : np.ndarray
        Donor post-period outcomes, shape ``(T1, Nd)``.
    """
    target_pre = np.ravel(target_pre)
    T0 = U_k.shape[0]
    T1, N = donor_post.shape
    k = U_k.shape[1]

    df1 = max(T0 - k, 1)
    resid1 = (np.eye(T0) - U_k @ U_k.T) @ target_pre
    var1 = float(np.linalg.norm(resid1) ** 2) / df1

    df2 = max(T1 * (N - k), 1)
    resid2 = (np.eye(N) - V_k @ V_k.T) @ donor_post.T
    var2 = float(np.linalg.norm(resid2, "fro") ** 2) / df2

    df = df1 + df2
    var = (df2 / df) * var1 + (df1 / df) * var2
    return float(np.sqrt(var)), float(np.sqrt(var1)), float(np.sqrt(var2))
