"""Numerical kernels for CAST (Xia, Yan & Wainwright 2025, arXiv:2412.09482).

A staggered-adoption panel, after sorting units by how long they are treated,
takes a *staircase* shape: the observed (control) cells form a descending
staircase and the counterfactual cells sit above it. The paper's insight is
that such a staircase decomposes into a collection of *four-block* problems

.. math::

   Y = \\begin{pmatrix} Y_a & Y_b \\\\ Y_c & ? \\end{pmatrix},

each of which is solved by a spectral routine (:func:`four_block_est`,
Algorithm 1) that estimates the column subspace from the left blocks, denoises
the upper blocks, and regresses one onto the other to impute ``?``. The same
subspaces yield a closed-form entrywise variance (:func:`entrywise_variance`,
equations 6a-6c), which is what makes per-cell confidence intervals possible.

* :func:`four_block_est` -- Algorithm 1 (FourBlockEst), paper S3.1.1.
* :func:`staircase_blocks` -- the partition of Appendix A.
* :func:`entrywise_variance` -- equations (6a) and (6c).
"""
from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError


def svd_r(M: np.ndarray, r: int):
    """Truncated rank-``r`` SVD, returning ``(U, s, Vh)``."""
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:                                # pragma: no cover - guarded by callers
        raise MlsynthConfigError("svd_r: M must be a 2D array.")
    if not (1 <= r <= min(M.shape)):
        raise MlsynthConfigError(
            f"svd_r: require 1 <= r <= min(N, T) = {min(M.shape)}; got r={r}."
        )
    u, s, vh = np.linalg.svd(M, full_matrices=False)
    return u[:, :r], s[:r], vh[:r, :]


def four_block_est(Y: np.ndarray, N1: int, T1: int, r: int) -> np.ndarray:
    """Algorithm 1 (FourBlockEst): impute the unobserved bottom-right block.

    Parameters
    ----------
    Y : np.ndarray
        ``(N, T)`` panel laid out as ``[[Ya, Yb], [Yc, ?]]``; the bottom-right
        block is unobserved and its content is ignored.
    N1, T1 : int
        Row/column size of the fully observed top-left block ``Ya``.
    r : int
        Rank of the underlying mean matrix.

    Returns
    -------
    np.ndarray
        ``(N - N1, T - T1)`` estimate of the missing block.

    Notes
    -----
    With a rank-``r`` mean matrix and noiseless observations the output equals
    the missing block exactly (paper S3.1.1).
    """
    Y = np.asarray(Y, dtype=float)
    N, T = Y.shape
    if not (r <= N1 < N) or not (r <= T1 < T):
        raise MlsynthConfigError(
            f"four_block_est: require r <= N1 < N and r <= T1 < T; got "
            f"r={r}, N1={N1}, N={N}, T1={T1}, T={T}."
        )
    # Step 1: left subspace, from the fully observed left blocks [Ya; Yc].
    U_left, _, _ = svd_r(Y[:, :T1], r)
    U1, U2 = U_left[:N1], U_left[N1:]
    # Step 2-3: denoise the upper blocks [Ya Yb] and read off M_b.
    U_up, s_up, Vh_up = svd_r(Y[:N1, :], r)
    Mb = (U_up * s_up) @ Vh_up[:, T1:]
    # Step 4: impute  M_d = U2 (U1'U1)^{-1} U1' M_b.
    try:
        return U2 @ np.linalg.solve(U1.T @ U1, U1.T @ Mb)
    except np.linalg.LinAlgError as exc:          # pragma: no cover - guarded above
        raise MlsynthConfigError(
            f"four_block_est: the top-left block is rank deficient ({exc})."
        ) from exc


def staircase_blocks(A: np.ndarray):
    """Sort a staggered design into staircase form (paper Appendix A).

    Parameters
    ----------
    A : np.ndarray
        ``(N, T)`` indicator, 1 where the outcome is observed under control and
        0 where it is an unobserved counterfactual.

    Returns
    -------
    (order, inv, N_all, T_all, k)
        ``order`` sorts units by ascending treatment duration (so the
        longest-observed units come first); ``inv`` restores the original row
        order. ``T_all`` holds the distinct control-run lengths in ascending
        order, ``N_all[i]`` the cumulative unit count down to the ``i``-th
        longest, and ``k`` the number of distinct blocks.
    """
    A = np.asarray(A)
    if A.ndim != 2:                                # pragma: no cover - guarded by setup
        raise MlsynthDataError("staircase_blocks: A must be a 2D array.")
    control_len = A.sum(axis=1).astype(int)
    if not (A == 0).any():
        raise MlsynthDataError(
            "staircase_blocks: no unobserved (treated) cells -- nothing to impute."
        )
    if not (control_len == A.shape[1]).any():
        raise MlsynthDataError(
            "staircase_blocks: CAST needs at least one never-treated unit to "
            "anchor the staircase."
        )
    order = np.argsort(A.shape[1] - control_len, kind="stable")
    inv = np.argsort(order)
    T_all = np.sort(np.unique(control_len)).astype(int)
    k = int(T_all.size)
    N_all = np.zeros(k, dtype=int)
    for i in range(k):
        N_all[i] = int((control_len == T_all[k - 1 - i]).sum()) + (N_all[i - 1] if i else 0)
    return order, inv, N_all, T_all, k


def _blocks(N_all, T_all, k):
    """Yield the four-block sub-problems covering every unobserved cell."""
    for i in range(k):
        for j in range(k - i, k):
            N1, T1 = int(N_all[k - j - 1]), int(T_all[k - i - 1])
            Nij, Tij = int(N_all[i]), int(T_all[j])
            r0, r1 = int(N_all[i - 1]) if i else 0, int(N_all[i])
            c0, c1 = int(T_all[j - 1]) if j else 0, int(T_all[j])
            yield N1, T1, Nij, Tij, r0, r1, c0, c1


def impute_missing(Y: np.ndarray, A: np.ndarray, r: int) -> np.ndarray:
    """Estimate every unobserved cell (Algorithm 3, estimation half).

    Returns an ``(N, T)`` matrix carrying the counterfactual estimates at the
    unobserved cells and zero elsewhere, in the original row order.
    """
    order, inv, N_all, T_all, k = staircase_blocks(A)
    Ys = np.asarray(Y, dtype=float)[order]
    M_est = np.zeros_like(Ys)
    for N1, T1, Nij, Tij, r0, r1, c0, c1 in _blocks(N_all, T_all, k):
        Md = four_block_est(Ys[:Nij, :Tij], N1, T1, r)
        M_est[r0:r1, c0:c1] = Md[r0 - N1:r1 - N1, c0 - T1:c1 - T1]
    return M_est[inv]


def entrywise_variance(Y: np.ndarray, A: np.ndarray, r: int) -> np.ndarray:
    """Entrywise variance of the imputed cells (paper equations 6a and 6c).

    The residuals are formed on the observed entries of the *sorted* staircase
    (eq. 6a) and combined with the estimated subspaces to give, for each
    unobserved ``(i, t)``, the variance of the imputation error (eq. 6c).

    Returns an ``(N, T)`` array of variances in the original row order; entries
    at observed cells are zero.

    Notes
    -----
    The mask is applied in the sorted frame, as eq. (6a) specifies. The
    authors' ``CAST-panel`` package (1.0.2) masks sorted residuals with the
    *unsorted* indicator, which understates the standard errors whenever the
    sort permutation is non-trivial; see ``docs/replications/cast.rst``.
    """
    order, inv, N_all, T_all, k = staircase_blocks(A)
    Ys = np.asarray(Y, dtype=float)[order]
    As = np.asarray(A)[order]

    M_imp = impute_missing(Y, A, r)[order] + Ys * As
    u, s, vh = svd_r(M_imp, r)
    E = (Ys - (u * s) @ vh) * As                  # eq. (6a), sorted frame
    U_hat, V_hat = u, vh.T

    var = np.zeros_like(Ys)
    for N1, T1, Nij, Tij, r0, r1, c0, c1 in _blocks(N_all, T_all, k):
        U1, U2 = U_hat[:N1], U_hat[N1:Nij]
        V1, V2 = V_hat[:T1], V_hat[T1:Tij]
        PU = np.square(U2 @ np.linalg.solve(U1.T @ U1, U1.T))
        PV = np.square(V2 @ np.linalg.solve(V1.T @ V1, V1.T))
        v = PU @ np.square(E[:N1, T1:Tij]) + np.square(E[N1:Nij, :T1]) @ PV.T
        var[r0:r1, c0:c1] = v[r0 - N1:r1 - N1, c0 - T1:c1 - T1]
    return var[inv]
