"""Numerical core for RMSI (Agarwal, Choi & Yuan 2026).

Implements the four-component sieve + nuclear-norm estimator (their Algorithm 1)
and the block-missing causal extension (their Algorithm 3). Both reduce to
projections and singular-value soft-thresholding -- no iterative solver.

Algorithm 1 decomposes ``M = M1 + M2 + M3 + M4``:

* ``M1 = PX Y PZ``                     -- explained by both row & column features
* ``M2 = svt(PX Y (I-PZ), nu2/2)``     -- row-feature-driven
* ``M3 = svt((I-PX) Y PZ, nu3/2)``     -- column-feature-driven
* ``M4 = svt((I-PX) Y (I-PZ), nu4/2)`` -- residual low-rank (no features)

where ``PX``/``PZ`` project onto sieve bases of the row/column covariates, the
penalised least-squares ``argmin ||B - A||_F^2 + nu||A||_*`` has the closed form
``svt(B, nu/2)``, and ``nu2 = C2 sqrt(T)``, ``nu3 = C3 sqrt(N)``,
``nu4 = C4 (sqrt(N) + sqrt(T))``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def sieve_poly(C: np.ndarray, J: int = 2) -> np.ndarray:
    """Polynomial sieve basis (with intercept) of a covariate matrix.

    For ``J = 2`` each covariate ``c`` contributes ``c`` and ``c**2``; a constant
    column is prepended so the projection captures means.

    Parameters
    ----------
    C : np.ndarray, shape (n, d)
        Covariate matrix (rows are units/periods).
    J : int
        Polynomial sieve order.

    Returns
    -------
    np.ndarray, shape (n, 1 + d*J)
        Basis matrix ``[1, c_1, c_1^2, ..., c_d, c_d^2, ...]``.
    """
    C = np.asarray(C, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    n, d = C.shape
    cols = [np.ones((n, 1))]
    for j in range(d):
        for p in range(1, J + 1):
            cols.append(C[:, j:j + 1] ** p)
    return np.hstack(cols)


def _proj(Phi: np.ndarray) -> np.ndarray:
    """Orthogonal projector ``Phi (Phi'Phi)^{-1} Phi'`` (pseudo-inverse form)."""
    return Phi @ np.linalg.pinv(Phi)


def _svt(B: np.ndarray, thr: float) -> np.ndarray:
    """Singular-value soft-thresholding: prox of ``thr * ||.||_*``."""
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    s = np.maximum(s - thr, 0.0)
    return (U * s) @ Vt


def _auto_rank(s: np.ndarray, rel_thr: float = 0.05) -> int:
    """Default factor rank: number of singular values above ``rel_thr * max``.

    A relative-magnitude threshold is used in preference to the Ahn-Horenstein
    eigenvalue-ratio because these panels are typically *level-dominated* (a
    single huge leading singular value), which collapses ratio-based estimators
    to rank one and discards the genuine factor structure the recombination
    needs. ``rel_thr = 0.05`` retains the components carrying material signal.
    """
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return 1
    return int(max(1, np.sum(s > rel_thr * s[0])))


def algorithm1(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, *, J: int = 2,
               C2: float = 1.0, C3: float = 1.0, C4: float = 1.0,
               PX: Optional[np.ndarray] = None, PZ: Optional[np.ndarray] = None):
    """Four-component sieve + SVT estimator (Algorithm 1) for a full matrix.

    Parameters
    ----------
    Y : np.ndarray, shape (N, T)
        Fully observed outcome matrix.
    X : np.ndarray, shape (N, d1)
        Row (unit) covariates.
    Z : np.ndarray, shape (T, d2)
        Column (time) covariates.
    J : int
        Sieve order.
    C2, C3, C4 : float
        Penalty constants for the three soft-thresholded components.
    PX, PZ : np.ndarray, optional
        Precomputed projectors (built from the sieve bases if omitted).

    Returns
    -------
    M_hat : np.ndarray, shape (N, T)
        The aggregated estimate ``M1 + M2 + M3 + M4``.
    components : dict
        ``{"M1", "M2", "M3", "M4"}`` -- the individual components.
    """
    Y = np.asarray(Y, dtype=float)
    N, T = Y.shape
    if PX is None:
        PX = _proj(sieve_poly(X, J))
    if PZ is None:
        PZ = _proj(sieve_poly(Z, J))
    IN, IT = np.eye(N), np.eye(T)
    M1 = PX @ Y @ PZ
    M2 = _svt(PX @ Y @ (IT - PZ), C2 * np.sqrt(T) / 2.0)
    M3 = _svt((IN - PX) @ Y @ PZ, C3 * np.sqrt(N) / 2.0)
    M4 = _svt((IN - PX) @ Y @ (IT - PZ), C4 * (np.sqrt(N) + np.sqrt(T)) / 2.0)
    return M1 + M2 + M3 + M4, {"M1": M1, "M2": M2, "M3": M3, "M4": M4}


def algorithm3(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, *,
               control_idx: np.ndarray, T0: int, J: int = 2,
               rank: Optional[int] = None, C2: float = 1.0, C3: float = 1.0,
               C4: float = 1.0):
    """Block-missing causal estimator (Algorithm 3).

    Imputes the full matrix from the "tall" submatrix (all units, pre-treatment
    periods) and the "wide" submatrix (control units, all periods), recombining
    via ``M_hat = U_tall H_adj D_wide V_wide'`` with a rotation ``H_adj`` that
    aligns the wide left-singular vectors to the tall ones on the control rows.

    Parameters
    ----------
    Y : np.ndarray, shape (N, T)
        Outcome matrix; the treated post-treatment block is treated as missing.
    X, Z : np.ndarray
        Row and column covariates.
    control_idx : np.ndarray
        Indices of never-treated (control) units (the "wide" rows).
    T0 : int
        Number of clean pre-treatment periods (the "tall" columns).
    J : int
        Sieve order.
    rank : int, optional
        Factor rank ``K``; a relative-magnitude singular-value threshold
        (:func:`_auto_rank`) is used when omitted.
    C2, C3, C4 : float
        Penalty constants forwarded to :func:`algorithm1`.

    Returns
    -------
    M_hat : np.ndarray, shape (N, T)
        The completed matrix (the imputed counterfactual everywhere).
    rank : int
        The factor rank used.
    """
    Y = np.asarray(Y, dtype=float)
    N, T = Y.shape
    control_idx = np.asarray(control_idx, dtype=int)

    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]

    # "tall": all units, clean pre-period.   "wide": control units, all periods.
    M_tall, _ = algorithm1(Y[:, :T0], X, Z[:T0], J=J, C2=C2, C3=C3, C4=C4)
    M_wide, _ = algorithm1(Y[np.ix_(control_idx, np.arange(T))], X[control_idx],
                           Z, J=J, C2=C2, C3=C3, C4=C4)

    Ut, st, _ = np.linalg.svd(M_tall, full_matrices=False)
    Uw, sw, Vwt = np.linalg.svd(M_wide, full_matrices=False)
    if rank is None:
        rank = max(_auto_rank(st), _auto_rank(sw))
    rank = int(min(rank, st.size, sw.size))

    Ut = Ut[:, :rank]
    Uw = Uw[:, :rank]
    Vw = Vwt[:rank].T
    Dw = np.diag(sw[:rank])

    # Rotation: align wide left-vectors to tall left-vectors on the control rows.
    H_adj = np.linalg.pinv(Ut[control_idx]) @ Uw
    M_hat = Ut @ H_adj @ Dw @ Vw.T
    return M_hat, int(rank)
