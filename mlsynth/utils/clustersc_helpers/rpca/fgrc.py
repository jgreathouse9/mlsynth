"""Functional Generalized Reduced Clustering (fGRC) for RPCA-SC donor selection.

Pure-Python port of Yamamoto & Hwang (2017), *Dimension-Reduced Clustering of
Functional Data via Subspace Separation* (Journal of Classification 34:294-326),
the functional sibling of their 2014 generalized reduced clustering.

Where the paper's silhouette-driven :math:`k`-means (and the FPCA + k-means used
in :mod:`.clustering`) cluster on the *max-variance* subspace, fGRC finds the
subspace that best *reveals* the cluster structure while projecting out a
separate "disturbing" subspace -- a shared, high-variance direction (a common
level/trend) that would otherwise mask the grouping. Concretely it minimises,
over orthonormal loadings :math:`A = [A_1 \\mid A_2]` (cluster subspace
:math:`A_1`, disturbing subspace :math:`A_2`) and a cluster indicator
:math:`U`,

.. math::

   \\| X - X A A^\\top \\|_F^2
   + \\rho_1 \\| (I - P_U) X A_1 \\|_F^2
   + \\rho_2 \\| P_U X A_1 \\|_F^2,
   \\qquad P_U = U (U^\\top U)^{-1} U^\\top,

by alternating a Lloyd :math:`k`-means U-update on :math:`X A_1` with an
A-update that is a closed-form eigendecomposition when there is no disturbing
subspace (:math:`c_2 = 0`) and a Stiefel gradient-projection (polar-retraction)
step otherwise. The functional part is a B-spline basis expansion of each unit's
trajectory feeding a Gram-weighted coefficient matrix into that core.

This module is a faithful transcription of the reference ``OptimGRC_C`` C
routine; the layer-by-layer equivalence (basis, loss, gradient, A-update, ALS,
end-to-end pool and ATT) is pinned against the compiled reference in
``mlsynth/tests/test_fgrc.py``.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from ....exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError


# --------------------------------------------------------------------------
# Functional layer: B-spline basis expansion  G = coef @ H^{1/2}
# --------------------------------------------------------------------------
def basis_expand(X: np.ndarray, n_knots: int, order: int) -> np.ndarray:
    """B-spline smooth each row of ``X`` (units x time) and return the
    Gram-weighted coefficient matrix :math:`G = \\mathrm{coef}\\,H^{1/2}`
    (fGRC's ``BasisExpand`` with smoothing parameter ``lambda=0``).

    Parameters
    ----------
    X : np.ndarray
        Trajectories, shape ``(n_units, n_time)`` (rows are units).
    n_knots : int
        Number of B-spline breakpoints (``par.Bsp[1]``); must be ``>= 2``.
    order : int
        B-spline order (``par.Bsp[2]``; 4 = cubic); must be ``>= 2``.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise MlsynthDataError("basis_expand: X must be a 2D array (units x time).")
    if n_knots < 2:
        raise MlsynthConfigError("basis_expand: n_knots must be >= 2.")
    if order < 2:
        raise MlsynthConfigError("basis_expand: order must be >= 2.")
    n_time = X.shape[1]
    if n_time < order:
        raise MlsynthConfigError(
            f"basis_expand: n_time ({n_time}) must be >= order ({order})."
        )
    X = np.asarray(X, dtype=float)
    br = np.linspace(1.0, n_time, n_knots)
    knots = np.concatenate([np.repeat(br[0], order - 1), br, np.repeat(br[-1], order - 1)])
    x = np.arange(1, n_time + 1, dtype=float)
    try:
        Phi = BSpline.design_matrix(x, knots, order - 1, extrapolate=True).toarray()
        coef = X @ Phi @ np.linalg.inv(Phi.T @ Phi)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise MlsynthEstimationError(f"fGRC basis expansion failed: {exc}") from exc
    Nb = Phi.shape[1]
    w = np.ones(n_time)
    w[0] = w[-1] = 0.5                                                 # trapezoidal weights
    H = np.zeros((Nb, Nb))
    for i in range(Nb):
        for j in range(i, Nb):
            H[i, j] = H[j, i] = float(np.sum(w * Phi[:, i] * Phi[:, j]))
    ev, evec = np.linalg.eigh(H)
    H_half = (evec * np.sqrt(np.abs(ev))) @ evec.T
    return coef @ H_half


# --------------------------------------------------------------------------
# GRC core: cluster indicator, loss, Euclidean gradient
# --------------------------------------------------------------------------
def indicator(labels: np.ndarray, k: int) -> np.ndarray:
    """One-hot cluster indicator ``U`` (n_units x k) from 1-based integer labels."""
    labels = np.asarray(labels, dtype=int)
    U = np.zeros((labels.size, k))
    U[np.arange(labels.size), labels - 1] = 1.0
    return U


def _proj_U(U: np.ndarray) -> np.ndarray:
    """Projection onto the cluster indicator's column space, :math:`P_U`."""
    return U @ np.linalg.solve(U.T @ U, U.T)


def grc_loss(X, U, A, c1: int, rho1: float, rho2: float) -> float:
    """fGRC/GRC objective (matches ``OptimGRC_C`` ``f_lossfunc``)."""
    A1 = A[:, :c1]
    P = _proj_U(U)
    Z1 = X - X @ A @ A.T
    Z2 = X @ A1 - P @ (X @ A1)
    Z3 = P @ (X @ A1)
    return float(np.sum(Z1 ** 2) + rho1 * np.sum(Z2 ** 2) + rho2 * np.sum(Z3 ** 2))


def grc_grad(X, U, A, c1: int, rho1: float, rho2: float) -> np.ndarray:
    """Euclidean gradient of the Stiefel-restricted loss (matches ``f_grad``):
    :math:`-2[X^\\top M X A_1 \\mid X^\\top X A_2]`, :math:`M=(1-\\rho_1)I+(\\rho_1-\\rho_2)P_U`."""
    ncomp = A.shape[1]
    A1 = A[:, :c1]
    P = _proj_U(U)
    M = (1.0 - rho1) * np.eye(X.shape[0]) + (rho1 - rho2) * P
    G = -2.0 * (X.T @ M @ X @ A1)
    if ncomp > c1:
        A2 = A[:, c1:]
        G = np.hstack([G, -2.0 * (X.T @ X @ A2)])
    return G


# --------------------------------------------------------------------------
# A-update: eigendecomposition (c2 = 0) or gradient-projection (c2 > 0)
# --------------------------------------------------------------------------
def eigen_update(X, U, c1: int, rho1: float, rho2: float) -> np.ndarray:
    """Closed-form A-update with no disturbing subspace (``c2 = 0``): the top
    ``c1`` eigenvectors of :math:`X^\\top M X`."""
    P = _proj_U(U)
    M = (1.0 - rho1) * np.eye(X.shape[0]) + (rho1 - rho2) * P
    _ev, evec = np.linalg.eigh(X.T @ M @ X)          # ascending
    return evec[:, ::-1][:, :c1].copy()              # top c1


def gp_algorithm(X, U, A, c1: int, rho1: float, rho2: float,
                 n_ite: int = 100, n_max_alpha: int = 15, eps: float = 1e-5,
                 alpha_ini: float = 1.0) -> np.ndarray:
    """Gradient-projection A-update on the Stiefel manifold (matches C
    ``GPAlgorithm``): step :math:`A \\leftarrow \\mathrm{polar}(A - \\alpha G)`
    with backtracking :math:`\\alpha` until the loss decreases."""
    A_new = np.array(A, dtype=float, copy=True)
    it = 0
    while True:
        it += 1
        A_cur = A_new
        G = grc_grad(X, U, A_cur, c1, rho1, rho2)
        lc = grc_loss(X, U, A_cur, c1, rho1, rho2)
        n_alpha = 0
        alpha = 2.0 * alpha_ini
        ln = lc
        while True:
            n_alpha += 1
            alpha /= 2.0
            A_tar = A_cur - alpha * G
            Us, _, Vt = np.linalg.svd(A_tar, full_matrices=False)
            A_new = Us @ Vt                            # polar retraction (unique)
            ln = grc_loss(X, U, A_new, c1, rho1, rho2)
            if not (ln > lc and n_alpha < n_max_alpha):
                break
        if n_alpha == n_max_alpha:                     # no improving step found -> revert
            A_new = A_cur
            ln = lc
        if not ((lc - ln) > eps and it < n_ite):
            break
    return A_new


# --------------------------------------------------------------------------
# Lloyd k-means (faithful to the C kmeans_Lloyd) + best-of-nstart restarts
# --------------------------------------------------------------------------
def _kmeans_lloyd(F, centers, max_iter: int = 1000):
    n = F.shape[0]
    k = centers.shape[0]
    cen = np.array(centers, dtype=float, copy=True)
    cl = -np.ones(n, dtype=int)
    for _ in range(max_iter):
        d = np.sum((F[:, None, :] - cen[None, :, :]) ** 2, axis=2)     # (n, k)
        new = d.argmin(axis=1)
        if np.array_equal(new, cl):
            break
        cl = new
        for j in range(k):
            m = cl == j
            if m.any():
                cen[j] = F[m].mean(axis=0)
    wss = np.array([np.sum((F[cl == j] - cen[j]) ** 2) if np.any(cl == j) else 0.0
                    for j in range(k)])
    return cl, wss


def _kmeans_best(F, k, nstart, rng, warm_centers=None):
    """Best-WSS Lloyd over ``nstart`` random-row starts (warm start on the first).

    Partitions that collapse a cluster to empty are rejected (faithful to the C
    routine, which lands empty starts on ``nan`` WSS and never selects them).
    Returns ``None`` when every start produced an empty cluster.
    """
    best_wss = np.inf
    best_cl = None
    for r in range(nstart):
        if warm_centers is not None and r == 0:
            centers = warm_centers
        else:
            idx = rng.choice(F.shape[0], size=k, replace=False)
            centers = F[idx]
        cl, wss = _kmeans_lloyd(F, centers)
        if np.unique(cl).size < k:            # empty cluster -> reject this start
            continue
        s = float(wss.sum())
        if s < best_wss:
            best_wss = s
            best_cl = cl
    return best_cl


def _random_orthonormal(n_var, n_comp, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n_var, n_comp)))
    return Q


# --------------------------------------------------------------------------
# One ALS run and the multi-restart OptimGRC (operates on the expanded G)
# --------------------------------------------------------------------------
def _als_once(X, A_init, c1, c2, k, rho1, rho2, n_ite, nstart, eps, rng):
    A = np.array(A_init, dtype=float, copy=True)
    loss = np.inf
    U = None
    for _ in range(n_ite):
        loss_old = loss
        F1 = X @ A[:, :c1]
        warm = None
        if U is not None:                                   # warm start = current cluster means
            warm = (U.T @ F1) / U.sum(axis=0)[:, None]
        labels0 = _kmeans_best(F1, k, nstart, rng, warm_centers=warm)
        if labels0 is None:                                 # no k-non-empty partition
            raise MlsynthEstimationError(
                "fGRC: k-means could not form k non-empty clusters (degenerate or "
                "collinear trajectories); reduce k or check the input panel."
            )
        U = indicator(labels0 + 1, k)
        A = eigen_update(X, U, c1, rho1, rho2) if c2 == 0 \
            else gp_algorithm(X, U, A, c1, rho1, rho2)
        loss = grc_loss(X, U, A, c1, rho1, rho2)
        if abs(loss_old - loss) <= eps:
            break
    return A, U, float(loss)


def optim_grc(X, c1: int, c2: int, k: int, rho1: float = 1.0, rho2: float = 0.0,
              n_random: int = 40, nstart: int = 40, n_ite: int = 100, eps: float = 1e-5,
              seed: int = 0):
    """Multi-restart GRC on an already-expanded matrix ``X`` (= ``G``). Returns
    ``(labels [1-based], A, loss)`` at the lowest-loss restart."""
    n_comp = c1 + c2
    best = None
    for r in range(n_random):
        rng = np.random.default_rng((seed + 1) * 100003 + r)
        A0 = _random_orthonormal(X.shape[1], n_comp, rng)
        try:
            A, U, loss = _als_once(X, A0, c1, c2, k, rho1, rho2, n_ite, nstart, eps, rng)
        except MlsynthEstimationError:
            continue                          # skip a restart that hit a degenerate partition
        if best is None or loss < best[2]:
            best = (U.argmax(axis=1) + 1, A, loss)
    if best is None:
        raise MlsynthEstimationError(
            "fGRC: every restart failed to form k non-empty clusters (degenerate or "
            "collinear trajectories); reduce k or check the input panel."
        )
    return best


def fgrc_cluster(trajectories, c1: int, c2: int, k: int, n_knots: int, order: int = 4,
                 rho1: float = 1.0, rho2: float = 0.0, center: bool = True,
                 n_random: int = 40, nstart: int = 40, n_ite: int = 100, eps: float = 1e-5,
                 seed: int = 0):
    """Cluster a panel of trajectories (units x time) with fGRC.

    Parameters
    ----------
    trajectories : np.ndarray
        Smooth series per unit, shape ``(n_units, n_time)``.
    c1, c2, k : int
        Cluster-subspace dim, disturbing-subspace dim, number of clusters.
    n_knots, order : int
        B-spline breakpoints and order (4 = cubic).
    center : bool
        Column-center the trajectory matrix before expansion (default True).

    Returns
    -------
    (labels, loss) : np.ndarray, float
        1-based cluster label per unit and the best objective value.
    """
    X = np.asarray(trajectories, dtype=float)
    if X.ndim != 2:
        raise MlsynthDataError("fgrc_cluster: trajectories must be 2D (units x time).")
    n_units = X.shape[0]
    if c1 < 1:
        raise MlsynthConfigError("fgrc_cluster: c1 (cluster-subspace dim) must be >= 1.")
    if c2 < 0:
        raise MlsynthConfigError("fgrc_cluster: c2 (disturbing-subspace dim) must be >= 0.")
    if k < 2:
        raise MlsynthConfigError("fgrc_cluster: k (number of clusters) must be >= 2.")
    if k > n_units:
        raise MlsynthConfigError(
            f"fgrc_cluster: k ({k}) cannot exceed the number of units ({n_units})."
        )
    if rho1 <= rho2:
        raise MlsynthConfigError("fgrc_cluster: require rho1 > rho2 (Yamamoto-Hwang constraint).")
    if center:
        X = X - X.mean(axis=0)
    G = basis_expand(X, n_knots, order)
    if c1 + c2 > G.shape[1]:
        raise MlsynthConfigError(
            f"fgrc_cluster: c1+c2 ({c1 + c2}) exceeds the number of B-spline "
            f"basis functions ({G.shape[1]}); reduce c1/c2 or raise n_knots."
        )
    labels, _A, loss = optim_grc(G, c1, c2, k, rho1, rho2, n_random, nstart, n_ite, eps, seed)
    return labels, loss
