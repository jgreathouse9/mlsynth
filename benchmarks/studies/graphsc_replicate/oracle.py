"""Graph-SC, ported from the paper as an oracle.

Zhang, Y., Chung, S. and Talebpour, A. (2026), "Graph-Regularized Synthetic
Control for Stable Counterfactual Estimation under Donor Redundancy",
arXiv:2610.24113v1.

* ``knn_laplacian`` is Algorithm 1 (p. 4) with Definition 3.1: a similarity
  matrix with ``-inf`` on the diagonal, each donor's ``k`` most similar donors,
  the symmetrized unweighted adjacency, and ``L = I - D^-1/2 A D^-1/2``.
* ``fista`` is Algorithm 2 (p. 14): step ``1 / L_f`` with ``L_f`` the largest
  eigenvalue of ``X0'X0 + lambda_g L + lambda_sum 11'`` (eq. (10)), and the
  proximal map ``(v - eta lambda_1)_+`` of Lemma A.3, so ``w >= 0`` as Algorithm
  2 enforces it (Definition 3.4 writes ``w`` in ``R^J``; the proofs and the
  algorithm use ``w >= 0``). The paper gives no stopping rule; the loop stops on
  the KKT residual and reports when it reaches its cap.
* ``solve_exact`` solves the same problem exactly. With ``w >= 0`` the l1 term
  is ``lambda_1 1'w``, linear, and
  ``lambda_1 1'w + lambda_sum/2 (1'w - 1)^2 = lambda_sum/2 (1'w - s)^2 + const``
  with ``s = 1 - lambda_1 / lambda_sum``. Eq. (2) is therefore the non-negative
  least-squares problem ``min ||M w - b||^2``, ``w >= 0``, with ``M`` stacking
  ``X0``, ``sqrt(lambda_g) R`` (``R'R = L``) and ``sqrt(lambda_sum) 1'``, and
  ``b`` stacking ``x1``, zeros and ``sqrt(lambda_sum) s``. The l1 penalty does
  no more than lower the target sum from 1 to ``s``: 0.999 at the paper's
  ``lambda_1 = 0.05``, ``lambda_sum = 50``.

* ``solve_signed`` solves eq. (2) as Definition 3.4 writes it, ``w`` in ``R^J``:
  a linear solve when ``lambda_1 = 0``, CLARABEL otherwise, and ``fista(...,
  signed=True)`` is Algorithm 2 with the soft-threshold prox that reading needs.
  Only this reading reproduces Table 1's ablation rows on Prop 99 (``run.py``).

Both solvers are certified by ``kkt_residual`` on the returned point: the
prox-gradient mapping ``w - (w - eta (grad f + lambda_1))_+``, zero exactly at
the minimiser, relative to ``max|w|`` with no absolute floor (Nocedal and
Wright 2006, pp. 26-27, on tolerances that do not depend on the data's units).
"""
from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np

from mlsynth.utils.solvers.nnls import nnls

_EPS = np.finfo(float).eps


# ------------------------------------------------------------------ the graph


def _similarity(F: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        sq = np.sum(F * F, axis=1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * F @ F.T, 0.0)
        return -np.sqrt(d2)                      # nearest = most similar
    if metric == "correlation":
        return np.corrcoef(F)
    if metric == "cosine":
        n = np.linalg.norm(F, axis=1)
        return (F @ F.T) / np.outer(n, n)
    raise ValueError(f"unknown similarity metric {metric!r}: euclidean, correlation or cosine")


def knn_laplacian(F: np.ndarray, k: int, metric: str = "euclidean") -> Tuple[np.ndarray, np.ndarray]:
    """Algorithm 1: ``(L, A)`` for donors with features ``F`` (one row per donor)."""
    F = np.asarray(F, dtype=float)
    J = F.shape[0]
    S = _similarity(F, metric)
    np.fill_diagonal(S, -np.inf)
    nbrs = np.argsort(-S, axis=1, kind="stable")[:, :k]
    A_dir = np.zeros((J, J), bool)
    np.put_along_axis(A_dir, nbrs, True, axis=1)
    A = (A_dir | A_dir.T).astype(float)
    np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1)
    inv = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
    L = np.eye(J) - inv[:, None] * A * inv[None, :]
    return L, A


def random_graph_laplacian(n_edges: int, J: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    """A uniformly random undirected graph with ``n_edges`` edges: the 100%-rewired graph of Appendix C."""
    iu, ju = np.triu_indices(J, 1)
    pick = rng.choice(len(iu), size=n_edges, replace=False)
    A = np.zeros((J, J))
    A[iu[pick], ju[pick]] = 1.0
    A = A + A.T
    d = A.sum(axis=1)
    inv = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
    return np.eye(J) - inv[:, None] * A * inv[None, :], A


# -------------------------------------------------------------- the objective


def objective(w, x1, X0, L, lam1, lamg, lamsum) -> float:
    """Eq. (2) at ``w`` (``||w||_1`` written out, so it is defined for any ``w``)."""
    r = x1 - X0 @ w
    return float(0.5 * r @ r + lam1 * np.abs(w).sum() + 0.5 * lamg * w @ L @ w
                 + 0.5 * lamsum * (w.sum() - 1.0) ** 2)


def _grad_f(w, x1, X0, L, lamg, lamsum):
    """Lemma A.1: the gradient of the smooth part."""
    return X0.T @ (X0 @ w - x1) + lamg * (L @ w) + lamsum * (w.sum() - 1.0)


def _lipschitz(X0, L, lamg, lamsum) -> float:
    """Eq. (10): ``L_f``, computed exactly (the matrix is ``J x J``)."""
    J = X0.shape[1]
    H = X0.T @ X0 + lamg * L + lamsum * np.ones((J, J))
    return float(np.linalg.eigvalsh(H)[-1])


def _prox(v, t, signed):
    """Lemma A.3's prox of ``t ||.||_1``: on ``w >= 0`` a shift and clip, on ``R^J`` a soft threshold."""
    if signed:
        return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)
    return np.maximum(v - t, 0.0)


def kkt_residual(w, x1, X0, L, lam1, lamg, lamsum, signed: bool = False) -> float:
    """``max|w - prox(w - eta grad f)| / max|w|`` with ``eta = 1 / L_f``.

    Zero exactly at the minimiser (the fixed point of Algorithm 2's step), over
    ``w >= 0`` or, with ``signed``, over ``R^J``. Relative to the size of ``w``,
    with no floor; at ``w = 0`` the residual is 0 if the step is, else infinite.
    """
    eta = 1.0 / _lipschitz(X0, L, lamg, lamsum)
    step = w - _prox(w - eta * _grad_f(w, x1, X0, L, lamg, lamsum), eta * lam1, signed)
    scale = np.abs(w).max()
    if scale > 0:
        return float(np.abs(step).max() / scale)
    return 0.0 if not np.any(step) else float("inf")


def fista(x1, X0, L, lam1, lamg, lamsum, w0=None, max_iter: int = 200_000,
          tol: float = 1e-10, signed: bool = False) -> Tuple[np.ndarray, Dict]:
    """Algorithm 2, stopped when ``kkt_residual <= tol`` (checked every 50 steps).

    ``signed`` swaps Lemma A.3's prox for the soft threshold, for ``w`` in ``R^J``.
    """
    x1, X0, L = (np.asarray(a, dtype=float) for a in (x1, X0, L))
    J = X0.shape[1]
    eta = 1.0 / _lipschitz(X0, L, lamg, lamsum)
    w = np.full(J, 1.0 / J) if w0 is None else np.asarray(w0, dtype=float).copy()
    z, t = w.copy(), 1.0
    res = np.inf
    for k in range(1, max_iter + 1):
        u = z - eta * _grad_f(z, x1, X0, L, lamg, lamsum)
        w_new = _prox(u, eta * lam1, signed)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = w_new + (t - 1.0) / t_new * (w_new - w)
        w, t = w_new, t_new
        if k % 50 == 0 or k == max_iter:
            res = kkt_residual(w, x1, X0, L, lam1, lamg, lamsum, signed)
            if res <= tol:
                return w, dict(iterations=k, converged=True, kkt=res)
    warnings.warn(RuntimeWarning(
        f"FISTA stopped at its cap of {max_iter} iterations with KKT residual {res:.2e}"), stacklevel=2)
    return w, dict(iterations=max_iter, converged=False, kkt=res)


def nnls_form(x1, X0, L, lam1, lamg, lamsum):
    """``(M, b, const)`` with eq. (2) ``= ||M w - b||^2 / 2 + const`` for every ``w >= 0``."""
    ev, U = np.linalg.eigh(L)
    R = np.sqrt(np.clip(ev, 0.0, None))[:, None] * U.T             # R'R = L
    s = 1.0 - lam1 / lamsum
    J = X0.shape[1]
    M = np.vstack([X0, np.sqrt(lamg) * R, np.sqrt(lamsum) * np.ones((1, J))])
    b = np.r_[x1, np.zeros(J), np.sqrt(lamsum) * s]
    const = 0.5 * lamsum * (1.0 - s * s)
    return M, b, const


def solve_exact(x1, X0, L, lam1, lamg, lamsum, kkt_tol: float = 1e-9) -> np.ndarray:
    """Eq. (2) as non-negative least squares (Lawson-Hanson), certified by its KKT residual."""
    M, b, _ = nnls_form(x1, X0, L, lam1, lamg, lamsum)
    w = nnls(M, b)[0]
    res = kkt_residual(w, x1, X0, L, lam1, lamg, lamsum)
    if not res <= kkt_tol:
        warnings.warn(RuntimeWarning(
            f"the active-set solve of eq. (2) failed its KKT test: residual {res:.2e}"), stacklevel=2)
    return w


def solve_signed(x1, X0, L, lam1, lamg, lamsum, kkt_tol: float = 1e-7) -> np.ndarray:
    """Eq. (2) over ``w`` in ``R^J``, as Definition 3.4 writes it, certified by its KKT residual.

    Without the l1 term it is the least-squares problem ``min ||M w - b||``
    (``nnls_form`` with ``lambda_1 = 0``), solved by an SVD; with it, by CLARABEL.
    """
    x1, X0, L = (np.asarray(a, dtype=float) for a in (x1, X0, L))
    if lam1 == 0:
        M, b, _ = nnls_form(x1, X0, L, 0.0, lamg, lamsum)
        w = np.linalg.lstsq(M, b, rcond=None)[0]
    else:
        import cvxpy as cp

        v = cp.Variable(X0.shape[1])
        obj = (0.5 * cp.sum_squares(x1 - X0 @ v) + lam1 * cp.norm1(v)
               + 0.5 * lamg * cp.quad_form(v, cp.psd_wrap(L)) + 0.5 * lamsum * cp.square(cp.sum(v) - 1.0))
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
        if prob.status != cp.OPTIMAL:
            warnings.warn(RuntimeWarning(f"CLARABEL returned {prob.status} for eq. (2)"), stacklevel=2)
        w = np.asarray(v.value, float)
    res = kkt_residual(w, x1, X0, L, lam1, lamg, lamsum, signed=True)
    if not res <= kkt_tol:
        warnings.warn(RuntimeWarning(
            f"the signed solve of eq. (2) failed its KKT test: residual {res:.2e}"), stacklevel=2)
    return w


def simplex_sc(x1, X0) -> np.ndarray:
    """Classic SC, eq. (1): ``min ||x1 - X0 w||^2`` on the simplex, via ``mlsynth``'s active-set QP."""
    from mlsynth.utils.solvers.active_set import solve_simplex_qp

    w, info = solve_simplex_qp(np.asarray(X0, float), np.asarray(x1, float), return_info=True)
    if not info["converged"]:
        warnings.warn(RuntimeWarning("the simplex QP for classic SC did not converge"), stacklevel=2)
    return np.asarray(w, float)


# ------------------------------------------------------------------ metrics


def weight_entropy(w, eps: float = 1e-12) -> float:
    """Table 1's ``H(w) = -sum w_j log(w_j + eps)``."""
    w = np.asarray(w, float)
    return float(-np.sum(w * np.log(w + eps)))


def weight_entropy_abs(w, eps: float = 1e-12) -> float:
    """Table 1's entropy as its numbers read: on ``|w|`` normalized to sum to one.

    The printed formula takes ``log(w_j + eps)``, undefined for negative weights;
    for signed weights this reading reproduces the printed Sparse-only and
    Graph-only entropies to four decimals.
    """
    a = np.abs(np.asarray(w, float))
    a = a / a.sum()
    return float(-np.sum(a * np.log(a + eps)))


def active_donors(w, tol: float = 1e-6) -> int:
    return int(np.sum(np.abs(np.asarray(w)) > tol))


def metrics(y, cf, T0, w) -> Dict:
    """Table 1's columns: pre-RMSPE, post-RMSPE, mean post gap magnitude, active donors, entropy."""
    gap = np.asarray(y, float) - np.asarray(cf, float)
    return dict(pre_rmspe=float(np.sqrt(np.mean(gap[:T0] ** 2))),
                post_rmspe=float(np.sqrt(np.mean(gap[T0:] ** 2))),
                post_gap=float(abs(gap[T0:].mean())),
                active=active_donors(w), negative=int(np.sum(np.asarray(w) < -1e-6)),
                entropy=weight_entropy(np.clip(w, 0.0, None)), entropy_abs=weight_entropy_abs(w),
                weight_sum=float(np.sum(w)))
