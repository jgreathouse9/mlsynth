"""Weight solve and confidence-set machinery for SCD.

Two pieces:

* the simplex point estimator -- fit ``w`` on the differenced pre-period
  group means via mlsynth's pure-NumPy active-set solver;
* the weight confidence set of Canen & Song (2025). Membership of a candidate
  ``w`` is a chi-squared test on the projected moment. The projection QP
  ``min_r (phi - r)' P (phi - r)`` s.t. ``w'r = 0, r >= 0`` has, for ``w, r >=
  0``, the property ``r_j = 0`` wherever ``w_j > 0`` -- so ``r`` is supported
  only on the (near-)zero set of ``w`` and the QP collapses to a small NNLS.
  Interior points (all ``w > tol``) need no solve at all, so a dense grid is
  swept with one batched quadratic form plus an NNLS only on boundary points.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import nnls
from scipy.stats import chi2

from ..bilevel.active_set import solve_simplex_qp

_INTERIOR_TOL = 1e-8  # a coordinate above this is treated as strictly positive


def lambda_vector(differencing: str, T0: int) -> np.ndarray:
    """Return the length-``T0`` differencing weight vector.

    ``"did"`` -> ``(0, ..., 0, 1)`` (off the last pre-period); ``"uniform"``
    -> ``(1/T0, ..., 1/T0)`` (off the pre-period average); ``"sc"`` ->
    zeros (no differencing).
    """
    if differencing == "did":
        v = np.zeros(T0)
        v[-1] = 1.0
        return v
    if differencing == "uniform":
        return np.full(T0, 1.0 / T0)
    if differencing == "sc":
        return np.zeros(T0)
    raise ValueError(f"Unknown differencing scheme {differencing!r}.")  # pragma: no cover


def solve_scd_weights(G: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Simplex least squares ``argmin_{w in Delta} ||g - G w||^2``."""
    return np.asarray(solve_simplex_qp(G, g)).ravel()


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``v`` onto the probability simplex."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.max(np.where(u > (css - 1) / np.arange(1, len(v) + 1))[0])
    theta = (css[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def _test_statistic(w: np.ndarray, phi: np.ndarray, ops: Any, tol: float):
    """Return ``(T_stat, df)`` for candidate ``w`` at deviation ``phi``."""
    K = ops.K
    if np.all(w > _INTERIOR_TOL):
        r = np.zeros(K)
    else:
        Z = np.where(w <= _INTERIOR_TOL)[0]
        rz, _ = nnls(ops.sqrtP[:, Z], ops.sqrtP @ phi)
        r = np.zeros(K)
        r[Z] = rz
    dv = phi - r
    T_stat = float(ops.n * dv @ ops.precomp @ dv)
    gamma = ops.precomp @ dv
    hat_d = int(np.sum((np.abs(gamma) < tol) & (w == 0)))
    df = max(K - 1 - hat_d, 1)
    return T_stat, df


def in_C(w: np.ndarray, ops: Any, kappa: float, tol: float) -> bool:
    """Is ``w`` in the ``1 - kappa`` weight confidence set?"""
    w = np.asarray(w, dtype=float).ravel()
    phi = ops.hat_H @ w - ops.hat_h
    T_stat, df = _test_statistic(w, phi, ops, tol)
    return bool(T_stat <= chi2.ppf(1 - kappa, df))


def confidence_set(
    ops: Any,
    kappa: float,
    tol: float,
    n_grid: int,
    radius: float,
    random_state: int,
) -> np.ndarray:
    """Sweep the weight confidence set; return the accepted candidates.

    The grid mirrors the reference construction: the fitted weights, a
    Gaussian cloud around them (simplex-projected), and a Dirichlet cloud.
    Dense (interior) candidates are tested with one batched quadratic form;
    sparse (boundary) candidates fall back to the per-point NNLS.
    """
    rng = np.random.default_rng(random_state)
    K = ops.K
    pts = [ops.hat_w]
    if n_grid > 0:
        cloud = ops.hat_w + rng.normal(0.0, radius, size=(n_grid, K))
        pts.extend(project_simplex(v) for v in cloud)
        pts.extend(rng.dirichlet(np.ones(K), n_grid))
    W = np.array(pts)

    Phi = ops.hat_H @ W.T - ops.hat_h[:, None]           # (K, N)
    interior = np.all(W > _INTERIOR_TOL, axis=1)
    keep = np.zeros(W.shape[0], dtype=bool)

    # interior: r = 0, df = K - 1, one batched quadratic form.
    if interior.any():
        Pi = ops.precomp @ Phi[:, interior]
        Ts = ops.n * np.einsum("ki,ki->i", Phi[:, interior], Pi)
        keep[interior] = Ts <= chi2.ppf(1 - kappa, K - 1)
    # boundary: small NNLS per point.
    for j in np.where(~interior)[0]:
        Ts, df = _test_statistic(W[j], Phi[:, j], ops, tol)
        keep[j] = Ts <= chi2.ppf(1 - kappa, df)

    return W[keep]
