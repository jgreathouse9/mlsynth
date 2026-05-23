"""Core estimation for the Continuous-Treatment Synthetic Control (CTSC).

Powell, D. (2022). *"Synthetic Control Estimation Beyond Comparative Case
Studies: Does the Minimum Wage Reduce Employment?"* Journal of Business &
Economic Statistics 40(3):1302-1314. (The paper calls the estimator
"GSC"; mlsynth names it CTSC to avoid collision with Xu (2017)'s
Generalized Synthetic Control.)

The estimator jointly fits, for every unit :math:`i`, a treatment-slope
vector :math:`\\alpha_i \\in \\mathbb{R}^K` and a synthetic control over the
*other* units' untreated outcomes (paper eq. 5):

.. math::

   \\min_{b, \\phi}\\ \\frac{1}{2nT}\\sum_i \\Omega_i^{-1} \\sum_t
     \\Bigl[ Y_{it} - D_{it}' b_i
            - \\sum_{j \\ne i} \\phi_j^i (Y_{jt} - D_{jt}' b_j) \\Bigr]^2,
   \\quad \\phi_j^i \\ge 0,\\ \\sum_{j \\ne i} \\phi_j^i = 1,

where :math:`Y_{it} - D_{it}' b_i` is unit :math:`i`'s untreated outcome
and :math:`\\Omega_i` is a per-unit fit weight (eq. 6) that down-weights
units without a good synthetic control.

The paper minimises this with Nelder-Mead over all
:math:`nK + n(n-1)` parameters. The objective is **biconvex** -- a
weighted linear least squares in the stacked slopes :math:`b` for fixed
weights :math:`\\phi`, and :math:`n` independent simplex-constrained least
squares in :math:`\\phi` for fixed :math:`b` -- so mlsynth uses block
coordinate descent (a single linear solve alternated with per-unit
simplex QPs), which optimises the same objective far more stably.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import nnls

from ...exceptions import MlsynthEstimationError

_TOL = 1e-9


def _simplex_ls(target: np.ndarray, donors: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to one minimising ``||target - donors @ w||``.

    Solved as non-negative least squares with the sum-to-one constraint
    imposed by a heavy penalty row, then renormalised. This is far faster
    and more robust than a QP solver when called in the inner loop of the
    block-coordinate descent (thousands of small solves).
    """
    m = donors.shape[1]
    if m == 0:
        return np.zeros(0)
    if m == 1:
        return np.ones(1)
    scale = float(np.sqrt((donors ** 2).sum() / max(donors.size, 1))) or 1.0
    penalty = 1e3 * scale
    A = np.vstack([donors, penalty * np.ones((1, m))])
    y = np.concatenate([target, [penalty]])
    try:
        w, _ = nnls(A, y, maxiter=max(50 * m, 200))
    except RuntimeError as exc:
        raise MlsynthEstimationError(f"CTSC simplex NNLS failed: {exc}") from exc
    total = w.sum()
    if total <= _TOL:
        return np.full(m, 1.0 / m)
    return w / total


def _phi_step(Y: np.ndarray, D: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Given slopes ``b`` (n, K), refit each unit's weights on untreated
    outcomes. Returns ``Phi`` (n, n) with zero diagonal, rows on the simplex."""
    n, T = Y.shape
    U = Y - np.einsum("itk,ik->it", D, b)        # untreated outcomes (n, T)
    Phi = np.zeros((n, n))
    for i in range(n):
        others = [j for j in range(n) if j != i]
        w = _simplex_ls(U[i], U[others].T)
        Phi[i, others] = w
    return Phi


def _b_step(
    Y: np.ndarray, D: np.ndarray, Phi: np.ndarray, omega: np.ndarray,
    *, pi: Optional[np.ndarray] = None, restrict_ae: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Given weights ``Phi``, solve the weighted linear least squares for the
    stacked slopes ``b`` (n, K) in closed form (paper eq. 5 is quadratic in b).

    If ``restrict_ae`` is given, impose the average-effect equality
    constraint ``sum_i pi_i b_{i,k} = restrict_ae[k]`` for each variable
    ``k`` via a KKT-augmented system (used by the inference null fit).
    """
    n, T, K = D.shape
    P = n * K
    # Residual e_it = c_it - g_it' beta, with
    #   c_it = Y_it - sum_{j!=i} phi_ij Y_jt
    #   g_it[block i] = D_it ; g_it[block j!=i] = -phi_ij D_jt
    c = Y - Phi @ Y                                  # (n, T)
    AtA = np.zeros((P, P))
    Atc = np.zeros(P)
    for i in range(n):
        wi = 1.0 / omega[i]
        G = np.zeros((T, P))
        G[:, i * K:(i + 1) * K] = D[i]               # +D_it on block i
        for j in range(n):
            if j == i:
                continue
            G[:, j * K:(j + 1) * K] = -Phi[i, j] * D[j]
        AtA += wi * (G.T @ G)
        Atc += wi * (G.T @ c[i])

    AtA += _TOL * np.eye(P)
    try:
        if restrict_ae is None:
            beta = np.linalg.solve(AtA, Atc)
        else:
            # KKT: [[AtA, C'], [C, 0]] [beta; nu] = [Atc; a0], with
            # C[k, i*K + k] = pi_i  (one constraint per variable).
            C = np.zeros((K, P))
            for k in range(K):
                for i in range(n):
                    C[k, i * K + k] = pi[i]
            KKT = np.block([[AtA, C.T], [C, np.zeros((K, K))]])
            rhs = np.concatenate([Atc, np.asarray(restrict_ae, dtype=float)])
            sol = np.linalg.solve(KKT, rhs)
            beta = sol[:P]
    except np.linalg.LinAlgError as exc:
        raise MlsynthEstimationError(f"CTSC b-step solve failed: {exc}") from exc
    return beta.reshape(n, K)


def _objective(
    Y: np.ndarray, D: np.ndarray, Phi: np.ndarray, b: np.ndarray, omega: np.ndarray
) -> float:
    U = Y - np.einsum("itk,ik->it", D, b)
    resid = U - Phi @ U
    n, T = Y.shape
    return float(np.sum((1.0 / omega)[:, None] * resid ** 2) / (2 * n * T))


def _alternate(
    Y: np.ndarray, D: np.ndarray, omega: np.ndarray,
    *, max_iter: int = 200, tol: float = 1e-8,
    pi: Optional[np.ndarray] = None, restrict_ae: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Block coordinate descent on (b, Phi). Returns (b, Phi, objective)."""
    n, T, K = D.shape
    b = np.zeros((n, K))
    Phi = _phi_step(Y, D, b)
    prev = np.inf
    obj = np.inf
    for _ in range(max_iter):
        b = _b_step(Y, D, Phi, omega, pi=pi, restrict_ae=restrict_ae)
        Phi = _phi_step(Y, D, b)
        obj = _objective(Y, D, Phi, b, omega)
        if abs(prev - obj) < tol * (1.0 + abs(prev)):
            break
        prev = obj
    return b, Phi, obj


def _per_unit_fit(Y: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Step-1 per-unit fit weights Omega_i (paper eq. 6).

    For each unit independently, fit its slope and a synthetic control from
    the other units' (observed) outcomes; ``Omega_i`` is the minimised
    mean squared residual -- small for well-fitting units.
    """
    n, T, K = D.shape
    omega = np.ones(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        Yo = Y[others].T                              # (T, n-1)
        bi = np.zeros(K)
        prev = np.inf
        for _ in range(100):
            target = Y[i] - D[i] @ bi
            w = _simplex_ls(target, Yo)
            resid_y = Y[i] - Yo @ w                    # (T,)
            # regress resid_y on D[i] (closed form OLS) for the slope
            Di = D[i]                                  # (T, K)
            bi, *_ = np.linalg.lstsq(Di, resid_y, rcond=None)
            e = Y[i] - Di @ bi - Yo @ w
            ss = float(np.mean(e ** 2))
            if abs(prev - ss) < 1e-10 * (1 + abs(prev)):
                break
            prev = ss
        omega[i] = max(ss, _TOL)
    return omega


def fit_ctsc(
    Y: np.ndarray,
    D: np.ndarray,
    *,
    population_weights: Optional[np.ndarray] = None,
    use_fit_weights: bool = True,
    restrict_ae: Optional[np.ndarray] = None,
    omega: Optional[np.ndarray] = None,
) -> dict:
    """Fit CTSC and return the unit-specific slopes and the average effect.

    Parameters
    ----------
    Y : np.ndarray
        Outcomes, shape ``(n, T)``.
    D : np.ndarray
        Treatment / explanatory variables, shape ``(n, T, K)``.
    population_weights : np.ndarray, optional
        Per-unit weights :math:`\\pi_i` for the average effect (eq. 7);
        defaults to uniform ``1/n``.
    use_fit_weights : bool
        If True, use the two-step per-unit fit weights :math:`\\Omega_i`
        (eq. 6); if False, weight all units equally.

    Returns
    -------
    dict with keys ``alpha`` (n, K) unit slopes, ``average_effect`` (K,),
    ``weights`` (n, n) synthetic-control matrix, ``omega`` (n,) fit weights,
    ``objective`` (float).
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    if D.ndim == 2:
        D = D[:, :, None]
    n, T, K = D.shape

    if omega is None:
        omega = _per_unit_fit(Y, D) if use_fit_weights else np.ones(n)

    if population_weights is None:
        pi = np.full(n, 1.0 / n)
    else:
        pi = np.asarray(population_weights, dtype=float)
        pi = pi / pi.sum()

    b, Phi, obj = _alternate(Y, D, omega, pi=pi, restrict_ae=restrict_ae)
    average_effect = pi @ b                            # (K,)

    return {
        "alpha": b,
        "average_effect": average_effect,
        "weights": Phi,
        "omega": omega,
        "objective": obj,
    }
