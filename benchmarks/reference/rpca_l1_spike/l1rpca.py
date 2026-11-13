"""Rank-k l1 low-rank fit and its criticality certificate.

Gong, Lai & Ma (2026), "Certifying optimality in nonconvex robust PCA"
(arXiv:2601.21333), study

    f(X, Y) = || X Y^T - M ||_1 ,   X in R^{m x k}, Y in R^{n x k}.

The solver here reaches exact zeros in the residual, which the certificate
needs (a subgradient run leaves the residual nonzero almost everywhere, so its
sign pattern is fixed on every entry and no certificate exists):

1. the paper's subgradient method from a small random start (step halved after
   10 non-improving iterations), from several starts and from the SVD;
2. alternating least absolute deviations from each: with Y fixed each row of X
   is an LAD regression, solved exactly as a linear program (HiGHS), and the
   same for Y. This finish makes the residual's zeros exact.

Alternating LAD alone, started at the SVD, stalled away from the truth on 2 of
3 noiseless rank-3 problems (relative error 0.7); block coordinate descent on a
nonsmooth objective can stop at a non-critical point, which is why it is not
used on its own.

Certificate (paper Sec. 2.1 and Lemma 1). (X, Y) is Clarke-critical iff some
Lambda in sign(X Y^T - M) has Lambda Y = 0 and Lambda^T X = 0. With U, V
orthonormal bases of col(L), row(L) and Omega the residual's support, that is
the linear program

    t* = min t  s.t.  Lambda_Omega = sign(R_Omega),  |Lambda_{Omega^c}| <= t,
                      Lambda V = 0,  U^T Lambda = 0,

and the fit is critical iff t* <= 1. At k = rank(L), Theorem 3's sharpness
constant is proportional to 1 - t*.
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog


def lad(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """argmin_x ||A x - b||_1, exactly, as a linear program."""
    n, p = A.shape
    c = np.r_[np.zeros(p), np.ones(n)]
    A_ub = np.block([[A, -np.eye(n)], [-A, -np.eye(n)]])
    res = linprog(c, A_ub=A_ub, b_ub=np.r_[b, -b],
                  bounds=[(None, None)] * p + [(0, None)] * n, method="highs")
    if res.status != 0:  # pragma: no cover - an LAD LP is always feasible and bounded
        raise RuntimeError(f"LAD LP failed: {res.message}")
    return res.x[:p]


@dataclass(frozen=True)
class L1Fit:
    low_rank: np.ndarray
    objective: float          # ||L - M||_1 on the original scale
    sweeps: int
    polish_gain: float        # relative drop in f from the LAD finish


def _alternate(Mn, X, Y, sweeps):
    prev = np.inf
    done = 0
    for done in range(1, sweeps + 1):
        X = np.array([lad(Y, Mn[i]) for i in range(Mn.shape[0])])
        Y = np.array([lad(X, Mn[:, j]) for j in range(Mn.shape[1])])
        f = np.abs(X @ Y.T - Mn).sum()
        if prev - f <= 1e-12 * max(1.0, f):
            break
        prev = f
    return X, Y, done


def _subgradient(Mn, k, iters, seed, init=1e-3, step0=0.05):
    """The paper's method (Fig. 1): small random init, normalised subgradient
    steps, step halved after 10 iterations without improvement."""
    rng = np.random.default_rng(seed)
    m, n = Mn.shape
    X = init * rng.standard_normal((m, k))
    Y = init * rng.standard_normal((n, k))
    best, bX, bY = np.inf, X, Y
    step, stall = step0, 0
    for _ in range(iters):
        R = X @ Y.T - Mn
        f = np.abs(R).sum()
        if f < best - 1e-14:
            best, bX, bY, stall = f, X.copy(), Y.copy(), 0
        else:
            stall += 1
            if stall >= 10:
                step, stall = step * 0.5, 0
        if step < 1e-12:
            break
        G = np.sign(R)
        gX, gY = G @ Y, G.T @ X
        nrm = np.sqrt((gX ** 2).sum() + (gY ** 2).sum()) + 1e-300
        X, Y = X - step * gX / nrm, Y - step * gY / nrm
    return bX, bY


def l1_lowrank(M: np.ndarray, k: int, restarts: int = 4, iters: int = 4000,
               sweeps: int = 40, seed: int = 0) -> L1Fit:
    """Rank-k l1 fit of M: the paper's subgradient method from ``restarts``
    small random starts plus one SVD start, each finished by alternating exact
    LAD so the residual's zeros are exact. Returns the lowest objective."""
    scale = float(np.abs(M).max()) or 1.0
    Mn = M / scale
    starts = []
    for r in range(restarts):
        starts.append(_subgradient(Mn, k, iters, seed + r))
    U, s, Vt = np.linalg.svd(Mn, full_matrices=False)
    starts.append((U[:, :k] * s[:k], Vt[:k].T))
    best = None
    for X0, Y0 in starts:
        f0 = np.abs(X0 @ Y0.T - Mn).sum()
        X, Y, n = _alternate(Mn, X0, Y0, sweeps)
        f = np.abs(X @ Y.T - Mn).sum()
        if best is None or f < best[0]:
            best = (f, X, Y, n, (f0 - f) / f0 if f0 > 0 else 0.0)
    f, X, Y, n, gain = best
    return L1Fit(low_rank=(X @ Y.T) * scale, objective=f * scale,
                 sweeps=n, polish_gain=float(gain))


@dataclass(frozen=True)
class Certificate:
    status: str
    t_star: float             # critical iff <= 1
    support_share: float      # share of entries where the residual is nonzero

    @property
    def critical(self) -> bool:
        return self.status == "optimal" and self.t_star <= 1.0 + 1e-7


def certify(M: np.ndarray, L: np.ndarray, r: int, tol: float = 1e-7) -> Certificate:
    """The criticality LP at rank r; ``tol`` is relative to max |M|."""
    U, _s, Vt = np.linalg.svd(L, full_matrices=False)
    U, V = U[:, :r], Vt[:r].T
    R = L - M
    omega = np.abs(R) > tol * np.abs(M).max()
    Lam = cp.Variable(M.shape)
    t = cp.Variable()
    cons = [Lam @ V == 0, U.T @ Lam == 0]
    if omega.any():
        cons.append(cp.multiply(omega, Lam) == np.sign(R) * omega)
    if (~omega).any():
        cons.append(cp.abs(cp.multiply(~omega, Lam)) <= t)
    else:
        cons.append(t == 0)
    prob = cp.Problem(cp.Minimize(t), cons)
    try:
        prob.solve(solver=cp.HIGHS)
    except cp.error.SolverError:  # pragma: no cover
        return Certificate("solver-error", float("nan"), float(omega.mean()))
    t_val = float(t.value) if t.value is not None else float("nan")
    return Certificate(prob.status, t_val, float(omega.mean()))
