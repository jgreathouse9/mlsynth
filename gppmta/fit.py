"""Penalised least-squares learning of the graphical point process (Sec. 4.1).

Each customer-initiated node ``e`` solves (EC.3)

    min_{theta >= 0}  1/2 theta' V theta - b_e' theta + gamma_e * sum_{k>=1} theta_k,

with ``theta = (mu_e, alpha_{1e}, ..., alpha_{pe})``. On the non-negative orthant
the L1 penalty is the linear term ``gamma_e * delta' theta`` with
``delta = (0, 1, ..., 1)``, so the problem is a bound-constrained convex QP. With
``V = L L'`` it is exactly the non-negative least-squares problem
``min ||L' theta - L^{-1} c||^2`` for ``c = b_e - gamma_e delta``, which
``scipy.optimize.nnls`` solves to optimality by an active-set method. The
paper's ADMM (Algorithm 3) is kept in :func:`admm` for comparison.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls

from .paths import Path
from .stats import SufficientStats, as_decay, sufficient_stats


@dataclass(frozen=True)
class GPPParams:
    """Parameters of model (1): baselines and Granger coefficients."""

    mu: np.ndarray  # (q,) baselines of the customer-initiated types
    alpha: np.ndarray  # (p, q): alpha[k, e] is the coefficient of edge k -> e
    decay: np.ndarray  # (p,) kernel decay rate of each source type

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu, dtype=float)
        alpha = np.asarray(self.alpha, dtype=float)
        p, q = alpha.shape
        if mu.shape != (q,):
            raise ValueError("mu must have one entry per customer-initiated type")
        if np.any(mu < 0) or np.any(alpha < 0):
            raise ValueError("mu and alpha must be non-negative")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "decay", as_decay(self.decay, p))

    @property
    def p(self) -> int:
        return self.alpha.shape[0]

    @property
    def q(self) -> int:
        return self.alpha.shape[1]

    @property
    def alpha_full(self) -> np.ndarray:
        """``(p, p)`` coefficient matrix; firm-initiated columns are zero."""
        out = np.zeros((self.p, self.p))
        out[:, : self.q] = self.alpha
        return out

    @property
    def edges(self) -> list[tuple[int, int]]:
        """Edges ``(k, e)`` of the estimated Granger causality graph."""
        return [tuple(map(int, kv)) for kv in np.argwhere(self.alpha > 0)]


def objective(V: np.ndarray, b: np.ndarray, gamma: float, theta: np.ndarray) -> float:
    return 0.5 * theta @ V @ theta - b @ theta + gamma * theta[1:].sum()


def _sqrt_factor(V: np.ndarray) -> np.ndarray:
    """Lower-triangular ``L`` with ``L L' = V`` (V symmetric PSD)."""
    try:
        return np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        ridge = 1e-12 * max(np.trace(V), 1.0)
        warnings.warn(
            "V is singular (collinear features); adding a ridge of "
            f"{ridge:.1e} to solve the node problem",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.linalg.cholesky(V + ridge * np.eye(len(V)))


def nonneg_qp(V: np.ndarray, c: np.ndarray, support: np.ndarray | None = None) -> np.ndarray:
    """``argmin_{theta >= 0} 1/2 theta' V theta - c' theta`` (exact, via NNLS).

    Coordinates whose feature never fires (zero diagonal of ``V``) or outside
    ``support`` are fixed at zero.
    """
    k = len(c)
    keep = np.diag(V) > 0
    if support is not None:
        keep &= support
    theta = np.zeros(k)
    if not keep.any():
        return theta
    L = _sqrt_factor(V[np.ix_(keep, keep)])
    y = solve_triangular(L, c[keep], lower=True)
    theta[keep], _ = nnls(L.T, y, maxiter=50 * k)
    return theta


def solve_node(V: np.ndarray, b: np.ndarray, gamma: float, refit: bool = False) -> np.ndarray:
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    delta = np.r_[0.0, np.ones(len(b) - 1)]
    theta = nonneg_qp(V, b - gamma * delta)
    if refit:
        # Relaxed lasso: unpenalised fit on the selected edges (mu always kept).
        support = theta > 0
        support[0] = True
        theta = nonneg_qp(V, b, support)
    return theta


def admm(
    V: np.ndarray,
    b: np.ndarray,
    gamma: float,
    eta: float = 1.0,
    max_iter: int = 100_000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, int]:
    """Algorithm 3 of the paper, transcribed as published.

    The ``theta`` step clips the unconstrained minimiser at zero, which equals
    the constrained minimiser only when no clipping binds (or V is diagonal);
    see ``tests/test_fit.py`` for how close it lands to the exact solution.
    Returns ``(theta, iterations)``.
    """
    p1 = len(b)
    D = np.zeros((p1, p1))
    D[1:, 1:] = eta * np.eye(p1 - 1)
    Minv = np.linalg.inv(V + D)
    a2 = np.zeros(p1 - 1)
    w = np.zeros(p1 - 1)
    theta = np.zeros(p1)
    for it in range(1, max_iter + 1):
        rhs = b.copy()
        rhs[1:] += eta * a2 - w
        theta = np.maximum(Minv @ rhs, 0.0)
        a2_new = np.maximum(theta[1:] + w / eta - gamma / eta, 0.0)
        w = w + eta * (theta[1:] - a2_new)
        primal = np.linalg.norm(theta[1:] - a2_new)
        dual = eta * np.linalg.norm(a2_new - a2)
        a2 = a2_new
        if primal < tol and dual < tol:
            break
    return np.r_[theta[0], a2], it


def gamma_max(V: np.ndarray, b: np.ndarray) -> float:
    """Smallest ``gamma`` at which every ``alpha_e`` is zero (KKT at alpha = 0)."""
    mu0 = max(b[0] / V[0, 0], 0.0)
    return float(max(np.max(b[1:] - V[1:, 0] * mu0), 0.0))


def fit_params(
    stats: SufficientStats, q: int, decay, gamma, refit: bool = False
) -> GPPParams:
    p = stats.B.shape[0]
    gam = np.broadcast_to(np.asarray(gamma, dtype=float), (q,))
    V = stats.V
    thetas = [solve_node(V, stats.b(e), float(gam[e]), refit) for e in range(q)]
    mu = np.array([th[0] for th in thetas])
    alpha = np.column_stack([th[1:] for th in thetas])
    return GPPParams(mu, alpha, as_decay(decay, p))


@dataclass(frozen=True)
class GammaCV:
    """Per-node cross-validation of ``gamma`` on the held-out contrast."""

    grid: np.ndarray  # (q, G)
    loss: np.ndarray  # (q, G) mean held-out loss per path
    best: np.ndarray  # (q,)


def cv_gamma(
    paths: Sequence[Path],
    p: int,
    q: int,
    decay,
    n_folds: int = 5,
    n_grid: int = 30,
    ratio: float = 1e-4,
    seed: int = 0,
) -> GammaCV:
    """K-fold CV of each node's ``gamma`` by the held-out least-squares contrast.

    The contrast is additive over paths, so fold statistics are computed once
    and training statistics are ``total - fold``.
    """
    if n_folds < 2 or n_folds > len(paths):
        raise ValueError("n_folds must be between 2 and the number of paths")
    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(len(paths)) % n_folds
    folds = [sufficient_stats([paths[j] for j in np.flatnonzero(fold_of == f)], p, decay)
             for f in range(n_folds)]
    total = folds[0]
    for f in folds[1:]:
        total = total + f
    grids = np.empty((q, n_grid + 1))
    loss = np.zeros((q, n_grid + 1))
    for e in range(q):
        gmax = gamma_max(total.V, total.b(e))
        grids[e] = np.r_[gmax * np.geomspace(1.0, ratio, n_grid), 0.0] if gmax > 0 else 0.0
    for test in folds:
        train = total - test
        Vtr, Vte = train.V, test.V
        for e in range(q):
            btr, bte = train.b(e), test.b(e)
            for g, gam in enumerate(grids[e]):
                th = solve_node(Vtr, btr, gam)
                loss[e, g] += (0.5 * th @ Vte @ th - bte @ th) * test.n / total.n
    best = grids[np.arange(q), loss.argmin(axis=1)]
    return GammaCV(grids, loss, best)


def cv_decay(
    paths: Sequence[Path],
    p: int,
    q: int,
    grid: Sequence[float],
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Pick a common kernel decay from ``grid`` by held-out contrast.

    The paper treats the kernels as known. The least-squares contrast is a
    proper score for the intensity, so held-out values are comparable across
    feature sets; each candidate is scored by its CV-tuned penalty, summed
    over the customer-initiated nodes. Returns ``(best, losses)``.
    """
    losses = np.array([
        cv_gamma(paths, p, q, d, n_folds, seed=seed).loss.min(axis=1).sum() for d in grid
    ])
    return float(np.asarray(grid)[losses.argmin()]), losses


@dataclass
class GraphicalPointProcess:
    """Estimator for the graphical point process model (1).

    ``gamma="cv"`` picks each node's penalty by K-fold CV; a number (or one per
    customer-initiated node) fixes it. ``refit=True`` re-estimates the selected
    edges without the penalty (relaxed lasso), removing the shrinkage bias.
    """

    decay: float | Sequence[float]
    gamma: float | Sequence[float] | str = "cv"
    refit: bool = False
    n_folds: int = 5
    seed: int = 0
    params_: GPPParams | None = field(default=None, init=False)
    cv_: GammaCV | None = field(default=None, init=False)

    def fit(self, paths: Sequence[Path], p: int, q: int) -> "GraphicalPointProcess":
        if not 1 <= q <= p:
            raise ValueError("need 1 <= q <= p")
        if isinstance(self.gamma, str):
            if self.gamma != "cv":
                raise ValueError("gamma must be a number, a sequence, or 'cv'")
            self.cv_ = cv_gamma(paths, p, q, self.decay, self.n_folds, seed=self.seed)
            gamma = self.cv_.best
        else:
            gamma = self.gamma
        stats = sufficient_stats(paths, p, self.decay)
        self.params_ = fit_params(stats, q, self.decay, gamma, self.refit)
        return self
