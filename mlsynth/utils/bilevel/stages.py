"""The three stages of the Malo et al. (2024) bilevel SCM algorithm.

* :func:`unconstrained_feasibility` -- Section 3.1: solve the constrained
  regression on outcomes for the lower bound, then check (via the predictor
  LP) whether that solution is already bilevel-optimal.
* :func:`corner_solutions` -- Section 3.2: evaluate the ``K`` basic predictor
  weightings (all weight on one predictor) and keep the best by upper-level
  loss.
* :func:`tykhonov_refine` -- Section 3.3: a pragmatic regularized descent over
  the predictor simplex, used only if a gap remains. (The paper notes the
  optimum is usually a corner found earlier, so this rarely changes anything.)
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np

from .simplex import mspe, project_simplex, simplex_lstsq
from .structure import BilevelProblem

_EPS = 1e-12


def warn_on_gap(gap: float, lower_bound: float, factor: float,
                *, stacklevel: int = 2) -> None:
    """Warn when the bilevel optimality gap is a large multiple of the bound.

    A large gap means the predictor-constrained fit is far worse than the best
    achievable outcome fit -- i.e. the predictors are hard to match on the donor
    simplex and the predictor weights ``V`` are weakly identified (the pathology
    behind diverging malo/mscmt estimates). Only fires for a non-trivial bound.
    """
    if lower_bound > 1e-9 and gap / lower_bound > factor:
        warnings.warn(
            f"bilevel optimality gap is {gap / lower_bound:.1f}x the outcome "
            f"lower bound: the predictors are weakly matchable and V is poorly "
            f"identified; treat the predictor weighting with caution.",
            RuntimeWarning,
            stacklevel=stacklevel,
        )


def _basis_vector(k: int, K: int) -> np.ndarray:
    v = np.zeros(K)
    v[k] = 1.0
    return v


def _lower_level_weights(prob: BilevelProblem, V: np.ndarray, eps: float) -> np.ndarray:
    """Lower-level donor weights for predictor weights ``V`` (Eq. 8/11).

    Solves ``min_W ||X1 - X0 W||^2_V + eps * ||y1_pre - Y0_pre W||^2`` on the
    simplex. The non-Archimedean ``eps`` term selects, among predictor-optimal
    weights, the one that best fits the outcome (Proposition 2).
    """
    sq = np.sqrt(np.clip(V, 0.0, None))
    A = np.vstack([sq[:, None] * prob.X0, np.sqrt(eps) * prob.Y0_pre])
    b = np.concatenate([sq * prob.X1, np.sqrt(eps) * prob.y1_pre])
    return simplex_lstsq(A, b)


def unconstrained_feasibility(
    prob: BilevelProblem, *, feas_tol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """Section 3.1: unconstrained simplex regression + feasibility check.

    Returns ``(W_unc, V_star, lower_bound, is_optimal)``. ``W_unc`` minimizes
    the outcome fit on the simplex (Eq. 3) and ``lower_bound = L(W**)``. The
    predictor LP (Eq. 10) is linear in ``V`` over the simplex, so its optimum
    is the basis vector on the best-matched predictor; if that predictor's
    discrepancy is ~0, ``W_unc`` is the global bilevel solution.
    """
    W_unc = simplex_lstsq(prob.Y0_pre, prob.y1_pre, warn=True)
    lower_bound = mspe(prob.y1_pre, prob.Y0_pre, W_unc)

    resid = prob.X1 - prob.X0 @ W_unc            # (K,)
    sq_disc = resid ** 2
    k_star = int(np.argmin(sq_disc))
    V_star = _basis_vector(k_star, prob.n_predictors)
    is_optimal = bool(sq_disc[k_star] <= feas_tol)
    return W_unc, V_star, float(lower_bound), is_optimal


def corner_solutions(
    prob: BilevelProblem, *, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
    """Section 3.2: best of the ``K`` corner (basic) predictor weightings.

    For each predictor ``k`` set ``V = e_k``, solve the lower-level problem for
    ``W_k``, and score the upper-level loss ``L_V``. Returns
    ``(V_best, W_best, upper_loss_best, all_upper_losses)``.
    """
    K = prob.n_predictors
    losses: List[float] = []
    best_k, best_loss, best_W = 0, np.inf, None
    for k in range(K):
        W_k = _lower_level_weights(prob, _basis_vector(k, K), eps)
        L_V = mspe(prob.y1_pre, prob.Y0_pre, W_k)
        losses.append(L_V)
        if L_V < best_loss:
            best_k, best_loss, best_W = k, L_V, W_k
    return _basis_vector(best_k, K), best_W, float(best_loss), losses


def tykhonov_refine(
    prob: BilevelProblem,
    V0: np.ndarray,
    *,
    eps0: float = 1e-3,
    shrink: float = 0.5,
    outer_iters: int = 6,
    inner_iters: int = 8,
    fd_step: float = 1e-3,
    lr: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Section 3.3: pragmatic regularized descent over the predictor simplex.

    For a decreasing sequence ``eps_k -> 0`` (Proposition 4) the implicit
    objective ``L_eps(V) = L_V(V, W*_eps(V))`` is minimized by projected
    gradient descent on ``V`` (finite-difference gradient, simplex projection).
    A lighter stand-in for the full Bouligand-stationary QP of Appendix B.1;
    it only moves ``V`` when doing so lowers the outcome loss.
    """
    K = prob.n_predictors
    V = project_simplex(np.asarray(V0, dtype=float))
    W = _lower_level_weights(prob, V, eps0)
    best_loss = mspe(prob.y1_pre, prob.Y0_pre, W)
    eps = eps0
    total = 0

    for _ in range(outer_iters):
        for _ in range(inner_iters):
            total += 1
            grad = np.zeros(K)
            base = mspe(prob.y1_pre, prob.Y0_pre, _lower_level_weights(prob, V, eps))
            for k in range(K):
                Vp = V.copy()
                Vp[k] += fd_step
                grad[k] = (mspe(prob.y1_pre, prob.Y0_pre,
                                _lower_level_weights(prob, Vp, eps)) - base) / fd_step
            V_new = project_simplex(V - lr * grad)
            W_new = _lower_level_weights(prob, V_new, eps)
            loss_new = mspe(prob.y1_pre, prob.Y0_pre, W_new)
            if loss_new < best_loss - _EPS:
                V, W, best_loss = V_new, W_new, loss_new
            else:
                break
        eps *= shrink

    return V, W, float(best_loss), total
