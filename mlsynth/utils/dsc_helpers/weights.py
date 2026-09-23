"""Simplex-constrained weight solver for Distributional Synthetic Controls.

For each pre-period :math:`t \\in \\mathcal T_0`, DSC solves

.. math::

   \\widehat w_t = \\arg\\min_{w \\in \\mathcal H}
                    \\bigl\\| \\widetilde Y_t\\, w - \\widehat Y_{1t} \\bigr\\|_2^2,
   \\qquad
   \\mathcal H = \\bigl\\{ w \\in [0, 1]^J : \\mathbf 1^\\top w = 1 \\bigr\\},

where :math:`\\widetilde Y_t` is the :math:`M \\times J` donor
pseudo-sample matrix and :math:`\\widehat Y_{1t}` is the
:math:`M \\times 1` treated pseudo-sample vector (Zhang, Zhang &
Zhang 2026 eq. 3; the loss is the squared 2-Wasserstein distance
approximated by Monte Carlo / QMC).
"""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthEstimationError


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``v`` onto the probability simplex.

    Implements the exact :math:`O(J \\log J)` algorithm of Duchi,
    Shalev-Shwartz, Singer & Chandra (2008): sort, find the threshold
    via the cumulative sum, and soft-threshold. Returns ``w >= 0`` with
    ``sum(w) == 1``.
    """
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho_idx = np.nonzero(u - css / np.arange(1, v.size + 1) > 0)[0]
    rho = rho_idx[-1] if rho_idx.size else 0
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def solve_simplex_weights(
    donor_matrix: np.ndarray,
    treated_vec: np.ndarray,
    *,
    max_iter: int = 5000,
    tol: float = 1e-12,
) -> np.ndarray:
    """Return the simplex-constrained least-squares weight vector.

    Solves the convex program

    .. math::

       \\widehat w = \\arg\\min_{w \\in \\mathcal H}
                     \\| \\widetilde Y_t\\, w - \\widehat Y_{1t} \\|_2^2,
       \\qquad
       \\mathcal H = \\{ w \\ge 0 : \\mathbf 1^\\top w = 1 \\},

    by **accelerated projected gradient descent** (FISTA; Beck &
    Teboulle 2009) with the exact simplex projection of Duchi et al.
    (2008). This replaces an earlier SLSQP solver that failed
    (``"Positive directional derivative for linesearch"``) once the
    donor pool grew past a few dozen units -- precisely the regime of
    Gunsilius (2023, Section 6.1), where the method is meant to use
    tens to hundreds of donors. The reference DiSCo R package solves the
    same program with a dedicated constrained least-squares routine
    (``pracma::lsqlincon``).

    An earlier version of this docstring claimed projected gradient
    "returns the identical optimum (the objective is convex with a unique
    minimum value over the simplex)". That reasoning is wrong, and the
    error is worth keeping visible: a unique minimum *value* does not
    imply a unique *argmin*, and the argmin is what gets reported. FISTA
    reaches the same objective as an exact QP to 0.00 percent while its
    weights differ by ~1e-3, which was enough to miss the reference's
    published weights by 0.0047 (issue #304). The projected-gradient pass
    is therefore a warm start, and :func:`_refine_exact` produces the
    returned answer.

    Parameters
    ----------
    donor_matrix : np.ndarray
        :math:`(M, J)` design matrix -- donor quantile functions
        evaluated on the grid.
    treated_vec : np.ndarray
        Length-``M`` target quantile function.
    max_iter : int
        Maximum FISTA iterations.
    tol : float
        Relative objective-change stopping tolerance.

    Returns
    -------
    np.ndarray
        Length-``J`` weight vector with ``w >= 0`` and ``sum(w) == 1``.
    """
    if donor_matrix.ndim != 2 or treated_vec.ndim != 1:
        raise MlsynthEstimationError(
            "donor_matrix must be 2-D and treated_vec must be 1-D."
        )
    if donor_matrix.shape[0] != treated_vec.shape[0]:
        raise MlsynthEstimationError(
            "donor_matrix and treated_vec must have the same number of rows."
        )

    A = np.asarray(donor_matrix, dtype=float)
    b = np.asarray(treated_vec, dtype=float)
    J = A.shape[1]
    if J == 1:
        return np.ones(1)

    AtA = A.T @ A
    Atb = A.T @ b
    # Lipschitz constant of the gradient of ||A w - b||^2 is 2 * lambda_max(AtA).
    lip = 2.0 * float(np.linalg.norm(AtA, 2))
    step = 1.0 / max(lip, 1e-12)

    w = np.full(J, 1.0 / J)
    y = w.copy()
    t = 1.0
    prev_obj = np.inf
    for _ in range(max_iter):
        grad = 2.0 * (AtA @ y - Atb)
        w_new = _project_to_simplex(y - step * grad)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        w, t = w_new, t_new
        diff = A @ w - b
        obj = float(diff @ diff)
        if abs(prev_obj - obj) <= tol * (1.0 + abs(obj)):
            break
        prev_obj = obj

    w = np.clip(w, 0.0, None)
    s = w.sum()
    w = w / s if s > 0 else np.full(J, 1.0 / J)

    # FISTA converges on the OBJECTIVE, and that is not the same as converging
    # on the argmin. Measured against an exact QP on the Dube pre-periods, it
    # reaches the same objective to 0.00 percent while its weights differ by
    # ~1e-3, and on the Stata Journal tenure panel that is enough to miss the
    # reference's published weights by 0.0047. Since the weights are the
    # reported quantity, the projected-gradient result is used only as a warm
    # start and the answer comes from an exact solve.
    return _refine_exact(A, b, w)


def _refine_exact(A: np.ndarray, b: np.ndarray, warm: np.ndarray) -> np.ndarray:
    """Exact simplex least squares, falling back to ``warm`` if unavailable.

    ``cvxpy`` is already required by the DSC test path and the bilevel engine
    keeps an equivalent fallback (``_simplex_qp_cvxpy``), so this adds no new
    dependency. The fallback exists so that an environment without a working
    solver degrades to the previous behaviour rather than failing outright.
    """
    try:
        import cvxpy as cp
    except Exception:  # pragma: no cover - cvxpy is a declared dependency
        return warm
    try:
        w = cp.Variable(A.shape[1], nonneg=True)
        cp.Problem(cp.Minimize(cp.sum_squares(A @ w - b)),
                   [cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
        if w.value is None:  # pragma: no cover - degenerate
            return warm
        out = np.clip(np.asarray(w.value, dtype=float).ravel(), 0.0, None)
        total = out.sum()
        return out / total if total > 0 else warm
    except Exception:  # pragma: no cover - solver failure
        return warm


def solve_sum_to_one_weights(
    donor_matrix: np.ndarray,
    treated_vec: np.ndarray,
) -> np.ndarray:
    """Least-squares weights that sum to one, without the non-negativity bound.

    Solves

    .. math::

       \\widehat w = \\arg\\min_{w} \\| \\widetilde Y_t\\, w
                     - \\widehat Y_{1t} \\|_2^2,
       \\qquad \\mathbf 1^\\top w = 1, \\quad w \\le 1,

    the feasible set the reference ``DiSCos`` package uses by default:
    ``DiSCo_weights_reg`` passes ``lb = NULL`` unless ``simplex = TRUE`` and
    ``ub = 1`` in either case, so a weight may go negative but none may exceed
    one. Zhang, Zhang & Zhang (2026) frame the same relaxation as a bounded
    extrapolation set :math:`[-C_L, C_U]^J`, which contains the simplex, so the
    fitted loss here is never above what :func:`solve_simplex_weights` attains.

    Allowing negative weights buys fit when the treated quantile function sits
    outside the donors' convex hull, and costs the interpretability the simplex
    provides: the synthetic unit is no longer a weighted average of observed
    donors. The simplex remains the default.

    Parameters
    ----------
    donor_matrix : np.ndarray
        :math:`(M, J)` design matrix -- donor quantile functions on the grid.
    treated_vec : np.ndarray
        Length-``M`` target quantile function.

    Returns
    -------
    np.ndarray
        Length-``J`` weight vector with ``sum(w) == 1`` and ``w <= 1``.
    """
    if donor_matrix.ndim != 2 or treated_vec.ndim != 1:
        raise MlsynthEstimationError(
            "donor_matrix must be 2-D and treated_vec must be 1-D."
        )
    if donor_matrix.shape[0] != treated_vec.shape[0]:
        raise MlsynthEstimationError(
            "donor_matrix and treated_vec must have the same number of rows."
        )
    A = np.asarray(donor_matrix, dtype=float)
    b = np.asarray(treated_vec, dtype=float)
    J = A.shape[1]
    if J == 1:
        return np.ones(1)
    try:
        import cvxpy as cp
    except Exception as exc:  # pragma: no cover - cvxpy is a declared dependency
        raise MlsynthEstimationError(
            "weight_constraint='sum_to_one' needs cvxpy, which is not importable."
        ) from exc
    w = cp.Variable(J)
    cp.Problem(cp.Minimize(cp.sum_squares(A @ w - b)),
               [cp.sum(w) == 1, w <= 1]).solve(solver=cp.CLARABEL)
    if w.value is None:  # pragma: no cover - degenerate
        raise MlsynthEstimationError("The sum-to-one weight solve did not converge.")
    out = np.asarray(w.value, dtype=float).ravel()
    # Renormalise against solver slack on the equality; the residual is ~1e-12
    # and left uncorrected it would surface as weights that do not sum to one.
    return out / out.sum()


def wasserstein_loss_at_weights(
    donor_matrix: np.ndarray,
    treated_vec: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Squared 2-Wasserstein loss :math:`\\|\\widetilde Y_t w - \\widehat Y_{1t}\\|_2^2 / M`.

    The :math:`1/M` normalisation makes this comparable across periods
    with different grid sizes.
    """
    diff = donor_matrix @ weights - treated_vec
    return float((diff @ diff) / max(donor_matrix.shape[0], 1))
