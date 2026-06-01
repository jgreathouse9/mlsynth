"""Multivariate Square-root Lasso solve + rolling-origin lambda selection.

Estimator (Shen, Song & Abadie 2025, eq. 5)::

    Theta_hat = argmin_Theta  (1/sqrt(T0)) * ||Y - X Theta||_*  +  lambda * ||Theta||_1

where ``||.||_*`` is the nuclear norm (the "square-root"/pivotal loss; tuning
does not depend on the unknown noise variance) and ``||.||_1`` is the
**elementwise** L1 penalty ``sum_{i,j} |Theta_ij|`` that drives donor selection
in the high-dimensional regime.

The default solver is a two-split (consensus) **ADMM** in which every
subproblem has a closed form:

* the nuclear-norm term -> singular-value soft-thresholding of the residual,
* the L1 term          -> elementwise soft-thresholding,
* the coupling          -> an *exact* least-squares step, solved with a
  Cholesky factorisation of ``X'X + I`` that is computed **once** and reused
  across every iteration (it does not depend on the penalty parameter ``rho``).

Replacing the linearised proximal-gradient step of a one-split scheme with this
exact step collapses the iteration count from thousands to a few hundred, which
is what makes the high-dimensional, many-treated regime tractable. A ``cvxpy``
path is retained for validation against a general conic solver.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve


# ---------------------------------------------------------------------------
# Proximal operators
# ---------------------------------------------------------------------------

def _soft_threshold(Z: np.ndarray, tau: float) -> np.ndarray:
    """Elementwise soft-thresholding operator.

    Evaluates the proximal map of ``tau * ||.||_1`` (the elementwise L1 norm),

    .. math::

        \\operatorname{prox}_{\\tau\\|\\cdot\\|_1}(Z)_{ij}
            = \\operatorname{sign}(Z_{ij})\\,\\max(|Z_{ij}| - \\tau,\\ 0).

    Parameters
    ----------
    Z : numpy.ndarray
        Input array of any shape.
    tau : float
        Non-negative threshold. Entries with magnitude ``<= tau`` are set to
        exactly zero; larger entries are shrunk toward zero by ``tau``.

    Returns
    -------
    numpy.ndarray
        Array with the same shape as ``Z`` containing the thresholded values.

    Notes
    -----
    This is the closed-form solution of
    ``argmin_S  0.5 ||S - Z||_F^2 + tau ||S||_1`` and is the L1 (``Z``-block)
    update of the ADMM in :func:`fit_msqrt_admm`.
    """
    return np.sign(Z) * np.maximum(np.abs(Z) - tau, 0.0)


def _svt(W: np.ndarray, tau: float) -> np.ndarray:
    """Singular-value (matrix) soft-thresholding operator.

    Evaluates the proximal map of ``tau * ||.||_*`` (the nuclear norm). If
    ``W = U diag(s) V'`` is the (thin) SVD of ``W``, then

    .. math::

        \\operatorname{prox}_{\\tau\\|\\cdot\\|_*}(W)
            = U\\,\\operatorname{diag}\\big(\\max(s - \\tau,\\ 0)\\big)\\,V'.

    Parameters
    ----------
    W : numpy.ndarray, shape (p, q)
        Input matrix.
    tau : float
        Non-negative threshold applied to each singular value.

    Returns
    -------
    numpy.ndarray, shape (p, q)
        ``W`` with its singular values shrunk toward zero by ``tau`` (negative
        results clipped to zero), i.e. the low-rank-promoting proximal step.

    Notes
    -----
    Computed with a thin SVD (``full_matrices=False``); its cost is governed by
    the smaller dimension ``min(p, q)``. This is the nuclear-norm (``R``-block)
    update of the ADMM in :func:`fit_msqrt_admm`.
    """
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    s_thr = np.maximum(s - tau, 0.0)
    return (U * s_thr) @ Vt


# ---------------------------------------------------------------------------
# ADMM solver
# ---------------------------------------------------------------------------

def fit_msqrt_admm(
    Y: np.ndarray,
    X: np.ndarray,
    lambd: float,
    *,
    rho: float = 1.0,
    over_relax: float = 1.5,
    max_iter: int = 5000,
    abstol: float = 1e-6,
    reltol: float = 1e-4,
    accelerate: bool = False,
    eta: float = 0.999,
    adaptive_rho: bool = True,
    warm_start: Optional[dict] = None,
    return_state: bool = False,
):
    """Solve the Multivariate Square-root Lasso (eq. 5) by two-split ADMM.

    Minimises

    .. math::

        \\frac{1}{\\sqrt{T}}\\,\\lVert Y - X\\Theta\\rVert_*
            + \\lambda \\lVert \\Theta \\rVert_1

    over ``Theta in R^{n x m}``. The problem is cast in the standard ADMM form
    ``min f(x) + g(z) s.t. A x + B z = c`` with ``x = Theta``, two consensus
    blocks ``z = (R, Z)`` for ``R = X Theta`` (nuclear-norm term) and
    ``Z = Theta`` (L1 term), ``A = [X; I]``, ``B = -I`` and ``c = 0``. The
    scaled-dual updates (Boyd et al. 2011, Sec. 3.1.1) are

    * ``Theta`` -- exact least squares:
      ``(X'X + I) Theta = X'(Rhat - Uhat) + (Zhat - What)``;
    * ``R``     -- ``R = Y - svt(Y - X Theta - Uhat, (1/sqrt(T)) / rho)``;
    * ``Z``     -- ``Z = soft_threshold(Theta + What, lambda / rho)``;
    * duals     -- ``U = Uhat + (X Theta - R)``, ``W = What + (Theta - Z)``.

    where ``(Rhat, Zhat, Uhat, What)`` are the (possibly extrapolated)
    inputs to each iteration -- equal to the previous iterate for vanilla ADMM,
    or a Nesterov-momentum extrapolation when ``accelerate=True``.

    The system matrix ``M = X'X + I`` does **not** depend on ``rho``, so it is
    Cholesky-factorised **once** and reused for every iteration.

    Parameters
    ----------
    Y : numpy.ndarray, shape (T, m)
        Treated-unit outcomes, time-major (rows are periods).
    X : numpy.ndarray, shape (T, n)
        Donor (control) outcomes, time-major.
    lambd : float
        Non-negative L1 penalty ``lambda`` in eq. (5).
    rho : float, optional
        ADMM penalty parameter (default ``1.0``). Affects only the convergence
        path, not the solution. Held fixed when ``accelerate=True``.
    over_relax : float, optional
        Over-relaxation parameter ``alpha`` in ``(0, 2)`` (default ``1.5``;
        Boyd et al. 2011, Sec. 3.4.3). In the ``R``/``Z`` and dual updates the
        coupling term ``A Theta`` is replaced by
        ``alpha (A Theta) + (1 - alpha) (previous z)``. Values in ``[1.5, 1.8]``
        typically cut the iteration count appreciably at no per-iteration cost;
        ``alpha = 1`` recovers the un-relaxed scheme.
    max_iter : int, optional
        Maximum number of iterations (default ``5000``); a safety cap rarely
        reached because the exact ``Theta`` step (and acceleration) converge
        quickly.
    abstol : float, optional
        Absolute tolerance for the primal/dual stopping rule (default ``1e-6``).
    reltol : float, optional
        Relative tolerance for the primal/dual stopping rule (default ``1e-4``).
    accelerate : bool, optional
        If True, apply the Nesterov-momentum acceleration with adaptive restart
        of Goldstein, O'Donoghue, Setzer & Baraniuk (2014), "Fast Alternating
        Direction Optimization Methods", SIAM J. Imaging Sci. 7(3) (UCLA CAM
        report 12-35), Algorithm 8 (the general-convex restart variant). ``rho``
        is held fixed in this mode. **Default False**: acceleration targets the
        convergence *order*, but for this problem the limiting factor is the
        mismatch in scale between the two constraints (``R = X Theta`` is on the
        scale of the data; ``Z = Theta`` on the scale of the weights). Boyd's
        adaptive-``rho`` residual balancing (Sec. 3.4.1) -- the practical proxy
        for the per-constraint penalties of Sec. 3.4.2 -- handles that directly
        and, empirically, converges to the cvxpy optimum (objective agreeing to
        five significant figures) far faster than fixed-``rho`` acceleration.
    eta : float, optional
        Restart threshold in ``(0, 1)`` (default ``0.999``): when accelerating,
        momentum is kept while the combined residual decreases by at least a
        factor ``eta``, otherwise the iterate restarts (momentum reset). Only
        used when ``accelerate=True``.
    adaptive_rho : bool, optional
        If True (default), and only when ``accelerate=False``, apply
        residual-balancing ``rho`` updates (Boyd et al. 2011, Sec. 3.4.1):
        double/halve ``rho`` when the primal and dual residuals differ by more
        than 10x (``mu = 10``, ``tau = 2``), rescaling the scaled duals
        accordingly. This is the default solving mode. Ignored when
        ``accelerate=True``.
    warm_start : dict or None, optional
        Initial solver state to start from, e.g. the ``state`` returned by a
        previous call (keys ``R``, ``Z``, ``U``, ``W``, ``rho``). Defaults to a
        cold start (zeros). Warm starting from a nearby solution -- as in the
        pathwise cross-validation of :func:`select_lambda_cv` -- sharply reduces
        the iteration count.
    return_state : bool, optional
        If True, also return the final solver state ``dict`` (for warm-starting
        a subsequent solve). Default False.

    Returns
    -------
    numpy.ndarray, shape (n, m)
        The estimated weight matrix ``Theta`` (the L1 ``Z``-block, hence exactly
        sparse). ``X @ Theta`` is the fitted treated-outcome matrix.
    state : dict, optional
        Returned only when ``return_state=True``: the final ``R``, ``Z``, ``U``,
        ``W`` and ``rho``, suitable as a ``warm_start`` for a related problem.

    Notes
    -----
    Termination uses the primal/dual residual criterion of Boyd et al. (2011,
    Sec. 3.3): stop when ``||r||_2 <= eps_pri`` and ``||s||_2 <= eps_dual`` with
    primal residual ``r = (X Theta - R, Theta - Z)``, dual residual
    ``s = rho * A'(z^k - z^{k-1})``, and tolerances
    ``eps_pri = sqrt((T+n) m) abstol + reltol max(||A Theta||, ||B z||)`` and
    ``eps_dual = sqrt(n m) abstol + reltol ||A' (rho u)||``. The accelerated
    restart uses the combined residual
    ``c_k = rho (||u^k - uhat^k||^2 + ||z^k - zhat^k||^2)``.
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    T, m = Y.shape
    n = X.shape[1]
    scale = 1.0 / np.sqrt(T)
    # Over-relaxation and Nesterov acceleration are distinct accelerations;
    # do not stack them. Acceleration takes precedence when both are requested.
    if accelerate:
        over_relax = 1.0

    # Cache the Cholesky factor of M = X'X + I (SPD, rho-independent).
    M = X.T @ X + np.eye(n)
    c_and_lower = cho_factor(M, lower=True, check_finite=False)
    Xt = X.T

    # Initialise iterate z^{k-1} = (R, Z), scaled dual u^{k-1} = (U, W).
    if warm_start is not None:
        R = np.array(warm_start.get("R", np.zeros((T, m))), dtype=float)
        Z = np.array(warm_start.get("Z", np.zeros((n, m))), dtype=float)
        U = np.array(warm_start.get("U", np.zeros((T, m))), dtype=float)
        W = np.array(warm_start.get("W", np.zeros((n, m))), dtype=float)
        rho = float(warm_start.get("rho", rho))
    else:
        R = np.zeros((T, m))
        Z = np.zeros((n, m))
        U = np.zeros((T, m))
        W = np.zeros((n, m))
    # Extrapolation inputs zhat^k, uhat^k (== previous iterate at k=1).
    Rh, Zh, Uh, Wh = R.copy(), Z.copy(), U.copy(), W.copy()
    alpha = 1.0
    c_prev = np.inf
    iters = 0

    for _ in range(max_iter):
        iters += 1
        # --- Theta step: exact least squares via the cached Cholesky ---
        Theta = cho_solve(c_and_lower, Xt @ (Rh - Uh) + (Zh - Wh),
                          check_finite=False)
        XTheta = X @ Theta

        # --- over-relaxation (Boyd Sec. 3.4.3): blend with the previous z ---
        XTheta_or = over_relax * XTheta + (1.0 - over_relax) * Rh
        Theta_or = over_relax * Theta + (1.0 - over_relax) * Zh

        # --- z step: nuclear-norm (R) and L1 (Z) proximal updates ---
        R_new = Y - _svt(Y - XTheta_or - Uh, scale / rho)
        Z_new = _soft_threshold(Theta_or + Wh, lambd / rho)

        # --- scaled dual update (on the relaxed coupling) ---
        U_new = Uh + (XTheta_or - R_new)
        W_new = Wh + (Theta_or - Z_new)

        # --- convergence on the true (un-relaxed) residuals (Boyd Sec. 3.3) ---
        r_primal_R = XTheta - R_new
        r_primal_Z = Theta - Z_new
        r_norm = np.sqrt(np.sum(r_primal_R ** 2) + np.sum(r_primal_Z ** 2))
        s_norm = rho * np.linalg.norm(Xt @ (R_new - R) + (Z_new - Z))
        eps_pri = np.sqrt((T + n) * m) * abstol + reltol * np.sqrt(
            max(np.sum(XTheta ** 2) + np.sum(Theta ** 2),
                np.sum(R_new ** 2) + np.sum(Z_new ** 2)))
        eps_dual = np.sqrt(n * m) * abstol + reltol * rho * np.linalg.norm(
            Xt @ U_new + W_new)
        if r_norm <= eps_pri and s_norm <= eps_dual:
            R, Z, U, W = R_new, Z_new, U_new, W_new
            break

        if accelerate:
            # Combined residual (Goldstein et al. 2014, restart test).
            c_k = rho * (np.sum((U_new - Uh) ** 2) + np.sum((W_new - Wh) ** 2)
                         + np.sum((R_new - Rh) ** 2) + np.sum((Z_new - Zh) ** 2))
            if c_k < eta * c_prev:
                alpha_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * alpha ** 2))
                beta = (alpha - 1.0) / alpha_new
                Rh = R_new + beta * (R_new - R)
                Zh = Z_new + beta * (Z_new - Z)
                Uh = U_new + beta * (U_new - U)
                Wh = W_new + beta * (W_new - W)
                alpha = alpha_new
                c_prev = c_k
            else:
                # Restart: drop momentum, fall back to the previous iterate.
                alpha = 1.0
                Rh, Zh, Uh, Wh = R, Z, U, W
                c_prev = c_prev / eta
            R, Z, U, W = R_new, Z_new, U_new, W_new
        else:
            R, Z, U, W = R_new, Z_new, U_new, W_new
            if adaptive_rho:
                if r_norm > 10.0 * s_norm:
                    rho *= 2.0
                    U /= 2.0
                    W /= 2.0
                elif s_norm > 10.0 * r_norm:
                    rho *= 0.5
                    U *= 2.0
                    W *= 2.0
            Rh, Zh, Uh, Wh = R, Z, U, W

    if return_state:
        return Z, {"R": R, "Z": Z, "U": U, "W": W, "rho": rho, "iters": iters}
    return Z


# ---------------------------------------------------------------------------
# cvxpy solver (validation / fallback)
# ---------------------------------------------------------------------------

def _fit_msqrt_cvxpy(Y: np.ndarray, X: np.ndarray, lambd: float) -> np.ndarray:
    """Reference solve of eq. (5) via cvxpy's conic reformulation.

    Provided for validating :func:`fit_msqrt_admm`; far slower because the
    nuclear norm is lifted to a semidefinite cone and handed to a general
    interior-point solver (CLARABEL).

    Parameters
    ----------
    Y : numpy.ndarray, shape (T, m)
        Treated-unit outcomes, time-major.
    X : numpy.ndarray, shape (T, n)
        Donor outcomes, time-major.
    lambd : float
        Non-negative L1 penalty.

    Returns
    -------
    numpy.ndarray, shape (n, m)
        The estimated weight matrix ``Theta``.

    Raises
    ------
    RuntimeError
        If the solver does not report an (inaccurate-)optimal status.

    Notes
    -----
    The penalty is ``cp.sum(cp.abs(Theta))`` -- the **elementwise** L1 of
    eq. (5). Note that ``cp.norm(Theta, 1)`` on a *matrix* would instead be the
    induced 1-norm (maximum absolute column sum), a different objective.
    """
    import cvxpy as cp

    T, m = Y.shape
    n = X.shape[1]
    Theta = cp.Variable((n, m))
    loss = (1.0 / np.sqrt(T)) * cp.norm(Y - X @ Theta, "nuc")
    reg = lambd * cp.sum(cp.abs(Theta))
    prob = cp.Problem(cp.Minimize(loss + reg))
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"MSQRT solve did not converge (status={prob.status}).")
    return np.asarray(Theta.value, dtype=float)


def fit_msqrt_weights(
    Y: np.ndarray,
    X: np.ndarray,
    lambd: float,
    *,
    tol: float = 1e-2,
    solver: str = "admm",
    **solver_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve eq. (5) for the donor-weight matrix and report fit + sparsity.

    Parameters
    ----------
    Y : numpy.ndarray, shape (T, m)
        Treated-unit outcomes, time-major.
    X : numpy.ndarray, shape (T, n)
        Donor outcomes, time-major.
    lambd : float
        Non-negative L1 penalty ``lambda``.
    tol : float, optional
        Magnitude below which a weight is treated as zero when counting active
        donors (default ``1e-2``). Does not affect the solve.
    solver : {"admm", "cvxpy"}, optional
        Which backend to use. ``"admm"`` (default) is the fast two-split solver
        :func:`fit_msqrt_admm`; ``"cvxpy"`` is the slow conic reference
        :func:`_fit_msqrt_cvxpy`.
    **solver_kwargs
        Extra keyword arguments forwarded to :func:`fit_msqrt_admm` (ignored for
        the cvxpy backend), e.g. ``rho``, ``max_iter``, ``abstol``, ``reltol``.

    Returns
    -------
    Theta_hat : numpy.ndarray, shape (n, m)
        Estimated weight matrix.
    Y_hat : numpy.ndarray, shape (T, m)
        Fitted treated outcomes ``X @ Theta_hat``.
    nonzero_per_col : numpy.ndarray, shape (m,)
        Number of active donors (``|Theta_ij| > tol``) for each treated unit.

    Raises
    ------
    ValueError
        If ``solver`` is not ``"admm"`` or ``"cvxpy"``.
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)

    if solver == "admm":
        Theta_hat = fit_msqrt_admm(Y, X, lambd, **solver_kwargs)
    elif solver == "cvxpy":
        Theta_hat = _fit_msqrt_cvxpy(Y, X, lambd)
    else:
        raise ValueError(f"Unknown solver {solver!r} (use 'admm' or 'cvxpy').")

    Y_hat = X @ Theta_hat
    nonzero_per_col = np.sum(np.abs(Theta_hat) > tol, axis=0)
    return Theta_hat, Y_hat, nonzero_per_col


# ---------------------------------------------------------------------------
# Cross-validation of the penalty
# ---------------------------------------------------------------------------

def _cv_schedule(
    T0: int,
    initial_train: Optional[int],
    val_window: Optional[int],
    step: Optional[int],
) -> Tuple[int, int, int]:
    """Resolve a rolling-origin cross-validation schedule for the pre-period.

    Fills in any unspecified schedule parameters with defaults that scale to the
    available number of pre-treatment periods ``T0``, and guarantees the initial
    training window leaves room for at least one validation block.

    Parameters
    ----------
    T0 : int
        Number of pre-treatment periods available for cross-validation.
    initial_train : int or None
        Length of the first (expanding) training window. If None, defaults to
        ``max(2, round(0.6 * T0))``.
    val_window : int or None
        Number of periods in each validation block. If None, defaults to
        ``max(1, T0 // 5)``.
    step : int or None
        Number of periods by which the training window expands between folds.
        If None, defaults to ``val_window``.

    Returns
    -------
    tuple of int
        ``(initial_train, val_window, step)`` with all values resolved to ints
        and ``initial_train`` capped at ``max(2, T0 - val_window)``.
    """
    val_window = val_window or max(1, T0 // 5)
    initial_train = initial_train or max(2, int(round(0.6 * T0)))
    step = step or val_window
    initial_train = min(initial_train, max(2, T0 - val_window))
    return int(initial_train), int(val_window), int(step)


def select_lambda_cv(
    Y_pre: np.ndarray,
    X_pre: np.ndarray,
    lambdas: Sequence[float],
    *,
    initial_train: Optional[int] = None,
    val_window: Optional[int] = None,
    step: Optional[int] = None,
    n_folds: Optional[int] = None,
    solver: str = "admm",
) -> float:
    """Select the L1 penalty by rolling-origin cross-validation on the pre-period.

    Uses an expanding training window: for each fold the model is fit on
    ``Y_pre[:train_end]`` and scored by mean squared error on the next
    ``val_window`` periods. The penalty minimising the mean validation MSE over
    all folds is returned.

    Parameters
    ----------
    Y_pre : numpy.ndarray, shape (T0, m)
        Pre-treatment treated-unit outcomes, time-major.
    X_pre : numpy.ndarray, shape (T0, n)
        Pre-treatment donor outcomes, time-major.
    lambdas : sequence of float
        Candidate penalty values to search.
    initial_train, val_window, step : int or None, optional
        Rolling-origin schedule overrides; see :func:`_cv_schedule` for the
        adaptive defaults used when these are None.
    n_folds : int or None, optional
        If given, use at most this many folds (the earliest ones).
    solver : {"admm", "cvxpy"}, optional
        Backend forwarded to the solver (default ``"admm"``).

    Returns
    -------
    float
        The cross-validated penalty. Falls back to ``min(lambdas)`` when the
        pre-period is too short to form any fold.

    Notes
    -----
    With the ADMM backend the search is **pathwise**: within each fold the
    candidate penalties are visited in *descending* order and each solve is
    warm-started from the previous (larger-``lambda``, sparser) solution. Since
    neighbouring penalties have nearby solutions, this cuts the per-solve
    iteration count substantially relative to cold starts, without changing the
    selected penalty.
    """
    Y_pre = np.asarray(Y_pre, dtype=float)
    X_pre = np.asarray(X_pre, dtype=float)
    T0 = Y_pre.shape[0]
    lambdas = list(lambdas)

    init, val, stp = _cv_schedule(T0, initial_train, val_window, step)
    fold_starts = list(range(init, T0 - val + 1, stp))
    if n_folds is not None:
        fold_starts = fold_starts[:n_folds]
    if not fold_starts:
        return float(min(lambdas))

    # Descending penalty path so each warm start is a sparser nearby solution.
    order = sorted(range(len(lambdas)), key=lambda i: -lambdas[i])
    totals = [0.0] * len(lambdas)
    for tr_end in fold_starts:
        Ytr, Xtr = Y_pre[:tr_end], X_pre[:tr_end]
        Yval, Xval = Y_pre[tr_end:tr_end + val], X_pre[tr_end:tr_end + val]
        state = None
        for i in order:
            lam = lambdas[i]
            if solver == "admm":
                Theta, state = fit_msqrt_admm(
                    Ytr, Xtr, lam, warm_start=state, return_state=True)
            else:
                Theta = _fit_msqrt_cvxpy(Ytr, Xtr, lam)
            err = Yval - Xval @ Theta
            totals[i] += float(np.mean(err ** 2))

    n_folds_used = len(fold_starts)
    best_i = min(range(len(lambdas)), key=lambda i: totals[i] / n_folds_used)
    return float(lambdas[best_i])
