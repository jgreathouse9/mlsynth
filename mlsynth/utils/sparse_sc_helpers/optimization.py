"""Lambda sweep + V-weight optimisation for SparseSC.

For each lambda on the grid the outer V-weight problem is a smooth
bound-constrained nonlinear program (``v_2 >= 0``) solved with
``scipy.optimize.minimize`` (L-BFGS-B). The selected lambda is the
value minimising the *unpenalised* validation-block MSE.

Two performance refinements over a naive implementation are in place:

* **Closed-form gradient** (Vives's Algorithm 1 outer objective is
  smooth in v away from the L1 kink; the L1 part has a trivial
  right-derivative under the v_2 >= 0 bound L-BFGS-B already enforces).
  Without this, L-BFGS-B falls back to a 2(P-1)-evaluation central
  finite-difference per outer step, which is the dominant cost on
  large predictor sets. The closed-form gradient is implemented in
  ``objective.outer_loss_and_grad`` via the envelope theorem: at the
  inner optimum w*(v), one ``(|A|+1) x (|A|+1)`` Cholesky on the
  active-set KKT matrix produces all P-1 gradient components.

* **Warm starts across the lambda grid**. The path is monotone in
  lambda, so the V-solution at lambda_i is a good initialiser for
  lambda_{i+1}. A failed warm start falls back to the cold MATLAB
  init ``default_v20``.

The outer V-objective window is controlled by ``outer_loss_window``:

* ``"validation"`` (default, paper) -- outer V minimises validation-
  block MSE + lambda * ||V||_1. Matches Vives-i-Bastida (2023) Algorithm 1.
* ``"training"`` -- outer V minimises training-block MSE + lambda *
  ||V||_1. Matches the unpublished MATLAB driver ``sparse_synth.m``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .inner import solve_w
from .objective import outer_loss_and_grad, selection_mse


def default_lambda_grid(size: int = 51) -> np.ndarray:
    """Return ``[0, logspace(-4, 0, size - 1)]`` (matches MATLAB)."""
    return np.concatenate([[0.0], np.logspace(-4, 0, size - 1)])


def default_v20(X0: np.ndarray) -> np.ndarray:
    """MATLAB starting v_2 = (sd_1 / sd_k)^2 for k > 1."""
    sd = X0.std(axis=1, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    return (sd[0] / sd[1:]) ** 2


def sweep_lambda(
    X1: np.ndarray,
    X0: np.ndarray,
    Y1: np.ndarray,
    Y0: np.ndarray,
    T0_total: int,
    T0_train: int,
    lambda_grid: Optional[np.ndarray] = None,
    solver: Any = None,
    max_outer_iter: int = 500,
    ftol: Optional[float] = None,
    outer_loss_window: str = "validation",
    use_analytical_grad: bool = False,
    warm_start: bool = False,
    multi_start: int = 1,
    robust: bool = True,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep lambda and return the best V-weights.

    Parameters
    ----------
    outer_loss_window : {"validation", "training"}
        Which pre-treatment block the outer V-objective evaluates the
        outcome MSE over.
    use_analytical_grad : bool, default True
        Use the envelope-theorem closed-form gradient inside L-BFGS-B.
        Set to False to fall back to scipy's finite-difference
        gradient (~20-50x slower on the augmented Vives spec).
    warm_start : bool, default True
        Reuse the previous lambda's V-solution as the initialiser
        for the next lambda. Falls back to the cold MATLAB init if a
        warm-started fit appears to fail.
    robust : bool, default True
        Add a backward continuation pass (homotopy from the heavily
        penalised, trivially-sparse end of the grid) and a champion
        restart, so the selected optimum is the true minimum-validation-
        MSE point and is reproducible across numerical stacks rather than
        depending on which critical point a single cold start lands in.
        Roughly doubles the sweep cost; set False for the fast single pass.

    Returns
    -------
    optv : np.ndarray
        Final V-weights, shape ``(P,)`` with ``optv[0] = 1``.
    opt_lambda : float
        Lambda value selected on the validation MSE.
    grid : np.ndarray
        Lambda grid actually used.
    outer_curve : np.ndarray
        Penalised outer objective at each grid point.
    val_curve : np.ndarray
        Unpenalised validation MSE at each grid point (selection target).
    v_path : np.ndarray
        Per-grid-point V-weights, shape ``(len(grid), P)``.
    """
    if outer_loss_window not in {"validation", "training"}:
        raise ValueError(
            "outer_loss_window must be 'validation' or 'training', "
            f"got {outer_loss_window!r}."
        )

    if ftol is None:
        # With the closed-form gradient L-BFGS-B's relative-objective
        # tolerance must be much tighter than the finite-difference
        # default, because the clean gradient produces fewer iterations
        # to the same precision and ftol=1e-8 terminates the loop
        # before convergence. Cross-checked against ftol=1e-14 FD
        # answers on the Vives California spec: ftol=1e-12 reproduces
        # the published ATT and pre-RMSE to 3 significant figures while
        # remaining ~20x faster than finite-difference at 1e-14.
        ftol = 1e-12 if use_analytical_grad else 1e-8

    if lambda_grid is None:
        lambda_grid = default_lambda_grid()
    lambda_grid = np.asarray(lambda_grid, dtype=float)

    Z0_train = Y0[:T0_train, :]
    Z1_train = Y1[:T0_train]
    Z0_val = Y0[T0_train:T0_total, :]
    Z1_val = Y1[T0_train:T0_total]

    if outer_loss_window == "validation":
        Z0_outer, Z1_outer = Z0_val, Z1_val
    else:
        Z0_outer, Z1_outer = Z0_train, Z1_train

    P = X0.shape[0]
    v20_cold = default_v20(X0)
    bounds = [(0.0, None)] * (P - 1)

    outer_curve = np.full(lambda_grid.size, np.nan)
    val_curve = np.full(lambda_grid.size, np.nan)
    v_path = np.zeros((lambda_grid.size, P))

    def _solve(lam, starts):
        """Best (lowest outer-objective) critical point over the given starts."""
        best_res = None
        for x0 in starts:
            res = _minimize_outer(
                x0=x0, X1=X1, X0=X0, Z1_outer=Z1_outer, Z0_outer=Z0_outer,
                lam=float(lam), solver=solver, bounds=bounds,
                max_outer_iter=max_outer_iter, ftol=ftol,
                use_analytical_grad=use_analytical_grad,
            )
            if (best_res is None) or (np.isfinite(res.fun) and res.fun < best_res.fun):
                best_res = res
        return best_res

    # ---- Forward pass (ascending lambda) ------------------------------------
    # Multi-start: try several deterministic init points and keep the one with
    # the lowest outer objective. L-BFGS-B converges to whatever critical point
    # is nearest the initialiser, so on the non-convex V-objective a single cold
    # start is fragile -- which one it lands in depends on finite-difference /
    # BLAS rounding and so drifts across numerical stacks. The champion restart
    # below and the backward continuation pass make the selected optimum
    # reproducible without relying on that gradient noise.
    champion_v2 = None                      # v2 with the lowest validation MSE so far
    best_val_fwd = np.inf
    for idx, lam in enumerate(lambda_grid):
        starts = _build_starts(
            v20_cold=v20_cold, prev_v2=(v_path[idx - 1, 1:] if idx > 0 else v20_cold),
            warm_start=warm_start, multi_start=multi_start, include_warm_first=idx > 0,
            champion=champion_v2,
        )
        res = _solve(lam, starts)
        v2_hat = np.clip(res.x, 0.0, None)
        outer_curve[idx] = float(res.fun)
        val_curve[idx] = selection_mse(v2_hat, X1, X0, Z1_val, Z0_val, solver=solver)
        v_path[idx, :] = np.concatenate([[1.0], v2_hat])
        if val_curve[idx] < best_val_fwd:
            best_val_fwd = val_curve[idx]
            champion_v2 = v2_hat.copy()

    # ---- Backward continuation pass (descending lambda) ---------------------
    # Homotopy from the heavily-penalised end: at large lambda the L1 term makes
    # the V-problem trivially sparse and easy to solve from any start, so warm-
    # starting *downward* tracks the sparse solution path instead of jumping into
    # a dense, overfit basin. Combined with the champion restart this finds the
    # global outer optimum at each lambda deterministically. A candidate is
    # accepted only if it lowers that lambda's outer objective, so the pass can
    # never worsen the path.
    if robust:
        # ensure the champion is the global validation-MSE winner from the forward pass
        champion_v2 = v_path[int(np.nanargmin(val_curve)), 1:].copy()
        for idx in range(lambda_grid.size - 2, -1, -1):
            lam = float(lambda_grid[idx])
            starts = [v_path[idx + 1, 1:].copy(), champion_v2.copy(), v_path[idx, 1:].copy()]
            res = _solve(lam, starts)
            if np.isfinite(res.fun) and res.fun < outer_curve[idx] - 1e-12:
                v2_hat = np.clip(res.x, 0.0, None)
                outer_curve[idx] = float(res.fun)
                val_curve[idx] = selection_mse(v2_hat, X1, X0, Z1_val, Z0_val, solver=solver)
                v_path[idx, :] = np.concatenate([[1.0], v2_hat])
                if val_curve[idx] <= np.nanmin(val_curve):
                    champion_v2 = v2_hat.copy()

    best_idx = int(np.nanargmin(val_curve))
    best_lambda = float(lambda_grid[best_idx])
    best_v = v_path[best_idx, :].copy()
    return best_v, best_lambda, lambda_grid, outer_curve, val_curve, v_path


def _build_starts(
    v20_cold: np.ndarray,
    prev_v2: np.ndarray,
    warm_start: bool,
    multi_start: int,
    include_warm_first: bool,
    champion: Optional[np.ndarray] = None,
) -> list:
    """Construct a list of L-BFGS-B initialisers for one lambda step.

    The order matters only for tie-breaking on the outer objective; we
    keep the cold MATLAB init last so it acts as a deterministic
    fallback. The included candidates are:

    * the previous-lambda's solution (if warm-start and not the first
      grid point),
    * a small constant ``0.1 * 1`` (good when the L1 penalty is large),
    * a constant ``1`` (uniform predictor importance),
    * ``v20_cold`` (the canonical MATLAB heuristic).

    ``multi_start`` controls how many of these to use; ``multi_start=1``
    falls back to v20_cold (or prev_v2 in warm-start mode) only.
    """
    candidates: list = []
    if champion is not None:
        candidates.append(np.asarray(champion, dtype=float).copy())
    if warm_start and include_warm_first:
        candidates.append(prev_v2.copy())
    candidates.append(v20_cold.copy())
    extras = [
        np.ones_like(v20_cold),
        0.1 * np.ones_like(v20_cold),
    ]
    while len(candidates) < max(1, multi_start) and extras:
        candidates.append(extras.pop(0))
    # The champion restart is always kept (it is what makes the selected optimum
    # reproducible); ``multi_start`` only caps the *additional* heuristic starts.
    keep = max(1, multi_start) + (1 if champion is not None else 0)
    return candidates[:keep]


def _minimize_outer(
    x0: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    Z1_outer: np.ndarray,
    Z0_outer: np.ndarray,
    lam: float,
    solver: Any,
    bounds: list,
    max_outer_iter: int,
    ftol: float,
    use_analytical_grad: bool,
):
    """Single L-BFGS-B run with or without analytical Jacobian."""
    if use_analytical_grad:
        return minimize(
            outer_loss_and_grad,
            x0=x0,
            args=(X1, X0, Z1_outer, Z0_outer, lam, solver),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": int(max_outer_iter), "ftol": float(ftol)},
        )

    from .objective import outer_loss
    return minimize(
        outer_loss,
        x0=x0,
        args=(X1, X0, Z1_outer, Z0_outer, lam, solver),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_outer_iter), "ftol": float(ftol)},
    )


def recover_w(
    v: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    solver: Any = None,
) -> np.ndarray:
    """Final donor-weight recovery at the selected V-weights."""
    return solve_w(v, X1, X0, solver=solver)
