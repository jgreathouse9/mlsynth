"""Coordinate-descent cross-validation for NSC tuning parameters.

Implements Tian (2023, page 13) selection of :math:`(a^*, b^*)` exactly
as in the reference R implementation (``NSC.R``, function ``fn_cv``):

1. Initialise ``b_star = 0``.
2. Sweep ``a_star`` on the grid ``[0, 1]`` with step ``grid_size``;
   pick the value minimising the CV-MSPE.
3. Holding ``a_star`` fixed, sweep ``b_star`` on the same grid.
4. Iterate steps 2-3 until ``(a_star, b_star)`` stops moving (or a
   max-iteration budget is reached).

CV objective (R-faithful, "controls" target)
--------------------------------------------

For each donor ``j``:

* Treat donor ``j`` as a pseudo-treated unit. The fitting target is its
  pre-period matching vector.
* The donor pool for the fit is the *other* ``J - 1`` donors PLUS one
  randomly drawn extra donor (with replacement-style indexing), so the
  per-fold pool size stays at ``J`` -- matching the dimensionality of
  the real treated-unit fit and keeping the eigenvalue-based scaling of
  ``(a, b)`` consistent across folds (NSC.R lines 110-115).
* Fit NSC weights at the candidate ``(a, b)`` on PRE-period matching
  data, then evaluate the **post-period** MSPE: actual donor ``j``
  outcomes vs. those weights applied to the other donors'
  post-period outcomes.

This is a *true* held-out test error -- the post-period outcomes never
enter the fit. The legacy ``target="treated"`` option (which scored on
the same data it fit to) has been removed; passing it now raises.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from .optimization import (
    design_eigenvalues,
    fit_nsc,
)
from .structures import NSCCVTrace


def _draw_extra_donor(rng: np.random.Generator, J: int, j: int) -> int:
    """Sample one donor index from ``{0, ..., J-1} \\ {j}``.

    Mirrors R's ``sample(setdiff(1:J, j), 1)``.
    """
    pool = np.array([k for k in range(J) if k != j])
    return int(rng.choice(pool))


def _cv_score_controls(
    a_star: float,
    b_star: float,
    Z0: np.ndarray,
    Y_donors: np.ndarray,
    T0: int,
    rng: np.random.Generator,
) -> float:
    """R-faithful CV: per-donor leave-one-out, MSPE on held-out POST.

    Parameters
    ----------
    a_star, b_star : float
        Candidate dimensionless tuning parameters.
    Z0 : np.ndarray
        Standardized donor matching matrix, shape ``(J, p)``. Rows are
        donors.
    Y_donors : np.ndarray
        Full-period donor outcomes, shape ``(T, J)``. Rows are time
        periods; columns are donors. Pre-period rows are ``[:T0]``,
        post-period rows are ``[T0:]``.
    T0 : int
        Number of pre-treatment periods.
    rng : np.random.Generator
        RNG used for the extra-donor draws (one draw per fold).

    Returns
    -------
    float
        ``sqrt(mean(per-fold post-period MSPE))``. ``+inf`` on solver
        failure.
    """
    J = Z0.shape[0]
    if J < 3:
        return np.inf
    n_post = Y_donors.shape[0] - T0
    if n_post <= 0:
        return np.inf

    sq_err = []
    for j in range(J):
        other = np.array([k for k in range(J) if k != j])
        idx = _draw_extra_donor(rng, J, j)
        # Fit pool: J - 1 other donors plus one duplicated extra.
        ZJ = np.vstack([Z0[other], Z0[idx][None, :]])
        Z1_j = Z0[j]
        try:
            eig = design_eigenvalues(ZJ)
            w, _, _ = fit_nsc(Z1_j, ZJ, a_star, b_star, eigvals=eig)
        except Exception:
            return np.inf
        # Score: predict donor j's POST outcomes from the same pool.
        Y_pool_post = np.column_stack(
            [Y_donors[T0:, other], Y_donors[T0:, idx][:, None]]
        )
        Y_pred_post = Y_pool_post @ w
        Y_true_post = Y_donors[T0:, j]
        sq_err.append(float(np.mean((Y_true_post - Y_pred_post) ** 2)))

    return float(np.sqrt(np.mean(sq_err)))


def cv_select(
    Z1: np.ndarray,
    Z0: np.ndarray,
    Y_donors: np.ndarray,
    T0: int,
    grid_size: float = 0.1,
    max_iterations: int = 3,
    target: str = "controls",
    seed: Union[int, np.random.Generator, None] = 123,
) -> Tuple[float, float, NSCCVTrace]:
    """Coordinate-descent selection of ``(a_star, b_star)`` on ``[0, 1]``.

    Parameters
    ----------
    Z1 : np.ndarray
        Treated matching vector, length ``p``. Currently unused by the
        R-faithful "controls" target (kept in the signature for symmetry
        with :func:`fit_nsc` and for future use).
    Z0 : np.ndarray
        Standardized donor matching matrix, shape ``(J, p)``.
    Y_donors : np.ndarray
        Full-period donor outcome matrix, shape ``(T, J)``.
    T0 : int
        Number of pre-treatment periods.
    grid_size : float, default 0.1
        Step of the search grid; the candidate set is
        ``[0, grid_size, 2*grid_size, ..., 1]``.
    max_iterations : int, default 3
        Hard cap on coordinate-descent iterations.
    target : {"controls"}
        Only the R-faithful "controls" target is supported. The legacy
        "treated" target (training-error MSPE on the treated unit's
        pretreatment fit) has been removed.
    seed : int, Generator, or None, default 123
        Seed (or RNG) for the extra-donor draws. Defaults to ``123`` to
        match the reference R script's ``set.seed(123)`` convention.

    Returns
    -------
    a_star : float
        Optimal dimensionless L1 multiplier on ``[0, 1]``.
    b_star : float
        Optimal dimensionless L2 multiplier on ``[0, 1]``.
    trace : NSCCVTrace
        Coordinate-descent diagnostics.
    """
    if target != "controls":
        raise ValueError(
            f"NSC cv_select supports only target='controls' "
            f"(R-faithful held-out POST MSPE); got {target!r}."
        )
    if not 0.0 < grid_size <= 0.5:
        raise ValueError(
            f"grid_size must lie in (0, 0.5]; got {grid_size}."
        )
    if T0 <= 0 or T0 >= Y_donors.shape[0]:
        raise ValueError(
            f"T0={T0} must satisfy 0 < T0 < T={Y_donors.shape[0]}."
        )

    grid = np.round(np.arange(0.0, 1.0 + grid_size / 2.0, grid_size), 6)

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    a_star, b_star = 0.0, 0.0
    a_curve = np.full(grid.size, np.inf)
    b_curve = np.full(grid.size, np.inf)
    converged = False
    iterations = 0

    prev_a, prev_b = -1.0, -1.0
    for _ in range(max_iterations):
        iterations += 1
        # Sweep a* | b*. Each grid evaluation uses a fresh fold of
        # extra-donor draws -- mirrors R's repeated calls to fn_cv
        # within fn_tuning_a (each call burns J samples from the
        # global RNG, so the per-grid-point draws are different).
        for i, a in enumerate(grid):
            a_curve[i] = _cv_score_controls(
                float(a), b_star, Z0, Y_donors, T0, rng
            )
        a_star = float(grid[int(np.argmin(a_curve))])

        # Sweep b* | a*.
        for j, b in enumerate(grid):
            b_curve[j] = _cv_score_controls(
                a_star, float(b), Z0, Y_donors, T0, rng
            )
        b_star = float(grid[int(np.argmin(b_curve))])

        if a_star == prev_a and b_star == prev_b:
            converged = True
            break
        prev_a, prev_b = a_star, b_star

    return a_star, b_star, NSCCVTrace(
        a_grid=grid.copy(),
        b_grid=grid.copy(),
        a_mspe_curve=a_curve.copy(),
        b_mspe_curve=b_curve.copy(),
        iterations=iterations,
        converged=bool(converged),
        target=target,
    )
