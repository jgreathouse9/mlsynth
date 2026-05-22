"""Coordinate-descent cross-validation for NSC tuning parameters.

Implements Tian (2023, page 13) selection of :math:`(a^*, b^*)`:

1. Initialise ``b_star = 0``.
2. Sweep ``a_star`` on the grid ``[0, 1]`` with step ``grid_size``;
   pick the value minimising the CV-MSPE.
3. Holding ``a_star`` fixed, sweep ``b_star`` on the same grid.
4. Iterate steps 2-3 until ``(a_star, b_star)`` stops moving (or a
   max-iteration budget is reached).

Two CV targets are exposed, matching the paper's choices:

* ``"controls"`` (default) -- for each donor ``k``, predict its
  pretreatment outcome series using the *other* donors via NSC
  weights with the candidate ``(a, b)``. Sum the MSPE across donors.
* ``"treated"`` -- predict the treated unit's pretreatment outcomes
  using all donors via NSC weights with the candidate ``(a, b)``.
  Less robust (the same outcomes are used for fitting and scoring
  on the treated side) but cheaper.

The function returns the optimal ``(a_star, b_star)`` plus a trace
object useful for diagnostics.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .optimization import (
    design_eigenvalues,
    fit_nsc,
)
from .structures import NSCCVTrace


def _cv_score_at(
    a_star: float,
    b_star: float,
    Z1: np.ndarray,
    Z0: np.ndarray,
    eigvals: np.ndarray,
    target: str,
) -> float:
    """MSPE at one ``(a_star, b_star)`` candidate."""
    if target == "treated":
        try:
            w, _, _ = fit_nsc(Z1, Z0, a_star, b_star, eigvals=eigvals)
        except Exception:
            return np.inf
        pred = Z0.T @ w
        return float(np.mean((Z1 - pred) ** 2))

    if target == "controls":
        J = Z0.shape[0]
        if J < 3:
            return np.inf
        sse = 0.0
        n = 0
        for k in range(J):
            mask = np.ones(J, dtype=bool)
            mask[k] = False
            Z1_loo = Z0[k]
            Z0_loo = Z0[mask]
            eig_loo = design_eigenvalues(Z0_loo)
            try:
                w, _, _ = fit_nsc(Z1_loo, Z0_loo, a_star, b_star, eigvals=eig_loo)
            except Exception:
                return np.inf
            pred = Z0_loo.T @ w
            resid = Z1_loo - pred
            sse += float(resid @ resid)
            n += resid.size
        if n == 0:
            return np.inf
        return sse / n

    raise ValueError(
        f"Unknown CV target {target!r}; expected 'controls' or 'treated'."
    )


def cv_select(
    Z1: np.ndarray,
    Z0: np.ndarray,
    grid_size: float = 0.1,
    max_iterations: int = 3,
    target: str = "controls",
) -> Tuple[float, float, NSCCVTrace]:
    """Coordinate-descent selection of ``(a_star, b_star)`` on ``[0, 1]``.

    Parameters
    ----------
    Z1, Z0 : np.ndarray
        Matching vector / matrix passed straight to
        :func:`fit_nsc`.
    grid_size : float, default 0.1
        Step of the search grid; the candidate set is
        ``[0, grid_size, 2*grid_size, ..., 1]``.
    max_iterations : int, default 3
        Hard cap on coordinate-descent iterations.
    target : {"controls", "treated"}
        CV target as described above.

    Returns
    -------
    a_star : float
        Optimal dimensionless L1 multiplier on ``[0, 1]``.
    b_star : float
        Optimal dimensionless L2 multiplier on ``[0, 1]``.
    trace : NSCCVTrace
        Coordinate-descent diagnostics.
    """

    if not 0.0 < grid_size <= 0.5:
        raise ValueError(
            f"grid_size must lie in (0, 0.5]; got {grid_size}."
        )
    eigvals = design_eigenvalues(Z0)
    grid = np.round(np.arange(0.0, 1.0 + grid_size / 2.0, grid_size), 6)

    a_star, b_star = 0.0, 0.0
    a_curve = np.full(grid.size, np.inf)
    b_curve = np.full(grid.size, np.inf)
    converged = False
    iterations = 0

    prev_a, prev_b = -1.0, -1.0
    for _ in range(max_iterations):
        iterations += 1
        # Sweep a* | b*.
        for i, a in enumerate(grid):
            a_curve[i] = _cv_score_at(float(a), b_star, Z1, Z0, eigvals, target)
        a_star = float(grid[int(np.argmin(a_curve))])

        # Sweep b* | a*.
        for j, b in enumerate(grid):
            b_curve[j] = _cv_score_at(a_star, float(b), Z1, Z0, eigvals, target)
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
