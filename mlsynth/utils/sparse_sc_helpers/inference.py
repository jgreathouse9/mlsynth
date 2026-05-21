"""Abadie-style placebo permutation inference for SparseSC.

For each donor ``j`` we treat that donor as the placebo treated unit,
swap it into the treated slot, refit SparseSC at the *already-selected*
lambda (to keep the placebo loop tractable), and record the post-period
placebo ATT. The two-sided placebo p-value compares ``|observed|``
against the distribution of ``|placebo|``.

Refitting the full lambda sweep for every placebo would multiply the
runtime by ~50x with little inferential gain, so by default we reuse
the lambda picked on the actual treated unit. Set ``resweep=True`` to
re-run the full lambda selection inside each placebo.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .objective import training_loss, validation_mse
from .optimization import default_v20, recover_w, sweep_lambda
from scipy.optimize import minimize


def _refit_at_lambda(
    X1: np.ndarray, X0: np.ndarray,
    Z1_train: np.ndarray, Z0_train: np.ndarray,
    lam: float, solver: Any,
) -> np.ndarray:
    """Refit V-weights at a fixed lambda; return full v including v[0] = 1."""
    P = X0.shape[0]
    v20 = default_v20(X0)
    bounds = [(0.0, None)] * (P - 1)
    res = minimize(
        training_loss, x0=v20,
        args=(X1, X0, Z1_train, Z0_train, float(lam), solver),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-8},
    )
    return np.concatenate([[1.0], np.clip(res.x, 0.0, None)])


def run_placebo(
    Y0: np.ndarray, Y1: np.ndarray,
    X0: np.ndarray, X1: np.ndarray,
    T0_total: int, T0_train: int,
    selected_lambda: float,
    observed_att: float,
    solver: Any = None,
    resweep: bool = False,
    lambda_grid: np.ndarray | None = None,
    n_placebo: int | None = None,
    seed: int = 1400,
) -> Tuple[np.ndarray, float, int]:
    """Return ``(placebo_atts, p_value, n_completed)``.

    Parameters
    ----------
    Y0, Y1, X0, X1 : np.ndarray
        Full pre-standardized panel + predictor matrices.
    T0_total, T0_train : int
        Pre-treatment window bounds.
    selected_lambda : float
        Lambda chosen on the actual treated unit. Reused for each
        placebo when ``resweep=False`` (default).
    observed_att : float
        ATT of the actual treated unit, used to construct the p-value.
    resweep : bool
        If True, re-run the full lambda grid for each placebo. Slow.
    lambda_grid : np.ndarray, optional
        Grid for the resweep case.
    n_placebo : int, optional
        Subsample of donors to use as placebos. ``None`` uses every donor.
    seed : int
        Seed for the subsample when ``n_placebo < N``.
    """
    rng = np.random.default_rng(seed)
    N = X0.shape[1]
    donor_indices = np.arange(N)
    if n_placebo is not None and n_placebo < N:
        donor_indices = rng.choice(donor_indices, size=int(n_placebo),
                                   replace=False)

    placebo_list = []
    for j in donor_indices:
        # Swap donor j into the treated slot.
        X0_loo = np.delete(X0, j, axis=1)
        Y0_loo = np.delete(Y0, j, axis=1)
        X1_placebo = X0[:, j].copy()
        Y1_placebo = Y0[:, j].copy()

        Z1t = Y1_placebo[:T0_train]
        Z0t = Y0_loo[:T0_train, :]

        if resweep:
            best_v, _, _, _, _, _ = sweep_lambda(
                X1=X1_placebo, X0=X0_loo,
                Y1=Y1_placebo, Y0=Y0_loo,
                T0_total=T0_total, T0_train=T0_train,
                lambda_grid=lambda_grid, solver=solver,
            )
        else:
            try:
                best_v = _refit_at_lambda(
                    X1=X1_placebo, X0=X0_loo,
                    Z1_train=Z1t, Z0_train=Z0t,
                    lam=float(selected_lambda), solver=solver,
                )
            except Exception:
                continue

        try:
            w = recover_w(best_v, X1_placebo, X0_loo, solver=solver)
        except Exception:
            continue
        cf = Y0_loo @ w
        placebo_att = float(np.mean((Y1_placebo - cf)[T0_total:]))
        if np.isfinite(placebo_att):
            placebo_list.append(placebo_att)

    placebo_atts = np.asarray(placebo_list, dtype=float)
    if placebo_atts.size == 0 or not np.isfinite(observed_att):
        return placebo_atts, float("nan"), 0
    p_value = float(
        (np.sum(np.abs(placebo_atts) >= abs(observed_att)) + 1)
        / (placebo_atts.size + 1)
    )
    return placebo_atts, p_value, int(placebo_atts.size)
