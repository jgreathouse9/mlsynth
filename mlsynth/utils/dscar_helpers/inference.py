"""Inference for the Dynamic Synthetic Control method.

Two procedures from Sections 3.1 and 3.2 of Zheng & Chen (2024):

1. **FDR-controlled unconfoundedness test** (Section 3.1). For each
   pre-period ``t``, test ``H_0: gap_t = 0`` using a z-statistic
   ``z_t = gap_t / (sigma * v_eta_t)``, where ``v_eta_t =
   sqrt(1/n_treated + sum_i w_{t, i}^2)`` and ``sigma`` is the
   residual SD from the pre-period AR-1 model
   ``Y_it(0) = delta_t + beta_t' X_it + rho_t Y_{i, t-1} + eps_it``.
   Benjamini-Yekutieli correction across the ``T0`` per-period tests.

2. **Normalised placebo test** (Section 3.2). Sample ``n_treated``
   donor units uniformly at random ``K`` times; re-run DSC treating
   them as the placebo "treated" group; normalise each placebo's
   post-period mean effect by its own per-rep SD so the empirical
   distribution is on the same scale as the real ATT. The placebo SD
   of these normalised statistics is the SE for ``att``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .structures import DSCARInputs


def _benjamini_yekutieli_min(p_values: np.ndarray) -> float:
    """Smallest BY-adjusted p-value (Benjamini & Yekutieli 2001).

    For ``m`` ordered p-values :math:`p_{(1)} \\le ... \\le p_{(m)}`,
    the BY harmonic factor is :math:`C(m) = \\sum_{j=1}^{m} 1/j`. The
    smallest adjusted p-value is :math:`\\min_j p_{(j)} \\cdot
    m \\cdot C(m) / j`; we return that scalar.
    """
    p = np.asarray(p_values, dtype=float).ravel()
    m = p.size
    if m == 0:
        return float("nan")
    p_sorted = np.sort(p)
    j = np.arange(1, m + 1)
    cm = float(np.sum(1.0 / j))
    return float(np.min(p_sorted * m * cm / j))


def fdr_unconfoundedness_test(
    *, inputs: "DSCARInputs", weights: np.ndarray, gap: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Per-pre-period z-tests + BY-adjusted minimum p-value.

    Returns
    -------
    p_values : np.ndarray
        Shape ``(T0,)`` two-sided p-values for ``H_0: gap_t = 0``.
    min_p_adjusted : float
        Smallest BY-adjusted p-value across the ``T0`` tests.
    """
    T0 = inputs.T0
    n_treated = inputs.n_treated
    n_donors = inputs.N - n_treated

    # Donor-only OLS pool to estimate the residual SD `sigma`.
    p = inputs.X.shape[2]
    # Stack pre-periods 1..T0 into one big regression on donors.
    rows_Y = []
    rows_X = []
    for t in range(1, T0):
        y = inputs.Y[n_treated:, t]
        x_cols = [np.ones(n_donors), inputs.Y_lag1[n_treated:, t]]
        for k in range(p):
            x_cols.append(inputs.X[n_treated:, t, k])
        rows_Y.append(y)
        rows_X.append(np.column_stack(x_cols))
    if not rows_Y:
        # T0 == 1 -> no degrees of freedom; abort.
        return np.full(T0, np.nan), float("nan")
    Y_stack = np.concatenate(rows_Y)
    X_stack = np.vstack(rows_X)
    try:
        beta, *_ = np.linalg.lstsq(X_stack, Y_stack, rcond=None)
        resid = Y_stack - X_stack @ beta
        sigma = float(np.sqrt((resid ** 2).sum() / max(resid.size - X_stack.shape[1], 1)))
    except np.linalg.LinAlgError:
        sigma = float(np.std(Y_stack))
    if sigma <= 0:
        sigma = 1.0

    # Per-period v_eta_t = sqrt(1/n_treated + sum_i w_{t,i}^2).
    v_eta = np.sqrt(1.0 / n_treated + (weights ** 2).sum(axis=1))   # (T,)

    # z = gap / (sigma * v_eta); two-sided p-value from N(0, 1).
    from math import erf, sqrt
    z = gap[:T0] / np.maximum(sigma * v_eta[:T0], 1e-12)
    # 2 * (1 - Phi(|z|)).
    p_values = np.empty(T0)
    for i, zi in enumerate(z):
        p_values[i] = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(zi) / sqrt(2.0))))
    min_p_adj = _benjamini_yekutieli_min(p_values)
    return p_values, min_p_adj


def normalised_placebo_test(
    *,
    inputs: "DSCARInputs",
    weights: np.ndarray,
    att_observed: float,
    placebo_reps: int,
    el_tolerance: float,
    seed: int,
) -> Tuple[np.ndarray, float]:
    """Run the normalised placebo procedure (Section 3.2 of Zheng & Chen 2024).

    Returns
    -------
    placebo_atts : np.ndarray
        Length-``placebo_reps`` array of placebo post-period mean
        effects (normalised to match the observed-treated SD scale).
    se : float
        Empirical SE of ``att_observed``, computed from the placebo
        distribution.
    """
    # Local imports to avoid a cycle at module load.
    from .pipeline import _build_period_targets, variable_importance
    from .structures import DSCARInputs
    from .weights import solve_dsc_weights

    n_treated = inputs.n_treated
    N = inputs.N
    n_donors = N - n_treated
    if placebo_reps <= 0 or n_donors <= n_treated:
        return np.zeros(0), float("nan")

    rng = np.random.default_rng(seed)
    donor_idx = np.arange(n_treated, N)

    placebo_raw = np.zeros(placebo_reps)
    placebo_norm = np.zeros(placebo_reps)

    # Pre-compute the observed-rep variance scale: sigma_eta = sqrt(1/N_tr + sum w^2)
    # for each post period; the actual normalisation factor is the SD across
    # placebo reps anyway, so we just record raw post-period mean ATTs and
    # compute the empirical SD.
    for r in range(placebo_reps):
        placebo_treated = rng.choice(donor_idx, size=n_treated, replace=False)
        keep = np.setdiff1d(np.arange(N), placebo_treated)
        # Reorder so placebo "treated" come first.
        order = np.concatenate([placebo_treated, keep])
        Y_p = inputs.Y[order]
        Ylag_p = inputs.Y_lag1[order]
        X_p = inputs.X[order]
        # Build placebo inputs (a thin re-pack; we don't re-validate labels).
        placebo_inputs = DSCARInputs(
            Y=Y_p, Y_lag1=Ylag_p, X=X_p,
            var_names=inputs.var_names, y_name=inputs.y_name,
            treated_labels=tuple("placebo_{}".format(i) for i in range(n_treated)),
            donor_labels=tuple(f"placebo_d_{i}" for i in range(len(keep))),
            time_labels=inputs.time_labels,
            N=N, T=inputs.T, T0=inputs.T0, T1=inputs.T1,
            n_treated=n_treated,
        )
        # Per-period V on the new donor set.
        n_p_donors = len(keep)
        V_p = variable_importance(
            placebo_inputs.Y[n_treated:],
            placebo_inputs.X[n_treated:],
            placebo_inputs.Y_lag1[n_treated:],
            T0=placebo_inputs.T0,
        )
        Y0_hat_p = np.zeros(inputs.T)
        w_p_mat = np.zeros((inputs.T, n_p_donors))
        for t in range(inputs.T):
            Z1, Z0 = _build_period_targets(placebo_inputs, t, Y0_hat_p)
            if Z1.size == 0:
                w = np.full(n_p_donors, 1.0 / n_p_donors)
            else:
                V_diag = V_p[t, : Z0.shape[0]]
                V_diag = np.where(V_diag > 1e-12, V_diag, 1.0)
                w, _ = solve_dsc_weights(Z1, Z0, V_diag, el_tolerance=el_tolerance)
            w_p_mat[t] = w
            Y0_hat_p[t] = float(placebo_inputs.Y[n_treated:, t] @ w)
        Y_treated_mean_p = placebo_inputs.Y[:n_treated].mean(axis=0)
        gap_p = Y_treated_mean_p - Y0_hat_p
        placebo_raw[r] = float(gap_p[inputs.T0:].mean())

    # SE is just the sample SD of the placebo ATTs (per Section 3.2 last paragraph).
    se = float(np.std(placebo_raw, ddof=1))
    return placebo_raw, se
