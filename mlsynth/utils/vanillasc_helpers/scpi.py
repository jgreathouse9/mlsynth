"""SCPI prediction intervals for the (simplex) synthetic control.

Cattaneo, Feng & Titiunik (2021, JASA) and Cattaneo, Feng, Palomba &
Titiunik (2025, JSS ``scpi``). The prediction error decomposes as

    tau_hat_T - tau_T = e_T - p_T' (beta_hat - beta_0),

an out-of-sample shock ``e_T`` plus an in-sample weight-estimation error.
The (1 - alpha1 - alpha2) prediction interval for tau_T is

    [ tau_hat_T + M1_L - M2_U ,  tau_hat_T + M1_U - M2_L ].

* **In-sample** (M1): a simulation-based bound. With ``Z = B`` (donor
  pre-outcomes), ``Q = Z'Z`` and pre-period residuals ``u = A - B w_hat``,
  draw ``G* ~ N(0, Sigma_hat)`` with ``Sigma_hat = Z' diag(Var[u]) Z``, and
  for each draw solve, over the *localised* simplex constraint set,

      min / max  p_T' delta   s.t.   delta'Q delta - 2 G*'delta <= 0.

  ``M1_L``/``M1_U`` are the ``alpha1/2`` / ``1 - alpha1/2`` quantiles of the
  resulting infima / suprema across draws.
* **Out-of-sample** (M2): the location-scale model -- ``e_T = E[e] +
  sqrt(Var[e]) * eps`` with Gaussian (or empirical) ``eps`` quantiles,
  ``E[e]``/``Var[e]`` estimated from the pre-period residuals.

This implements the canonical simplex case (``w >= 0``, ``sum w = 1``),
which is the ``scpi`` default and the standard synthetic control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:  # pragma: no cover
    _HAS_CVXPY = False

from scipy.stats import norm

_EPS = 1e-10


@dataclass
class SCPIResult:
    """Per-post-period SCPI prediction intervals (arrays of length T_post)."""

    tau: np.ndarray            # observed effect Y1 - synthetic
    lower: np.ndarray          # PI lower bound for tau_T
    upper: np.ndarray          # PI upper bound for tau_T
    cf_lower: np.ndarray       # PI lower bound for the counterfactual Y1(0)
    cf_upper: np.ndarray
    M1_lower: np.ndarray
    M1_upper: np.ndarray
    M2_lower: float
    M2_upper: float
    metadata: Dict[str, Any]


def _localized_lower_bounds(W: np.ndarray, B: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Lower bounds on ``delta`` for the localised simplex set ``Delta*``.

    For the simplex (``-w_j <= 0``) the second derivative is zero (linear
    constraints), so ``kappa = 0``. A donor is "near-binding" (weight ~ 0)
    when ``w_j < rho`` -- those directions may only increase (``delta_j >= 0``);
    interior donors keep the original bound ``delta_j >= -w_j``.
    """
    T0, J = B.shape
    d0 = int(np.sum(W > _EPS))
    d = J
    sig_u = float(np.std(u)) + _EPS
    sig_b = np.std(B, axis=0) + _EPS
    # scpi's data-driven near-zero threshold (reported for provenance).
    rho_data = (np.sqrt(max(d0, 1) * np.log(max(d, 2)) * np.log(max(T0, 2)))
                * sig_b.max() * sig_u) / (np.sqrt(T0) * (sig_b.min() ** 2))
    # For *linear* constraints (the simplex) the local set Delta* equals the
    # exact tangent cone at w_hat with kappa = 0: directions with zero weight
    # may only increase. Use the numerical active set (w_j ~ 0) and cap the
    # data-driven rho so it can never swallow every donor (which would force
    # delta = 0 and a degenerate, zero-width in-sample band).
    rho = float(min(rho_data, 0.5 * float(W.max()) if W.max() > 0 else 1e-6))
    near_zero = W <= max(rho, 1e-7)
    lb = np.where(near_zero, 0.0, -W)
    return lb, float(rho_data)


def scpi_intervals(
    y: np.ndarray,
    Y0: np.ndarray,
    pre: int,
    W: np.ndarray,
    *,
    sims: int = 200,
    u_alpha: float = 0.05,
    e_alpha: float = 0.05,
    u_missp: bool = True,
    e_method: str = "gaussian",
    seed: int = 0,
) -> SCPIResult:
    """Compute SCPI prediction intervals for a simplex synthetic control.

    Parameters
    ----------
    y : np.ndarray
        Treated outcome over all periods, shape ``(T,)``.
    Y0 : np.ndarray
        Donor outcomes over all periods, shape ``(T, J)`` (columns match ``W``).
    pre : int
        Number of pre-treatment periods ``T0``.
    W : np.ndarray
        Fitted simplex donor weights, shape ``(J,)``.
    sims : int
        Number of Gaussian draws for the in-sample simulation.
    u_alpha, e_alpha : float
        In-sample (``alpha1``) and out-of-sample (``alpha2``) levels.
    u_missp : bool
        If True, allow ``E[u | H] != 0`` (estimated by regressing the
        pre-period residuals on the active-donor design); else assume 0.
    e_method : {"gaussian", "empirical"}
        Location-scale tabulation for the out-of-sample shock.
    seed : int
        RNG seed for the simulation.
    """
    if not _HAS_CVXPY:  # pragma: no cover
        raise ImportError("SCPI inference requires cvxpy.")
    y = np.asarray(y, float).ravel()
    Y0 = np.asarray(Y0, float)
    W = np.asarray(W, float).ravel()
    T0, J = pre, Y0.shape[1]
    A = y[:T0]
    B = Y0[:T0]
    u = A - B @ W                                   # pre-period residuals
    T_post = y.shape[0] - T0
    if T_post < 1:
        raise ValueError("SCPI needs at least one post-treatment period.")

    # --- degrees of freedom (simplex) + variance-correction ---
    df = max(int(np.sum(W > _EPS)) - 1, 0)
    vc = T0 / (T0 - df) if df < T0 - 1 else 1.0

    # --- conditional mean/variance of the pseudo-residuals ---
    active = W > _EPS
    if u_missp and active.any():
        Du = np.column_stack([np.ones(T0), B[:, active]])
        coef, *_ = np.linalg.lstsq(Du, u, rcond=None)
        u_resid = u - Du @ coef
    else:
        u_resid = u
    var_u = vc * (u_resid ** 2)                      # diag of Var[u | H]

    # --- Sigma = Z' diag(Var[u]) Z, Q = Z'Z (D = I) ---
    Z = B
    Q = Z.T @ Z
    Q = 0.5 * (Q + Q.T) + 1e-10 * np.eye(J)
    Sigma = Z.T @ (var_u[:, None] * Z)
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-12 * np.eye(J)

    lb_delta, rho = _localized_lower_bounds(W, B, u)

    # --- compiled QCQP: maximise c'delta over the localised, simulated set ---
    delta = cp.Variable(J)
    c = cp.Parameter(J)
    Gstar = cp.Parameter(J)
    constraints = [
        cp.quad_form(delta, cp.psd_wrap(Q)) - 2.0 * Gstar @ delta <= 0.0,
        cp.sum(delta) == 0.0,
        delta >= lb_delta,
    ]
    prob = cp.Problem(cp.Maximize(c @ delta), constraints)

    # Cholesky-free Gaussian draws via eigendecomposition (Sigma may be near-PSD).
    rng = np.random.default_rng(seed)
    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.clip(vals, 0.0, None)
    L = vecs @ np.diag(np.sqrt(vals))
    draws = (L @ rng.standard_normal((J, sims))).T   # (sims, J)

    # Post-period predictors; append the post-period average as an extra column
    # so the average-effect (ATT) interval is computed in the same simulation.
    P = np.vstack([Y0[T0:], Y0[T0:].mean(axis=0, keepdims=True)])  # (T_post+1, J)
    n_cols = T_post + 1
    sup = np.full((sims, n_cols), np.nan)
    inf = np.full((sims, n_cols), np.nan)
    for s in range(sims):
        Gstar.value = draws[s]
        for t in range(n_cols):
            c.value = P[t]
            try:
                prob.solve(solver=cp.CLARABEL, warm_start=True)
                if prob.status in ("optimal", "optimal_inaccurate") and delta.value is not None:
                    sup[s, t] = float(P[t] @ delta.value)
            except Exception:
                pass
            c.value = -P[t]
            try:
                prob.solve(solver=cp.CLARABEL, warm_start=True)
                if prob.status in ("optimal", "optimal_inaccurate") and delta.value is not None:
                    inf[s, t] = float(-(P[t] @ delta.value))
            except Exception:
                pass

    with np.errstate(invalid="ignore"):
        M1_L = np.nanquantile(inf, u_alpha / 2.0, axis=0)
        M1_U = np.nanquantile(sup, 1.0 - u_alpha / 2.0, axis=0)
    M1_L = np.nan_to_num(M1_L, nan=0.0)
    M1_U = np.nan_to_num(M1_U, nan=0.0)

    # --- out-of-sample (location-scale) ---
    e_mean = float(np.mean(u))
    e_sd = float(np.std(u, ddof=1)) if T0 > 1 else float(np.std(u))
    if e_method == "empirical":
        std_resid = (u - e_mean) / (e_sd + _EPS)
        q_lo = np.quantile(std_resid, e_alpha / 2.0)
        q_hi = np.quantile(std_resid, 1.0 - e_alpha / 2.0)
    else:  # gaussian
        q_lo = norm.ppf(e_alpha / 2.0)
        q_hi = norm.ppf(1.0 - e_alpha / 2.0)
    M2_L = e_mean + e_sd * q_lo
    M2_U = e_mean + e_sd * q_hi

    # Split per-period bounds from the appended average-effect column.
    M1_L_avg, M1_U_avg = float(M1_L[-1]), float(M1_U[-1])
    M1_L, M1_U = M1_L[:T_post], M1_U[:T_post]

    # --- assemble per-period intervals ---
    cf = Y0[T0:] @ W                                  # synthetic counterfactual
    tau = y[T0:] - cf
    lower = tau + M1_L - M2_U
    upper = tau + M1_U - M2_L
    cf_lower = cf - M1_U + M2_L                       # counterfactual band (mirror)
    cf_upper = cf - M1_L + M2_U
    # average effect (ATT) over the post-period
    att = float(np.mean(tau))
    att_lower = att + M1_L_avg - M2_U
    att_upper = att + M1_U_avg - M2_L
    return SCPIResult(
        tau=tau, lower=lower, upper=upper,
        cf_lower=cf_lower, cf_upper=cf_upper,
        M1_lower=M1_L, M1_upper=M1_U, M2_lower=M2_L, M2_upper=M2_U,
        metadata={"sims": sims, "u_alpha": u_alpha, "e_alpha": e_alpha,
                  "df": df, "rho": rho, "e_method": e_method,
                  "u_missp": bool(u_missp),
                  "att": att, "att_lower": att_lower, "att_upper": att_upper},
    )
