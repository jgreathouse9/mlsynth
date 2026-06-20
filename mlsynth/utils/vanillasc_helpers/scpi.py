"""SCPI prediction intervals for the (simplex) synthetic control.

Cattaneo, Feng & Titiunik (2021, JASA) and Cattaneo, Feng, Palomba &
Titiunik (2025, JSS ``scpi``). The prediction error of the synthetic-control
counterfactual decomposes as

    tau_hat_T - tau_T = e_T - p_T' (beta_hat - beta_0),

an out-of-sample shock ``e_T`` plus an in-sample weight-estimation error.
This module is a *from-scratch* (MIT-licensed) re-derivation of the algorithm
described in those papers -- it does **not** import the GPL ``scpi`` package --
and has been validated to reproduce ``scpi``'s ``CI_all_gaussian`` for the
canonical simplex control to within Monte-Carlo error.

The counterfactual prediction band is assembled period-by-period as

    [ Y_fit + w_lb + e_lb ,  Y_fit + w_ub + e_ub ],

with the treatment-effect interval ``[Y_obs - cf_upper, Y_obs - cf_lower]``.

In-sample component (``w_lb``/``w_ub``)
---------------------------------------
With ``Z = B`` (donor pre-outcomes), ``Q = Z'Z / T0`` and pre-period
residuals ``u = A - B w_hat``, draw ``G* ~ N(0, Sigma)`` with
``Sigma = Z' diag(omega) Z / T0**2`` and ``omega_t = (T0/(T0-df)) (u_t -
E[u_t])**2`` (HC1; ``E[u]`` from a regression of ``u`` on the active-donor
design when ``u_missp``). For each draw and post-period predictor ``p_T``
solve, over the *localised* simplex set,

    min / max  p_T' x   s.t.   (x - w_hat)'Q(x - w_hat) - 2 G*'(x - w_hat) <= 0,
                               sum(x) = 1,  x >= lb,

where ``lb_j = w_hat_j`` if ``w_hat_j < rho`` else ``0`` (the local geometry of
Cattaneo et al.; ``rho`` is the data-driven regularisation parameter, capped at
``rho_max = 0.2``). ``Q`` is reduced via a thresholded eigen-square-root so that
collinear (near-null) donor directions are left unconstrained, exactly as in the
reference conic reformulation. ``w_lb``/``w_ub`` are the ``alpha1/2`` /
``1 - alpha1/2`` quantiles, across draws, of ``p_T'(w_hat - x)`` for the
maximising / minimising branch.

Out-of-sample component (``e_lb``/``e_ub``)
-------------------------------------------
A location-scale model for ``e_T``: regress ``u`` on the active-donor design
to get the conditional mean ``E[e]`` and a log-variance model for the scale
``sqrt(Var[e])`` (Gaussian), capped by the inter-quartile range of the
residuals (``IQR / 1.34``). The Gaussian band is ``E[e] +/- sqrt(-2 ln alpha2)
* scale``; ``"ls"`` uses standardized-residual quantiles, ``"empirical"`` the
raw residual quantiles.

This implements the canonical simplex case (``w >= 0``, ``sum w = 1``), the
``scpi`` default and the standard synthetic control.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Any, Dict

import numpy as np
from scipy.linalg import eigh, sqrtm

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:  # pragma: no cover
    _HAS_CVXPY = False

_EPS = 1e-10
_RHO_MAX = 0.2          # scpi's default cap on the regularisation parameter
_NZ = 1e-6              # "non-zero weight" threshold (scpi convention)


@dataclass
class SCPIResult:
    """Per-post-period SCPI prediction intervals (arrays of length T_post)."""

    tau: np.ndarray            # observed effect Y1 - synthetic
    lower: np.ndarray          # PI lower bound for tau_T
    upper: np.ndarray          # PI upper bound for tau_T
    cf_lower: np.ndarray       # PI lower bound for the counterfactual Y1(0)
    cf_upper: np.ndarray
    M1_lower: np.ndarray       # in-sample band (w_lb)
    M1_upper: np.ndarray       # in-sample band (w_ub)
    M2_lower: np.ndarray       # out-of-sample band (e_lb), per period
    M2_upper: np.ndarray       # out-of-sample band (e_ub), per period
    metadata: Dict[str, Any]


def _regularization_rho(u: np.ndarray, B: np.ndarray, d0: int) -> float:
    """Data-driven ``rho`` (scpi ``type-1``), capped at ``rho_max = 0.2``.

    Mirrors ``regularize_w`` / ``regularize_check_lb``: a too-small ``rho`` is
    bumped up (via the ``type-2`` rule, then to ``rho_max``) to favour shrinkage
    and avoid overfitting the out-of-sample variance.
    """
    T0, J = B.shape
    d = J
    sig_u = sqrt(float(np.mean((u - u.mean()) ** 2)))
    std_b = B.std(axis=0, ddof=0)
    fac = sqrt(log(max(d, 2)) * max(d0, 1) * log(max(T0, 2))) / sqrt(T0)

    rho = (sig_u / max(std_b.min(), _EPS)) * fac
    rho = min(rho, _RHO_MAX)

    if rho < 0.001:  # regularize_check_lb
        rho1 = min((sig_u / max(std_b.min(), _EPS)) * fac, _RHO_MAX)
        rho2 = min((sig_u * std_b.max() / max(B.var(axis=0, ddof=0).min(), _EPS)) * fac, _RHO_MAX)
        rho = max(rho1, rho2)
        if rho < 0.05:
            rho = _RHO_MAX
    return float(rho)


def _mat_regularize(Q: np.ndarray):
    """Thresholded eigen-square-root of a PSD ``Q`` (scpi ``matRegularize``).

    Returns ``(scale, Qreg)`` with ``Qreg' Qreg = Q / scale`` over the retained
    directions, so that ``x' Q x = scale * ||Qreg x||**2``. Directions with
    eigenvalue at or below ``1e6 * eps * scale`` are dropped (left
    unconstrained), which is what lets collinear donors move freely in the
    in-sample simulation.
    """
    w, V = eigh(Q)
    cond = 1e6 * np.finfo(float).eps
    scale = float(np.max(np.abs(w)))
    if scale < cond:
        return 0.0, None
    w_scaled = w / scale
    mask = w_scaled > cond
    Qreg = (V[:, mask] * np.sqrt(w_scaled[mask])).T   # (k, J)
    return scale, Qreg


def _out_of_sample(
    u: np.ndarray,
    Xe0: np.ndarray,
    Xe1: np.ndarray,
    e_alpha: float,
    e_method: str,
):
    """Out-of-sample location-scale band ``(e_lb, e_ub)`` per predictor row.

    ``Xe0`` is the in-sample (T0 x k) active-donor design, ``Xe1`` the
    out-of-sample (T_rows x k) design. Returns Gaussian / ls / empirical bands
    following ``scpi_out``.
    """
    coef, *_ = np.linalg.lstsq(Xe0, u, rcond=None)
    e_mean = Xe1 @ coef
    y_fit = Xe0 @ coef
    resid = u - y_fit

    if e_method == "empirical":
        q_lo = np.quantile(resid, e_alpha / 2.0)
        q_hi = np.quantile(resid, 1.0 - e_alpha / 2.0)
        return e_mean + q_lo, e_mean + q_hi

    # location-scale: log-variance model, capped by the IQR of the residuals
    log_var = np.log(np.maximum(resid ** 2, _EPS))
    vcoef, *_ = np.linalg.lstsq(Xe0, log_var, rcond=None)
    sig_lv = np.sqrt(np.exp(Xe1 @ vcoef))

    try:  # IQR via quantile regression (scpi); fall back to a marginal IQR
        import statsmodels.api as sm
        qr = sm.QuantReg(resid, Xe0)
        q1 = np.asarray(qr.fit(q=0.25).predict(Xe1)).ravel()
        q3 = np.asarray(qr.fit(q=0.75).predict(Xe1)).ravel()
        iqr = np.abs(q3 - q1)
    except Exception:  # pragma: no cover - statsmodels missing / rank issue
        iqr = np.full(Xe1.shape[0], np.subtract(*np.percentile(resid, [75, 25])))
    e_sig = np.minimum(sig_lv, np.abs(iqr) / 1.34)

    if e_method == "ls":
        std_resid = resid / np.maximum(sig_lv[:1].mean(), _EPS)
        q_lo = np.quantile(std_resid, e_alpha / 2.0)
        q_hi = np.quantile(std_resid, 1.0 - e_alpha / 2.0)
        return e_mean + e_sig * q_lo, e_mean + e_sig * q_hi

    eps = sqrt(-log(e_alpha) * 2.0) * e_sig   # gaussian
    return e_mean - eps, e_mean + eps


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
    e_method : {"gaussian", "ls", "empirical"}
        Tabulation for the out-of-sample shock.
    seed : int
        RNG seed for the simulation.
    """
    if not _HAS_CVXPY:  # pragma: no cover
        raise ImportError("SCPI inference requires cvxpy.")
    y = np.asarray(y, float).ravel()
    Y0 = np.asarray(Y0, float)
    W = np.asarray(W, float).ravel()
    W = np.where(W < 0, 0.0, W)
    T0, J = pre, Y0.shape[1]
    A = y[:T0]
    B = Y0[:T0]
    P = Y0[T0:]                                       # (T_post, J)
    T_post = P.shape[0]
    if T_post < 1:  # pragma: no cover - VanillaSC guarantees a post period
        raise ValueError("SCPI needs at least one post-treatment period.")
    u = A - B @ W                                    # pre-period residuals

    # --- degrees of freedom (simplex): #nonzero - 1 (+ KM = 0) ---
    d0 = int(np.sum(np.abs(W) >= _NZ))
    df = max(d0 - 1, 0)
    vc = T0 / (T0 - df) if df < T0 else 1.0

    # --- regularisation parameter and localised lower bounds ---
    rho = _regularization_rho(u, B, d0)
    idxw = W > rho
    if not idxw.any():  # pragma: no cover - degenerate: every weight below rho
        idxw = np.zeros(J, dtype=bool)
        idxw[int(np.argmax(W))] = True
    lb = np.where(W < rho, W, 0.0)

    # --- conditional mean of the residuals (u_missp) ---
    Xd = np.column_stack([B[:, idxw], np.ones(T0)])
    if u_missp:
        coef, *_ = np.linalg.lstsq(Xd, u, rcond=None)
        u_mean = Xd @ coef
    else:
        u_mean = np.zeros(T0)
    omega = vc * (u - u_mean) ** 2                   # HC1 diagonal

    # --- Q = Z'Z / T0,  Sigma = Z' diag(omega) Z / T0**2 ---
    Q = (B.T @ B) / T0
    Q = 0.5 * (Q + Q.T)
    Sigma = (B.T * omega) @ B / (T0 ** 2)
    Sigma = 0.5 * (Sigma + Sigma.T)
    S_root = sqrtm(Sigma).real
    scale, Qreg = _mat_regularize(Q)

    # --- compiled QCQP over the localised, simulated simplex set ---
    x = cp.Variable(J)
    c = cp.Parameter(J)
    Gstar = cp.Parameter(J)
    if Qreg is None:  # pragma: no cover - degenerate Q (no donor variation)
        quad = cp.Constant(0.0)
    else:
        quad = scale * cp.sum_squares(Qreg @ (x - W))
    constraints = [quad - 2.0 * Gstar @ (x - W) <= 0.0, cp.sum(x) == 1.0, x >= lb]
    prob_min = cp.Problem(cp.Minimize(c @ x), constraints)
    prob_max = cp.Problem(cp.Maximize(c @ x), constraints)

    rng = np.random.default_rng(seed)
    # Predictor rows: each post period plus the post-period mean (for the ATT).
    P_aug = np.vstack([P, P.mean(axis=0, keepdims=True)])     # (T_post + 1, J)
    n_rows = T_post + 1
    lo = np.full((sims, n_rows), np.nan)   # min-branch  p'(w - x)
    hi = np.full((sims, n_rows), np.nan)   # max-branch  p'(w - x)

    def _solve(prob):
        for solver in (cp.CLARABEL,):
            try:
                prob.solve(solver=solver, warm_start=True)
                if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
                    return np.asarray(x.value).ravel()
            except Exception:
                continue
        return None

    for s in range(sims):
        Gstar.value = S_root @ rng.standard_normal(J)
        for t in range(n_rows):
            c.value = P_aug[t]
            xs = _solve(prob_max)                    # maximise p'x -> min p'(w-x)
            if xs is not None:
                lo[s, t] = float(P_aug[t] @ (W - xs))
            xs = _solve(prob_min)                    # minimise p'x -> max p'(w-x)
            if xs is not None:
                hi[s, t] = float(P_aug[t] @ (W - xs))

    with np.errstate(invalid="ignore"):
        w_lb = np.nanquantile(lo, u_alpha / 2.0, axis=0)
        w_ub = np.nanquantile(hi, 1.0 - u_alpha / 2.0, axis=0)
    w_lb = np.nan_to_num(w_lb, nan=0.0)
    w_ub = np.nan_to_num(w_ub, nan=0.0)

    # --- out-of-sample band (per post period plus the averaged row) ---
    Xe0 = np.column_stack([B[:, idxw], np.ones(T0)])
    Xe1 = np.column_stack([P[:, idxw], np.ones(T_post)])
    Xe1_aug = np.vstack([Xe1, Xe1.mean(axis=0, keepdims=True)])
    e_lb_aug, e_ub_aug = _out_of_sample(u, Xe0, Xe1_aug, e_alpha, e_method)

    # split per-period from the appended average (ATT) row
    w_lb_avg, w_ub_avg = float(w_lb[-1]), float(w_ub[-1])
    e_lb_avg, e_ub_avg = float(e_lb_aug[-1]), float(e_ub_aug[-1])
    w_lb, w_ub = w_lb[:T_post], w_ub[:T_post]
    e_lb, e_ub = e_lb_aug[:T_post], e_ub_aug[:T_post]

    # --- assemble intervals ---
    cf = P @ W                                       # synthetic counterfactual (Y_fit)
    obs = y[T0:]
    tau = obs - cf
    cf_lower = cf + w_lb + e_lb
    cf_upper = cf + w_ub + e_ub
    lower = obs - cf_upper                           # effect PI = obs - cf band
    upper = obs - cf_lower

    att = float(np.mean(tau))
    cf_avg = float(np.mean(cf))
    obs_avg = float(np.mean(obs))
    att_lower = obs_avg - (cf_avg + w_ub_avg + e_ub_avg)
    att_upper = obs_avg - (cf_avg + w_lb_avg + e_lb_avg)

    return SCPIResult(
        tau=tau, lower=lower, upper=upper,
        cf_lower=cf_lower, cf_upper=cf_upper,
        M1_lower=w_lb, M1_upper=w_ub, M2_lower=e_lb, M2_upper=e_ub,
        metadata={"sims": sims, "u_alpha": u_alpha, "e_alpha": e_alpha,
                  "df": df, "rho": rho, "e_method": e_method,
                  "u_missp": bool(u_missp), "n_active": int(idxw.sum()),
                  "att": att, "att_lower": att_lower, "att_upper": att_upper},
    )
