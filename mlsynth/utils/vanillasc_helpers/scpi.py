"""SCPI prediction intervals for the synthetic control (full constraint family).

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

Weight-constraint family (``w_constr``)
---------------------------------------
The compatible weight set in the in-sample QCQP and the effective degrees of
freedom follow ``scpi``'s Table 2 (``w_constr_prep`` / ``local_geom`` /
``df_EST``):

    ols       no constraint;                          df = J
    simplex   sum w = 1, w >= 0 (the default);          df = #nonzero - 1
    lasso     ||w||_1 <= Q;                             df = #nonzero
    ridge     ||w||_2 <= Q;                             df = sum_k d_k^2/(d_k^2+lambda)
    L1-L2     sum w = 1, w >= 0, ||w||_2 <= Q2;         df = sum_k d_k^2/(d_k^2+lambda)

where ``d_k`` are the singular values of the pre-period donor design and ``Q`` /
``lambda`` for ridge / L1-L2 come from ``scpi``'s data-driven shrinkage
rule-of-thumb (``shrinkage_EST``) unless supplied. The simplex case is the
``scpi`` default and the standard synthetic control; the ridge case is
``scpi``'s Table-3 inference setting for Amjad, Kim, Shah & Shen (2018) Robust
Synthetic Control (used by CLUSTERSC's RSC / PCR path). The out-of-sample
component is constraint-independent.
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
    lower: np.ndarray          # pointwise PI lower bound for tau_T
    upper: np.ndarray          # pointwise PI upper bound for tau_T
    cf_lower: np.ndarray       # pointwise PI lower bound for the counterfactual Y1(0)
    cf_upper: np.ndarray
    M1_lower: np.ndarray       # in-sample band (w_lb)
    M1_upper: np.ndarray       # in-sample band (w_ub)
    M2_lower: np.ndarray       # out-of-sample band (e_lb), per period
    M2_upper: np.ndarray       # out-of-sample band (e_ub), per period
    lower_simul: np.ndarray    # simultaneous (joint-coverage) PI lower for tau_T
    upper_simul: np.ndarray    # simultaneous PI upper for tau_T
    cf_lower_simul: np.ndarray  # simultaneous PI lower for the counterfactual
    cf_upper_simul: np.ndarray
    metadata: Dict[str, Any]


def _regularization_rho(u: np.ndarray, B: np.ndarray, d0: int, KM: int = 0) -> float:
    """Data-driven ``rho`` (scpi's default ``type-2``), capped at ``rho_max``.

    Mirrors ``regularize_w`` and the ``regularize_check_lb`` bump. scpi defaults
    to the ``type-2`` rule

        rho = (sigma_u * max_j sd(B_j) / min_j var(B_j))
                 * sqrt(log(J + KM) * (d0 + KM) * log(T0)) / sqrt(T0),

    (standard deviations / variances with ``ddof = 1``, as in pandas); a
    too-small ``rho`` is raised (to the larger of the type-1 / type-2 rules,
    then to ``rho_max``) to favour shrinkage and avoid overfitting the
    out-of-sample variance. ``KM`` is the covariate/constant block size.
    """
    T0, J = B.shape
    d = J + KM
    d0e = max(d0 + KM, 1)
    sig_u = sqrt(float(np.mean((u - u.mean()) ** 2)))
    std_b = B.std(axis=0, ddof=1)                     # pandas-style, as scpi
    var_b = B.var(axis=0, ddof=1)
    fac = sqrt(log(max(d, 2)) * d0e * log(max(T0, 2))) / sqrt(T0)

    type1 = (sig_u / max(std_b.min(), _EPS)) * fac
    type2 = (sig_u * std_b.max() / max(var_b.min(), _EPS)) * fac

    rho = min(type2, _RHO_MAX)                        # scpi's default rho type
    if rho < 0.001:  # regularize_check_lb
        rho = min(max(type1, type2), _RHO_MAX)
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


_W_CONSTR_NAMES = ("ols", "simplex", "lasso", "ridge", "L1-L2")


def _floors_at_zero(w_constr) -> bool:
    """True if the constraint lower-bounds the donor weights at zero.

    Simplex and L1-L2 impose ``w >= 0``; ols / lasso / ridge allow signed
    weights (``lb = -inf``), so their fitted weights must not be clipped.
    """
    name = (w_constr if isinstance(w_constr, str)
            else w_constr.get("name") if isinstance(w_constr, dict) else None)
    return name in ("simplex", "L1-L2")


def _shrinkage_ridge(A: np.ndarray, B: np.ndarray):
    """scpi's ridge ``shrinkage_EST`` rule-of-thumb, returning ``(Q, lambda)``.

    Fits an unweighted (identity ``V``) OLS of the treated pre-outcome on the
    donors, then ``lambda = sigma^2 (J + KM) / ||b||**2`` and ``Q = ||b|| /
    (1 + lambda)`` (``KM = 0`` for the outcome-only single-unit case). When
    observations are scarce (``T0 <= J + 10``) scpi lasso-screens the donors to
    ``max(T0 - 10, 2)`` columns and refits; we mirror that with a cvxpy L1 solve.
    """
    A = np.asarray(A, float).ravel()
    B = np.asarray(B, float)
    T0, J = B.shape

    def _rule(Bsub: np.ndarray):
        b, *_ = np.linalg.lstsq(Bsub, A, rcond=None)
        resid = A - Bsub @ b
        k = Bsub.shape[1]
        sig = float(resid @ resid) / max(T0 - k, 1)      # OLS scale (WLS, V=I)
        L2 = float(b @ b)
        lam = sig * k / max(L2, _EPS)
        return sqrt(L2) / (1.0 + lam), lam, L2

    if T0 > J + 10:
        Q, lam, _ = _rule(B)
        return Q, lam

    # lasso pre-selection (scpi's scarce-observation branch)
    if _HAS_CVXPY:
        z = cp.Variable(J)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A - B @ z)),
                          [cp.norm1(z) <= 1.0])
        try:
            prob.solve(solver=cp.CLARABEL)
            coefs = np.abs(np.asarray(z.value).ravel())
        except Exception:  # pragma: no cover - solver fallback
            coefs = np.abs(np.linalg.lstsq(B, A, rcond=None)[0])
    else:  # pragma: no cover - cvxpy is required for SCPI anyway
        coefs = np.abs(np.linalg.lstsq(B, A, rcond=None)[0])
    keep = max(min(T0 - 10, J), 2)
    active = np.argsort(coefs)[::-1][:keep]
    Q, lam, _ = _rule(B[:, np.sort(active)])
    return Q, lam


def _prep_w_constr(w_constr, A: np.ndarray, B: np.ndarray, W: np.ndarray) -> Dict[str, Any]:
    """Normalise a weight-constraint spec to scpi's canonical dict.

    Mirrors ``scpi``'s ``w_constr_prep``: accepts a family name (``"ols"``,
    ``"simplex"``, ``"lasso"``, ``"ridge"``, ``"L1-L2"``) or an explicit
    ``{"name": ..., "Q": ..., "lambda": ..., "Q2": ...}``, and fills in the norm
    ``p``, direction ``dir``, lower bound ``lb`` and -- for ridge / L1-L2 -- the
    data-driven budget ``Q`` and penalty ``lambda`` via ``_shrinkage_ridge``.
    """
    if isinstance(w_constr, str):
        spec: Dict[str, Any] = {"name": w_constr}
    elif isinstance(w_constr, dict):
        spec = dict(w_constr)
    else:
        raise ValueError(
            "w_constr must be a constraint name or a dict; got "
            f"{type(w_constr).__name__}.")
    name = spec.get("name")
    if name not in _W_CONSTR_NAMES:
        raise ValueError(
            "w_constr 'name' must be one of "
            f"{'/'.join(_W_CONSTR_NAMES)}; got {name!r}.")

    if name == "simplex":
        return {"name": "simplex", "p": "L1", "dir": "==", "lb": 0.0,
                "Q": float(spec.get("Q", 1.0)), "Q2": None, "lambda": None}
    if name == "ols":
        return {"name": "ols", "p": "no norm", "dir": None, "lb": -np.inf,
                "Q": None, "Q2": None, "lambda": None}
    if name == "lasso":
        return {"name": "lasso", "p": "L1", "dir": "<=", "lb": -np.inf,
                "Q": float(spec.get("Q", 1.0)), "Q2": None, "lambda": None}
    if name == "ridge":
        Q = spec.get("Q")
        lam = spec.get("lambda")
        if Q is None or lam is None:
            Qe, lame = _shrinkage_ridge(A, B)
            Q = Qe if Q is None else Q
            lam = lame if lam is None else lam
        return {"name": "ridge", "p": "L2", "dir": "<=", "lb": -np.inf,
                "Q": float(Q), "Q2": None, "lambda": float(lam)}
    # L1-L2
    Q2 = spec.get("Q2")
    lam = spec.get("lambda")
    if Q2 is None or lam is None:
        Qe, lame = _shrinkage_ridge(A, B)
        Q2 = Qe if Q2 is None else Q2
        lam = lame if lam is None else lam
    return {"name": "L1-L2", "p": "L1-L2", "dir": "==/<=", "lb": 0.0,
            "Q": float(spec.get("Q", 1.0)), "Q2": float(Q2), "lambda": float(lam)}


@dataclass
class _LocalGeom:
    """Localised compatible-set geometry for the in-sample QCQP (scpi ``local_geom``)."""

    idxw: np.ndarray            # active-donor mask for the E[u]/E[e] design
    lb: np.ndarray              # localised lower bounds (used iff use_lb)
    use_lb: bool
    has_sum: bool               # sum(x) == Q_sum   (simplex, L1-L2)
    Q_sum: float
    has_l1: bool                # ||x||_1 <= Q_l1   (lasso)
    Q_l1: float
    has_l2: bool                # ||x||_2 <= Q_l2   (ridge, L1-L2)
    Q_l2: float


def _local_geom(wc: Dict[str, Any], W: np.ndarray, rho: float, B: np.ndarray) -> _LocalGeom:
    """Localised weight-set for the compatible-region QCQP, per constraint.

    Follows ``scpi``'s ``local_geom``: for norm-bounded constraints the budget is
    relaxed to the realised norm when the fit sits within ``rho`` of the bound
    (so the localised set contains the estimate); ``lb`` pins the near-zero
    donors for the lower-bounded (simplex / L1-L2) constraints; ridge and ols
    leave every donor active with no lower bound.
    """
    W = np.asarray(W, float).ravel()
    J = W.shape[0]
    name = wc["name"]
    all_active = np.ones(J, dtype=bool)

    if name == "simplex":
        idxw = W > rho
        if not idxw.any():
            idxw = np.zeros(J, dtype=bool)
            idxw[int(np.argmax(W))] = True
        lb = np.where(W < rho, W, 0.0)
        return _LocalGeom(idxw, lb, True, True, 1.0, False, 0.0, False, 0.0)

    if name == "ols":
        return _LocalGeom(all_active, np.zeros(J), False, False, 0.0,
                          False, 0.0, False, 0.0)

    if name == "lasso":
        idxw = np.abs(W) > rho
        if not idxw.any():
            idxw = np.zeros(J, dtype=bool)
            idxw[int(np.argmax(np.abs(W)))] = True
        Q = wc["Q"]
        l1 = float(np.sum(np.abs(W)))
        Q_star = l1 if (Q - rho * sqrt(J) <= l1 <= Q) else Q
        return _LocalGeom(idxw, np.zeros(J), False, False, 0.0,
                          True, Q_star, False, 0.0)

    if name == "ridge":
        Q = wc["Q"]
        l2 = float(sqrt(np.sum(W ** 2)))
        Q_star = l2 if (Q - rho <= l2 <= Q) else Q
        return _LocalGeom(all_active, np.zeros(J), False, False, 0.0,
                          False, 0.0, True, Q_star)

    # L1-L2: localised sum (as simplex) plus a localised L2 budget
    idxw = np.abs(W) > rho
    if not idxw.any():
        idxw = np.zeros(J, dtype=bool)
        idxw[int(np.argmax(np.abs(W)))] = True
    lb = np.where(W < rho, W, 0.0)
    Q, Q2 = wc["Q"], wc["Q2"]
    l2 = float(sqrt(np.sum(W ** 2)))
    Q2_star = l2 if (Q - rho <= l2 <= Q) else Q2
    return _LocalGeom(idxw, lb, True, True, 1.0, False, 0.0, True, Q2_star)


def _df_est(wc: Dict[str, Any], W: np.ndarray, B: np.ndarray) -> float:
    """Effective degrees of freedom per constraint (scpi ``df_EST``, ``KM = 0``).

    ols: ``J``; lasso: ``#nonzero``; simplex: ``#nonzero - 1``; ridge / L1-L2:
    ``sum_k d_k^2 / (d_k^2 + lambda)`` over the positive singular values ``d_k``
    of the pre-period donor design.
    """
    W = np.asarray(W, float).ravel()
    name = wc["name"]
    if name == "ols":
        return float(B.shape[1])
    if name == "lasso":
        return float(int(np.sum(np.abs(W) >= _NZ)))
    if name == "simplex":
        return float(max(int(np.sum(np.abs(W) >= _NZ)) - 1, 0))
    # ridge / L1-L2
    d = np.linalg.svd(np.asarray(B, float), compute_uv=False)
    d = d[d > 0]
    lam = wc["lambda"]
    return float(np.sum(d ** 2 / (d ** 2 + lam)))


def _qcqp_weight_constraints(x, lg: _LocalGeom):
    """cvxpy weight-set constraints for the in-sample QCQP from ``_local_geom``."""
    cons = []
    if lg.use_lb:
        cons.append(x >= lg.lb)
    if lg.has_sum:
        cons.append(cp.sum(x) == lg.Q_sum)
    if lg.has_l1:
        cons.append(cp.norm1(x) <= lg.Q_l1)
    if lg.has_l2:
        cons.append(cp.sum_squares(x) <= lg.Q_l2 ** 2)
    return cons


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
    w_constr: Any = "simplex",
    constant: bool = False,
    sims: int = 200,
    u_alpha: float = 0.05,
    e_alpha: float = 0.05,
    u_missp: bool = True,
    e_method: str = "gaussian",
    cointegrated: bool = False,
    seed: int = 0,
) -> SCPIResult:
    """Compute SCPI prediction intervals for a synthetic control.

    Parameters
    ----------
    y : np.ndarray
        Treated outcome over all periods, shape ``(T,)``.
    Y0 : np.ndarray
        Donor outcomes over all periods, shape ``(T, J)`` (columns match ``W``).
    pre : int
        Number of pre-treatment periods ``T0``.
    W : np.ndarray
        Fitted weights (columns match ``Y0``): shape ``(J,)`` when
        ``constant=False``, or ``(J + 1,)`` when ``constant=True`` with the
        trailing entry the intercept coefficient.
    w_constr : str or dict
        Weight-constraint family the fit obeys, controlling the in-sample
        compatible set and the degrees of freedom. One of ``"ols"``,
        ``"simplex"`` (default), ``"lasso"``, ``"ridge"``, ``"L1-L2"``, or an
        explicit dict ``{"name": ..., "Q": ..., "lambda": ..., "Q2": ...}``.
        Ridge is scpi's Table-3 setting for Amjad et al. (2018) Robust SC.
        The donor weights are floored at zero only for lower-bounded
        constraints (simplex / L1-L2); signed families (ols / lasso / ridge)
        keep their weights as given.
    constant : bool
        If True, append an unconstrained constant (intercept) to the donor
        design (scpi's ``constant=True``): the weight constraint binds only the
        donor block while the intercept is free, and the degrees of freedom gain
        ``KM = 1``. ``W`` must then be length ``J + 1`` (donors + intercept).
        The budget ``Q``, penalty ``lambda`` and degrees of freedom reproduce
        scpi exactly with the constant, and the prediction band reproduces
        scpi's ``CI_all_gaussian`` to Monte-Carlo error.
    sims : int
        Number of Gaussian draws for the in-sample simulation.
    u_alpha, e_alpha : float
        In-sample (``alpha1``) and out-of-sample (``alpha2``) levels.
    u_missp : bool
        If True, allow ``E[u | H] != 0`` (estimated by regressing the
        pre-period residuals on the active-donor design); else assume 0.
    e_method : {"gaussian", "ls", "empirical"}
        Tabulation for the out-of-sample shock.
    cointegrated : bool
        If True, fit the in-sample ``E[u]`` and out-of-sample ``E[e]`` models on
        first differences of the donor design (scpi's ``cointegrated_data=True``),
        appropriate when the outcome and donors are cointegrated (I(1) levels).
        The point counterfactual is unchanged; only the prediction bands move.
    seed : int
        RNG seed for the simulation.
    """
    if not _HAS_CVXPY:  # pragma: no cover
        raise ImportError("SCPI inference requires cvxpy.")
    y = np.asarray(y, float).ravel()
    Y0 = np.asarray(Y0, float)
    W = np.asarray(W, float).ravel()
    T0, J = pre, Y0.shape[1]
    KM = 1 if constant else 0
    if W.shape[0] != J + KM:
        raise ValueError(
            f"W must have length J{'+1' if constant else ''} = {J + KM}; "
            f"got {W.shape[0]}.")
    # Donor weights are floored at zero only for lower-bounded constraints
    # (simplex / L1-L2); signed families keep their weights as given.
    if _floors_at_zero(w_constr):
        W = W.copy()
        W[:J] = np.where(W[:J] < 0, 0.0, W[:J])

    A = y[:T0]
    Bdon = Y0[:T0]                                   # donor design (constrained block)
    Pdon = Y0[T0:]                                   # (T_post, J)
    T_post = Pdon.shape[0]
    if T_post < 1:  # pragma: no cover - VanillaSC guarantees a post period
        raise ValueError("SCPI needs at least one post-treatment period.")
    # Augmented design: donors plus the optional unconstrained constant column
    # (scpi's ``constant=True`` / the ``KM`` covariate block). Q / Sigma / the
    # compatible-set QCQP run on the augmented design; the constraint binds only
    # the donor block.
    if constant:
        Bdes = np.column_stack([Bdon, np.ones(T0)])
        Pdes = np.column_stack([Pdon, np.ones(T_post)])
    else:
        Bdes, Pdes = Bdon, Pdon
    u = A - Bdes @ W                                 # pre-period residuals (full model)

    # --- cointegration (scpi's ``cointegrated_data=True``): when the outcome
    #     and donors are cointegrated the levels are I(1), so the uncertainty
    #     models are fit on first differences. Difference the donor design for
    #     the ``E[u]`` / ``E[e]`` regressions and drop the first pre-period (the
    #     NaN differencing introduces, which scpi removes via complete_cases);
    #     the (augmented) level design still drives ``Q``/``Sigma`` with that row
    #     removed. Off (the default) leaves the levels path untouched. ---
    if cointegrated:
        B_mean = Bdon[1:] - Bdon[:-1]  # differenced DONOR design for E[u]/E[e]
        B_q = Bdes[1:]                 # augmented level design for Q/Sigma
        Bdon_q = Bdon[1:]              # donor level design for df/rho/localisation
        u_w = u[1:]                    # residuals aligned to the kept rows
        T0e = T0 - 1
    else:
        B_mean, B_q, Bdon_q, u_w, T0e = Bdon, Bdes, Bdon, u, T0

    # --- weight-constraint spec (scpi w_constr_prep): fills Q / lambda from the
    #     augmented design (shrinkage uses J + KM columns). Align the treated
    #     pre-outcome to the rows kept for Q/Sigma. ---
    A_q = A[1:] if cointegrated else A
    wc = _prep_w_constr(w_constr, A_q, B_q, W)
    Wd = W[:J]                                        # donor (constrained) block

    # --- degrees of freedom per constraint (scpi df_EST) plus KM ---
    d0 = int(np.sum(np.abs(Wd) >= _NZ))
    df = _df_est(wc, Wd, Bdon_q) + KM
    if df >= T0e:  # scpi guard: never spend more dof than observations
        df = T0e - 1
    vc = T0e / (T0e - df) if df < T0e else 1.0

    # --- regularisation parameter and localised compatible set (scpi local_geom),
    #     computed on the donor block ---
    rho = _regularization_rho(u_w, Bdon_q, d0, KM)
    lg = _local_geom(wc, Wd, rho, Bdon_q)
    idxw = lg.idxw

    # --- conditional mean of the residuals (u_missp) ---
    Xd = np.column_stack([B_mean[:, idxw], np.ones(T0e)])
    if u_missp:
        coef, *_ = np.linalg.lstsq(Xd, u_w, rcond=None)
        u_mean = Xd @ coef
    else:
        u_mean = np.zeros(T0e)
    omega = vc * (u_w - u_mean) ** 2                 # HC1 diagonal

    # --- Q = Z'Z / T0,  Sigma = Z' diag(omega) Z / T0**2 ---
    Q = (B_q.T @ B_q) / T0e
    Q = 0.5 * (Q + Q.T)
    Sigma = (B_q.T * omega) @ B_q / (T0e ** 2)
    Sigma = 0.5 * (Sigma + Sigma.T)
    S_root = sqrtm(Sigma).real
    scale, Qreg = _mat_regularize(Q)

    # --- compiled QCQP over the localised, simulated compatible set. The
    #     variable spans donors + covariates; the weight-set constraints bind
    #     only the donor block (``x[:J]``), the constant/covariates stay free. ---
    nfull = J + KM
    x = cp.Variable(nfull)
    c = cp.Parameter(nfull)
    Gstar = cp.Parameter(nfull)
    if Qreg is None:  # pragma: no cover - degenerate Q (no donor variation)
        quad = cp.Constant(0.0)
    else:
        quad = scale * cp.sum_squares(Qreg @ (x - W))
    constraints = [quad - 2.0 * Gstar @ (x - W) <= 0.0]
    constraints += _qcqp_weight_constraints(x[:J] if KM else x, lg)
    prob_min = cp.Problem(cp.Minimize(c @ x), constraints)
    prob_max = cp.Problem(cp.Maximize(c @ x), constraints)

    rng = np.random.default_rng(seed)
    # Predictor rows: each post period plus the post-period mean (for the ATT).
    P_aug = np.vstack([Pdes, Pdes.mean(axis=0, keepdims=True)])   # (T_post + 1, nfull)
    n_rows = T_post + 1
    lo = np.full((sims, n_rows), np.nan)   # min-branch  p'(w - x)
    hi = np.full((sims, n_rows), np.nan)   # max-branch  p'(w - x)

    def _solve(prob):
        for solver in (cp.CLARABEL, cp.ECOS):
            try:
                prob.solve(solver=solver, warm_start=True)
                if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
                    return np.asarray(x.value).ravel()
            except Exception:
                continue
        return None

    for s in range(sims):
        Gstar.value = S_root @ rng.standard_normal(nfull)
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

    # --- out-of-sample band (per post period plus the averaged row). Under
    #     cointegration the design is the differenced donors and the predictand
    #     is the differenced post outcomes, bridged from the pre-period by
    #     ``dP[0] = P[0] - B[T0-1]`` (scpi's e_des_prep). ---
    if cointegrated:
        P_oos = np.empty_like(Pdon)
        P_oos[0] = Pdon[0] - Bdon[T0 - 1]
        P_oos[1:] = Pdon[1:] - Pdon[:-1]
    else:
        P_oos = Pdon
    Xe0 = np.column_stack([B_mean[:, idxw], np.ones(T0e)])
    Xe1 = np.column_stack([P_oos[:, idxw], np.ones(T_post)])
    Xe1_aug = np.vstack([Xe1, Xe1.mean(axis=0, keepdims=True)])
    e_lb_aug, e_ub_aug = _out_of_sample(u_w, Xe0, Xe1_aug, e_alpha, e_method)

    # split per-period from the appended average (ATT) row
    w_lb_avg, w_ub_avg = float(w_lb[-1]), float(w_ub[-1])
    e_lb_avg, e_ub_avg = float(e_lb_aug[-1]), float(e_ub_aug[-1])
    w_lb, w_ub = w_lb[:T_post], w_ub[:T_post]
    e_lb, e_ub = e_lb_aug[:T_post], e_ub_aug[:T_post]

    # --- simultaneous (joint-coverage) band: a uniform in-sample bound plus the
    #     out-of-sample component inflated by sqrt(log(T_post + 1)) (scpi's
    #     ``simultaneousPredGet``). The in-sample joint bound is the alpha/2 (resp.
    #     1 - alpha/2) quantile, across post periods, of the per-period extreme
    #     deviation over draws -- always at least as wide as any pointwise bound. ---
    with np.errstate(invalid="ignore"):
        lo_min = np.nanmin(lo[:, :T_post], axis=0)   # per-period min over draws
        hi_max = np.nanmax(hi[:, :T_post], axis=0)   # per-period max over draws
        w_lb_joint = float(np.nan_to_num(np.nanquantile(lo_min, u_alpha / 2.0)))
        w_ub_joint = float(np.nan_to_num(np.nanquantile(hi_max, 1.0 - u_alpha / 2.0)))
    eps_joint = sqrt(log(T_post + 1.0))              # out-of-sample inflation

    # --- assemble intervals ---
    cf = Pdes @ W                                    # synthetic counterfactual (Y_fit)
    obs = y[T0:]
    tau = obs - cf
    cf_lower = cf + w_lb + e_lb
    cf_upper = cf + w_ub + e_ub
    lower = obs - cf_upper                           # effect PI = obs - cf band
    upper = obs - cf_lower

    cf_lower_simul = cf + w_lb_joint + eps_joint * e_lb
    cf_upper_simul = cf + w_ub_joint + eps_joint * e_ub
    lower_simul = obs - cf_upper_simul
    upper_simul = obs - cf_lower_simul

    att = float(np.mean(tau))
    cf_avg = float(np.mean(cf))
    obs_avg = float(np.mean(obs))
    att_lower = obs_avg - (cf_avg + w_ub_avg + e_ub_avg)
    att_upper = obs_avg - (cf_avg + w_lb_avg + e_lb_avg)

    return SCPIResult(
        tau=tau, lower=lower, upper=upper,
        cf_lower=cf_lower, cf_upper=cf_upper,
        M1_lower=w_lb, M1_upper=w_ub, M2_lower=e_lb, M2_upper=e_ub,
        lower_simul=lower_simul, upper_simul=upper_simul,
        cf_lower_simul=cf_lower_simul, cf_upper_simul=cf_upper_simul,
        metadata={"sims": sims, "u_alpha": u_alpha, "e_alpha": e_alpha,
                  "df": df, "rho": rho, "e_method": e_method,
                  "u_missp": bool(u_missp), "cointegrated": bool(cointegrated),
                  "n_active": int(idxw.sum()),
                  "w_constr": wc["name"], "Q": wc["Q"], "lambda": wc["lambda"],
                  "constant": bool(constant), "KM": KM,
                  "eps_joint": eps_joint,
                  "att": att, "att_lower": att_lower, "att_upper": att_upper},
    )
