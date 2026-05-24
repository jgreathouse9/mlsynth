"""Single Proxy Synthetic Control (SPSC).

Implements:

    Park, C., & Tchetgen Tchetgen, E. J. (2025). "Single Proxy Synthetic
    Control." Journal of Causal Inference 13(1), 20230079.
    https://doi.org/10.1515/jci-2023-0079

Unlike the two-proxy proximal estimators (PI/PIS/PIPost), SPSC needs only
**one** type of proxy: the donor outcomes themselves. It views the donor
outcomes ``W`` as error-prone proxies of the treated unit's treatment-free
potential outcome, and uses the treated unit's *own* (optionally detrended)
pre-treatment outcome as the instrument. A ridge-regularized GMM recovers
the synthetic-control weights ``gamma``; the ATT is the mean post-period
gap, with a GMM sandwich (HAC) standard error.

This is a faithful port of the authors' reference R package
(``github.com/qkrcks0218/SPSC``), validated value-for-value on the Panic of
1907 application (Table 3): SPSC-NoDT ATT -0.811 / SE 0.085 (paper -0.813 /
0.084) and SPSC-DT ATT -0.815 / SE 0.067 (paper -0.816 / 0.066).
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

# R's MASS::ginv truncation tolerance; needed to match the reference on the
# rank-deficient GMM bread (dim(instruments) < number of donors).
_GINV_RCOND = np.sqrt(np.finfo(float).eps)
_LAMBDA_GRID = np.arange(-6.0, 2.0 + 1e-9, 0.5)


def _spline_basis(T0: int, df: int) -> np.ndarray:
    """Cubic B-spline trend basis on ``1..T0+1`` (R ``bs(df, intercept=TRUE)``)."""
    from patsy import dmatrix

    grid = np.arange(1, T0 + 2)
    return np.asarray(
        dmatrix(
            f"bs(x, df={df}, degree=3, include_intercept=True) - 1",
            {"x": grid},
            return_type="dataframe",
        )
    )


def _build_detrend_matrix(T0: int, T: int, df: int) -> np.ndarray:
    """Length-``T`` detrend basis; post-period rows are held at the ``T0+1`` value."""
    B = _spline_basis(T0, df)
    rows = np.minimum(np.arange(1, T + 1), T0 + 1) - 1
    return B[rows]


def _instruments(y: np.ndarray, D_pre: Optional[np.ndarray], T0: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Pre-period instrument matrix ``g`` and detrend coefficients ``eta``.

    With detrend, ``g = [D, y - D eta]`` (trend basis plus the detrended
    treated residual). Without detrend, ``g = y``.
    """
    y_pre = y[:T0]
    if D_pre is None:
        return y_pre.reshape(-1, 1), None
    eta = np.linalg.lstsq(D_pre, y_pre, rcond=None)[0]
    residual = y_pre - D_pre @ eta
    return np.column_stack([D_pre, residual]), eta


def _ridge_gamma(g_pre: np.ndarray, y_pre: np.ndarray, W_pre: np.ndarray, lam: float) -> np.ndarray:
    """Ridge-GMM weights solving ``(GW' GW + 10^lam I) gamma = GW' GY``.

    The ``10^lam I`` ridge makes the system full rank, so a deterministic
    ``solve`` matches R's ``ginv`` here while avoiding the run-to-run
    instability of a multithreaded pseudo-inverse on a near-singular matrix.
    """
    N = W_pre.shape[1]
    GY = (g_pre * y_pre[:, None]).mean(0)
    GW = np.stack([(g_pre * W_pre[:, n: n + 1]).mean(0) for n in range(N)], axis=1)
    return np.linalg.solve(GW.T @ GW + (10.0 ** lam) * np.eye(N), GW.T @ GY)


def _cv_lambda(g_pre: np.ndarray, y_pre: np.ndarray, W_pre: np.ndarray, grid: np.ndarray) -> float:
    """Leave-one-out CV over the log10 ridge grid (R ``CV.lambda``)."""
    T0, N = W_pre.shape
    best_lam, best_err = grid[0], np.inf
    for lam in grid:
        resid = np.empty(T0)
        for j in range(T0):
            idx = np.arange(T0) != j
            g, yy, ww = g_pre[idx], y_pre[idx], W_pre[idx]
            GY = (g * yy[:, None]).mean(0)
            GW = np.stack([(g * ww[:, n: n + 1]).mean(0) for n in range(N)], axis=1)
            gamma = np.linalg.solve(GW.T @ GW + (10.0 ** lam) * np.eye(N), GW.T @ GY)
            resid[j] = y_pre[j] - W_pre[j] @ gamma
        err = np.log(np.mean(resid ** 2))
        if err < best_err:
            best_err, best_lam = err, lam
    return float(best_lam)


def _effect(y, W, A, D, lam, lam_grid):
    """One estimation pass: detrend, ridge gamma, constant-ATT beta."""
    pre = A == 0
    T0 = int(pre.sum())
    D_pre = D[pre] if D is not None else None
    g_pre, eta = _instruments(y, D_pre, T0)
    y_pre, W_pre = y[pre], W[pre]
    if lam is None:
        lam = _cv_lambda(g_pre, y_pre, W_pre, lam_grid)
    gamma = _ridge_gamma(g_pre, y_pre, W_pre, lam)
    beta = np.array([float(np.mean(y[A == 1] - W[A == 1] @ gamma))])  # constant ATT
    return dict(eta=eta, gamma=gamma, beta=beta, lam=lam)


def _psi(theta, y, W, A, D, B_post) -> np.ndarray:
    """Stacked moment matrix (rows = periods). Mirrors the reference ``Psi.Ft``."""
    pre = 1 - A
    gamma, beta = theta["gamma"], theta["beta"]
    res_gamma = y - W @ gamma
    blocks = []
    if theta["eta"] is not None:
        eta = theta["eta"]
        blocks.append((pre[:, None] * D) * (y - D @ eta)[:, None])      # YDT
        blocks.append((pre[:, None] * D) * res_gamma[:, None])           # GDT
    # GYb instrument is the ORIGINAL treated outcome (phi = identity), per the reference.
    blocks.append((pre[:, None] * y[:, None]) * res_gamma[:, None])      # GYb
    res_beta = y - W @ gamma - B_post @ beta
    blocks.append((A[:, None] * B_post) * res_beta[:, None])             # Beta
    return np.column_stack(blocks)


def _grad_psi(theta, y, W, A, D, B_post, eps: float = 1e-6) -> np.ndarray:
    """Numerical Jacobian of the mean moments w.r.t. (eta, gamma, beta)."""
    detrend = theta["eta"] is not None
    nd = len(theta["eta"]) if detrend else 0
    ng, nb = len(theta["gamma"]), len(theta["beta"])
    vec = np.concatenate(([theta["eta"]] if detrend else []) + [theta["gamma"], theta["beta"]])

    def unpack(v):
        out = {"eta": v[:nd] if detrend else None,
               "gamma": v[nd: nd + ng], "beta": v[nd + ng:]}
        return out

    ncol = _psi(theta, y, W, A, D, B_post).shape[1]
    G = np.zeros((ncol, len(vec)))
    for j in range(len(vec)):
        vp, vm = vec.copy(), vec.copy()
        vp[j] += eps
        vm[j] -= eps
        G[:, j] = (
            _psi(unpack(vp), y, W, A, D, B_post).mean(0)
            - _psi(unpack(vm), y, W, A, D, B_post).mean(0)
        ) / (2 * eps)
    return G


def _ar1_params(x: np.ndarray) -> Tuple[float, float]:
    """AR(1) coefficient and innovation variance for the auto-bandwidth rule."""
    from statsmodels.tsa.arima.model import ARIMA

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(x, order=(1, 0, 0)).fit(method_kwargs={"warn_convergence": False})
        return float(fit.arparams[0]), float(fit.params[-1])
    except Exception:
        return 0.0, float(np.var(x))


def _hac_meat(Psi: np.ndarray, beta_pos: list) -> np.ndarray:
    """Andrews (1991) auto-bandwidth HAC long-run variance (QS vs Bartlett)."""
    T, K = Psi.shape
    cross = []
    for lag in range(T):
        M = Psi[: T - lag].T @ Psi[lag:]
        cross.append(M + M.T if lag > 0 else M)

    rho = np.empty(len(beta_pos))
    s2 = np.empty(len(beta_pos))
    for i, bp in enumerate(beta_pos):
        rho[i], s2[i] = _ar1_params(Psi[:, bp])
    a1 = np.sum(4 * rho ** 2 * s2 ** 2 / (1 - rho) ** 6 / (1 + rho) ** 2) / np.sum(s2 ** 2 / (1 - rho) ** 4)
    a2 = np.sum(4 * rho ** 2 * s2 ** 2 / (1 - rho) ** 8) / np.sum(s2 ** 2 / (1 - rho) ** 4)
    qs_bw = 1.3221 * (a2 * T) ** 0.2

    lags = np.arange(T)
    z_qs = 6 * np.pi / 5 * lags / qs_bw
    w_qs = np.ones(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_qs[1:] = 3 / z_qs[1:] ** 2 * (np.sin(z_qs[1:]) / z_qs[1:] - np.cos(z_qs[1:]))
    z_b = lags / qs_bw
    w_b = np.where(np.abs(z_b) <= 1, 1 - np.abs(z_b), 0.0)
    w_b[0] = 1.0

    meat_b = sum(cross[l] * w_b[l] for l in range(T)) / T
    meat_qs = sum(cross[l] * w_qs[l] for l in range(T)) / T
    bp = np.array(beta_pos)
    eig_b = np.mean(np.linalg.eigvalsh(meat_b[np.ix_(bp, bp)]))
    eig_qs = np.mean(np.linalg.eigvalsh(meat_qs[np.ix_(bp, bp)]))
    return meat_b if eig_b > eig_qs else meat_qs


def estimate_spsc(
    outcome_vector: np.ndarray,
    donor_outcomes: np.ndarray,
    num_pre_treatment_periods: int,
    detrend: bool = True,
    spline_df: int = 5,
    ridge_lambda: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, float]:
    """Single Proxy Synthetic Control estimate.

    Parameters
    ----------
    outcome_vector : np.ndarray
        Treated outcome over all ``T`` periods, shape ``(T,)``.
    donor_outcomes : np.ndarray
        Donor outcomes ``W``, shape ``(T, N)`` -- the single proxy group.
    num_pre_treatment_periods : int
        Number of pre-treatment periods ``T0``.
    detrend : bool, default True
        If True, residualize the treated outcome against a cubic B-spline
        time trend (SPSC-DT); otherwise SPSC-NoDT.
    spline_df : int, default 5
        Degrees of freedom of the detrend B-spline basis.
    ridge_lambda : float or None, default None
        log10 ridge penalty. ``None`` selects it by leave-one-out CV over
        ``10**[-6, ..., 2]``.

    Returns
    -------
    counterfactual : np.ndarray
        Synthetic control ``W gamma`` over all periods, shape ``(T,)``.
    gamma : np.ndarray
        Donor weights.
    att : float
        Mean post-treatment gap.
    se : float
        GMM/HAC standard error of the ATT (``np.nan`` if ``T1 <= 1``).
    trend : np.ndarray
        Estimated treated-outcome trend (zeros if ``detrend=False``).
    lambda_opt : float
        Selected log10 ridge penalty.
    """

    y = np.asarray(outcome_vector, dtype=float).ravel()
    W = np.asarray(donor_outcomes, dtype=float)
    T, N = W.shape
    T0 = int(num_pre_treatment_periods)
    T1 = T - T0
    A = (np.arange(T) >= T0).astype(float)
    B_post = A.reshape(-1, 1)  # constant-ATT basis

    D = _build_detrend_matrix(T0, T, spline_df) if detrend else None
    theta = _effect(y, W, A, D, ridge_lambda, _LAMBDA_GRID)
    nd = len(theta["eta"]) if detrend else 0

    # Detrend rescaling: balance the trend-moment and proxy-moment magnitudes
    # before the variance step (reference SPSC(), "Scale").
    if detrend and T1 > 1:
        Psi0 = _psi(theta, y, W, A, D, B_post)
        Sig0 = _hac_meat(Psi0, [Psi0.shape[1] - 1])
        diag = np.diag(Sig0)
        ydt_mean = np.mean(diag[0:nd])
        scale = np.sqrt(diag[2 * nd] / ydt_mean) if ydt_mean > 0 else 1.0
        if not np.isfinite(scale):
            scale = 1.0
        D = D * scale
        theta = _effect(y, W, A, D, ridge_lambda, _LAMBDA_GRID)

    gamma = theta["gamma"]
    counterfactual = W @ gamma
    att = float(np.mean(y[T0:] - counterfactual[T0:]))
    trend = D @ theta["eta"] if detrend else np.zeros(T)

    se = np.nan
    if T1 > 1:
        ng = N
        G = _grad_psi(theta, y, W, A, D, B_post)
        Psi = _psi(theta, y, W, A, D, B_post)
        Sigma = _hac_meat(Psi, [Psi.shape[1] - 1])
        npar = nd + ng + 1
        grad_lambda = np.zeros((npar, npar))
        grad_lambda[nd: nd + ng, nd: nd + ng] = np.eye(ng)
        S1 = G.T @ G + (10.0 ** theta["lam"]) * grad_lambda
        S2 = G.T @ Sigma @ G
        S1_inv = np.linalg.pinv(S1, rcond=_GINV_RCOND)  # match R MASS::ginv tolerance
        avar = S1_inv @ S2 @ S1_inv.T / T
        var_beta = avar[nd + ng, nd + ng]
        se = float(np.sqrt(var_beta)) if var_beta >= 0 else np.nan

    return counterfactual, gamma, att, se, trend, float(theta["lam"])
