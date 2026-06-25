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


def _build_detrend_matrix(T0: int, T: int, df: int,
                          basis: str = "bspline", degree: int = 1) -> np.ndarray:
    """Length-``T`` detrend basis.

    ``basis="bspline"`` is the cubic B-spline trend (``df`` degrees of freedom),
    with post-period rows held at the ``T0+1`` value -- the default. ``basis=
    "poly"`` is the polynomial trend ``[1, t, ..., t^degree]`` over the full
    ``1..T`` grid, reproducing the reference's ``detrend.ft`` (``degree=1`` is the
    linear ``(1, t)`` trend used in the authors' California example). Only the
    pre-period rows enter the GMM moments, so the post-row convention does not
    affect the weights either way.
    """
    if basis == "poly":
        t = np.arange(1, T + 1, dtype=float)
        return np.column_stack([t ** k for k in range(degree + 1)])
    B = _spline_basis(T0, df)
    rows = np.minimum(np.arange(1, T + 1), T0 + 1) - 1
    return B[rows]


def _att_basis(T: int, T0: int, att_degree: int) -> np.ndarray:
    """Post-treatment ATT basis (reference ``att.ft``), shape ``(T, att_degree+1)``.

    Pre-period rows are zero; post rows are ``[s^0, ..., s^att_degree]`` in the
    within-post index ``s = 1..T1``. ``att_degree=0`` gives the single constant
    column (the treatment indicator on post rows), i.e. a constant ATT; ``att_
    degree=1`` is the linear-in-time effect path of the authors' California
    example.
    """
    T1 = T - T0
    s = np.arange(1, T1 + 1, dtype=float)
    post = np.column_stack([s ** k for k in range(att_degree + 1)])
    B = np.zeros((T, att_degree + 1))
    B[T0:] = post
    return B


def _poly_basis(u: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial sieve ``[u, u^2, ..., u^degree]`` of a 1-D instrument.

    ``degree=1`` reproduces the linear single-proxy instrument (the reference's
    default ``Y.basis``); ``degree>=2`` is the **nonparametric** (series/sieve)
    SPSC instrument of Park & Tchetgen Tchetgen's supplement S1.6, which spans
    a richer space of the treated outcome and so over-identifies the bridge.
    """
    u = np.asarray(u, dtype=float).ravel()
    return np.column_stack([u ** k for k in range(1, degree + 1)])


def _instruments(y: np.ndarray, D_pre: Optional[np.ndarray], T0: int,
                 degree: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Pre-period instrument matrix ``g`` and detrend coefficients ``eta``.

    With detrend, ``g = [D, phi(y - D eta)]`` (trend basis plus a sieve basis of
    the detrended treated residual). Without detrend, ``g = phi(y)``. The sieve
    ``phi`` is the degree-``degree`` polynomial basis (``degree=1`` is linear).
    """
    y_pre = y[:T0]
    if D_pre is None:
        return _poly_basis(y_pre, degree), None
    eta = np.linalg.lstsq(D_pre, y_pre, rcond=None)[0]
    residual = y_pre - D_pre @ eta
    return np.column_stack([D_pre, _poly_basis(residual, degree)]), eta


def _ridge_ginv(GW: np.ndarray, GY: np.ndarray, lam: float) -> np.ndarray:
    """Ridge-GMM donor weights ``(GW' GW + 10^lam I)^+ (GW' GY)``, solved stably.

    The GMM is under-identified -- ``GW`` is ``(m, N)`` with only ``m`` moments
    for ``N`` donors -- so ``GW' GW`` is rank-``m`` and ``10^lam I`` merely
    floors its ``N - m`` null eigenvalues at ``10^lam``. A plain ``solve``
    inverts that floor (``1 / 10^lam``) and blows the null-space noise up into
    ``gamma`` when the penalty is small (the CV choice can be ``10^-5`` or
    less), which is what skewed and inflated the conformal band; R's
    ``MASS::ginv`` truncates those directions instead.

    The real instability is a *near-collinear* instrument: ``GW`` (``m`` rows)
    can have a singular value so small that ``solve`` inverts it and lets its
    direction dominate ``gamma``, while ``ginv`` truncates it. Solve through the
    thin SVD of ``GW = U diag(s) V^T``. The non-trivial eigenvalues of
    ``GW'GW + 10^lam I`` are ``s^2 + 10^lam`` (the ``N - m`` null directions sit
    at ``10^lam`` but carry no ``GW'GY`` mass, so they never enter ``gamma``).
    Drop exactly the directions ``MASS::ginv`` would -- those whose eigenvalue
    ``s^2 + 10^lam`` is below ``sqrt(eps)`` times the largest -- then
    ``gamma = V diag(s / (s^2 + 10^lam)) U^T GY`` over the kept directions. The
    cutoff is on the *penalised* eigenvalue, so the ridge floor ``10^lam`` makes
    it ``lambda``-aware exactly as the reference: a near-collinear direction is
    truncated at a small penalty but retained once ``10^lam`` lifts it above the
    tolerance, which is what keeps the CV ``lambda`` choice (and hence the
    conformal band) aligned with ``MASS::ginv``. This avoids forming the
    rank-deficient ``N x N`` Gram and matches the reference to machine precision.
    """
    U, s, Vt = np.linalg.svd(GW, full_matrices=False)   # GW (m, N): U (m, m), s (m,), Vt (m, N)
    eig = s ** 2 + 10.0 ** lam                           # eigenvalues of GW'GW + 10^lam I (instrument block)
    keep = eig > _GINV_RCOND * eig.max()                 # MASS::ginv sqrt(eps) tol on that matrix
    scale = np.where(keep, s / eig, 0.0)
    return Vt.T @ (scale * (U.T @ GY))


def _ridge_gamma(g_pre: np.ndarray, y_pre: np.ndarray, W_pre: np.ndarray, lam: float) -> np.ndarray:
    """Ridge-GMM donor weights (reference ``SPSC.Effect``); see :func:`_ridge_ginv`."""
    N = W_pre.shape[1]
    GY = (g_pre * y_pre[:, None]).mean(0)
    GW = np.stack([(g_pre * W_pre[:, n: n + 1]).mean(0) for n in range(N)], axis=1)
    return _ridge_ginv(GW, GY, lam)


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
            gamma = _ridge_ginv(GW, GY, lam)
            resid[j] = y_pre[j] - W_pre[j] @ gamma
        err = np.log(np.mean(resid ** 2))
        if err < best_err:
            best_err, best_lam = err, lam
    return float(best_lam)


def _effect(y, W, A, D, lam, lam_grid, B_post, degree=1):
    """One estimation pass: detrend, ridge gamma, ATT-basis beta.

    ``beta`` is the least-squares fit of the post-treatment gap on the ATT basis
    ``B_post`` (post rows); a single constant column reproduces the mean-gap
    (constant-ATT) estimate, a ``[1, s]`` basis gives the linear effect path.
    """
    pre = A == 0
    T0 = int(pre.sum())
    D_pre = D[pre] if D is not None else None
    g_pre, eta = _instruments(y, D_pre, T0, degree)
    y_pre, W_pre = y[pre], W[pre]
    if lam is None:
        lam = _cv_lambda(g_pre, y_pre, W_pre, lam_grid)
    gamma = _ridge_gamma(g_pre, y_pre, W_pre, lam)
    post = A == 1
    beta = np.linalg.lstsq(B_post[post], (y - W @ gamma)[post], rcond=None)[0]
    return dict(eta=eta, gamma=gamma, beta=beta, lam=lam)


def _psi(theta, y, W, A, D, B_post, degree=1) -> np.ndarray:
    """Stacked moment matrix (rows = periods). Mirrors the reference ``Psi.Ft``."""
    pre = 1 - A
    gamma, beta = theta["gamma"], theta["beta"]
    res_gamma = y - W @ gamma
    blocks = []
    if theta["eta"] is not None:
        eta = theta["eta"]
        blocks.append((pre[:, None] * D) * (y - D @ eta)[:, None])      # YDT
        blocks.append((pre[:, None] * D) * res_gamma[:, None])           # GDT
    # GYb instrument is the sieve basis of the ORIGINAL treated outcome, per the
    # reference ``Psi.Ft`` (degree=1 is the identity / linear single proxy).
    blocks.append((pre[:, None] * _poly_basis(y, degree)) * res_gamma[:, None])  # GYb
    res_beta = y - W @ gamma - B_post @ beta
    blocks.append((A[:, None] * B_post) * res_beta[:, None])             # Beta
    return np.column_stack(blocks)


def _grad_psi(theta, y, W, A, D, B_post, degree=1, eps: float = 1e-6) -> np.ndarray:
    """Numerical Jacobian of the mean moments w.r.t. (eta, gamma, beta)."""
    detrend = theta["eta"] is not None
    nd = len(theta["eta"]) if detrend else 0
    ng, nb = len(theta["gamma"]), len(theta["beta"])
    vec = np.concatenate(([theta["eta"]] if detrend else []) + [theta["gamma"], theta["beta"]])

    def unpack(v):
        out = {"eta": v[:nd] if detrend else None,
               "gamma": v[nd: nd + ng], "beta": v[nd + ng:]}
        return out

    ncol = _psi(theta, y, W, A, D, B_post, degree).shape[1]
    G = np.zeros((ncol, len(vec)))
    for j in range(len(vec)):
        vp, vm = vec.copy(), vec.copy()
        vp[j] += eps
        vm[j] -= eps
        G[:, j] = (
            _psi(unpack(vp), y, W, A, D, B_post, degree).mean(0)
            - _psi(unpack(vm), y, W, A, D, B_post, degree).mean(0)
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
    basis_degree: int = 1,
    att_degree: int = 0,
    detrend_basis: str = "bspline",
    detrend_degree: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, float, np.ndarray, np.ndarray]:
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
    basis_degree : int, default 1
        Degree of the polynomial sieve applied to the treated-outcome
        instrument (the reference's ``Y.basis``). ``1`` is the linear single
        proxy; ``>=2`` is the **nonparametric** (series) SPSC, which spans a
        richer space of the outcome and over-identifies the bridge -- useful
        when the synthetic-control bridge is nonlinear in the donor outcomes.

    Returns
    -------
    counterfactual : np.ndarray
        Synthetic control ``W gamma`` over all periods, shape ``(T,)``.
    gamma : np.ndarray
        Donor weights.
    att : float
        Average post-treatment effect (mean of the fitted ATT path; equals the
        mean post-treatment gap when ``att_degree=0``).
    se : float
        GMM/HAC standard error of ``att`` (``np.nan`` if ``T1 <= 1``).
    trend : np.ndarray
        Estimated treated-outcome trend (zeros if ``detrend=False``).
    lambda_opt : float
        Selected log10 ridge penalty.
    effect_path : np.ndarray
        Fitted per-post-period effect ``B_post @ beta``, shape ``(T1,)``. A flat
        path at ``att`` when ``att_degree=0``.
    path_se : np.ndarray
        Per-post-period standard error of ``effect_path`` (delta method), shape
        ``(T1,)``; all ``np.nan`` if ``T1 <= 1``.

    Other Parameters
    ----------------
    att_degree : int, default 0
        Polynomial degree of the ATT basis in post-treatment time (reference
        ``att.ft``). ``0`` is a constant ATT; ``1`` is the linear effect path of
        the authors' California example.
    detrend_basis : {"bspline", "poly"}, default "bspline"
        Detrend trend family. ``"poly"`` uses ``[1, t, ..., t^detrend_degree]``
        (the reference ``detrend.ft``; ``detrend_degree=1`` is the linear trend).
    detrend_degree : int, default 1
        Polynomial degree when ``detrend_basis="poly"``.
    """

    if int(basis_degree) < 1:
        raise ValueError("basis_degree must be a positive integer.")
    if int(att_degree) < 0:
        raise ValueError("att_degree must be a non-negative integer.")
    if int(detrend_degree) < 1:
        raise ValueError("detrend_degree must be a positive integer.")
    degree = int(basis_degree)

    y = np.asarray(outcome_vector, dtype=float).ravel()
    W = np.asarray(donor_outcomes, dtype=float)
    T, N = W.shape
    T0 = int(num_pre_treatment_periods)
    T1 = T - T0
    A = (np.arange(T) >= T0).astype(float)
    B_post = _att_basis(T, T0, int(att_degree))      # (T, att_degree+1)
    nb = B_post.shape[1]

    D = (_build_detrend_matrix(T0, T, spline_df, detrend_basis, int(detrend_degree))
         if detrend else None)
    theta = _effect(y, W, A, D, ridge_lambda, _LAMBDA_GRID, B_post, degree)
    nd = len(theta["eta"]) if detrend else 0
    _ncol = _psi(theta, y, W, A, D, B_post, degree).shape[1]
    beta_pos = list(range(_ncol - nb, _ncol))         # the ATT-basis moment block

    # Detrend rescaling: balance the trend-moment and proxy-moment magnitudes
    # before the variance step (reference SPSC(), "Scale"): the ratio of the
    # mean GYb-block diagonal to the mean YDT-block diagonal of the meat.
    if detrend and T1 > 1:
        Psi0 = _psi(theta, y, W, A, D, B_post, degree)
        Sig0 = _hac_meat(Psi0, beta_pos)
        diag = np.diag(Sig0)
        ydt_mean = np.mean(diag[0:nd])
        gyb_mean = np.mean(diag[2 * nd: 2 * nd + degree])
        scale = np.sqrt(gyb_mean / ydt_mean) if ydt_mean > 0 else 1.0
        if not np.isfinite(scale):
            scale = 1.0
        D = D * scale
        theta = _effect(y, W, A, D, ridge_lambda, _LAMBDA_GRID, B_post, degree)

    gamma = theta["gamma"]
    counterfactual = W @ gamma
    B_post_post = B_post[T0:]                          # (T1, nb)
    effect_path = B_post_post @ theta["beta"]          # (T1,)
    att = float(np.mean(effect_path))
    trend = D @ theta["eta"] if detrend else np.zeros(T)

    se = np.nan
    path_se = np.full(T1, np.nan)
    if T1 > 1:
        ng = N
        G = _grad_psi(theta, y, W, A, D, B_post, degree)
        Psi = _psi(theta, y, W, A, D, B_post, degree)
        Sigma = _hac_meat(Psi, beta_pos)
        npar = nd + ng + nb
        grad_lambda = np.zeros((npar, npar))
        grad_lambda[nd: nd + ng, nd: nd + ng] = np.eye(ng)
        S1 = G.T @ G + (10.0 ** theta["lam"]) * grad_lambda
        S2 = G.T @ Sigma @ G
        S1_inv = np.linalg.pinv(S1, rcond=_GINV_RCOND)  # match R MASS::ginv tolerance
        avar = S1_inv @ S2 @ S1_inv.T / T
        cov_beta = avar[nd + ng: nd + ng + nb, nd + ng: nd + ng + nb]  # (nb, nb)
        # Average-ATT variance (delta method on the mean ATT-basis row) and the
        # per-period path variance b_t' cov b_t.
        bbar = B_post_post.mean(0)
        var_att = float(bbar @ cov_beta @ bbar)
        se = float(np.sqrt(var_att)) if var_att >= 0 else np.nan
        pv = np.einsum("ti,ij,tj->t", B_post_post, cov_beta, B_post_post)
        path_se = np.sqrt(np.where(pv >= 0, pv, np.nan))

    return (counterfactual, gamma, att, se, trend, float(theta["lam"]),
            effect_path, path_se)
