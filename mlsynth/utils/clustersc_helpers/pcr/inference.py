"""Shen-Ding-Sekhon-Yu (2023) confidence intervals for PCR-SC.

Implements the variance estimators from Section 4 of *Same Root
Different Leaves: Time Series and Cross-Sectional Methods in Panel
Data* (Econometrica 91(6):2125-2154). The closed-form formulas below
are ports of ``var.py`` from the authors' reference implementation
at https://github.com/deshen24/panel-data-regressions, adapted to
mlsynth's column-orientation (donors as columns of ``Y_pre``).

The three variance estimators correspond to three assumed sources of
randomness:

* **HZ** (horizontal / time-series). Assumption 1 of the paper:
  donors' post-period outcome :math:`y_T` is generated from their
  pre-period outcomes plus mean-zero noise that varies *across
  donors*. The relevant residual variance comes from the
  cross-sectional dimension. Confidence intervals built from
  :math:`\\sigma^2_{\\mathrm{hz}}` quantify uncertainty under the
  HZ generative model.
* **VT** (vertical / cross-sectional). Assumption 2 of the paper:
  treated unit's pre-period outcome :math:`y_N` is generated from
  the donors' pre-period outcomes plus mean-zero noise that varies
  *across pre-periods*. The relevant residual variance comes from
  the time-series dimension.
* **DR** (doubly robust). Assumption 3: noise varies across both
  units and time. The DR variance is :math:`v_{\\mathrm{hz}} +
  v_{\\mathrm{vt}} - \\mathrm{tr}(A)` where :math:`A` is an
  interaction term, clipped at zero (paper eq. 17 / var.py).

Three variance estimators are available:

* ``"homoskedastic"`` -- paper Section 4.1.3, eq 19. Closed-form,
  assumes errors are i.i.d.
* ``"jackknife"`` -- paper Supplemental Material. Diagonal weighting.
* ``"hrk"`` -- Hartley-Rao-Kish, paper Supplemental Material. Only
  valid when ``max(1 - diag(H_perp)) < 1/2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from scipy.stats import norm

from ....exceptions import MlsynthEstimationError


_VarKind = Literal["homoskedastic", "jackknife", "hrk"]
_Source = Literal["hz", "vt", "dr"]
_EPS = 1e-10


def _threshold(err: np.ndarray, tol: float = _EPS) -> np.ndarray:
    """Zero out numerically-tiny residual entries (var.py:err_threshold)."""
    out = err.copy()
    out[np.abs(out) <= tol] = 0.0
    return out


def _pcr_weights(X: np.ndarray, y: np.ndarray, rank: int) -> np.ndarray:
    """Truncated pseudo-inverse PCR weights: ``((V_r/s_r) @ U_r^T) @ y``.

    Equivalent to ``pinv(HSVT_r(X)) @ y`` but uses the SVD of ``X``
    directly so we avoid recomputing the truncation.
    """
    u, sv, vt = np.linalg.svd(X, full_matrices=False)
    r = max(1, min(int(rank), sv.size))
    return ((vt[:r, :].T / sv[:r]) @ u[:, :r].T) @ y


# ---------------------------------------------------------------------------
# Vectorised (batched-over-post-periods) variance estimators.
#
# These compute the same (var_hz, var_vt, trA) triples as the per-period
# ``_var_homo`` / ``_var_jack`` / ``_var_hrk`` above, but for *all* post-periods
# at once and with every period-invariant quantity (SVD-derived pseudo-inverses,
# projections, the y_n-side residuals, the interaction trace) computed a single
# time. They are what :func:`shen_inference` calls; the scalar versions are kept
# as the reference-matched source of truth (and are cross-validated against the
# authors' ``var.py`` in the benchmark suite). Inputs:
#   Y_post : (T1, J) donor outcomes over the post-period.
#   W_hz   : (T0, T1) HZ weights, column ``ti`` = w_hz for post-period ``ti``.
#   w_vt   : (J,) period-invariant VT weights.
#   Y0     : (J, T0) rank-k donor pre-matrix (``Y0_low``).
# ---------------------------------------------------------------------------
def _batch_var_homo(y_n, Y_post, Y0, W_hz, w_vt, Hu_perp, Hv_perp):
    T1, J = Y_post.shape
    T0d = y_n.shape[0]
    R = int(round(np.trace(Y0 @ np.linalg.pinv(Y0))))

    sig2_hz = np.zeros(T1)
    if R != J:
        E = _threshold(Hu_perp @ Y_post.T)            # (J, T1)
        sig2_hz = (E ** 2).sum(axis=0) / (J - R)      # (T1,)
    v_hz = sig2_hz * float(np.dot(w_vt, w_vt))

    sig2_vt = 0.0
    if R != T0d:
        err_vt = _threshold(Hv_perp @ y_n)
        sig2_vt = (np.linalg.norm(err_vt) ** 2) / (T0d - R)
    v_vt = sig2_vt * (W_hz ** 2).sum(axis=0)          # dot(w_hz, w_hz) per period

    trace_A0 = float(np.trace(np.linalg.pinv(Y0) @ np.linalg.pinv(Y0.T)))
    trA = sig2_hz * sig2_vt * trace_A0
    return v_hz, v_vt, trA


def _batch_var_jack(y_n, Y_post, Y0, W_hz, w_vt, Hu_perp, Hv_perp):
    T1, J = Y_post.shape
    T0d = y_n.shape[0]
    R = int(round(np.trace(Y0 @ np.linalg.pinv(Y0))))
    P1 = np.linalg.pinv(Y0)          # (T0, J)
    P2 = np.linalg.pinv(Y0.T)        # (J, T0)
    Q = P1 * P2.T                    # (T0, J): Q[i,j] = P1[i,j] P2[j,i]

    D_hz = np.zeros((J, T1))
    if R != J:
        H_inv_hz = np.linalg.pinv((Hu_perp * Hu_perp) * np.eye(J))
        E_hz = _threshold((Hu_perp @ Y_post.T) ** 2)  # (J, T1)
        D_hz = H_inv_hz @ E_hz                         # (J, T1)
    v_hz = ((w_vt[:, None] ** 2) * D_hz).sum(axis=0)

    d_vt = np.zeros(T0d)
    if R != T0d:
        H_inv_vt = np.linalg.pinv((Hv_perp * Hv_perp) * np.eye(T0d))
        e_vt = _threshold((Hv_perp @ y_n) ** 2)
        d_vt = H_inv_vt @ e_vt                          # (T0,)
    v_vt = ((W_hz ** 2) * d_vt[:, None]).sum(axis=0)

    trA = d_vt @ (Q @ D_hz)                              # (T1,)
    return v_hz, v_vt, trA


def _batch_var_hrk(y_n, Y_post, Y0, W_hz, w_vt, Hu_perp, Hv_perp):
    T1, J = Y_post.shape
    P1 = np.linalg.pinv(Y0)
    P2 = np.linalg.pinv(Y0.T)
    Q = P1 * P2.T

    H_inv_hz = np.linalg.pinv(Hu_perp * Hu_perp)
    D_hz = H_inv_hz @ _threshold((Hu_perp @ Y_post.T) ** 2)   # (J, T1)
    v_hz = ((w_vt[:, None] ** 2) * D_hz).sum(axis=0)

    H_inv_vt = np.linalg.pinv(Hv_perp * Hv_perp)
    d_vt = H_inv_vt @ _threshold((Hv_perp @ y_n) ** 2)        # (T0,)
    v_vt = ((W_hz ** 2) * d_vt[:, None]).sum(axis=0)

    trA = d_vt @ (Q @ D_hz)
    return v_hz, v_vt, trA


_BATCH_VARIANCE_FNS = {
    "homoskedastic": _batch_var_homo,
    "jackknife": _batch_var_jack,
    "hrk": _batch_var_hrk,
}


def _var_homo(
    y_n: np.ndarray,
    y_t: np.ndarray,
    Y0: np.ndarray,
    w_hz: np.ndarray,
    w_vt: np.ndarray,
    Hu_perp: np.ndarray,
    Hv_perp: np.ndarray,
) -> Tuple[float, float, float]:
    """Homoskedastic variance estimator (paper eq 19; var.py:var_homo)."""
    R = int(round(np.trace(Y0 @ np.linalg.pinv(Y0))))

    if R == len(y_t):
        sig2_hz = 0.0
        var_hz = 0.0
    else:
        err_hz = _threshold(Hu_perp @ y_t)
        sig2_hz = (np.linalg.norm(err_hz) ** 2) / (len(y_t) - R)
        var_hz = sig2_hz * float(np.dot(w_vt, w_vt))

    if R == len(y_n):
        sig2_vt = 0.0
        var_vt = 0.0
    else:
        err_vt = _threshold(Hv_perp @ y_n)
        sig2_vt = (np.linalg.norm(err_vt) ** 2) / (len(y_n) - R)
        var_vt = sig2_vt * float(np.dot(w_hz, w_hz))

    A = (sig2_hz * sig2_vt) * (np.linalg.pinv(Y0) @ np.linalg.pinv(Y0.T))
    return float(var_hz), float(var_vt), float(np.trace(A))


def _var_jack(
    y_n: np.ndarray,
    y_t: np.ndarray,
    Y0: np.ndarray,
    w_hz: np.ndarray,
    w_vt: np.ndarray,
    Hu_perp: np.ndarray,
    Hv_perp: np.ndarray,
) -> Tuple[float, float, float]:
    """Jackknife variance estimator (var.py:var_jack)."""
    R = int(round(np.trace(Y0 @ np.linalg.pinv(Y0))))

    if R == len(y_t):
        var_hz = 0.0
        sigma_hz = np.zeros((len(y_t), len(y_t)))
    else:
        diag_u = Hu_perp * Hu_perp * np.eye(Hu_perp.shape[0])
        H_inv_hz = np.linalg.pinv(diag_u)
        err_hz = _threshold((Hu_perp @ y_t) * (Hu_perp @ y_t))
        sigma_hz = np.diag(H_inv_hz @ err_hz)
        var_hz = float(np.dot(w_vt, sigma_hz @ w_vt))

    if R == len(y_n):
        var_vt = 0.0
        sigma_vt = np.zeros((len(y_n), len(y_n)))
    else:
        diag_v = Hv_perp * Hv_perp * np.eye(Hv_perp.shape[0])
        H_inv_vt = np.linalg.pinv(diag_v)
        err_vt = _threshold((Hv_perp @ y_n) * (Hv_perp @ y_n))
        sigma_vt = np.diag(H_inv_vt @ err_vt)
        var_vt = float(np.dot(w_hz, sigma_vt @ w_hz))

    A = np.linalg.pinv(Y0) @ sigma_hz @ np.linalg.pinv(Y0.T) @ sigma_vt
    return var_hz, var_vt, float(np.trace(A))


def _var_hrk(
    y_n: np.ndarray,
    y_t: np.ndarray,
    Y0: np.ndarray,
    w_hz: np.ndarray,
    w_vt: np.ndarray,
    Hu_perp: np.ndarray,
    Hv_perp: np.ndarray,
) -> Tuple[float, float, float]:
    """HRK variance estimator (var.py:var_hrk).

    Valid only when ``max(1 - diag(H_perp)) < 1/2`` for both
    projections; the caller is responsible for that check.
    """
    H_inv_hz = np.linalg.pinv(Hu_perp * Hu_perp)
    err_hz = _threshold((Hu_perp @ y_t) * (Hu_perp @ y_t))
    sigma_hz = np.diag(H_inv_hz @ err_hz)
    var_hz = float(np.dot(w_vt, sigma_hz @ w_vt))

    H_inv_vt = np.linalg.pinv(Hv_perp * Hv_perp)
    err_vt = _threshold((Hv_perp @ y_n) * (Hv_perp @ y_n))
    sigma_vt = np.diag(H_inv_vt @ err_vt)
    var_vt = float(np.dot(w_hz, sigma_vt @ w_hz))

    A = np.linalg.pinv(Y0) @ sigma_hz @ np.linalg.pinv(Y0.T) @ sigma_vt
    return var_hz, var_vt, float(np.trace(A))


_VARIANCE_FNS = {
    "homoskedastic": _var_homo,
    "jackknife": _var_jack,
    "hrk": _var_hrk,
}


@dataclass(frozen=True)
class ShenInference:
    """Output of :func:`shen_inference` for a PCR-SC fit.

    Attributes
    ----------
    method : str
        Variance estimator tag.
    alpha : float
        Nominal level used to form the CIs.
    att : float
        Point estimate of the ATT (average gap over the post-period).
    per_period_gap : np.ndarray
        Per-period treatment effect estimates, shape ``(T1,)``.
    per_period_se_{hz,vt,dr} : np.ndarray
        Per-period standard errors under each source-of-randomness
        assumption, shape ``(T1,)``.
    per_period_ci_{hz,vt,dr} : np.ndarray
        Per-period :math:`(1-\\alpha)` CIs, shape ``(T1, 2)``.
    att_se_{hz,vt,dr} : float
        ATT standard errors (variance-of-mean under independence).
    att_ci_{hz,vt,dr} : tuple of float
        ATT :math:`(1-\\alpha)` CIs.
    rank : int
        PCR rank used.
    """

    method: str
    alpha: float
    att: float
    per_period_gap: np.ndarray
    per_period_se_hz: np.ndarray
    per_period_se_vt: np.ndarray
    per_period_se_dr: np.ndarray
    per_period_ci_hz: np.ndarray
    per_period_ci_vt: np.ndarray
    per_period_ci_dr: np.ndarray
    att_se_hz: float
    att_se_vt: float
    att_se_dr: float
    att_ci_hz: Tuple[float, float]
    att_ci_vt: Tuple[float, float]
    att_ci_dr: Tuple[float, float]
    rank: int


def shen_inference(
    treated_outcome: np.ndarray,
    donor_outcomes: np.ndarray,
    T0: int,
    rank: int,
    *,
    variance: _VarKind = "homoskedastic",
    alpha: float = 0.05,
) -> ShenInference:
    """Per-period and ATT CIs for PCR-SC, following Shen et al. (2023).

    Parameters
    ----------
    treated_outcome : np.ndarray
        Treated outcome series, shape ``(T,)``.
    donor_outcomes : np.ndarray
        Donor outcomes (columns = donors), shape ``(T, J)``.
    T0 : int
        Number of pre-treatment periods.
    rank : int
        PCR truncation rank :math:`k` already chosen by the
        pipeline. Reused so the inference uses the same low-rank
        donor matrix the weights were fit on.
    variance : {"homoskedastic", "jackknife", "hrk"}
        Variance estimator. Defaults to homoskedastic (paper eq 19).
    alpha : float
        Two-sided level for the returned CIs.
    """
    if variance not in _VARIANCE_FNS:
        raise MlsynthEstimationError(
            f"Unknown variance estimator {variance!r}; expected one of "
            f"{sorted(_VARIANCE_FNS)}."
        )
    if not (0.0 < alpha < 1.0):
        raise MlsynthEstimationError("alpha must lie in (0, 1).")

    y = np.asarray(treated_outcome, dtype=float).flatten()
    Y_full = np.asarray(donor_outcomes, dtype=float)
    T = y.shape[0]
    T1 = T - T0
    if T1 < 1:
        raise MlsynthEstimationError("Need at least one post-period to infer.")
    J = Y_full.shape[1]
    y_n = y[:T0]                # treated pre-period (T0,)
    Y_pre = Y_full[:T0]         # (T0, J) donors as columns

    # Paper convention: Y0 has donors as ROWS, so transpose.
    Y0 = Y_pre.T                # (J, T0)

    # Rank-k SVD; build paper-notation projections.
    U, sv, Vt = np.linalg.svd(Y0, full_matrices=False)
    k = max(1, min(int(rank), sv.size))
    U_k = U[:, :k]; Vt_k = Vt[:k, :]
    Hu = U_k @ U_k.T            # (J, J)
    Hv = Vt_k.T @ Vt_k          # (T0, T0)
    Hu_perp = np.eye(J) - Hu
    Hv_perp = np.eye(T0) - Hv
    Y0_low = (U_k * sv[:k]) @ Vt_k  # rank-k Y0

    if variance == "hrk":
        u_excess = float(np.max(1.0 - np.diagonal(Hu_perp)))
        v_excess = float(np.max(1.0 - np.diagonal(Hv_perp)))
        if u_excess >= 0.5 or v_excess >= 0.5:
            raise MlsynthEstimationError(
                "HRK variance requires max(1 - diag(H_perp)) < 1/2 for both "
                "projections; got "
                f"max_u={u_excess:.3f}, max_v={v_excess:.3f}."
            )

    z = norm.ppf(1.0 - alpha / 2.0)
    batch_var_fn = _BATCH_VARIANCE_FNS[variance]

    # Period-invariant pieces, computed once. The truncated pseudo-inverses
    # reuse the SVD of Y0 already taken above (P_hz @ y = pinv_k(Y0) @ y,
    # P_vt @ y = pinv_k(Y0^T) @ y), so the per-period loop becomes pure
    # matrix-vector work -- no SVD/pinv is recomputed across post-periods.
    P_hz = (Vt_k.T / sv[:k]) @ U_k.T            # (T0, J): w_hz = P_hz @ y_t
    P_vt = (U_k / sv[:k]) @ Vt_k                # (J, T0): w_vt = P_vt @ y_n
    w_vt = P_vt @ y_n                           # (J,) -- period-invariant
    Y_post = Y_full[T0:]                        # (T1, J)
    W_hz = P_hz @ Y_post.T                      # (T0, T1): columns are w_hz

    # Theorem 1: HZ and VT predictions coincide for PCR at the same rank.
    gaps = y[T0:] - Y_post @ w_vt               # (T1,)

    v_hz, v_vt, trA = batch_var_fn(
        y_n, Y_post, Y0_low, W_hz, w_vt, Hu_perp, Hv_perp,
    )
    v_dr = np.maximum(0.0, v_hz + v_vt - trA)

    se_hz = np.sqrt(v_hz)
    se_vt = np.sqrt(v_vt)
    se_dr = np.sqrt(v_dr)
    per_ci_hz = np.column_stack([gaps - z * se_hz, gaps + z * se_hz])
    per_ci_vt = np.column_stack([gaps - z * se_vt, gaps + z * se_vt])
    per_ci_dr = np.column_stack([gaps - z * se_dr, gaps + z * se_dr])

    att = float(gaps.mean())
    # ATT variance under independence-across-post-periods (first-pass
    # standard; the paper does not derive a multi-period closed form).
    se_att_hz = float(np.sqrt(v_hz.mean() / T1))
    se_att_vt = float(np.sqrt(v_vt.mean() / T1))
    se_att_dr = float(np.sqrt(v_dr.mean() / T1))
    att_ci_hz = (att - z * se_att_hz, att + z * se_att_hz)
    att_ci_vt = (att - z * se_att_vt, att + z * se_att_vt)
    att_ci_dr = (att - z * se_att_dr, att + z * se_att_dr)

    return ShenInference(
        method=f"shen_{variance}",
        alpha=float(alpha),
        att=att,
        per_period_gap=gaps,
        per_period_se_hz=se_hz,
        per_period_se_vt=se_vt,
        per_period_se_dr=se_dr,
        per_period_ci_hz=per_ci_hz,
        per_period_ci_vt=per_ci_vt,
        per_period_ci_dr=per_ci_dr,
        att_se_hz=se_att_hz,
        att_se_vt=se_att_vt,
        att_se_dr=se_att_dr,
        att_ci_hz=att_ci_hz,
        att_ci_vt=att_ci_vt,
        att_ci_dr=att_ci_dr,
        rank=k,
    )
