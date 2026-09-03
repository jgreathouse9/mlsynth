"""The TWSF method: two regressions and a bilinear combination.

Section references are to Shen (arXiv:2606.18512v2): ``algorithm.tex`` for the
one-step estimator, ``horizon.tex`` for the direct and recursive multi-step
extensions, and ``results.tex`` for the pooled variance.

The linear-algebra kernel is :mod:`mlsynth.utils.pcr.core`, shared with SI,
ClusterSC and SNN. Both pseudo-inverses are built from the truncated factors
:func:`~mlsynth.utils.pcr.core.hsvt` returns: ``HSVT(A, k)`` has rank exactly
``k``, so inverting it with a tolerance-based pseudo-inverse keeps directions
that are numerically zero and inflates the result without bound.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm

from ...exceptions import MlsynthConfigError, MlsynthEstimationError
from ..pcr.core import hsvt, pcr_weights
from .structures import TWSFFit


# --------------------------------------------------------------------------
# Page construction
# --------------------------------------------------------------------------

def page_blocks(Y_post: np.ndarray, L: int, lead: int = 1
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(Z_lag, z_next, W)`` from eqs. (page.train)-(W.block).

    Each donor's treated series is cut into non-overlapping blocks of length
    ``L + lead``; the first ``L`` rows of a block are a lag vector and the last
    is its ``lead``-step-ahead response. Stacking the donors horizontally gives
    the ``L x M`` design and the length-``M`` response. ``W`` is the donors'
    terminal ``L`` observations, the state the forecast starts from.

    ``lead = 1`` is the one-step and recursive construction; ``lead = h`` is the
    direct estimator's, which needs longer blocks and so fewer of them.
    """
    N1, T1 = Y_post.shape
    width = L + lead
    n_blocks = T1 // width
    if n_blocks < 2:
        raise MlsynthConfigError(
            f"the treated window supplies {n_blocks} Page block(s) of length "
            f"{width} (L = {L}, lead = {lead}, T1 = {T1}); at least 2 are "
            "needed, since the last is held back for the forecast state. "
            "Shorten L, shorten the horizon, or use multistep='recursive', "
            "which needs blocks of length L + 1 instead of L + h."
        )
    lags, resp = [], []
    for j in range(N1):
        for b in range(n_blocks - 1):
            seg = Y_post[j, b * width:(b + 1) * width]
            lags.append(seg[:L])
            resp.append(seg[-1])
    Z_lag = np.stack(lags, axis=1)          # L x M
    z_next = np.asarray(resp, dtype=float)  # M
    W = Y_post[:, T1 - L:]                  # N1 x L
    return Z_lag, z_next, W


# --------------------------------------------------------------------------
# the companion recursion
# --------------------------------------------------------------------------

def companion(x: np.ndarray) -> np.ndarray:
    """``Pi(x)`` of eq. (pi.matrix): shift the state, append ``x'`` as its law."""
    L = x.size
    P = np.zeros((L, L))
    P[:L - 1, 1:] = np.eye(L - 1)
    P[L - 1] = x
    return P


def lead_map(x: np.ndarray, ell: int) -> np.ndarray:
    """``g_ell(x) = (Pi(x)^ell)' e_L``: the L-lag state's lead-``ell`` forecast."""
    e = np.zeros(x.size); e[-1] = 1.0
    return np.linalg.matrix_power(companion(x), ell).T @ e


def lead_jacobian(x: np.ndarray, ell: int) -> np.ndarray:
    """Jacobian of :func:`lead_map`, by the representation in horizon.tex.

    This is how a first-order perturbation in the estimated one-step rule
    propagates into the lead-``ell`` forecast, and it is why the recursive
    interval is wider than the one-step interval scaled by the horizon.
    """
    e = np.zeros(x.size); e[-1] = 1.0
    Pi = companion(x)
    out = np.zeros((x.size, x.size))
    for a in range(ell):
        out += float(e @ np.linalg.matrix_power(Pi, a) @ e) * \
            np.linalg.matrix_power(Pi, ell - 1 - a).T
    return out


def _truncated_pinv(A: np.ndarray, k: int) -> np.ndarray:
    """Pseudo-inverse of ``HSVT(A, k)``, taken at its construction rank."""
    _, U, s, Vt = hsvt(A, k)
    return (Vt.T / s) @ U.T


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------

def fit_twsf(y_target_pre: np.ndarray, Y_donors_pre: np.ndarray,
             Y_donors_post: np.ndarray, L: int, k_y: int, k_z: int,
             horizon: int, multistep: str = "recursive",
             alpha_level: float = 0.10,
             interval: str = "confidence") -> TWSFFit:
    """Fit TWSF and return the forecast path with its pointwise interval.

    The two halves are estimated separately -- the unit side on the control
    window, the time side on the donors' treated window -- and combined as
    ``theta = <alpha, W' beta>``: the unit weights place the target inside the
    treated regime, and the temporal rule advances it.
    """
    k_y = min(k_y, *Y_donors_pre.shape)
    beta = pcr_weights(Y_donors_pre.T, y_target_pre, k_y)   # eq. (pcr.beta.hat)

    Z1, z1, W = page_blocks(Y_donors_post, L, lead=1)
    k_z_eff = min(k_z, *Z1.shape)
    alpha_1 = pcr_weights(Z1.T, z1, k_z_eff)                # eq. (pcr.alpha.hat)
    state = W.T @ beta                                       # imputed treated state

    # pooled unit- and time-side PCR residuals, eq. (var.sigma.pool)
    ru = y_target_pre - Y_donors_pre.T @ beta
    rt = z1 - Z1.T @ alpha_1
    dof = max((y_target_pre.size - k_y) + (z1.size - k_z_eff), 1)
    sigma2 = float((np.sum(ru ** 2) + np.sum(rt ** 2)) / dof)

    Yk_pinv = _truncated_pinv(Y_donors_pre, k_y)
    Zk_pinv = _truncated_pinv(Z1, k_z_eff)
    n_blocks = Y_donors_post.shape[1] // (L + 1)

    forecast, se = np.empty(horizon), np.empty(horizon)
    for m in range(1, horizon + 1):
        if multistep == "recursive":
            a_m = lead_map(alpha_1, m)
            J_m = lead_jacobian(alpha_1, m)
            q_a = Zk_pinv @ (J_m.T @ state)
            a_ref = alpha_1
        else:
            Zm, zm, _ = page_blocks(Y_donors_post, L, lead=m)
            k_m = min(k_z, *Zm.shape)
            a_m = pcr_weights(Zm.T, zm, k_m)                # eq. (alpha.direct)
            q_a = _truncated_pinv(Zm, k_m) @ state
            a_ref = a_m
        forecast[m - 1] = float(a_m @ state)
        q_b = Yk_pinv @ (W @ a_m)
        var = sigma2 * (np.sum(a_m ** 2) * np.sum(beta ** 2)
                        + np.sum(q_b ** 2) * (1 + np.sum(beta ** 2))
                        + np.sum(q_a ** 2) * (1 + np.sum(a_ref ** 2)))
        if interval == "prediction":
            var += sigma2                                    # future innovation
        se[m - 1] = float(np.sqrt(max(var, 0.0)))

    z = float(norm.ppf(1.0 - alpha_level / 2.0))
    return TWSFFit(forecast=forecast, std_error=se,
                   lower=forecast - z * se, upper=forecast + z * se,
                   beta=beta, alpha=alpha_1, sigma2=sigma2,
                   n_blocks=int(n_blocks), multistep=multistep)
