"""Numerical core for PROPSC: common-weights SC / SDID / DID for proportions.

A faithful, NumPy-only port of the multivariate estimator in the authors' R
package ``propsdid`` (``lstoetze/propsdid``; ``R/{solver,utils,synthdid,
vcov_multivar}.R``), which accompanies Bogatyrev & Stoetzer (2026), "Estimating
Treatment Effects on Proportions with Synthetic Controls," *Political Analysis*.

The estimator fits a *single* set of unit weights :math:`\\omega` and time
weights :math:`\\lambda`, shared across all :math:`K` compositional outcomes
(the "constant control comparison"), then reads off each outcome's ATT via the
usual synthetic-DID double difference. Because the weights are common, the
:math:`K` ATTs sum to zero by construction.

The weights come from the ``synthdid`` Frank-Wolfe solver with an exact line
search and the two-round sparsify pass (not a one-shot QP); the 3-D objective is
the per-outcome objective stacked over the ``K`` slices. This module is the
oracle-matching core: it reproduces the R package cell-by-cell (agreement at the
level of floating-point reordering) on the tiny fixture and on the paper's Spain
and Poland applications. See ``benchmarks/cases/propsc_spain.py``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# collapsed forms (utils.R: collapsed.form / collapsed.form.3d)
# ---------------------------------------------------------------------------
def collapsed_form(Y: np.ndarray, N0: int, T0: int) -> np.ndarray:
    """Collapse ``Y`` to an ``(N0+1) x (T0+1)`` block-mean matrix."""
    top = np.hstack([Y[:N0, :T0], Y[:N0, T0:].mean(axis=1, keepdims=True)])
    bot = np.hstack([Y[N0:, :T0].mean(axis=0, keepdims=True),
                     np.array([[Y[N0:, T0:].mean()]])])
    return np.vstack([top, bot])


def collapsed_form_3d(Y3d: np.ndarray, N0: int, T0: int) -> np.ndarray:
    """Apply :func:`collapsed_form` to each of the ``K`` outcome slices."""
    N, T, K = Y3d.shape
    out = np.empty((N0 + 1, T0 + 1, K))
    for k in range(K):
        out[:, :, k] = collapsed_form(Y3d[:, :, k], N0, T0)
    return out


# ---------------------------------------------------------------------------
# Frank-Wolfe simplex solver (solver.R: fw.step / sc.weight.fw)
# ---------------------------------------------------------------------------
def _fw_step(A: np.ndarray, x: np.ndarray, b: np.ndarray, eta: float) -> np.ndarray:
    """One Frank-Wolfe step for ``||Ax - b||^2 + eta*||x||^2`` on the simplex."""
    Ax = A @ x
    half_grad = (Ax - b) @ A + eta * x
    i = int(np.argmin(half_grad))
    d_x = -x.copy()
    d_x[i] = 1.0 - x[i]
    if np.all(d_x == 0):  # pragma: no cover - already at the optimal vertex
        return x
    d_err = A[:, i] - Ax
    step = -(half_grad @ d_x) / (np.sum(d_err ** 2) + eta * np.sum(d_x ** 2))
    return x + min(1.0, max(0.0, step)) * d_x


def sc_weight_fw(Y: np.ndarray, zeta: float, intercept: bool = True,
                 lam: Optional[np.ndarray] = None, min_decrease: float = 1e-3,
                 max_iter: int = 1000) -> np.ndarray:
    """Frank-Wolfe SC weights. ``Y`` is ``(N0, T0+1)`` or ``(N0, T0+1, K)``.

    For the 3-D (common-weights) case each outcome slice is column-centered
    (when ``intercept``) and the slices are stacked row-wise, so the single
    weight vector minimizes the summed pre-treatment fit across all ``K``
    outcomes. The ridge coefficient is ``eta = N0 * zeta**2`` with the
    pre-stack ``N0``.
    """
    Y = np.asarray(Y, dtype=float)
    N0 = Y.shape[0]
    T0 = Y.shape[1] - 1
    if lam is None:
        lam = np.full(T0, 1.0 / T0)
    else:
        lam = np.asarray(lam, dtype=float).copy()

    if Y.ndim == 3:
        K = Y.shape[2]
        Ys = Y.copy()
        if intercept:
            for k in range(K):
                Ys[:, :, k] -= Ys[:, :, k].mean(axis=0, keepdims=True)
        Ymat = np.vstack([Ys[:, :, k] for k in range(K)])
    else:
        Ymat = Y.copy()
        if intercept:
            Ymat -= Ymat.mean(axis=0, keepdims=True)

    A = Ymat[:, :T0]
    b = Ymat[:, T0]
    eta = N0 * (zeta ** 2)

    vals = []
    t = 0
    while t < max_iter and (t < 2 or vals[t - 2] - vals[t - 1] > min_decrease ** 2):
        lam = _fw_step(A, lam, b, eta)
        err = Ymat @ np.append(lam, -1.0)
        vals.append((zeta ** 2) * np.sum(lam ** 2) + np.sum(err ** 2) / len(b))
        t += 1
    return lam


def _sparsify(v: np.ndarray) -> np.ndarray:
    """synthdid.R sparsify_function: zero out small weights, renormalize."""
    v = v.copy()
    v[v <= v.max() / 4.0] = 0.0
    return v / v.sum()


def _noise_level(Y: np.ndarray, N0: int, T0: int) -> float:
    """Pooled sd of pre-period control first differences (``ddof=1``)."""
    diffs = np.diff(Y[:N0, :T0, :], axis=1)
    return float(diffs.std(ddof=1))


# ---------------------------------------------------------------------------
# multivariate estimator (synthdid.R: synthdid_estimate_multivar / sc_estimate)
# ---------------------------------------------------------------------------
def estimate_common_weights(
    Y: np.ndarray, N0: int, T0: int, method: str = "sdid",
    weights: Optional[Dict[str, Optional[np.ndarray]]] = None,
    max_iter: int = 10000, max_iter_pre_sparsify: int = 100,
) -> Dict[str, object]:
    """Common-weights estimate over a ``(N, T, K)`` proportion array.

    Parameters
    ----------
    Y : np.ndarray
        ``(N, T, K)`` outcomes, controls in rows ``0..N0-1``, treated last;
        pre-treatment periods in columns ``0..T0-1``.
    N0, T0 : int
        Control-unit count and pre-treatment period count.
    method : {"sdid", "sc", "did"}
        ``sdid`` estimates both unit and time weights; ``sc`` uses zero time
        weights and an intercept-free unit fit with negligible ridge; ``did``
        fixes uniform unit and time weights.

    Returns
    -------
    dict
        ``estimate`` (length-``K`` ATT vector), ``omega`` (unit weights),
        ``lambda`` (time weights).
    """
    Y = np.asarray(Y, dtype=float)
    N, T, K = Y.shape
    N1, T1 = N - N0, T - T0
    noise = _noise_level(Y, N0, T0)

    if method == "did":
        omega = np.full(N0, 1.0 / N0)
        lam = np.full(T0, 1.0 / T0)
    elif method == "sc":
        lam = np.zeros(T0)
        omega = _solve_omega(Y, N0, T0, eta_omega=1e-6, noise=noise,
                             intercept=False, max_iter=max_iter,
                             max_iter_pre=max_iter_pre_sparsify)
    elif method == "sdid":
        eta_omega = ((N1) * (T - T0) * K) ** 0.25
        lam = _solve_lambda(Y, N0, T0, noise=noise, max_iter=max_iter,
                            max_iter_pre=max_iter_pre_sparsify)
        omega = _solve_omega(Y, N0, T0, eta_omega=eta_omega, noise=noise,
                             intercept=True, max_iter=max_iter,
                             max_iter_pre=max_iter_pre_sparsify)
    else:
        raise ValueError('method must be "sdid", "sc", or "did"')

    ovec = np.append(-omega, np.full(N1, 1.0 / N1))
    lvec = np.append(-lam, np.full(T1, 1.0 / T1))
    est = np.array([ovec @ Y[:, :, k] @ lvec for k in range(K)])
    return {"estimate": est, "omega": omega, "lambda": lam, "noise_level": noise}


def _solve_lambda(Y, N0, T0, noise, max_iter, max_iter_pre):
    zeta_lambda = 1e-6 * noise
    md = 1e-5 * noise
    Yc = collapsed_form_3d(Y, N0, T0)
    lam = sc_weight_fw(Yc[:N0, :, :], zeta_lambda, True, None, md, max_iter_pre)
    lam = sc_weight_fw(Yc[:N0, :, :], zeta_lambda, True, _sparsify(lam), md, max_iter)
    return lam


def _solve_omega(Y, N0, T0, eta_omega, noise, intercept, max_iter, max_iter_pre):
    zeta_omega = eta_omega * noise
    md = 1e-5 * noise
    Yc = collapsed_form_3d(Y, N0, T0)
    Yct = np.transpose(Yc[:, :T0, :], (1, 0, 2))  # (T0, N0+1, K)
    om = sc_weight_fw(Yct, zeta_omega, intercept, None, md, max_iter_pre)
    om = sc_weight_fw(Yct, zeta_omega, intercept, _sparsify(om), md, max_iter)
    return om


# ---------------------------------------------------------------------------
# fixed-weights jackknife (vcov_multivar.R: jackknife_se_multi)
# ---------------------------------------------------------------------------
def _sum_normalize(x: np.ndarray) -> np.ndarray:
    s = x.sum()
    return x / s if s != 0 else np.full(len(x), 1.0 / len(x))


def _estimate_fixed(Y, N0, omega, lam):
    N, T, K = Y.shape
    N1, T1 = N - N0, T - len(lam)
    ovec = np.append(-omega, np.full(N1, 1.0 / N1))
    lvec = np.append(-lam, np.full(T1, 1.0 / T1))
    return np.array([ovec @ Y[:, :, k] @ lvec for k in range(K)])


def jackknife_se(Y: np.ndarray, N0: int, omega: np.ndarray,
                 lam: np.ndarray) -> np.ndarray:
    """Fixed-weights leave-one-unit-out jackknife SE per outcome.

    Algorithm 3 of Arkhangelsky et al. (2021) as adapted to the multivariate
    estimator in ``vcov_multivar.R``: weights are held fixed (unit weights
    renormalized over the retained controls) and the ``K``-vector estimate is
    recomputed on each leave-one-out panel. Returns ``NaN`` when there is a
    single treated unit.
    """
    Y = np.asarray(Y, dtype=float)
    N, T, K = Y.shape
    omega = np.asarray(omega, dtype=float)
    lam = np.asarray(lam, dtype=float)
    if N0 >= N - 1:
        return np.full(K, np.nan)
    u = np.empty((N, K))
    for i in range(N):
        keep = [j for j in range(N) if j != i]
        Yk = Y[keep, :, :]
        if i < N0:  # dropped a control -> renormalize its weight away
            om = _sum_normalize(np.delete(omega, i))
            N0_new = N0 - 1
        else:       # dropped a treated unit -> weights unchanged
            om = omega
            N0_new = N0
        u[i, :] = _estimate_fixed(Yk, N0_new, om, lam)
    return np.sqrt(((N - 1) / N) * (N - 1) * u.var(axis=0, ddof=1))
