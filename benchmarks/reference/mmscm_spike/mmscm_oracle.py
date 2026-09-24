"""Readable port of Kato & Ohda (arXiv 2307.11127v5) MMSCM, plus tie-breaks.

The published estimator (their eq. 4) matches ``G`` scalar moments of the
treated unit's pre-period outcome to a simplex-weighted sum of the donors':

    w_hat = argmin_{w in Delta}  sum_g v_g | m_0[g] - (A w)[g] |

with ``A[g, j] = mean_t (Y_jt)^g`` and ``m_0[g] = mean_t (Y_0t)^g``, on
outcomes rescaled by the panel's column-wise max (their ``_data_setup``).

With ``G`` moment equations and ``J`` donors and ``G << J`` the minimiser is a
face of the simplex, not a point. Their Theorem 5.8 says as much -- it defines
``Phi_dagger`` as an argmin *set* and proves convergence to that set. This
module solves the problem lexicographically instead: stage 1 finds the optimal
moment loss, stage 2 selects one point of the optimal face by a stated rule.

Selectors
---------
``minnorm``  minimise ||w||_2 -- the point of the face closest to uniform.
``pathfit``  minimise the pre-period path SSE ||y_0 - Y_d w||^2 -- moments
             first, then spend the remaining J - G degrees of freedom the way
             synthetic control ordinarily does.
``none``     the published estimator, solved by SLSQP from a given start, kept
             so the seed dependence stays visible.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
import scipy.optimize


def moment_design(Y_pre: np.ndarray, n_moments: int, scale: np.ndarray | None = None):
    """Return ``(A, m0, scale)`` for eq. (4).

    ``Y_pre`` is ``T0 x (1 + J)`` with the treated unit in column 0. Outcomes
    are divided by the column-wise max absolute value over all units, which is
    the reference's normalisation and what makes Assumption 5.4's [0, 1]
    bound hold.
    """
    if scale is None:
        scale = np.abs(Y_pre).max()
    Z = Y_pre / scale
    powers = np.arange(1, n_moments + 1)
    M = np.stack([(Z ** g).mean(axis=0) for g in powers])   # (G, 1+J)
    return M[:, 1:], M[:, 0], scale


def _moment_loss(A, m0, w, v):
    r = m0 - A @ w
    return float(np.sum(v * r ** 2))


def fit(Y_pre: np.ndarray, n_moments: int = 5, selector: str = "pathfit",
        v: np.ndarray | None = None, slack: float = 1e-6,
        x0: np.ndarray | None = None) -> dict:
    """Fit MMSCM weights on the pre-period block ``Y_pre`` (``T0 x (1+J)``)."""
    A, m0, scale = moment_design(Y_pre, n_moments)
    G, J = A.shape
    v = np.ones(G) if v is None else np.asarray(v, float)

    if selector == "none":
        # the published solve: SLSQP on the squared-mean loss from a start
        start = np.full(J, 1.0 / J) if x0 is None else np.asarray(x0, float)
        r = scipy.optimize.minimize(
            lambda w: _moment_loss(A, m0, w, v), start, method="SLSQP",
            bounds=[(0.0, 1.0)] * J,
            constraints=({"type": "eq", "fun": lambda w: 1.0 - w.sum()},))
        w = np.clip(r.x, 0, None)
        return {"w": w / w.sum(), "moment_loss": float(r.fun), "selector": selector}

    # ---- stage 1: the optimal moment loss --------------------------------
    sv = np.sqrt(v)
    scale2 = float(np.sum((sv * m0) ** 2))          # the loss's natural scale
    wv = cp.Variable(J, nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(cp.multiply(sv, m0 - A @ wv))),
               [cp.sum(wv) == 1]).solve(solver=cp.CLARABEL)
    L_star = float(np.sum((sv * (m0 - A @ np.asarray(wv.value, float))) ** 2))

    # ---- stage 2: select one point of the near-optimal set ---------------
    # The slack is absolute against the loss's own scale, not relative to
    # L_star: L_star can be ~1e-10 while the objective stays flat to ~1e-6, so a
    # slack proportional to L_star is a razor-thin constraint no solver accepts,
    # and one proportional to the data scale asks the interpretable question --
    # among weights whose moment loss is negligible on the scale of the moments
    # themselves, which one do we take?
    cap = L_star + slack * scale2
    w2 = cp.Variable(J, nonneg=True)
    cons2 = [cp.sum(w2) == 1,
             cp.sum_squares(cp.multiply(sv, m0 - A @ w2)) <= cap]
    if selector == "minnorm":
        obj2 = cp.Minimize(cp.sum_squares(w2))
    elif selector == "pathfit":
        obj2 = cp.Minimize(cp.sum_squares(Y_pre[:, 0] - Y_pre[:, 1:] @ w2))
    else:
        raise ValueError(f"unknown selector {selector!r}")
    prob2 = cp.Problem(obj2, cons2)
    for solver in (cp.CLARABEL, cp.SCS, cp.ECOS):
        try:
            prob2.solve(solver=solver)
        except Exception:
            continue
        if w2.value is not None:
            break
    if w2.value is None:
        raise RuntimeError(f"stage 2 failed ({selector!r}, L*={L_star:.3e})")
    w = np.clip(np.asarray(w2.value, float), 0, None)
    w = w / w.sum()
    return {"w": w, "moment_loss": _moment_loss(A, m0, w, v),
            "moment_loss_optimal": L_star, "loss_scale": scale2,
            "slack": slack, "selector": selector}


def moment_weights(m0: np.ndarray, weighting: str = "unit") -> np.ndarray:
    """The v_gamma of eq. (4).

    ``unit`` is the paper's own choice ("we set v_gamma = 1"). On outcomes
    rescaled into [0, 1] the gamma-th raw moment decays geometrically, so a
    unit-weighted loss is dominated by the first moment and the higher ones
    barely enter. ``scaled`` sets v_gamma = 1 / m0[gamma]^2, which makes every
    matched moment contribute equally -- still admissible, since the paper
    allows any v_gamma in (0, inf).
    """
    if weighting == "unit":
        return np.ones_like(m0)
    if weighting == "scaled":
        d = np.where(np.abs(m0) > 0, m0, 1.0)
        return 1.0 / d ** 2
    raise ValueError(f"unknown weighting {weighting!r}")


def argmin_set_box(Y_pre: np.ndarray, n_moments: int, v: np.ndarray | None = None,
                   slack: float = 1e-6):
    """Per-donor range of the near-optimal set, and what it implies for the ATT.

    For each donor j, minimise and maximise ``w_j`` subject to the moment loss
    staying within ``slack * loss_scale`` of its optimum. The width of that box
    is how much the weights can move without the objective noticing.
    """
    A, m0, _ = moment_design(Y_pre, n_moments)
    G, J = A.shape
    v = np.ones(G) if v is None else np.asarray(v, float)
    sv = np.sqrt(v)
    scale2 = float(np.sum((sv * m0) ** 2))
    wv = cp.Variable(J, nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(cp.multiply(sv, m0 - A @ wv))),
               [cp.sum(wv) == 1]).solve(solver=cp.CLARABEL)
    L_star = float(np.sum((sv * (m0 - A @ np.asarray(wv.value, float))) ** 2))
    cap = L_star + slack * scale2

    lo, hi = np.zeros(J), np.zeros(J)
    for j in range(J):
        for sense, out in ((cp.Minimize, lo), (cp.Maximize, hi)):
            x = cp.Variable(J, nonneg=True)
            pr = cp.Problem(sense(x[j]),
                            [cp.sum(x) == 1,
                             cp.sum_squares(cp.multiply(sv, m0 - A @ x)) <= cap])
            for solver in (cp.CLARABEL, cp.SCS, cp.ECOS):
                try:
                    pr.solve(solver=solver)
                except Exception:
                    continue
                if x.value is not None:
                    break
            out[j] = np.nan if x.value is None else float(x.value[j])
    return {"lo": lo, "hi": hi, "L_star": L_star, "cap": cap,
            "width_l1": float(np.nansum(hi - lo))}


def counterfactual(Y: np.ndarray, w: np.ndarray, T0: int, bias: bool = True):
    """Counterfactual path, with the reference's additive pre-period bias term."""
    cf = Y[:, 1:] @ w
    if bias:
        cf = cf + float(np.mean(Y[:T0, 0] - cf[:T0]))
    return cf


def demeaned_sc(Y_pre: np.ndarray) -> np.ndarray:
    """Ferman & Pinto (2021) demeaned SC: SC on unit-demeaned pre-period data.

    Their own fix for the bias MMSCM targets, and the baseline the paper's
    simulations never run.
    """
    mu = Y_pre.mean(axis=0)
    Z = Y_pre - mu
    J = Y_pre.shape[1] - 1
    w = cp.Variable(J, nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(Z[:, 0] - Z[:, 1:] @ w)),
               [cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
    out = np.clip(np.asarray(w.value, float), 0, None)
    return out / out.sum()


def abadie_sc(Y_pre: np.ndarray) -> np.ndarray:
    """Plain simplex SC on the pre-period outcome path."""
    J = Y_pre.shape[1] - 1
    w = cp.Variable(J, nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(Y_pre[:, 0] - Y_pre[:, 1:] @ w)),
               [cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
    out = np.clip(np.asarray(w.value, float), 0, None)
    return out / out.sum()
