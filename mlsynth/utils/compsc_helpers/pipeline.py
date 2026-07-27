"""Numerical core for COMPSC: Compositional Synthetic Controls.

A NumPy port of Algorithm 1 in Boussim (2026), "Compositional Synthetic
Controls", with one deliberate correction.

The paper transforms shares with the additive log-ratio map
``l_k(pi) = log(pi_k / pi_p)`` (eq. 9) and states that this makes ``l`` an
isometry between the simplex under the Aitchison metric and Euclidean space
(Remark 3), so that minimizing the stacked squared error (eqs. 17-18) minimizes
pre-treatment Aitchison distance. That is not so: the additive log-ratio map is
not an isometry -- the centered log-ratio is. The practical consequence is that
Algorithm 1's fitted weights depend on which category is nominated as the
baseline, an arbitrary labelling choice.

Both maps are implemented. ``geometry="clr"`` is the default and does what the
paper says it wants: it minimizes the pre-treatment Aitchison distance and is
invariant to relabelling the categories. ``geometry="alr"`` reproduces the
paper's published estimates cell-for-cell. The gap between them is reported as
a diagnostic rather than hidden.

Given weights, the counterfactual is the same object under either map: the
Frechet barycenter of the donor compositions under the Aitchison metric, which
is the closure of their weighted geometric mean (eq. 16).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..bilevel.active_set import solve_simplex_qp


# --------------------------------------------------------------------------- #
# simplex geometry
# --------------------------------------------------------------------------- #
def closure(x: np.ndarray) -> np.ndarray:
    """Renormalize so the last axis sums to one (Aitchison's closure operator).

    Parameters
    ----------
    x : np.ndarray
        Non-negative array with categories on the last axis.

    Returns
    -------
    np.ndarray
        Same shape, with ``out.sum(axis=-1) == 1``.
    """
    x = np.asarray(x, dtype=float)
    return x / x.sum(axis=-1, keepdims=True)


def clr(P: np.ndarray) -> np.ndarray:
    """Centered log-ratio transform, categories on the last axis.

    ``clr(pi)_k = log(pi_k) - mean_m log(pi_m)``. This is the isometry between
    the simplex under the Aitchison metric and Euclidean space, so Euclidean
    distance between clr vectors *is* Aitchison distance.
    """
    L = np.log(np.asarray(P, dtype=float))
    return L - L.mean(axis=-1, keepdims=True)


def alr(P: np.ndarray, base: int = -1) -> np.ndarray:
    """Additive log-ratio transform against category ``base`` (paper eq. 9).

    Returns the ``K-1`` log-odds ``log(pi_k / pi_base)``, dropping the baseline
    coordinate. Not an isometry for the Aitchison metric; see the module
    docstring.
    """
    P = np.asarray(P, dtype=float)
    L = np.log(P)
    k = base % P.shape[-1]
    return np.delete(L - L[..., [k]], k, axis=-1)


def aitchison_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Aitchison distance between compositions (paper eq. 4).

    Equal to ``||clr(a) - clr(b)||_2``, i.e. the square root of the mean over
    all ordered category pairs of the squared log-ratio discrepancy.

    Parameters
    ----------
    a, b : np.ndarray
        Compositions with categories on the last axis; broadcastable.

    Returns
    -------
    np.ndarray
        Distances, shaped like the broadcast of ``a`` and ``b`` minus the last
        axis.
    """
    return np.linalg.norm(clr(a) - clr(b), axis=-1)


def _transform(P: np.ndarray, geometry: str, base: int) -> np.ndarray:
    """Dispatch to the configured log-ratio map."""
    return alr(P, base=base) if geometry == "alr" else clr(P)


# --------------------------------------------------------------------------- #
# estimation
# --------------------------------------------------------------------------- #
def compsc_weights(P: np.ndarray, T0: int, geometry: str = "clr",
                   base: int = -1) -> np.ndarray:
    """Donor weights from a ``(J, T, K)`` share array; row 0 is the treated unit.

    Algorithm 1, Steps 1-3: transform to log-ratio coordinates, stack the
    coordinate equations across the ``T0`` pre-periods into a single regression
    with ``N = (K-1) * T0`` scalar observations (the multi-outcome stacking of
    Sun, Ben-Michael & Feller 2025), and solve the simplex-constrained QP.

    Parameters
    ----------
    P : np.ndarray
        ``(J, T, K)`` compositions, treated unit first, strictly positive.
    T0 : int
        Number of pre-treatment periods.
    geometry : {"clr", "alr"}
        Log-ratio map defining the objective.
    base : int
        Baseline category index, used only by ``geometry="alr"``.

    Returns
    -------
    np.ndarray
        Weights over the ``J-1`` donors; non-negative, summing to one.
    """
    L = _transform(P[:, :T0, :], geometry, base)
    y = L[0].T.ravel()                                    # category-major stack
    X = np.column_stack([L[j].T.ravel() for j in range(1, P.shape[0])])
    return solve_simplex_qp(X, y)


def compsc_counterfactual(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Counterfactual composition for every period (paper eq. 16).

    The Frechet barycenter of the donor compositions under the Aitchison
    metric, i.e. the closure of their weighted geometric mean. Symmetric in the
    categories, so this does not depend on any baseline choice; it is a valid
    composition (strictly positive, summing to one) by construction.

    Parameters
    ----------
    P : np.ndarray
        ``(J, T, K)`` compositions, treated unit first.
    w : np.ndarray
        Donor weights, shape ``(J-1,)``.

    Returns
    -------
    np.ndarray
        ``(T, K)`` counterfactual compositions.
    """
    logmix = np.tensordot(np.asarray(w, dtype=float), np.log(P[1:]), axes=(0, 0))
    logmix = logmix - logmix.max(axis=-1, keepdims=True)   # overflow guard
    return closure(np.exp(logmix))


def log_ratio_effects(observed: np.ndarray, counterfactual: np.ndarray) -> np.ndarray:
    """All pairwise log-ratio treatment effects (paper eq. 3).

    Parameters
    ----------
    observed, counterfactual : np.ndarray
        ``(T, K)`` compositions.

    Returns
    -------
    np.ndarray
        ``(T, K, K)`` array whose ``[t, k, l]`` entry is
        ``log(pi_k/pi_l) - log(cf_k/cf_l)`` at period ``t``: the treatment-
        induced change in the odds of category ``k`` against category ``l``.
        Antisymmetric in ``(k, l)`` with a zero diagonal.
    """
    d = np.log(observed) - np.log(counterfactual)          # (T, K)
    return d[:, :, None] - d[:, None, :]


def fit_compsc(P: np.ndarray, T0: int, geometry: str = "clr",
               base: int = -1) -> Dict[str, np.ndarray]:
    """Run Algorithm 1 Steps 1-5 for the treated unit in row 0 of ``P``.

    Returns
    -------
    dict
        ``weights``, ``counterfactual``, ``observed``, ``share_effects``
        (eq. 2), ``log_ratio_effects`` (eq. 3), ``aitchison_distance`` (eq. 4),
        and the pre/post Aitchison RMSPEs used by the placebo test.
    """
    w = compsc_weights(P, T0, geometry=geometry, base=base)
    cf = compsc_counterfactual(P, w)
    obs = P[0]
    dA = aitchison_distance(obs, cf)
    return {
        "weights": w,
        "counterfactual": cf,
        "observed": obs,
        "share_effects": obs - cf,
        "log_ratio_effects": log_ratio_effects(obs, cf),
        "aitchison_distance": dA,
        "rmspe_pre": float(np.sqrt(np.mean(dA[:T0] ** 2))),
        "rmspe_post": float(np.sqrt(np.mean(dA[T0:] ** 2))),
        "share_rmspe_pre": float(np.sqrt(np.mean(((obs[:T0] - cf[:T0]) ** 2).sum(axis=1)))),
    }


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def placebo_test(P: np.ndarray, T0: int, geometry: str = "clr", base: int = -1,
                 screen: float = 5.0) -> Dict[str, object]:
    """Abadie placebo permutation test on the Aitchison RMSPE ratio (Step 6).

    Each donor is treated in turn as if it had received the intervention at
    ``T0``, refitting against the remaining units. The test statistic is the
    post-to-pre root mean squared prediction error ratio, measured in the
    Aitchison metric so it is consistent with the estimation objective. Units
    whose pre-treatment fit is worse than ``screen`` times the treated unit's
    are dropped before the comparison.

    Each placebo fit draws on every other unit, the genuinely treated one
    included -- the convention the source paper uses, and what reproduces its
    published p-value. Post-treatment periods of the treated unit therefore
    enter the donor pool for placebo fits, which can flatter a placebo whose
    mix moved the same way; read the placebo gaps, not only the p-value.

    Returns
    -------
    dict
        ``p_value``, ``ratios`` (per unit, treated first), ``pre_rmspe``,
        ``retained`` (boolean mask), ``n_retained`` and ``treated_ratio``.
    """
    J = P.shape[0]
    ratios = np.empty(J)
    pre = np.empty(J)
    for j in range(J):
        order = [j] + [i for i in range(J) if i != j]
        f = fit_compsc(P[order], T0, geometry=geometry, base=base)
        pre[j] = f["rmspe_pre"]
        ratios[j] = (f["rmspe_post"] / f["rmspe_pre"]
                     if f["rmspe_pre"] > 0 else np.inf)
    retained = pre <= screen * pre[0]
    retained[0] = True
    M = int(retained.sum())
    p = float(np.count_nonzero(ratios[retained] >= ratios[0])) / M
    return {"p_value": p, "ratios": ratios, "pre_rmspe": pre,
            "retained": retained, "n_retained": M,
            "treated_ratio": float(ratios[0])}


def geometry_sensitivity(P: np.ndarray, T0: int,
                         fitted: np.ndarray) -> Optional[Dict[str, float]]:
    """How far the fit moves across the arbitrary baseline choices.

    Refits under ``geometry="alr"`` with each category in turn as the baseline
    and reports the largest L1 departure from ``fitted``. Under ``clr`` the
    objective has no baseline, so a large value here says the paper's Algorithm
    1 would have produced a materially different synthetic unit depending on
    which category was nominated as the reference.

    Returns ``None`` when the composition has fewer than three categories, where
    every baseline gives the same fit.
    """
    K = P.shape[2]
    if K < 3:
        return None
    spread = max(
        float(np.abs(compsc_weights(P, T0, geometry="alr", base=b) - fitted).sum())
        for b in range(K)
    )
    return {"max_l1_weight_shift": spread}
