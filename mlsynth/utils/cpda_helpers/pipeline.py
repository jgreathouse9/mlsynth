"""CPDA end to end: Hsiao and Zhou (2019) Equations 11 to 15.

The four steps, in the paper's order:

1. estimate the covariate slope ``beta`` on the pre-period (:mod:`.beta`);
2. residualise, ``v_t = y_t - X_t beta``, for the treated unit and every donor;
3. choose ``mu`` and ``w`` to minimise ``sum_t (v_1t - mu - w' v*_t)^2`` over
   the pre-period, on a subset ``v*`` a selector picks (:mod:`.selection`);
4. predict ``y^0_1t = x'_1t beta + w' v*_t + mu``, and difference.

What the method buys over a donor-only regression is step 2. Donors have to
explain the factor part of the treated unit alone, so a covariate that moves
the treated unit and no donor stops being unexplained error.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ..pda_helpers.inference import hac_lrv, normal_test
from .beta import estimate_beta
from .selection import SELECTORS, select_donors
from .structures import CPDAFit, CPDAInputs


def residualise(inputs: CPDAInputs, beta: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Equation 12: strip the covariate part from the treated unit and donors."""
    v1 = inputs.y - inputs.x @ beta
    vco = inputs.Yco - np.einsum("tnk,k->tn", inputs.Xco, beta)
    return v1, vco


def _refit(target_pre: np.ndarray, design_pre: np.ndarray, design_all: np.ndarray,
           keep: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Equation 13: least squares for ``mu`` and ``w`` on the kept donors.

    ``lstsq`` is the minimum-norm solution, so a collinear or rank-deficient
    pool returns a fit instead of raising. The fitted path is unique even where
    the coefficients are not.
    """
    T0 = len(target_pre)
    if keep.size == 0:
        mu = float(np.mean(target_pre))
        return mu, np.zeros(0), np.full(design_all.shape[0], mu)
    A = np.column_stack([np.ones(T0), design_pre[:, keep]])
    coef = np.linalg.lstsq(A, target_pre, rcond=None)[0]
    fitted = np.column_stack([np.ones(design_all.shape[0]),
                              design_all[:, keep]]) @ coef
    return float(coef[0]), coef[1:], fitted


def _fit_once(inputs: CPDAInputs, beta: np.ndarray, v1: np.ndarray,
              vco: np.ndarray, *, selector: str, standardize: bool, seed: int
              ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    T0 = inputs.T0
    keep = select_donors(vco[:T0], v1[:T0], selector=selector,
                         standardize=standardize, seed=seed)
    mu, w, fitted = _refit(v1[:T0], vco[:T0], vco, keep)
    counterfactual = inputs.x @ beta + fitted          # Equation 15
    return keep, mu, w, counterfactual


def run_cpda(inputs: CPDAInputs, *, beta_method: str = "cce", r: int = 2,
             selector: str = "lasso_cv", standardize_selection: bool = False,
             alpha: float = 0.05, seed: int = 0, lrvar_lag: int | None = None,
             sensitivity: bool = False) -> CPDAFit:
    """Run the construction and package the fit.

    Parameters
    ----------
    inputs : CPDAInputs
        Preprocessed panel.
    beta_method : str
        ``cce`` or ``bai``; see :mod:`.beta`.
    r : int
        Factor count, read only by ``bai``.
    selector : str
        Which rule picks the donor subset; see :mod:`.selection`.
    standardize_selection : bool
        Rescale the selection design before the penalty.
    alpha : float
        Two-sided level for the interval.
    seed : int
        Seed for any cross-validated penalty.
    lrvar_lag : int, optional
        Bartlett truncation lag for the ATT's long-run variance. ``None``
        takes the usual ``floor(4 (T2/100)^(2/9))`` rule.
    sensitivity : bool
        Also fit under every other selector and record the spread. The
        selector is not pinned by the paper and moves the estimate, so a
        caller reporting a point estimate alone is claiming an identification
        the method does not have.

    Returns
    -------
    CPDAFit
    """
    T0, T = inputs.T0, inputs.T
    beta = estimate_beta(inputs.Yco, inputs.Xco, T0, method=beta_method, r=r)
    v1, vco = residualise(inputs, beta)

    keep, mu, w, counterfactual = _fit_once(
        inputs, beta, v1, vco, selector=selector,
        standardize=standardize_selection, seed=seed)
    gap = inputs.y - counterfactual

    post = gap[T0:]
    if post.size:
        att = float(np.mean(post))
        lrv = hac_lrv(post, lag=lrvar_lag)       # de-means internally
        att_se = float(np.sqrt(max(lrv, 0.0) / post.size))
    else:                                    # pragma: no cover - setup forbids
        att, att_se = float("nan"), float("nan")
    p_value, ci = normal_test(att, att_se, alpha=alpha)

    sweep: Dict[str, Dict[str, float]] | None = None
    if sensitivity:
        sweep = {}
        for name in SELECTORS:
            k2, _, _, cf2 = _fit_once(inputs, beta, v1, vco, selector=name,
                                      standardize=standardize_selection, seed=seed)
            g2 = inputs.y - cf2
            sweep[name] = {
                "att": float(np.mean(g2[T0:])) if T > T0 else float("nan"),
                "n_selected": int(k2.size)}

    # ``.tolist()`` so a numeric label comes back as a Python int or float
    # instead of a numpy scalar, which prints as ``np.int64(1)``.
    labels: List[Any] = np.asarray(inputs.unit_index.labels)[keep].tolist()
    return CPDAFit(
        beta=np.asarray(beta, dtype=float),
        beta_method=beta_method,
        selector=selector,
        selected_donors=labels,
        standardize_selection=bool(standardize_selection),
        intercept=mu,
        weights={lab: float(val) for lab, val in zip(labels, w)},
        counterfactual=counterfactual,
        gap=gap,
        att=att,
        att_se=att_se,
        ci=ci,
        p_value=p_value,
        sensitivity=sweep,
        metadata={"r": r, "alpha": alpha, "seed": seed},
    )
