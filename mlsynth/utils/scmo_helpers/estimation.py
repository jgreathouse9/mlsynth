"""SCMO estimation: weighting schemes, demeaning, counterfactual, effects.

All functions are pure NumPy and operate on the arrays in
:class:`SCMOInputs`, so the same core drives both the real fit and the
permutation placebos in :mod:`mlsynth.utils.scmo_helpers.inference`.

Schemes
-------
* ``concatenated`` (Tian-Lee-Panchenko): match the full standardized matching
  matrix ``Z``.
* ``averaged`` (Sun-Ben-Michael-Feller): match the average of the standardized
  outcomes within each period.
* ``separate`` (conventional): match the primary outcome's standardized
  pre-treatment trajectory alone.
* ``MA``: convex model-average of the concatenated and averaged
  counterfactuals (chosen by pre-treatment fit).

Demeaning (``demean=True``) is the Doudchenko-Imbens / Sun-Ben-Michael-Feller
intercept shift: the counterfactual is the treated unit's pre-period mean
plus the weighted donor *deviations*, allowing levels to differ.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import cvxpy as cp

from .solvers import simplex_weights
from .structures import (
    AVERAGED,
    CONCATENATED,
    MA,
    SEPARATE,
    SCMOInputs,
    SCMOMethodFit,
)


def _standardize_cols(M: np.ndarray) -> np.ndarray:
    """Divide each column by its cross-unit SD (no centering)."""
    sd = M.std(axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    return M / sd


def _averaged_Z(Z: np.ndarray, col_period: Optional[np.ndarray]) -> np.ndarray:
    """Average the standardized columns within each period across outcomes."""
    if col_period is None:
        return Z.mean(axis=1, keepdims=True)
    periods = np.unique(col_period)
    return np.column_stack([Z[:, col_period == p].mean(axis=1) for p in periods])


def _effects(y: np.ndarray, cf: np.ndarray, T0: int) -> Tuple[float, float, np.ndarray]:
    gap = y - cf
    att = float(np.mean(gap[T0:])) if gap.shape[0] > T0 else float("nan")
    pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2)))
    return att, pre_rmse, gap


def _matching_vectors(
    Z: np.ndarray, Y: np.ndarray, T0: int, scheme: str,
    col_period: Optional[np.ndarray], demean: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the (N, .) matching matrix the scheme solves on (rows = units)."""
    if scheme == CONCATENATED:
        M = Z
    elif scheme == AVERAGED:
        M = _averaged_Z(Z, col_period)
    elif scheme == SEPARATE:
        pre = Y[:, :T0]
        if demean:
            pre = pre - pre.mean(axis=1, keepdims=True)
        M = _standardize_cols(pre)
    else:
        raise ValueError(f"Unknown scheme: {scheme!r}")
    return M


def _counterfactual(w: np.ndarray, Y_donors: np.ndarray, y: np.ndarray, T0: int, demean: bool) -> np.ndarray:
    if demean:
        donor_pre = Y_donors[:, :T0].mean(axis=1)            # (J,)
        return y[:T0].mean() + w @ (Y_donors - donor_pre[:, None])
    return w @ Y_donors


def _scheme_weights(M, treated_idx, donor_idx, augment, ridge_lambda):
    """Simplex SC weights on the matching matrix, optionally ridge-augmented."""
    if augment == "ridge":
        from ..bilevel.ridge_augment import ridge_augment_weights
        ra = ridge_augment_weights(
            M[treated_idx], M[donor_idx].T, lambda_=ridge_lambda)
        return np.asarray(ra.W, dtype=float)
    return simplex_weights(M[treated_idx], M[donor_idx])


def _fit_core(
    Z: np.ndarray, Y: np.ndarray, treated_idx: int, donor_idx: np.ndarray,
    T0: int, scheme: str, col_period: Optional[np.ndarray], demean: bool,
    augment: Optional[str] = None, ridge_lambda: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Solve weights + counterfactual for any (treated, donors) selection."""
    M = _matching_vectors(Z, Y, T0, scheme, col_period, demean)
    w = _scheme_weights(M, treated_idx, donor_idx, augment, ridge_lambda)
    cf = _counterfactual(w, Y[donor_idx], Y[treated_idx], T0, demean)
    att, pre_rmse, gap = _effects(Y[treated_idx], cf, T0)
    return w, cf, att, pre_rmse, gap


def fit_scheme(inputs: SCMOInputs, scheme: str, demean: bool = False,
               augment: Optional[str] = None,
               ridge_lambda: Optional[float] = None) -> SCMOMethodFit:
    """Fit one weighting scheme on the real treated unit."""
    w, cf, att, pre_rmse, gap = _fit_core(
        inputs.Z, inputs.Y, inputs.treated_idx, inputs.donor_idx,
        inputs.T0, scheme, inputs.col_period, demean, augment, ridge_lambda,
    )
    donor_labels = inputs.donor_labels
    donor_weights = {donor_labels[i]: float(round(w[i], 4)) for i in range(len(w))}
    return SCMOMethodFit(
        name=scheme, weights=w, counterfactual=cf, gap=gap, att=att,
        pre_rmse=pre_rmse, donor_weights=donor_weights,
        metadata={"demean": demean, "augment": augment},
    )


def model_average(inputs: SCMOInputs, fits: List[SCMOMethodFit]) -> SCMOMethodFit:
    """Convex model-average of several fits by pre-treatment fit (the old BOTH)."""
    T0 = inputs.T0
    y = inputs.y_treated
    cfs = np.column_stack([f.counterfactual for f in fits])      # (T, M)
    M = cfs.shape[1]
    lam = cp.Variable(M)
    cp.Problem(cp.Minimize(cp.sum_squares(y[:T0] - cfs[:T0] @ lam)),
               [lam >= 0, cp.sum(lam) == 1]).solve(solver=cp.OSQP)
    lam = np.clip(np.asarray(lam.value).ravel(), 0.0, None)
    cf = cfs @ lam
    att, pre_rmse, gap = _effects(y, cf, T0)
    w = sum(lam[m] * fits[m].weights for m in range(M))
    donor_labels = inputs.donor_labels
    donor_weights = {donor_labels[i]: float(round(w[i], 4)) for i in range(len(w))}
    return SCMOMethodFit(
        name=MA, weights=w, counterfactual=cf, gap=gap, att=att,
        pre_rmse=pre_rmse, donor_weights=donor_weights,
        metadata={"lambdas": {fits[m].name: float(round(lam[m], 4)) for m in range(M)}},
    )
