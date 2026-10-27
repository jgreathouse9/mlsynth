"""Per-unit synthetic-control construction for the inclusive SCM method.

A synthetic control is built for one target unit from a donor pool. Two
matching modes:

* **outcome-only** -- simplex least squares on pre-treatment outcomes (the
  classic Abadie objective), solved exactly by the active set behind
  :func:`~mlsynth.utils.solvers.ridge_augment.simplex_qp`. It replaced the FISTA
  primitive ``simplex_lstsq``, which on German reunification ran to its
  2000-iteration cap on 31 of 34 leave-one-out fits and stopped short of the
  optimum on 32, at about 37 ms a fit against the active set's 0.7 ms;
* **covariate** -- predictor matching through the FSCM/MASC bilevel solver,
  with the predictor weights ``V`` jointly optimized with the donor weights
  ``W``. The bilevel *backend* is the caller's choice -- ``"malo"`` (corner
  search) or ``"mscmt"`` (global differential-evolution search) -- so INCSCM
  users pick the same solver the FSCM/MASC estimators expose.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ....exceptions import MlsynthDataError
from ...bilevel import (
    BilevelProblem,
    bias_corrected_gaps,
    solve_bilevel,
)
from ...solvers.ridge_augment import simplex_qp


def _standardize(P_target: np.ndarray, P_donors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score predictors using the donor pool's mean and spread (FSCM style)."""
    center = P_donors.mean(axis=1)
    scale = P_donors.std(axis=1) + 1e-8
    X1 = (P_target - center) / scale
    X0 = (P_donors - center[:, None]) / scale[:, None]
    return X1, X0


def build_unit_sc(
    target_idx: int,
    donor_idx: np.ndarray,
    Y: np.ndarray,
    T0: int,
    *,
    predictors: Optional[np.ndarray] = None,
    predictor_names: Optional[List] = None,
    solver: str = "malo",
    bias_correct: bool = False,
    intercept: bool = False,
):
    """Synthetic control for ``target_idx`` built from donors ``donor_idx``.

    Parameters
    ----------
    bias_correct : bool
        If True (and covariates are supplied), apply the Abadie-L'Hour bias
        correction to the gap, removing the part attributable to residual
        covariate imbalance.
    intercept : bool
        If True (outcome-only mode), fit a **demeaned** simplex synthetic
        control with an unpenalised level shift -- each series is centred by its
        own pre-period mean before the simplex least squares, and the fitted
        intercept ``a = mean(y1_pre) - mean(Y0_pre) . w`` is added back. This is
        the SCM-with-intercept of Doudchenko & Imbens (2016) and the backend of
        Di Stefano & Mellace's inclusive SCM reference (Melnychuk's
        ``scm_weights``); it generally assigns the affected neighbour a larger
        weight than the plain simplex. Ignored in covariate mode.

    Returns
    -------
    w : np.ndarray
        Donor weights aligned with ``donor_idx`` (on the simplex).
    cf : np.ndarray
        Full-length synthetic counterfactual ``(T,)``.
    gap : np.ndarray
        ``Y[target] - cf`` over all periods ``(T,)`` (bias-corrected if
        requested).
    pre_rmspe : float
        Root mean squared pre-treatment fit error (of the reported gap).
    sol : BilevelSolution or None
        The bilevel solution (covariate mode) or ``None`` (outcome-only).

    Raises
    ------
    MlsynthDataError
        If the donor pool is empty, or the target's or a donor's outcomes are
        not finite. The active set returns weights for a design carrying a NaN,
        so the check is made here and not left to the solver.
    """
    donor_idx = np.asarray(donor_idx)
    if donor_idx.size == 0:
        raise MlsynthDataError(
            f"SPILLSYNTH: unit {target_idx} has an empty donor pool; a "
            "synthetic control needs at least one donor."
        )
    y1 = Y[target_idx]
    Y0 = Y[donor_idx].T                       # (T, J)
    if not (np.all(np.isfinite(y1)) and np.all(np.isfinite(Y0))):
        raise MlsynthDataError(
            f"SPILLSYNTH: unit {target_idx} or its donors have non-finite "
            "outcomes; a synthetic control cannot be fit to them."
        )

    if predictors is None:
        sol = None
        if intercept:
            mu1 = y1[:T0].mean()
            mu0 = Y0[:T0].mean(axis=0)         # per-donor pre-period mean
            w = simplex_qp(Y0[:T0] - mu0, y1[:T0] - mu1)
            a = float(mu1 - mu0 @ w)           # unpenalised level shift
            gap = y1 - (a + Y0 @ w)
        else:
            w = simplex_qp(Y0[:T0], y1[:T0])
            gap = y1 - Y0 @ w
    else:
        X1, X0 = _standardize(predictors[target_idx], predictors[donor_idx].T)
        prob = BilevelProblem(
            y1_pre=y1[:T0], Y0_pre=Y0[:T0], X1=X1, X0=X0,
            predictor_names=list(predictor_names or []),
        )
        sol = solve_bilevel(prob, method=solver)
        w = sol.W
        if bias_correct:
            gap = bias_corrected_gaps(w, X1, X0, y1, Y0)
        else:
            gap = y1 - Y0 @ w

    cf = y1 - gap
    pre_rmspe = float(np.sqrt(np.mean(gap[:T0] ** 2)))
    return w, cf, gap, pre_rmspe, sol
