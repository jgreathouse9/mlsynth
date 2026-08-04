"""Xu (2017) Steps 2 and 3, and Algorithm 1.

Step 1 (:mod:`.ife`) fits the interactive fixed effects model to the
never-treated units, returning ``\\hat\\beta``, ``\\hat F`` and the additive
effects. What remains is to place the treated units in that estimated factor
space and difference:

Step 2 recovers each treated unit's loadings from its own pre-treatment
periods, holding the control-fit quantities fixed,

.. math::

   \\hat\\lambda_i = (F^{0\\prime} F^0)^{-1} F^{0\\prime}
                     (Y^0_i - X^0_i \\hat\\beta), \\qquad i \\in \\mathcal{T}.

Step 3 imputes and differences,

.. math::

   \\hat Y_{it}(0) = x_{it}'\\hat\\beta + \\hat\\lambda_i'\\hat f_t,
   \\qquad
   \\widehat{ATT}_t = \\frac{1}{N_{tr}}
       \\sum_{i \\in \\mathcal{T}} [Y_{it}(1) - \\hat Y_{it}(0)].

Under additive unit effects a column of ones is appended to ``\\hat F`` before
Step 2, so a treated unit's own level is estimated jointly with its loadings off
its pre-periods -- the level is not identified any other way, since the control
units' unit effects say nothing about a unit outside that group.

Algorithm 1 chooses the factor count. For each candidate rank it refits Step 1
once, then walks the treated units' pre-treatment periods, holding one period
back, refitting the loadings on the rest, and predicting the held-out cell. The
rank minimizing the mean squared prediction error wins. The procedure draws no
random numbers, so the selected rank is a property of the panel.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ...exceptions import MlsynthEstimationError
from .ife import ControlFit, interactive_fixed_effects
from .structures import GSYNTHCrossValidation, GSYNTHFit, GSYNTHInputs


def _regressors(fit: ControlFit, T: int, two_way: bool) -> np.ndarray:
    """The matrix Step 2 projects the treated pre-periods onto."""
    if two_way:
        return np.hstack([fit.factor, np.ones((T, 1))])
    return fit.factor


def _treated_residual(inputs: GSYNTHInputs, fit: ControlFit) -> np.ndarray:
    """Treated outcomes net of covariates and the control fit's additive terms.

    The unit effect is deliberately not among them: it belongs to the treated
    unit and is recovered in Step 2.
    """
    tr = inputs.treated_index
    U = inputs.Y[:, tr]
    if inputs.p:
        U = U - np.einsum("tnk,k->tn", inputs.X[:, tr, :], fit.beta)
    return U - fit.mu - fit.xi[:, None]


def _rank_ceiling(inputs: GSYNTHInputs, two_way: bool) -> int:
    """Largest rank the shortest treated pre-period history can identify.

    A unit with ``T0`` pre-periods supports ``T0`` regressors, one of which is
    the intercept under two-way effects. One fewer than that keeps the loading
    regression overdetermined, which is what the cross-validation needs so a
    period can be held back.
    """
    return max(int(inputs.min_pre_periods) - (2 if two_way else 1), 0)


def fit_control_model(
    inputs: GSYNTHInputs, r: int, *, two_way: bool = True,
    tol: float = 1e-5, max_iter: int = 500,
) -> ControlFit:
    """Run Step 1 on the never-treated units."""
    co = inputs.control_index
    return interactive_fixed_effects(
        inputs.Y[:, co], inputs.X[:, co, :], r,
        two_way=two_way, tol=tol, max_iter=max_iter,
    )


def gsc_fit(
    inputs: GSYNTHInputs, r: int, *, two_way: bool = True,
    tol: float = 1e-5, max_iter: int = 500,
    control_fit: Optional[ControlFit] = None,
) -> GSYNTHFit:
    """Run Steps 1-3 and return the effect matrix and the ATT.

    Parameters
    ----------
    inputs : GSYNTHInputs
        Preprocessed panel.
    r : int
        Number of factors.
    two_way : bool
        Include additive unit and period effects.
    tol, max_iter : float, int
        Passed to the Step 1 alternating least squares.
    control_fit : ControlFit, optional
        A Step 1 fit to reuse. Supplying it skips refitting the control model,
        which the cross-validation and the bootstrap both rely on.

    Returns
    -------
    GSYNTHFit

    Raises
    ------
    MlsynthEstimationError
        If a treated unit has fewer pre-treatment periods than the loading
        regression has regressors, or if that regression is singular.
    """
    T = inputs.T
    k = int(r) + (1 if two_way else 0)
    shortest = int(inputs.min_pre_periods)
    if k > shortest:
        raise MlsynthEstimationError(
            f"r = {r} needs {k} regressors per treated unit but the shortest "
            f"pre-treatment history is {shortest} period(s); lower r to at most "
            f"{max(shortest - (1 if two_way else 0), 0)}."
        )

    fit = control_fit if control_fit is not None else fit_control_model(
        inputs, r, two_way=two_way, tol=tol, max_iter=max_iter)
    U = _treated_residual(inputs, fit)
    F = _regressors(fit, T, two_way)

    loadings = np.zeros((inputs.n_treated, F.shape[1]))
    for j, t0 in enumerate(inputs.adoption_index):
        Fp, up = F[:t0], U[:t0, j]
        try:
            loadings[j] = np.linalg.solve(Fp.T @ Fp, Fp.T @ up)
        except np.linalg.LinAlgError as exc:
            raise MlsynthEstimationError(
                f"The loading regression for treated unit "
                f"{inputs.unit_labels[inputs.treated_index[j]]!r} is singular "
                f"at r = {r}; the estimated factors are collinear over its "
                f"pre-treatment periods. Lower r."
            ) from exc

    effect = U - F @ loadings.T
    counterfactual = inputs.Y[:, inputs.treated_index] - effect
    post = inputs.D[:, inputs.treated_index] > 0
    unit_effects = loadings[:, -1] if two_way else np.zeros(inputs.n_treated)

    return GSYNTHFit(
        effect=effect,
        counterfactual=counterfactual,
        att=float(effect[post].mean()),
        n_post_cells=int(post.sum()),
        loadings=loadings,
        unit_effects=unit_effects,
        factors_used=F,
        control_fit=fit,
    )


def cross_validate_rank(
    inputs: GSYNTHInputs, *, r_max: int = 5, two_way: bool = True,
    tol: float = 1e-5, max_iter: int = 500,
) -> GSYNTHCrossValidation:
    """Algorithm 1: leave-one-period-out selection of the factor count.

    Parameters
    ----------
    inputs : GSYNTHInputs
        Preprocessed panel.
    r_max : int
        Largest rank considered. Lowered to what the shortest treated
        pre-period history can identify.
    two_way : bool
        Include additive unit and period effects.
    tol, max_iter : float, int
        Passed to the Step 1 alternating least squares.

    Returns
    -------
    GSYNTHCrossValidation

    Raises
    ------
    MlsynthEstimationError
        If no rank could be scored, which happens when no treated unit has a
        pre-period history long enough to hold a period back from.
    """
    ceiling = min(int(r_max), _rank_ceiling(inputs, two_way))
    T = inputs.T

    mspe: Dict[int, float] = {}
    scored: Dict[int, int] = {}
    for r in range(0, ceiling + 1):
        fit = fit_control_model(inputs, r, two_way=two_way, tol=tol,
                                max_iter=max_iter)
        U = _treated_residual(inputs, fit)
        F = _regressors(fit, T, two_way)
        k = F.shape[1]

        sse, n = 0.0, 0
        for j, t0 in enumerate(inputs.adoption_index):
            pre = np.arange(int(t0))
            if pre.size <= k:
                continue
            for s in pre:
                keep = pre[pre != s]
                Fk = F[keep]
                try:
                    lam = np.linalg.solve(Fk.T @ Fk, Fk.T @ U[keep, j])
                except np.linalg.LinAlgError:  # pragma: no cover - collinear fold
                    continue
                sse += float((U[s, j] - F[s] @ lam) ** 2)
                n += 1
        if n:
            mspe[r] = sse / n
            scored[r] = n

    if not mspe:
        raise MlsynthEstimationError(
            "Cross-validation could not score any rank: no treated unit has "
            "enough pre-treatment periods to hold one back. Fix the rank with "
            "the `r` option instead."
        )
    best = min(mspe, key=lambda r: mspe[r])
    return GSYNTHCrossValidation(r_selected=int(best), mspe=mspe, n_scored=scored)
