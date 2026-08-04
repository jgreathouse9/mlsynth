"""Xu (2017) Algorithm 2: the parametric bootstrap.

The estimator's uncertainty comes from the idiosyncratic errors, which are
assumed independent of treatment assignment, covariates, factors and loadings.
So the bootstrap holds the estimated conditional mean fixed and resamples errors
around it.

Treated and control cells draw from different pools, and that asymmetry is the
point. The factor space and the coefficient vector are estimated from the
control units, so the fitted surface tracks a control unit better than it tracks
a treated counterfactual, and the treated prediction error is the larger of the
two. Loop 1 measures it: pick a control unit, treat it as if it had adopted,
predict it from a resample of the remaining controls, and keep the difference.
Repeating that builds an empirical distribution of prediction errors for a unit
the model did not see.

Loop 2 then rebuilds panels from the fitted values plus resampled errors --
treated columns drawing from Loop 1's pool, control columns from the in-sample
residuals -- refits, and collects the ATT. The simulated treated columns carry
no treatment effect, so the spread of those refits is the sampling distribution
of the estimator under the estimated data-generating process; the point estimate
is added back when the intervals are formed.

Draws whose refit fails are discarded and counted, and the routine raises if too
few survive to estimate a variance from.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...exceptions import MlsynthEstimationError
from .pipeline import fit_control_model, gsc_fit
from .structures import (
    GSYNTHFit,
    GSYNTHInference,
    GSYNTHInputs,
    event_study_from_effect,
)

_MIN_DRAWS = 5


def _rebuild(inputs: GSYNTHInputs, Y: np.ndarray) -> GSYNTHInputs:
    """The same panel with a new outcome matrix."""
    return GSYNTHInputs(
        Y=Y, D=inputs.D, X=inputs.X,
        treated_index=inputs.treated_index,
        control_index=inputs.control_index,
        adoption_index=inputs.adoption_index,
        unit_labels=inputs.unit_labels,
        time_labels=inputs.time_labels,
        covariate_names=inputs.covariate_names,
    )


def _control_fitted_and_residuals(inputs: GSYNTHInputs, fit) -> tuple:
    """The control units' fitted values and in-sample residuals."""
    co = inputs.control_index
    fitted = (fit.mu + fit.alpha[None, :] + fit.xi[:, None]
              + fit.factor @ fit.lam.T)
    if inputs.p:
        fitted = fitted + np.einsum("tnk,k->tn", inputs.X[:, co, :], fit.beta)
    return fitted, inputs.Y[:, co] - fitted


def _prediction_error_pool(
    inputs: GSYNTHInputs, r: int, n_draws: int, rng: np.random.Generator,
    *, two_way: bool, tol: float, max_iter: int,
) -> np.ndarray:
    """Loop 1: prediction errors for a unit the control model did not fit.

    Each draw promotes one control unit to stand in for every treated unit --
    inheriting their adoption dates, so the held-out periods match the ones the
    real fit predicts -- and resamples the remaining controls with replacement
    as its donor pool.

    Returns
    -------
    np.ndarray
        Shape ``(T, N_tr, n_kept)``. Columns are aligned with
        ``inputs.treated_index``, so a treated unit draws errors generated at
        its own adoption date.
    """
    co = inputs.control_index
    n_tr = inputs.n_treated
    pool = []
    for _ in range(n_draws):
        fake = int(rng.choice(co))
        rest = co[co != fake]
        # Resampled to the full control count, not to the size of what is left
        # after removing the fake unit, so the pseudo-fit sees a donor pool the
        # same width as the real one.
        donors = rng.choice(rest, size=co.size, replace=True)
        cols = np.concatenate([np.full(n_tr, fake, dtype=int), donors])

        pseudo = GSYNTHInputs(
            Y=inputs.Y[:, cols], D=inputs.D[:, np.concatenate(
                [inputs.treated_index, donors])],
            X=inputs.X[:, cols, :],
            treated_index=np.arange(n_tr),
            control_index=np.arange(n_tr, cols.size),
            adoption_index=inputs.adoption_index,
            unit_labels=inputs.unit_labels[cols],
            time_labels=inputs.time_labels,
            covariate_names=inputs.covariate_names,
        )
        try:
            out = gsc_fit(pseudo, r, two_way=two_way, tol=tol, max_iter=max_iter)
        except (MlsynthEstimationError, np.linalg.LinAlgError):
            continue
        pool.append(out.effect)
    if not pool:
        raise MlsynthEstimationError(
            "Algorithm 2 could not simulate any prediction error: every "
            "leave-one-control-out refit failed. Lower r or turn inference off."
        )
    return np.stack(pool, axis=2)


def parametric_bootstrap(
    inputs: GSYNTHInputs,
    fit: GSYNTHFit,
    r: int,
    *,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
    two_way: bool = True,
    tol: float = 1e-5,
    max_iter: int = 500,
) -> GSYNTHInference:
    """Run Algorithm 2 and return the ATT and horizon-path uncertainty.

    Parameters
    ----------
    inputs : GSYNTHInputs
        Preprocessed panel.
    fit : GSYNTHFit
        The fit on the observed data, whose control model is reused.
    r : int
        Factor count, held fixed across the bootstrap so the intervals describe
        the estimator at this rank.
    n_bootstrap : int
        Draws in each loop.
    alpha : float
        Two-sided significance level.
    seed : int
        Seed for the resampling.
    two_way, tol, max_iter
        Passed through to the refits.

    Returns
    -------
    GSYNTHInference

    Raises
    ------
    MlsynthEstimationError
        If too few draws survive to estimate a variance from.
    """
    rng = np.random.default_rng(seed)
    control = fit.control_fit
    fitted_co, resid_co = _control_fitted_and_residuals(inputs, control)
    fitted_tr = fit.counterfactual

    pool = _prediction_error_pool(
        inputs, r, n_bootstrap, rng,
        two_way=two_way, tol=tol, max_iter=max_iter)
    n_pool = pool.shape[2]

    es = event_study_from_effect(fit.effect, inputs.adoption_index)
    horizons = es.horizons
    index = {int(h): i for i, h in enumerate(horizons)}

    atts = np.empty(n_bootstrap)
    paths = np.full((n_bootstrap, horizons.size), np.nan)
    kept = 0
    for _ in range(n_bootstrap):
        Y = np.empty_like(inputs.Y)
        draw = rng.integers(0, n_pool, size=inputs.n_treated)
        Y[:, inputs.treated_index] = fitted_tr + pool[:, np.arange(
            inputs.n_treated), draw]
        pick = rng.integers(0, inputs.n_control, size=inputs.n_control)
        Y[:, inputs.control_index] = fitted_co + resid_co[:, pick]

        boot = _rebuild(inputs, Y)
        try:
            control_boot = fit_control_model(
                boot, r, two_way=two_way, tol=tol, max_iter=max_iter)
            out = gsc_fit(boot, r, two_way=two_way, tol=tol, max_iter=max_iter,
                          control_fit=control_boot)
        except (MlsynthEstimationError, np.linalg.LinAlgError):
            continue
        atts[kept] = out.att
        boot_es = event_study_from_effect(out.effect, inputs.adoption_index)
        for h, v in zip(boot_es.horizons, boot_es.tau):
            slot = index.get(int(h))
            if slot is not None:
                paths[kept, slot] = v
        kept += 1

    n_failed = n_bootstrap - kept
    if kept < _MIN_DRAWS:
        raise MlsynthEstimationError(
            f"Algorithm 2 kept only {kept} of {n_bootstrap} bootstrap draws, "
            f"too few to estimate a variance from. Lower r or turn inference "
            f"off."
        )
    atts = atts[:kept]
    paths = paths[:kept]

    # The simulated treated columns carry no effect, so the draws are centered
    # near zero; the point estimate is added back to form the intervals.
    att = float(fit.att)
    se = float(atts.std(ddof=1))
    lo, hi = np.percentile(atts, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    ci_lower, ci_upper = att + float(lo), att + float(hi)
    p_value = float(np.mean(np.abs(atts) >= abs(att)))

    with np.errstate(invalid="ignore"):
        tau_se = np.nanstd(paths, axis=0, ddof=1)
        q = np.nanpercentile(paths, [100 * alpha / 2, 100 * (1 - alpha / 2)],
                             axis=0)
    tau = np.asarray(es.tau, dtype=float)

    return GSYNTHInference(
        n_bootstrap=int(n_bootstrap),
        alpha=float(alpha),
        att=att,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        tau=tau,
        tau_se=np.asarray(tau_se, dtype=float),
        tau_lower=tau + np.asarray(q[0], dtype=float),
        tau_upper=tau + np.asarray(q[1], dtype=float),
        horizons=np.asarray(horizons, dtype=int),
        n_failed=int(n_failed),
    )
