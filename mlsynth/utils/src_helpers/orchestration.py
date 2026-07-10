"""Assemble the SRC point estimate from pivoted inputs (``run_src``)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...exceptions import MlsynthEstimationError
from .estimation import counterfactual, src_weights
from .screening import screen_donors, screen_inputs
from .structures import SRCFit, SRCInputs
from .vsearch import optimize_v


def _build_matching_matrix(inputs: SRCInputs) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return ``(X, y, n_outcome_rows)`` for the SRC weight computation.

    The outcome matching rows are the treated/donor outcomes over the fit window
    (``inputs.fit_idx``; all pre-periods by default). Pure Algorithm 1 (no
    covariates) returns just those rows. With covariates, each covariate's
    (windowed) mean is scaled to the mean standard deviation of the outcome rows
    -- the reference's equal-variance scaling -- and stacked on top, so
    predictors and outcomes enter on a common scale. Outcome rows sit last so the
    ``V`` search can address them as the held-out target.
    """
    idx = inputs.fit_idx if inputs.fit_idx is not None else np.arange(inputs.T0)
    y_out = inputs.Y_treated[idx]                    # (n_out,)
    X_out = inputs.Y_donors[idx, :]                  # (n_out, J)
    if not inputs.has_covariates:
        return X_out, y_out, X_out.shape[0]

    out_sd = X_out.std(axis=1, ddof=0)
    scale = float(np.mean(out_sd[out_sd > 0])) if np.any(out_sd > 0) else 1.0
    cov_all = np.column_stack([inputs.cov_treated, inputs.cov_donors])  # (P, J+1)
    cov_sd = cov_all.std(axis=1, ddof=0)
    cov_sd = np.where(cov_sd > 0, cov_sd, 1.0)
    y_cov = inputs.cov_treated / cov_sd * scale       # (P,)
    X_cov = inputs.cov_donors / cov_sd[:, None] * scale
    X = np.vstack([X_cov, X_out])                     # (P + n_out, J)
    y = np.concatenate([y_cov, y_out])
    return X, y, X_out.shape[0]


def run_src(
    inputs: SRCInputs,
    *,
    ridge: float = 1e-3,
    screen: str = "none",
    n_screen: Optional[int] = None,
    v_search: str = "none",
    v_seed: int = 0,
    v_maxiter: int = 60,
    v_popsize: int = 12,
) -> SRCFit:
    """Fit SRC (deterministic Algorithm 1 / 3) and return the point estimate.

    With ``screen="sirs"`` the donor pool is first reduced by SIRS screening
    (Algorithm 2) to ``n_screen`` donors (default the paper's count). With
    ``v_search="de"`` the predictor weights ``V`` are chosen by a seeded global
    search (the paper's Algorithm 3); otherwise ``V = I``.
    """
    n_screened_out = 0
    if screen == "sirs":
        keep = screen_donors(
            inputs.Y_donors[:inputs.T0], inputs.Y_treated[:inputs.T0],
            n_screen=n_screen,
        )
        if keep.size < inputs.J:
            n_screened_out = int(inputs.J - keep.size)
            inputs = screen_inputs(inputs, keep)
    try:
        X, y, n_out = _build_matching_matrix(inputs)
        V: Optional[np.ndarray] = None
        if v_search == "de":
            V = optimize_v(X, y, n_out, ridge=ridge, seed=v_seed,
                           maxiter=v_maxiter, popsize=v_popsize)
        weights = src_weights(X, y, ridge=ridge, V=V)
        cf = counterfactual(inputs.Y_donors, weights)
    except Exception as exc:  # pragma: no cover - defensive; src_weights is total
        raise MlsynthEstimationError(f"SRC estimation failed: {exc}") from exc

    obs = inputs.Y_treated
    gap = obs - cf
    T0 = inputs.T0
    pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2)))
    att = float(np.mean(gap[T0:])) if T0 < inputs.T else float("nan")
    donor_weights = {
        lbl: float(c) for lbl, c in zip(inputs.donor_labels, weights.combined)
    }
    return SRCFit(
        att=att,
        weights=weights.combined,
        theta=weights.theta,
        w=weights.w,
        bias=weights.bias,
        sigma2=weights.sigma2,
        counterfactual=cf,
        gap=gap,
        pre_rmse=pre_rmse,
        n_matching_rows=int(X.shape[0]),
        n_covariates=len(inputs.covariate_names),
        v_search=v_search,
        v=V,
        donor_weights=donor_weights,
        n_screened_out=n_screened_out,
    )
