"""MOSC's estimation pipeline: factor model, outcome regression, diagnostics.

The stages are the paper's Algorithm 1. Fit a probabilistic factor model to the
pre-intervention panel; read off each unit's latent loadings; regress every
unit's post-period outcome on its loadings and a treatment dummy; predict the
treated unit's row with the dummy switched off. The counterfactual is that
prediction, and the effect is the observed path minus it.

Two things here are the estimator's own and not the paper's, both established by
``benchmarks/reference/mosc_spike/``. The effect takes equation 43's sign, which
the authors' code inverts. And the design carries no lagged outcome: the authors'
code adds the last pre-intervention outcome to every regression that produced
their published figure, gives the baseline no equivalent term, and does not
mention it in equations 40-41.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from ...exceptions import MlsynthEstimationError
from .factor import FACTOR_MODELS, FactorDraws, heldout_log_density
from .structures import MOSCDiagnostics, MOSCInputs, MOSCPosterior


def fit_factor_model(
    pre_panel: np.ndarray, factor_model: str, n_factors: int,
    n_samples: int, n_warmup: int, mask: np.ndarray, seed: int,
) -> FactorDraws:
    """Draw the estimated confounding structure from the pre-intervention panel."""
    engine = FACTOR_MODELS[factor_model]
    return engine(
        pre_panel, n_factors, mask=mask, n_samples=n_samples,
        n_warmup=n_warmup, seed=seed,
    )


def counterfactual_draws(
    loadings: np.ndarray, panel: np.ndarray, pre_periods: int,
    ridge_alphas: Tuple[float, ...],
) -> np.ndarray:
    """Post-period counterfactual for the treated unit, one draw at a time.

    Each draw gets its own outcome model. Averaging the loadings first would be
    ill-defined -- a factor model is identified only up to relabelling, so the
    mean of the draws need not be a configuration any draw took.
    """
    return np.array([
        _counterfactual_from_draw(loadings[draw], panel, pre_periods, ridge_alphas)
        for draw in range(loadings.shape[0])
    ])


def _counterfactual_from_draw(
    loadings: np.ndarray, panel: np.ndarray, pre_periods: int,
    ridge_alphas: Tuple[float, ...],
) -> np.ndarray:
    design, response = _regression_arrays(loadings, panel, pre_periods, treated_on=True)
    search = GridSearchCV(Ridge(), {"alpha": list(ridge_alphas)}, scoring="r2", cv=_folds(design))
    search.fit(design, response)
    counterfactual_design, _ = _regression_arrays(
        loadings, panel, pre_periods, treated_on=False
    )
    return np.asarray(search.best_estimator_.predict(counterfactual_design)[0], dtype=float)


def _regression_arrays(
    loadings: np.ndarray, panel: np.ndarray, pre_periods: int, treated_on: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Design and response for the outcome model, one row per unit."""
    n_units = loadings.shape[1]
    treatment = np.zeros((n_units, 1))
    if treated_on:
        treatment[0] = 1.0                       # the treated unit is column 0
    design = np.concatenate([loadings.T, treatment], axis=1)
    return design, panel[pre_periods:].T


def _folds(design: np.ndarray) -> int:
    """Cross-validation folds that a small unit count can actually supply."""
    return int(max(2, min(5, design.shape[0] // 2)))


def diagnose(
    pre_panel: np.ndarray, mask: np.ndarray, draws: FactorDraws, outcome_scale: str,
) -> MOSCDiagnostics:
    """Score the fit, and measure whether the model's own assumptions hold.

    Both quantities are evaluated on held-out cells. A rank-``K`` fit to a panel
    with few units interpolates, so an in-sample dispersion would read near 1
    whatever the data did.
    """
    fitted = np.mean([draws.mean(draw) for draw in range(draws.n_draws)], axis=0)
    heldout = ~np.asarray(mask, dtype=bool)

    scale = np.clip(np.abs(fitted), 1e-8, None)
    scorable = heldout & (scale >= 1.0)
    # The ratio is unbounded where the fit sends a rate toward zero, so a median
    # over cells the model gives real mass to is what the statistic can support.
    ratio = (pre_panel[scorable] - fitted[scorable]) ** 2 / scale[scorable]
    dispersion = float(np.median(ratio)) if ratio.size else float("nan")

    pearson = (pre_panel - fitted) / np.sqrt(scale)
    correlations = [
        float(np.corrcoef(pearson[:-1, unit], pearson[1:, unit])[0, 1])
        for unit in range(pearson.shape[1])
        if pearson[:, unit].std() > 1e-12
    ]
    autocorrelation = float(np.mean(correlations)) if correlations else 0.0

    return MOSCDiagnostics(
        heldout_log_density=heldout_log_density(pre_panel, mask, draws),
        pearson_dispersion=dispersion,
        residual_autocorrelation=autocorrelation,
        outcome_scale=outcome_scale,
        n_heldout_cells=int(heldout.sum()),
    )


def run_mosc(inputs: MOSCInputs, config) -> Tuple[MOSCPosterior, MOSCDiagnostics]:
    """Fit the factor model and turn its loadings into counterfactual draws."""
    panel, pre = inputs.panel, inputs.pre_periods
    modelled, rebase = _to_modelling_scale(panel, pre, config.outcome_scale)
    modelled_pre = modelled[:_pre_length(pre, config.outcome_scale)]

    rng = np.random.default_rng(config.seed)
    mask = rng.random(modelled_pre.shape) > config.heldout_fraction
    if mask.all():  # pragma: no cover - only when the draw holds nothing out
        mask[0, 0] = False

    try:
        draws = fit_factor_model(
            modelled_pre, config.factor_model, config.n_factors,
            config.n_samples, config.n_warmup, mask, config.seed,
        )
        raw = counterfactual_draws(
            draws.loadings, modelled, _pre_length(pre, config.outcome_scale),
            tuple(config.ridge_alphas),
        )
    except MlsynthEstimationError:
        raise
    except Exception as exc:
        raise MlsynthEstimationError(f"MOSC estimation failed: {exc}") from exc

    posterior = MOSCPosterior(
        loadings=draws.loadings,
        counterfactual=rebase(raw),
        n_factors=config.n_factors,
        n_draws=draws.n_draws,
        factor_model=config.factor_model,
    )
    return posterior, diagnose(modelled_pre, mask, draws, config.outcome_scale)


def _pre_length(pre_periods: int, outcome_scale: str) -> int:
    """Pre-period length on the modelling scale; differencing costs one period."""
    return pre_periods - 1 if outcome_scale == "difference" else pre_periods


def _to_modelling_scale(panel: np.ndarray, pre_periods: int, outcome_scale: str):
    """Panel as the factor model sees it, plus the inverse of that transform.

    Differencing is re-integrated from the last observed pre-intervention level,
    so a caller always reads a counterfactual on the outcome's own scale whatever
    the factor model was fit to.
    """
    if outcome_scale == "level":
        return panel, lambda draws: draws

    differenced = np.diff(panel, axis=0)
    anchor = float(panel[pre_periods - 1, 0])

    def rebase(draws: np.ndarray) -> np.ndarray:
        return anchor + np.cumsum(draws, axis=1)

    return differenced, rebase
