"""End-to-end FSC: embed, weight, augment, project, infer, assemble."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...config_models import (EffectsResults, FitDiagnosticsResults,
                              MethodDetailsResults, TimeSeriesResults,
                              WeightsResults)
from .basis import cubic_bspline_basis
from .config import FSCConfig
from .project import (frobenius_weights, project_increasing, project_psd,
                      triangular_dimension)
from .structures import FSCCurve, FSCInputs, FSCResults
from .weights import (augmented_weights, fsc_weights, pre_treatment_fit,
                      select_lambda, synthetic_objects)
from .inference import (band_from_halfwidth, conformal_halfwidth,
                        placebo_p_values)


def _embedding(inputs: FSCInputs, space: str) -> np.ndarray:
    """Coordinate weights realising the isometry ``Psi`` on the argument axis.

    For curves and quantile functions the sampled values already are the
    coordinates. For matrices they are not: the Frobenius metric counts each
    off-diagonal entry twice, so the half-vectorisation becomes an isometry only
    once the off-diagonal coordinates are scaled by ``sqrt(2)``.
    """
    if space != "matrix":
        return np.ones(inputs.grid.size)
    return frobenius_weights(triangular_dimension(inputs.grid.size))


def _project(objects: np.ndarray, inputs: FSCInputs, config: FSCConfig,
             scale: np.ndarray) -> np.ndarray:
    """Project each object back onto ``Psi(M)``, in embedded coordinates."""
    if config.space == "function":
        return objects
    if config.space == "distribution":
        return np.array([project_increasing(inputs.grid, o,
                                            bounds=config.value_bounds)
                         for o in objects])
    dim = triangular_dimension(inputs.grid.size)
    return np.array([project_psd(o / scale, dim) * scale for o in objects])


def run_fsc(inputs: FSCInputs, config: FSCConfig) -> FSCResults:
    """Fit FSC and return the typed results."""
    scale = _embedding(inputs, config.space)
    values = inputs.values * scale                 # embedded coordinates
    n_pre = inputs.pre_periods

    basis: Optional[np.ndarray] = None
    if config.space != "matrix":
        basis = cubic_bspline_basis(inputs.grid, config.n_basis)

    gamma = fsc_weights(values, n_pre)
    synthetic_fsc = synthetic_objects(values, gamma)
    fit_fsc = pre_treatment_fit(values, n_pre, synthetic_fsc)

    lam: Optional[float] = None
    lam_relative: Optional[float] = None
    by_cv: Optional[bool] = None
    weights = gamma
    augmented: Optional[np.ndarray] = None
    if config.augment:
        by_cv = config.ridge_lambda is None
        if by_cv:
            lam, lam_relative = select_lambda(
                values, n_pre, config.lambda_bounds, basis=basis,
                grid=inputs.grid)
        else:
            lam = float(config.ridge_lambda)
        augmented = augmented_weights(values, n_pre, lam, basis=basis,
                                      grid=inputs.grid)
        weights = augmented

    synthetic = _project(synthetic_objects(values, weights), inputs, config,
                         scale)
    fit = pre_treatment_fit(values, n_pre, synthetic)

    lower = upper = None
    if config.inference in ("conformal", "both"):
        halfwidth = conformal_halfwidth(values, n_pre, synthetic,
                                        config.conformal_alpha)
        lower, upper = band_from_halfwidth(synthetic, halfwidth)

    p_values = None
    if config.inference in ("placebo", "both"):
        p_values = placebo_p_values(
            values, n_pre,
            lambda cube: _project(
                synthetic_objects(
                    cube,
                    augmented_weights(cube, n_pre, lam, basis=basis,
                                      grid=inputs.grid)
                    if config.augment else fsc_weights(cube, n_pre)),
                inputs, config, scale))

    curves = _build_curves(inputs, values, synthetic, scale, lower, upper)
    return _assemble(inputs, config, curves, gamma, augmented, fit, fit_fsc,
                     lam, lam_relative, by_cv, p_values, basis)


def _build_curves(inputs: FSCInputs, values: np.ndarray, synthetic: np.ndarray,
                  scale: np.ndarray, lower, upper) -> List[FSCCurve]:
    """Un-embed and package one :class:`FSCCurve` per period."""
    curves: List[FSCCurve] = []
    for t, label in enumerate(inputs.time_labels):
        gap = values[0, t] - synthetic[t]
        curves.append(FSCCurve(
            time=label,
            is_post=t >= inputs.pre_periods,
            observed=values[0, t] / scale,
            counterfactual=synthetic[t] / scale,
            effect=gap / scale,
            magnitude=float(np.sqrt(np.sum(gap ** 2))),
            lower=None if lower is None else lower[t] / scale,
            upper=None if upper is None else upper[t] / scale,
        ))
    return curves


def _assemble(inputs: FSCInputs, config: FSCConfig, curves: List[FSCCurve],
              gamma: np.ndarray, augmented: Optional[np.ndarray], fit: float,
              fit_fsc: float, lam: Optional[float],
              lam_relative: Optional[float], by_cv: Optional[bool],
              p_values, basis: Optional[np.ndarray]) -> FSCResults:
    """Lift the curve-valued output into the standardized sub-models."""
    observed = np.array([c.observed.mean() for c in curves])
    counterfactual = np.array([c.counterfactual.mean() for c in curves])
    gap = observed - counterfactual
    post = np.array([c.is_post for c in curves])
    weights = gamma if augmented is None else augmented

    band: Dict[str, Any] = {}
    if curves and curves[0].lower is not None:
        band = {
            "counterfactual_lower": np.array([c.lower.mean() for c in curves]),
            "counterfactual_upper": np.array([c.upper.mean() for c in curves]),
            "prediction_interval_level": 1.0 - config.conformal_alpha,
            "prediction_interval_kind": "conformal",
        }

    return FSCResults(
        curves=curves,
        grid=inputs.grid,
        fsc_weights=gamma,
        augmented_weights=augmented,
        pre_treatment_fit=fit,
        pre_treatment_fit_fsc=fit_fsc,
        ridge_lambda=lam,
        ridge_lambda_relative=lam_relative,
        lambda_selected_by_cv=by_cv,
        placebo_p_values=p_values,
        n_negative_weights=None if augmented is None
        else int((augmented < 0).sum()),
        effects=EffectsResults(
            att=float(gap[post].mean()),
            additional_effects={
                "mean_effect_magnitude": float(
                    np.mean([c.magnitude for c in curves if c.is_post])),
                "effect_magnitude_by_period": {
                    str(c.time): c.magnitude for c in curves if c.is_post},
            }),
        fit_diagnostics=FitDiagnosticsResults(
            rmse_pre=float(np.sqrt(np.mean(gap[~post] ** 2))),
            rmse_post=float(np.sqrt(np.mean(gap[post] ** 2))),
            additional_metrics={
                "pre_treatment_fit": fit,
                "pre_treatment_fit_fsc": fit_fsc,
                "fit_improvement": (float("nan") if fit_fsc == 0.0
                                    else 1.0 - fit / fit_fsc),
            }),
        time_series=TimeSeriesResults(
            observed_outcome=observed,
            counterfactual_outcome=counterfactual,
            estimated_gap=gap,
            time_periods=np.asarray(inputs.time_labels, dtype=object),
            intervention_time=inputs.time_labels[inputs.pre_periods]
            if inputs.post_periods else None,
            **band),
        weights=WeightsResults(
            donor_weights={str(u): float(v)
                           for u, v in zip(inputs.donor_names, weights)},
            summary_stats={
                "constraint": ("simplex" if augmented is None
                               else "sum to one (negative weights allowed)"),
                "sum": float(weights.sum())}),
        method_details=MethodDetailsResults(
            method_name="FSC",
            parameters_used={
                "space": config.space,
                "augment": config.augment,
                "n_basis": None if basis is None else int(config.n_basis),
                "ridge_lambda": lam,
                "ridge_lambda_relative": lam_relative,
                "inference": config.inference,
                "n_donors": inputs.n_donors,
                "n_pre": inputs.pre_periods,
                "n_grid": int(inputs.grid.size),
            }),
        metadata={
            "treated_unit": inputs.treated_unit_name,
            "donor_names": list(inputs.donor_names),
            "argument": inputs.argument,
            "outcome": inputs.outcome,
        },
    )
