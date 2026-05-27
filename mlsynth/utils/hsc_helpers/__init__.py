"""Harmonic Synthetic Control (HSC) helper package.

Implements the algorithmic pieces of:

    Liu, Z., & Xu, Y. (2026). "The Harmonic Synthetic Control Method."

Module layout (one responsibility each):

    structures.py    : frozen dataclass containers (inputs / design / results)
    formulation.py   : the profiled metric W_{rho,q} and the simplex donor QP
    forecast.py      : forecasters for the smooth residual E (ARIMA(1,1,0), ...)
    optimization.py  : HSC fit at a fixed rho + rolling-origin CV over rho
    orchestration.py : top-level solve_hsc + effect summary
    plotter.py       : observed-vs-counterfactual plot

Following Liu & Xu (2026), this module ships the HSC *point* estimator only;
formal uncertainty quantification is deliberately out of scope (see the docs).
"""

from .formulation import (
    difference_operator,
    fit_donor_weights,
    roughness_matrix,
    sdid_ridge_coefficient,
    smoother_and_metric,
)
from .forecast import arima110_forecast, forecast_smooth, last_value_forecast
from .optimization import fit_at_rho, select_rho_by_cv
from .orchestration import solve_hsc, summarize_effects
from .setup import prepare_hsc_inputs
from .structures import HSCDesign, HSCInputs, HSCResults

__all__ = [
    "difference_operator",
    "roughness_matrix",
    "smoother_and_metric",
    "fit_donor_weights",
    "sdid_ridge_coefficient",
    "arima110_forecast",
    "last_value_forecast",
    "forecast_smooth",
    "fit_at_rho",
    "select_rho_by_cv",
    "solve_hsc",
    "summarize_effects",
    "prepare_hsc_inputs",
    "HSCInputs",
    "HSCDesign",
    "HSCResults",
]
