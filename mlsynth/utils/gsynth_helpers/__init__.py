"""Helper utilities for the generalized synthetic control (GSYNTH) estimator.

Implements Xu, Y. (2017), *Generalized Synthetic Control Method: Causal
Inference with Interactive Fixed Effects Models*, Political Analysis
25(1):57-76. An interactive fixed effects model is fit to the never-treated
units, each treated unit's factor loadings are recovered from its own
pre-treatment periods, and the untreated potential outcome follows from the two.
The factor count comes from the paper's Algorithm 1 and the uncertainty from its
Algorithm 2.
"""

from .config import GSYNTHConfig
from .ife import ControlFit, interactive_fixed_effects, panel_factor
from .inference import parametric_bootstrap
from .pipeline import cross_validate_rank, fit_control_model, gsc_fit
from .plotter import plot_gsynth
from .setup import prepare_gsynth_inputs
from .structures import (
    GSYNTHAggregate,
    GSYNTHCrossValidation,
    GSYNTHDesign,
    GSYNTHEventStudy,
    GSYNTHFit,
    GSYNTHInference,
    GSYNTHInputs,
    GSYNTHResults,
    GSYNTHUnitFit,
    event_study_from_effect,
)

__all__ = [
    "ControlFit",
    "GSYNTHAggregate",
    "GSYNTHConfig",
    "GSYNTHCrossValidation",
    "GSYNTHDesign",
    "GSYNTHEventStudy",
    "GSYNTHFit",
    "GSYNTHInference",
    "GSYNTHInputs",
    "GSYNTHResults",
    "GSYNTHUnitFit",
    "cross_validate_rank",
    "event_study_from_effect",
    "fit_control_model",
    "gsc_fit",
    "interactive_fixed_effects",
    "panel_factor",
    "parametric_bootstrap",
    "plot_gsynth",
    "prepare_gsynth_inputs",
]
