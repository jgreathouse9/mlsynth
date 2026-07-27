"""Helper package for the COMPSC estimator (compositional / Aitchison SC)."""

from __future__ import annotations

from .assembly import assemble_compsc_results
from .config import COMPSCConfig
from .pipeline import (
    aitchison_distance,
    alr,
    closure,
    clr,
    compsc_counterfactual,
    compsc_weights,
    fit_compsc,
    geometry_sensitivity,
    log_ratio_effects,
    placebo_test,
)
from .plotter import plot_compsc
from .setup import prepare_compsc_inputs
from .structures import (
    CompscCategoryFit,
    CompscInputs,
    CompscPlacebo,
    COMPSCResults,
)

__all__ = [
    "COMPSCConfig",
    "CompscInputs",
    "CompscCategoryFit",
    "CompscPlacebo",
    "COMPSCResults",
    "prepare_compsc_inputs",
    "closure",
    "clr",
    "alr",
    "aitchison_distance",
    "compsc_weights",
    "compsc_counterfactual",
    "log_ratio_effects",
    "fit_compsc",
    "placebo_test",
    "geometry_sensitivity",
    "assemble_compsc_results",
    "plot_compsc",
]
