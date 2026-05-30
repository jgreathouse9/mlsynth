"""Helper modules for the Modified Unbiased Synthetic Control estimator.

See :mod:`mlsynth.estimators.musc` for the public estimator class.
"""

from __future__ import annotations

from .estimation import att_for_unit, predict_counterfactual, solve_musc_qp
from .inference import (
    normal_ci_from_variance,
    randomization_ci,
    unbiased_variance,
)
from .orchestration import derive_treatment, run_musc
from .plotter import plot_musc
from .setup import prepare_musc_inputs
from .structures import (
    MUSC,
    SC,
    MUSCInference,
    MUSCInputs,
    MUSCResults,
    MUSCVariantFit,
)

__all__ = [
    "MUSC",
    "SC",
    "MUSCInference",
    "MUSCInputs",
    "MUSCResults",
    "MUSCVariantFit",
    "att_for_unit",
    "derive_treatment",
    "normal_ci_from_variance",
    "plot_musc",
    "predict_counterfactual",
    "prepare_musc_inputs",
    "randomization_ci",
    "run_musc",
    "solve_musc_qp",
    "unbiased_variance",
]
