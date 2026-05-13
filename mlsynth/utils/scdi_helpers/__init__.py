"""Helper utilities for the SCDI estimator."""

from .formulation import (
    build_global_2way_components,
    build_global_2way_constraints,
    build_global_2way_objective,
    build_per_unit_components,
    build_per_unit_constraints,
    build_per_unit_objective,
    build_scdi_problem_components,
)
from .inference import permutation_test_global
from .optimization import estimate_lambda, solve_synthetic_design
from .setup import prepare_scdi_inputs
from .structures import (
    SCDIDesign,
    SCDIInference,
    SCDIInputs,
    SCDIProblemComponents,
    SCDIResults,
)

__all__ = [
    "SCDIDesign",
    "SCDIInference",
    "SCDIInputs",
    "SCDIProblemComponents",
    "SCDIResults",
    "build_global_2way_components",
    "build_global_2way_constraints",
    "build_global_2way_objective",
    "build_per_unit_components",
    "build_per_unit_constraints",
    "build_per_unit_objective",
    "build_scdi_problem_components",
    "estimate_lambda",
    "permutation_test_global",
    "prepare_scdi_inputs",
    "solve_synthetic_design",
]
