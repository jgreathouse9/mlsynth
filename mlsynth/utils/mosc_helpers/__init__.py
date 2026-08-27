"""Helpers for MOSC -- many-outcomes synthetic control."""

from .config import MOSCConfig
from .factor import FACTOR_MODELS, FactorDraws, gap_gibbs, heldout_log_density, ppca_em
from .pipeline import (
    counterfactual_draws,
    diagnose,
    fit_factor_model,
    resample_columns,
    run_mosc,
)
from .plotter import plot_mosc_posterior
from .setup import prepare_mosc_inputs
from .structures import (
    MOSCDiagnostics,
    MOSCInference,
    MOSCInputs,
    MOSCPosterior,
    MOSCResults,
)

__all__ = [
    "FACTOR_MODELS",
    "FactorDraws",
    "MOSCConfig",
    "MOSCDiagnostics",
    "MOSCInference",
    "MOSCInputs",
    "MOSCPosterior",
    "MOSCResults",
    "counterfactual_draws",
    "diagnose",
    "fit_factor_model",
    "gap_gibbs",
    "heldout_log_density",
    "plot_mosc_posterior",
    "ppca_em",
    "prepare_mosc_inputs",
    "resample_columns",
    "run_mosc",
]
