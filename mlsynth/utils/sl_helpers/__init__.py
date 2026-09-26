"""Helpers for SL, the synthetic learner of Viviano and Bradic (2023)."""

from .config import SLConfig
from .diagnostics import (
    error_correlation,
    error_matrix,
    flag_degenerate,
    participation_ratio,
)
from .experts import EXPERTS, ExpertLibrary, build_experts
from .inference import (
    BootstrapResult,
    bias_adjusted_att,
    block_bootstrap_test,
    test_statistic,
)
from .pipeline import resolve_split, run_sl
from .plotter import plot_sl
from .setup import prepare_sl_inputs
from .structures import SLFit, SLInputs, SLResults
from .weights import effective_k, exponential_weights, paper_eta

__all__ = [
    "EXPERTS",
    "BootstrapResult",
    "ExpertLibrary",
    "SLConfig",
    "SLFit",
    "SLInputs",
    "SLResults",
    "bias_adjusted_att",
    "block_bootstrap_test",
    "build_experts",
    "effective_k",
    "error_correlation",
    "error_matrix",
    "exponential_weights",
    "flag_degenerate",
    "paper_eta",
    "participation_ratio",
    "plot_sl",
    "prepare_sl_inputs",
    "resolve_split",
    "run_sl",
    "test_statistic",
]
