"""Helpers for the SPOTSYNTH (spillover-detection synthetic control) estimator.

O'Riordan, M. & Gilligan-Lee, C. M. (2025). *"Spillover detection for donor
selection in synthetic control models."* Journal of Causal Inference
13:20240036. doi:10.1515/jci-2024-0036.

SPOTSYNTH screens every candidate donor for spillover contamination (Algorithm
1: forecast the donor's first post-intervention value from pre-intervention
donor data and flag it if the forecast fails), excludes the contaminated
donors, and builds a simplex synthetic control on the donors judged valid.

Module layout:

* :mod:`.screen` -- the Algorithm 1 forecast test (S1 / S2, two forecast
  anchors, time-averaging).
* :mod:`.sc` -- simplex synthetic-control weight solver.
* :mod:`.structures` -- :class:`SpotSynthInputs`, :class:`SpilloverScreen`,
  :class:`SpotSynthResults`.
* :mod:`.setup` -- single-treated-unit panel ingestion.
* :mod:`.pipeline` -- :func:`run_spotsynth` orchestration.
* :mod:`.plotter` -- treated-vs-screened-control chart.
* :mod:`.simulation` -- the Appendix B local-linear-trend DGP.
* :mod:`.replication` -- Figure 2 simulation + Figure 6 real-data demos.
"""

from __future__ import annotations

from .bayes import BayesianSCFit, bayesian_simplex_sc
from .debias import ProximalDebiasFit, proximal_debias
from .pipeline import run_spotsynth
from .replication import (
    DEMO,
    PAPER,
    SpotSimConfig,
    replicate_all_spillover,
    replicate_basque_spillover,
    replicate_germany_spillover,
    replicate_prop99_spillover,
    run_forecast_power_analysis,
    run_spotsynth_simulation,
)
from .sc import simplex_weights
from .screen import spillover_screen
from .setup import prepare_spotsynth_inputs
from .simulation import simulate_spillover_panel
from .structures import SpilloverScreen, SpotSynthInputs, SpotSynthResults

__all__ = [
    "DEMO",
    "PAPER",
    "BayesianSCFit",
    "ProximalDebiasFit",
    "bayesian_simplex_sc",
    "proximal_debias",
    "SpilloverScreen",
    "SpotSimConfig",
    "SpotSynthInputs",
    "SpotSynthResults",
    "prepare_spotsynth_inputs",
    "replicate_all_spillover",
    "replicate_basque_spillover",
    "replicate_germany_spillover",
    "replicate_prop99_spillover",
    "run_forecast_power_analysis",
    "run_spotsynth",
    "run_spotsynth_simulation",
    "simplex_weights",
    "simulate_spillover_panel",
    "spillover_screen",
]
