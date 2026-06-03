"""Spillover-aware synthetic-control helper package.

Currently houses the Cao & Dowd (2023) Sp-adjusted SCM machinery under
:mod:`.cd`. Additional methods (e.g.\ iSCM Di Stefano-Mellace, distance-
weighted spillover variants) will be added as sibling subpackages and
dispatched through ``SPILLSYNTHConfig.method``.
"""

from __future__ import annotations

from .cd import run_cd
from .cd.inference import (
    KappaATestResult, PTestResult, kappa_A_test, p_test, select_A_by_kappa,
    signed_ci,
)
from .cd.sensitivity import PureDonorSensitivity, pure_donor_sensitivity
from .grossi import run_grossi
from .iscm import build_omega, build_unit_sc, run_iscm, solve_inclusive
from .plotter import plot_spillsynth
from .setup import (
    build_A_distance_decay, build_A_example3, build_A_homogeneous,
    build_A_per_unit, prepare_spillsynth_inputs,
)
from .structures import (
    CDFit, GrossiFit, ISCMFit, SpillSynthInputs, SpillSynthResults)

__all__ = [
    "CDFit",
    "GrossiFit",
    "ISCMFit",
    "KappaATestResult",
    "PTestResult",
    "PureDonorSensitivity",
    "SpillSynthInputs",
    "SpillSynthResults",
    "build_A_distance_decay",
    "build_A_example3",
    "build_A_homogeneous",
    "build_A_per_unit",
    "build_omega",
    "build_unit_sc",
    "kappa_A_test",
    "p_test",
    "plot_spillsynth",
    "prepare_spillsynth_inputs",
    "pure_donor_sensitivity",
    "run_cd",
    "run_grossi",
    "run_iscm",
    "select_A_by_kappa",
    "signed_ci",
    "solve_inclusive",
]
