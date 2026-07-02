"""Helpers for Partially Pooled SCM with staggered adoption.

A faithful port of augsynth::multisynth (Ben-Michael, Feller & Rothstein 2022,
*JRSS-B* 84(2):351-381). For ``J`` treated units/cohorts (each adopting at its
own period) and never-treated controls, PPSCM removes two-way fixed effects,
balances the residuals, and picks donor weights minimizing

    nu * ||pooled imbalance||^2 / norm_pool + (1 - nu) * mean_j ||sep_j||^2 / norm_sep

over the donor simplex. ``nu`` interpolates between a separate SCM per treated
unit (small ``nu``) and a fully pooled SCM (large ``nu``); when ``nu="auto"`` it
is the triangle-inequality ratio ``global_l2 * sqrt(d) / avg_l2`` of the
separate fit. The pooled imbalance is aligned by **relative time** (time since
treatment). ``time_cohort=True`` collapses units sharing an adoption time into
one fully-pooled cohort.

Layout:
    structures.py : frozen dataclasses for inputs / design / event study / results
    setup.py      : staggered long->wide formatting (the only pandas touchpoint)
    engine.py     : fixed effects (fit_feff), the partially-pooled QP, auto-nu,
                    and the relative-time event study / ATT
    inference.py  : the paper's delete-one jackknife
    plotter.py    : event-study chart with the jackknife CI band
"""

from .structures import (
    PPSCMInputs, PPSCMDesign, PPSCMEventStudy, PPSCMInference, PPSCMResults,
    PPSCMUnitFit,
)
from .setup import prepare_ppscm_inputs
from .engine import run_multisynth, fit_feff
from .inference import jackknife_inference
from .plotter import plot_ppscm

__all__ = [
    "PPSCMInputs", "PPSCMDesign", "PPSCMEventStudy", "PPSCMInference",
    "PPSCMResults", "PPSCMUnitFit", "prepare_ppscm_inputs", "run_multisynth",
    "fit_feff", "jackknife_inference", "plot_ppscm",
]
