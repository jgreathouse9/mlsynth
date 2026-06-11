"""CTSC Path-B: Powell (2022) continuous-treatment SC vs two-way FE (Table 1).

Validates mlsynth's ``CTSC`` (the paper's generalized synthetic control) against
the calibrated Monte Carlo of Powell (2022), Section 5 / Table 1. The DGP
(:func:`mlsynth.utils.ctsc_helpers.simulation.generate_model`) has a **continuous
treatment correlated with the interactive fixed effects** and a true average
effect of exactly zero each draw, so two-way fixed effects is badly biased while
CTSC is not.

  ========  =================  =================
  Quantity  CTSC               two-way FE
  ========  =================  =================
  mean bias ~0.00              ~0.82-0.87
  RMSE      small (~0.04)      ~0.85
  ========  =================  =================

Path B (the paper's simulation): the case asserts CTSC's near-zero bias and that
it is an order of magnitude less biased than two-way FE on Models 1-2 -- the
paper's headline -- not exact Monte Carlo cells. Deterministic (seeded).
"""
from __future__ import annotations

import warnings


def run() -> dict:
    from mlsynth.utils.ctsc_helpers.simulation import run_simulation

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s1 = run_simulation(1, n_sims=40, seed=0)
        s2 = run_simulation(2, n_sims=25, seed=0)
    return {
        "ctsc_bias_m1": s1.ctsc_mean_bias,
        "fe_bias_m1": s1.fe_mean_bias,
        "ctsc_rmse_m1": s1.ctsc_rmse,
        "ctsc_bias_m2": s2.ctsc_mean_bias,
        "fe_bias_m2": s2.fe_mean_bias,
        "ctsc_beats_fe_m1": float(abs(s1.ctsc_mean_bias) < 0.25 * abs(s1.fe_mean_bias)),
        "ctsc_beats_fe_m2": float(abs(s2.ctsc_mean_bias) < 0.25 * abs(s2.fe_mean_bias)),
    }


# Deterministic (seeded). Tolerances absorb the Monte Carlo noise at n_sims=40/25.
# Reproduces Powell (2022) Table 1: CTSC is ~unbiased (true effect 0) where
# two-way FE carries a large bias (~0.82-0.87) from the treatment's correlation
# with the interactive fixed effects.
EXPECTED = {
    "ctsc_bias_m1": (0.0, 0.08),
    "fe_bias_m1": (0.82, 0.18),
    "ctsc_rmse_m1": (0.04, 0.06),
    "ctsc_bias_m2": (0.0, 0.06),
    "fe_bias_m2": (0.87, 0.18),
    "ctsc_beats_fe_m1": (1.0, 0.0),
    "ctsc_beats_fe_m2": (1.0, 0.0),
}
