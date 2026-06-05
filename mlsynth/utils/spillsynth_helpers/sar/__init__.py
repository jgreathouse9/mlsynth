"""SAR spillover synthetic-control subpackage (Sakaguchi & Tagawa 2026).

Spatial-autoregressive Bayesian SCM that relaxes SUTVA: the control outcomes
follow a SAR panel, so a treatment on the treated unit spills onto the
controls. Two-step inference estimates the synthetic weights ``alpha`` by a
Bayesian horseshoe regression and the spatial parameter ``rho`` by a SAR
likelihood, then identifies the treatment effect on the treated unit and the
spillover effects on the controls in closed form.

Module layout:

* :mod:`.sampler` -- the horseshoe-``alpha`` Gibbs sampler, the SAR ``rho``
  Metropolis sampler (with AR(1) factor + covariate blocks), and the
  identification plug-ins.
* :mod:`.setup` -- :func:`prepare_sar_inputs` (long DataFrame + spatial-weight
  spec -> :class:`SARInputs`).
* :mod:`.pipeline` -- :func:`run_sar`, the two-step driver returning a
  :class:`SARFit`.
* :mod:`.structures` -- :class:`SARInputs`, :class:`SARFit`.
"""
from __future__ import annotations

from .pipeline import run_sar
from .sampler import (
    hs_alpha_gibbs,
    row_normalize,
    sar_full_sampler,
    spillover_effects,
    treated_counterfactual,
)
from .setup import prepare_sar_inputs
from .structures import SARFit, SARInputs

__all__ = [
    "SARFit",
    "SARInputs",
    "hs_alpha_gibbs",
    "prepare_sar_inputs",
    "row_normalize",
    "run_sar",
    "sar_full_sampler",
    "spillover_effects",
    "treated_counterfactual",
]
