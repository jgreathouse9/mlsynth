"""Helper package for the PROPSC estimator (compositional common-weights SC/SDID)."""

from __future__ import annotations

from .assembly import assemble_propsc_results
from .config import PROPSCConfig
from .pipeline import (
    estimate_common_weights,
    jackknife_se,
    sc_weight_fw,
)
from .plotter import plot_propsc
from .setup import prepare_propsc_inputs
from .structures import (
    PropscInputs,
    PropscProportionFit,
    PROPSCResults,
)

__all__ = [
    "PROPSCConfig",
    "PropscInputs",
    "PropscProportionFit",
    "PROPSCResults",
    "prepare_propsc_inputs",
    "estimate_common_weights",
    "jackknife_se",
    "sc_weight_fw",
    "assemble_propsc_results",
    "plot_propsc",
]
