"""Helpers for CPDA, Hsiao and Zhou (2019) Section 3.

* :mod:`.setup`      -- long DataFrame to NumPy, the only pandas touchpoint.
* :mod:`.beta`       -- Step 1, the covariate slope (Pesaran CCE or Bai IFE).
* :mod:`.selection`  -- Step 3's donor subset, and why the rule is a choice.
* :mod:`.pipeline`   -- Equations 11 to 15 end to end.
* :mod:`.structures` -- frozen inputs, fit and results.
* :mod:`.plotter`    -- returns a Figure; showing is the caller's job.
"""

from .beta import bai_objective, beta_bai, beta_cce, estimate_beta
from .config import CPDAConfig
from .pipeline import residualise, run_cpda
from .plotter import plot_cpda
from .selection import SELECTORS, select_donors
from .setup import prepare_cpda_inputs
from .structures import CPDAFit, CPDAInputs, CPDAResults

__all__ = [
    "CPDAConfig", "CPDAFit", "CPDAInputs", "CPDAResults", "SELECTORS",
    "bai_objective", "beta_bai", "beta_cce", "estimate_beta", "plot_cpda",
    "prepare_cpda_inputs", "residualise", "run_cpda", "select_donors",
]
