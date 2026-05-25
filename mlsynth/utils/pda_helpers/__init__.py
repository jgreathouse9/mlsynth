"""Helper subpackage for the Panel Data Approach (PDA).

NumPy-first, IndexSet-indexed. Each high-dimensional PDA variant lives in its
own subpackage with the inference theory from its source paper:

* :mod:`mlsynth.utils.pda_helpers.l2`    -- L2-relaxation (Shi & Wang 2024).
* :mod:`mlsynth.utils.pda_helpers.lasso` -- L1/LASSO (Li & Bell 2017).
* :mod:`mlsynth.utils.pda_helpers.fs`    -- forward selection (Shi & Huang 2023).

The only DataFrame touchpoint is :func:`setup.prepare_pda_inputs`.
"""

from .structures import FS, L2, LASSO, PDAInputs, PDAMethodFit, PDAResults
from .setup import derive_treatment, prepare_pda_inputs
from .inference import hac_lrv, newey_west_lag, normal_test

__all__ = [
    "L2", "LASSO", "FS",
    "PDAInputs", "PDAMethodFit", "PDAResults",
    "derive_treatment", "prepare_pda_inputs",
    "hac_lrv", "newey_west_lag", "normal_test",
]
