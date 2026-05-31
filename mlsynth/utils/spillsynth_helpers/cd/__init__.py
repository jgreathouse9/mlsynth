"""Cao & Dowd (2023) spillover-adjusted SCM subpackage.

Module layout:

* :mod:`.scm_core` -- leave-one-out demeaned simplex SCM weights
  (eq. 1-2 of the paper).
* :mod:`.estimation` -- per-period spillover-adjusted treatment-effect
  formula (eq. 5).
* :mod:`.pipeline` -- public dispatcher :func:`run_cd`.
"""

from __future__ import annotations

from .estimation import build_M, sp_estimate, vanilla_scm_path
from .pipeline import run_cd
from .scm_core import fit_demeaned_sc, fit_leave_one_out_sc

__all__ = [
    "build_M",
    "fit_demeaned_sc",
    "fit_leave_one_out_sc",
    "run_cd",
    "sp_estimate",
    "vanilla_scm_path",
]
