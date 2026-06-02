"""Inclusive SCM (Di Stefano & Mellace 2024) subpackage.

Keeps the "potentially affected" control units in the donor pool -- instead
of discarding them as the conventional spillover-robust recipe does -- and
removes the bias their inclusion causes by solving a small cross-weight
linear system.

Module layout:

* :mod:`.weights` -- per-unit synthetic control (outcome-only or covariate
  matching through the FSCM/MASC bilevel solver, with a ``malo`` / ``mscmt``
  backend choice).
* :mod:`.system` -- the ``Omega`` cross-weight system and its inverse (eq. 6).
* :mod:`.pipeline` -- public dispatcher :func:`run_iscm`.
"""

from __future__ import annotations

from .pipeline import run_iscm
from .system import build_omega, solve_inclusive
from .weights import build_unit_sc

__all__ = [
    "build_omega",
    "build_unit_sc",
    "run_iscm",
    "solve_inclusive",
]
