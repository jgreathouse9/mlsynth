"""Helpers for PANGEO -- parallel-trends supergeo experimental design.

PANGEO keeps the Supergeo (Chen et al. 2023) set-partitioning MIP but
replaces its scalar sum-matching objective with a **difference-in-
differences parallelism** score on the full pre-period trajectory: within
each treatment arm it partitions the geos into supergeo pairs whose
treatment/control halves run as parallel as possible before treatment,
so a later DiD/synthetic-control analysis enjoys clean parallel trends.

Module layout:

* :mod:`.parallelism` -- DiD pre-period gap-variance / R^2 scoring and
  admissible-pair enumeration.
* :mod:`.mip` -- the cvxpy/HiGHS set-partitioning MIP.
* :mod:`.setup` -- historical-panel ingestion + per-arm pools.
* :mod:`.pipeline` -- :func:`run_pangeo` dispatcher.
* :mod:`.simulation` -- seasonal factor-model sales generator.
* :mod:`.structures` -- :class:`PangeoResults`, :class:`ArmDesign`,
  :class:`SupergeoPair`.
"""

from __future__ import annotations

from .parallelism import (
    enumerate_candidate_pairs,
    gap_variance,
    parallelism_r2,
)
from .pipeline import run_pangeo
from .setup import PangeoInputs, prepare_pangeo_inputs
from .simulation import make_seasonal_sales_panel
from .structures import ArmDesign, PangeoResults, SupergeoPair

__all__ = [
    "ArmDesign",
    "PangeoInputs",
    "PangeoResults",
    "SupergeoPair",
    "enumerate_candidate_pairs",
    "gap_variance",
    "make_seasonal_sales_panel",
    "parallelism_r2",
    "prepare_pangeo_inputs",
    "run_pangeo",
]
