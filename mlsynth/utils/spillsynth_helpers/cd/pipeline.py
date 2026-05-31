"""Cao-Dowd orchestration: leave-one-out SCM, spillover solve, packaging.

Public entry point: :func:`run_cd`. Composes
:func:`fit_leave_one_out_sc` (from :mod:`.scm_core`), :func:`build_M`,
:func:`sp_estimate`, and :func:`vanilla_scm_path` (from
:mod:`.estimation`) into a single :class:`CDFit` artifact.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..structures import CDFit, SpillSynthInputs
from .estimation import build_M, sp_estimate, vanilla_scm_path
from .scm_core import fit_leave_one_out_sc


def run_cd(inputs: SpillSynthInputs, *, solver: Optional[str] = None) -> CDFit:
    """Run the Cao-Dowd spillover-adjusted SCM end-to-end.

    Parameters
    ----------
    inputs : SpillSynthInputs
        Output of :func:`prepare_spillsynth_inputs`.
    solver : str, optional
        cvxpy solver name forwarded to the per-unit SCM fits.

    Returns
    -------
    CDFit
        Bundle of intercepts, weights, per-period spillover and
        treatment-effect estimates, and the SCM/SP counterfactual paths.
    """
    a, B = fit_leave_one_out_sc(inputs.Y_pre, solver=solver)
    M = build_M(B)
    gamma, alpha, cond_AMA = sp_estimate(
        inputs.Y_post, a=a, B=B, M=M, A=inputs.A,
    )

    # Treated-unit gap and counterfactual under SP and vanilla SCM.
    y_treated_post = inputs.Y_post[0]
    gap_sp = alpha[0]                                       # = y_post - sp_cf
    counterfactual_sp = y_treated_post - gap_sp
    counterfactual_scm = vanilla_scm_path(inputs.Y_post, a=a, B=B)
    gap_scm = y_treated_post - counterfactual_scm

    # Per-affected-unit spillover panel: {label: trajectory over T1 periods}.
    spillover_panel = {
        label: gamma[1 + k].copy()
        for k, label in enumerate(inputs.affected_labels)
    }

    return CDFit(
        a=a,
        B=B,
        M=M,
        gamma=gamma,
        alpha=alpha,
        counterfactual_sp=counterfactual_sp,
        counterfactual_scm=counterfactual_scm,
        gap_sp=gap_sp,
        gap_scm=gap_scm,
        att_sp=float(gap_sp.mean()),
        att_scm=float(gap_scm.mean()),
        spillover_panel=spillover_panel,
        cond_AMA=cond_AMA,
    )
