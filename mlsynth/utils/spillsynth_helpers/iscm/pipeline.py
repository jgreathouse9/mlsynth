"""Public dispatcher for the inclusive SCM method (Di Stefano & Mellace 2024).

Pipeline:

1. Build a synthetic control for the treated unit *and* every affected unit,
   each fit keeping the other affected units in its donor pool (the
   "inclusive" donor pool).
2. Read off the cross-weights -- the weight each affected unit receives in
   another's synthetic control -- and assemble the system matrix ``Omega``.
3. Invert ``Omega`` to de-contaminate the post-period gaps into the treated
   unit's effect and the affected units' spillover effects (eq. 6).
4. Report the inclusive ATT next to the naive (contaminated) SCM ATT, plus
   the pre-fit gain from keeping the affected units in the pool.
"""

from __future__ import annotations

import numpy as np

from ..structures import ISCMFit, SpillSynthInputs
from .system import build_omega, solve_inclusive
from .weights import build_unit_sc


def run_iscm(inputs: SpillSynthInputs, *, bilevel_solver: str = "malo") -> ISCMFit:
    """Run inclusive SCM and assemble an :class:`ISCMFit`.

    Parameters
    ----------
    inputs : SpillSynthInputs
        Preprocessed panel (row 0 treated, rows ``1 .. p`` affected). The
        affected set ``S`` is the treated unit plus the ``p`` affected units.
    bilevel_solver : {"malo", "mscmt"}
        Bilevel backend for covariate matching. Ignored (no predictor block)
        in outcome-only mode.
    """
    Y, T0, N, p = inputs.Y, inputs.T0, inputs.N, inputs.p
    P = inputs.predictors
    pnames = list(inputs.predictor_names) if inputs.predictor_names else None
    solver_label = bilevel_solver if P is not None else "outcome-only"

    names = [inputs.treated_label, *inputs.affected_labels, *inputs.clean_labels]
    S = list(range(p + 1))            # affected-set row indices: treated + affected
    m = len(S)

    # --- synthetic control for every unit in the affected set ---------------
    gaps = np.zeros((m, inputs.T))
    cross = np.zeros((m, m))
    treated_w = None
    treated_sol = None
    treated_pre_rmspe = np.nan
    treated_synthetic_pre = None
    for si, i in enumerate(S):
        donor_idx = np.array([j for j in range(N) if j != i])
        w, cf, gap, pre_rmspe, sol = build_unit_sc(
            i, donor_idx, Y, T0,
            predictors=P, predictor_names=pnames, solver=bilevel_solver,
        )
        gaps[si] = gap
        # weight that each *other* affected-set unit receives in unit i's SC
        for sk, k in enumerate(S):
            if k != i:
                pos = int(np.where(donor_idx == k)[0][0])
                cross[si, sk] = w[pos]
        if i == 0:                    # treated unit
            treated_w = dict(zip([names[j] for j in donor_idx], w))
            treated_sol = sol
            treated_pre_rmspe = pre_rmspe
            treated_synthetic_pre = cf[:T0]

    # --- de-contaminate via the cross-weight system -------------------------
    omega = build_omega(cross)
    omega_det = float(np.linalg.det(omega))
    theta = solve_inclusive(omega, gaps[:, T0:])     # (m, T1)

    gap_incl = theta[0]
    gap_scm = gaps[0, T0:]
    y_treated_post = Y[0, T0:]
    cf_incl = y_treated_post - gap_incl
    cf_scm = y_treated_post - gap_scm

    # --- restricted pre-fit: exclude affected units from the treated pool ---
    clean_idx = np.array([j for j in range(N) if j > p])  # clean controls only
    if clean_idx.size >= 1:
        _, _, _, pre_rmspe_restr, _ = build_unit_sc(
            0, clean_idx, Y, T0,
            predictors=P, predictor_names=pnames, solver=bilevel_solver,
        )
    else:
        pre_rmspe_restr = np.nan

    # --- cross-weight labels + spillover panel ------------------------------
    cross_weights = {}
    for si, i in enumerate(S):
        for sk, k in enumerate(S):
            if k != i:
                cross_weights[f"{names[k]} in {names[i]}"] = float(cross[si, sk])

    spillover_panel = {names[k]: theta[sk] for sk, k in enumerate(S) if k != 0}
    spillover_att = {lab: float(np.mean(traj)) for lab, traj in spillover_panel.items()}

    predictor_weights = None
    if treated_sol is not None and pnames is not None:
        predictor_weights = {n: float(v) for n, v in zip(pnames, treated_sol.V)}

    return ISCMFit(
        att=float(np.mean(gap_incl)),
        att_scm=float(np.mean(gap_scm)),
        gap=gap_incl,
        gap_scm=gap_scm,
        counterfactual=cf_incl,
        counterfactual_scm=cf_scm,
        theta=theta,
        omega=omega,
        omega_det=omega_det,
        cross_weights=cross_weights,
        weight_matrix=cross,            # cross-weights among the affected set
        donor_weights={k: float(v) for k, v in (treated_w or {}).items() if abs(v) > 1e-10},
        pre_rmspe=float(treated_pre_rmspe),
        pre_rmspe_restricted=float(pre_rmspe_restr),
        spillover_panel=spillover_panel,
        spillover_att=spillover_att,
        bilevel_solver=solver_label,
        predictor_weights=predictor_weights,
        treated_synthetic_pre=treated_synthetic_pre,
    )
