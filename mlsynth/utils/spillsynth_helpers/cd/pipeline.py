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
from .estimation import (
    build_M, estimate_omega_from_pre_residuals, sp_estimate,
    sp_estimate_weighted, vanilla_scm_path,
)
from .inference import kappa_A_test, run_per_period_tests
from .scm_core import fit_leave_one_out_sc
from .sensitivity import pure_donor_sensitivity


def run_cd(
    inputs: SpillSynthInputs,
    *,
    solver: Optional[str] = None,
    weighting: str = "identity",
) -> CDFit:
    """Run the Cao-Dowd spillover-adjusted SCM end-to-end.

    Parameters
    ----------
    inputs : SpillSynthInputs
        Output of :func:`prepare_spillsynth_inputs`.
    solver : str, optional
        cvxpy solver name forwarded to the per-unit SCM fits.
    weighting : {"identity", "efficient"}
        ``"identity"`` -- the standard Cao-Dowd estimator with
        :math:`W = I`. ``"efficient"`` -- additionally compute the
        GMM-weighted variant of Proposition S.1 (v3 Section S.1.1)
        using :math:`\\widehat W = \\widehat \\Omega^{-1}`, where
        :math:`\\widehat \\Omega` is the sample covariance of the
        pre-period residuals. Both the identity and efficient results
        are returned; the headline ``att_sp`` and ``alpha`` reflect the
        identity (W=I) fit, while the efficient variant is exposed via
        ``CDFit.efficient_fit``.

    Returns
    -------
    CDFit
        Bundle of intercepts, weights, per-period spillover and
        treatment-effect estimates, P-test inference, CIs, the
        :math:`\\kappa_A` specification test, and (optionally) the
        efficient-weighted variant.
    """
    if weighting not in ("identity", "efficient"):
        raise ValueError(
            f"run_cd: weighting must be 'identity' or 'efficient'; "
            f"got {weighting!r}."
        )

    a, B = fit_leave_one_out_sc(inputs.Y_pre, solver=solver)
    M = build_M(B)
    gamma, alpha, cond_AMA = sp_estimate(
        inputs.Y_post, a=a, B=B, M=M, A=inputs.A, warn=True,
    )

    n_treated = inputs.n_treated
    # Per-treated paths (rows 0..n_treated-1 are the treated units).
    y_treated_post = inputs.Y_post[:n_treated]              # (n_treated, T1)
    gaps_sp = alpha[:n_treated]                              # (n_treated, T1)
    counterfactuals_sp = y_treated_post - gaps_sp

    # Vanilla SCM counterfactual: each treated unit gets its own
    # leave-one-out fit (row i of (a, B)).
    counterfactuals_scm = np.stack([
        a[i] + B[i] @ inputs.Y_post for i in range(n_treated)
    ])
    gaps_scm = y_treated_post - counterfactuals_scm

    # Per-treated dicts keyed by label.
    gaps_sp_by_unit = {
        label: gaps_sp[i].copy()
        for i, label in enumerate(inputs.treated_labels)
    }
    gaps_scm_by_unit = {
        label: gaps_scm[i].copy()
        for i, label in enumerate(inputs.treated_labels)
    }
    atts_sp_by_unit = {label: float(g.mean()) for label, g in gaps_sp_by_unit.items()}
    atts_scm_by_unit = {label: float(g.mean()) for label, g in gaps_scm_by_unit.items()}

    # Back-compat: scalar/vector fields reflect treated unit 0 (the only
    # treated unit in the single-treated case).
    gap_sp = gaps_sp[0]
    counterfactual_sp = counterfactuals_sp[0]
    gap_scm = gaps_scm[0]
    counterfactual_scm = counterfactuals_scm[0]

    # Per-affected-unit spillover panel: {label: trajectory over T1 periods}.
    # alpha is partitioned as
    #   rows 0..n_treated-1                 -> treated-unit gaps
    #   rows n_treated..n_treated+p-1       -> affected-unit spillovers
    #   rows n_treated+p..                  -> clean (identically zero, except in
    #                                          distance_decay where every control
    #                                          gets a decayed spillover).
    spillover_panel: dict = {}
    if inputs.spillover_structure == "distance_decay":
        decay = inputs.A[n_treated:, n_treated]
        control_labels_in_order = (*inputs.affected_labels, *inputs.clean_labels)
        for k, label in enumerate(control_labels_in_order):
            if decay[k] > 0:
                spillover_panel[label] = alpha[n_treated + k].copy()
    else:
        for k, label in enumerate(inputs.affected_labels):
            spillover_panel[label] = alpha[n_treated + k].copy()

    # Cao-Dowd Section 4 P-test inference + signed CIs + joint test.
    # ``run_per_period_tests`` now returns per-treated dicts.
    (treatment_tests, spillover_tests, treatment_cis_95,
     spillover_ci_95, joint_spillover_test) = run_per_period_tests(
        alpha_hat=alpha,
        Y_pre=inputs.Y_pre,
        a=a,
        B=B,
        A=inputs.A,
        treated_labels=inputs.treated_labels,
        affected_labels=inputs.affected_labels,
    )

    # Back-compat: the legacy ``treatment_test`` / ``treatment_ci_95``
    # fields point at the FIRST treated unit (= the only one in the
    # single-treated case).
    first_label = inputs.treated_labels[0]
    treatment_test = treatment_tests[first_label]
    treatment_ci_95 = treatment_cis_95[first_label]

    # Cao-Dowd v3 Section 5.1.2 specification test for A.
    kA = kappa_A_test(
        Y_post=inputs.Y_post, alpha_hat=alpha,
        Y_pre=inputs.Y_pre, a=a, B=B, A=inputs.A,
    )

    # Cao-Dowd v3 Section 5.2 pure-donor sensitivity. Only meaningful
    # when at least one column of A is a unit basis vector for an
    # affected (non-clean) control -- otherwise there are no
    # "assumed-clean" units against which to bound misspecification.
    pd_sens = None
    if (np.abs(inputs.A).sum(axis=1) == 0).any():
        pd_sens = pure_donor_sensitivity(inputs.Y_pre, B=B, A=inputs.A)

    # Optional GMM-efficient variant (Proposition S.1).
    efficient_fit = None
    if weighting == "efficient":
        Omega = estimate_omega_from_pre_residuals(inputs.Y_pre, a=a, B=B)
        W = np.linalg.inv(Omega)
        gamma_W, alpha_W, cond_AMA_W = sp_estimate_weighted(
            inputs.Y_post, a=a, B=B, A=inputs.A, W=W, warn=True,
        )
        efficient_fit = {
            "gamma_W": gamma_W,
            "alpha_W": alpha_W,
            "W": W,
            "Omega_hat": Omega,
            "cond_AMA_W": cond_AMA_W,
            "att_sp_W": float(alpha_W[0].mean()),
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
        treatment_test=treatment_test,
        spillover_tests=spillover_tests,
        treatment_ci_95=treatment_ci_95,
        spillover_ci_95=spillover_ci_95,
        joint_spillover_test=joint_spillover_test,
        kappa_A_test=kA,
        pure_donor_sensitivity=pd_sens,
        efficient_fit=efficient_fit,
        # Multi-treated extensions (Section S.1.2).
        gaps_sp_by_unit=gaps_sp_by_unit,
        gaps_scm_by_unit=gaps_scm_by_unit,
        atts_sp_by_unit=atts_sp_by_unit,
        atts_scm_by_unit=atts_scm_by_unit,
        treatment_tests=treatment_tests,
        treatment_cis_95=treatment_cis_95,
    )
