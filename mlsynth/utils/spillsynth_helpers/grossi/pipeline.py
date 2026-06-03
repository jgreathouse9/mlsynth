"""Grossi et al. (2025) direct + spillover effects under partial interference.

Grossi, G., Mariani, M., Mattei, A., Lattarulo, P., & Oener, O. (2025).
"Direct and spillover effects of a new tramway line on the commercial vitality
of peripheral streets: a synthetic-control approach." JRSS-A 188(1):223-240.

Partial interference (their Assumption 1) restricts spillover to occur only
*within* the treated unit's cluster (neighbourhood), never between clusters.
That makes every unit in a *different* cluster a clean control. The method
imputes the untreated potential outcome for the treated unit **and** for each
of its cluster-mates by a penalized synthetic control (Abadie-L'Hour 2021)
built from the clean controls only:

* the treated unit's gap is the **direct effect** (eq. 3.4);
* each cluster-mate's gap is a **spillover effect**, averaged into the
  average spillover (eq. 3.5).

This is the opposite design to the inclusive method: rather than keeping the
affected neighbours in the donor pool and correcting (``method='iscm'``), it
*excludes* the whole treated cluster and leans on the far controls -- trading
fit for an interference-free counterfactual.
"""

from __future__ import annotations

import numpy as np

from ....exceptions import MlsynthConfigError, MlsynthDataError
from ..iscm.weights import build_unit_sc
from ..structures import GrossiFit, SpillSynthInputs
from .inference import residual_resampling


def run_grossi(inputs: SpillSynthInputs, *, bilevel_solver: str = "penalized",
               bias_correct: bool = False, n_boot: int = 0,
               ci_level: float = 0.90, seed: int = 0) -> GrossiFit:
    """Run the Grossi et al. partial-interference estimator.

    Parameters
    ----------
    inputs : SpillSynthInputs
        Preprocessed panel (row 0 treated, rows ``1 .. p`` the affected
        cluster-mates, rows ``p+1 ..`` the clean controls).
    bilevel_solver : {"penalized", "malo", "mscmt"}
        Backend used to build each cluster unit's synthetic control from the
        clean controls. Defaults to ``"penalized"`` (the paper's estimator).
    bias_correct : bool
        Apply the Abadie-L'Hour bias correction to each gap (covariate mode).
    n_boot : int
        Number of residual-resampling draws for the pivotal bias-corrected
        confidence intervals (eqs. 3.6-3.7). ``0`` (default) skips inference.
        Supported for the penalized backend and outcome-only matching.
    ci_level : float
        Confidence level for the intervals (the paper uses 0.90).
    seed : int
        RNG seed for the resampling.
    """
    Y, T0, N, p = inputs.Y, inputs.T0, inputs.N, inputs.p
    if p < 1:
        raise MlsynthConfigError(
            "SPILLSYNTH method='grossi' needs at least one affected unit "
            "(the treated unit's cluster-mates) in affected_units."
        )
    clean_idx = np.arange(p + 1, N)                 # far / clean controls
    if clean_idx.size < 2:
        raise MlsynthDataError(
            "SPILLSYNTH method='grossi' needs >= 2 clean controls (units "
            "outside the treated unit's cluster) to build a synthetic control."
        )

    P = inputs.predictors
    pnames = list(inputs.predictor_names) if inputs.predictor_names else None
    names = [inputs.treated_label, *inputs.affected_labels, *inputs.clean_labels]
    cluster = list(range(p + 1))                     # treated + affected

    # Penalized SC for each cluster unit, built from the clean controls only.
    gaps_full = {}
    lam_cluster: list = []
    treated_w = None
    treated_pre = None
    direct_pre_rmspe = float("nan")
    lam = float("nan")
    for i in cluster:
        w, cf, gap, pre_rmspe, sol = build_unit_sc(
            i, clean_idx, Y, T0, predictors=P, predictor_names=pnames,
            solver=bilevel_solver, bias_correct=bias_correct,
        )
        gaps_full[i] = gap
        lam_cluster.append(
            float(sol.metadata.get("lambda", float("nan"))) if sol is not None else None)
        if i == 0:
            treated_w = dict(zip([names[j] for j in clean_idx], w))
            treated_pre = cf[:T0]
            direct_pre_rmspe = pre_rmspe
            if sol is not None:
                lam = float(sol.metadata.get("lambda", float("nan")))

    direct_gap = gaps_full[0][T0:]
    spillover_panel = {names[i]: gaps_full[i][T0:] for i in cluster if i != 0}
    avg_spillover_gap = np.mean(
        np.vstack([gaps_full[i][T0:] for i in cluster if i != 0]), axis=0)

    # Naive SCM using ALL controls (incl. the affected units) for comparison.
    all_donors = np.arange(1, N)
    _, _, gap_naive, _, _ = build_unit_sc(
        0, all_donors, Y, T0, predictors=P, predictor_names=pnames,
        solver=bilevel_solver, bias_correct=bias_correct,
    )
    gap_scm = gap_naive[T0:]

    # Residual-resampling inference (penalized backend or outcome-only).
    direct_ci = avg_spillover_ci = None
    if n_boot > 0:
        if P is not None and bilevel_solver != "penalized":
            raise MlsynthConfigError(
                "SPILLSYNTH method='grossi' inference (n_boot>0) is supported "
                "for bilevel_solver='penalized' or outcome-only matching; got "
                f"bilevel_solver={bilevel_solver!r} with covariates."
            )
        direct_ci, avg_spillover_ci = residual_resampling(
            Y, T0, p, N, P, lam_cluster, n_boot=n_boot, ci_level=ci_level,
            seed=seed, bias_correct=bias_correct,
            direct_att=direct_gap, avg_spill_att=avg_spillover_gap,
        )

    y_treated_post = Y[0, T0:]
    return GrossiFit(
        direct_att=float(np.mean(direct_gap)),
        att_scm=float(np.mean(gap_scm)),
        gap=direct_gap,
        gap_scm=gap_scm,
        counterfactual=y_treated_post - direct_gap,
        counterfactual_scm=y_treated_post - gap_scm,
        avg_spillover_att=float(np.mean(avg_spillover_gap)),
        avg_spillover_gap=avg_spillover_gap,
        spillover_panel=spillover_panel,
        spillover_att={k: float(np.mean(v)) for k, v in spillover_panel.items()},
        direct_pre_rmspe=float(direct_pre_rmspe),
        donor_weights={k: float(v) for k, v in (treated_w or {}).items()
                       if abs(v) > 1e-10},
        treated_synthetic_pre=treated_pre,
        n_clean=int(clean_idx.size),
        lam=lam,
        direct_ci=direct_ci,
        avg_spillover_ci=avg_spillover_ci,
        bilevel_solver=bilevel_solver,
    )
