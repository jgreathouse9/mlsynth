"""Public dispatcher for the Iterative ("waterfall") SCM (Melnychuk 2024).

Pipeline (a port of Melnychuk's ``runIterativeSCM`` /
``runIterativeSCMwithCov`` in waterfall mode):

1. **Clean** each spillover-affected control in turn. A synthetic control is
   built for the affected unit from the *clean* controls plus any
   already-cleaned affected units (the treated unit and not-yet-cleaned affected
   units are excluded). The affected unit's outcomes are replaced by this
   spillover-free synthetic. In waterfall mode each cleaning step may reuse the
   donors cleaned before it.
2. **Refit** the treated unit's synthetic control on the cleaned donor pool, so
   the affected donors no longer carry the treatment's spillover.

``replace_pre`` is Melnychuk's ``replacePreTreatData``, and it decides how much
of the affected donor's series step 1 overwrites. The two settings are
different estimators, so the choice is not cosmetic:

``False`` (the default)
    Only the post-treatment outcomes are replaced. The treated unit's fitting
    data is bit-identical to the naive fit's, so the refit returns the naive
    weights and the correction reaches the estimate only through the cleaned
    post-period counterfactual.
``True``
    The whole series is replaced. The cleaned donor's pre-period is then an
    exact convex combination of the pool it was built from, so the refit has no
    reason to load on it and the weights move -- on German reunification
    (outcome-only) Austria's weight goes 0.46 to 0.00 and the ATT moves by
    ~320 USD.

The reference's function signature defaults to ``FALSE``, but its call sites --
``runAllMethods`` and ``runAllMethodsWithCov``, which produce the paper's tables
-- pass ``TRUE``. The default here stays ``False`` for backward compatibility;
set ``iterative_replace_pre=True`` to reproduce the published configuration.

The per-unit SCM backend is the caller's choice: outcome-only (optionally
demeaned-with-intercept) or covariate matching through the FSCM/MASC bilevel
solver. Melnychuk's headline German result uses the *covariate* backend
(Abadie's ``synth`` with special predictors).
"""

from __future__ import annotations

import numpy as np

from ..structures import IterativeFit, SpillSynthInputs
from ..iscm.weights import build_unit_sc


def run_iterative(inputs: SpillSynthInputs, *, bilevel_solver: str = "mscmt",
                  bias_correct: bool = False, intercept: bool = False,
                  replace_pre: bool = False) -> IterativeFit:
    """Run the Iterative ("waterfall") SCM and assemble an :class:`IterativeFit`.

    Parameters
    ----------
    inputs : SpillSynthInputs
        Preprocessed panel (row 0 treated, rows ``1 .. p`` affected, the rest
        clean controls).
    bilevel_solver : {"mscmt", "malo", "penalized"}
        Backend for covariate matching (ignored without a predictor block).
    bias_correct : bool
        Apply the Abadie-L'Hour bias correction to each unit's gap (covariate
        mode only).
    intercept : bool
        Use the demeaned simplex SCM with a level shift (outcome-only mode).
    replace_pre : bool
        Melnychuk's ``replacePreTreatData``. False replaces each affected
        donor's post-treatment outcomes only; True replaces its whole series,
        which is what the reference's call sites do. See the module docstring.
    """
    Y = np.array(inputs.Y, dtype=float)               # working copy (N, T)
    Y_orig = inputs.Y
    T0, N, p = inputs.T0, inputs.N, inputs.p
    P = inputs.predictors
    pnames = list(inputs.predictor_names) if inputs.predictor_names else None
    names = [inputs.treated_label, *inputs.affected_labels, *inputs.clean_labels]
    solver_label = bilevel_solver if P is not None else (
        "intercept" if intercept else "outcome-only")

    affected = list(range(1, p + 1))
    clean = list(range(p + 1, N))
    cleaned: list = []
    spillover_panel: dict = {}
    spillover_panel_pre: dict = {}
    spillover_att: dict = {}

    # --- waterfall: clean each affected control's outcomes ------------------
    for i in affected:
        donors = np.array(sorted(clean + cleaned))
        if donors.size == 0:                          # nothing clean to learn from
            cleaned.append(i)
            continue
        _, cf, _, _, _ = build_unit_sc(
            i, donors, Y, T0, predictors=P, predictor_names=pnames,
            solver=bilevel_solver, bias_correct=bias_correct, intercept=intercept)
        spillover_panel[names[i]] = cf[T0:]
        spillover_att[names[i]] = float(np.mean(Y_orig[i, T0:] - cf[T0:]))
        Y[i, T0:] = cf[T0:]
        if replace_pre:                               # replacePreTreatData=TRUE
            spillover_panel_pre[names[i]] = np.asarray(cf[:T0])
            Y[i, :T0] = cf[:T0]
        cleaned.append(i)

    # --- refit the treated unit on the cleaned pool -------------------------
    donors_final = np.array([j for j in range(N) if j != 0])
    w_f, cf_f, gap_f, pre_rmspe, _ = build_unit_sc(
        0, donors_final, Y, T0, predictors=P, predictor_names=pnames,
        solver=bilevel_solver, bias_correct=bias_correct, intercept=intercept)

    # The naive counterfactual reuses the refit weights on the original
    # (contaminated) outcomes. Without replace_pre the treated unit's fitting
    # data never changed, so those weights *are* the naive ones and the two
    # pre-period fits coincide; with it they separate.
    cf_naive = Y_orig[donors_final].T @ w_f
    if intercept and P is None:                       # add back the level shift
        mu1 = Y_orig[0, :T0].mean()
        mu0 = Y_orig[donors_final][:, :T0].mean(axis=1)
        cf_naive = cf_naive + float(mu1 - mu0 @ w_f)
    gap_naive = Y_orig[0] - cf_naive

    donor_weights = {names[int(donors_final[k])]: float(v)
                     for k, v in enumerate(w_f) if abs(v) > 1e-10}

    return IterativeFit(
        att=float(np.mean(gap_f[T0:])),
        att_scm=float(np.mean(gap_naive[T0:])),
        gap=gap_f[T0:],
        gap_scm=gap_naive[T0:],
        counterfactual=cf_f[T0:],
        counterfactual_scm=cf_naive[T0:],
        spillover_panel=spillover_panel,
        spillover_panel_pre=spillover_panel_pre,
        spillover_att=spillover_att,
        donor_weights=donor_weights,
        cleaned_units=[names[i] for i in affected],
        n_clean=len(clean),
        pre_rmspe=float(pre_rmspe),
        treated_synthetic_pre=np.asarray(cf_f[:T0]),
        naive_synthetic_pre=np.asarray(cf_naive[:T0]),
        replace_pre=replace_pre,
        bilevel_solver=solver_label,
    )
