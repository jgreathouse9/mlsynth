"""Orchestration for the MC-NNM estimator (Athey et al. 2021)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...config_models import InferenceResults, WeightsResults
from ..results_helpers import build_effect_submodels
from .completion import mcnnm_cv, mcnnm_fit
from .structures import MCNNMInference, MCNNMInputs, MCNNMResults

_EPS = 1e-12


def _factors_and_weights(inputs: MCNNMInputs, L: np.ndarray):
    """SVD factor decomposition of L plus implied (non-unique) donor weights.

    Returns (unit_factors, time_factors, singular_values, WeightsResults).
    Implied weights project each treated unit's low-rank row onto the
    control units' low-rank rows by least squares.
    """
    U, s, Vt = np.linalg.svd(L, full_matrices=False)
    r = int((s > 1e-6 * (s[0] if s.size else 1.0)).sum())
    r = max(r, 1)
    sqrt_s = np.sqrt(s[:r])
    unit_factors = U[:, :r] * sqrt_s
    time_factors = Vt[:r].T * sqrt_s

    names = inputs.unit_names
    control_idx = np.array([i for i in range(inputs.N)
                            if i not in set(inputs.treated_idx.tolist())])
    Lc = L[control_idx]                      # (n_control, T)
    per_unit = {}
    for i in inputs.treated_idx:
        # min_w || L[i] - sum_j w_j L[j] ||  over control rows j
        w, *_ = np.linalg.lstsq(Lc.T, L[i], rcond=None)
        per_unit[str(names[i])] = {str(names[control_idx[k]]): float(w[k])
                                   for k in range(control_idx.size)}
    if len(per_unit) == 1:
        donor_weights = next(iter(per_unit.values()))
    else:
        donor_weights = {}
        for d in {dn for u in per_unit.values() for dn in u}:
            donor_weights[d] = float(np.mean([u.get(d, 0.0)
                                              for u in per_unit.values()]))
    w_arr = np.array(list(donor_weights.values())) if donor_weights else np.array([])
    summary = {
        "weights_are": "implied_non_unique",
        "note": ("MC-NNM is a factorisation estimator; these donor weights "
                 "are derived by projecting the treated unit's low-rank row "
                 "onto the control rows and are not unique. The estimator's "
                 "actual object is the factor decomposition (unit_factors, "
                 "time_factors)."),
        "rank": r,
        "n_donors": int(w_arr.size),
        "sum_of_weights": float(w_arr.sum()) if w_arr.size else 0.0,
    }
    if len(per_unit) > 1:
        summary["per_unit_donor_weights"] = per_unit
    weights = WeightsResults(donor_weights=donor_weights, summary_stats=summary)
    return unit_factors, time_factors, s[:r], weights


def _att_from_fit(Y, D, completed, T0, time_labels):
    treated = D > 0
    effects = np.full_like(Y, np.nan)
    effects[treated] = Y[treated] - completed[treated]
    att = float(np.nanmean(effects[treated])) if treated.any() else np.nan
    att_by_period = {}
    for t in range(T0, Y.shape[1]):
        col = treated[:, t]
        if col.any():
            att_by_period[time_labels[t]] = float(np.nanmean(effects[col, t]))
    return att, effects, att_by_period


def _adoption_indices(inputs: MCNNMInputs, adoption_times) -> np.ndarray:
    """Per-unit adoption period index (-1 for never-treated).

    ``adoption_times`` (from ``datautils.dataprep``) maps unit name -> the
    time *label* of first treatment; this resolves it to a column index.
    Falls back to argmax of the treatment matrix if a unit is absent.
    """
    label_to_idx = {lab: t for t, lab in enumerate(inputs.time_labels)}
    name_to_row = {n: i for i, n in enumerate(inputs.unit_names)}
    adopt = np.full(inputs.N, -1, dtype=int)
    if adoption_times:
        for name, lab in adoption_times.items():
            i = name_to_row.get(name)
            if i is not None and lab in label_to_idx:
                adopt[i] = label_to_idx[lab]
    # Fallback for any treated unit not covered.
    for i in inputs.treated_idx:
        if adopt[i] < 0:
            adopt[i] = int(np.argmax(inputs.D[i] == 1))
    return adopt


def _staggered_aggregations(inputs, completed, adoption_times):
    """Cohort ATTs and an event-study curve from the per-cell gaps."""
    Y, D = inputs.Y, inputs.D
    adopt = _adoption_indices(inputs, adoption_times)
    gap = Y - completed                       # full gap matrix

    # Cohort ATTs: group treated units by adoption index; average post cells.
    cohort_att = {}
    cohorts = {}
    for i in inputs.treated_idx:
        cohorts.setdefault(adopt[i], []).append(i)
    for a, members in sorted(cohorts.items()):
        post = D[members, a:] > 0
        vals = gap[members, a:][post]
        if vals.size:
            cohort_att[inputs.time_labels[a]] = float(np.mean(vals))

    # Event study: average gap by relative time e = t - adoption (all t).
    by_e = {}
    for i in inputs.treated_idx:
        a = adopt[i]
        for t in range(inputs.T):
            e = t - a
            by_e.setdefault(e, []).append(gap[i, t])
    event_study = {int(e): float(np.mean(v)) for e, v in sorted(by_e.items())}
    return cohort_att, event_study


def run_mcnnm(
    inputs: MCNNMInputs,
    *,
    est_u: bool = True,
    est_v: bool = True,
    n_lam: int = 40,
    n_folds: int = 5,
    max_iter: int = 400,
    tol: float = 1e-5,
    inference: bool = False,
    alpha_level: float = 0.05,
    random_state: int = 0,
    adoption_times: Optional[dict] = None,
) -> MCNNMResults:
    """Run MC-NNM (CV over the threshold) and assemble :class:`MCNNMResults`.

    Parameters
    ----------
    inputs : MCNNMInputs
    est_u, est_v : bool
        Estimate unit / time fixed effects (recommended; default True).
    n_lam : int
        Number of candidate thresholds in the CV grid.
    n_folds : int
        Cross-validation folds over observed cells.
    inference : bool
        If True, run a leave-one-control jackknife (at the CV-selected
        threshold) for the ATT SE / CI.
    """
    Y, mask, D, T0 = inputs.Y, inputs.mask, inputs.D, inputs.T0

    fit = mcnnm_cv(Y, mask, est_u=est_u, est_v=est_v, n_lam=n_lam,
                   n_folds=n_folds, max_iter=max_iter, tol=tol,
                   random_state=random_state)
    completed = fit["completed"]
    att, effects, att_by_period = _att_from_fit(
        Y, D, completed, T0, inputs.time_labels
    )
    cohort_att, event_study = _staggered_aggregations(
        inputs, completed, adoption_times
    )
    s = np.linalg.svd(fit["L"], compute_uv=False)
    rank = int((s > 1e-6 * (s[0] if s.size else 1.0)).sum())
    unit_factors, time_factors, singular_values, weights = _factors_and_weights(
        inputs, fit["L"]
    )

    inf = None
    if inference:
        inf = _jackknife(inputs, fit["best_lambda"], est_u, est_v,
                         max_iter, tol, alpha_level)

    metadata = {
        "N": inputs.N, "T": inputs.T, "T0": T0,
        "n_treated": int(inputs.treated_idx.size),
        "n_control": int(inputs.N - inputs.treated_idx.size),
        "estimate_unit_fe": est_u, "estimate_time_fe": est_v,
        "n_missing": int((1.0 - mask).sum()),
    }
    # Cross-treated-unit observed / imputed paths drive the standardized time
    # series (and result.plot()); T0 is the common adoption reference.
    tr = inputs.treated_idx
    observed_path = Y[tr].mean(axis=0)
    counterfactual_path = completed[tr].mean(axis=0)
    std_inference = None
    if inf is not None:
        lo, hi = inf.ci
        std_inference = InferenceResults(
            method=inf.method, standard_error=float(inf.se),
            ci_lower=float(lo), ci_upper=float(hi),
            confidence_level=float(1.0 - inf.alpha_level),
            details={"n_jackknife": int(inf.n_jackknife)},
        )
    submodels = build_effect_submodels(
        observed_outcome=observed_path,
        counterfactual_outcome=counterfactual_path,
        n_pre_periods=int(T0),
        n_post_periods=int(inputs.T - T0),
        time_periods=np.asarray(inputs.time_labels),
        weights=weights,
        inference=std_inference,
        method_name="MCNNM",
        effects_overrides={"att": float(att)},
        intervention_time=(inputs.time_labels[T0] if T0 < inputs.T
                           else inputs.time_labels[-1]),
    )
    return MCNNMResults(
        **submodels,
        inputs=inputs, counterfactual_matrix=completed, effects_matrix=effects,
        att_by_period=att_by_period, cohort_att=cohort_att,
        event_study=event_study, L=fit["L"], gamma=fit["gamma"],
        delta=fit["delta"], best_lambda=float(fit["best_lambda"]), rank=rank,
        unit_factors=unit_factors, time_factors=time_factors,
        singular_values=singular_values, inference_jackknife=inf,
        metadata=metadata,
    )


def _jackknife(inputs, thr, est_u, est_v, max_iter, tol, alpha_level):
    """Leave-one-control-out jackknife ATT SE at the fixed CV threshold."""
    Y, mask, D = inputs.Y, inputs.mask, inputs.D
    control_idx = np.array([i for i in range(inputs.N)
                            if i not in set(inputs.treated_idx.tolist())])
    atts = []
    for c in control_idx:
        keep = np.array([i for i in range(inputs.N) if i != c])
        fit = mcnnm_fit(Y[keep], mask[keep], thr, est_u=est_u, est_v=est_v,
                        max_iter=max_iter, tol=tol)
        Dk = D[keep]
        tp = Dk > 0
        if tp.any():
            atts.append(float((Y[keep] - fit["completed"])[tp].mean()))
    atts = np.asarray(atts)
    if atts.size < 2:
        return MCNNMInference("jackknife", float("nan"),
                              (float("nan"), float("nan")),
                              float(alpha_level), int(atts.size))
    q = atts.size
    se = float(np.sqrt((q - 1) / q * np.sum((atts - atts.mean()) ** 2)))
    from scipy.stats import norm
    z = float(norm.ppf(1.0 - alpha_level / 2.0))
    point = float(atts.mean())
    return MCNNMInference("jackknife", se, (point - z * se, point + z * se),
                          float(alpha_level), int(q))
