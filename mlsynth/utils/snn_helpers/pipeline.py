"""Orchestration for the SNN estimator (Agarwal et al. 2021).

In the causal/panel setting, SNN masks the treated post-treatment cells
as missing, imputes their untreated potential outcomes by synthetic
nearest-neighbors matrix completion, and forms treatment effects as
observed minus imputed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...config_models import WeightsResults
from .completion import snn_complete, snn_donor_weights
from .structures import SNNInference, SNNInputs, SNNResults

_EPS = 1e-12


def _build_weights(
    inputs: SNNInputs, *, n_neighbors, max_rank, spectral_energy, universal,
    random_state,
) -> WeightsResults:
    """Assemble a standardized WeightsResults from the per-treated-unit
    PCR donor weights."""
    X = inputs.Y.copy()
    X[inputs.D > 0] = np.nan
    names = inputs.unit_names
    per_unit = {}
    for i in inputs.treated_idx:
        AR, w = snn_donor_weights(
            X, inputs.mask if hasattr(inputs, "mask") else (inputs.D == 0).astype(float),
            int(i), n_neighbors=n_neighbors, max_rank=max_rank,
            spectral_energy=spectral_energy, universal=universal,
            random_state=random_state,
        )
        if AR.size:
            per_unit[str(names[i])] = {str(names[AR[k]]): float(w[k])
                                       for k in range(AR.size)}
    if not per_unit:
        return WeightsResults(donor_weights={}, summary_stats={"note": "no anchors"})

    if len(per_unit) == 1:
        donor_weights = next(iter(per_unit.values()))
    else:
        # Cross-treated average weight per donor.
        donor_weights = {}
        for d in {dn for u in per_unit.values() for dn in u}:
            vals = [u.get(d, 0.0) for u in per_unit.values()]
            donor_weights[d] = float(np.mean(vals))
    w_arr = np.array(list(donor_weights.values()))
    summary = {
        "sum_of_weights": float(w_arr.sum()),
        "n_donors": int(w_arr.size),
        "n_negative": int((w_arr < 0).sum()),
        "max_abs_weight": float(np.abs(w_arr).max()) if w_arr.size else 0.0,
        "constraint": "unconstrained PCR weights (need not sum to 1 or be >= 0)",
    }
    if len(per_unit) > 1:
        summary["per_unit_donor_weights"] = per_unit
    return WeightsResults(donor_weights=donor_weights, summary_stats=summary)


def _impute_counterfactual(
    Y: np.ndarray, D: np.ndarray, *, n_neighbors: int, max_rank: Optional[int],
    spectral_energy: float, universal: bool, clip: bool, random_state: int,
) -> tuple:
    """Return (counterfactual matrix, feasible mask) with treated cells imputed."""
    X = Y.copy()
    treated_cells = D > 0
    X[treated_cells] = np.nan
    lo = float(np.nanmin(X)) if clip else None
    hi = float(np.nanmax(X)) if clip else None
    completed, feasible = snn_complete(
        X, n_neighbors=n_neighbors, max_rank=max_rank,
        spectral_energy=spectral_energy, universal=universal,
        min_value=lo, max_value=hi, random_state=random_state,
    )
    # Observed (control / pre) cells keep their observed values.
    completed[~treated_cells] = Y[~treated_cells]
    feasible[~treated_cells] = True
    return completed, feasible


def _att_from_counterfactual(Y, D, counterfactual, feasible, T0, time_labels):
    treated_post = (D > 0) & feasible
    effects = np.full_like(Y, np.nan)
    effects[treated_post] = Y[treated_post] - counterfactual[treated_post]
    att = float(np.nanmean(effects[treated_post])) if treated_post.any() else np.nan
    att_by_period = {}
    for t in range(T0, Y.shape[1]):
        col = treated_post[:, t]
        if col.any():
            att_by_period[time_labels[t]] = float(np.nanmean(effects[col, t]))
    return att, effects, att_by_period


def run_snn(
    inputs: SNNInputs,
    *,
    n_neighbors: int = 1,
    max_rank: Optional[int] = None,
    spectral_energy: float = 0.95,
    universal: bool = True,
    clip: bool = True,
    inference: bool = False,
    alpha_level: float = 0.05,
    random_state: int = 0,
) -> SNNResults:
    """Run SNN and assemble :class:`SNNResults`.

    Parameters
    ----------
    inputs : SNNInputs
        Preprocessed panel.
    n_neighbors : int
        Number of synthetic neighbours (anchor-row groups) to average.
    max_rank : int, optional
        Fixed PCR truncation rank (overrides spectral/universal rule).
    spectral_energy : float
        Energy threshold for spectral rank selection.
    universal : bool
        Use the Donoho-Gavish universal hard threshold for the rank
        (default True; well-calibrated for small low-rank panels).
    clip : bool
        Clip imputations to the observed value range.
    inference : bool
        If True, run a leave-one-control jackknife for the ATT SE/CI.
    alpha_level : float
        Two-sided level for the jackknife CI.
    random_state : int
        Seed for anchor-row splitting.
    """
    Y, D, T0 = inputs.Y, inputs.D, inputs.T0

    counterfactual, feasible = _impute_counterfactual(
        Y, D, n_neighbors=n_neighbors, max_rank=max_rank,
        spectral_energy=spectral_energy, universal=universal,
        clip=clip, random_state=random_state,
    )
    att, effects, att_by_period = _att_from_counterfactual(
        Y, D, counterfactual, feasible, T0, inputs.time_labels,
    )

    weights = _build_weights(
        inputs, n_neighbors=n_neighbors, max_rank=max_rank,
        spectral_energy=spectral_energy, universal=universal,
        random_state=random_state,
    )

    inf = None
    if inference:
        inf = _jackknife_inference(
            inputs, n_neighbors=n_neighbors, max_rank=max_rank,
            spectral_energy=spectral_energy, universal=universal, clip=clip,
            alpha_level=alpha_level, random_state=random_state,
        )

    treated_post = (D > 0) & feasible
    metadata = {
        "N": inputs.N, "T": inputs.T, "T0": T0,
        "n_treated": int(inputs.treated_idx.size),
        "n_control": int(inputs.N - inputs.treated_idx.size),
        "treated_cells": int((D > 0).sum()),
        "imputed_cells": int(treated_post.sum()),
        "infeasible_cells": int(((D > 0) & ~feasible).sum()),
    }
    return SNNResults(
        inputs=inputs, att=att, counterfactual=counterfactual, effects=effects,
        att_by_period=att_by_period, feasible=feasible, weights=weights,
        inference=inf, metadata=metadata,
    )


def _jackknife_inference(
    inputs, *, n_neighbors, max_rank, spectral_energy, universal, clip,
    alpha_level, random_state,
) -> SNNInference:
    """Leave-one-control-out jackknife ATT standard error."""
    Y, D = inputs.Y, inputs.D
    control_idx = np.array([i for i in range(inputs.N)
                            if i not in set(inputs.treated_idx.tolist())])
    atts = []
    for c in control_idx:
        keep = np.array([i for i in range(inputs.N) if i != c])
        Yk, Dk = Y[keep], D[keep]
        cf, feas = _impute_counterfactual(
            Yk, Dk, n_neighbors=n_neighbors, max_rank=max_rank,
            spectral_energy=spectral_energy, universal=universal,
            clip=clip, random_state=random_state,
        )
        tp = (Dk > 0) & feas
        if tp.any():
            atts.append(float(np.nanmean((Yk - cf)[tp])))
    atts = np.asarray(atts)
    if atts.size < 2:
        return SNNInference("jackknife", float("nan"), (float("nan"), float("nan")),
                            float(alpha_level), int(atts.size))
    q = atts.size
    mean = atts.mean()
    se = float(np.sqrt((q - 1) / q * np.sum((atts - mean) ** 2)))
    from scipy.stats import norm
    z = float(norm.ppf(1.0 - alpha_level / 2.0))
    point = float(np.nanmean((Y - _impute_counterfactual(
        Y, D, n_neighbors=n_neighbors, max_rank=max_rank,
        spectral_energy=spectral_energy, universal=universal, clip=clip,
        random_state=random_state)[0])[(D > 0)]))
    ci = (point - z * se, point + z * se)
    return SNNInference("jackknife", se, ci, float(alpha_level), int(q))
