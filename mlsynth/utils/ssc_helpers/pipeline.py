"""Orchestration for the SSC estimator (Cao, Lu & Wu 2026)."""

from __future__ import annotations

import numpy as np

from ...config_models import WeightsResults
from .estimation import (
    aggregate,
    build_treatment_structure,
    estimate_tau,
    event_time_maps,
    placebo_windows,
)
from .structures import SSCInference, SSCInputs, SSCResults
from .weights import synthetic_control_batch

_TOL = 1e-4


def _build_weights(B_hat: np.ndarray, inputs: SSCInputs) -> WeightsResults:
    """Per-treated-unit donor weights + an aggregate summary."""
    names = [str(u) for u in inputs.unit_names]
    per_unit = {}
    for i in inputs.treated_idx:
        row = B_hat[i]
        per_unit[names[i]] = {names[j]: float(row[j])
                              for j in range(len(names)) if abs(row[j]) > _TOL}
    if inputs.treated_idx.size == 1:
        donor_weights = next(iter(per_unit.values()))
    else:
        donor_weights = {}
        for d in names:
            vals = [B_hat[i, names.index(d)] for i in inputs.treated_idx]
            w = float(np.mean(vals))
            if abs(w) > _TOL:
                donor_weights[d] = w
    nonzero = np.sum(np.abs(B_hat[inputs.treated_idx]) > _TOL, axis=1)
    summary = {
        "weights_are": "ssc_per_unit_simplex",
        "note": ("Each unit's untreated outcome is a demeaned simplex synthetic "
                 "control on all other units (Cao, Lu & Wu 2026, eq. 2.1). "
                 "donor_weights averages the treated units' rows of B_hat."),
        "n_treated": int(inputs.treated_idx.size),
        "avg_active_donors_per_treated": float(np.mean(nonzero)) if nonzero.size else 0.0,
        "per_unit_donor_weights": per_unit,
    }
    return WeightsResults(donor_weights=donor_weights, summary_stats=summary)


def run_ssc(inputs: SSCInputs, *, inference: bool = True,
            alpha: float = 0.1) -> SSCResults:
    """Run SSC end to end and assemble :class:`SSCResults`.

    Parameters
    ----------
    inputs : SSCInputs
    inference : bool
        Attach Andrews end-of-sample bands/p-values to the event-time and
        overall ATT (default True).
    alpha : float
        Two-sided level for the bands.
    """
    Y, D, T0 = inputs.Y, inputs.D, inputs.T0
    N, S, K = inputs.N, inputs.S, inputs.K

    a_hat, B_hat = synthetic_control_batch(Y[:, :T0])
    index, A = build_treatment_structure(D, T0)
    tau, gram, residuals = estimate_tau(Y, T0, A, a_hat, B_hat)

    V = placebo_windows(gram, A, B_hat, residuals, T0) if inference else None

    # Per-cell effects on the post grid (NaN where untreated).
    effects = np.full((N, S), np.nan)
    for k in range(K):
        s, i, _ = index[k]
        effects[i, s - 1] = tau[k]

    # Overall ATT.
    L_all = np.full((1, K), 1.0 / K)
    att = float(np.ravel(L_all @ tau)[0])
    att_band = aggregate(L_all, tau, V, alpha, None, K) if inference else None

    # Event-time ATT.
    emaps = event_time_maps(index)
    event_att = {e: float(L @ tau) for e, L in emaps.items()}
    if inference:
        event_bands = {e: aggregate(L.reshape(1, -1), tau, V, alpha, e,
                                    int((index[:, 2] == e).sum()))
                       for e, L in emaps.items()}
    else:
        event_bands = {}

    inf = SSCInference("andrews_eos", float(alpha),
                       int(max(0, T0 - S))) if inference else None

    metadata = {
        "N": N, "T0": T0, "S": S, "K": K,
        "n_treated": int(inputs.treated_idx.size),
        "n_never_treated": int(N - inputs.treated_idx.size),
        "n_adoption_times": int(len(set(inputs.adoption[inputs.treated_idx].tolist()))),
        "estimator": "SSC",
    }
    return SSCResults(
        inputs=inputs, tau=tau, index=index, att=att, att_band=att_band,
        event_att=event_att, event_bands=event_bands, effects=effects,
        a_hat=a_hat, B_hat=B_hat, weights=_build_weights(B_hat, inputs),
        residuals=residuals, inference=inf, metadata=metadata,
    )
