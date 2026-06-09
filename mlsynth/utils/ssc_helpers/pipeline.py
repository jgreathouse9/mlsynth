"""Orchestration for the SSC estimator (Cao, Lu & Wu 2026)."""

from __future__ import annotations

import numpy as np

from ...config_models import (
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)
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

    # Assumption 2.1 (invertibility) diagnostic: smallest eigenvalue of the
    # design Gram sum_s A_s' M A_s (the paper's Table 1). A value near zero
    # signals a near-singular, numerically unstable problem.
    try:
        gram_min_eig = float(np.linalg.eigvalsh(gram).min())
    except np.linalg.LinAlgError:
        gram_min_eig = float("nan")

    metadata = {
        "N": N, "T0": T0, "S": S, "K": K,
        "n_treated": int(inputs.treated_idx.size),
        "n_never_treated": int(N - inputs.treated_idx.size),
        "n_adoption_times": int(len(set(inputs.adoption[inputs.treated_idx].tolist()))),
        "gram_min_eigenvalue": gram_min_eig,
        "estimator": "SSC",
    }
    # Standardized event-time series: gap = event-study ATT_e curve,
    # counterfactual = no-effect baseline.
    events = sorted(event_att)
    event_curve = np.array([event_att[e] for e in events], dtype=float)
    cf = np.zeros_like(event_curve)
    std_inference = InferenceResults(
        ci_lower=(att_band.lower if att_band is not None else None),
        ci_upper=(att_band.upper if att_band is not None else None),
        p_value=(att_band.p_value if att_band is not None else None),
        method=(inf.method if inf is not None else None),
        details=inf,
    )
    return SSCResults(
        inputs=inputs, tau=tau, index=index, att_band=att_band,
        event_att=event_att, event_bands=event_bands, effects_matrix=effects,
        a_hat=a_hat, B_hat=B_hat, weights=_build_weights(B_hat, inputs),
        residuals=residuals, inference_detail=inf, metadata=metadata,
        effects=EffectsResults(att=att),
        time_series=TimeSeriesResults(
            observed_outcome=event_curve,
            counterfactual_outcome=cf,
            estimated_gap=event_curve,
            time_periods=np.asarray(events),
            intervention_time=(0 if 0 in events else None),
        ),
        fit_diagnostics=FitDiagnosticsResults(
            rmse_pre=float(np.sqrt(np.mean(np.asarray(residuals, dtype=float) ** 2)))
            if np.size(residuals) else None),
        inference=std_inference,
        method_details=MethodDetailsResults(method_name="SSC", is_recommended=True),
    )
