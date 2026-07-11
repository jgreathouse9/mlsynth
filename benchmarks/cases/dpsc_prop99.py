"""Cross-validation benchmark: DPSC vs the authors' ``srho1/dpsc`` (Prop 99).

Cross-validates ``mlsynth``'s differentially private synthetic-control mechanisms
against the authors' own ``PrivateSC`` (Rho, Cummings & Misra 2023) on the
Abadie-Diamond-Hainmueller Proposition 99 smoking panel
(``basedata/smoking_data.csv``: California treated from 1989, 38 donors,
``T0 = 19`` pre-periods).

Both implementations are randomized. ``numpy.random`` seeds the authors' global
RNG; ``numpy.random.RandomState`` seeds mlsynth's -- the two are the same
Mersenne-Twister stream, so a shared seed makes the privatized output
reproducible on both sides. On the identical panel, matched seed, and matched
budget (``lambda = 10``, ``eps1 = eps2 = 50``), mlsynth reproduces the authors'
private coefficients and private counterfactual value-for-value, for both the
output-perturbation (Algorithm 2) and objective-perturbation (Algorithm 3)
mechanisms. Path A / cross-validation (scenario 3). Skips gracefully when the
authors' clone is unavailable.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_T0, _LAMBDA, _EPS = 19, 10.0, 50.0
_SEED = 0


def _panel_arrays():
    from mlsynth.utils.datautils import dataprep
    df = pd.read_csv(_BASE / "smoking_data.csv")
    df["treat"] = df["Proposition 99"].astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prep = dataprep(df, "state", "year", "cigsale", "treat")
    y = np.asarray(prep["y"], float).flatten()
    D = np.asarray(prep["donor_matrix"], float)
    return y, D


def _reference(pre_donor, pre_target, donor_full, mechanism):
    """Run the authors' PrivateSC under the global-RNG seed; return (weights, pred)."""
    from benchmarks.reference.clone_dpsc import import_private_sc
    psc_mod = import_private_sc()
    method = "out" if mechanism == "output" else "obj"
    pre_donor_df = pd.DataFrame(pre_donor)
    pre_target_df = pd.DataFrame(pre_target.reshape(-1, 1))
    donor_full_df = pd.DataFrame(donor_full)
    np.random.seed(_SEED)
    psc = psc_mod.PrivateSC()
    psc.set_params(method=method, lmbda=_LAMBDA, eps1=_EPS, eps2=_EPS, delta=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        psc.fit(pre_donor_df, pre_target_df, fit_intercept=False)
        pred = psc.predict(donor_full_df).values.flatten()
    return np.asarray(psc.weights, float).flatten(), pred


def _mlsynth(pre_donor, pre_target, donor_full, mechanism):
    from mlsynth.utils.dpsc_helpers.mechanisms import (
        run_objective_perturbation, run_output_perturbation)
    run = run_output_perturbation if mechanism == "output" else run_objective_perturbation
    cf, w, _ = run(np.random.RandomState(_SEED), pre_donor, pre_target, donor_full,
                   _LAMBDA, _EPS, _EPS)
    return w, cf


def run() -> dict:
    y, D = _panel_arrays()
    pre_donor, pre_target, donor_full = D[:_T0], y[:_T0], D
    out = {"n_donors": float(D.shape[1]), "T0": float(_T0)}
    for mech in ("output", "objective"):
        wr, pr = _reference(pre_donor, pre_target, donor_full, mech)   # skips if no clone
        wm, pm = _mlsynth(pre_donor, pre_target, donor_full, mech)
        tag = "out" if mech == "output" else "obj"
        out[f"{tag}_weights_vs_ref"] = float(np.max(np.abs(wm - wr)))
        out[f"{tag}_pred_vs_ref"] = float(np.max(np.abs(pm - pr)))
    return out


def comparison() -> dict:
    """mlsynth DPSC mechanisms vs the authors' PrivateSC on Prop 99, matched seed.

    Pairs the privatized coefficient norm and the privatized counterfactual's
    post-period mean for both mechanisms -- the quantities the mechanism actually
    releases -- side by side.
    """
    y, D = _panel_arrays()
    pre_donor, pre_target, donor_full = D[:_T0], y[:_T0], D
    post = slice(_T0, D.shape[0])
    rows = []
    for mech in ("output", "objective"):
        wr, pr = _reference(pre_donor, pre_target, donor_full, mech)
        wm, pm = _mlsynth(pre_donor, pre_target, donor_full, mech)
        tag = "OutputPerturb" if mech == "output" else "ObjectivePerturb"
        rows.append({"quantity": f"{tag}: ||private weights||",
                     "mlsynth": round(float(np.linalg.norm(wm)), 6),
                     "reference": round(float(np.linalg.norm(wr)), 6)})
        rows.append({"quantity": f"{tag}: private counterfactual mean (post)",
                     "mlsynth": round(float(np.mean(pm[post])), 6),
                     "reference": round(float(np.mean(pr[post])), 6)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "DPSC",
                         "config": {"mechanism": "output|objective",
                                    "ridge_lambda": _LAMBDA,
                                    "epsilon1": _EPS, "epsilon2": _EPS, "seed": _SEED}},
        "reference": {"impl": "srho1/dpsc PrivateSC (differentially private SC)",
                      "version": "@0be4eba"},
    }


# Validated value-for-value against srho1/dpsc @ 0be4eba on Prop 99 (lambda=10,
# eps1=eps2=50, seed 0): under the shared Mersenne-Twister stream mlsynth
# reproduces the authors' private coefficients and private counterfactual to
# solver tolerance, for both mechanisms (objective solved in closed form vs the
# authors' cvxpy).
EXPECTED = {
    "n_donors": (38.0, 0.0),
    "T0": (19.0, 0.0),
    "out_weights_vs_ref": (0.0, 1e-9),
    "out_pred_vs_ref": (0.0, 1e-8),
    "obj_weights_vs_ref": (0.0, 1e-5),
    "obj_pred_vs_ref": (0.0, 1e-5),
}
