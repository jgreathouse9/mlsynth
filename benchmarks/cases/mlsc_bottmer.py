"""Cross-validation: mlsynth ``MLSC`` vs the author's ``mlSC_estimator``.

Matches mlsynth's multi-level SC engine against the reference implementation of
Bottmer (2024/2025) -- the ``multi-level-sc-estimator`` repository
(https://github.com/leabottmer/multi-level-sc-estimator) -- cell by cell on the
hierarchical-factor DGP from the author's README. Both estimators solve the same
penalized state-county program (classical-SC warm start, block penalty matrix
``Q``, heuristic penalty :math:`\\lambda = 2\\sigma^2_\\varepsilon/\\sigma^2_y`)
with ``cvxpy`` + ``SCS``, so they agree to solver precision.

The README DGP is :func:`mlsynth.utils.mlsc_helpers.simulation.simulate_mlsc_sample`
(``N_s = 10`` states, ``C_s = 10`` counties, ``T = 20``), which feeds *both*
sides: the reference takes the raw ``(data_s, data_c, idx, n_c, t, w_c)`` arrays,
mlsynth takes the equivalent long ``df_agg`` / ``df_disagg`` frames. No treatment
effect is injected, so the true ATT is zero and Path B is an unbiasedness check.

* **Path A** (one draw, ``rng=default_rng(42)``): the selected penalty and the
  reported ATT match the reference -- ``dtau`` is solver noise on the ATT,
  ``dlambda`` is machine precision on the penalty.
* **Path B** (``M = 200`` draws): both estimators are unbiased for the true zero
  ATT and stay in near-machine agreement throughout (worst-case ``|dtau|``).

Provenance / scenario
---------------------
* Full repo (scenario 3): cross-validation is mandatory and done here.
* Reference: ``multi-level-sc-estimator`` pinned at commit ``0fb2639``; solves
  with ``cvxpy`` + ``SCS`` (open source). Its unused top-level ``import ray`` is
  stubbed by the clone helper.
* The case **skips gracefully** (``BenchmarkSkipped``) when the reference clone
  or its deps are unavailable.
"""
from __future__ import annotations

import warnings

import numpy as np

SEED_A = 42      # Path A: the README's exact panel
M = 200          # Path B: independent draws


def _ml_fit(sample):
    from mlsynth import MLSC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MLSC({
            "df_agg": sample.df_agg, "df_disagg": sample.df_disagg,
            "outcome": "y", "time": "time", "treat": "treated",
            "unitid_agg": "state", "unitid_disagg": "county",
            "agg_id": "state", "lambda_est": "heuristic",
            "display_graphs": False,
        }).fit()


def run() -> dict:
    from benchmarks.compare import BenchmarkSkipped
    from benchmarks.reference.clone_mlsc import import_mlsc
    from mlsynth.utils.mlsc_helpers.simulation import simulate_mlsc_sample

    ref = import_mlsc()
    mlSC_estimator = ref.mlSC_estimator

    def ref_tau_lambda(sample):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                tau, lam, _ = mlSC_estimator(
                    sample.data_s, sample.data_c, sample.idx,
                    sample.n_c, sample.t, sample.w_c, lambda_est="heuristic",
                )
            except Exception as exc:  # pragma: no cover - reference solve failure
                raise BenchmarkSkipped(f"reference mlSC_estimator failed: {exc}")
        return float(tau), float(lam)

    # --- Path A: the README panel, value for value -----------------------
    s = simulate_mlsc_sample(rng=np.random.default_rng(SEED_A))
    tau_ref_a, lam_ref_a = ref_tau_lambda(s)
    res_a = _ml_fit(s)
    out = {
        "path_a_dtau": float(abs(res_a.att - tau_ref_a)),
        "path_a_dlambda": float(abs(res_a.design.lambda_used - lam_ref_a)),
    }

    # --- Path B: M independent draws, unbiasedness + agreement -----------
    tau_ref = np.empty(M)
    tau_ml = np.empty(M)
    for i in range(M):
        si = simulate_mlsc_sample(rng=np.random.default_rng(i))
        tau_ref[i], _ = ref_tau_lambda(si)
        tau_ml[i] = float(_ml_fit(si).att)

    out["path_b_max_dtau"] = float(np.abs(tau_ml - tau_ref).max())
    out["path_b_mean_ml"] = float(tau_ml.mean())
    out["path_b_rmse_ml"] = float(np.sqrt((tau_ml ** 2).mean()))
    out["path_b_drmse"] = float(
        abs(np.sqrt((tau_ml ** 2).mean()) - np.sqrt((tau_ref ** 2).mean()))
    )
    return out


# Two independent implementations of the same penalized program on a shared DGP.
# The cross-validation cells (dtau/dlambda/max_dtau/drmse) pin the agreement to
# SCS solver precision; the unbiasedness cell pins the deterministic M=200 mean,
# which sits within one MC SE (sigma/sqrt(200) ~ 0.012) of the true zero ATT.
EXPECTED = {
    "path_a_dtau": (0.0, 5e-3),        # measured 1.5e-6 (ATT solver noise)
    "path_a_dlambda": (0.0, 1e-6),     # measured 4.4e-16 (machine precision)
    "path_b_max_dtau": (0.0, 5e-3),    # measured 5.76e-4 (worst of 200)
    "path_b_mean_ml": (-0.0103, 0.02), # ~0; |bias| < 1 MC SE
    "path_b_rmse_ml": (0.1662, 0.02),  # matches reference RMSE
    "path_b_drmse": (0.0, 2e-3),       # ml vs ref RMSE agreement
}
