"""Path B benchmark: LEXSCM recovers the planted effect on the Abadie-Zhao DGP.

Reproduces the central finding of the synthetic-experimental-design simulation
study (Abadie & Zhao 2026, Sec. 5; the lexicographic solve is Vives-i-Bastida
2022): a synthetic-control *design* chosen before the intervention recovers the
average treatment effect with mean-absolute-error far below the effect's own
scale, and the error *shrinks* as more units are admitted to the treated group.

Design then realize (the experimental-design loop)
--------------------------------------------------
For each draw from the paper's linear-factor DGP:

1. LEXSCM picks ``m`` treated units (and their synthetic control) from the
   pre-period untreated outcomes ``Y^N`` -- the design is fixed *before* any
   intervention.
2. The experiment realizes the treated potential outcome ``Y^I`` on exactly
   those units in the post-period.
3. The design's estimator ``tau_hat = w'Y_obs - v'Y_obs`` is compared to the
   true effect ``tau``; we report the MAE relative to the effect scale.

Running this at two cardinalities (``m = 2`` and ``m = 4``) shows the paper's
monotonicity: more treated units -> lower error.

Provenance
----------
* DGP: :func:`mlsynth.utils.marex_helpers.simulation.generate_marex_sample`
  -- the Abadie & Zhao (2026) baseline linear-factor model (eqs 12a/12b),
  shared across the MAREX family.
* Headline: paper Table 2 -- the design's MAE is a small fraction of the effect
  scale (here ratio ~0.17 at m=2, ~0.09 at m=4), decreasing in the number of
  treated units.
"""
from __future__ import annotations

import warnings

import numpy as np

J, T, T0 = 12, 30, 25
SEED = 0
N_REPS = 4
M_VALUES = (2, 4)


def _design_mae(sample, m: int):
    """MAE of the LEXSCM design's estimator vs the true effect, and the scale."""
    from mlsynth import LEXSCM
    import pandas as pd

    Jn, Tn = sample.Y_N.shape
    pre = pd.DataFrame([
        {"unit": f"u{j:02d}", "time": t, "y": float(sample.Y_N[j, t]), "candidate": 1}
        for j in range(Jn) for t in range(sample.T0)
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = LEXSCM({
            "df": pre, "outcome": "y", "unitid": "unit", "time": "time",
            "candidate_col": "candidate", "m": m, "top_K": 3,
            "n_sims": 40, "n_post_grid": [5], "mde_horizon": "late",
            "verbose": False,
        }).fit()
    bc = res.best_candidate
    w = np.zeros(Jn)
    for label, wj in bc.treated_weight_dict.items():
        w[int(str(label)[1:])] = wj          # treated representativeness weights
    v = np.asarray(bc.weights.control, dtype=float)  # full-length control weights
    treated = np.where(w > 1e-8)[0]

    Y_obs = sample.Y_N.copy()
    Y_obs[treated, sample.T0:] = sample.Y_I[treated, sample.T0:]   # realize Y^I
    tau_hat = w @ Y_obs[:, sample.T0:] - v @ Y_obs[:, sample.T0:]
    mae = float(np.mean(np.abs(tau_hat - sample.tau[sample.T0:])))
    scale = float(np.mean(np.abs(sample.tau[sample.T0:])))
    return mae, scale


def run() -> dict:
    from mlsynth.utils.marex_helpers.simulation import generate_marex_sample

    ratios = {}
    maes = {}
    for m in M_VALUES:
        rng = np.random.default_rng(SEED)        # same draws across m (paired)
        mlist, slist = [], []
        for _ in range(N_REPS):
            sample = generate_marex_sample(J=J, T=T, T0=T0, rng=rng)
            mae, scale = _design_mae(sample, m)
            mlist.append(mae)
            slist.append(scale)
        maes[m] = float(np.mean(mlist))
        ratios[m] = float(np.mean(mlist) / np.mean(slist))

    return {
        "sim_ratio_m2": ratios[2],
        "sim_ratio_m4": ratios[4],
        # 1.0 iff the design recovers the effect (MAE well below its scale).
        "recovers_effect": float(ratios[2] < 0.3 and ratios[4] < 0.3),
        # 1.0 iff error shrinks with more treated units (paper's monotonicity).
        "error_decreases_with_m": float(maes[2] > maes[4]),
    }


# Deterministic (fixed DGP seed; draws paired across m). The binding facts are
# `recovers_effect == 1` (MAE far below the effect scale) and
# `error_decreases_with_m == 1` (Table-2 monotonicity); the per-m ratios are
# pinned with bands wide enough to absorb numpy-RNG / solver drift.
EXPECTED = {
    "sim_ratio_m2": (0.171, 0.10),
    "sim_ratio_m4": (0.089, 0.08),
    "recovers_effect": (1.0, 0.0),
    "error_decreases_with_m": (1.0, 0.0),
}
