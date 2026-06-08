"""Path B benchmark: LEXSCM recovers the planted effect on the Abadie-Zhao DGP.

Reproduces the central finding of the synthetic-experimental-design simulation
study (Abadie & Zhao 2026, Sec. 5; the lexicographic solve is Vives-i-Bastida
2022): a synthetic-control *design* chosen before the intervention recovers the
average treatment effect with mean-absolute-error far below the effect's own
scale, and the error *shrinks* as more units are admitted to the treated group
(the paper: "performance metrics improve substantially when allowing m = 2 or
m = 3" over the single-treated-unit design).

Exactly the paper's Section 5 setup
-----------------------------------
``J = 15`` units, ``R = 7`` observed covariates, ``F = 11`` unobserved,
``T = 30`` periods with ``T0 = 25`` pre-intervention (``T_E = 20`` fitting + 5
blank, 5 experimental), ``sigma^2 = 1``, uniform population weights, every unit
a treatment candidate. The MAE is taken over the experimental periods exactly as
the paper defines it (eq. in Sec. 5.2.1).

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

Running at ``m = 1`` (the single-treated-unit design the paper flags as worst)
and ``m = 2`` shows the paper's monotonicity: more treated units -> lower error.

Provenance
----------
* DGP: :func:`mlsynth.utils.marex_helpers.simulation.generate_marex_sample`
  -- the Abadie & Zhao (2026) baseline linear-factor model (eqs 12a/12b) with
  the Section-5 parameters above (the generator's defaults are R=7, F=11).
* Headline: paper Table 2 -- the design's MAE is a small fraction of the effect
  scale (here ratio ~0.24 at m=1, ~0.16 at m=2), decreasing in the number of
  treated units.
"""
from __future__ import annotations

import warnings

import numpy as np

# Abadie & Zhao (2026) Section 5 design.
J, R, F = 15, 7, 11
T, T0 = 30, 25
T_E = 20                       # fitting periods (frac_E = T_E / T0); 5 blank, 5 post
SIGMA = 1.0
SEED = 0
N_REPS = 4
M_VALUES = (1, 2)


def _design_mae(sample, m: int):
    """MAE of the LEXSCM design's estimator vs the true effect, and the scale."""
    import pandas as pd

    from mlsynth import LEXSCM

    Jn = sample.Y_N.shape[0]
    pre = pd.DataFrame([
        {"unit": f"u{j:02d}", "time": t, "y": float(sample.Y_N[j, t]), "candidate": 1}
        for j in range(Jn) for t in range(sample.T0)
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = LEXSCM({
            "df": pre, "outcome": "y", "unitid": "unit", "time": "time",
            "candidate_col": "candidate", "m": m, "frac_E": T_E / T0,
            "top_K": 3, "n_sims": 40, "n_post_grid": [5], "mde_horizon": "late",
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

    ratios, maes = {}, {}
    for m in M_VALUES:
        rng = np.random.default_rng(SEED)        # same draws across m (paired)
        mlist, slist = [], []
        for _ in range(N_REPS):
            sample = generate_marex_sample(J=J, R=R, F=F, T=T, T0=T0, sigma=SIGMA, rng=rng)
            mae, scale = _design_mae(sample, m)
            mlist.append(mae)
            slist.append(scale)
        maes[m] = float(np.mean(mlist))
        ratios[m] = float(np.mean(mlist) / np.mean(slist))

    return {
        "sim_ratio_m1": ratios[1],
        "sim_ratio_m2": ratios[2],
        # 1.0 iff the design recovers the effect (MAE well below its scale).
        "recovers_effect": float(ratios[1] < 0.35 and ratios[2] < 0.35),
        # 1.0 iff error shrinks with more treated units (paper's monotonicity).
        "error_decreases_with_m": float(maes[1] > maes[2]),
    }


# Deterministic (fixed DGP seed; draws paired across m). The binding facts are
# `recovers_effect == 1` (MAE far below the effect scale) and
# `error_decreases_with_m == 1` (Table-2 monotonicity from the single-treated
# design to m=2); the per-m ratios are pinned with bands wide enough to absorb
# numpy-RNG / solver drift.
EXPECTED = {
    "sim_ratio_m1": (0.244, 0.12),
    "sim_ratio_m2": (0.164, 0.10),
    "recovers_effect": (1.0, 0.0),
    "error_decreases_with_m": (1.0, 0.0),
}
