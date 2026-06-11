"""SHC Path-B: latent-confounder recovery (Chen, Yang & Yang 2024, Sec 3.1).

Validates mlsynth's ``SHC`` against the paper's simulation design. The DGP
(:func:`mlsynth.utils.shc_helpers.simulation`) is a single series
``y_t = ell_t + delta_t d_t + eps_t`` with **no intervention effect**
(``delta_t = 0``), so the exercise measures how well SHC recovers the latent
time-varying confounder ``ell_t`` in the post-intervention period. The latent is a
globally C^1 curve of alternating cosine local-trends and cubic-Hermite
connectors; the treated block's shape is a convex combination of the historical
shapes (Assumption 2(b)).

With ``delta_t = 0``, both the pre-period matching error (``MSE_pre``, Eq. 31) and
the post-period prediction error against the *true* latent (``MSE_post(k)``, Eq.
38) are near zero, and ``MSE_post`` rises mildly with the horizon -- consistent
with the paper's bias bound (Proposition 1) growing with the horizon but staying
small for a smooth, regularly-recurring latent at low noise:

  ============  ===============
  Quantity      value
  ============  ===============
  MSE_pre       ~0.001
  MSE_post(1)   ~0.001
  MSE_post(25)  ~0.002
  ============  ===============

Path B (the paper's simulation): the case asserts SHC recovers the latent to
near-zero error pre and post -- the recovery property the method rests on -- not
exact cells. Deterministic (seeded).
"""
from __future__ import annotations


def run() -> dict:
    from mlsynth.utils.shc_helpers import monte_carlo_shc

    out = monte_carlo_shc(
        n_reps=5, m=25, h=4, n=25, P=10, sigma=0.1,
        w_f=(1, 0, 0, 0), regular=True, k_grid=(1, 5, 10, 25), seed=0,
    )
    mse_pre = float(out["mse_pre"])
    mp = {int(k): float(v) for k, v in out["mse_post"].items()}
    return {
        "mse_pre": mse_pre,
        "mse_post_1": mp[1],
        "mse_post_25": mp[25],
        "recovery_near_zero": float(mse_pre < 0.02 and mp[1] < 0.02 and mp[25] < 0.02),
        "post_at_least_pre": float(mp[25] >= 0.5 * mse_pre),
    }


# Deterministic (seeded). Tolerances absorb the Monte Carlo noise at n_reps=5.
# Reproduces Chen-Yang-Yang Sec 3.1 (Regular-l, sigma=0.1): SHC recovers the
# latent confounder to ~1e-3 matching error pre and post, with the post-period
# prediction error rising only mildly with the horizon.
EXPECTED = {
    "mse_pre": (0.0011, 0.01),
    "mse_post_1": (0.0011, 0.01),
    "mse_post_25": (0.0017, 0.015),
    "recovery_near_zero": (1.0, 0.0),
    "post_at_least_pre": (1.0, 0.0),
}
