"""SPSC Path-B: the authors' interactive-fixed-effects Monte Carlo.

Validates mlsynth's Single Proxy Synthetic Control against the data-generating
process the SPSC authors ship in their reference R package README (the *"Toy
Example from Interactive Fixed Effect Models,"* ``github.com/qkrcks0218/SPSC``),
accompanying

    Park, C., & Tchetgen Tchetgen, E. J. (2025). "Single Proxy Synthetic
    Control." Journal of Causal Inference 13(1), 20230079.

The DGP (:func:`mlsynth.utils.proximal_helpers.simulation.simulate_spsc_ifem`)
gives the donor pool a deterministic baseline **trend**, so the untreated
trajectories drift. That is exactly the regime the paper highlights: the
detrended estimator **SPSC-DT** stays (near-)unbiased with close-to-nominal
coverage, while the un-detrended **SPSC-NoDT** -- which forces a constant gap
through a trending counterfactual -- under-covers because its sandwich SE cannot
see the trend misspecification. Both recover the true ``ATT = 3`` essentially
without bias (the bridge is identified by the single proxy regardless of the
trend); detrending is what buys honest inference.

  ================  ===============  ===============  =================
  Quantity          SPSC-DT          SPSC-NoDT        target
  ================  ===============  ===============  =================
  bias (true 3.0)   ~0.00            ~0.00            ~unbiased
  95% coverage      ~0.93            ~0.76            DT near nominal
  ================  ===============  ===============  =================

Path B (scenario 3, the authors' own DGP): the case asserts the geometry -- both
estimators unbiased, DT covers near nominal, and **DT covers strictly better than
NoDT under the trend** -- not exact Monte Carlo cells.
"""
from __future__ import annotations

import warnings

import numpy as np

from mlsynth.utils.proximal_helpers.simulation import simulate_spsc_ifem

M = 60          # Monte Carlo replications (runtime vs. binomial noise trade-off)
_TRUE_ATT = 3.0


def _mc(detrend: bool, base_seed: int):
    """Mean ATT, bias, and 95% Wald coverage of SPSC over ``M`` draws."""
    from mlsynth.utils.proximal_helpers.spsc.estimation import estimate_spsc

    rng = np.random.default_rng(base_seed)
    atts = np.empty(M)
    covers = np.empty(M)
    for i in range(M):
        s = simulate_spsc_ifem(rng=rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, att, se, _, _, _, _ = estimate_spsc(s.y, s.donors, s.T0,
                                                      detrend=detrend, spline_df=5)
        atts[i] = att
        lo, hi = att - 1.96 * se, att + 1.96 * se
        covers[i] = 1.0 if lo <= s.true_att <= hi else 0.0
    return float(atts.mean()), float(covers.mean())


def run() -> dict:
    dt_mean, dt_cover = _mc(detrend=True, base_seed=4100)
    nodt_mean, nodt_cover = _mc(detrend=False, base_seed=4100)
    return {
        "spsc_dt_bias": dt_mean - _TRUE_ATT,
        "spsc_nodt_bias": nodt_mean - _TRUE_ATT,
        "spsc_dt_coverage": dt_cover,
        "spsc_nodt_coverage": nodt_cover,
        "dt_covers_better": float(dt_cover > nodt_cover),
    }


# Deterministic (seeded). Tolerances absorb the binomial Monte Carlo noise at
# M=60 (a coverage SE ~ sqrt(.9*.1/60) ~ 0.04). The reproduced facts are: SPSC
# recovers the true ATT=3 essentially without bias under either detrending
# choice, SPSC-DT covers near the nominal 95%, and -- the paper's point -- DT
# covers strictly better than NoDT once the donor pool trends.
EXPECTED = {
    "spsc_dt_bias": (0.0, 0.04),          # detrended: unbiased
    "spsc_nodt_bias": (0.0, 0.05),        # un-detrended: also ~unbiased in point
    "spsc_dt_coverage": (0.93, 0.10),     # near nominal 0.95
    "spsc_nodt_coverage": (0.76, 0.18),   # under-covers under the trend
    "dt_covers_better": (1.0, 0.0),       # detrending buys honest inference
}
