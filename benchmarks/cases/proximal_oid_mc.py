"""PIOID Path-B: the authors' linear interactive-fixed-effects Monte Carlo.

Validates mlsynth's over-identified proximal-inference estimator (``PIOID``)
against the linear simulation the authors ship in their manuscript replication
(``KenLi93/proximal_sc_manuscript``, ``simulation/run_sim_linear_est.R``),
accompanying

    Shi, X., Li, K. T., Yu, Y., Miao, W., Kuchibhotla, A. K., Hu, W., &
    Tchetgen Tchetgen, E. J. (2026). "Theory for Identification and Inference
    with Synthetic Controls: A Proximal Causal Inference Framework." JASA.

The DGP (:func:`mlsynth.utils.proximal_helpers.simulation.simulate_pioid_linear`)
is a linear factor model ``Y_it = U_i' lambda_t + eps_it`` with ``true.beta = 2``:
one treated unit and ``n_units - 1`` controls, split into treatment proxies ``Z``
and donor outcomes ``W`` that share the (diagonal) loading structure but carry
*independent* idiosyncratic noise. That shared error-in-variables structure is
the point: a naive synthetic control regresses the treated series on the noisy
donors ``W`` and inherits an attenuation-style bias, while PIOID instruments
``W`` with ``Z`` and stays consistent.

The case asserts the two headline facts of the manuscript's linear table:

1. Recovery + honest inference. PIOID recovers ``true.beta = 2`` (mean ATT ~ 2.0,
   bias ~ 0) with over-identified GMM/Newey-West Wald coverage near the nominal
   95%.
2. Proximal beats naive SC. A simplex-constrained synthetic control fit to the
   same donors ``W`` carries a sizeable positive bias (~ +0.55), and PIOID is
   strictly less biased on every seed.

  =====================  ===========  ===============
  Quantity               mlsynth      reference
  =====================  ===========  ===============
  PIOID mean ATT         ~2.0         2.0 (true.beta)
  PIOID 95% coverage     ~0.94        ~0.95 (nominal)
  naive SC bias          ~ +0.55      biased (EIV)
  PIOID less biased      True         yes
  =====================  ===========  ===============

Path B (the authors' own DGP): the case asserts the geometry -- PIOID unbiased
with near-nominal coverage and strictly less biased than the naive SC it is
meant to correct -- not exact Monte Carlo cells. The estimators are driven at
the array level (:func:`..pi.overid.estimate_pi_overid` for the over-identified
2SLS bridge, :func:`..bilevel.active_set.solve_simplex_qp` for the simplex SC)
for speed, so the case is cheap enough to run under ``--all``.

The default grid is the paper's ``n_units = 7``, ``t0 = 80`` (so ``T = 160``,
equal pre/post), unconstrained treated loading, stationary factors, i.i.d.
errors at ``mysd = 1.5``. ``M = 100`` replications keep the case a fast
regression guard; the manuscript's own tables use ~4000 reps, which reproduce
the published precision at ~3 s array-level but add nothing the geometry here
does not already pin. The GMM HAC lag is the manuscript's Newey-West ``q = 10``.
"""
from __future__ import annotations

import warnings

import numpy as np

from mlsynth.utils.proximal_helpers.simulation import simulate_pioid_linear

_TRUE = 2.0
_HAC = 10        # manuscript's Newey-West lag
M = 100          # recovery / coverage replications


def _mc(base_seed: int):
    """Mean ATT + 95% coverage of PIOID, and the naive-SC bias, over ``M`` draws."""
    from mlsynth.utils.proximal_helpers.pi.overid import estimate_pi_overid
    from mlsynth.utils.bilevel.active_set import solve_simplex_qp

    rng = np.random.default_rng(base_seed)
    pioid_att = np.empty(M)
    pioid_cov = np.empty(M)
    sc_att = np.empty(M)
    for i in range(M):
        s = simulate_pioid_linear(
            n_units=7, t0=80, true_att=_TRUE, u_setting="unconstrained",
            dist_lambda="stationary", dist_epsilon="iid", rng=rng)
        T = 2 * s.T0
        n_post = T - s.T0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cf, _alpha, se = estimate_pi_overid(
                s.y, s.donor_outcomes, s.donor_proxies, s.T0, n_post, T, _HAC)
            # Naive SCM: simplex-constrained fit of the treated series on the
            # noisy donors W (min ||W w - y||^2 on the pre-period, w on the
            # simplex) -- the estimator PIOID is meant to de-bias.
            Wp, Yp = s.donor_outcomes[:s.T0], s.y[:s.T0]
            w_sc = solve_simplex_qp(Wp, Yp)
        tau = float(np.mean((s.y - cf)[s.T0:]))
        pioid_att[i] = tau
        pioid_cov[i] = abs(tau - _TRUE) <= 1.96 * se if np.isfinite(se) else np.nan
        sc_att[i] = float(np.mean((s.y - s.donor_outcomes @ w_sc)[s.T0:]))
    return (float(pioid_att.mean()), float(np.nanmean(pioid_cov)),
            float(sc_att.mean()))


def run() -> dict:
    pioid_mean, pioid_cov, sc_mean = _mc(2026)
    pioid_bias = pioid_mean - _TRUE
    sc_bias = sc_mean - _TRUE
    return {
        "pioid_att": pioid_mean,
        "pioid_bias": pioid_bias,
        "pioid_coverage": pioid_cov,
        "naive_sc_att": sc_mean,
        "naive_sc_bias": sc_bias,
        "pioid_less_biased_than_sc": float(abs(pioid_bias) < abs(sc_bias)),
    }


# Deterministic (seeded at base 2026). Tolerances absorb the Monte Carlo noise at
# M=100 (across ten base seeds: PIOID bias in [-0.07, +0.19], coverage in
# [0.87, 0.97], naive-SC bias in [+0.52, +0.58]). The reproduced facts: PIOID
# recovers true.beta=2 with near-nominal over-identified GMM coverage, and is
# strictly less biased than the simplex SC fit to the same error-prone donors.
EXPECTED = {
    "pioid_bias": (0.0, 0.25),                  # PIOID recovers true.beta=2
    "pioid_coverage": (0.93, 0.10),             # near nominal 0.95
    "naive_sc_bias": (0.55, 0.15),              # naive SC biased by EIV in the donors
    "pioid_less_biased_than_sc": (1.0, 0.0),
}
