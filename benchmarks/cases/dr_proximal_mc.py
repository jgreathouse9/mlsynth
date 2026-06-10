"""DR-proximal Path-B: the authors' doubly-robust ``normal`` Monte Carlo.

Validates mlsynth's doubly-robust (``DR``) and proximal-IPW (``PIPW``)
estimators against the data-generating process the authors ship in their
reference repo (``DR_Proximal_SC/simulation/normal``), accompanying

    Qiu, H., Shi, X., Miao, W., Dobriban, E., & Tchetgen Tchetgen, E. J.
    (2024). "Doubly robust proximal synthetic controls." Biometrics 80(2),
    ujae055.

The DGP (:func:`mlsynth.utils.proximal_helpers.simulation.simulate_dr_proximal_normal`)
has ``true.ATE = 2``, two AR(1) latent confounders, and independent error-prone
proxy blocks ``W`` (outcome proxies) and ``Z`` (treatment proxies). The case
asserts the two headline facts of the paper:

1. **Recovery + honest inference.** At ``T = 1000`` both ``DR`` and ``PIPW``
   recover the truth (mean ATT ~ 2.0) with Wald coverage near the nominal 95%.
2. **Double robustness.** When the outcome bridge is misspecified (a nonlinear
   confounding signal an intercept-free linear bridge cannot absorb), the
   outcome-only ``PI`` estimator collapses (mean ATT ~ 4.3) while ``DR`` --
   rescued by a correctly-specified treatment bridge -- stays at ~ 2.0.

  ====================  ===========  ===============
  Quantity              mlsynth      reference
  ====================  ===========  ===============
  DR mean ATT           ~2.0         2.0 (true)
  PIPW mean ATT         ~2.0         2.0 (true)
  PI under misspec      ~4.3         collapses
  DR under misspec      ~2.0         holds
  ====================  ===========  ===============

In the **just-identified** case the DR and PIPW *point* estimates coincide
exactly: the treatment bridge solves the balancing moment
:math:`\\mathbb{E}_{\\text{pre}}[q\\,(1,W)] = \\mathbb{E}_{\\text{post}}[(1,W)]`,
which makes the DR outcome-bridge correction cancel, so ``dr_bias`` equals
``pipw_bias`` here. The two still differ in **inference** -- the sandwich SEs,
and hence the Wald coverage, are not identical.

Path B (scenario 3, the authors' own DGP): the case asserts the geometry --
both estimators unbiased with reasonable coverage, and DR robust to outcome-
bridge misspecification where PI is not -- not exact Monte Carlo cells. The
estimators are driven at the array level (the just-identified GMM is closed
form) for speed.
"""
from __future__ import annotations

import warnings

import numpy as np

from mlsynth.utils.proximal_helpers.simulation import simulate_dr_proximal_normal

T = 1000
M = 60          # recovery / coverage replications
M_DR = 40       # double-robustness replications
_TRUE = 2.0


def _bandwidth(n_post: int) -> int:
    from mlsynth.utils.proximal_helpers.setup import _hac_bandwidth
    return _hac_bandwidth(n_post)


def _recovery_and_coverage(base_seed: int):
    """Mean ATT and 95% Wald coverage of DR and PIPW over ``M`` draws."""
    from mlsynth.utils.proximal_helpers.dr.estimation import estimate_dr
    from mlsynth.utils.proximal_helpers.pipw.estimation import estimate_pipw

    rng = np.random.default_rng(base_seed)
    dr_att = np.empty(M)
    pipw_att = np.empty(M)
    dr_cov = np.empty(M)
    pipw_cov = np.empty(M)
    for i in range(M):
        s = simulate_dr_proximal_normal(T=T, rng=rng)
        bw = _bandwidth(T - s.T0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, _, da, ds = estimate_dr(s.y, s.donor_outcomes, s.donor_proxies, s.T0, bw)
            _, pa, ps = estimate_pipw(s.y, s.donor_outcomes, s.donor_proxies, s.T0, bw)
        dr_att[i], pipw_att[i] = da, pa
        dr_cov[i] = abs(da - _TRUE) <= 1.96 * ds
        pipw_cov[i] = abs(pa - _TRUE) <= 1.96 * ps
    return (float(dr_att.mean()), float(dr_cov.mean()),
            float(pipw_att.mean()), float(pipw_cov.mean()))


def _double_robustness(base_seed: int):
    """Mean outcome-only PI vs DR ATT when the outcome bridge is misspecified."""
    from mlsynth.utils.proximal_helpers.dr.estimation import estimate_dr
    from mlsynth.utils.proximal_helpers.pi.estimation import estimate_pi

    rng = np.random.default_rng(base_seed)
    pi_att = np.empty(M_DR)
    dr_att = np.empty(M_DR)
    for i in range(M_DR):
        s = simulate_dr_proximal_normal(T=T, misspecify=True, rng=rng)
        bw = _bandwidth(T - s.T0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cf_pi, _, _ = estimate_pi(s.y, s.donor_outcomes, s.donor_proxies,
                                      s.T0, T - s.T0, T, bw)
            _, _, _, da, _ = estimate_dr(s.y, s.donor_outcomes, s.donor_proxies, s.T0, bw)
        pi_att[i] = float(np.mean(s.y[s.T0:] - cf_pi[s.T0:]))
        dr_att[i] = da
    return float(pi_att.mean()), float(dr_att.mean())


def run() -> dict:
    dr_mean, dr_cov, pipw_mean, pipw_cov = _recovery_and_coverage(2200)
    pi_mis, dr_mis = _double_robustness(3300)
    return {
        "dr_bias": dr_mean - _TRUE,
        "pipw_bias": pipw_mean - _TRUE,
        "dr_coverage": dr_cov,
        "pipw_coverage": pipw_cov,
        "pi_att_misspecified": pi_mis,
        "dr_att_misspecified": dr_mis,
        "dr_beats_pi_under_misspec": float(abs(dr_mis - _TRUE) < abs(pi_mis - _TRUE)),
    }


# Deterministic (seeded). Tolerances absorb the binomial / Monte Carlo noise at
# M=60 / M_DR=40. The reproduced facts: DR and PIPW recover true.ATE=2 with
# near-nominal Wald coverage, and -- the paper's headline -- DR is robust to an
# outcome-bridge misspecification that collapses the outcome-only PI estimator
# (PI ~ 4.3, DR ~ 2.0). Centres match the docs' runnable Monte Carlo.
EXPECTED = {
    "dr_bias": (0.0, 0.06),                 # DR recovers ATE=2
    "pipw_bias": (0.0, 0.06),               # PIPW recovers ATE=2
    "dr_coverage": (0.91, 0.12),            # near nominal 0.95
    "pipw_coverage": (0.99, 0.08),          # conservative (high) coverage
    "pi_att_misspecified": (4.30, 0.6),     # PI collapses under nonlinear confounding
    "dr_att_misspecified": (2.0, 0.25),     # DR holds (correct treatment bridge)
    "dr_beats_pi_under_misspec": (1.0, 0.0),
}
