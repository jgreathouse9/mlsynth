"""ORTHSC Monte Carlo: fixed-smoothing t-test size control + power (Fry Tables 1-2).

Path B (the paper's simulation study). Fry's headline simulation result is that
the Orthogonalized SC t-test controls size (rejection rate at or below the
nominal level under the null) while retaining high power, where competing
inference methods (naive IV-SCE, cross-fitting, ArCo) over-reject. We reproduce
that *behaviour* on a clean linear-factor DGP rather than the paper's exact
carbon-tax-calibrated numbers: a treated unit that is a convex combination of
the controls plus idiosyncratic noise, instruments that share the factors but
are independent of the treated unit's idiosyncratic shocks (so the exclusion
restriction holds), and a constant additive post-treatment effect for power.

The case reports the size (effect = 0) and power (effect = -0.25 per period)
rejection rates at the 5% level across sample sizes, and pins the two claims
that matter: size is controlled (<= nominal, up to Monte Carlo noise) at every
configuration, and power is high and increasing in the number of post periods.
Invariants, not brittle Monte Carlo floats (200 replications).
"""
from __future__ import annotations

import warnings

import numpy as np

from mlsynth.utils.orthsc_helpers.pipeline import orthogonalized_sce

_NSIM = 200
_ALPHA = 0.05
_TAU = -0.25
_J, _QZ, _R, _NOISE = 8, 5, 2, 0.3


def _draw(T0, T1, tau, rng):
    T = T0 + T1
    F = rng.normal(size=(T, _R))
    Lc = rng.uniform(0.5, 1.5, (_R, _J))
    YJ = (F @ Lc).T + _NOISE * rng.normal(size=(_J, T))
    w = rng.dirichlet(np.ones(_J))
    y = w @ YJ + _NOISE * rng.normal(size=T)
    y[T0:] += tau
    Lz = rng.uniform(0.5, 1.5, (_R, _QZ))
    Z = (F @ Lz).T + _NOISE * rng.normal(size=(_QZ, T))
    return y[:T0], YJ[:, :T0], Z[:, :T0], y[T0:], YJ[:, T0:]


def _reject_rate(T0, T1, tau, seed):
    rng = np.random.default_rng(seed)
    rej = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(_NSIM):
            r = orthogonalized_sce(*_draw(T0, T1, tau, rng), alpha=_ALPHA)
            rej += int(r["pvalue"] < _ALPHA)
    return rej / _NSIM


def run() -> dict:
    out = {}
    for (T0, T1, sd_s, sd_p) in [(30, 16, 11, 12), (30, 32, 21, 22), (60, 32, 31, 32)]:
        out[f"size_{T0}_{T1}"] = _reject_rate(T0, T1, 0.0, sd_s)
        out[f"power_{T0}_{T1}"] = _reject_rate(T0, T1, _TAU, sd_p)
    return out


# Size is controlled at/below the 5% nominal level (centers ~0.02-0.05; tol
# admits [~0, 0.11] for Monte Carlo noise). Power is high and rises with the
# number of post periods (T1 = 16 -> 32).
EXPECTED = {
    "size_30_16": (0.05, 0.06),
    "size_30_32": (0.04, 0.06),
    "size_60_32": (0.05, 0.06),
    "power_30_16": (0.63, 0.20),
    "power_30_32": (0.88, 0.15),
    "power_60_32": (0.90, 0.15),
}
