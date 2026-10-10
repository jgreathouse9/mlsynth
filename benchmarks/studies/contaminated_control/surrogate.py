"""Correcting contamination through the delivery profile instead of the cross-section.

Every other arm works across markets: find the contaminated one, rebuild it,
or down-weight it. This one works across time. A marketing experiment observes
a second metric in every market -- impressions, reach, delivered spend -- and
that metric tracks the campaign intensity that drives the effect. Liu, Tchetgen
Tchetgen and Varjao (2024) call such a metric a surrogate: it loads on the same
factors as the causal effect.

Write the campaign's intensity at time t as rho_t. If the effect is driven by
delivery, the post-period gap between the synthetic treated and synthetic
control aggregates is

    g_t = theta * rho_t + delta_t - v_k * gamma_t

for delta_t the design's own per-period fit error and gamma_t the contamination
on market k*. Regressing g_t on a constant and rho_t gives

    b_hat = theta + [ cov(delta, rho) - v_k cov(gamma, rho) ] / var(rho)

so the contamination leaves the slope exactly when its time profile is
uncorrelated with the delivery profile, and a contamination that is constant
over the post-period is absorbed by the intercept at no cost. The contaminated
market is never named. What replaces that knowledge is a claim about shapes.

Two failure modes follow and both are swept here: contamination that tracks the
campaign, and a flat campaign with no variation in rho to project on. The
design's own fit error contributes a second term that does not vanish with the
contamination, so the arm also measures what the projection costs when there is
nothing to correct.

    python surrogate.py 60 results/surrogate.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from mlsynth.utils.solvers.simplex import simplex_lstsq

from dgps import DGPS
from run import design

# panels with enough post-period to regress a profile on
PANELS = ("rank_shift_ok", "rank_shift_dormant", "fdid_ar1_rho.7",
          "hsc_shared_trend", "pangeo_seasonal")
CORRS = (0.0, 0.25, 0.5, 0.75, 1.0)      # corr(gamma_t, rho_t)
PI_MULT = 2.0                             # contamination level, in panel scales
VAR_FRAC = 1.0                            # its time variation, relative to that level
NOISE = 0.10                              # measurement noise on impressions


def flighted(n, rng, flat=False):
    """Campaign intensity: flighted weeks on and off, or a near-flat always-on.

    The flat arm keeps a little jitter on purpose. A perfectly constant profile
    leaves the correlation with it undefined; a nearly constant one leaves the
    projection weakly identified, which is the case a real always-on campaign
    presents.
    """
    if flat:
        return np.ones(n) * rng.uniform(0.97, 1.03, size=n)
    rho = 0.35 + 0.65 * ((np.arange(n) % 3) != 2).astype(float)
    return rho * rng.uniform(0.85, 1.15, size=n)


def profile_with_corr(rho, target, rng):
    """A unit-variance profile whose correlation with rho is `target`."""
    r = (rho - rho.mean()) / (rho.std() + 1e-12)
    z = rng.normal(size=len(rho))
    z = z - z.mean()
    z = z - (z @ r) / (r @ r) * r          # orthogonal to rho
    z = z / (z.std() + 1e-12)
    g = target * r + np.sqrt(max(0.0, 1 - target ** 2)) * z
    return g / (g.std() + 1e-12)


def replication(name, seed):
    YN, _, T0 = DGPS[name](seed)
    T, J = YN.shape
    Tp = T - T0
    w, v = design(YN, T0, max(2, J // 6))
    treated = np.flatnonzero(w > 1e-8)
    controls = np.flatnonzero(v > 1e-12)
    kstar = int(np.argmax(v)); vk = float(v[kstar])
    clean = np.array([j for j in controls if j != kstar])
    scale = float(np.median(YN[:T0].std(axis=0)))
    post = slice(T0, T)

    rows = []
    for flat in (False, True):
        rows += _one(name, seed, YN, T0, Tp, J, w, v, treated, controls,
                     kstar, vk, clean, scale, post, flat)
    return rows


def _one(name, seed, YN, T0, Tp, J, w, v, treated, controls, kstar, vk,
         clean, scale, post, flat):
    """One campaign shape, reusing the design solved once per replication."""
    rng = np.random.default_rng(50_000 + seed + (7 if flat else 0))
    # campaign intensity, and the effect it drives
    rho = flighted(Tp, rng, flat=flat)
    theta = scale                                   # effect per unit intensity
    tau_t = theta * rho
    true_att = float(tau_t.mean())

    # impressions: intensity times market size, observed with noise, treated only
    size = np.abs(YN[:T0].mean(axis=0)) + 1.0
    imp = np.zeros((Tp, J))
    for j in treated:
        imp[:, j] = rho * size[j] * rng.lognormal(0.0, NOISE, size=Tp)
    rho_hat = imp[:, treated].sum(axis=1)
    rho_hat = rho_hat / rho_hat.mean()              # observed delivery profile

    rows = []
    for corr in CORRS:
        # a level shift plus time variation: the level is what biases the
        # average, the variation is what the projection can or cannot remove
        gam = PI_MULT * scale * (1.0 + VAR_FRAC * profile_with_corr(rho, corr, rng))
        Y = YN.copy()
        Y[post, treated] += tau_t[:, None]
        Yc = Y.copy()
        Yc[post, kstar] += gam

        A = Yc @ w
        g = A[post] - Yc[post] @ v                  # contaminated gap
        g_clean = (Y @ w)[post] - Y[post] @ v

        naive = float(g.mean())
        oracle = float(g_clean.mean())

        # repair that is told the market, for reference
        fit = simplex_lstsq(Yc[:T0, clean], Yc[:T0, kstar])
        Yi = Yc.copy(); Yi[post, kstar] = Yc[post][:, clean] @ fit
        known = float(((Yi @ w)[post] - Yi[post] @ v).mean())

        # surrogate projection: gap on (1, delivery profile)
        X = np.column_stack([np.ones(Tp), rho_hat])
        b = np.linalg.lstsq(X, g, rcond=None)[0]
        surrogate = float(b[1] * rho_hat.mean())
        b_clean = np.linalg.lstsq(X, g_clean, rcond=None)[0]
        surrogate_clean = float(b_clean[1] * rho_hat.mean())

        rows.append(dict(dgp=name, seed=seed, flat=flat, corr=corr, vk=vk,
                         Tpost=Tp, true_att=true_att, oracle=oracle,
                         naive=naive, known=known, surrogate=surrogate,
                         surrogate_clean=surrogate_clean,
                         rho_sd=float(rho_hat.std())))
    return rows


def main(reps, out):
    warnings.filterwarnings("ignore")
    rows, fails = [], {}
    for name in PANELS:
        for seed in range(reps):
            try:
                rows += replication(name, seed)
            except Exception as exc:                   # noqa: BLE001
                fails[name] = fails.get(name, 0) + 1
                if fails[name] <= 2:
                    print(f"{name} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
        print(f"done {name} ({fails.get(name, 0)} failures)", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows to {out}; failures={fails}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
