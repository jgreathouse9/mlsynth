"""The surrogate projection when the response is not linear in delivery.

The surrogate arm imposes an effect proportional to delivery at the same
instant. Real response has carryover, saturation and lag, so the effect is a
transform of the delivery path and not the path itself. Writing that transform
as m_t = Sat(Adstock(rho))_{t-L}, normalised to average one, the effect is
theta * m_t and the average effect is theta whatever the transform.

The analyst regressing the gap on raw delivery is then projecting onto the
wrong regressor. The slope picks up only the part of m that delivery explains
linearly, and the rest joins the error, so a misspecified shape puts bias into
the estimate through the same channel the contamination uses -- and it does so
with no contamination present at all.

Three regressors are compared against that. Raw delivery, which is the
surrogate arm's choice. The true transform, which no analyst has but which
bounds what the channel can deliver. And a transform whose carryover,
saturation and lag are picked by grid search on fit, which is what an analyst
would actually do.

The prediction for the fitted one is that it is not free. The grid search
maximises fit against a gap that contains the contamination, so the
contamination gets a say in which transform is chosen. It should therefore
track the oracle transform when the profiles are uncorrelated and fall away
from it faster than the oracle does as they align.

    python response_shape.py 60 results/response_shape.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from mlsynth.utils.solvers.simplex import simplex_lstsq

from dgps import DGPS
from run import design
from surrogate import PANELS, CORRS, PI_MULT, VAR_FRAC, NOISE, flighted, profile_with_corr

# phi: carryover.  kappa: saturation (None = linear).  lag: periods of delay.
REGIMES = {
    "linear":     dict(phi=0.0, kappa=None, lag=0),
    "carryover":  dict(phi=0.6, kappa=None, lag=0),
    "saturating": dict(phi=0.0, kappa=1.0,  lag=0),
    "lagged":     dict(phi=0.0, kappa=None, lag=1),
    "realistic":  dict(phi=0.6, kappa=1.0,  lag=1),
}
GRID_PHI = (0.0, 0.25, 0.5, 0.75)
GRID_KAPPA = (None, 2.0, 1.0, 0.5)
GRID_LAG = (0, 1)


def adstock(rho, phi):
    out = np.empty_like(rho, dtype=float)
    acc = 0.0
    for t, r in enumerate(rho):
        acc = r + phi * acc
        out[t] = acc
    return out


def shape(rho, phi, kappa, lag):
    """Delivery path -> effect driver, normalised to average one."""
    a = adstock(np.asarray(rho, float), phi)
    s = a if kappa is None else a / (a + kappa * a.mean())
    if lag:
        s = np.concatenate([np.full(lag, s[0]), s[:-lag]])
    m = s / s.mean()
    return m


def project(g, m):
    """OLS of the gap on (1, m). With m averaging one the slope is the effect."""
    X = np.column_stack([np.ones(len(m)), m])
    b, *_ = np.linalg.lstsq(X, g, rcond=None)
    resid = g - X @ b
    ss = float(((g - g.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss if ss > 0 else -np.inf
    return float(b[1]), r2


def fitted_projection(g, rho_hat):
    """Pick the transform by fit, which lets the contamination have a say."""
    best = (-np.inf, np.nan, None)
    for phi in GRID_PHI:
        for kappa in GRID_KAPPA:
            for lag in GRID_LAG:
                b, r2 = project(g, shape(rho_hat, phi, kappa, lag))
                if r2 > best[0]:
                    best = (r2, b, (phi, kappa, lag))
    return best[1], best[2]


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

    rng = np.random.default_rng(90_000 + seed)
    rho = flighted(Tp, rng, flat=False)
    size = np.abs(YN[:T0].mean(axis=0)) + 1.0
    imp = np.zeros((Tp, J))
    for j in treated:
        imp[:, j] = rho * size[j] * rng.lognormal(0.0, NOISE, size=Tp)
    rho_hat = imp[:, treated].sum(axis=1)
    rho_hat = rho_hat / rho_hat.mean()

    theta = scale
    rows = []
    for rname, rp in REGIMES.items():
        m = shape(rho, **rp)                    # true effect driver
        tau_t = theta * m
        true_att = float(tau_t.mean())          # equals theta by normalisation
        m_oracle = shape(rho_hat, **rp)         # true transform, observed path
        m_raw = rho_hat / rho_hat.mean()

        for corr in CORRS:
            gam = PI_MULT * scale * (1.0 + VAR_FRAC * profile_with_corr(m, corr, rng))
            Y = YN.copy(); Y[post, treated] += tau_t[:, None]
            Yc = Y.copy(); Yc[post, kstar] += gam

            g = (Yc @ w)[post] - Yc[post] @ v
            g_clean = (Y @ w)[post] - Y[post] @ v
            naive = float(g.mean())
            oracle = float(g_clean.mean())

            fit = simplex_lstsq(Yc[:T0, clean], Yc[:T0, kstar])
            Yi = Yc.copy(); Yi[post, kstar] = Yc[post][:, clean] @ fit
            known = float(((Yi @ w)[post] - Yi[post] @ v).mean())

            s_raw, _ = project(g, m_raw)
            s_orc, _ = project(g, m_oracle)
            s_fit, picked = fitted_projection(g, rho_hat)

            rows.append(dict(dgp=name, seed=seed, regime=rname, corr=corr, vk=vk,
                             Tpost=Tp, theta=theta, true_att=true_att,
                             oracle=oracle, naive=naive, known=known,
                             surrogate_raw=s_raw, surrogate_oracle=s_orc,
                             surrogate_fitted=s_fit,
                             picked_phi=picked[0], picked_kappa=(-1.0 if picked[1] is None else picked[1]),
                             picked_lag=picked[2],
                             true_phi=rp["phi"],
                             true_kappa=(-1.0 if rp["kappa"] is None else rp["kappa"]),
                             true_lag=rp["lag"]))
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
