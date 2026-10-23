"""Per-unit and aggregate inference over a long pre-period.

``units.py`` asks whether per-unit inference holds on Abadie and Zhao's Section
5 panel. This asks how to split a long pre-period between fitting and blank
periods when per-unit inference is wanted: 60 units in four clusters of 10,
15, 15 and 20, 140 pre-treatment periods and 12 post periods.

The panel is a linear factor model on the log scale,

    log y_jt = a_j + delta_t + lambda_j' f_t + eps_jt,

with round values:

* ``a_j ~ N(6, 1)``, the unit's level;
* ``delta_t``, common to every unit: a cycle of period 52 with amplitude 0.2,
  plus AR(1) shocks at 0.5 with sd 0.15;
* three factors, AR(1) at 0.5 with unit variance; each unit's loadings are its
  cluster's mean loadings, ``N(0, 0.03^2)``, plus its own, ``N(0, 0.03^2)``;
* ``eps_jt``, AR(1) at 0.2 with sd 0.06.

MAREX runs on the levels ``y``, clustered, two treated per cluster, eight in
all. Each draw solves a ``standard`` and a ``unit_penalized`` design
(``xi = 1``) at each blank length, 20 to 70 of the 140 periods.

Per treated unit ``j``, ``v_.j`` is refitted on the fitting periods from its
cluster's untreated units (Abadie and Zhao's eq. 10), and on ``u_j = y_j -
Y0 v_.j``:

* Theorem 2's p-value under no effect, 999 drawn arrangements;
* the split-conformal pointwise band, and its half-width relative to the
  unit's mean level (the smallest single-period effect it could detect);
* the band for the unit's total by inversion at the true total, under a 5%
  multiplicative lift, which in levels follows the cycle;
* across the eight units: any raw rejection, Bonferroni, and the joint test.

The aggregate, shared contrast and eq. 11, by the same measures.

Run from the repository root::

    python -m benchmarks.studies.marex_unit_conformal.long_pre [draws] [Tb|all] [seed0]
"""
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from benchmarks.studies.marex_unit_conformal.units import (B, cyclic_signed_p,
                                                           joint_p, thm2_p)
from mlsynth.utils.conformal.quantile import split_conformal_quantile
from mlsynth.utils.conformal.refit import conformal_refit_gaps
from mlsynth.utils.solvers.simplex import simplex_lstsq

CLUSTERS = (10, 15, 15, 20)
J = sum(CLUSTERS)
T0, H = 140, 12
BLANKS = (20, 26, 39, 52, 70)
ALPHAS = (0.05, 0.1)
M_PER_CLUSTER = 2
LIFT = 0.05
DESIGNS = {"standard": {}, "unit_penalized": {"xi": 1.0}}


def _ar(T, phi, rng, burn=200):
    z = np.zeros(T + burn)
    e = rng.standard_normal(T + burn) * np.sqrt(1 - phi ** 2)
    for t in range(1, T + burn):
        z[t] = phi * z[t - 1] + e[t]
    return z[burn:]


def factor_panel(rng, T=T0 + H):
    """Levels ``(T, J)`` and each unit's cluster."""
    cluster = np.repeat(np.arange(len(CLUSTERS)), CLUSTERS)
    a = rng.normal(6.0, 1.0, J)
    t = np.arange(T)
    delta = (0.2 * np.sin(2 * np.pi * t / 52 + rng.uniform(0, 2 * np.pi))
             + 0.15 * _ar(T, 0.5, rng))
    F = np.column_stack([_ar(T, 0.5, rng) for _ in range(3)])
    lam = (rng.normal(0, 0.03, (len(CLUSTERS), 3))[cluster]
           + rng.normal(0, 0.03, (J, 3)))
    eps = 0.06 * np.column_stack([_ar(T, 0.2, rng) for _ in range(J)])
    return np.exp(a + delta[:, None] + F @ lam.T + eps), cluster


def solve(Y, cluster, Tb, design):
    from mlsynth import MAREX
    T = Y.shape[0]
    df = pd.DataFrame({"unit": np.repeat(np.arange(J), T),
                       "time": np.tile(np.arange(T), J),
                       "y": Y.T.reshape(-1),
                       "cluster": np.repeat(cluster, T)})
    res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                 "T0": T0, "blank_periods": Tb, "T_post": H,
                 "m_eq": M_PER_CLUSTER, "cluster": "cluster", "backend": "exact",
                 "inference": False, "design": design, "display_graph": False,
                 **DESIGNS[design]}).fit()
    g = res.globres
    return (np.asarray(g.treated_weights_agg, float),
            np.asarray(g.control_weights_agg, float))


def one(Y, cluster, Tb, design, draws):
    TE = T0 - Tb
    w, v = solve(Y, cluster, Tb, design)
    treated = np.flatnonzero(w > 1e-8)
    U, scale, p_tot = [], [], []
    for j in treated:
        pool = np.flatnonzero((cluster == cluster[j]) & (w <= 1e-8))
        vj = simplex_lstsq(Y[:TE, pool], Y[:TE, j])
        U.append(Y[:, j] - Y[:, pool] @ vj)
        scale.append(Y[:T0, j].mean())
        # a 5% lift in levels follows the unit's own cycle
        y = Y[:, j].copy()
        lift = LIFT * Y[T0:, j]
        y[T0:] += lift
        C = float(lift.sum())
        yc = y.copy()
        yc[T0:] -= C / H
        r = conformal_refit_gaps(yc, Y[:, pool], rule="sc")
        p_tot.append(cyclic_signed_p(r[TE:], H))
    U, scale, p_tot = np.array(U), np.array(scale), np.array(p_tot)
    wt = w[treated] / w[treated].sum()
    agg = {"shared": Y @ w - Y @ v, "eq11": wt @ U}
    agg_scale = float((Y[:T0] @ w).mean())

    p_unit = np.array([thm2_p(x, H, draws) for x in U[:, TE:]])
    p_joint = joint_p(U[:, TE:], H, draws)
    out = {}
    for a in ALPHAS:
        out[("unit", "size", a)] = float(np.mean(p_unit <= a))
        out[("unit", "fwer_raw", a)] = float(np.any(p_unit <= a))
        out[("unit", "fwer_bonf", a)] = float(np.any(p_unit <= a / p_unit.size))
        out[("unit", "fwer_joint", a)] = float(p_joint <= a)
        q = np.array([split_conformal_quantile(x[TE:T0], a) for x in U])
        out[("unit", "pointwise", a)] = float(np.mean(np.abs(U[:, T0:]) <= q[:, None]))
        out[("unit", "rel_halfwidth", a)] = float(np.median(q / scale))
        out[("unit", "total_lift", a)] = float(np.mean(p_tot > a))
        for name, g in agg.items():
            out[(name, "size", a)] = float(thm2_p(g[TE:], H, draws) <= a)
            qa = split_conformal_quantile(g[TE:T0], a)
            out[(name, "pointwise", a)] = float(np.mean(np.abs(g[T0:]) <= qa))
            out[(name, "rel_halfwidth", a)] = float(qa / agg_scale)
    for name, g in agg.items():
        out[(name, "rel_rmse_post", 0)] = float(np.sqrt(np.mean(g[T0:] ** 2)) / agg_scale)
    out[("unit", "rel_rmse_blank", 0)] = float(np.median(
        np.sqrt(np.mean(U[:, TE:T0] ** 2, axis=1)) / scale))
    return out


def run(draws, blanks, seed0):
    acc = {}
    for d in range(draws):
        rng = np.random.default_rng(seed0 + d)
        Y, cluster = factor_panel(rng)
        for Tb in blanks:
            idx = np.array([rng.choice(Tb + H, size=H, replace=False)
                            for _ in range(B)])
            for design in DESIGNS:
                for k, val in one(Y, cluster, Tb, design, idx).items():
                    acc.setdefault((Tb, design) + k, []).append(val)
    for key in sorted(acc, key=str):
        vals = np.asarray(acc[key], float)
        stat = np.median(vals) if "rel_" in key[3] else np.mean(vals)
        print(f"long_pre draws={draws} seed0={seed0} Tb={key[0]} {key[1]} {key[2]} "
              f"{key[3]} alpha={key[4]}: {stat:.4f}", flush=True)


if __name__ == "__main__":
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    blanks = (BLANKS if len(sys.argv) <= 2 or sys.argv[2] == "all"
              else (int(sys.argv[2]),))
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    run(draws, blanks, seed0)
