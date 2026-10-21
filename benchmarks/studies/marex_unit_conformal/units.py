"""Conformal inference for each treated unit of a MAREX design, and the aggregate.

MAREX tests and bands the aggregate contrast, the synthetic treated minus the
synthetic control. In Abadie and Zhao's Unit-level design (their formulation
10) each treated unit ``j`` has its own synthetic control ``v_.j``, fitted on the
fitting periods from the untreated units, and the aggregate is the
``w``-weighted average of the per-unit contrasts (their eq. 11),

    u_jt = Y_jt - sum_i v_ij Y_it,        tau_hat_t = sum_j w_j u_jt.

Each ``u_j`` has a blank stretch and a post stretch, so the aggregate's inference
can run on it unit by unit. This measures whether that holds.

Per-unit contrasts:

* ``unit`` -- ``v_.j`` refitted after the design, on the fitting periods, from
  the untreated units alone: eq. 10's constraint ``v_ij = 0`` for treated ``i``,
  which can be imposed once the assignment is known.
* ``naive`` -- the group's shared control, ``u_j = Y_j - Y v``: what reading one
  treated unit off today's result gives.

Tests and bands on each:

* Theorem 2's permutation p-value on ``S = mean |u|`` over the post block,
  against ``B`` draws of post blocks from the blank and post periods,
  ``(1 + k) / (1 + B)``.
* Theorem 3's pointwise band with the split-conformal quantile, the
  ``ceil((Tb + 1)(1 - alpha))``-th order statistic of ``|u|`` over the blank
  periods; infinite below ``ceil(1 / alpha) - 1`` blanks.
* The band for the unit's total by inversion, evaluated at the true total: the
  signed post sum of the residuals of an SC refit on every period, against its
  cyclic shifts over the blank and post periods (the construction
  ``marex_cumulative_conformal`` settled on). ``constant`` is a constant effect
  per unit; ``ramp`` adds a ramp-shaped departure from it.

Across the ``m`` treated units, under no effect: any raw ``p_j <= alpha``,
Bonferroni (under no effect at all Holm rejects exactly when Bonferroni does),
and a joint test whose statistic is
``max_j mean_post |u_jt| / mean_window |u_jt|`` with the same draw of periods
applied to every unit, so the units' common factors stay aligned. The scale is
a mean over the whole blank-and-post window, which no draw changes, so the joint
test is a permutation test of its own.

The aggregate: the shared contrast ``Y w - Y v`` against eq. 11's
``sum_j w_j u_j``, by size, pointwise coverage, band half-width, and the RMSE
of the per-period error on the fitting, blank and post periods.

Designs ``standard`` and ``unit_penalized`` (``xi = 1``) run on the same panels,
``m = 3`` treated of ``J = 15``, the panel of ``marex_cumulative_conformal``.

Run from the repository root::

    python -m benchmarks.studies.marex_unit_conformal.units [draws] [dgp] [config|all] [seed0]
"""
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from benchmarks.studies.marex_cumulative_conformal.candidate2 import shapes
from benchmarks.studies.marex_cumulative_conformal.designs import J, panel
from mlsynth.utils.conformal.quantile import split_conformal_quantile
from mlsynth.utils.conformal.refit import conformal_refit_gaps
from mlsynth.utils.solvers.simplex import simplex_lstsq

ALPHAS = (0.05, 0.2)
#: (fitting periods, blank periods, post periods)
CONFIGS = {"mid": (40, 26, 13), "short": (40, 12, 13)}
DESIGNS = {"standard": {}, "unit_penalized": {"xi": 1.0}}
M = 3
B = 999
SCALE = 2.0


def solve(Y, TE, Tb, H, design):
    from mlsynth import MAREX
    T0 = TE + Tb
    df = pd.DataFrame({"unit": np.repeat(np.arange(J), Y.shape[0]),
                       "time": np.tile(np.arange(Y.shape[0]), J),
                       "y": Y.T.reshape(-1)})
    res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                 "T0": T0, "blank_periods": Tb, "T_post": H, "m_eq": M,
                 "backend": "exact", "inference": False, "design": design,
                 "display_graph": False, **DESIGNS[design]}).fit()
    g = res.globres
    return (np.asarray(g.treated_weights_agg, float),
            np.asarray(g.control_weights_agg, float))


def thm2_p(x, H, draws):
    """Theorem 2's p-value: ``x`` is the blank-and-post window, post last."""
    s_obs = np.mean(np.abs(x[-H:]))
    s = np.mean(np.abs(x[draws]), axis=1)
    return (1 + np.sum(s >= s_obs - 1e-12 * (1 + s_obs))) / (1 + len(draws))


def joint_p(X, H, draws):
    """Max over units of the scale-free statistic, one draw of periods for all."""
    Z = np.abs(X) / np.mean(np.abs(X), axis=1, keepdims=True)
    t_obs = np.max(np.mean(Z[:, -H:], axis=1))
    t = np.max(np.mean(Z[:, draws], axis=2), axis=0)
    return (1 + np.sum(t >= t_obs - 1e-12 * (1 + t_obs))) / (1 + len(draws))


def cyclic_signed_p(r, H):
    """Signed post sum against every cyclic shift of ``r``; post is the last ``H``."""
    N = r.size
    idx = (N - H + np.arange(N)[:, None] + np.arange(H)[None, :]) % N
    S = np.abs(r[idx].sum(axis=1))
    return float(np.mean(S >= S[0] - 1e-9 * (1.0 + S[0])))


def unit_total_p(y, Y0, T0, TE, H, C):
    """The inversion's p-value at the true total ``C``."""
    yc = y.copy()
    yc[T0:] -= C / H
    r = conformal_refit_gaps(yc, Y0, rule="sc")
    return cyclic_signed_p(r[TE:], H)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def one_design(Y, TE, Tb, H, design, draws, ramp):
    T0 = TE + Tb
    w, v = solve(Y, TE, Tb, H, design)
    treated = np.flatnonzero(w > 1e-8)
    untreated = np.flatnonzero(w <= 1e-8)
    Y0 = Y[:, untreated]
    U = {"unit": [], "naive": []}
    for j in treated:
        vj = simplex_lstsq(Y0[:TE], Y[:TE, j])
        U["unit"].append(Y[:, j] - Y0 @ vj)
        U["naive"].append(Y[:, j] - Y @ v)
    U = {k: np.array(x) for k, x in U.items()}
    wt = w[treated] / w[treated].sum()
    agg = {"shared": Y @ w - Y @ v, "eq11": wt @ U["unit"]}

    p_unit = {arm: np.array([thm2_p(x, H, draws) for x in X[:, TE:]])
              for arm, X in U.items()}
    p_joint = {arm: joint_p(X[:, TE:], H, draws) for arm, X in U.items()}
    p_agg = {name: thm2_p(g[TE:], H, draws) for name, g in agg.items()}
    p_total = {}
    for k, j in enumerate(treated):
        c = SCALE * (k + 1)
        for path, dev in (("constant", 0.0), ("ramp", SCALE * ramp)):
            y = Y[:, j].copy()
            y[T0:] += c + dev
            p_total.setdefault(path, []).append(
                unit_total_p(y, Y0, T0, TE, H, c * H))

    out = {}
    for a in ALPHAS:
        for arm, X in U.items():
            p = p_unit[arm]
            out[(arm, "size", a)] = float(np.mean(p <= a))
            out[(arm, "fwer_raw", a)] = float(np.any(p <= a))
            out[(arm, "fwer_bonf", a)] = float(np.any(p <= a / p.size))
            out[(arm, "fwer_joint", a)] = float(p_joint[arm] <= a)
            q = np.array([split_conformal_quantile(x[TE:T0], a) for x in X])
            out[(arm, "pointwise", a)] = float(np.mean(np.abs(X[:, T0:]) <= q[:, None]))
            out[(arm, "halfwidth", a)] = float(np.median(q))
        for name, g in agg.items():
            out[(name, "size", a)] = float(p_agg[name] <= a)
            q = split_conformal_quantile(g[TE:T0], a)
            out[(name, "pointwise", a)] = float(np.mean(np.abs(g[T0:]) <= q))
            out[(name, "halfwidth", a)] = float(q)
        for path, ps in p_total.items():
            out[("unit", f"total_{path}", a)] = float(np.mean(np.array(ps) > a))
    for name, g in agg.items():
        out[(name, "rmse_fit", 0)] = rms(g[:TE])
        out[(name, "rmse_blank", 0)] = rms(g[TE:T0])
        out[(name, "rmse_post", 0)] = rms(g[T0:])
    for arm, X in U.items():
        out[(arm, "rmse_blank", 0)] = rms(X[:, TE:T0])
        out[(arm, "rmse_post", 0)] = rms(X[:, T0:])
    return out


def run(dgp, config, draws, seed0):
    TE, Tb, H = CONFIGS[config]
    ramp = shapes(H)["ramp"]
    acc = {}
    for d in range(draws):
        rng = np.random.default_rng(seed0 + d)
        Y = panel(TE + Tb + H, dgp, rng)
        draws_idx = np.array([rng.choice(Tb + H, size=H, replace=False)
                              for _ in range(B)])
        for design in DESIGNS:
            for k, val in one_design(Y, TE, Tb, H, design, draws_idx, ramp).items():
                acc.setdefault((design,) + k, []).append(val)
    head = f"units {dgp} {config} TE={TE} Tb={Tb} H={H} draws={draws} seed0={seed0}"
    for key in sorted(acc, key=str):
        vals = np.asarray(acc[key], float)
        stat = (np.median(vals) if key[2] in ("halfwidth", "rmse_fit",
                                              "rmse_blank", "rmse_post")
                else np.mean(vals))
        print(f"{head} {key[0]} {key[1]} {key[2]} alpha={key[3]}: {stat:.4f}",
              flush=True)


if __name__ == "__main__":
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    dgps = (sys.argv[2],) if len(sys.argv) > 2 else ("iid", "ar5", "ar", "sorted")
    configs = ((sys.argv[3],) if len(sys.argv) > 3 and sys.argv[3] != "all"
               else tuple(CONFIGS))
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    for g in dgps:
        for c in configs:
            run(g, c, draws, seed0)
