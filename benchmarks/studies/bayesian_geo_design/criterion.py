"""Arm 3: what the MDE actually ranks, and what to rank on instead.

This arm is independent of which engine does the fitting. It scores the same
990-design pool (Walmart, cardinality-2 MAREX designs) by several criteria and
compares each against the realised out-of-sample contrast error, over three
origins.

    python criterion.py [results/criterion.json]

The defect. GEOX ranks by the minimum detectable effect.
``placebo_detection_boundary`` gives ``up = (z sigma - tau0)/baseline`` and
``down = (-z sigma - tau0)/baseline``, an interval displaced by ``tau0``, the
backtest's own estimation error. ``compute_mde`` then keeps the smaller
magnitude of the two directions, which is by construction the side ``tau0``
pushes toward, so a biased backtest reports a smaller MDE and looks more
sensitive. Measured: rho(|tau0|/sigma, |MDE|) = -0.79 (sdid), -0.93 (tasc);
detection-interval asymmetry |up|/|down| = 0.73 and 6.64 against 1.00 for a
centred one; sign(MDE) equals sign(tau0) in 96% of designs. ``tau0`` is a
persistent property of the candidate, not backtest noise -- sign consistency
0.77 and 0.99 across eight backtests, against 0.35 for a coin flip -- so it
does not average out.

The measured consequence, three origins, mean over them:

    criterion                rho     selected design's realised error
    mde                   -0.018                      11.94%
    power at 5%           +0.468                       4.31%
    power at 10%          +0.524                       6.89%
    centred mde           +0.414                      44.21%
    backtest rmse         +0.243                      11.40%
    in-sample contrast    +0.862                       2.93%
    best in pool               -                       1.40%

Two readings the numbers force. Centring the interval confirms the diagnosis --
rho moves from -0.018 to +0.414, positive at every origin -- and makes the
decision worse, because ``z sigma / baseline`` ranks on noise relative to
treated volume and so favours large markets; ``tau0`` was partly offsetting
that. Fixing the effect size in advance never invokes the min-magnitude rule at
all, and that is the one change that improves both the ordering and the choice.
``power at 2%`` is degenerate on this panel: nothing has power there, so its
apparent numbers are tie-breaking artifacts.
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import norm, spearmanr

from mlsynth.utils.geox_helpers.aggregate import (compute_accuracy, compute_mde,
                                                  compute_power)
from mlsynth.utils.geox_helpers.batch import run_simulations

DATA = "basedata/walmart_weekly_sales_covariates.csv"
COVS = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
HORIZON, BIGM = 15, 1e6


def simplex_ls(A, b):
    """min ||A x - b||^2 subject to x >= 0 and sum(x) == 1."""
    k = A.shape[1]
    if k == 1:
        return np.ones(1)
    x, _ = nnls(np.vstack([A, BIGM * np.ones((1, k))]),
                np.concatenate([b, [BIGM]]))
    s = x.sum()
    return x / s if s > 0 else np.ones(k) / k


def design_pool(Y, X, K=2):
    """Every |S| = K MAREX design: treated and control weights, and the contrast."""
    n = X.shape[0]
    xbar, xt = X.mean(axis=0), X.T
    pool = []
    for S in itertools.combinations(range(n), K):
        S = list(S)
        Sc = [j for j in range(n) if j not in S]
        w, v = simplex_ls(xt[:, S], xbar), simplex_ls(xt[:, Sc], xbar)
        c = np.zeros(n)
        c[S], c[Sc] = w, -v
        pool.append({"S": tuple(S), "c": c})
    return pool


def predictors(Y_fit, cov, standardize=True):
    X = np.hstack([Y_fit, np.asarray(cov, dtype=float)])
    if standardize:
        sd = X.std(axis=0)
        X = X / np.where(sd == 0, 1.0, sd)
    return X


def run(data_path: str = DATA, origins=(100, 114, 128), n_sample=150) -> dict:
    df = pd.read_csv(data_path)
    units = df["store"].unique()
    wide = df.pivot(index="week", columns="store", values="sales").reindex(columns=units)
    Y = wide.to_numpy().T
    grid = np.concatenate([np.arange(0.002, 0.0501, 0.002), [0.06, 0.08, 0.10]])
    effects = sorted({round(float(x), 4) for x in np.concatenate([grid, -grid])})
    z = float(norm.ppf(0.95))
    out = []
    for T0 in origins:
        fit_end = T0 - 28
        hold = np.arange(T0, min(T0 + HORIZON, Y.shape[1]))
        if len(hold) < 8:
            continue
        weeks = np.sort(df["week"].unique())[:T0]
        cov = (df[df["week"].isin(weeks)].groupby("store")[COVS].mean()
               .reindex(units).to_numpy())
        pool = design_pool(Y, predictors(Y[:, :fit_end], cov))
        C = np.stack([d["c"] for d in pool])
        realised = np.sqrt(((Y[:, hold].T @ C.T) ** 2).mean(axis=0))
        volume = np.array([Y[list(d["S"]), :T0].mean() for d in pool])
        relative = realised / volume
        in_sample = np.sqrt(((Y[:, :fit_end].T @ C.T) ** 2).mean(axis=0))
        idx = np.random.default_rng(T0).choice(len(pool), size=n_sample, replace=False)
        cands = [frozenset(int(units[j]) for j in pool[i]["S"]) for i in idx]
        cube = run_simulations(wide.iloc[:T0], cands, durations=[HORIZON],
                               n_backtests=6, effect_sizes=effects, n_draws=200,
                               seed=0, n_jobs=1, engine="sdid", alpha=0.1)
        power = compute_power(cube, alpha=0.1)
        acc = compute_accuracy(cube).set_index("candidate")
        mde = compute_mde(power, power_threshold=0.8).set_index("candidate")
        base = {k: float(wide.iloc[:T0][list(k)].mean(axis=1).iloc[-HORIZON:].mean())
                for k in cands}
        at = {e: power[np.isclose(power["effect_size"].abs(), e)]
              .groupby("candidate", observed=True)["power"].mean()
              for e in (0.05, 0.10)}
        rows = []
        for k, i in zip(cands, idx):
            b = base[k]
            rows.append(dict(
                mde=abs(mde["mde"].get(k, np.nan)),
                centred_mde=z * acc["placebo_sigma_mean"].get(k, np.nan) / b,
                bt_rmse=acc["att_error_rmse"].get(k, np.nan) / b,
                neg_power_5=-float(at[0.05].get(k, np.nan)),
                neg_power_10=-float(at[0.10].get(k, np.nan)),
                contrast=in_sample[i], realised=relative[i]))
        M = pd.DataFrame(rows).astype(float)
        rec = {"T0": int(T0), "best": float(M["realised"].min())}
        for c in [x for x in M.columns if x != "realised"]:
            ok = np.isfinite(M[c]) & np.isfinite(M["realised"])
            if ok.sum() < 5:
                continue
            rec["rho_" + c] = float(spearmanr(M[c][ok], M["realised"][ok]).statistic)
            rec["pick_" + c] = float(M["realised"][M[c][ok].idxmin()])
        out.append(rec)
        print("origin T0=%d done" % T0, flush=True)
    return {"origins": out}


if __name__ == "__main__":
    res = run()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            json.dump(res, fh, indent=2)
