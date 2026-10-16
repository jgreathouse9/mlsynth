"""Path-B Monte Carlo: CSC against fDiD, iDiD and PSC.

Reproduces the design of ``Simulation/3_Simulation_CSC_IDiD_FDiD.R`` and
reports the quantity the paper's Table 1 reports -- the average estimation
error ``mean_r (tau_hat_r - tau)`` -- alongside the average *absolute*
estimation error, because the signed mean cancels a bias whose sign flips
across replications and the paper's footnote 27 says the absolute version was
never run.

Two protocols, selected by ``--fixed-factors``:

* redrawn -- fresh ``lambda`` and ``gamma`` every replication, which is what
  the shipped driver does;
* fixed   -- both held at the values the paper prints in eq. (26) and (27),
  which is what Appendix A.f says was done.

Both DiD variants are run: the reference's own treatment column (off by one
unit, see ``baselines.twfe_did``) and the correct one, so the size of that bug
is measured, not assumed. CSC is run raw and with the ridge tie-break
that makes its answer solver-independent (``tiebreak.py``).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from baselines import idid, psc_weights_all, twfe_did
from core import csc_counterfactual, csc_weights
from dgp import simulate
from fixed_factors import GAMMA_PAPER, LAMBDA_PAPER

BASE = dict(N=100, T=5, num_categories=5, num_factors=2, prob_tr=0.15, treat_not_at_random=True)

GRID = {
    # The shipped driver's grid (``tunning_params``).
    "baseline": dict(BASE),
    "N = 200": dict(BASE, N=200),
    "T = 4": dict(BASE, T=4),
    "T = 7": dict(BASE, T=7),
    "F = 4": dict(BASE, num_factors=4),
    "at random": dict(BASE, treat_not_at_random=False),
    # Rows the paper reports that the shipped driver does not carry.
    "E[pi] = 0.40": dict(BASE, prob_tr=0.40),
    # The selection dial behind the paper's central claim.
    "corr x0.50": dict(BASE, corr_scale=0.5),
    "corr x0.00": dict(BASE, corr_scale=0.0),
}

METHODS = ("csc", "csc_ridge", "csc_pinned", "fdid", "fdid_ref", "idid", "psc")
RIDGE = 1e-2


def one_rep(args) -> dict | None:
    seed, cfg = args
    p = simulate(rng=np.random.default_rng(seed), **cfg)

    tr, do = p.treated_idx, p.donor_idx
    T = p.Y.shape[1]
    T0 = T - 1
    if tr.size < 2 or do.size < 2:
        return None

    Y1_pre, Y0_pre = p.Y[tr, :T0], p.Y[do, :T0]
    Y1_post, Y0_post = p.Y[tr, T0:], p.Y[do, T0:]
    truth = p.Y0[tr, T0:]  # untreated potential outcome, observable only in simulation
    X1 = p.X_dummies[tr]

    err, rmse = {}, {}

    def record(name, counterfactual):
        cf = np.asarray(counterfactual, dtype=float).reshape(tr.size, -1)
        err[name] = float(np.mean(Y1_post - cf)) - p.tau
        rmse[name] = float(np.sqrt(np.mean((cf - truth) ** 2)))

    try:
        record("csc", csc_counterfactual(csc_weights(Y1_pre, Y0_pre, X1), Y0_post))
        record(
            "csc_ridge",
            csc_counterfactual(csc_weights(Y1_pre, Y0_pre, X1, ridge=RIDGE), Y0_post),
        )
        record(
            "csc_pinned",
            csc_counterfactual(
                csc_weights(Y1_pre, Y0_pre, X1, intercept="pin_first"), Y0_post
            ),
        )
    except Exception:  # noqa: BLE001
        return None

    for name, offby in (("fdid", False), ("fdid_ref", True)):
        att, cf = twfe_did(p.Y, p.D, reference_offbyone=offby)
        err[name] = att - p.tau
        rmse[name] = float(np.sqrt(np.mean((cf.reshape(-1, 1) - truth) ** 2)))

    att, cf = idid(p.Y, p.D, p.mu, p.lam, p.mu_mean, p.lambda_mean)
    err["idid"] = att - p.tau
    rmse["idid"] = float(np.sqrt(np.mean((cf.reshape(-1, 1) - truth) ** 2)))

    W, _ = psc_weights_all(Y1_pre, Y0_pre)
    record("psc", W @ Y0_post)

    return {"err": err, "rmse": rmse, "n1": int(tr.size)}


def run(rows, reps, workers, fixed_factors, seed0=1000):
    out = {}
    for name in rows:
        cfg = dict(GRID[name])
        if fixed_factors:
            cfg.update(fixed_lam=LAMBDA_PAPER, fixed_Gamma=GAMMA_PAPER)
        jobs = [(seed0 + 7919 * i, cfg) for i in range(reps)]
        t = time.time()
        if workers > 1:
            with mp.Pool(workers) as pool:
                res = [r for r in pool.map(one_rep, jobs, chunksize=2) if r]
        else:
            res = [r for r in map(one_rep, jobs) if r]

        row = {"n_ok": len(res), "n1_mean": float(np.mean([r["n1"] for r in res]))}
        for m in METHODS:
            e = np.array([r["err"][m] for r in res])
            row[f"err_{m}"] = float(e.mean())
            row[f"abserr_{m}"] = float(np.abs(e).mean())
            row[f"se_{m}"] = float(e.std(ddof=1) / np.sqrt(len(e)))
            row[f"rmse_{m}"] = float(np.mean([r["rmse"][m] for r in res]))
        out[name] = row
        print(
            f"{name:<14} n1~{row['n1_mean']:5.1f} | "
            + " ".join(f"{m}={row['err_'+m]:+.2f}/{row['abserr_'+m]:.2f}" for m in METHODS)
            + f" [{time.time()-t:.0f}s]",
            flush=True,
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=250)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--rows", nargs="*", default=list(GRID))
    ap.add_argument("--fixed-factors", action="store_true")
    ap.add_argument("--out", default="mc_results.json")
    a = ap.parse_args()
    print("signed mean error / mean absolute error; true tau = 1.0", flush=True)
    res = run(a.rows, a.reps, a.workers, a.fixed_factors)
    with open(a.out, "w") as fh:
        json.dump(
            {"reps": a.reps, "fixed_factors": a.fixed_factors, "rows": res}, fh, indent=2
        )
    print("wrote", a.out)
